"""Per-run budget enforcement for ReproBench.

Two hard limits are enforced around every agent run (paper Fig. 3, Rules 2 and 3):

* a **cost** cap of ``$5.00`` (``config/benchmark.yaml`` -> ``budget.max_cost_usd``) with a
  soft warning at ``warn_fraction`` (default 0.80, i.e. ``$4.00``), and
* a **wall clock** cap of ``24h`` (``budget``/``wall_clock.max_hours``).

The module is deliberately clock- and sink-injectable so the whole policy can be unit
tested offline with a fake ``$6`` token stream or a fake 5-second clock (plan: "budget
abort tested with a fake $6 stream; 24h watchdog tested with a fake 5s clock").
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..utils.io import load_config

# Fallbacks mirror config/benchmark.yaml; config remains the single source of truth.
DEFAULT_MAX_COST_USD = 5.00
DEFAULT_MAX_HOURS = 24.0
DEFAULT_WARN_FRACTION = 0.80
DEFAULT_ABORT_GRACE_SECONDS = 30
DEFAULT_SIGTERM_GRACE_SECONDS = 60

#: Reasons a run may be terminated. ``None`` means "still running / finished normally".
TERMINATION_BUDGET = "budget_abort"
TERMINATION_WATCHDOG = "watchdog_24h"

__all__ = [
    "Budget",
    "BudgetEvent",
    "TERMINATION_BUDGET",
    "TERMINATION_WATCHDOG",
    "budget_from_config",
]


def _cfg(path: List[str], default: Any) -> Any:
    """Best-effort dotted lookup into ``benchmark.yaml`` (never raises)."""
    try:
        cfg: Any = load_config("benchmark")
    except Exception:  # pragma: no cover - defensive
        return default
    for key in path:
        if not isinstance(cfg, dict) or key not in cfg:
            return default
        cfg = cfg[key]
    return cfg if cfg is not None else default


def budget_from_config() -> "Budget":
    """Build a :class:`Budget` from ``config/benchmark.yaml``."""
    return Budget(
        max_cost_usd=float(_cfg(["budget", "max_cost_usd"], DEFAULT_MAX_COST_USD)),
        max_hours=float(_cfg(["wall_clock", "max_hours"], DEFAULT_MAX_HOURS)),
        warn_fraction=float(_cfg(["budget", "warn_fraction"], DEFAULT_WARN_FRACTION)),
        abort_grace_seconds=int(
            _cfg(["budget", "abort_grace_seconds"], DEFAULT_ABORT_GRACE_SECONDS)
        ),
        sigterm_grace_seconds=int(
            _cfg(["wall_clock", "sigterm_grace_seconds"], DEFAULT_SIGTERM_GRACE_SECONDS)
        ),
    )


@dataclass
class BudgetEvent:
    """One observable budget checkpoint."""

    kind: str
    detail: str = ""
    cost_usd: float = 0.0
    elapsed_seconds: float = 0.0
    fraction: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "detail": self.detail,
            "cost_usd": round(self.cost_usd, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "fraction": round(self.fraction, 3),
        }


@dataclass
class Budget:
    """Tracks cumulative cost and wall clock for a single run.

    Parameters
    ----------
    max_cost_usd:
        Hard cap; exceeding it must terminate the run (``$5.00``).
    max_hours:
        Hard wall-clock cap in hours (``24``).
    warn_fraction:
        Soft warning threshold as a fraction of ``max_cost_usd`` (``0.80`` -> ``$4.00``).
    clock:
        Injectable monotonic clock (tests pass a fake).
    """

    max_cost_usd: float = DEFAULT_MAX_COST_USD
    max_hours: float = DEFAULT_MAX_HOURS
    warn_fraction: float = DEFAULT_WARN_FRACTION
    abort_grace_seconds: int = DEFAULT_ABORT_GRACE_SECONDS
    sigterm_grace_seconds: int = DEFAULT_SIGTERM_GRACE_SECONDS
    clock: Callable[[], float] = time.monotonic
    cost_usd: float = 0.0
    started_at: Optional[float] = None
    terminated_by: Optional[str] = None
    warned: bool = False
    events: List[BudgetEvent] = field(default_factory=list)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "Budget":
        """Record the run start time (idempotent)."""
        if self.started_at is None:
            self.started_at = self.clock()
        self.events.append(BudgetEvent(kind="start", cost_usd=self.cost_usd))
        return self

    @property
    def max_seconds(self) -> float:
        return max(0.0, float(self.max_hours)) * 3600.0

    @property
    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        return max(0.0, self.clock() - self.started_at)

    @property
    def elapsed_hours(self) -> float:
        return self.elapsed_seconds / 3600.0

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.max_cost_usd - self.cost_usd)

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.max_seconds - self.elapsed_seconds)

    @property
    def warn_usd(self) -> float:
        return self.max_cost_usd * self.warn_fraction

    @property
    def cost_fraction(self) -> float:
        if self.max_cost_usd <= 0:
            return 1.0 if self.cost_usd > 0 else 0.0
        return self.cost_usd / self.max_cost_usd

    @property
    def exhausted(self) -> bool:
        """True once either limit has been crossed."""
        return self.terminated_by is not None or self.cost_over() or self.time_over()

    # ------------------------------------------------------------- cost accounting
    def add_cost(self, amount: float, detail: str = "") -> bool:
        """Accumulate cost; returns ``False`` when the cap is now exceeded."""
        try:
            delta = float(amount)
        except (TypeError, ValueError):
            delta = 0.0
        if delta > 0:
            self.cost_usd += delta
        if not self.warned and self.cost_usd >= self.warn_usd:
            self.warned = True
            self.events.append(
                BudgetEvent(
                    kind="warn",
                    detail=detail or f"cost warning at ${self.warn_usd:.2f}",
                    cost_usd=self.cost_usd,
                    elapsed_seconds=self.elapsed_seconds,
                    fraction=self.cost_fraction,
                )
            )
        if self.cost_over():
            self._terminate(TERMINATION_BUDGET, detail or "cost cap exceeded")
            return False
        return True

    def add_tokens(self, tokens: Any, pricing: Any = None, detail: str = "") -> bool:
        """Accumulate cost from a token usage mapping and/or precomputed ``cost``.

        ``tokens`` may be a mapping (``{"cost_usd": ..}`` or the trace schema
        ``{"input", "output", "reasoning", "cache": {"read", "write"}}`` merged with a
        ``pricing`` map of per-million-token prices).
        """
        cost = 0.0
        if isinstance(tokens, dict):
            if tokens.get("cost_usd") is not None:
                cost = float(tokens.get("cost_usd") or 0.0)
            elif isinstance(pricing, dict):
                cost = price_tokens(tokens, pricing)
        elif tokens is not None:
            cost = float(tokens)
        return self.add_cost(cost, detail=detail)

    def cost_over(self, extra: float = 0.0) -> bool:
        return (self.cost_usd + max(0.0, float(extra or 0.0))) > self.max_cost_usd

    def time_over(self) -> bool:
        return self.elapsed_seconds > self.max_seconds

    # ------------------------------------------------------------------ watchdog
    def check(self) -> Optional[str]:
        """Evaluate both limits; returns the termination reason (or ``None``)."""
        if self.terminated_by:
            return self.terminated_by
        if self.cost_over():
            self._terminate(TERMINATION_BUDGET, "cost cap exceeded")
            return self.terminated_by
        if self.time_over():
            self._terminate(TERMINATION_WATCHDOG, "wall clock cap exceeded")
            return self.terminated_by
        return None

    def should_abort(self) -> bool:
        return self.check() is not None

    def _terminate(self, reason: str, detail: str) -> None:
        if self.terminated_by is None:
            self.terminated_by = reason
            self.events.append(
                BudgetEvent(
                    kind="abort",
                    detail=detail,
                    cost_usd=self.cost_usd,
                    elapsed_seconds=self.elapsed_seconds,
                    fraction=self.cost_fraction,
                )
            )

    # --------------------------------------------------------------- guard wrapper
    def guard(self, fn: Callable[[], Any]) -> Any:
        """Run ``fn`` only if the budget still allows work."""
        reason = self.check()
        if reason:
            raise BudgetExceeded(reason, self)
        return fn()

    # ------------------------------------------------------------------- reporting
    def summary(self) -> Dict[str, Any]:
        return {
            "max_cost_usd": self.max_cost_usd,
            "max_hours": self.max_hours,
            "warn_fraction": self.warn_fraction,
            "cost_usd": round(self.cost_usd, 4),
            "cost_fraction": round(self.cost_fraction, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "elapsed_hours": round(self.elapsed_hours, 3),
            "remaining_usd": round(self.remaining_usd, 4),
            "remaining_seconds": round(self.remaining_seconds, 1),
            "warned": self.warned,
            "terminated_by": self.terminated_by,
            "events": [event.as_dict() for event in self.events],
        }


class BudgetExceeded(RuntimeError):
    """Raised by :meth:`Budget.guard` when a limit has already been crossed."""

    def __init__(self, reason: str, budget: Budget) -> None:
        super().__init__(f"budget exceeded: {reason}")
        self.reason = reason
        self.budget = budget


def price_tokens(tokens: Dict[str, Any], pricing: Dict[str, Any]) -> float:
    """Price a trace token mapping with a per-million-token pricing table.

    Recognised pricing keys: ``input``, ``output``, ``reasoning``, ``cache_read``,
    ``cache_write``. Reasoning tokens are billed at the output rate unless an explicit
    ``reasoning`` price is configured (documented deviation (f): the paper does not
    define whether ``output`` includes reasoning).
    """

    def _num(value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    cache = tokens.get("cache") if isinstance(tokens.get("cache"), dict) else {}
    cached_in = _num(tokens.get("cache_read", cache.get("read") if cache else 0))
    cache_write = _num(tokens.get("cache_write", cache.get("write") if cache else 0))
    inp = _num(tokens.get("input"))
    out = _num(tokens.get("output"))
    reasoning = _num(tokens.get("reasoning"))

    per_m = 1_000_000.0
    cost = 0.0
    cost += inp * _num(pricing.get("input")) / per_m
    cost += out * _num(pricing.get("output")) / per_m
    cost += reasoning * _num(pricing.get("reasoning", pricing.get("output"))) / per_m
    cost += cached_in * _num(pricing.get("cache_read", pricing.get("input"))) / per_m
    cost += cache_write * _num(pricing.get("cache_write", pricing.get("input"))) / per_m
    return cost
