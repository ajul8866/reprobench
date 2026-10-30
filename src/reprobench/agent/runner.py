"""Trial orchestration for ReproBench (paper Section 4, harness description).

`runner.py` is the thin layer between the CLI/scripts and the concrete harness
adapters (``replay``, ``opencode``, ``openai_compat``).  It owns:

* ``RunConfig`` -- the canonical description of one ``(CVE, model, trial)`` attempt,
* ``run_trial`` -- execute a single attempt and normalise the result metadata,
* ``run_matrix`` -- iterate the 30 x 5 x 3 matrix (optionally filtered),
* ``select_harness`` -- adapter resolution with a safe replay fallback.

The module is deliberately defensive: harness imports are performed lazily so
that ``import reprobench.agent.runner`` works in the offline environment even if
an optional adapter module cannot be imported (e.g. a missing vendor SDK).  No
network calls happen unless a live adapter is explicitly requested.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from ..utils.log import get_logger

LOG = get_logger("agent.runner")

DEFAULT_RUNS_DIR = Path("results/runs")
DEFAULT_ADAPTER = "replay"
DEFAULT_BUDGET_USD = 5.0
DEFAULT_MAX_HOURS = 24.0
DEFAULT_IMAGE = "reprobench/agent:24.04"

ADAPTER_CHOICES = ("replay", "opencode", "openai_compat")

__all__ = [
    "ADAPTER_CHOICES",
    "DEFAULT_ADAPTER",
    "DEFAULT_BUDGET_USD",
    "DEFAULT_IMAGE",
    "DEFAULT_MAX_HOURS",
    "DEFAULT_RUNS_DIR",
    "RunConfig",
    "RunResult",
    "run_matrix",
    "run_trial",
    "select_harness",
]


def _slug(value: str) -> str:
    """Filesystem-safe fragment used inside run ids."""

    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-") or "x"


@dataclass
class RunConfig:
    """Configuration for exactly one benchmark attempt.

    Mirrors :class:`reprobench.agent.harness.RunConfig`; kept here as well so
    that orchestration code has no hard import dependency on an adapter module.
    """

    cve_id: str
    model: str = "glm-5.2"
    trial: int = 1
    adapter: str = DEFAULT_ADAPTER
    runs_dir: Path = DEFAULT_RUNS_DIR
    fixture_dir: Optional[Path] = None
    budget_usd: float = DEFAULT_BUDGET_USD
    max_hours: float = DEFAULT_MAX_HOURS
    image: str = DEFAULT_IMAGE
    provider: Optional[str] = None
    judge_model: Optional[str] = None
    opencode_bin: str = "opencode"
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cve_id = str(self.cve_id)
        self.model = str(self.model)
        self.trial = int(self.trial)
        self.adapter = str(self.adapter or DEFAULT_ADAPTER)
        self.runs_dir = Path(self.runs_dir)
        if self.fixture_dir is not None:
            self.fixture_dir = Path(self.fixture_dir)
        self.budget_usd = float(self.budget_usd)
        self.max_hours = float(self.max_hours)

    # ------------------------------------------------------------------ ids
    @property
    def run_id(self) -> str:
        """Stable identifier: ``<CVE>_<model>_t<trial>`` (sanitised)."""

        return f"{_slug(self.cve_id)}_{_slug(self.model)}_t{int(self.trial)}"

    @property
    def provider_model(self) -> str:
        """``provider/model`` string handed to the OpenCode CLI."""

        if self.provider:
            if "/" in self.model:
                return self.model
            return f"{self.provider}/{self.model}"
        if "/" in self.model:
            return self.model
        lowered = self.model.lower()
        if lowered.startswith("claude"):
            return f"anthropic/{self.model}"
        if lowered.startswith(("gpt", "o1", "o3", "o4")):
            return f"openai/{self.model}"
        if lowered.startswith("glm"):
            return f"zhipu/{self.model}"
        if lowered.startswith("mimo"):
            return f"xiaomi/{self.model}"
        if lowered.startswith("deepseek"):
            return f"deepseek/{self.model}"
        return self.model

    @property
    def run_dir(self) -> Path:
        return Path(self.runs_dir) / self.run_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cve_id": self.cve_id,
            "model": self.model,
            "trial": self.trial,
            "adapter": self.adapter,
            "runs_dir": str(self.runs_dir),
            "fixture_dir": None if self.fixture_dir is None else str(self.fixture_dir),
            "budget_usd": self.budget_usd,
            "max_hours": self.max_hours,
            "image": self.image,
            "provider": self.provider,
            "judge_model": self.judge_model,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunConfig":
        known = {
            "cve_id",
            "model",
            "trial",
            "adapter",
            "runs_dir",
            "fixture_dir",
            "budget_usd",
            "max_hours",
            "image",
            "provider",
            "judge_model",
            "opencode_bin",
            "extra",
        }
        payload = {k: v for k, v in dict(data).items() if k in known}
        payload.setdefault("cve_id", data.get("cve", ""))
        return cls(**payload)


@dataclass
class RunResult:
    """Normalised outcome of one attempt (JSON friendly)."""

    run_id: str
    run_dir: str
    cve_id: str
    model: str
    trial: int
    adapter: str
    terminated_by: str = "completed"
    cost_usd: float = 0.0
    wall_seconds: float = 0.0
    fixture_derived: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "cve_id": self.cve_id,
            "model": self.model,
            "trial": self.trial,
            "adapter": self.adapter,
            "terminated_by": self.terminated_by,
            "cost_usd": round(float(self.cost_usd), 6),
            "wall_seconds": round(float(self.wall_seconds), 3),
            "fixture_derived": bool(self.fixture_derived),
        }
        payload.update(self.extra)
        return payload


# ---------------------------------------------------------------------------
# adapter resolution
# ---------------------------------------------------------------------------
def _harness_module() -> Any:
    """Import the harness module lazily; ``None`` when it is unavailable."""

    try:  # pragma: no cover - depends on optional adapter deps
        from . import harness as harness_mod  # type: ignore

        return harness_mod
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("harness module unavailable: %s", exc)
        return None


class _InlineReplayHarness:
    """Minimal replay adapter used when :mod:`reprobench.agent.harness` fails.

    It materialises the standard run directory layout from a fixture (or an
    empty placeholder) so that the scoring pipeline can still be exercised
    completely offline.
    """

    name = "replay"

    def __init__(self, fixtures_root: Optional[Path] = None) -> None:
        self.fixtures_root = Path(fixtures_root) if fixtures_root else None

    # -- fixture discovery -------------------------------------------------
    def _candidates(self, config: RunConfig) -> List[Path]:
        roots: List[Path] = []
        for root in (config.fixture_dir, self.fixtures_root):
            if root is not None:
                roots.append(Path(root))
        names = [
            config.run_id,
            str(config.trial),
            f"{config.cve_id}_{config.model}_t{config.trial}",
            f"{config.cve_id}_{config.model}",
            config.cve_id,
        ]
        found: List[Path] = []
        for root in roots:
            found.append(root)
            for name in names:
                found.append(root / name)
                found.append(root / config.cve_id / name)
                found.append(root / config.model / name)
        return found

    def fixture_for(self, config: RunConfig) -> Optional[Path]:
        for candidate in self._candidates(config):
            try:
                if candidate.is_dir() and (
                    (candidate / "output").is_dir() or (candidate / "events.jsonl").is_file()
                ):
                    return candidate
                if candidate.is_dir() and any(candidate.iterdir()):
                    if (candidate / "output").is_dir() or (candidate / "trace").is_dir():
                        return candidate
            except OSError:
                continue
        return None

    # -- execution ---------------------------------------------------------
    def run(self, config: RunConfig) -> Dict[str, Any]:
        from ..utils.io import ensure_dir, write_text

        run_dir = config.run_dir
        output_dir = ensure_dir(run_dir / "output")
        trace_dir = ensure_dir(run_dir / "trace")

        fixture = self.fixture_for(config)
        if fixture is not None:
            src_output = fixture / "output"
            if src_output.is_dir():
                for item in sorted(src_output.rglob("*")):
                    if not item.is_file():
                        continue
                    rel = item.relative_to(src_output)
                    dest = output_dir / rel
                    ensure_dir(dest.parent)
                    try:
                        dest.write_bytes(item.read_bytes())
                    except OSError:  # pragma: no cover - defensive
                        LOG.warning("could not copy fixture file %s", item)
            src_trace = fixture / "trace"
            if src_trace.is_dir():
                for item in sorted(src_trace.rglob("*")):
                    if not item.is_file():
                        continue
                    rel = item.relative_to(src_trace)
                    dest = trace_dir / rel
                    ensure_dir(dest.parent)
                    try:
                        dest.write_bytes(item.read_bytes())
                    except OSError:  # pragma: no cover
                        LOG.warning("could not copy fixture trace %s", item)
            elif (fixture / "events.jsonl").is_file():
                (trace_dir / "events.jsonl").write_bytes((fixture / "events.jsonl").read_bytes())
            terminated_by = "fixture_replay"
        else:
            write_text(
                output_dir / "plan.md",
                "# Reproduction plan\n\nFixture missing: no replay workspace was found.\n",
            )
            write_text(
                output_dir / "report.md",
                "# Reproduction report\n\nFixture missing: no replay workspace was found.\n",
            )
            write_text(trace_dir / "events.jsonl", "")
            terminated_by = "fixture_missing"

        return {
            "run_id": config.run_id,
            "run_dir": str(run_dir),
            "adapter": self.name,
            "terminated_by": terminated_by,
            "cost_usd": 0.0,
            "fixture": None if fixture is None else str(fixture),
        }


def select_harness(config: RunConfig, **kwargs: Any) -> Any:
    """Return the adapter instance for ``config`` (always succeeds)."""

    adapter = (config.adapter or DEFAULT_ADAPTER).lower()
    module = _harness_module()

    if module is not None:
        factory = getattr(module, "get_harness", None)
        if callable(factory):
            try:
                return factory(adapter, **kwargs)
            except Exception as exc:  # pragma: no cover - depends on adapter deps
                LOG.warning("adapter %r unavailable (%s); falling back to replay", adapter, exc)
        else:  # pragma: no cover - older harness revisions
            registry = getattr(module, "ADAPTERS", {}) or {}
            cls = registry.get(adapter)
            if cls is not None:
                try:
                    return cls(**kwargs)
                except Exception as exc:  # pragma: no cover
                    LOG.warning("adapter %r failed to initialise (%s)", adapter, exc)

    if adapter != "replay":
        LOG.warning("falling back to the in-memory replay adapter")

    fixtures_root = config.fixture_dir if config.fixture_dir is not None else kwargs.get("fixtures_root")
    return _InlineReplayHarness(fixtures_root=fixtures_root)


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------
def _normalise_result(
    config: RunConfig,
    raw: Any,
    started: float,
) -> RunResult:
    """Coerce whatever the adapter returned into a :class:`RunResult`."""

    data: Dict[str, Any] = {}
    if isinstance(raw, dict):
        data = dict(raw)
    elif raw is not None:
        for key in ("run_id", "run_dir", "terminated_by", "adapter", "cost_usd", "wall_seconds"):
            if hasattr(raw, key):
                data[key] = getattr(raw, key)

    wall = data.get("wall_seconds")
    if wall is None:
        wall = time.monotonic() - started

    terminated = data.get("terminated_by") or "completed"
    adapter = data.get("adapter") or config.adapter

    extra = {k: v for k, v in data.items() if k not in {
        "run_id",
        "run_dir",
        "terminated_by",
        "adapter",
        "cost_usd",
        "wall_seconds",
    }}

    return RunResult(
        run_id=str(data.get("run_id") or config.run_id),
        run_dir=str(data.get("run_dir") or config.run_dir),
        cve_id=config.cve_id,
        model=config.model,
        trial=int(config.trial),
        adapter=str(adapter),
        terminated_by=str(terminated),
        cost_usd=float(data.get("cost_usd") or 0.0),
        wall_seconds=float(wall or 0.0),
        fixture_derived=bool(adapter == "replay" or extra.get("fixture")),
        extra=extra,
    )


def run_trial(config: RunConfig, *, harness_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute one ``(CVE, model, trial)`` attempt and return its metadata.

    The returned dict always contains ``run_id``, ``run_dir``,
    ``terminated_by`` and ``adapter``; the CLI and the matrix driver rely on
    those keys.  Failures inside an adapter never propagate -- they surface as
    ``terminated_by="harness_error"`` so that the matrix keeps going and the
    failure taxonomy can attribute the run.
    """

    if isinstance(config, dict):
        config = RunConfig.from_dict(config)

    started = time.monotonic()
    harness = select_harness(config, **(harness_kwargs or {}))

    try:
        raw = harness.run(config)
    except Exception as exc:  # pragma: no cover - adapter dependent
        LOG.error("run %s failed in adapter %s: %s", config.run_id, config.adapter, exc)
        raw = {
            "run_id": config.run_id,
            "run_dir": str(config.run_dir),
            "terminated_by": "harness_error",
            "adapter": config.adapter,
            "error": f"{type(exc).__name__}: {exc}",
        }

    result = _normalise_result(config, raw, started)

    # Best effort: persist / refresh meta.json so the evidence builder can pick
    # up cve/model/trial even for adapters that do not write it themselves.
    _ensure_meta(config, result)

    LOG.info(
        "run %s adapter=%s terminated_by=%s",
        result.run_id,
        result.adapter,
        result.terminated_by,
    )
    return result.as_dict()


def _ensure_meta(config: RunConfig, result: RunResult) -> None:
    try:
        from ..utils.io import read_json, write_json

        meta_path = Path(result.run_dir) / "meta.json"
        existing = read_json(meta_path, default=None)
        payload: Dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
        payload.setdefault("run_id", result.run_id)
        payload.setdefault("cve_id", config.cve_id)
        payload.setdefault("model", config.model)
        payload.setdefault("trial", int(config.trial))
        payload["adapter"] = payload.get("adapter") or result.adapter
        payload["terminated_by"] = result.terminated_by
        payload.setdefault("budget_usd", config.budget_usd)
        payload.setdefault("max_hours", config.max_hours)
        payload["fixture_derived"] = bool(
            payload.get("fixture_derived", result.fixture_derived)
        )
        if result.terminated_by in {"harness_error", "provider_error", "budget_abort", "watchdog_24h"}:
            payload.setdefault("infrastructure_failure", result.terminated_by in {"harness_error", "provider_error"})
        write_json(meta_path, payload)
    except OSError:  # pragma: no cover - defensive
        LOG.warning("could not write meta.json for %s", result.run_id)


# ---------------------------------------------------------------------------
# matrix driver
# ---------------------------------------------------------------------------
def run_matrix(
    cves: Iterable[str],
    models: Sequence[str],
    trials: int = 3,
    *,
    adapter: str = DEFAULT_ADAPTER,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    fixtures: Optional[Path] = None,
    budget_usd: float = DEFAULT_BUDGET_USD,
    max_hours: float = DEFAULT_MAX_HOURS,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run the full ``|CVEs| x |models| x trials`` matrix sequentially.

    Returns one result dict per attempt (same shape as :func:`run_trial`).
    ``limit`` caps the number of attempts for pilot runs.
    """

    results: List[Dict[str, Any]] = []
    for cve in cves:
        for model in models:
            for trial in range(1, int(trials) + 1):
                if limit is not None and len(results) >= int(limit):
                    return results
                config = RunConfig(
                    cve_id=str(cve),
                    model=str(model),
                    trial=trial,
                    adapter=adapter,
                    runs_dir=Path(runs_dir),
                    fixture_dir=Path(fixtures) if fixtures is not None else None,
                    budget_usd=budget_usd,
                    max_hours=max_hours,
                )
                results.append(run_trial(config))
    return results


def iter_runs(
    cves: Iterable[str],
    models: Sequence[str],
    trials: int = 3,
) -> Iterator[RunConfig]:
    """Yield :class:`RunConfig` objects for the matrix without executing them."""

    for cve in cves:
        for model in models:
            for trial in range(1, int(trials) + 1):
                yield RunConfig(cve_id=str(cve), model=str(model), trial=trial)
