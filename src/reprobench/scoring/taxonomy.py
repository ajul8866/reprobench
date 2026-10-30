"""Failure taxonomy attribution for ReproBench (paper Table 7).

Implements *first-match* attribution of a failed run to one of the twelve
taxonomy modes (M01--M12) declared in ``config/failure_taxonomy.yaml``, plus an
extension slot for future ``X*`` modes.

Attribution order (first match wins):

1. Pre-phase rules, evaluated in the order declared in the config:
   ``M09 infrastructure_failure`` -> ``M10 policy_refusal`` ->
   ``M11 premature_abandonment`` -> ``M08 simulation_substitution``.
2. Phase mapping: the earliest phase ``p`` in canonical order (skipping ``P1``)
   whose score is below ``threshold_fraction * max_points`` decides the mode.
   ``P5``/``P6`` are overridden to ``M08`` when gating fired.
3. Default: ``M12 ambiguous``.

All identifiers, thresholds and predicate strings live in
``config/failure_taxonomy.yaml``; this module contains no hardcoded mode ids
beyond the defensive fallback used when the config cannot be loaded.

The published Table 7 distribution is exposed by :func:`published_distribution`
for the analysis layer / claims ledger, and is *never* used to score a run.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..utils.io import pct

CONFIG_NAME = "failure_taxonomy.yaml"

#: Defensive fallback (config is shipped with the repo; this only guards
#: against a broken/missing install so imports never fail).
DEFAULT_THRESHOLD_FRACTION = 0.5
DEFAULT_MODE = "M12"

__all__ = [
    "CONFIG_NAME",
    "DEFAULT_MODE",
    "DEFAULT_THRESHOLD_FRACTION",
    "Taxonomy",
    "attribute",
    "attribute_failure",
    "attribute_run",
    "attribution_for",
    "families",
    "first_match_phase",
    "load_taxonomy",
    "mode_by_id",
    "modes",
    "published_distribution",
    "unsuccessful_phases",
]


# ---------------------------------------------------------------------------
# config loading
# ---------------------------------------------------------------------------
def _load_config(name: str) -> Dict[str, Any]:
    """Load a YAML config by name, returning ``{}`` on failure."""
    try:  # pragma: no cover - trivial import guard
        from ..utils.io import load_config

        data = load_config(name)
        return data if isinstance(data, dict) else {}
    except Exception:  # pragma: no cover - defensive
        return {}


class Taxonomy:
    """Loaded view over ``config/failure_taxonomy.yaml``.

    Parsed once and cached in :func:`load_taxonomy`; consumers may also build it
    directly from a dict (useful in tests / offline self-test).
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = dict(config or {})
        self.families: List[Dict[str, Any]] = [
            dict(f) for f in (self.config.get("families") or []) if isinstance(f, Mapping)
        ]
        self.modes: List[Dict[str, Any]] = [
            dict(m) for m in (self.config.get("modes") or []) if isinstance(m, Mapping)
        ]
        self.extension: Dict[str, Any] = dict(self.config.get("extension") or {})
        self.attribution_cfg: Dict[str, Any] = dict(self.config.get("attribution") or {})
        self.distribution_cfg: Dict[str, Any] = dict(self.config.get("distribution") or {})

        self.by_id: Dict[str, Dict[str, Any]] = {}
        for mode in self.modes:
            mid = mode.get("id")
            if mid:
                self.by_id[str(mid)] = mode
        for mode in self.extension.get("registered") or []:
            if isinstance(mode, Mapping) and mode.get("id"):
                self.by_id[str(mode["id"])] = dict(mode)

        self.pre_phase_rules: List[Dict[str, Any]] = [
            dict(r) for r in (self.attribution_cfg.get("pre_phase_rules") or []) if isinstance(r, Mapping)
        ]
        self.phase_mapping: Dict[str, Any] = dict(self.attribution_cfg.get("phase_mapping") or {})
        self.gated_overrides: Dict[str, str] = {
            str(k): str(v) for k, v in (self.attribution_cfg.get("gated_overrides") or {}).items()
        }
        self.default_mode: str = str(self.attribution_cfg.get("default_mode") or DEFAULT_MODE)
        self.threshold_fraction: float = float(
            self.phase_mapping.get("threshold_fraction", DEFAULT_THRESHOLD_FRACTION)
        )
        self.canonical_order: List[str] = [
            str(p) for p in (self.phase_mapping.get("canonical_order") or ["P1", "P2", "P3", "P4", "P5", "P6"])
        ]
        self.skip_phases: List[str] = [str(p) for p in (self.phase_mapping.get("skip_phases") or ["P1"])]

    # -- lookups ---------------------------------------------------------
    def mode(self, mode_id: Optional[str]) -> Dict[str, Any]:
        """Return the mode record for ``mode_id`` (empty dict if unknown)."""
        if not mode_id:
            return {}
        return dict(self.by_id.get(str(mode_id), {}))

    def family_of(self, mode_id: Optional[str]) -> str:
        mode = self.mode(mode_id)
        if mode.get("family"):
            return str(mode["family"])
        return "ambiguous"

    def family(self, family_id: Optional[str]) -> Dict[str, Any]:
        for fam in self.families:
            if str(fam.get("id")) == str(family_id):
                return dict(fam)
        return {}

    def mode_ids(self) -> List[str]:
        return list(self.by_id.keys())

    # -- phase helpers ---------------------------------------------------
    def phase_candidates(self, phase: str) -> List[str]:
        spec = self.phase_mapping.get(str(phase)) or {}
        return [str(c) for c in (spec.get("candidates") or [])]

    def phase_max(self, phase: str) -> float:
        """Phase maxima from ``config/phases.yaml`` (fallback 15/15/15/15/20/20)."""
        maxima = _phase_maxima()
        return float(maxima.get(str(phase), 15.0))

    def phase_failed(self, phase: str, score: float) -> bool:
        return float(score) < self.threshold_fraction * self.phase_max(phase)


_PHASE_MAXIMA_CACHE: Optional[Dict[str, float]] = None


def _phase_maxima() -> Dict[str, float]:
    global _PHASE_MAXIMA_CACHE
    if _PHASE_MAXIMA_CACHE is not None:
        return _PHASE_MAXIMA_CACHE
    maxima: Dict[str, float] = {p: m for p, m in zip(
        ["P1", "P2", "P3", "P4", "P5", "P6"], [15.0, 15.0, 15.0, 15.0, 20.0, 20.0]
    )}
    try:
        from ..utils.io import load_config

        cfg = load_config("phases")
        derived = (cfg or {}).get("derived") or {}
        values = derived.get("max_points")
        ids = derived.get("phase_ids") or ["P1", "P2", "P3", "P4", "P5", "P6"]
        if isinstance(values, Sequence) and len(values) == len(ids):
            maxima = {str(p): float(v) for p, v in zip(ids, values)}
        else:
            parsed: Dict[str, float] = {}
            for ph in (cfg or {}).get("phases") or []:
                pid = ph.get("id")
                if pid and ph.get("max_points") is not None:
                    parsed[str(pid)] = float(ph["max_points"])
            if parsed:
                maxima = parsed
    except Exception:  # pragma: no cover - defensive
        pass
    _PHASE_MAXIMA_CACHE = maxima
    return maxima


_TAXONOMY_CACHE: Optional[Taxonomy] = None


def load_taxonomy(config: Optional[Mapping[str, Any]] = None, refresh: bool = False) -> Taxonomy:
    """Load (and cache) the taxonomy configuration."""
    global _TAXONOMY_CACHE
    if config is not None:
        return Taxonomy(config)
    if _TAXONOMY_CACHE is None or refresh:
        _TAXONOMY_CACHE = Taxonomy(_load_config(CONFIG_NAME))
    return _TAXONOMY_CACHE


def modes(taxonomy: Optional[Taxonomy] = None) -> List[Dict[str, Any]]:
    """Return the declared mode records in config order."""
    return list((taxonomy or load_taxonomy()).modes)


def families(taxonomy: Optional[Taxonomy] = None) -> List[Dict[str, Any]]:
    """Return the declared failure families."""
    return list((taxonomy or load_taxonomy()).families)


def mode_by_id(mode_id: str, taxonomy: Optional[Taxonomy] = None) -> Dict[str, Any]:
    """Look up a mode record by id."""
    return (taxonomy or load_taxonomy()).mode(mode_id)


def published_distribution(taxonomy: Optional[Taxonomy] = None) -> Dict[str, Any]:
    """Return the published Table 7 distribution (test/analysis data only)."""
    cfg = (taxonomy or load_taxonomy()).distribution_cfg
    counts = dict(cfg.get("counts") or {})
    percentages = dict(cfg.get("percentages") or {})
    total = float(cfg.get("total_runs") or 450) or 450.0
    # Recompute percentages so the analysis layer never trusts stale literals.
    recomputed = {name: pct(float(value), total) for name, value in counts.items()}
    return {
        "total_runs": total,
        "counts": counts,
        "percentages": percentages,
        "recomputed_percentages": recomputed,
        "sum_counts": sum(float(v) for v in counts.values()),
        "published_percentage_sum": cfg.get("published_percentage_sum"),
        "residual_pct": cfg.get("residual_pct"),
        "residual_note": cfg.get("residual_note", ""),
    }


# ---------------------------------------------------------------------------
# tolerant accessors (pydantic models, dicts, or plain objects)
# ---------------------------------------------------------------------------
def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        if name in obj:
            return obj[name]
        return default
    value = getattr(obj, name, default)
    return default if value is None else value


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return float(default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _phase_scores(run: Any) -> Dict[str, float]:
    """Extract ``{"P1": score, ...}`` from a run / score record."""
    phases: Dict[str, float] = {}
    raw = _get(run, "phases")
    if isinstance(raw, Mapping):
        for pid, value in raw.items():
            if isinstance(value, Mapping) or hasattr(value, "score"):
                phases[str(pid)] = _as_float(_get(value, "score"))
            else:
                phases[str(pid)] = _as_float(value)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for entry in raw:
            pid = _get(entry, "phase")
            if pid:
                phases[str(pid)] = _as_float(_get(entry, "score"))
    if not phases:
        raw_scores = _get(run, "phase_scores")
        if isinstance(raw_scores, Mapping):
            phases = {str(k): _as_float(v) for k, v in raw_scores.items()}
        elif isinstance(raw_scores, Sequence) and not isinstance(raw_scores, (str, bytes)):
            ids = ["P1", "P2", "P3", "P4", "P5", "P6"]
            for idx, value in enumerate(raw_scores):
                if idx < len(ids):
                    phases[ids[idx]] = _as_float(value)
    return phases


def _gating(run: Any) -> Any:
    gating = _get(run, "gating")
    if gating is None:
        gating = _get(run, "gating_verdict")
    return gating


def _gated_phases(run: Any) -> List[str]:
    gating = _gating(run)
    phases = _get(gating, "gated_phases")
    if isinstance(phases, Sequence) and not isinstance(phases, (str, bytes)):
        return [str(p) for p in phases]
    return []


def _simulation_detected(run: Any) -> bool:
    gating = _gating(run)
    if _as_bool(_get(gating, "gated")):
        return True
    gated = _gated_phases(run)
    return "P5" in gated or "P6" in gated


# ---------------------------------------------------------------------------
# predicate evaluation for the config's `when` / `prefer_when` strings
# ---------------------------------------------------------------------------
def _resolve(path: str, run: Any, ctx: Dict[str, Any]) -> Any:
    """Resolve ``a.b.c`` against the run record / context dict."""
    path = path.strip()
    if not path:
        return None
    parts = path.split(".")
    root = parts[0]
    rest = parts[1:]
    if root in ctx:
        value: Any = ctx[root]
    else:
        value = _get(run, root)
        if value is None:
            value = ctx.get(root)
    for part in rest:
        value = _get(value, part)
    return value


def _split_bool_expr(expr: str) -> Tuple[str, List[str]]:
    """Split a boolean expression into (joiner, atoms)."""
    expr = expr.strip()
    lower = expr.lower()
    if " or " in lower:
        return "or", re.split(r"\s+or\s+", expr, flags=re.IGNORECASE)
    if " and " in lower:
        return "and", re.split(r"\s+and\s+", expr, flags=re.IGNORECASE)
    return "and", [expr]


def _eval_atom(atom: str, run: Any, ctx: Dict[str, Any]) -> bool:
    atom = atom.strip().strip("()").strip()
    if not atom:
        return True
    # `X contains Y`
    match = re.match(r"^(?P<lhs>[\w\.]+)\s+contains\s+(?P<rhs>.+)$", atom, re.IGNORECASE)
    if match:
        haystack = _resolve(match.group("lhs"), run, ctx)
        needle = match.group("rhs").strip().strip("'\"")
        if isinstance(haystack, Mapping):
            return needle in {str(k) for k in haystack.keys()}
        if isinstance(haystack, Sequence) and not isinstance(haystack, (str, bytes)):
            return needle in [str(item) for item in haystack]
        return needle in str(haystack or "")
    # `X in {a,b}` / `X in [a, b]`
    match = re.match(r"^(?P<lhs>[\w\.]+)\s+in\s+\{?(?P<rhs>[^\}]+)\}?$", atom, re.IGNORECASE)
    if match and not re.match(r"^\s*[\w\.]+\s*==", atom):
        value = str(_resolve(match.group("lhs"), run, ctx))
        options = [o.strip().strip("'\"") for o in match.group("rhs").split(",")]
        return value in options
    # `X == Y` / `X != Y`
    match = re.match(r"^(?P<lhs>[\w\.]+)\s*(?P<op>==|!=)\s*(?P<rhs>.+)$", atom)
    if match:
        lhs = _resolve(match.group("lhs"), run, ctx)
        rhs = match.group("rhs").strip().strip("'\"")
        if rhs.lower() in {"true", "false"}:
            equal = _as_bool(lhs) == (rhs.lower() == "true")
        elif re.match(r"^-?\d+(\.\d+)?$", rhs):
            equal = _as_float(lhs) == float(rhs)
        elif rhs.lower() in {"null", "none"}:
            equal = lhs is None
        else:
            equal = str(lhs) == rhs
        return equal if match.group("op") == "==" else not equal
    # bare truthiness
    return _as_bool(_resolve(atom, run, ctx))


def eval_predicate(expr: str, run: Any, ctx: Optional[Mapping[str, Any]] = None) -> bool:
    """Evaluate a config predicate string against a run record.

    Supports ``==``, ``!=``, ``contains``, ``in``, bare truthiness, and a single
    level of ``and`` / ``or`` composition. Unknown expressions degrade to
    ``False`` so attribution never raises on a config typo.
    """
    if not expr:
        return False
    context: Dict[str, Any] = dict(ctx or {})
    joiner, atoms = _split_bool_expr(str(expr))
    values = [_eval_atom(atom, run, context) for atom in atoms]
    if not values:
        return False
    return all(values) if joiner == "and" else any(values)


def _pre_phase_ctx(run: Any) -> Dict[str, Any]:
    trace = _get(run, "trace")
    if trace is None:
        trace = _get(run, "trace_summary") or {}
    gating = _gating(run)
    if gating is None:
        gating = {}
    evidence = _get(run, "evidence") or {}
    return {
        "run": run,
        "trace": trace,
        "gating": gating,
        "evidence": evidence,
    }


# ---------------------------------------------------------------------------
# attribution
# ---------------------------------------------------------------------------
def first_match_phase(
    run: Any,
    taxonomy: Optional[Taxonomy] = None,
) -> Tuple[Optional[str], float]:
    """Return the earliest sub-threshold phase (skipping configured phases)."""
    tax = taxonomy or load_taxonomy()
    scores = _phase_scores(run)
    for phase in tax.canonical_order:
        if phase in tax.skip_phases:
            continue
        if phase not in scores:
            continue
        if tax.phase_failed(phase, scores[phase]):
            return phase, float(scores[phase])
    return None, 0.0


def unsuccessful_phases(run: Any, taxonomy: Optional[Taxonomy] = None) -> List[str]:
    """List every phase below the failure threshold, in canonical order."""
    tax = taxonomy or load_taxonomy()
    scores = _phase_scores(run)
    return [
        phase
        for phase in tax.canonical_order
        if phase not in tax.skip_phases and phase in scores and tax.phase_failed(phase, scores[phase])
    ]


def _phase_mode(
    phase: str,
    run: Any,
    taxonomy: Taxonomy,
    ctx: Dict[str, Any],
) -> Optional[str]:
    """Pick the mode for a failing phase, honouring gated overrides."""
    if phase in taxonomy.gated_overrides and _simulation_detected(run):
        return taxonomy.gated_overrides[phase]
    spec = taxonomy.phase_mapping.get(phase) or {}
    override = spec.get("override_when_gated")
    if override and _simulation_detected(run):
        return str(override)
    candidates = taxonomy.phase_candidates(phase)
    if not candidates:
        return None
    prefer = spec.get("prefer_when") or {}
    if isinstance(prefer, Mapping):
        for mode_id, predicate in prefer.items():
            if eval_predicate(str(predicate), run, ctx):
                # `prefer_when` names the mode that wins when its predicate holds
                if str(mode_id) in candidates or str(mode_id) in taxonomy.by_id:
                    return str(mode_id)
    return candidates[0]


def attribute_run(
    run: Any,
    taxonomy: Optional[Taxonomy] = None,
    *,
    mode: Optional[str] = None,
    family: Optional[str] = None,
    phase: Optional[str] = None,
    rule: str = "",
    rationale: str = "",
) -> Dict[str, Any]:
    """Attribute ``run`` to a taxonomy mode, returning a plain dict.

    The returned dict is shaped to populate ``schema.FailureAttribution``:
    ``mode``, ``mode_name``, ``family``, ``phase``, ``rule``, ``rationale``,
    ``gated``. An explicit ``mode`` argument short-circuits the first-match
    engine (used by the judge / manual review).
    """
    tax = taxonomy or load_taxonomy()
    ctx = _pre_phase_ctx(run)

    if mode:
        record = tax.mode(mode)
        return {
            "mode": str(mode),
            "mode_name": str(record.get("name") or record.get("id") or mode),
            "family": str(family or record.get("family") or "ambiguous"),
            "phase": phase if phase is not None else record.get("applies_to_phase"),
            "rule": rule or "explicit",
            "rationale": rationale,
            "gated": _simulation_detected(run),
        }

    # 1) pre-phase rules, in declared order
    for rule_cfg in tax.pre_phase_rules:
        expr = str(rule_cfg.get("when") or "")
        try:
            matched = eval_predicate(expr, run, ctx)
        except Exception:  # pragma: no cover - defensive
            matched = False
        if matched:
            mode_id = str(rule_cfg.get("mode") or "")
            record = tax.mode(mode_id)
            return {
                "mode": mode_id,
                "mode_name": str(record.get("name") or mode_id),
                "family": str(record.get("family") or "ambiguous"),
                "phase": record.get("applies_to_phase"),
                "rule": str(rule_cfg.get("id") or "pre_phase"),
                "rationale": f"pre-phase rule matched: {expr}",
                "gated": _simulation_detected(run),
            }

    # 2) earliest sub-threshold phase
    failing_phase, failing_score = first_match_phase(run, tax)
    if failing_phase:
        mode_id = _phase_mode(failing_phase, run, tax, ctx)
        if mode_id:
            record = tax.mode(mode_id)
            gated = _simulation_detected(run) and failing_phase in tax.gated_overrides
            return {
                "mode": mode_id,
                "mode_name": str(record.get("name") or mode_id),
                "family": str(record.get("family") or "ambiguous"),
                "phase": failing_phase,
                "rule": f"phase_threshold:{failing_phase}",
                "rationale": (
                    f"{failing_phase} scored {failing_score:.2f} < "
                    f"{tax.threshold_fraction:.1f} x {tax.phase_max(failing_phase):.0f}"
                    + (" (gated override)" if gated else "")
                ),
                "gated": gated,
            }

    # 3) default
    record = tax.mode(tax.default_mode)
    return {
        "mode": tax.default_mode,
        "mode_name": str(record.get("name") or tax.default_mode),
        "family": str(record.get("family") or "ambiguous"),
        "phase": record.get("applies_to_phase"),
        "rule": "default",
        "rationale": "no pre-phase rule or sub-threshold phase matched",
        "gated": _simulation_detected(run),
    }


# Friendly aliases used across the codebase / tests.
def attribute(run: Any, taxonomy: Optional[Taxonomy] = None, **kwargs: Any) -> Dict[str, Any]:
    """Alias of :func:`attribute_run`."""
    return attribute_run(run, taxonomy, **kwargs)


def attribute_failure(run: Any, taxonomy: Optional[Taxonomy] = None, **kwargs: Any) -> Dict[str, Any]:
    """Alias of :func:`attribute_run` (explicit failure nomenclature)."""
    return attribute_run(run, taxonomy, **kwargs)


def attribution_for(run: Any, taxonomy: Optional[Taxonomy] = None, **kwargs: Any) -> Dict[str, Any]:
    """Alias of :func:`attribute_run`."""
    return attribute_run(run, taxonomy, **kwargs)
