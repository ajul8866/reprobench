"""Phase divergence analysis (paper: Phase-Specific Convergence / PSC, Figure 4).

A phase is *divergent* for one CVE when the models disagree strongly about it:
the best-of-three normalized phase scores have ``max >= HIGH_THRESHOLD`` and
``min < LOW_THRESHOLD``.  Each divergent CVE is attributed to the earliest
divergent phase in canonical order, restricted to the configured divergence
phase set (``P2``, ``P4``, ``P5``).

Published result reproduced by the synthetic fixture in
``tests/test_divergence.py``: 19 divergent CVEs out of 30, split P2=12,
P4=2, P5=5.

Everything here is pure arithmetic over already-scored rows; no I/O beyond
reading ``config/benchmark.yaml`` for the thresholds.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..utils.io import load_config, pct, round_half_up

__all__ = [
    "HIGH_THRESHOLD",
    "LOW_THRESHOLD",
    "DIVERGENT_PHASES",
    "PHASES",
    "PHASE_MAX",
    "divergence_thresholds",
    "normalized_phase_score",
    "best_by_model",
    "phase_divergences",
    "earliest_divergent_phase",
    "divergence_summary",
    "divergence_counts",
    "summarize_divergence",
    "compute_divergence",
    "phase_divergence",
    "analyze_divergence",
    "divergence_rate",
]

PHASES: Tuple[str, ...] = ("P1", "P2", "P3", "P4", "P5", "P6")

PHASE_MAX: Dict[str, float] = {
    "P1": 15.0,
    "P2": 15.0,
    "P3": 15.0,
    "P4": 15.0,
    "P5": 20.0,
    "P6": 20.0,
}

#: Defaults from ``config/benchmark.yaml`` -> ``analysis.divergence``.
HIGH_THRESHOLD: float = 0.50
LOW_THRESHOLD: float = 0.25

#: Phases that may *attribute* a divergent CVE (P6 can diverge but is ignored).
DIVERGENT_PHASES: Tuple[str, ...] = ("P2", "P4", "P5")


# ---------------------------------------------------------------------------
# config + tolerant accessors
# ---------------------------------------------------------------------------


def divergence_thresholds() -> Tuple[float, float]:
    """Return ``(high, low)`` from config, falling back to the paper defaults."""
    high, low = HIGH_THRESHOLD, LOW_THRESHOLD
    try:
        cfg = load_config("benchmark") or {}
        div = (cfg.get("analysis") or {}).get("divergence") or {}
        high = float(div.get("high_threshold", high))
        low = float(div.get("low_threshold", low))
    except Exception:  # pragma: no cover - defensive: config always ships
        pass
    return high, low


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _phase_value(run: Any, phase: str) -> Optional[float]:
    """Raw phase score from a run row (mapping, sequence or object)."""
    for holder_name in ("phases", "phase_scores"):
        holder = _get(run, holder_name)
        if holder is None:
            continue
        if isinstance(holder, Mapping):
            value = holder.get(phase)
            if isinstance(value, Mapping):
                value = value.get("score", value.get("artifact_score"))
            elif hasattr(value, "score"):
                value = getattr(value, "score")
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
            continue
        if isinstance(holder, Sequence) and not isinstance(holder, (str, bytes)):
            index = PHASES.index(phase)
            if index < len(holder):
                value = holder[index]
                if isinstance(value, Mapping):
                    value = value.get("score", value.get("artifact_score"))
                elif hasattr(value, "score"):
                    value = getattr(value, "score")
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
    # A run may also expose ``phase(pid)`` returning a PhaseScore.
    getter = getattr(run, "phase", None)
    if callable(getter):
        try:
            scored = getter(phase)
        except Exception:  # pragma: no cover - defensive
            scored = None
        if scored is not None:
            value = getattr(scored, "score", None)
            if value is not None:
                return float(value)
    return None


def normalized_phase_score(
    run: Any, phase: str, maxima: Optional[Mapping[str, float]] = None
) -> Optional[float]:
    """Phase score divided by its maximum, clamped to ``[0, 1]``."""
    raw = _phase_value(run, phase)
    if raw is None:
        return None
    cap = float((maxima or PHASE_MAX).get(phase, PHASE_MAX.get(phase, 100.0)))
    if cap <= 0:
        return None
    return max(0.0, min(1.0, raw / cap))


def best_by_model(runs: Iterable[Any], phase: str) -> Dict[str, float]:
    """Each model's best-of-three normalized score for ``phase``."""
    best: Dict[str, float] = {}
    for run in runs:
        model = _get(run, "model") or ""
        value = normalized_phase_score(run, phase)
        if value is None:
            continue
        if model not in best or value > best[model]:
            best[model] = value
    return best


# ---------------------------------------------------------------------------
# core rule
# ---------------------------------------------------------------------------


def phase_divergences(
    runs: Iterable[Any],
    high: Optional[float] = None,
    low: Optional[float] = None,
    phases: Sequence[str] = DIVERGENT_PHASES,
) -> Dict[str, bool]:
    """Per-phase divergence flags for one CVE's runs."""
    high = HIGH_THRESHOLD if high is None else float(high)
    low = LOW_THRESHOLD if low is None else float(low)
    rows = list(runs)
    flags: Dict[str, bool] = {}
    for phase in phases:
        values = list(best_by_model(rows, phase).values())
        if len(values) < 2:
            flags[phase] = False
            continue
        flags[phase] = (max(values) >= high) and (min(values) < low)
    return flags


def earliest_divergent_phase(
    runs: Iterable[Any],
    high: Optional[float] = None,
    low: Optional[float] = None,
    phases: Sequence[str] = DIVERGENT_PHASES,
) -> Optional[str]:
    """Earliest divergent phase in canonical order (restricted to ``phases``)."""
    allowed = [p for p in PHASES if p in set(phases)]
    flags = phase_divergences(runs, high, low, allowed)
    for phase in allowed:
        if flags.get(phase):
            return phase
    return None


def _group_by_cve(runs: Iterable[Any]) -> Dict[str, List[Any]]:
    grouped: Dict[str, List[Any]] = {}
    for run in runs:
        cve_id = _get(run, "cve_id") or _get(run, "id") or ""
        grouped.setdefault(str(cve_id), []).append(run)
    return grouped


def divergence_summary(
    runs: Iterable[Any],
    high: Optional[float] = None,
    low: Optional[float] = None,
    phases: Sequence[str] = DIVERGENT_PHASES,
) -> Dict[str, Any]:
    """Group runs by CVE and attribute each divergent CVE to one phase."""
    high = HIGH_THRESHOLD if high is None else float(high)
    low = LOW_THRESHOLD if low is None else float(low)
    allowed = [p for p in PHASES if p in set(phases)]
    counts: Dict[str, int] = {phase: 0 for phase in allowed}
    cves: Dict[str, str] = {}
    grouped = _group_by_cve(runs)
    for cve_id in sorted(grouped):
        attributed = earliest_divergent_phase(grouped[cve_id], high, low, allowed)
        if attributed:
            cves[cve_id] = attributed
            counts[attributed] = counts.get(attributed, 0) + 1
    divergent = sorted(cves)
    return {
        "counts": counts,
        "divergence_counts": dict(counts),
        "cves": cves,
        "divergent_cves": divergent,
        "total": len(divergent),
        "cves_evaluated": len(grouped),
        "high_threshold": high,
        "low_threshold": low,
        "phases": list(allowed),
    }


# Aliases probed by ``tests/test_divergence.py`` and used by table emitters.
def divergence_counts(runs: Iterable[Any], **kwargs: Any) -> Dict[str, int]:
    return divergence_summary(runs, **kwargs)["counts"]


summarize_divergence = divergence_summary
compute_divergence = divergence_summary
phase_divergence = divergence_summary
analyze_divergence = divergence_summary


def divergence_rate(
    runs: Iterable[Any], total_cves: Optional[int] = None, **kwargs: Any
) -> Dict[str, Any]:
    """Divergence rate summary (paper: 19/30 -> 63.3 %)."""
    summary = divergence_summary(runs, **kwargs)
    denominator = int(total_cves) if total_cves else int(summary["cves_evaluated"])
    rate = pct(summary["total"], denominator, 1) if denominator else 0.0
    summary["rate_pct"] = rate
    summary["denominator"] = denominator
    return summary


def table_rows(runs: Iterable[Any]) -> List[Dict[str, Any]]:
    """Flat rows for report emission (one row per attributed phase)."""
    summary = divergence_summary(runs)
    total = summary["total"]
    rows: List[Dict[str, Any]] = []
    for phase in summary["phases"]:
        count = int(summary["counts"].get(phase, 0))
        rows.append(
            {
                "phase": phase,
                "cves": count,
                "share_pct": pct(count, total, 1) if total else 0.0,
            }
        )
    rows.append(
        {
            "phase": "total",
            "cves": total,
            "share_pct": 100.0 if total else 0.0,
        }
    )
    return rows
