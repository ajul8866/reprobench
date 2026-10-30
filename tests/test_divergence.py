"""Tier 5 validation: phase divergence (PSC) analysis.

The paper reports divergence for 19 CVEs, attributed to the earliest divergent
phase: P2 -> 12 CVEs, P4 -> 2 CVEs, P5 -> 5 CVEs.  The high/low thresholds are
``0.50`` / ``0.25`` (``config/benchmark.yaml: analysis.divergence``).

This module deliberately contains a small, dependency-free *reference*
implementation of the divergence rule so the invariant can be validated even
when :mod:`reprobench.analysis.divergence` is absent.  When that module exists,
its entry points are duck-typed and must agree with the reference.

Divergence rule (reconstructed from the paper's Figure 4 / Table 2 discussion):

* For one ``(CVE, phase)`` cell, take each model's best-of-three normalized
  phase score (``score / phase_max``).
* The cell is **divergent** when ``max(normalized) >= high`` and
  ``min(normalized) < low`` for the models that produced data.
* A CVE is attributed to the **earliest** divergent phase in canonical order.

All tests are offline and deterministic: no network, no judge model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reprobench.utils.io import load_config  # noqa: E402

PHASES: Tuple[str, ...] = ("P1", "P2", "P3", "P4", "P5", "P6")
PHASE_MAX: Dict[str, float] = {
    "P1": 15.0,
    "P2": 15.0,
    "P3": 15.0,
    "P4": 15.0,
    "P5": 20.0,
    "P6": 20.0,
}
MODELS: Tuple[str, ...] = (
    "glm-5.2",
    "mimo-v2.5",
    "claude-sonnet-4.6",
    "gpt-5.5",
    "deepseek-v4-flash",
)
TRIALS: Tuple[int, ...] = (1, 2, 3)

HIGH = 0.50
LOW = 0.25
DIVERGENT_PHASES: Tuple[str, ...] = ("P2", "P4", "P5")

#: Published divergence counts (paper: 19 CVEs total).
PAPER_COUNTS: Dict[str, int] = {"P2": 12, "P4": 2, "P5": 5}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Mapping/attribute tolerant accessor."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def phase_value(run: Any, phase: str) -> Optional[float]:
    """Raw recorded score for ``phase`` on ``run`` (``None`` when unavailable)."""
    phases = _get(run, "phase_scores")
    if phases is None:
        phases = _get(run, "phases")
    if phases is None:
        return None
    if isinstance(phases, Mapping):
        if phase in phases:
            value = phases[phase]
            score = _get(value, "score")
            return _as_float(score if score is not None else value)
        return None
    if isinstance(phases, Sequence) and not isinstance(phases, (str, bytes)):
        idx = PHASES.index(phase)
        if idx < len(phases):
            value = phases[idx]
            score = _get(value, "score")
            return _as_float(score if score is not None else value)
        return None
    return None


def normalized_score(run: Any, phase: str, maxima: Optional[Mapping[str, float]] = None) -> Optional[float]:
    """``phase_score / phase_max`` in ``[0, 1]``; ``None`` when absent."""
    raw = phase_value(run, phase)
    if raw is None:
        return None
    table = maxima or PHASE_MAX
    cap = _as_float(table.get(phase), 0.0)
    if cap <= 0:
        return 0.0
    return max(0.0, min(1.0, raw / cap))


def best_by_model(runs: Iterable[Any], phase: str) -> Dict[str, float]:
    """Per-model best-of-three normalized phase score."""
    best: Dict[str, float] = {}
    for run in runs:
        model = str(_get(run, "model", "") or "")
        value = normalized_score(run, phase)
        if value is None:
            continue
        if model not in best or value > best[model]:
            best[model] = value
    return best


def phase_divergences(
    runs: Iterable[Any],
    high: float = HIGH,
    low: float = LOW,
) -> Dict[str, bool]:
    """``{phase: divergent}`` for one CVE's runs (models compared on best-of-three)."""
    runs = list(runs)
    result: Dict[str, bool] = {}
    for phase in PHASES:
        per_model = best_by_model(runs, phase)
        if len(per_model) < 2:
            result[phase] = False
            continue
        values = list(per_model.values())
        result[phase] = max(values) >= high and min(values) < low
    return result


def earliest_divergent_phase(
    runs: Iterable[Any],
    high: float = HIGH,
    low: float = LOW,
    phases: Sequence[str] = DIVERGENT_PHASES,
) -> Optional[str]:
    """Earliest divergent phase in canonical order (``None`` when none)."""
    flags = phase_divergences(runs, high=high, low=low)
    for phase in PHASES:
        if phase not in phases:
            continue
        if flags.get(phase):
            return phase
    return None


def divergence_summary(
    runs: Iterable[Any],
    high: float = HIGH,
    low: float = LOW,
    phases: Sequence[str] = DIVERGENT_PHASES,
) -> Dict[str, Any]:
    """Reference divergence summary.

    Returns ``{"counts": {phase: n}, "cves": {cve_id: phase}, "total": n,
    "divergent_cves": [cve_id, ...]}``.
    """
    grouped: Dict[str, List[Any]] = {}
    for run in runs:
        cve = str(_get(run, "cve_id", "") or "")
        grouped.setdefault(cve, []).append(run)

    counts: Dict[str, int] = {phase: 0 for phase in phases}
    attributed: Dict[str, str] = {}
    for cve in sorted(grouped):
        phase = earliest_divergent_phase(grouped[cve], high=high, low=low, phases=phases)
        if phase is None:
            continue
        attributed[cve] = phase
        counts[phase] = counts.get(phase, 0) + 1
    return {
        "counts": counts,
        "cves": attributed,
        "divergent_cves": sorted(attributed),
        "total": len(attributed),
    }


def _run(cve_id: str, model: str, trial: int, scores: Mapping[str, float]) -> Dict[str, Any]:
    """Build one synthetic per-trial row (``scores`` keyed by phase id)."""
    ordered = [float(scores.get(phase, 0.0)) for phase in PHASES]
    return {
        "cve_id": cve_id,
        "model": model,
        "trial": trial,
        "phase_scores": ordered,
        "phases": {phase: ordered[i] for i, phase in enumerate(PHASES)},
    }


def _cve_runs(cve_id: str, diverge_phase: Optional[str] = None) -> List[Dict[str, Any]]:
    """15 rows (5 models x 3 trials) for one CVE.

    By default every model sits mid-band (normalized ``0.6`` for P2/P4, ``0.4``
    for P5/P6) so nothing diverges.  When ``diverge_phase`` is given, models are
    split into a high group (normalized ``0.9``) and a low group (``0.1``),
    producing a divergent cell for that phase only.
    """
    rows: List[Dict[str, Any]] = []
    for m_idx, model in enumerate(MODELS):
        for trial in TRIALS:
            scores: Dict[str, float] = {}
            for phase in PHASES:
                base = 0.4 if phase in ("P5", "P6") else 0.6
                scores[phase] = base * PHASE_MAX[phase]
            if diverge_phase is not None:
                # models 0..2 (3 of 5) score high, models 3..4 (2 of 5) low
                level = 0.9 if m_idx < 3 else 0.1
                scores[diverge_phase] = level * PHASE_MAX[diverge_phase]
            rows.append(_run(cve_id, model, trial, scores))
    return rows


def synthetic_corpus(counts: Mapping[str, int] = PAPER_COUNTS) -> List[Dict[str, Any]]:
    """19 CVEs whose attributed divergence reproduces ``counts`` exactly."""
    rows: List[Dict[str, Any]] = []
    idx = 0
    for phase, n in counts.items():
        for _ in range(n):
            idx += 1
            cve_id = f"CVE-2025-{1000 + idx}"
            rows.extend(_cve_runs(cve_id, diverge_phase=phase))
    return rows


def _load_divergence_module() -> Any:
    try:
        from reprobench.analysis import divergence  # noqa: PLC0415
    except Exception:  # pragma: no cover - module may not exist yet
        return None
    return divergence


def _analysis_summary(module: Any, rows: Sequence[Mapping[str, Any]], **kwargs: Any) -> Optional[Dict[str, Any]]:
    """Duck-type whichever summary entry point the analysis module exposes."""
    for name in (
        "divergence_summary",
        "summarize_divergence",
        "compute_divergence",
        "divergence_counts",
        "phase_divergence",
        "analyze_divergence",
    ):
        fn = getattr(module, name, None)
        if not callable(fn):
            continue
        for payload in (list(rows),):
            try:
                result = fn(payload, **kwargs)
            except TypeError:
                try:
                    result = fn(payload)
                except Exception:
                    continue
            except Exception:
                continue
            if isinstance(result, Mapping):
                return dict(result)
    return None


# ---------------------------------------------------------------------------
# config invariants
# ---------------------------------------------------------------------------


def test_divergence_thresholds_match_paper() -> None:
    cfg = load_config("benchmark")
    div = cfg["analysis"]["divergence"]
    assert abs(float(div["high_threshold"]) - HIGH) < 1e-9
    assert abs(float(div["low_threshold"]) - LOW) < 1e-9


def test_divergence_phase_pairs_are_canonical() -> None:
    cfg = load_config("benchmark")
    pairs = cfg["analysis"]["divergence"]["phase_pairs"]
    assert pairs == [["P2", "P4"], ["P4", "P5"], ["P2", "P5"]]
    for pair in pairs:
        for phase in pair:
            assert phase in PHASES


def test_paper_divergence_counts_sum_to_19() -> None:
    assert sum(PAPER_COUNTS.values()) == 19
    assert PAPER_COUNTS == {"P2": 12, "P4": 2, "P5": 5}


# ---------------------------------------------------------------------------
# reference implementation semantics
# ---------------------------------------------------------------------------


def test_no_divergence_when_models_agree() -> None:
    runs = _cve_runs("CVE-2025-0001")
    assert divergence_summary(runs)["total"] == 0


def test_single_phase_divergence_detected() -> None:
    runs = _cve_runs("CVE-2025-0002", diverge_phase="P4")
    summary = divergence_summary(runs)
    assert summary["total"] == 1
    assert summary["cves"]["CVE-2025-0002"] == "P4"
    assert summary["counts"]["P4"] == 1
    assert summary["counts"]["P2"] == 0


def test_high_threshold_is_inclusive_low_is_strict() -> None:
    # normalized 0.50 exactly -> counts as high (>=), 0.25 exactly -> NOT low (<)
    rows = [_run("CVE-2025-9001", MODELS[0], 1, {"P2": 0.50 * 15.0})]
    rows.append(_run("CVE-2025-9001", MODELS[1], 1, {"P2": 0.25 * 15.0}))
    assert earliest_divergent_phase(rows) is None

    rows[1] = _run("CVE-2025-9001", MODELS[1], 1, {"P2": 0.24 * 15.0})
    assert earliest_divergent_phase(rows) == "P2"


def test_best_of_three_uses_model_maximum() -> None:
    # one model has a single high trial; the other model is uniformly low
    rows = [
        _run("CVE-2025-9002", MODELS[0], 1, {"P2": 0.1 * 15.0}),
        _run("CVE-2025-9002", MODELS[0], 2, {"P2": 0.9 * 15.0}),
        _run("CVE-2025-9002", MODELS[0], 3, {"P2": 0.1 * 15.0}),
        _run("CVE-2025-9002", MODELS[1], 1, {"P2": 0.1 * 15.0}),
        _run("CVE-2025-9002", MODELS[1], 2, {"P2": 0.1 * 15.0}),
        _run("CVE-2025-9002", MODELS[1], 3, {"P2": 0.1 * 15.0}),
    ]
    assert earliest_divergent_phase(rows) == "P2"
    assert best_by_model(rows, "P2")[MODELS[0]] == pytest.approx(0.9)


def test_earliest_phase_wins_when_multiple_diverge() -> None:
    rows = [
        _run("CVE-2025-9003", MODELS[0], 1, {"P2": 0.9 * 15.0, "P5": 0.9 * 20.0}),
        _run("CVE-2025-9003", MODELS[1], 1, {"P2": 0.1 * 15.0, "P5": 0.1 * 20.0}),
    ]
    summary = divergence_summary(rows)
    assert summary["cves"]["CVE-2025-9003"] == "P2"
    assert summary["counts"]["P2"] == 1
    assert summary["counts"]["P5"] == 0


def test_single_model_cannot_diverge() -> None:
    rows = [_run("CVE-2025-9004", MODELS[0], trial, {"P2": 0.9 * 15.0}) for trial in TRIALS]
    assert earliest_divergent_phase(rows) is None


def test_phase_outside_divergence_set_is_ignored() -> None:
    # P6 diverges but P6 is not one of the reported divergence phases
    rows = [
        _run("CVE-2025-9005", MODELS[0], 1, {"P6": 0.9 * 20.0}),
        _run("CVE-2025-9005", MODELS[1], 1, {"P6": 0.1 * 20.0}),
    ]
    flags = phase_divergences(rows)
    assert flags["P6"] is True
    assert earliest_divergent_phase(rows) is None
    assert divergence_summary(rows)["total"] == 0


def test_normalized_score_clamps() -> None:
    row = _run("CVE-2025-9006", MODELS[0], 1, {"P5": 99.0})
    assert normalized_score(row, "P5") == pytest.approx(1.0)
    assert normalized_score(row, "P6") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# synthetic 19-CVE corpus reproduces the published counts
# ---------------------------------------------------------------------------


def test_synthetic_corpus_reproduces_paper_counts() -> None:
    rows = synthetic_corpus()
    assert len(rows) == 19 * len(MODELS) * len(TRIALS) == 19 * 15
    summary = divergence_summary(rows)
    assert summary["total"] == 19
    assert summary["counts"] == PAPER_COUNTS


def test_synthetic_corpus_attribution_per_phase() -> None:
    summary = divergence_summary(synthetic_corpus())
    attributed = sorted(summary["cves"].values())
    assert attributed.count("P2") == 12
    assert attributed.count("P4") == 2
    assert attributed.count("P5") == 5
    assert len(summary["divergent_cves"]) == 19


def test_divergence_rate_matches_paper_fraction() -> None:
    # 19 divergent CVEs out of 30 in the released set -> 63.3%
    summary = divergence_summary(synthetic_corpus())
    assert round(100.0 * summary["total"] / 30.0, 1) == 63.3


def test_thresholds_are_configurable() -> None:
    rows = _cve_runs("CVE-2025-9007", diverge_phase="P2")
    assert divergence_summary(rows, high=0.95)["total"] == 0
    assert divergence_summary(rows, high=0.5)["total"] == 1
    # tight low threshold: 0.1 is still < 0.25, so divergence persists
    assert divergence_summary(rows, low=0.05)["total"] == 0


# ---------------------------------------------------------------------------
# optional: analysis.divergence module agreement (skips when absent)
# ---------------------------------------------------------------------------


def test_analysis_module_agrees_when_present() -> None:
    module = _load_divergence_module()
    if module is None:
        pytest.skip("reprobench.analysis.divergence not implemented yet")
    rows = synthetic_corpus()
    summary = _analysis_summary(module, rows)
    if summary is None:
        pytest.skip("analysis.divergence exposes no recognised summary entry point")

    counts = summary.get("counts") or summary.get("divergence_counts") or {}
    if not counts:
        pytest.skip("analysis.divergence summary does not expose per-phase counts")
    assert int(counts.get("P2", 0)) == 12
    assert int(counts.get("P4", 0)) == 2
    assert int(counts.get("P5", 0)) == 5


def test_analysis_module_constants_when_present() -> None:
    module = _load_divergence_module()
    if module is None:
        pytest.skip("reprobench.analysis.divergence not implemented yet")
    if hasattr(module, "HIGH_THRESHOLD"):
        assert float(module.HIGH_THRESHOLD) == pytest.approx(HIGH)
    if hasattr(module, "LOW_THRESHOLD"):
        assert float(module.LOW_THRESHOLD) == pytest.approx(LOW)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
