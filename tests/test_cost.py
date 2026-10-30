"""Tier 5 — cost accounting / Table 11 reproduction tests (offline, deterministic).

The paper reports (Table 11):

* 369 of 450 runs traced
* 1.19B total tokens
* $329.36 total cost
* 88.9 h total wall clock
* per-run means: 3,203,509 input tokens, 22,495 output tokens,
  3,226,004 total tokens, $0.89, 868 s

These tests never touch the network.  They (a) assert the published
figures are internally consistent with each other, and (b) reproduce them
from a deterministic synthetic fixture using either
``reprobench.analysis.cost`` (when that module is importable) or the
generic helpers in ``reprobench.utils.io`` plus ``schema.CostRecord``.

The cache-write nuance from the paper is covered too: cache-write tokens are
only ever reported for the claude model, so a fixture without cache-write
must not inflate the token means.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# --------------------------------------------------------------------------
# import bootstrap: allow `pytest tests/` from the repo root without install
# --------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from reprobench.schema import CostRecord, TokenUsage  # noqa: E402
from reprobench.utils.io import mean, pct, round_half_up  # noqa: E402

# --------------------------------------------------------------------------
# published Table 11 values (paper text, hard-coded on purpose)
# --------------------------------------------------------------------------
PAPER = {
    "runs": 450,
    "traced": 369,
    "total_tokens": 1_190_000_000,
    "total_cost_usd": 329.36,
    "total_hours": 88.9,
    "mean_input": 3_203_509,
    "mean_output": 22_495,
    "mean_total": 3_226_004,
    "mean_cost": 0.89,
    "mean_seconds": 868.0,
}

TOL = 1e-6


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _cost_module() -> Any:
    """Return ``reprobench.analysis.cost`` if it exists, else ``None``."""
    try:
        from reprobench.analysis import cost  # type: ignore

        return cost
    except Exception:  # pragma: no cover - module not implemented yet
        return None


def _synthetic_runs(n: int = 369, traced: int = 369) -> List[Dict[str, Any]]:
    """Build a deterministic fixture whose means equal the paper's.

    Every traced run carries the same per-run usage, so the arithmetic is
    exact and the test only checks aggregation, never sampling noise.
    """
    runs: List[Dict[str, Any]] = []
    for i in range(n):
        is_traced = i < traced
        runs.append(
            {
                "run_id": f"run-{i:04d}",
                "cve_id": f"CVE-2024-{i:04d}",
                "model": "glm-5.2",
                "trial": 1 + (i % 3),
                "traced": is_traced,
                "cost_usd": PAPER["mean_cost"] if is_traced else 0.0,
                "wall_seconds": PAPER["mean_seconds"] if is_traced else 0.0,
                "tokens": {
                    "input": PAPER["mean_input"],
                    "output": PAPER["mean_output"],
                    "reasoning": 0,
                    "cache_read": 0,
                    "cache_write": 0,
                },
            }
        )
    return runs


def _as_cost_record(raw: Dict[str, Any]) -> CostRecord:
    tok = raw.get("tokens", {})
    return CostRecord(
        tokens=TokenUsage(**{k: int(tok.get(k, 0) or 0) for k in
                             ("input", "output", "reasoning", "cache_read", "cache_write")}),
        cost_usd=float(raw.get("cost_usd", 0.0) or 0.0),
        wall_seconds=float(raw.get("wall_seconds", 0.0) or 0.0),
        traced=bool(raw.get("traced", True)),
    )


def _local_summary(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate cost rows without relying on ``analysis.cost``."""
    traced = [r for r in runs if r.get("traced", True)]
    n = len(traced)
    inputs = [int(r.get("tokens", {}).get("input", 0) or 0) for r in traced]
    outputs = [int(r.get("tokens", {}).get("output", 0) or 0) for r in traced]
    totals = [
        int(r.get("tokens", {}).get("input", 0) or 0)
        + int(r.get("tokens", {}).get("output", 0) or 0)
        + int(r.get("tokens", {}).get("reasoning", 0) or 0)
        + int(r.get("tokens", {}).get("cache_read", 0) or 0)
        + int(r.get("tokens", {}).get("cache_write", 0) or 0)
        for r in traced
    ]
    costs = [float(r.get("cost_usd", 0.0) or 0.0) for r in traced]
    secs = [float(r.get("wall_seconds", 0.0) or 0.0) for r in traced]
    return {
        "runs": len(runs),
        "traced": n,
        "mean_input": mean(inputs),
        "mean_output": mean(outputs),
        "mean_total": mean(totals),
        "mean_cost": mean(costs),
        "mean_seconds": mean(secs),
        "total_tokens": int(sum(totals)),
        "total_cost_usd": round_half_up(sum(costs), 2),
        "total_seconds": round_half_up(sum(secs), 1),
    }


# --------------------------------------------------------------------------
# 1. published figures are internally consistent
# --------------------------------------------------------------------------
def test_paper_traced_fraction() -> None:
    assert pct(PAPER["traced"], PAPER["runs"]) == 82.0


def test_paper_means_reconcile_with_totals() -> None:
    # mean_total * traced ~= total tokens (order of magnitude check)
    approx_total = PAPER["mean_total"] * PAPER["traced"]
    assert abs(approx_total - PAPER["total_tokens"]) / PAPER["total_tokens"] < 0.01
    # mean cost * traced ~= total cost
    assert abs(PAPER["mean_cost"] * PAPER["traced"] - PAPER["total_cost_usd"]) < 1.0
    # wall clock: 88.9 h = 320040 s; mean 868 s * 369 = 320292 s
    assert abs(PAPER["mean_seconds"] * PAPER["traced"] - PAPER["total_hours"] * 3600) / (
        PAPER["total_hours"] * 3600
    ) < 0.01


def test_paper_input_output_sum_below_total() -> None:
    # total >= input + output because reasoning/cache tokens are included
    assert PAPER["mean_total"] >= PAPER["mean_input"] + PAPER["mean_output"]


# --------------------------------------------------------------------------
# 2. fixture reproduces Table 11 means through the local aggregation
# --------------------------------------------------------------------------
def test_fixture_reproduces_traced_count() -> None:
    runs = _synthetic_runs(n=450, traced=369)
    summary = _local_summary(runs)
    assert summary["runs"] == 450
    assert summary["traced"] == 369


def test_fixture_reproduces_means() -> None:
    summary = _local_summary(_synthetic_runs())
    assert summary["mean_input"] == pytest.approx(PAPER["mean_input"], abs=TOL)
    assert summary["mean_output"] == pytest.approx(PAPER["mean_output"], abs=TOL)
    assert summary["mean_total"] == pytest.approx(PAPER["mean_total"], abs=TOL)
    assert summary["mean_cost"] == pytest.approx(PAPER["mean_cost"], abs=TOL)
    assert summary["mean_seconds"] == pytest.approx(PAPER["mean_seconds"], abs=TOL)


def test_fixture_totals_are_order_of_magnitude_correct() -> None:
    summary = _local_summary(_synthetic_runs())
    assert summary["total_tokens"] == PAPER["mean_total"] * PAPER["traced"]
    assert summary["total_cost_usd"] == pytest.approx(
        PAPER["mean_cost"] * PAPER["traced"], abs=0.01
    )


def test_untraced_runs_do_not_dilute_means() -> None:
    """Only traced runs contribute to token/cost means (paper: 369/450)."""
    traced_only = _local_summary(_synthetic_runs(n=369, traced=369))
    with_untraced = _local_summary(_synthetic_runs(n=450, traced=369))
    assert with_untraced["mean_input"] == pytest.approx(traced_only["mean_input"], abs=TOL)
    assert with_untraced["mean_cost"] == pytest.approx(traced_only["mean_cost"], abs=TOL)
    assert with_untraced["traced"] == traced_only["traced"] == 369


# --------------------------------------------------------------------------
# 3. schema.CostRecord / TokenUsage semantics
# --------------------------------------------------------------------------
def test_token_usage_total_includes_reasoning_and_cache() -> None:
    usage = TokenUsage(input=10, output=5, reasoning=2, cache_read=3, cache_write=4)
    assert usage.total == 24


def test_cost_record_from_fixture_row() -> None:
    rec = _as_cost_record(_synthetic_runs(n=1)[0])
    assert isinstance(rec, CostRecord)
    assert rec.traced is True
    assert rec.tokens.total == PAPER["mean_input"] + PAPER["mean_output"]
    assert rec.cost_usd == pytest.approx(PAPER["mean_cost"], abs=TOL)


def test_cache_write_only_on_claude_model() -> None:
    """Cache-write tokens are reported only for the claude model."""
    claude = _as_cost_record(
        {
            "traced": True,
            "cost_usd": 1.0,
            "wall_seconds": 100.0,
            "tokens": {"input": 1000, "output": 100, "cache_read": 50, "cache_write": 25},
        }
    )
    other = _as_cost_record(
        {
            "traced": True,
            "cost_usd": 1.0,
            "wall_seconds": 100.0,
            "tokens": {"input": 1000, "output": 100, "cache_read": 50, "cache_write": 0},
        }
    )
    assert claude.tokens.cache_write == 25
    assert other.tokens.cache_write == 0
    assert claude.tokens.total - other.tokens.total == 25


def test_untraced_defaults_are_zero() -> None:
    rec = _as_cost_record({"traced": False, "tokens": {}})
    assert rec.traced is False
    assert rec.cost_usd == 0.0
    assert rec.tokens.total == 0


# --------------------------------------------------------------------------
# 4. analysis.cost integration (skipped when the module is absent)
# --------------------------------------------------------------------------
def test_analysis_cost_summary_matches_fixture() -> None:
    cost_mod = _cost_module()
    if cost_mod is None:
        pytest.skip("reprobench.analysis.cost not implemented yet")

    runs = _synthetic_runs()
    summary = None
    for name in ("summarize_costs", "summarize", "cost_summary", "aggregate_costs"):
        fn = getattr(cost_mod, name, None)
        if callable(fn):
            try:
                summary = fn(runs)
            except TypeError:
                records = [_as_cost_record(r) for r in runs]
                summary = fn(records)
            break
    if summary is None:
        pytest.skip("analysis.cost exposes no recognised summary entry point")

    assert isinstance(summary, dict)
    assert int(summary.get("traced", summary.get("traced_runs", -1))) == PAPER["traced"]
    for key in ("mean_input", "mean_output", "mean_total", "mean_cost", "mean_seconds"):
        if key in summary:
            assert float(summary[key]) == pytest.approx(PAPER[key], abs=1.0)


def test_analysis_cost_table_renderer_if_present() -> None:
    cost_mod = _cost_module()
    if cost_mod is None:
        pytest.skip("reprobench.analysis.cost not implemented yet")
    renderer = None
    for name in ("table11", "render_table11", "cost_table", "table_rows"):
        fn = getattr(cost_mod, name, None)
        if callable(fn):
            renderer = fn
            break
    if renderer is None:
        pytest.skip("analysis.cost exposes no table renderer")
    try:
        out = renderer(_synthetic_runs())
    except TypeError:
        out = renderer([_as_cost_record(r) for r in _synthetic_runs()])
    text = out if isinstance(out, str) else str(out)
    assert "369" in text


# --------------------------------------------------------------------------
# 5. cost monitor semantics (budget abort at $5.00)
# --------------------------------------------------------------------------
def test_budget_abort_threshold_matches_paper() -> None:
    """Rule -2: total cost must not exceed $5.00 per run."""
    from reprobench.utils.io import load_config

    cfg = load_config("benchmark")
    assert float(cfg["budget"]["max_cost_usd"]) == 5.00


def test_warn_fraction_matches_config() -> None:
    from reprobench.utils.io import load_config

    cfg = load_config("benchmark")
    warn = float(cfg["budget"]["warn_fraction"])
    assert 0.0 < warn < 1.0
    assert round_half_up(5.00 * warn, 2) == 4.00


def test_sum_of_costs_is_finite() -> None:
    summary = _local_summary(_synthetic_runs())
    assert math.isfinite(summary["total_cost_usd"])
    assert summary["total_cost_usd"] > 0
