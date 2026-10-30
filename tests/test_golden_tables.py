"""Tier 1 — golden arithmetic oracles (must be exact, fully offline).

These values are the paper's worked examples and are the hard acceptance
interface for the measurement instrument:

    (C, D, F, Task) -> Overall
    (100, 100, 50, 100) -> 99.0
    ( 90, 100,  0,  53) -> 59.2
    ( 52.5, 100, 50, 14) -> 24.5

plus the Table 8-10 reconciliation identity
``12.0 + 6.0 + 1.0 + 12*4 + 16*2 = 99.0``.

Nothing here touches the network, a judge model, or a run directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow running ``pytest`` from the repository root without ``pip install -e .``.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_SRC))

from reprobench.scoring.aggregate import (  # noqa: E402
    DECOMPOSED_WEIGHTS,
    best_of,
    decomposed_contribution,
    overall_score,
)
from reprobench.scoring.plan import (  # noqa: E402
    COVERAGE_WEIGHTS,
    FALLBACK_ANCHORS,
    PHASES,
    coverage,
    dependency_detail,
    plan_score,
)
from reprobench.scoring.rubric import (  # noqa: E402
    PHASE_ITEM_POINTS,
    PHASE_MAX_POINTS,
    credit_for,
    score_phase,
    score_task,
    task_score,
)
from reprobench.utils.io import load_config, load_dataset, round_half_up  # noqa: E402

TOL = 1e-9

# --------------------------------------------------------------------------
# Golden plan arithmetic (Eq. 1-3)
# --------------------------------------------------------------------------


def test_plan_golden_100_100_50() -> None:
    assert plan_score(100.0, 100.0, 50.0) == pytest.approx(95.0, abs=TOL)


def test_plan_golden_90_100_0() -> None:
    assert plan_score(90.0, 100.0, 0.0) == pytest.approx(84.0, abs=TOL)


def test_plan_golden_52_5_100_50() -> None:
    assert plan_score(52.5, 100.0, 50.0) == pytest.approx(66.5, abs=TOL)


def test_plan_weights_are_sixty_thirty_ten() -> None:
    # 0.6*C + 0.3*D + 0.1*F exactly.
    c, d, f = 80.0, 60.0, 100.0
    expected = 0.6 * c + 0.3 * d + 0.1 * f
    assert plan_score(c, d, f) == pytest.approx(expected, abs=1e-6)


def test_coverage_golden_52_5() -> None:
    """q = [1, .5, 1, 1, 0, 0] -> C = 15 + 7.5 + 15 + 15 = 52.5."""
    q = {"P1": 1.0, "P2": 0.5, "P3": 1.0, "P4": 1.0, "P5": 0.0, "P6": 0.0}
    assert coverage(q) == pytest.approx(52.5, abs=TOL)


def test_coverage_max_is_100() -> None:
    assert coverage({p: 1.0 for p in PHASES}) == pytest.approx(100.0, abs=TOL)


def test_dependency_golden_one_inversion() -> None:
    """P1 -> P3 -> P2 -> P4 -> P5 -> P6 has exactly one inverted pair."""
    detail = dependency_detail(["P1", "P3", "P2", "P4", "P5", "P6"])
    assert detail["violations"] == 1
    assert detail["score"] == pytest.approx(80.0, abs=TOL)


def test_dependency_golden_five_inversions_floors_at_zero() -> None:
    """A fully reversed order yields >= 5 inversions, so D = max(0, 100 - 20*V) = 0."""
    detail = dependency_detail(["P6", "P5", "P4", "P3", "P2", "P1"])
    assert detail["violations"] >= 5
    assert detail["score"] == pytest.approx(0.0, abs=TOL)


def test_dependency_canonical_order_is_perfect() -> None:
    assert dependency_detail(list(PHASES))["score"] == pytest.approx(100.0, abs=TOL)


# --------------------------------------------------------------------------
# Golden task arithmetic (Table 4)
# --------------------------------------------------------------------------


def test_task_golden_full_marks() -> None:
    assert task_score([15, 15, 15, 15, 20, 20]) == pytest.approx(100.0, abs=TOL)


def test_task_golden_53() -> None:
    assert task_score([15, 14, 15, 9, 0, 0]) == pytest.approx(53.0, abs=TOL)


def test_task_golden_14() -> None:
    assert task_score([14, 0, 0, 0, 0, 0]) == pytest.approx(14.0, abs=TOL)


def test_task_golden_from_phase_scores() -> None:
    phases = score_task(
        {
            "P1": {"P1.1": 1.0, "P1.2": 1.0, "P1.3": 1.0, "P1.4": 1.0, "P1.5": 1.0},
            "P2": {"P2.1": 1.0},
            "P3": {"P3.1": 1.0},
            "P4": {"P4.1": 1.0},
            "P5": {},
            "P6": {},
        },
        require_citation=False,
    )
    assert task_score(phases) == pytest.approx(24.0, abs=TOL)


def test_item_credit_rounding_is_half_up() -> None:
    # 4 * 0.25 = 1.0; 3 * 0.25 = 0.75; 5 * 0.5 = 2.5
    assert credit_for(4, 0.25) == pytest.approx(1.0, abs=TOL)
    assert credit_for(3, 0.25) == pytest.approx(0.75, abs=TOL)
    assert credit_for(5, 0.5) == pytest.approx(2.5, abs=TOL)


def test_phase_capped_at_max() -> None:
    # Every item at full fraction with bogus extra credit passed in.
    phase = score_phase(
        "P2",
        {"P2.1": 1.0, "P2.2": 1.0, "P2.3": 1.0},
        require_citation=False,
        attempt_credit=5.0,
    )
    assert phase.artifact_score == pytest.approx(15.0, abs=TOL)
    assert phase.score <= phase.max_points


# --------------------------------------------------------------------------
# Golden overall arithmetic (0.2 * Plan + 0.8 * Task)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "plan, task, expected",
    [
        (95.0, 100.0, 99.0),
        (84.0, 53.0, 59.2),
        (66.5, 14.0, 24.5),
    ],
)
def test_overall_golden(plan: float, task: float, expected: float) -> None:
    assert overall_score(plan, task) == pytest.approx(expected, abs=TOL)


def test_overall_golden_chain_from_raw_components() -> None:
    """Full chain: (C, D, F, Task) -> Plan -> Overall."""
    plan_1 = plan_score(100.0, 100.0, 50.0)
    assert overall_score(plan_1, 100.0) == pytest.approx(99.0, abs=TOL)

    plan_2 = plan_score(90.0, 100.0, 0.0)
    assert overall_score(plan_2, 53.0) == pytest.approx(59.2, abs=TOL)

    plan_3 = plan_score(52.5, 100.0, 50.0)
    assert overall_score(plan_3, 14.0) == pytest.approx(24.5, abs=TOL)


def test_decomposed_reconciliation_identity() -> None:
    """Table 8-10 reconciliation: 12.0 + 6.0 + 1.0 + 12*4 + 16*2 = 99.0."""
    parts = decomposed_contribution(
        plan_coverage=100.0,
        plan_dependency=100.0,
        plan_fallback=50.0,
        task_phases={p: 100.0 for p in PHASES},
    )
    assert parts["plan_coverage"] == pytest.approx(12.0, abs=TOL)
    assert parts["plan_dependency"] == pytest.approx(6.0, abs=TOL)
    assert parts["plan_fallback"] == pytest.approx(1.0, abs=TOL)
    for pid in ("P1", "P2", "P3", "P4"):
        assert parts[f"task_{pid}"] == pytest.approx(12.0, abs=TOL)
    for pid in ("P5", "P6"):
        assert parts[f"task_{pid}"] == pytest.approx(16.0, abs=TOL)
    assert sum(parts.values()) == pytest.approx(99.0, abs=TOL)


def test_decomposed_weights_sum_to_one() -> None:
    assert sum(DECOMPOSED_WEIGHTS.values()) == pytest.approx(1.0, abs=TOL)


# --------------------------------------------------------------------------
# Config <-> frozen-table cross checks
# --------------------------------------------------------------------------


def test_phase_maxima_from_config_sum_to_100() -> None:
    cfg = load_config("phases")
    maxima = [p["max_points"] for p in cfg["phases"]]
    assert maxima == [15, 15, 15, 15, 20, 20]
    assert sum(maxima) == 100


def test_config_items_sum_to_phase_maxima() -> None:
    cfg = load_config("phases")
    for phase in cfg["phases"]:
        total = sum(item["points"] for item in phase["items"])
        assert total == phase["max_points"], f"{phase['id']} items sum to {total}"


def test_config_agrees_with_python_fallback_table() -> None:
    cfg = load_config("phases")
    for phase in cfg["phases"]:
        pid = phase["id"]
        from_config = {item["id"]: item["points"] for item in phase["items"]}
        assert from_config == PHASE_ITEM_POINTS[pid]
        assert PHASE_MAX_POINTS[pid] == phase["max_points"]


def test_benchmark_yaml_phase_maxima_match() -> None:
    cfg = load_config("benchmark")
    assert cfg["phases"]["max_points"] == [15, 15, 15, 15, 20, 20]
    assert cfg["plan"]["coverage_weights"] == list(COVERAGE_WEIGHTS)
    assert cfg["plan"]["weights"] == {"coverage": 0.6, "dependency": 0.3, "fallback": 0.1}
    assert cfg["aggregation"]["overall_plan_weight"] == 0.2
    assert cfg["aggregation"]["overall_task_weight"] == 0.8
    assert cfg["trials"] == 3
    assert cfg["total_runs"] == 450


def test_fallback_anchors_match_paper() -> None:
    assert list(FALLBACK_ANCHORS) == [0, 25, 50, 75, 100]


def test_coverage_weights_match_paper() -> None:
    assert list(COVERAGE_WEIGHTS) == [15, 15, 15, 15, 20, 20]


# --------------------------------------------------------------------------
# Dataset invariants (Table 3)
# --------------------------------------------------------------------------


def test_dataset_has_30_unique_cves() -> None:
    data = load_dataset("cves.yaml")
    ids = [c["id"] for c in data["cves"]]
    assert len(ids) == 30
    assert len(set(ids)) == 30


def test_dataset_is_ten_per_class() -> None:
    data = load_dataset("cves.yaml")
    counts: dict[str, int] = {}
    for cve in data["cves"]:
        counts[cve["class"]] = counts.get(cve["class"], 0) + 1
    assert counts == {"BOF": 10, "CMDI": 10, "AUTH": 10}


def test_dataset_has_14_vendors() -> None:
    data = load_dataset("cves.yaml")
    vendors = {cve["vendor"] for cve in data["cves"]}
    assert len(vendors) == 14


def test_dataset_years_in_range() -> None:
    data = load_dataset("cves.yaml")
    for cve in data["cves"]:
        assert 2020 <= int(cve["year"]) <= 2026, cve["id"]


def test_dataset_cwe_maps_to_single_class() -> None:
    data = load_dataset("cves.yaml")
    classes = data["classes"]
    for cve in data["cves"]:
        allowed = set(classes[cve["class"]]["cwe"])
        assert cve["cwe"], cve["id"]
        assert set(cve["cwe"]).issubset(allowed), cve["id"]
        # F3: must not map into any *other* class.
        for other, spec in classes.items():
            if other == cve["class"]:
                continue
            assert not (set(cve["cwe"]) & set(spec["cwe"])), cve["id"]


def test_dataset_derived_block_matches_entries() -> None:
    data = load_dataset("cves.yaml")
    derived = data["derived"]
    assert derived["total"] == len(data["cves"])
    assert derived["per_class"] == {"BOF": 10, "CMDI": 10, "AUTH": 10}
    assert derived["vendor_count"] == 14
    assert set(derived["vendors"]) == {cve["vendor"] for cve in data["cves"]}


def test_dataset_trigger_evidence_type_is_class_derived() -> None:
    data = load_dataset("cves.yaml")
    for cve in data["cves"]:
        expected = data["classes"][cve["class"]]["trigger_evidence_type"]
        assert cve["trigger_evidence_type"] == expected, cve["id"]


# --------------------------------------------------------------------------
# Rounding helper (shared by all of the above)
# --------------------------------------------------------------------------


def test_round_half_up_is_not_bankers_rounding() -> None:
    assert round_half_up(0.5) == 1.0
    assert round_half_up(1.5) == 2.0
    assert round_half_up(2.5) == 3.0
    assert round_half_up(59.25, 1) == 59.3


def test_best_of_empty_is_zeroed() -> None:
    row = best_of([])
    assert row["plan"] == 0.0
    assert row["task"] == 0.0
    assert row["overall"] == 0.0
    assert row["max_overall"] == 0.0
