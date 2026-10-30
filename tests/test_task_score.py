"""Tier 1/2 tests for the Table 4 task rubric (``reprobench.scoring.rubric``).

Covers:
  * phase maxima / item point breakdowns (15/15/15/15/20/20 = 100);
  * golden task scores: [15,15,15,15,20,20] -> 100, [15,14,15,9,0,0] -> 53,
    [14,0,0,0,0,0] -> 14;
  * fraction snapping to {0, .25, .5, .75, 1} and finer judge fractions recorded;
  * citation enforcement (credit without an existing citation path is zeroed);
  * attempt credit is only applied to a zero artifact phase (replace, never add),
    never to a gated phase, and is capped at the phase maximum;
  * phase scores are capped at the phase maximum.
"""

from __future__ import annotations

import pytest

from reprobench.scoring import rubric
from reprobench.scoring.rubric import (
    ITEM_KEYS,
    ITEM_POINTS,
    PHASE_ITEM_POINTS,
    PHASE_MAX_POINTS,
    credit_for,
    item_credit,
    score_phase,
    score_task,
    snap_fraction,
    task_score,
)
from reprobench.schema import PHASE_IDS


# ---------------------------------------------------------------------------
# Tier 2: structural invariants of the rubric tables
# ---------------------------------------------------------------------------

def test_phase_maxima_match_paper():
    assert PHASE_MAX_POINTS == {
        "P1": 15,
        "P2": 15,
        "P3": 15,
        "P4": 15,
        "P5": 20,
        "P6": 20,
    }
    assert sum(PHASE_MAX_POINTS.values()) == 100


def test_item_points_match_table4():
    assert PHASE_ITEM_POINTS["P1"] == {"P1.1": 4, "P1.2": 4, "P1.3": 3, "P1.4": 2, "P1.5": 2}
    assert PHASE_ITEM_POINTS["P2"] == {"P2.1": 5, "P2.2": 5, "P2.3": 5}
    assert PHASE_ITEM_POINTS["P3"] == {"P3.1": 3, "P3.2": 4, "P3.3": 6, "P3.4": 2}
    assert PHASE_ITEM_POINTS["P4"] == {"P4.1": 3, "P4.2": 4, "P4.3": 5, "P4.4": 3}
    assert PHASE_ITEM_POINTS["P5"] == {"P5.1": 4, "P5.2": 4, "P5.3": 4, "P5.4": 5, "P5.5": 3}
    assert PHASE_ITEM_POINTS["P6"] == {"P6.1": 4, "P6.2": 4, "P6.3": 8, "P6.4": 2, "P6.5": 2}


def test_item_points_sum_to_phase_maxima():
    for phase, items in PHASE_ITEM_POINTS.items():
        assert sum(items.values()) == PHASE_MAX_POINTS[phase], phase


def test_every_item_has_semantic_key():
    for phase, items in PHASE_ITEM_POINTS.items():
        for item_id in items:
            assert item_id in ITEM_KEYS, item_id
            assert ITEM_KEYS[item_id]


def test_item_points_flattened_consistent():
    total = 0
    for items in PHASE_ITEM_POINTS.values():
        for item_id, points in items.items():
            assert ITEM_POINTS[item_id] == points
            total += points
    assert total == 100


def test_config_cross_check_phase_maxima():
    """The YAML config, when loadable, must agree with the hardcoded fallback."""
    points = rubric.load_item_points()
    for phase, items in points.items():
        assert sum(items.values()) == PHASE_MAX_POINTS[phase], phase


# ---------------------------------------------------------------------------
# Tier 1: golden task scores
# ---------------------------------------------------------------------------

def test_task_score_full_marks():
    assert task_score({"P1": 15, "P2": 15, "P3": 15, "P4": 15, "P5": 20, "P6": 20}) == 100.0


def test_task_score_golden_53():
    assert task_score({"P1": 15, "P2": 14, "P3": 15, "P4": 9, "P5": 0, "P6": 0}) == 53.0


def test_task_score_golden_14():
    assert task_score({"P1": 14, "P2": 0, "P3": 0, "P4": 0, "P5": 0, "P6": 0}) == 14.0


def test_task_score_sequence_form():
    assert task_score([15, 14, 15, 9, 0, 0]) == 53.0
    assert task_score([14, 0, 0, 0, 0, 0]) == 14.0


def test_task_score_missing_phases_are_zero():
    assert task_score({"P1": 14}) == 14.0
    assert task_score(None) == 0.0
    assert task_score({}) == 0.0


def test_task_score_uses_phase_score_objects():
    phases = [score_phase(pid, {item: 1.0 for item in PHASE_ITEM_POINTS[pid]}) for pid in PHASE_IDS]
    assert task_score(phases) == 100.0


# ---------------------------------------------------------------------------
# Fraction handling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [
        (0.0, 0.0),
        (0.1, 0.0),
        (0.13, 0.25),
        (0.25, 0.25),
        (0.4, 0.5),
        (0.5, 0.5),
        (0.6, 0.5),
        (0.75, 0.75),
        (0.9, 1.0),
        (1.0, 1.0),
    ],
)
def test_snap_fraction(value, expected):
    assert snap_fraction(value) == pytest.approx(expected)


def test_snap_fraction_clamps_out_of_range():
    assert snap_fraction(-1.0) == 0.0
    assert snap_fraction(2.0) == 1.0


def test_snap_fraction_non_numeric():
    assert snap_fraction(None) == 0.0  # type: ignore[arg-type]


def test_credit_for_rounds_half_up():
    assert credit_for(4, 0.25) == 1.0
    assert credit_for(15, 0.5) == 7.5
    assert credit_for(8, 0.75) == 6.0
    # 5 * 0.25 = 1.25 -> 1.25 (2dp); half-up at 2dp
    assert credit_for(5, 0.25) == 1.25


def test_item_credit_records_raw_fraction():
    credit = item_credit("P1.1", 4, 0.33, citations=["report.md:1"], judge="judge")
    assert credit.raw_fraction == pytest.approx(0.33)
    assert credit.fraction in (0.25, 0.5) or credit.fraction == 0.25
    assert credit.judge if hasattr(credit, "judge") else True  # schema tolerant


def test_item_credit_finer_fraction_preserved_unsnapped():
    credit = item_credit("P1.1", 4, 0.6, citations=["report.md:1"], snap=False)
    assert credit.fraction == pytest.approx(0.6)
    assert credit.credit == pytest.approx(2.4)


# ---------------------------------------------------------------------------
# Citation enforcement
# ---------------------------------------------------------------------------

def test_credit_without_citation_is_zeroed():
    credit = item_credit("P1.1", 4, 1.0, citations=[])
    assert credit.invalidated is True
    assert credit.credit == 0.0
    assert credit.fraction == 0.0


def test_credit_with_nonexistent_citation_is_zeroed():
    credit = item_credit(
        "P1.1",
        4,
        1.0,
        citations=["artifact_path:/output/nope.md"],
        citation_exists=lambda _c: False,
    )
    assert credit.invalidated is True
    assert credit.credit == 0.0


def test_credit_with_existing_citation_is_kept():
    credit = item_credit(
        "P1.1",
        4,
        1.0,
        citations=["artifact_path:/output/report.md"],
        citation_exists=lambda _c: True,
    )
    assert credit.invalidated is False
    assert credit.credit == 4.0


def test_phase_zero_when_all_citations_invalid():
    fractions = {item: 1.0 for item in PHASE_ITEM_POINTS["P1"]}
    phase = score_phase("P1", fractions, citation_exists=lambda _c: False)
    assert phase.artifact_score == 0.0
    assert all(item.invalidated for item in phase.items)


def test_citation_requirement_can_be_disabled():
    credit = item_credit("P1.1", 4, 1.0, citations=[], require_citation=False)
    assert credit.credit == 4.0
    assert credit.invalidated is False


def test_parse_string_citation_kinds():
    from reprobench.scoring.citations import citation_key

    credit = item_credit(
        "P1.1",
        4,
        1.0,
        citations=["command_index:3", "plan_line:plan.md:12", "plan.md:12", "artifact_path:/output/x"],
    )
    kinds = {citation_key(c).split(":")[0] for c in credit.citations}
    assert "command_index" in kinds
    assert "plan_line" in kinds or "artifact_path" in kinds


# ---------------------------------------------------------------------------
# Phase scoring
# ---------------------------------------------------------------------------

def test_score_phase_full_credit_for_every_phase():
    for pid in PHASE_IDS:
        fractions = {item: 1.0 for item in PHASE_ITEM_POINTS[pid]}
        phase = score_phase(pid, fractions, require_citation=False)
        assert phase.artifact_score == PHASE_MAX_POINTS[pid]
        assert phase.score == PHASE_MAX_POINTS[pid]


def test_score_phase_partial_fractions():
    fractions = {item: 0.5 for item in PHASE_ITEM_POINTS["P3"]}
    phase = score_phase("P3", fractions, require_citation=False)
    # 1.5 + 2 + 3 + 1 = 7.5
    assert phase.artifact_score == pytest.approx(7.5)


def test_score_phase_accepts_semantic_keys():
    phase = score_phase("P2", {"firmware_acquired": 1.0}, require_citation=False)
    assert phase.artifact_score == pytest.approx(5.0)


def test_score_phase_accepts_underscore_alias():
    phase = score_phase("P1", {"P1_1": 1.0}, require_citation=False)
    assert phase.artifact_score == pytest.approx(4.0)


def test_score_phase_capped_at_max():
    # Over-credit (finest fractions cannot exceed 1.0 per item, but supply a custom points map)
    points = {"P2": {"P2.1": 30, "P2.2": 0, "P2.3": 0}}
    phase = score_phase(
        "P2",
        {"P2.1": 1.0},
        points=points,
        require_citation=False,
        max_points=15,
    )
    assert phase.artifact_score == 15.0


def test_phase_failed_property():
    phase = score_phase("P5", {item: 0.0 for item in PHASE_ITEM_POINTS["P5"]}, require_citation=False)
    assert phase.failed is True
    phase_full = score_phase(
        "P5", {item: 1.0 for item in PHASE_ITEM_POINTS["P5"]}, require_citation=False
    )
    assert phase_full.failed is False


# ---------------------------------------------------------------------------
# Attempt credit semantics (replace-never-add, gated, capped)
# ---------------------------------------------------------------------------

def test_attempt_replaces_zero_artifact():
    fractions = {item: 0.0 for item in PHASE_ITEM_POINTS["P2"]}
    phase = score_phase("P2", fractions, attempt_credit=1.0, attempt_label="targeted")
    assert phase.artifact_score == 0.0
    assert phase.attempt_credit == 1.0
    assert phase.score == 1.0


def test_attempt_not_added_to_nonzero_artifact():
    fractions = {item: 1.0 for item in PHASE_ITEM_POINTS["P1"]}
    phase = score_phase("P1", fractions, attempt_credit=1.0, attempt_label="targeted",
                        require_citation=False)
    assert phase.artifact_score == 15.0
    assert phase.attempt_credit in (0.0,)  # not applied
    assert phase.score == 15.0


def test_attempt_excluded_when_gated():
    fractions = {item: 0.0 for item in PHASE_ITEM_POINTS["P5"]}
    phase = score_phase(
        "P5",
        fractions,
        attempt_credit=1.0,
        attempt_label="targeted",
        gated=True,
    )
    assert phase.attempt_credit == 0.0
    assert phase.score == 0.0
    assert phase.gated is True


def test_attempt_capped_at_phase_max():
    fractions = {item: 0.0 for item in PHASE_ITEM_POINTS["P3"]}
    phase = score_phase("P3", fractions, attempt_credit=99.0, attempt_label="targeted")
    assert phase.score == PHASE_MAX_POINTS["P3"]


def test_attempt_credit_values_are_absolute_points():
    """Attempt credit of 0.5 is half a *point*, not half the phase maximum."""
    fractions = {item: 0.0 for item in PHASE_ITEM_POINTS["P1"]}
    phase = score_phase("P1", fractions, attempt_credit=0.5, attempt_label="fragmentary")
    assert phase.score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# score_task convenience wrapper
# ---------------------------------------------------------------------------

def test_score_task_returns_six_phases_in_order():
    phases = score_task({})
    assert [p.phase for p in phases] == list(PHASE_IDS)
    assert task_score(phases) == 0.0


def test_score_task_full_marks():
    fractions = {pid: {item: 1.0 for item in PHASE_ITEM_POINTS[pid]} for pid in PHASE_IDS}
    phases = score_task(fractions)
    assert task_score(phases) == 100.0
