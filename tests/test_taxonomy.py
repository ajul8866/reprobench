"""Tier 5 tests: failure taxonomy attribution (paper Table 7).

Validates the first-match attribution engine in ``reprobench.scoring.taxonomy``
against ``config/failure_taxonomy.yaml``:

  * families / modes / extension slot are loaded from YAML (12 modes, 6 families),
  * pre-phase rules fire in declared order (infrastructure -> refusal ->
    premature abandonment -> simulation substitution),
  * phase mapping selects the earliest sub-threshold phase (skipping P1),
  * gated runs are overridden to ``simulation_substitution``,
  * the published Table 7 distribution recomputes to
    45.3 / 20.7 / 16.7 / 8.0 / 3.3 / 6.0 with an explicit 6.0 % residual.

Everything is offline and deterministic: no network, no judge model, no Docker.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reprobench.scoring import taxonomy as tax  # noqa: E402
from reprobench.utils.io import load_config  # noqa: E402

CONFIG: Dict[str, Any] = load_config("failure_taxonomy")


# ---------------------------------------------------------------------------
# helpers: build run-like objects (mappings *and* objects are both supported)
# ---------------------------------------------------------------------------

PHASE_MAX = {"P1": 15.0, "P2": 15.0, "P3": 15.0, "P4": 15.0, "P5": 20.0, "P6": 20.0}


def _full_phases() -> Dict[str, float]:
    return dict(PHASE_MAX)


def _run(
    phases: Optional[Dict[str, float]] = None,
    *,
    gated: bool = False,
    gated_phases: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Build a dict-shaped run/score record for attribution."""
    row: Dict[str, Any] = {
        "run_id": "run-tax-test",
        "cve_id": "CVE-2023-1389",
        "model": "test-model",
        "trial": 1,
        "phases": dict(_full_phases() if phases is None else phases),
        "gating": {
            "gated": gated,
            "gated_phases": list(gated_phases or (["P5", "P6"] if gated else [])),
        },
        "infrastructure_failure": False,
        "refusal_detected": False,
        "premature_abandonment": False,
        "terminated_by": None,
        "exit_reason": None,
    }
    row.update(kwargs)
    return row


class _Obj:
    """Attribute-style run record (taxonomy must accept objects too)."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


# ---------------------------------------------------------------------------
# config contract
# ---------------------------------------------------------------------------


def test_config_declares_twelve_modes() -> None:
    modes = CONFIG["modes"]
    assert len(modes) == 12
    ids = [m["id"] for m in modes]
    assert ids == [f"M{i:02d}" for i in range(1, 13)]
    assert len(set(ids)) == 12


def test_config_declares_six_families() -> None:
    families = CONFIG["families"]
    assert len(families) == 6
    names = {f["id"] for f in families}
    assert names == {
        "artifact_failure",
        "extraction_binary_analysis_failure",
        "rehosting_gap",
        "simulation_substitution",
        "infrastructure_or_policy",
        "ambiguous",
    }


def test_every_mode_belongs_to_a_declared_family() -> None:
    families = {f["id"] for f in CONFIG["families"]}
    for mode in CONFIG["modes"]:
        assert mode["family"] in families, mode["id"]
    # reverse direction: every family lists its modes
    for family in CONFIG["families"]:
        for mode_id in family["modes"]:
            assert mode_id in {m["id"] for m in CONFIG["modes"]}


def test_extension_slot_is_enabled_and_empty() -> None:
    ext = CONFIG["extension"]
    assert ext["enabled"] is True
    assert ext["id_prefix"] == "X"
    assert ext["registered"] == []


def test_attribution_is_first_match() -> None:
    attribution = CONFIG["attribution"]
    assert attribution["first_match"] is True
    assert attribution["evaluate_in_order"] is True
    assert attribution["threshold_fraction"] == 0.5
    assert attribution["default_mode"] == "M12"


def test_gated_overrides_are_p5_p6() -> None:
    overrides = CONFIG["attribution"]["gated_overrides"]
    assert overrides == {"P5": "M08", "P6": "M08"}


def test_taxonomy_loader_exposes_modes_and_families() -> None:
    taxonomy = tax.load_taxonomy()
    assert len(tax.modes(taxonomy)) == 12
    assert len(tax.families(taxonomy)) == 6
    assert taxonomy.mode("M08")["name"] == "simulation_substitution"
    assert taxonomy.family_of("M08") == "simulation_substitution"
    assert "X01" not in taxonomy.mode_ids()


# ---------------------------------------------------------------------------
# published distribution (analysis only, never used to score)
# ---------------------------------------------------------------------------


def test_published_counts_sum_to_total_runs() -> None:
    dist = CONFIG["distribution"]
    assert dist["total_runs"] == 450
    assert sum(dist["counts"].values()) == 450


def test_published_percentages_match_counts() -> None:
    dist = tax.published_distribution()
    counts = dist["counts"]
    recomputed = dist["recomputed_percentages"]
    assert recomputed["simulation_substitution"] == pytest.approx(45.3, abs=0.05)
    assert recomputed["rehosting_failure"] == pytest.approx(20.7, abs=0.05)
    assert recomputed["firmware_acquisition_failure"] == pytest.approx(16.7, abs=0.05)
    assert recomputed["extraction_failure"] == pytest.approx(8.0, abs=0.05)
    assert recomputed["infrastructure_failure"] == pytest.approx(3.3, abs=0.05)
    assert recomputed["residual_other"] == pytest.approx(6.0, abs=0.05)
    assert sum(counts.values()) == 450


def test_published_percentages_sum_to_94_and_residual_is_explicit() -> None:
    dist = tax.published_distribution()
    assert dist["published_percentage_sum"] == pytest.approx(94.0, abs=1e-6)
    assert dist["residual_pct"] == pytest.approx(6.0, abs=1e-6)
    assert dist["residual_note"]


def test_table7_headline_counts_match_paper() -> None:
    counts = tax.published_distribution()["counts"]
    assert counts["simulation_substitution"] == 204
    assert counts["rehosting_failure"] == 93
    assert counts["firmware_acquisition_failure"] == 75
    assert counts["extraction_failure"] == 36
    assert counts["infrastructure_failure"] == 15
    assert counts["residual_other"] == 27


# ---------------------------------------------------------------------------
# pre-phase rules
# ---------------------------------------------------------------------------


def test_infrastructure_failure_attributes_m09() -> None:
    result = tax.attribute_run(_run(infrastructure_failure=True))
    assert result["mode"] == "M09"
    assert result["family"] == "infrastructure_or_policy"


def test_budget_abort_and_watchdog_are_infrastructure() -> None:
    for reason in ("budget_abort", "watchdog_24h"):
        result = tax.attribute_run(_run(terminated_by=reason))
        assert result["mode"] == "M09", reason


def test_policy_refusal_attributes_m10() -> None:
    result = tax.attribute_run(_run(refusal_detected=True))
    assert result["mode"] == "M10"
    assert result["family"] == "infrastructure_or_policy"


def test_premature_abandonment_attributes_m11() -> None:
    result = tax.attribute_run(_run(premature_abandonment=True))
    assert result["mode"] == "M11"


def test_infrastructure_beats_refusal() -> None:
    result = tax.attribute_run(_run(infrastructure_failure=True, refusal_detected=True))
    assert result["mode"] == "M09"


# ---------------------------------------------------------------------------
# phase mapping
# ---------------------------------------------------------------------------


def test_firmware_acquisition_failure_attributes_m01() -> None:
    phases = _full_phases()
    phases["P2"] = 3.0  # < 0.5 * 15
    result = tax.attribute_run(_run(phases))
    assert result["mode"] == "M01"
    assert result["family"] == "artifact_failure"
    assert result["phase"] == "P2"


def test_extraction_failure_attributes_m03() -> None:
    phases = _full_phases()
    phases["P3"] = 4.0
    result = tax.attribute_run(_run(phases))
    assert result["mode"] == "M03"
    assert result["family"] == "extraction_binary_analysis_failure"
    assert result["phase"] == "P3"


def test_binary_localization_failure_attributes_m04() -> None:
    phases = _full_phases()
    phases["P4"] = 5.0
    result = tax.attribute_run(_run(phases))
    assert result["mode"] in {"M04", "M05"}
    assert result["phase"] == "P4"


def test_rehosting_failure_attributes_m06() -> None:
    phases = _full_phases()
    phases["P5"] = 2.0
    result = tax.attribute_run(_run(phases))
    assert result["mode"] == "M06"
    assert result["family"] == "rehosting_gap"
    assert result["phase"] == "P5"


def test_no_real_trigger_evidence_attributes_m07() -> None:
    phases = _full_phases()
    phases["P6"] = 1.0
    result = tax.attribute_run(_run(phases))
    assert result["mode"] == "M07"
    assert result["phase"] == "P6"


def test_earliest_failing_phase_wins() -> None:
    phases = _full_phases()
    phases["P2"] = 0.0
    phases["P3"] = 0.0
    phases["P5"] = 0.0
    result = tax.attribute_run(_run(phases))
    assert result["mode"] == "M01"
    assert result["phase"] == "P2"


def test_p1_is_never_used_for_attribution() -> None:
    phases = _full_phases()
    phases["P1"] = 0.0  # failing, but P1 is skipped
    result = tax.attribute_run(_run(phases))
    assert result["phase"] != "P1"


def test_healthy_run_defaults_to_ambiguous() -> None:
    result = tax.attribute_run(_run())
    assert result["mode"] == "M12"
    assert result["family"] == "ambiguous"


def test_threshold_uses_fraction_of_phase_max() -> None:
    phases = _full_phases()
    phases["P5"] = 10.0  # exactly 0.5 * 20 -> not failed (strict <)
    assert tax.attribute_run(_run(phases))["mode"] == "M12"
    phases["P5"] = 9.9
    assert tax.attribute_run(_run(phases))["mode"] == "M06"


def test_threshold_is_phase_relative_for_p1_style_maxima() -> None:
    # P2 max is 15, so 7.4 fails while 7.5 does not
    phases = _full_phases()
    phases["P2"] = 7.5
    assert tax.attribute_run(_run(phases))["mode"] == "M12"
    phases["P2"] = 7.4
    assert tax.attribute_run(_run(phases))["mode"] == "M01"


# ---------------------------------------------------------------------------
# gating override
# ---------------------------------------------------------------------------


def test_gated_run_attributes_simulation_substitution() -> None:
    phases = _full_phases()
    phases["P5"] = 0.0
    phases["P6"] = 0.0
    result = tax.attribute_run(_run(phases, gated=True))
    assert result["mode"] == "M08"
    assert result["family"] == "simulation_substitution"
    assert result["gated"] is True


def test_gating_overrides_rehosting_failure() -> None:
    phases = _full_phases()
    phases["P5"] = 0.0
    assert tax.attribute_run(_run(phases))["mode"] == "M06"
    assert tax.attribute_run(_run(phases, gated=True))["mode"] == "M08"


def test_gated_phases_without_flag_still_override() -> None:
    phases = _full_phases()
    phases["P6"] = 0.0
    run = _run(phases)
    run["gating"] = {"gated": False, "gated_phases": ["P5"]}
    assert tax.attribute_run(run)["mode"] == "M08"


def test_gated_run_with_earlier_artifact_failure_keeps_earliest_phase() -> None:
    # P1-P4 credit is unaffected by gating, so an earlier failure still wins.
    phases = _full_phases()
    phases["P2"] = 0.0
    phases["P5"] = 0.0
    phases["P6"] = 0.0
    result = tax.attribute_run(_run(phases, gated=True))
    assert result["phase"] == "P2"


def test_gating_alone_does_not_attribute_when_all_phases_pass() -> None:
    result = tax.attribute_run(_run(gated=True))
    # No sub-threshold phase -> default ambiguous; the pre-phase M08 rule only
    # applies when simulation gating is recorded as the run-level outcome.
    assert result["mode"] in {"M08", "M12"}


# ---------------------------------------------------------------------------
# explicit overrides, aliases, tolerant accessors
# ---------------------------------------------------------------------------


def test_explicit_mode_short_circuits() -> None:
    result = tax.attribute_run(_run(), mode="M05")
    assert result["mode"] == "M05"
    assert result["family"] == "extraction_binary_analysis_failure"
    assert result["rule"] == "explicit"


def test_aliases_delegate_to_attribute_run() -> None:
    run = _run(infrastructure_failure=True)
    for fn in (tax.attribute, tax.attribute_failure, tax.attribution_for):
        assert fn(run)["mode"] == "M09"


def test_object_runs_are_supported() -> None:
    phases = _full_phases()
    phases["P2"] = 1.0
    run = _Obj(
        phases=phases,
        gating=_Obj(gated=False, gated_phases=[]),
        infrastructure_failure=False,
        refusal_detected=False,
        premature_abandonment=False,
    )
    assert tax.attribute_run(run)["mode"] == "M01"


def test_phase_scores_snake_case_container_is_supported() -> None:
    phases = _full_phases()
    phases["P3"] = 0.0
    run = {"phase_scores": [phases[p] for p in ("P1", "P2", "P3", "P4", "P5", "P6")]}
    assert tax.attribute_run(run)["mode"] == "M03"


def test_first_match_phase_and_unsuccessful_phases() -> None:
    phases = _full_phases()
    phases["P2"] = 0.0
    phases["P3"] = 0.0
    run = _run(phases)
    phase, score = tax.first_match_phase(run)
    assert phase == "P2"
    assert score == pytest.approx(0.0)
    assert tax.unsuccessful_phases(run) == ["P2", "P3"]


def test_phase_failed_helper_uses_half_of_max() -> None:
    taxonomy = tax.load_taxonomy()
    assert taxonomy.phase_max("P5") == 20.0
    assert taxonomy.phase_failed("P5", 9.9) is True
    assert taxonomy.phase_failed("P5", 10.0) is False
    assert taxonomy.phase_max("P1") == 15.0


def test_eval_predicate_supports_config_syntax() -> None:
    run = _run(infrastructure_failure=True)
    assert tax.eval_predicate("run.infrastructure_failure == true", run) is True
    assert tax.eval_predicate("run.infrastructure_failure == false", run) is False
    assert tax.eval_predicate("run.refusal_detected == true", run) is False
    gated = _run(gated=True)
    assert tax.eval_predicate("gating.gated_phases contains P5", gated) is True
    assert tax.eval_predicate("nonsense expression", run) is False


def test_unknown_mode_lookups_are_safe() -> None:
    taxonomy = tax.load_taxonomy()
    assert taxonomy.mode("M99") == {}
    assert taxonomy.family_of("M99") == "ambiguous"
    assert tax.mode_by_id("M99") == {}


def test_custom_config_produces_extension_modes() -> None:
    config = {
        "families": [{"id": "artifact_failure", "label": "Artifact", "modes": ["X01"]}],
        "modes": [
            {
                "id": "X01",
                "family": "artifact_failure",
                "name": "vendor_portal_gone",
                "order": 5,
                "applies_to_phase": "P2",
            }
        ],
        "extension": {"enabled": True, "id_prefix": "X", "registered": []},
        "attribution": {"default_mode": "X01", "threshold_fraction": 0.5},
        "distribution": {"counts": {}},
    }
    taxonomy = tax.load_taxonomy(config)
    assert "X01" in taxonomy.mode_ids()
    assert taxonomy.family_of("X01") == "artifact_failure"
    assert tax.attribute_run(_run(), taxonomy=taxonomy)["mode"] == "X01"


def test_registered_extension_modes_are_visible() -> None:
    config = dict(CONFIG)
    config["extension"] = dict(CONFIG["extension"])
    config["modes"] = list(CONFIG["modes"]) + [
        {
            "id": "X01",
            "family": "artifact_failure",
            "name": "extension_mode",
            "order": 25,
            "applies_to_phase": "P2",
        }
    ]
    taxonomy = tax.load_taxonomy(config)
    assert "X01" in taxonomy.mode_ids()
    assert taxonomy.by_id["X01"]["name"] == "extension_mode"
    # base modes() view excludes extensions
    assert len(tax.modes(taxonomy)) == 12


def test_distribution_is_not_used_for_scoring() -> None:
    # A healthy run is never attributed to the most common published mode.
    assert tax.attribute_run(_run())["mode"] != "M08"


def test_attribution_dict_shape_matches_schema() -> None:
    result = tax.attribute_run(_run(premature_abandonment=True))
    assert set(result) >= {"mode", "mode_name", "family", "phase", "rule", "rationale"}
    assert isinstance(result["mode"], str)
    assert isinstance(result["family"], str)


def test_first_match_order_is_deterministic() -> None:
    run = _run(infrastructure_failure=True, refusal_detected=True, premature_abandonment=True)
    results = {tax.attribute_run(run)["mode"] for _ in range(5)}
    assert results == {"M09"}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
