"""ReproBench scoring engine.

Public surface (paper Sections 4-6, Tables 4/5/7, Appendix B.2):

* :mod:`reprobench.scoring.plan`      -- Eq. 1-3: coverage / dependency / fallback -> Plan.
* :mod:`reprobench.scoring.rubric`    -- Table 4 item credits -> phase scores -> Task.
* :mod:`reprobench.scoring.gating`    -- Appendix B.2 real-target gating (P5/P6 -> 0).
* :mod:`reprobench.scoring.attempt`   -- Table 5 attempt credit (absolute 0/0.5/1 points).
* :mod:`reprobench.scoring.aggregate` -- Overall = 0.2*Plan + 0.8*Task, best-of-three.
* :mod:`reprobench.scoring.taxonomy`  -- Table 7 first-match failure attribution.
* :mod:`reprobench.scoring.citations` -- citation normalization / existence predicates.
* :mod:`reprobench.scoring.evidence`  -- EvidenceBundle: manifest, commands, provenance.
* :mod:`reprobench.scoring.judge`     -- two-pass LLM judge + offline heuristic fallback.

Import policy
-------------
This façade is deliberately **lightweight**: only the deterministically importable
numeric cores are imported eagerly (they are stdlib-only / pydantic-only and are
needed by nearly every consumer).  The heavier or optional modules are exposed
lazily through :func:`__getattr__` so that ``import reprobench.scoring`` never
pulls in ``httpx``, the LLM SDKs, ``python-magic`` or the profile builder.
"""

from __future__ import annotations

from typing import Any, Dict, List

__all__: List[str] = [
    # aggregate
    "BEST_OF",
    "DECOMPOSED_WEIGHTS",
    "OVERALL_PLAN_WEIGHT",
    "OVERALL_TASK_WEIGHT",
    "aggregate_all",
    "aggregate_cve_model",
    "best_of",
    "count_p5_p6_thresholds",
    "decomposed_contribution",
    "model_table_rows",
    "overall_score",
    "success_counts",
    # plan
    "COVERAGE_WEIGHTS",
    "FALLBACK_ANCHORS",
    "PHASES",
    "PLAN_WEIGHTS",
    "classify_fallback",
    "coverage",
    "dependency_detail",
    "dependency_score",
    "fallback_score",
    "plan_from_text",
    "plan_score",
    "score_plan",
    "snap_q",
    # rubric
    "ITEM_KEYS",
    "ITEM_POINTS",
    "PHASE_ITEM_POINTS",
    "PHASE_MAX_POINTS",
    "credit_for",
    "item_credit",
    "score_phase",
    "score_task",
    "snap_fraction",
    "task_score",
]

# ---------------------------------------------------------------------------------
# Eager imports: pure numeric cores (no network, no optional dependencies).
# ---------------------------------------------------------------------------------
from .aggregate import (  # noqa: E402
    BEST_OF,
    DECOMPOSED_WEIGHTS,
    OVERALL_PLAN_WEIGHT,
    OVERALL_TASK_WEIGHT,
    aggregate_all,
    aggregate_cve_model,
    best_of,
    count_p5_p6_thresholds,
    decomposed_contribution,
    model_table_rows,
    overall_score,
    success_counts,
)
from .plan import (  # noqa: E402
    COVERAGE_WEIGHTS,
    FALLBACK_ANCHORS,
    PHASES,
    PLAN_WEIGHTS,
    classify_fallback,
    coverage,
    dependency_detail,
    dependency_score,
    fallback_score,
    plan_from_text,
    plan_score,
    score_plan,
    snap_q,
)
from .rubric import (  # noqa: E402
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

# ---------------------------------------------------------------------------------
# Lazy modules: import on first attribute access.
# ---------------------------------------------------------------------------------
_LAZY_MODULES: Dict[str, str] = {
    "attempt": "reprobench.scoring.attempt",
    "citations": "reprobench.scoring.citations",
    "evidence": "reprobench.scoring.evidence",
    "gating": "reprobench.scoring.gating",
    "judge": "reprobench.scoring.judge",
    "taxonomy": "reprobench.scoring.taxonomy",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - trivially thin
    """Resolve the heavier scoring submodules only when they are actually used."""
    module_name = _LAZY_MODULES.get(name)
    if module_name is not None:
        import importlib

        module = importlib.import_module(module_name)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:  # pragma: no cover - introspection helper
    return sorted(set(__all__) | set(_LAZY_MODULES) | set(globals()))
