"""Failure-distribution analysis (paper Table 7 / Section 6).

Consumes scored run rows (or a ``results/scores`` directory), attributes each
unscored/failed run to a taxonomy mode via :mod:`reprobench.scoring.taxonomy`,
and emits the published Table 7 breakdown in the form the report emitters use.

The module is pure and offline: it reads rows, never writes, never calls a
judge model.  All family/mode identifiers, published counts and percentages
live in ``config/failure_taxonomy.yaml``; nothing is hardcoded here except
small defensive fallbacks.

Note on the paper's arithmetic
------------------------------
The published percentages (45.3 / 20.7 / 16.7 / 8.0 / 3.3) sum to 94.0 %.
The residual 6.0 % is reported explicitly (``residual_pct``) rather than
redistributed across the other modes.  ``PAPER_FAILURE_DISTRIBUTION`` is
reference data for the claims ledger only - never an input to scoring.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..utils.io import pct, round_half_up
from . import taxonomy as taxonomy_mod

__all__ = [
    "PAPER_FAILURE_DISTRIBUTION",
    "FAILURE_FAMILIES",
    "attribute_failures",
    "family_distribution",
    "failure_distribution",
    "distribution_rows",
    "render_failures",
    "render_failure_table",
    "failure_table",
    "summarize_failures",
]

# --------------------------------------------------------------------------
# Published reference values (Table 7).  Analysis/claims only.
# --------------------------------------------------------------------------

PAPER_FAILURE_DISTRIBUTION: Dict[str, Any] = {
    "total_runs": 450,
    "counts": {
        "simulation_substitution": 204,
        "rehosting_failure": 93,
        "firmware_acquisition_failure": 75,
        "extraction_failure": 36,
        "infrastructure_failure": 15,
        "residual_other": 27,
    },
    "percentages": {
        "simulation_substitution": 45.3,
        "rehosting_failure": 20.7,
        "firmware_acquisition_failure": 16.7,
        "extraction_failure": 8.0,
        "infrastructure_failure": 3.3,
        "residual_other": 6.0,
    },
    "published_percentage_sum": 94.0,
    "residual_pct": 6.0,
}

#: Family ids in presentation order (matches ``config/failure_taxonomy.yaml``).
FAILURE_FAMILIES: Tuple[str, ...] = (
    "artifact_failure",
    "extraction_binary_analysis_failure",
    "rehosting_gap",
    "simulation_substitution",
    "infrastructure_or_policy",
    "ambiguous",
)

#: Mapping from mode name (as published in Table 7) to taxonomy mode id.
_MODE_NAME_TO_ID: Dict[str, str] = {
    "simulation_substitution": "M08",
    "rehosting_failure": "M06",
    "firmware_acquisition_failure": "M01",
    "extraction_failure": "M03",
    "infrastructure_failure": "M09",
    "policy_refusal": "M10",
    "premature_abandonment": "M11",
    "binary_mismatch": "M04",
    "tool_misuse": "M05",
    "no_real_trigger_evidence": "M07",
    "version_or_artifact_mismatch": "M02",
    "ambiguous": "M12",
}

#: Table 7 presentation labels (short) used by the Markdown renderer.
_MODE_LABELS: Dict[str, str] = {
    "M08": "Simulation substitution",
    "M06": "Rehosting failure",
    "M01": "Firmware acquisition failure",
    "M03": "Extraction failure",
    "M09": "Infrastructure failure",
    "M10": "Policy refusal",
    "M11": "Premature abandonment",
    "M04": "Binary mismatch",
    "M05": "Tool misuse",
    "M07": "No real trigger evidence",
    "M02": "Version/artifact mismatch",
    "M12": "Ambiguous / other",
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Duck-typed attribute/mapping accessor."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _first(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        value = _get(obj, name, None)
        if value is not None:
            return value
    return default


def _mode_of(row: Any) -> Optional[str]:
    """Best-effort extraction of an already-attributed mode id."""
    failure = _get(row, "failure", None)
    if failure is not None:
        mode = _get(failure, "mode", None)
        if mode:
            return str(mode)
    for key in ("failure_mode", "mode", "attributed_mode"):
        value = _get(row, key, None)
        if value:
            return str(value)
    return None


def _family_of(row: Any, tax: "taxonomy_mod.Taxonomy") -> Optional[str]:
    failure = _get(row, "failure", None)
    if failure is not None:
        family = _get(failure, "family", None)
        if family:
            return str(family)
    family = _get(row, "failure_family", None)
    if family:
        return str(family)
    mode = _mode_of(row)
    if mode:
        return tax.family_of(mode)
    return None


def _is_failed(row: Any, tax: "taxonomy_mod.Taxonomy") -> bool:
    """A run counts towards the failure distribution when it is not a success.

    Runs are treated as failures when they carry a taxonomy mode, an explicit
    ``failed``/``failure`` flag, or when any gated/zero phase leaves the run
    below full marks.  Successful runs (Task == 100) are excluded.
    """
    if _mode_of(row) is not None:
        return True
    flag = _get(row, "failed", None)
    if flag is None:
        flag = _get(row, "is_failure", None)
    if flag is not None:
        return bool(flag)
    task = _get(row, "task_score", None)
    if task is None:
        task = _get(row, "task", None)
    if task is None:
        return True
    try:
        return float(task) < 100.0
    except (TypeError, ValueError):
        return True


# --------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------


def attribute_failures(
    rows: Iterable[Any],
    tax: Optional["taxonomy_mod.Taxonomy"] = None,
    *,
    rescore: bool = True,
) -> List[Dict[str, Any]]:
    """Attribute every failing row to a taxonomy mode.

    Rows already carrying a mode are passed through untouched (``rescore``
    False) or re-attributed through :func:`taxonomy.attribute_run`.  Returns
    one dict per failing row: ``{run_id, cve_id, model, trial, mode, mode_name,
    family, phase, rule}``.
    """
    tax = tax or taxonomy_mod.load_taxonomy()
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not _is_failed(row, tax):
            continue
        mode = None if rescore else _mode_of(row)
        if mode:
            attribution = {
                "mode": mode,
                "mode_name": tax.mode(mode).get("name", mode),
                "family": tax.family_of(mode),
                "phase": tax.mode(mode).get("applies_to_phase"),
                "rule": "precomputed",
            }
        else:
            try:
                attribution = taxonomy_mod.attribute_run(row, tax)
            except Exception:  # pragma: no cover - defensive
                attribution = {
                    "mode": tax.default_mode or "M12",
                    "mode_name": "ambiguous",
                    "family": "ambiguous",
                    "phase": None,
                    "rule": "attribution_error",
                }
        out.append(
            {
                "run_id": _get(row, "run_id", ""),
                "cve_id": _get(row, "cve_id", ""),
                "model": _get(row, "model", ""),
                "trial": _get(row, "trial", 1),
                "class": _first(row, ("class", "class_name", "cve_class"), ""),
                "mode": attribution.get("mode") or "M12",
                "mode_name": attribution.get("mode_name") or "ambiguous",
                "family": attribution.get("family") or "ambiguous",
                "phase": attribution.get("phase"),
                "rule": attribution.get("rule", ""),
            }
        )
    return out


# --------------------------------------------------------------------------
# Distributions
# --------------------------------------------------------------------------


def failure_distribution(
    rows: Iterable[Any],
    tax: Optional["taxonomy_mod.Taxonomy"] = None,
    *,
    by_family: bool = False,
    include_residual: bool = True,
) -> Dict[str, Any]:
    """Count attributed modes (or families) plus their percentages.

    Returns ``{"counts", "percentages", "total", "by_mode", "by_family",
    "residual_pct"}``.  Percentages use :func:`utils.io.pct` so 45.3-style
    values round half-up exactly as the paper reports them.
    """
    tax = tax or taxonomy_mod.load_taxonomy()
    attributed = attribute_failures(rows, tax)

    counts: Dict[str, int] = {}
    mode_counts: Dict[str, int] = {}
    family_counts: Dict[str, int] = {}
    for rec in attributed:
        mode = rec["mode"]
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        family = rec["family"]
        family_counts[family] = family_counts.get(family, 0) + 1

    total = len(attributed)
    if by_family:
        counts = dict(family_counts)
    else:
        # Present the six published modes first, then everything else.
        ordered = [m for m in _MODE_NAME_TO_ID.values() if m in mode_counts]
        ordered += sorted(m for m in mode_counts if m not in ordered)
        counts = {m: mode_counts[m] for m in ordered}

    percentages = {k: pct(v, total) for k, v in counts.items()}
    covered = round_half_up(sum(percentages.values()), 1)
    residual = round_half_up(max(0.0, 100.0 - covered), 1) if include_residual else 0.0

    return {
        "counts": counts,
        "percentages": percentages,
        "total": total,
        "by_mode": mode_counts,
        "by_family": family_counts,
        "residual_pct": residual,
        "percentage_sum": covered,
    }


def family_distribution(
    rows: Iterable[Any],
    tax: Optional["taxonomy_mod.Taxonomy"] = None,
) -> Dict[str, Any]:
    """Failure counts grouped by the six taxonomy families."""
    return failure_distribution(rows, tax, by_family=True)


# --------------------------------------------------------------------------
# Table rows / rendering
# --------------------------------------------------------------------------


def distribution_rows(
    summary: Optional[Mapping[str, Any]] = None,
    *,
    label: str = "fixture-derived",
) -> List[Dict[str, Any]]:
    """Formatted Table 7 rows with counts, percentages and labels.

    When ``summary`` is omitted the published Table 7 values are used so that
    ``render_failures()`` can be called with no data at all.
    """
    if summary is None:
        counts = dict(PAPER_FAILURE_DISTRIBUTION["counts"])
        percentages = dict(PAPER_FAILURE_DISTRIBUTION["percentages"])
        total = int(PAPER_FAILURE_DISTRIBUTION["total_runs"])
        return [
            {
                "mode": _MODE_NAME_TO_ID.get(name, name),
                "mode_name": name,
                "label": _MODE_LABELS.get(_MODE_NAME_TO_ID.get(name, ""), name),
                "count": counts[name],
                "pct": percentages[name],
                "total": total,
                "source": "paper",
            }
            for name in counts
        ]

    counts = dict(summary.get("counts") or {})
    percentages = dict(summary.get("percentages") or {})
    total = int(summary.get("total") or 0)
    rows: List[Dict[str, Any]] = []
    for key, count in counts.items():
        mode_id = key if str(key).startswith(("M", "X")) else _MODE_NAME_TO_ID.get(str(key), "")
        name = str(key)
        if mode_id:
            try:
                mode_record = taxonomy_mod.mode_by_id(mode_id)
                name = mode_record.get("name", name)
            except Exception:  # pragma: no cover - defensive
                pass
        rows.append(
            {
                "mode": mode_id or "",
                "mode_name": name,
                "label": _MODE_LABELS.get(mode_id, name),
                "count": int(count),
                "pct": percentages.get(key, pct(count, total)),
                "total": total,
                "source": label,
            }
        )
    return rows


def render_failures(
    summary: Optional[Mapping[str, Any]] = None,
    *,
    label: str = "fixture-derived",
    title: str = "Table 7: Failure taxonomy distribution",
) -> str:
    """Markdown rendering of the failure distribution (Table 7)."""
    rows = distribution_rows(summary, label=label)
    total = rows[0]["total"] if rows else 0
    lines = [f"# {title}", ""]
    lines.append(
        f"_Source: {label}. Published paper values are reproduced only when the "
        "full 450-run matrix has completed; fixture-derived numbers are "
        "not the paper's Table 7._"
    )
    lines.append("")
    lines.append("| Mode | Family label | Count | Share |")
    lines.append("|---|---|---:|---:|")
    family_counts: Dict[str, Any] = {}
    if summary is not None:
        family_counts = dict(summary.get("by_family") or {})
    for row in rows:
        lines.append(
            "| {label} | {name} | {count} | {pct:.1f}% |".format(
                label=row["label"],
                name=row["mode_name"],
                count=row["count"],
                pct=float(row["pct"] or 0.0),
            )
        )
    lines.append("")
    lines.append(f"Total attributed runs: **{total}**")
    if summary is not None and summary.get("residual_pct"):
        lines.append(
            "Residual not attributable to a listed mode: "
            f"**{float(summary['residual_pct']):.1f}%** (Table 7's published "
            "percentages sum to 94.0%, so the 6.0% remainder is reported "
            "explicitly rather than redistributed)."
        )
    if family_counts:
        lines.append("")
        lines.append("| Family | Count |")
        lines.append("|---|---:|")
        for family in FAILURE_FAMILIES:
            if family in family_counts:
                lines.append(f"| {family} | {family_counts[family]} |")
    lines.append("")
    return "\n".join(lines)


def render_failure_table(
    summary: Optional[Mapping[str, Any]] = None,
    *,
    label: str = "fixture-derived",
) -> str:
    """Alias of :func:`render_failures` used by some emitters/tests."""
    return render_failures(summary, label=label)


def failure_table(
    summary: Optional[Mapping[str, Any]] = None,
    *,
    label: str = "fixture-derived",
) -> str:
    """Alias of :func:`render_failures` (short form)."""
    return render_failures(summary, label=label)


def summarize_failures(
    rows: Iterable[Any],
    tax: Optional["taxonomy_mod.Taxonomy"] = None,
) -> Dict[str, Any]:
    """Distribution + Table 7 rows, ready for the report emitters."""
    summary = failure_distribution(rows, tax)
    summary["rows"] = distribution_rows(summary)
    summary["families"] = family_distribution(rows, tax)
    return summary
