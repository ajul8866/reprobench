"""Report orchestration for ReproBench analysis.

This module is the single entry point used by ``reprobench analyze`` and by
``scripts/reproduce_tables.sh``.  It loads the per-trial score records that the
scoring engine wrote under ``results/scores`` and emits every analysis artifact
described in the paper:

* ``table2.md`` / ``table2.json``  -- Table 2 (AvgPlan, AvgTask, AvgOverall,
  MaxOverall, P5>=15, P6>=15) plus the macro-averaged "All" row.
* ``fig4.md`` / ``fig4.json``      -- Figure 4 per-phase normalized achievement
  rates for the ``all``/``BOF``/``CMDI``/``AUTH`` panels.
* ``fig5.md`` / ``fig5.json``      -- Figure 5 stage-level means under
  best-of-three.
* ``classes.md``                   -- per-class results table.
* ``divergence.md`` / ``divergence.json`` -- PSC phase-divergence counts.
* ``failures.md`` / ``failures.json`` -- failure-taxonomy distribution.
* ``table11.md``                   -- Table 11 cost accounting.
* ``claims.md``                    -- claims ledger (``write_claims``).

Nothing here is allowed to claim the paper's 450-run numbers: every rendered
document carries a ``label`` (``"fixture-derived"`` by default) and an explicit
disclaimer, and the claims ledger refuses to mark run-dependent claims as
"verified" unless a completed run matrix is present.

All functions are offline: they read JSON files from disk and write Markdown /
JSON reports.  No network, no Docker, no judge model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..utils.io import (
    ensure_dir,
    load_dataset,
    read_json,
    read_jsonl,
    relpath,
    write_json,
    write_text,
)
from ..utils.log import get_logger
from . import classwise, cost, divergence, failures, phases

LOG = get_logger("analysis.tables")

PHASES_IDS: Tuple[str, ...] = ("P1", "P2", "P3", "P4", "P5", "P6")

DEFAULT_LABEL = "fixture-derived"

DISCLAIMER = (
    "> **Source: {label}.** These values were produced from the runs present in "
    "this workspace. They are *not* the paper's published 450-run results and "
    "must not be reported as Table 2 / Figure 4 / Figure 5 / Table 11 of "
    "ReproBench unless the complete 30 CVE x 5 model x 3 trial matrix has been "
    "executed and scored."
)

#: Files emitted by :func:`build_reports`, in a stable order.
REPORT_FILES: Tuple[str, ...] = (
    "table2.md",
    "table2.json",
    "fig4.md",
    "fig4.json",
    "fig5.md",
    "fig5.json",
    "classes.md",
    "divergence.md",
    "divergence.json",
    "failures.md",
    "failures.json",
    "table11.md",
    "summary.json",
)

__all__ = [
    "DEFAULT_LABEL",
    "DISCLAIMER",
    "REPORT_FILES",
    "build_reports",
    "claims_ledger",
    "render_fig4",
    "render_fig5",
    "render_table2",
    "write_claims",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Render a GitHub-flavoured Markdown table."""
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        cells = ["" if cell is None else str(cell) for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _fmt(value: Any, digits: int = 1) -> str:
    """Format a number for a table cell without exploding on ``None``."""
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if digits == 0:
        return f"{number:.0f}"
    return f"{number:.{digits}f}"


def _disclaimer(label: str) -> str:
    return DISCLAIMER.format(label=label)


# ---------------------------------------------------------------------------
# Table 2
# ---------------------------------------------------------------------------
def render_table2(payload: Mapping[str, Any], label: str = DEFAULT_LABEL,
                  title: str = "Table 2: Best-of-three results per model") -> str:
    """Render the Table 2 payload returned by :func:`phases.table2`."""
    rows = payload.get("rows") or []
    headers = ["Model", "AvgPlan", "AvgTask", "AvgOverall", "MaxOverall",
               "P5>=15", "P6>=15", "CVEs"]
    body = [
        [
            row.get("model", ""),
            _fmt(row.get("avg_plan"), 1),
            _fmt(row.get("avg_task"), 1),
            _fmt(row.get("avg_overall"), 1),
            _fmt(row.get("max_overall"), 1),
            row.get("p5_at_least_15", 0),
            row.get("p6_at_least_15", 0),
            row.get("cves", 0),
        ]
        for row in rows
    ]
    success = payload.get("success") or {}
    parts = [
        f"# {title}",
        "",
        _disclaimer(label),
        "",
        _md_table(headers, body),
        "",
        "Best-of: {best} (mode `{mode}`), {cells} (CVE, model) cells from "
        "{trials} trials.".format(
            best=payload.get("best_of", 3),
            mode=payload.get("best_by", "independent"),
            cells=payload.get("cells", 0),
            trials=payload.get("trials", 0),
        ),
    ]
    if success:
        parts.append(
            "Successful reproductions (P5>=15 and P6>=15): "
            f"{success.get('successes', 0)}/{success.get('total', 0)} "
            f"({_fmt(success.get('rate_pct'), 1)}%)."
        )
    parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Figure 4 / Figure 5
# ---------------------------------------------------------------------------
def render_fig4(panel_data: Mapping[str, Any], label: str = DEFAULT_LABEL,
                title: str = "Figure 4: Per-phase achievement rate (normalized)") -> str:
    """Render per-phase normalized rates for the ``all``/class panels."""
    panels = panel_data.get("panels") or panel_data
    headers = ["Panel", "Runs"] + list(PHASES_IDS)
    body: List[List[Any]] = []
    for panel in classwise.DIVERGENCE_PANELS if hasattr(classwise, "DIVERGENCE_PANELS") else ("all", "BOF", "CMDI", "AUTH"):
        data = panels.get(panel)
        if not isinstance(data, Mapping):
            continue
        phase_values = data.get("phases") or {}
        body.append(
            [panel, data.get("runs", 0)]
            + [_fmt(phase_values.get(pid), 1) for pid in PHASES_IDS]
        )
    if not body:
        for panel, data in panels.items():
            if not isinstance(data, Mapping):
                continue
            phase_values = data.get("phases") or {}
            body.append(
                [panel, data.get("runs", 0)]
                + [_fmt(phase_values.get(pid), 1) for pid in PHASES_IDS]
            )
    return "\n".join(
        [
            f"# {title}",
            "",
            _disclaimer(label),
            "",
            _md_table(headers, body),
            "",
            "Each cell is the mean of `phase_score / phase_max * 100` over the "
            "best-of-three aggregate for that panel.",
            "",
        ]
    )


def render_fig5(stage_data: Mapping[str, Any], label: str = DEFAULT_LABEL,
                title: str = "Figure 5: Stage-level averages (best-of-three)") -> str:
    """Render Figure 5 stage means (paper: P1 13.9/15, P5 3.7/20, P6 2.8/20)."""
    means = stage_data.get("means") or stage_data.get("phases") or {}
    headers = ["Stage", "Mean", "Max", "Normalized"]
    body: List[List[Any]] = []
    for pid in PHASES_IDS:
        entry = means.get(pid)
        if isinstance(entry, Mapping):
            mean_value = entry.get("mean")
            max_value = entry.get("max", _phase_max(pid))
            norm = entry.get("normalized")
        else:
            mean_value = entry
            max_value = _phase_max(pid)
            norm = (float(mean_value) / max_value * 100.0) if mean_value else 0.0
        body.append(
            [
                pid,
                _fmt(mean_value, 1),
                _fmt(max_value, 0),
                _fmt(norm, 1),
            ]
        )
    return "\n".join(
        [
            f"# {title}",
            "",
            _disclaimer(label),
            "",
            _md_table(headers, body),
            "",
            "Reference values from the paper (not reproduced offline): "
            "P1 13.9/15, P5 3.7/20, P6 2.8/20.",
            "",
        ]
    )


def _phase_max(phase_id: str) -> float:
    return 20.0 if phase_id in ("P5", "P6") else 15.0


# ---------------------------------------------------------------------------
# report builder
# ---------------------------------------------------------------------------
def build_reports(scores_dir: str | Path, out_dir: str | Path,
                  label: str = DEFAULT_LABEL,
                  best_by: str = "independent",
                  models: Optional[Sequence[str]] = None,
                  dataset_path: Optional[str | Path] = None) -> List[Path]:
    """Build every analysis artifact from a ``results/scores`` directory.

    Parameters
    ----------
    scores_dir:
        Directory containing per-run score JSON files (possibly nested).
    out_dir:
        Destination directory for the rendered reports.
    label:
        Provenance label embedded in every rendered document.  Defaults to
        ``"fixture-derived"`` so offline numbers cannot be mistaken for the
        paper's published results.
    best_by:
        ``"independent"`` (default, per-metric maxima) or ``"same_run"``.
    models:
        Optional model ordering for Table 2; defaults to discovery order.
    dataset_path:
        Optional ``cves.yaml`` override for class resolution.

    Returns
    -------
    list[pathlib.Path]
        Every written report path, in a stable order.
    """
    out = ensure_dir(Path(out_dir))
    trials = _load_trials(scores_dir)
    LOG.info("build_reports: %d trial records from %s", len(trials), scores_dir)

    written: List[Path] = []

    # ---- Table 2 -----------------------------------------------------
    try:
        table2_payload = phases.table2(trials, models=models, best_by=best_by)
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("table2 failed: %s", exc)
        table2_payload = {"rows": [], "cells": 0, "trials": len(trials), "success": {}}
    written.append(write_text(out / "table2.md", render_table2(table2_payload, label)))
    written.append(write_json(out / "table2.json", table2_payload))

    # ---- Figure 4 ----------------------------------------------------
    try:
        fig4_payload = phases.fig4(trials)
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("fig4 failed: %s", exc)
        fig4_payload = {"panels": {}}
    written.append(write_text(out / "fig4.md", render_fig4(fig4_payload, label)))
    written.append(write_json(out / "fig4.json", fig4_payload))

    # ---- Figure 5 ----------------------------------------------------
    try:
        fig5_payload = phases.fig5(trials, best_by=best_by)
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("fig5 failed: %s", exc)
        fig5_payload = {"means": {}}
    written.append(write_text(out / "fig5.md", render_fig5(fig5_payload, label)))
    written.append(write_json(out / "fig5.json", fig5_payload))

    # ---- per-class table ---------------------------------------------
    cve_map = _cve_class_map(dataset_path)
    try:
        classes_md = classwise.render_class_table(
            trials, cve_map=cve_map or None, label=label
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("class table failed: %s", exc)
        classes_md = f"# Per-class results\n\n{_disclaimer(label)}\n"
    written.append(write_text(out / "classes.md", classes_md))

    # ---- divergence --------------------------------------------------
    try:
        div_payload = divergence.divergence_summary(trials)
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("divergence failed: %s", exc)
        div_payload = {"counts": {}, "cves": {}, "divergent_cves": [], "total": 0}
    written.append(write_text(out / "divergence.md", _render_divergence(div_payload, label)))
    written.append(write_json(out / "divergence.json", div_payload))

    # ---- failure taxonomy --------------------------------------------
    try:
        fail_payload = failures.failure_distribution(trials)
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("failure distribution failed: %s", exc)
        fail_payload = {"counts": {}, "percentages": {}, "total": 0}
    written.append(write_text(out / "failures.md", _render_failures(fail_payload, label)))
    written.append(write_json(out / "failures.json", fail_payload))

    # ---- Table 11 (cost) ---------------------------------------------
    try:
        cost_summary = cost.aggregate_costs(trials)
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("cost aggregation failed: %s", exc)
        cost_summary = {}
    try:
        table11_md = cost.render_table11(cost_summary or None, label=label)
    except Exception:  # pragma: no cover - defensive
        table11_md = f"# Table 11\n\n{_disclaimer(label)}\n"
    written.append(write_text(out / "table11.md", table11_md))

    # ---- machine-readable summary ------------------------------------
    summary = {
        "label": label,
        "best_by": best_by,
        "scores_dir": str(scores_dir),
        "trials": len(trials),
        "models": [row.get("model") for row in table2_payload.get("rows", [])],
        "cells": table2_payload.get("cells", 0),
        "success": table2_payload.get("success", {}),
        "divergence": div_payload.get("counts", {}),
        "failures": fail_payload.get("percentages", {}),
        "cost": {k: v for k, v in (cost_summary or {}).items()
                 if k in ("runs", "traced", "total_tokens", "total_cost_usd",
                          "total_hours", "mean_total", "mean_cost")},
    }
    written.append(write_json(out / "summary.json", summary))

    LOG.info("build_reports: wrote %d files to %s", len(written), out)
    return written


def _render_divergence(payload: Mapping[str, Any], label: str) -> str:
    counts = payload.get("counts") or {}
    total = payload.get("total", sum(counts.values()) if counts else 0)
    rows = [[pid, counts.get(pid, 0)] for pid in ("P2", "P4", "P5")]
    return "\n".join(
        [
            "# Phase divergence (PSC)",
            "",
            _disclaimer(label),
            "",
            _md_table(["Phase", "Divergent CVEs"], rows),
            "",
            f"Total divergent CVEs: {total} (earliest-phase attribution).",
            "",
            "Paper reference counts (not reproduced offline): P2 12, P4 2, P5 5.",
            "",
        ]
    )


def _render_failures(payload: Mapping[str, Any], label: str) -> str:
    counts = payload.get("counts") or {}
    percentages = payload.get("percentages") or {}
    headers = ["Mode", "Count", "Percent"]
    rows = [
        [name, counts.get(name, 0), _fmt(percentages.get(name), 1)]
        for name in sorted(counts)
    ]
    if not rows:
        rows = [["(no attributed failures)", 0, "0.0"]]
    return "\n".join(
        [
            "# Failure distribution",
            "",
            _disclaimer(label),
            "",
            _md_table(headers, rows),
            "",
            "Paper distribution (not reproduced offline): simulation 204 (45.3%), "
            "rehosting 93 (20.7%), acquisition 75 (16.7%), extraction 36 (8.0%), "
            "infrastructure 15 (3.3%), residual 27 (6.0%).",
            "",
        ]
    )


# ---------------------------------------------------------------------------
# loading helpers
# ---------------------------------------------------------------------------
def _load_trials(scores_dir: str | Path) -> List[Dict[str, Any]]:
    """Load trial rows, delegating to :func:`analysis.phases.load_trials`."""
    try:
        rows = phases.load_trials(scores_dir)
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("phases.load_trials failed (%s); falling back to raw JSON", exc)
        rows = []
    if rows:
        return list(rows)
    return _raw_load(scores_dir)


def _raw_load(scores_dir: str | Path) -> List[Dict[str, Any]]:
    """Minimal fallback loader for plain score JSON files."""
    root = Path(scores_dir)
    if not root.exists():
        return []
    if root.is_file():
        payload = read_json(root, default=None)
        return [payload] if isinstance(payload, dict) else []
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        payload = read_json(path, default=None)
        if isinstance(payload, dict):
            rows.append(payload)
        elif isinstance(payload, list):
            rows.extend(p for p in payload if isinstance(p, dict))
    return rows


def _cve_class_map(dataset_path: Optional[str | Path]) -> Dict[str, str]:
    """Build a CVE id -> class map from ``data/cves.yaml``."""
    try:
        if dataset_path is not None:
            from ..utils.io import load_yaml

            doc = load_yaml(dataset_path)
        else:
            doc = load_dataset("cves.yaml")
    except Exception:  # pragma: no cover - defensive
        return {}
    mapping: Dict[str, str] = {}
    for entry in (doc or {}).get("cves", []) or []:
        cve_id = str(entry.get("id", "")).strip()
        cve_class = str(entry.get("class", "")).strip().upper()
        if cve_id and cve_class:
            mapping[cve_id] = cve_class
    return mapping


# ---------------------------------------------------------------------------
# claims ledger
# ---------------------------------------------------------------------------
def claims_ledger(dataset_path: Optional[str | Path] = None,
                  scores_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    """Build the paper-claim ledger with evidence pointers.

    Every claim is marked ``verified`` / ``partial`` / ``not_verifiable``
    (blocked) / ``deviation``.  Run-dependent claims are only marked
    ``verified`` when a completed run matrix is detected under ``scores_dir``.
    """
    dataset = _dataset_stats(dataset_path)
    run_stats = _run_stats(scores_dir)
    complete = bool(run_stats.get("complete_matrix"))

    claims: List[Dict[str, Any]] = [
        _claim(
            "C1", "Benchmark contains 30 CVEs, 10 per class, 14 vendors, 2020-2026",
            status="verified" if dataset.get("valid") else "partial",
            evidence=dataset.get("evidence", "data/cves.yaml"),
            note=dataset.get("note", ""),
        ),
        _claim(
            "C2", "Six-phase rubric with maxima 15/15/15/15/20/20 summing to 100",
            status="verified",
            evidence="config/phases.yaml; tests/test_task_score.py::test_phase_maxima_match_paper",
            note="Item points transcribed from Table 4.",
        ),
        _claim(
            "C3", "Plan = 0.6C + 0.3D + 0.1F with golden values 95.0 / 84.0 / 66.5",
            status="verified",
            evidence="src/reprobench/scoring/plan.py; tests/test_golden_tables.py",
            note="Dependency unit resolved as pairwise inversions (deviation a).",
            deviation="a",
        ),
        _claim(
            "C4", "Overall = 0.2*Plan + 0.8*Task reproduces 99.0 / 59.2 / 24.5",
            status="verified",
            evidence="src/reprobench/scoring/aggregate.py; tests/test_golden_tables.py",
            note="Best-of-three uses independent per-metric maxima (deviation c).",
            deviation="c",
        ),
        _claim(
            "C5", "Real-target gating zeroes P5/P6 on simulation substitution",
            status="verified",
            evidence="src/reprobench/scoring/gating.py; tests/test_gating.py (6 cases)",
            note="Judge confirmation injected; deterministic pre-scan drives nomination.",
        ),
        _claim(
            "C6", "Attempt credit replaces a zero artifact score, never adds",
            status="verified",
            evidence="src/reprobench/scoring/attempt.py; tests/test_attempt.py",
            note="",
        ),
        _claim(
            "C7", "Failure taxonomy has 6 families / 13 modes",
            status="deviation",
            evidence="config/failure_taxonomy.yaml",
            note=("Paper text says 6 families/13 modes but Table 7 lists 5/12; "
                  "we implement 12 modes plus the `ambiguous` residual family and "
                  "an extension slot (deviation d)."),
            deviation="d",
        ),
        _claim(
            "C8", "Failure percentages sum to 100%",
            status="deviation",
            evidence="config/failure_taxonomy.yaml distribution block",
            note=("Published percentages sum to 94.0%; the 6.0% residual is "
                  "reported explicitly rather than redistributed (deviation e)."),
            deviation="e",
        ),
        _claim(
            "C9", "Phase-failure threshold and early-abandonment window",
            status="deviation",
            evidence="config/benchmark.yaml; src/reprobench/scoring/taxonomy.py",
            note=("Paper does not specify them; we default to 0.5*max and k=5 "
                  "no-tool turns (deviation g)."),
            deviation="g",
        ),
        _claim(
            "C10", "Judge repeats use median of three",
            status="deviation",
            evidence="config/benchmark.yaml judge.repeats",
            note="Observed default is 1 repeat; repeats=3 median is opt-in (deviation h).",
            deviation="h",
        ),
        _claim(
            "C11", "Same-architecture chroot claims require arm64",
            status="deviation",
            evidence="config/benchmark.yaml harness.host_arch_paper",
            note=("Paper host is ARM64 (GB10); x86_64 development needs qemu-aarch64 "
                  "for same-arch chroot claims (deviation i)."),
            deviation="i",
        ),
        _claim(
            "C12", "Table 2: AvgPlan/AvgTask/AvgOverall/MaxOverall per model (450 runs)",
            status="verified" if complete else "not_verifiable",
            evidence="results/reports/table2.md" if complete else "requires 450 completed runs",
            note=("No completed 30x5x3 matrix in this workspace; the emitter is "
                  "implemented and validated on fixtures only."),
            blocked=not complete,
        ),
        _claim(
            "C13", "Figure 4 per-phase rates across all/BOF/CMDI/AUTH panels",
            status="verified" if complete else "not_verifiable",
            evidence="results/reports/fig4.md" if complete else "requires 450 completed runs",
            blocked=not complete,
        ),
        _claim(
            "C14", "Figure 5 stage means (P1 13.9/15, P5 3.7/20, P6 2.8/20)",
            status="verified" if complete else "not_verifiable",
            evidence="results/reports/fig5.md" if complete else "requires 450 completed runs",
            blocked=not complete,
        ),
        _claim(
            "C15", "Table 11 cost accounting (369/450 traced, 1.19B tokens, $329.36)",
            status="verified" if complete else "not_verifiable",
            evidence=("src/reprobench/analysis/cost.py; tests/test_cost.py"
                      if not complete else "results/reports/table11.md"),
            note=("Accounting logic reproduces the published means from fixtures; "
                  "the published totals themselves require live runs."),
            blocked=not complete,
        ),
        _claim(
            "C16", "Divergence counts P2 12 / P4 2 / P5 5 (19 of 30 CVEs)",
            status="verified" if complete else "not_verifiable",
            evidence="src/reprobench/analysis/divergence.py; tests/test_divergence.py",
            note="Rule and thresholds verified on synthetic corpora.",
            blocked=not complete,
        ),
    ]

    summary = {
        "total": len(claims),
        "verified": sum(1 for c in claims if c["status"] == "verified"),
        "partial": sum(1 for c in claims if c["status"] == "partial"),
        "deviation": sum(1 for c in claims if c["status"] == "deviation"),
        "not_verifiable": sum(1 for c in claims if c["status"] == "not_verifiable"),
    }
    return {
        "claims": claims,
        "summary": summary,
        "dataset": dataset,
        "runs": run_stats,
    }


def _claim(claim_id: str, text: str, *, status: str, evidence: str,
           note: str = "", deviation: str = "", blocked: bool = False) -> Dict[str, Any]:
    return {
        "id": claim_id,
        "claim": text,
        "status": status,
        "evidence": evidence,
        "note": note,
        "deviation": deviation,
        "blocked": blocked,
    }


def _dataset_stats(dataset_path: Optional[str | Path]) -> Dict[str, Any]:
    """Validate the released dataset for the ledger's C1 claim."""
    try:
        from ..dataset.load import validate_dataset

        problems = validate_dataset(dataset_path)
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "valid": False,
            "evidence": "data/cves.yaml",
            "note": f"dataset validation unavailable: {exc}",
        }
    if problems:
        return {
            "valid": False,
            "evidence": "data/cves.yaml",
            "note": "; ".join(str(p) for p in problems[:5]),
        }
    return {
        "valid": True,
        "evidence": "data/cves.yaml; tests/test_golden_tables.py::test_dataset_has_30_unique_cves",
        "note": "30 unique CVEs, 10 per class, 14 vendors, years 2020-2026, F3 CWE mapping holds.",
    }


def _run_stats(scores_dir: Optional[str | Path]) -> Dict[str, Any]:
    """Detect whether a completed 450-run matrix exists."""
    stats: Dict[str, Any] = {
        "scores_dir": str(scores_dir) if scores_dir else "",
        "trials": 0,
        "cves": 0,
        "models": 0,
        "complete_matrix": False,
    }
    if scores_dir is None:
        return stats
    trials = _load_trials(scores_dir)
    stats["trials"] = len(trials)
    cves = {row.get("cve_id") for row in trials if row.get("cve_id")}
    models = {row.get("model") for row in trials if row.get("model")}
    stats["cves"] = len(cves)
    stats["models"] = len(models)
    stats["complete_matrix"] = bool(
        len(cves) >= 30 and len(models) >= 5 and len(trials) >= 450
    )
    return stats


def render_claims(ledger: Mapping[str, Any], label: str = DEFAULT_LABEL) -> str:
    """Render the claims ledger as Markdown."""
    claims = ledger.get("claims") or []
    summary = ledger.get("summary") or {}
    headers = ["ID", "Claim", "Status", "Evidence", "Notes"]
    rows = [
        [
            c.get("id", ""),
            c.get("claim", ""),
            c.get("status", ""),
            c.get("evidence", ""),
            c.get("note", ""),
        ]
        for c in claims
    ]
    lines = [
        "# Claims ledger",
        "",
        f"Generated from **{label}** artifacts.",
        "",
        "Statuses: `verified` (mechanically checked), `partial`, "
        "`not_verifiable` (blocked on unavailable data), `deviation` "
        "(documented divergence from the paper).",
        "",
        _md_table(headers, rows),
        "",
        "Summary: " + ", ".join(f"{k}={v}" for k, v in sorted(summary.items())),
        "",
    ]
    runs = ledger.get("runs") or {}
    if runs:
        lines.append(
            "Run matrix detected: {trials} trials across {cves} CVEs and "
            "{models} models (complete matrix: {complete}).".format(
                trials=runs.get("trials", 0),
                cves=runs.get("cves", 0),
                models=runs.get("models", 0),
                complete=runs.get("complete_matrix", False),
            )
        )
        lines.append("")
    return "\n".join(lines)


def write_claims(out_dir: str | Path,
                 dataset_path: Optional[str | Path] = None,
                 scores_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    """Write ``claims.md`` (+ JSON) into ``out_dir`` and return the ledger."""
    out = ensure_dir(Path(out_dir))
    ledger = claims_ledger(dataset_path=dataset_path, scores_dir=scores_dir)
    write_text(out / "claims.md", render_claims(ledger))
    write_json(out / "claims.json", ledger)
    LOG.info("write_claims: %d claims written to %s", len(ledger["claims"]), out)
    return ledger
