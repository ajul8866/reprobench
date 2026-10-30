"""Per-class (BOF / CMDI / AUTH) analysis for ReproBench.

Implements the class breakdown reported in the paper (class means BOF 58.5,
CMDI 48.9, AUTH 46.0) and the per-class panels of Figure 4.

The module is a pure consumer of already-scored run/aggregate rows: it never
touches the network, the judge, or Docker, so it is safe for the offline
self-test.

Row shapes accepted (duck-typed via :func:`_get`):

* aggregate rows from :mod:`reprobench.scoring.aggregate` --
  ``{cve_id, model, plan, task, overall, max_overall, phases}``
* per-trial :class:`reprobench.schema.TrialRecord` objects
* plain dicts from ``results/scores/*.json``

Class resolution order: explicit ``class``/``class_name``/``cve_class`` field,
else the CVE id is looked up in ``data/cves.yaml`` via
:func:`reprobench.dataset.load.load_cves`.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..utils.io import mean, pct, round_half_up

__all__ = [
    "CLASSES",
    "CLASS_LABELS",
    "class_of",
    "class_means",
    "class_table",
    "class_panels",
    "class_distribution",
    "render_class_table",
    "render_class_figure_data",
    "PAPER_CLASS_MEANS",
]

CLASSES: Tuple[str, ...] = ("BOF", "CMDI", "AUTH")

CLASS_LABELS: Dict[str, str] = {
    "BOF": "Buffer overflow",
    "CMDI": "Command injection",
    "AUTH": "Authentication bypass",
}

#: Published class means (Table 2 companion numbers); analysis-only, never used
#: for scoring.  Reported so a claims ledger can compare fixture output.
PAPER_CLASS_MEANS: Dict[str, float] = {"BOF": 58.5, "CMDI": 48.9, "AUTH": 46.0}

_PHASES: Tuple[str, ...] = ("P1", "P2", "P3", "P4", "P5", "P6")
_PHASE_MAX: Dict[str, float] = {
    "P1": 15.0,
    "P2": 15.0,
    "P3": 15.0,
    "P4": 15.0,
    "P5": 20.0,
    "P6": 20.0,
}

_CLASS_KEYS = ("class", "class_name", "cve_class")
_METRIC_ALIASES = {
    "plan": ("plan", "plan_score"),
    "task": ("task", "task_score"),
    "overall": ("overall", "overall_score"),
    "max_overall": ("max_overall", "best_overall"),
}


# ---------------------------------------------------------------------------
# tolerant accessors
# ---------------------------------------------------------------------------


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a mapping or an object, tolerating absence."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        if name in obj:
            return obj[name]
        return default
    return getattr(obj, name, default)


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out:  # NaN
        return default
    return out


def _metric(row: Any, metric: str) -> float:
    for key in _METRIC_ALIASES.get(metric, (metric,)):
        value = _get(row, key)
        if value is not None:
            return _as_float(value)
    return 0.0


# ---------------------------------------------------------------------------
# class resolution
# ---------------------------------------------------------------------------


def _cve_class_map() -> Dict[str, str]:
    """Lazy ``CVE id -> class`` map from ``data/cves.yaml`` (cached)."""
    cached = globals().get("_CVE_CLASS_CACHE")
    if cached is not None:
        return cached  # type: ignore[return-value]
    mapping: Dict[str, str] = {}
    try:  # pragma: no cover - exercised through dataset module
        from ..dataset.load import load_cves

        for entry in load_cves():
            mapping[str(entry.id).strip().upper()] = str(entry.cve_class).strip().upper()
    except Exception:
        mapping = {}
    globals()["_CVE_CLASS_CACHE"] = mapping
    return mapping


def class_of(row: Any, cve_map: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Resolve the vulnerability class of a run/aggregate row.

    Returns ``"BOF"``/``"CMDI"``/``"AUTH"`` or ``None`` when unresolvable.
    """
    for key in _CLASS_KEYS:
        value = _get(row, key)
        if value:
            text = str(value).strip().upper()
            if text in CLASSES:
                return text
    cve_id = _get(row, "cve_id") or _get(row, "id") or _get(row, "cve")
    if not cve_id:
        return None
    text = str(cve_id).strip().upper()
    mapping = cve_map if cve_map is not None else _cve_class_map()
    return mapping.get(text)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def _group_by_class(
    rows: Iterable[Any], cve_map: Optional[Mapping[str, str]] = None
) -> Dict[str, List[Any]]:
    groups: Dict[str, List[Any]] = {name: [] for name in CLASSES}
    for row in rows:
        klass = class_of(row, cve_map)
        if klass in groups:
            groups[klass].append(row)
    return groups


def class_means(
    rows: Iterable[Any],
    metric: str = "overall",
    cve_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, float]:
    """Mean ``metric`` per class.

    ``metric`` is one of ``plan``/``task``/``overall``/``max_overall`` or a
    phase id (``P1``..``P6``), in which case the per-phase value is read from
    ``row.phases``/``row.phase_scores``.
    """
    groups = _group_by_class(rows, cve_map)
    out: Dict[str, float] = {}
    for klass in CLASSES:
        members = groups[klass]
        if metric in _PHASES:
            values = [_phase_value(r, metric) for r in members]
            values = [v for v in values if v is not None]
        else:
            values = [_metric(r, metric) for r in members]
        out[klass] = round_half_up(mean(values), 1) if values else 0.0
    return out


def _phase_value(row: Any, phase: str) -> Optional[float]:
    for key in ("phases", "phase_scores", "scores"):
        container = _get(row, key)
        if container is None:
            continue
        if isinstance(container, Mapping):
            if phase in container:
                return _as_float(container[phase], 0.0)
            continue
        if isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
            try:
                idx = _PHASES.index(phase)
            except ValueError:  # pragma: no cover - guarded by caller
                return None
            if idx < len(container):
                item = container[idx]
                if isinstance(item, Mapping):
                    return _as_float(
                        item.get("score", item.get("artifact_score")), 0.0
                    )
                return _as_float(item, 0.0)
    return None


def class_distribution(
    rows: Iterable[Any], cve_map: Optional[Mapping[str, str]] = None
) -> Dict[str, Any]:
    """Counts of rows and distinct CVEs per class."""
    rows = list(rows)
    groups = _group_by_class(rows, cve_map)
    counts: Dict[str, int] = {}
    cve_counts: Dict[str, int] = {}
    for klass in CLASSES:
        members = groups[klass]
        counts[klass] = len(members)
        cves = {
            str(_get(r, "cve_id") or "").strip().upper()
            for r in members
            if _get(r, "cve_id")
        }
        cve_counts[klass] = len(cves)
    total = sum(counts.values())
    return {
        "counts": counts,
        "cve_counts": cve_counts,
        "total": total,
        "percentages": {
            klass: pct(counts[klass], total) if total else 0.0 for klass in CLASSES
        },
    }


def class_table(
    rows: Iterable[Any],
    cve_map: Optional[Mapping[str, str]] = None,
    label: str = "fixture-derived",
) -> List[Dict[str, Any]]:
    """One row per class with plan/task/overall/phase means.

    This is the data behind the class-means sentence in the paper
    (BOF 58.5 / CMDI 48.9 / AUTH 46.0) and the per-class Figure 4 panels.
    """
    rows = list(rows)
    groups = _group_by_class(rows, cve_map)
    out: List[Dict[str, Any]] = []
    for klass in CLASSES:
        members = groups[klass]
        cves = {
            str(_get(r, "cve_id") or "").strip().upper()
            for r in members
            if _get(r, "cve_id")
        }
        models = {
            str(_get(r, "model") or "").strip() for r in members if _get(r, "model")
        }
        row: Dict[str, Any] = {
            "class": klass,
            "label": CLASS_LABELS.get(klass, klass),
            "runs": len(members),
            "cves": len(cves),
            "models": len(models),
            "avg_plan": round_half_up(mean([_metric(r, "plan") for r in members]), 1)
            if members
            else 0.0,
            "avg_task": round_half_up(mean([_metric(r, "task") for r in members]), 1)
            if members
            else 0.0,
            "avg_overall": round_half_up(
                mean([_metric(r, "overall") for r in members]), 1
            )
            if members
            else 0.0,
            "max_overall": round_half_up(
                max([_metric(r, "max_overall") for r in members], default=0.0), 1
            )
            if members
            else 0.0,
            "label_source": label,
        }
        for phase in _PHASES:
            values = [_phase_value(r, phase) for r in members]
            values = [v for v in values if v is not None]
            row[f"avg_{phase.lower()}"] = (
                round_half_up(mean(values), 2) if values else 0.0
            )
            row[f"norm_{phase.lower()}"] = (
                round_half_up(100.0 * mean(values) / _PHASE_MAX[phase], 1)
                if values
                else 0.0
            )
        out.append(row)
    return out


def class_panels(
    rows: Iterable[Any],
    cve_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Figure 4 panel data: normalised per-phase rates per class.

    Returns ``{panel_name: {"phases": {phase: normalised_pct}, "runs": n}}``
    with the special ``"all"`` panel first, mirroring the paper's four panels
    (all / BOF / CMDI / AUTH).
    """
    rows = list(rows)
    groups = _group_by_class(rows, cve_map)
    panels: Dict[str, Dict[str, Any]] = {}

    def _panel(members: Sequence[Any]) -> Dict[str, Any]:
        phases: Dict[str, float] = {}
        for phase in _PHASES:
            values = [_phase_value(r, phase) for r in members]
            values = [v for v in values if v is not None]
            phases[phase] = (
                round_half_up(100.0 * mean(values) / _PHASE_MAX[phase], 1)
                if values
                else 0.0
            )
        return {"phases": phases, "runs": len(members)}

    panels["all"] = _panel(rows)
    for klass in CLASSES:
        panels[klass] = _panel(groups[klass])
    return panels


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def render_class_table(
    rows: Iterable[Any],
    cve_map: Optional[Mapping[str, str]] = None,
    label: str = "fixture-derived",
    title: str = "Per-class results",
) -> str:
    """Markdown table of the per-class means."""
    table = class_table(rows, cve_map=cve_map, label=label)
    lines = [
        f"# {title}",
        "",
        f"_Source: {label}. Values are not the paper's Table 2 numbers unless "
        "the full 450-run matrix was executed._",
        "",
        "| Class | CVEs | Runs | Avg Plan | Avg Task | Avg Overall | Max Overall |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in table:
        lines.append(
            "| {klass} ({label}) | {cves} | {runs} | {avg_plan:.1f} | "
            "{avg_task:.1f} | {avg_overall:.1f} | {max_overall:.1f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def render_class_figure_data(
    rows: Iterable[Any],
    cve_map: Optional[Mapping[str, str]] = None,
    label: str = "fixture-derived",
) -> str:
    """Markdown rendering of the Figure 4 panel matrix."""
    panels = class_panels(rows, cve_map=cve_map)
    order = ["all"] + list(CLASSES)
    lines = [
        "# Figure 4 data: per-phase normalised rates by class panel",
        "",
        f"_Source: {label}._",
        "",
        "| Panel | Runs | " + " | ".join(_PHASES) + " |",
        "| --- | --- | " + " | ".join("---" for _ in _PHASES) + " |",
    ]
    for name in order:
        panel = panels.get(name, {"phases": {}, "runs": 0})
        cells = " | ".join(
            f"{panel['phases'].get(p, 0.0):.1f}" for p in _PHASES
        )
        lines.append(f"| {name} | {panel['runs']} | {cells} |")
    return "\n".join(lines) + "\n"
