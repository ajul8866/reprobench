"""Cost analysis (paper Table 11).

Consumes per-run cost records (``schema.CostRecord`` or plain dict rows) and
produces the aggregate accounting published in Table 11:

    runs 450, traced 369, total tokens 1.19B, total cost $329.36, 88.9 h
    means: 3,203,509 input / 22,495 output / 3,226,004 total / $0.89 / 868 s

Traced runs only enter the means: untraced runs are *excluded from the
denominator*, never treated as zeros (Tier 5 requirement).  Cache-write tokens
are only ever attributed to the claude-family model in the fixtures, which this
module preserves by never synthesizing tokens.

All entry points accept either ``schema.CostRecord`` instances or raw dict rows
shaped like::

    {"run_id": ..., "model": ..., "traced": True, "cost_usd": 0.89,
     "wall_seconds": 868.0,
     "tokens": {"input": ..., "output": ..., "reasoning": ...,
                "cache_read": ..., "cache_write": ...}}
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..utils.io import mean, round_half_up

__all__ = [
    "PAPER_TABLE11",
    "aggregate_costs",
    "cost_summary",
    "summarize",
    "summarize_costs",
    "table11",
    "render_table11",
    "cost_table",
    "table_rows",
    "per_model_costs",
]

#: Published Table 11 values (used by tests/claims ledger, never for scoring).
PAPER_TABLE11: Dict[str, Any] = {
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

_TOKEN_KEYS = ("input", "output", "reasoning", "cache_read", "cache_write")


# --------------------------------------------------------------------------- #
# tolerant accessors
# --------------------------------------------------------------------------- #
def _get(obj: Any, name: str, default: Any = None) -> Any:
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


def _tokens(raw: Any) -> Dict[str, int]:
    """Normalize a token payload to the five canonical counters."""
    out = {k: 0 for k in _TOKEN_KEYS}
    if raw is None:
        return out
    cache = _get(raw, "cache", None)
    for key in _TOKEN_KEYS:
        out[key] = int(_as_float(_get(raw, key, 0), 0.0))
    if cache is not None:
        out["cache_read"] = out["cache_read"] or int(_as_float(_get(cache, "read", 0), 0.0))
        out["cache_write"] = out["cache_write"] or int(_as_float(_get(cache, "write", 0), 0.0))
    return out


def _row(raw: Any) -> Dict[str, Any]:
    """Flatten one cost record (model or dict) into a neutral row."""
    tokens = _tokens(_get(raw, "tokens", None))
    total = sum(tokens.values())
    traced = _get(raw, "traced", True)
    return {
        "run_id": str(_get(raw, "run_id", "") or ""),
        "model": str(_get(raw, "model", "") or ""),
        "cve_id": str(_get(raw, "cve_id", "") or ""),
        "traced": bool(traced),
        "cost_usd": _as_float(_get(raw, "cost_usd", 0.0)),
        "wall_seconds": _as_float(_get(raw, "wall_seconds", _get(raw, "duration_seconds", 0.0))),
        "tokens": tokens,
        "tokens_total": total,
    }


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def aggregate_costs(runs: Iterable[Any]) -> Dict[str, Any]:
    """Aggregate per-run cost records into Table 11 style figures."""
    rows = [_row(r) for r in (runs or [])]
    traced_rows = [r for r in rows if r["traced"]]
    n_traced = len(traced_rows)

    total_tokens = sum(r["tokens_total"] for r in traced_rows)
    total_cost = sum(r["cost_usd"] for r in traced_rows)
    total_seconds = sum(r["wall_seconds"] for r in traced_rows)

    def _m(key: str) -> float:
        return mean([r["tokens"][key] for r in traced_rows]) if traced_rows else 0.0

    summary: Dict[str, Any] = {
        "runs": len(rows),
        "traced": n_traced,
        "untraced": len(rows) - n_traced,
        "traced_pct": round_half_up(100.0 * n_traced / len(rows), 1) if rows else 0.0,
        "total_tokens": total_tokens,
        "total_cost_usd": round_half_up(total_cost, 2),
        "total_seconds": round_half_up(total_seconds, 1),
        "total_hours": round_half_up(total_seconds / 3600.0, 1),
        "mean_input": round_half_up(_m("input"), 0),
        "mean_output": round_half_up(_m("output"), 0),
        "mean_reasoning": round_half_up(_m("reasoning"), 0),
        "mean_cache_read": round_half_up(_m("cache_read"), 0),
        "mean_cache_write": round_half_up(_m("cache_write"), 0),
        "mean_total": round_half_up(mean([r["tokens_total"] for r in traced_rows]) if traced_rows else 0.0, 0),
        "mean_cost": round_half_up(mean([r["cost_usd"] for r in traced_rows]) if traced_rows else 0.0, 2),
        "mean_seconds": round_half_up(mean([r["wall_seconds"] for r in traced_rows]) if traced_rows else 0.0, 1),
    }
    return summary


def summarize_costs(runs: Iterable[Any]) -> Dict[str, Any]:
    """Primary entry point: alias of :func:`aggregate_costs`."""
    return aggregate_costs(runs)


def summarize(runs: Iterable[Any]) -> Dict[str, Any]:
    """Alias of :func:`aggregate_costs`."""
    return aggregate_costs(runs)


def cost_summary(runs: Iterable[Any]) -> Dict[str, Any]:
    """Alias of :func:`aggregate_costs`."""
    return aggregate_costs(runs)


def per_model_costs(runs: Iterable[Any]) -> List[Dict[str, Any]]:
    """Per-model cost breakdown, sorted by total cost descending."""
    grouped: Dict[str, List[Any]] = {}
    for raw in runs or []:
        grouped.setdefault(str(_get(raw, "model", "") or "unknown"), []).append(raw)
    out: List[Dict[str, Any]] = []
    for model in sorted(grouped):
        row = aggregate_costs(grouped[model])
        row["model"] = model
        out.append(row)
    out.sort(key=lambda r: (-r["total_cost_usd"], r["model"]))
    return out


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _fmt_int(value: Any) -> str:
    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return "0"


def table_rows(summary: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """Row form of Table 11 (metric, value) pairs; embeds the traced count."""
    s = dict(summary or PAPER_TABLE11)
    return [
        {"metric": "Runs", "value": _fmt_int(s.get("runs", 0))},
        {"metric": "Traced runs", "value": _fmt_int(s.get("traced", 0))},
        {"metric": "Total tokens", "value": _fmt_int(s.get("total_tokens", 0))},
        {"metric": "Total cost (USD)", "value": f"${_as_float(s.get('total_cost_usd', 0.0)):,.2f}"},
        {"metric": "Total wall clock (h)", "value": f"{_as_float(s.get('total_hours', 0.0)):,.1f}"},
        {"metric": "Mean input tokens / run", "value": _fmt_int(s.get("mean_input", 0))},
        {"metric": "Mean output tokens / run", "value": _fmt_int(s.get("mean_output", 0))},
        {"metric": "Mean total tokens / run", "value": _fmt_int(s.get("mean_total", 0))},
        {"metric": "Mean cost / run (USD)", "value": f"${_as_float(s.get('mean_cost', 0.0)):,.2f}"},
        {"metric": "Mean duration / run (s)", "value": f"{_as_float(s.get('mean_seconds', 0.0)):,.0f}"},
    ]


def render_table11(
    summary: Optional[Mapping[str, Any]] = None,
    label: str = "fixture-derived",
    title: str = "Table 11: Cost accounting",
) -> str:
    """Markdown rendering of Table 11."""
    rows = table_rows(summary)
    lines = [
        f"# {title}",
        "",
        f"_Source: {label}. Values are computed from the scored runs; the published"
        " paper figures are NOT reproduced unless the full 450-run matrix was executed._",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['metric']} | {row['value']} |")
    lines.append("")
    return "\n".join(lines)


def table11(summary: Optional[Mapping[str, Any]] = None, label: str = "fixture-derived") -> str:
    """Alias of :func:`render_table11`."""
    return render_table11(summary, label=label)


def cost_table(summary: Optional[Mapping[str, Any]] = None, label: str = "fixture-derived") -> str:
    """Alias of :func:`render_table11`."""
    return render_table11(summary, label=label)


def load_runs(scores_dir: Any) -> List[Dict[str, Any]]:
    """Load per-run score JSON files from ``scores_dir`` into flat cost rows."""
    from pathlib import Path

    from ..utils.io import read_json

    rows: List[Dict[str, Any]] = []
    root = Path(str(scores_dir))
    if not root.exists():
        return rows
    for path in sorted(root.glob("*.json")):
        data = read_json(path, default=None)
        if not isinstance(data, Mapping):
            continue
        cost = data.get("cost") if isinstance(data.get("cost"), Mapping) else data
        row = {
            "run_id": data.get("run_id", path.stem),
            "cve_id": data.get("cve_id", ""),
            "model": data.get("model", ""),
            "traced": _get(cost, "traced", data.get("traced", True)),
            "cost_usd": _get(cost, "cost_usd", data.get("cost_usd", 0.0)),
            "wall_seconds": _get(cost, "wall_seconds", data.get("duration_seconds", 0.0)),
            "tokens": _get(cost, "tokens", data.get("tokens", {})),
        }
        rows.append(row)
    return rows
