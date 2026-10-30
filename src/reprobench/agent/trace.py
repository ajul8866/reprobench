"""Run-trace recording/reading for ReproBench.

The trace is the *raw evidence* for attempt scoring, gating and cost
accounting.  It is written as an append-only JSONL stream of
OpenAI-compatible message objects so that :mod:`reprobench.scoring.evidence`
can parse it without any knowledge of a particular agent scaffold:

* assistant messages may carry ``tool_calls`` with
  ``function.name`` / ``function.arguments`` (arguments are a JSON string),
* tool result messages carry the ``content`` output,
* the final assistant message carries ``usage`` and ``cost``.

A ``summary.json`` next to the stream stores the aggregate token/cost record
(``tokens{input,output,reasoning,cache{read,write}}`` and ``cost``).

Nothing here performs network or model calls; the harness adapters feed the
recorder.  All writes are best effort: tracing failures must never abort a
run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..utils.io import ensure_dir, read_json, read_jsonl, write_json
from ..utils.log import get_logger

__all__ = [
    "TRACE_FILENAME",
    "SUMMARY_FILENAME",
    "TraceRecorder",
    "iter_tool_calls",
    "normalize_tokens",
    "read_trace",
    "trace_path_for",
]

log = get_logger("agent.trace")

TRACE_FILENAME = "events.jsonl"
SUMMARY_FILENAME = "summary.json"

_TOKEN_KEYS = ("input", "output", "reasoning", "cache_read", "cache_write")


def trace_path_for(run_dir: str | Path) -> Path:
    """Return the trace directory for ``run_dir`` (``<run_dir>/trace``)."""
    return Path(run_dir) / "trace"


def normalize_tokens(raw: Any) -> Dict[str, int]:
    """Normalize any token payload into the five canonical counters.

    Accepts nested ``cache: {read, write}`` (Anthropic/OpenCode style) as well
    as flat ``cache_read``/``cache_write`` keys.
    """
    out = {key: 0 for key in _TOKEN_KEYS}
    if not isinstance(raw, dict):
        return out
    for key in ("input", "output", "reasoning"):
        for alias in (key, f"{key}_tokens"):
            value = raw.get(alias)
            if isinstance(value, (int, float)):
                out[key] = int(value)
                break
    cache = raw.get("cache") if isinstance(raw.get("cache"), dict) else {}
    for key, aliases in (
        ("cache_read", ("cache_read", "cache_read_input_tokens", "cache_creation_input_tokens")),
        ("cache_write", ("cache_write", "cache_write_input_tokens")),
    ):
        for alias in aliases:
            if isinstance(raw.get(alias), (int, float)):
                out[key] = int(raw[alias])  # type: ignore[assignment]
                break
        else:
            flat = key.split("_", 1)[1]  # "read" / "write"
            if isinstance(cache.get(flat), (int, float)):
                out[key] = int(cache[flat])  # type: ignore[assignment]
    return out


class TraceRecorder:
    """Append-only recorder for one trial."""

    def __init__(
        self,
        trace_dir: str | Path,
        run_id: str = "run",
        cve_id: str = "",
        model: str = "",
        trial: int = 1,
    ) -> None:
        self.dir = ensure_dir(Path(trace_dir))
        self.run_id = run_id
        self.cve_id = cve_id
        self.model = model
        self.trial = trial
        self.path = self.dir / TRACE_FILENAME
        self.messages: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.tokens: Dict[str, int] = {key: 0 for key in _TOKEN_KEYS}
        self.cost_usd: float = 0.0
        self.status: str = "running"
        # truncate a previous trace for the same run id so reruns are clean
        try:
            self.path.write_text("", encoding="utf-8")
        except OSError:  # pragma: no cover - defensive
            pass

    # ------------------------------------------------------------------ write
    def _append(self, record: Dict[str, Any]) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        except OSError as exc:  # pragma: no cover - defensive
            log.warning("trace write failed: %s", exc)

    def message(
        self,
        role: str,
        content: str,
        tool_calls: Optional[Iterable[Dict[str, Any]]] = None,
        usage: Optional[Dict[str, Any]] = None,
        cost: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Record one message; returns the stored record."""
        record: Dict[str, Any] = {"role": role, "content": content or ""}
        calls = list(tool_calls or [])
        if calls:
            record["tool_calls"] = calls
            for index, call in enumerate(calls):
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                self.tool_calls.append(
                    {
                        "index": len(self.tool_calls),
                        "name": fn.get("name") or call.get("name", ""),
                        "input": fn.get("arguments", call.get("input", "")),
                    }
                )
        if usage:
            record["usage"] = usage
            merged = normalize_tokens(usage)
            for key, value in merged.items():
                self.tokens[key] += value
        if cost is not None:
            record["cost"] = float(cost)
            self.cost_usd += float(cost)
        self.messages.append(record)
        self._append(record)
        return record

    def tool_call(
        self,
        name: str,
        arguments: str | Dict[str, Any],
        output: str = "",
        exit_code: Optional[int] = None,
    ) -> None:
        """Convenience: record an assistant tool call plus its result."""
        args = arguments if isinstance(arguments, str) else json.dumps(arguments, sort_keys=True)
        call_id = f"call_{len(self.tool_calls)}"
        self.message(
            "assistant",
            "",
            tool_calls=[
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                }
            ],
        )
        payload: Dict[str, Any] = {"role": "tool", "tool_call_id": call_id, "content": output}
        if exit_code is not None:
            payload["exit_code"] = exit_code
        if self.tool_calls:
            self.tool_calls[-1]["exit_code"] = exit_code
            self.tool_calls[-1]["output"] = (output or "")[-4000:]
        self.messages.append(payload)
        self._append(payload)

    # ----------------------------------------------------------------- final
    def finalize(
        self,
        tokens: Optional[Dict[str, Any]] = None,
        cost_usd: Optional[float] = None,
        status: str = "ok",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Write ``summary.json`` and return the summary dict."""
        if tokens:
            self.tokens = normalize_tokens(tokens)
        if cost_usd is not None:
            self.cost_usd = float(cost_usd)
        self.status = status
        summary: Dict[str, Any] = {
            "run_id": self.run_id,
            "cve_id": self.cve_id,
            "model": self.model,
            "trial": self.trial,
            "status": status,
            "messages": len(self.messages),
            "tool_calls": len(self.tool_calls),
            "tokens": dict(self.tokens),
            "cost_usd": round(self.cost_usd, 6),
        }
        if extra:
            summary.update(extra)
        try:
            write_json(self.dir / SUMMARY_FILENAME, summary)
        except OSError as exc:  # pragma: no cover - defensive
            log.warning("trace summary write failed: %s", exc)
        return summary

    # ------------------------------------------------------------------ read
    def tool_log(self) -> List[Dict[str, Any]]:
        """Return the in-memory tool-call log (name/input/output/exit_code)."""
        return list(self.tool_calls)


def iter_tool_calls(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract a flat tool-call list from an iterable of trace records."""
    calls: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        for call in record.get("tool_calls") or []:
            fn = call.get("function", {}) if isinstance(call, dict) else {}
            calls.append(
                {
                    "name": fn.get("name") or call.get("name", ""),
                    "input": fn.get("arguments", call.get("input", "")),
                }
            )
    return calls


def read_trace(trace_dir: str | Path) -> Dict[str, Any]:
    """Read a trace directory into ``{messages, tool_calls, tokens, cost, ...}``."""
    directory = Path(trace_dir)
    if directory.is_file():
        records = read_jsonl(directory)
        base = directory.parent
    else:
        records = []
        for path in sorted(directory.glob("*.jsonl")) if directory.is_dir() else []:
            records.extend(read_jsonl(path))
        base = directory
    summary = read_json(base / SUMMARY_FILENAME, default={}) or {}
    tokens = normalize_tokens(summary.get("tokens"))
    if not any(tokens.values()):
        for record in records:
            payload = record.get("usage") or record.get("tokens") or {}
            merged = normalize_tokens(payload)
            for key, value in merged.items():
                tokens[key] += value
    cost = summary.get("cost_usd")
    if cost is None:
        cost = max([float(r.get("cost", 0.0)) for r in records if isinstance(r, dict)] or [0.0])
    return {
        "messages": records,
        "tool_calls": iter_tool_calls(records),
        "tokens": tokens,
        "cost_usd": float(cost or 0.0),
        "status": summary.get("status", "unknown"),
        "summary": summary,
    }
