"""Evidence bundle construction (paper Sec. 4 / Appendix B).

Walks ``/output`` and the run trace, producing an :class:`EvidenceBundle`
that every downstream scorer consumes:

* ``manifest`` -- one :class:`ArtifactEntry` per file (path, size, sha256,
  magic/mime sniff, kind classification, capped excerpt).
* ``commands`` -- a command timeline parsed from the trace (bash command,
  exit code, stdout tail, curl/wget targets + HTTP status, binwalk, qemu /
  chroot invocation, listening ports, crash signals).
* ``provenance`` -- the firmware -> extraction -> binary -> executed chain.

The provenance chain is what powers real-target gating: it only closes when
a binary that lives inside an extracted rootfs is the very same sha256 that
was actually executed.  Everything here is deterministic and offline; no
network calls are made (``magic`` is optional and we fall back to a small
built-in signature table).

This module is intentionally tolerant: trace files come from several
harness adapters (opencode JSON, OpenAI-compatible tool loop, replay), so
the parsers accept both the canonical trace schema and looser shapes.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..schema import (
    ArtifactEntry,
    CommandEntry,
    EvidenceBundle,
    ProvenanceChain,
)
from ..utils.io import iter_files, read_text, sha256_file, stable_json

__all__ = [
    "ARTIFACT_KINDS",
    "build_bundle",
    "build_evidence_bundle",
    "classify_artifact",
    "excerpt_for",
    "parse_commands",
    "parse_trace",
    "sniff",
    "build_provenance",
]

# ---------------------------------------------------------------------------
# artifact classification
# ---------------------------------------------------------------------------

ARTIFACT_KINDS: Tuple[str, ...] = (
    "firmware_image",
    "archive",
    "rootfs_dir",
    "elf_binary",
    "script",
    "log",
    "pcap",
    "report",
    "plan",
    "poc",
    "other",
)

# magic-byte signatures: (prefix, magic string, mime, kind)
_SIGNATURES: Sequence[Tuple[bytes, str, str, str]] = (
    (b"\x7fELF", "ELF", "application/x-executable", "elf_binary"),
    (b"\x1f\x8b", "gzip compressed data", "application/gzip", "archive"),
    (b"hsqs", "Squashfs filesystem", "application/x-squashfs", "archive"),
    (b"sqsh", "Squashfs filesystem (big endian)", "application/x-squashfs", "archive"),
    (b"UBI#", "UBI erase count header", "application/x-ubi", "archive"),
    (b"UBI\x01", "UBI image", "application/x-ubi", "archive"),
    (b"\xd0\x0d\xfe\xed", "uImage header", "application/x-uimage", "firmware_image"),
    (b"070701", "ASCII cpio archive", "application/x-cpio", "archive"),
    (b"070702", "ASCII cpio archive (crc)", "application/x-cpio", "archive"),
    (b"\x27\x05\x19\x56", "u-boot legacy image", "application/x-uimage", "firmware_image"),
    (b"\x0a\x0d\x0d\x0a", "pcap capture file", "application/vnd.tcpdump.pcap", "pcap"),
    (b"\xa1\xb2\xc3\xd4", "pcap capture file (big endian)", "application/vnd.tcpdump.pcap", "pcap"),
    (b"PK\x03\x04", "Zip archive data", "application/zip", "archive"),
    (b"BZh", "bzip2 compressed data", "application/x-bzip2", "archive"),
    (b"\xfd7zXZ\x00", "XZ compressed data", "application/x-xz", "archive"),
    (b"7z\xbc\xaf\x27\x1c", "7-zip archive data", "application/x-7z-compressed", "archive"),
    (b"\x1f\x8b\x08", "gzip compressed data", "application/gzip", "archive"),
    (b"ustar", "POSIX tar archive", "application/x-tar", "archive"),
)

_TEXT_EXT = {
    ".sh": ("shell script", "text/x-shellscript", "script"),
    ".py": ("Python script", "text/x-python", "script"),
    ".js": ("JavaScript source", "application/javascript", "script"),
    ".rb": ("Ruby script", "text/x-ruby", "script"),
    ".pl": ("Perl script", "text/x-perl", "script"),
    ".c": ("C source", "text/x-c", "script"),
    ".h": ("C header", "text/x-c", "script"),
    ".md": ("ASCII text (markdown)", "text/markdown", "other"),
    ".txt": ("ASCII text", "text/plain", "other"),
    ".json": ("JSON data", "application/json", "other"),
    ".yaml": ("YAML data", "application/x-yaml", "other"),
    ".yml": ("YAML data", "application/x-yaml", "other"),
    ".log": ("ASCII text (log)", "text/plain", "log"),
    ".pcap": ("pcap capture file", "application/vnd.tcpdump.pcap", "pcap"),
    ".cap": ("pcap capture file", "application/vnd.tcpdump.pcap", "pcap"),
    ".pem": ("PEM certificate/key", "application/x-pem-file", "other"),
    ".bin": ("data", "application/octet-stream", "firmware_image"),
    ".img": ("data", "application/octet-stream", "firmware_image"),
    ".trx": ("data", "application/octet-stream", "firmware_image"),
    ".chk": ("data", "application/octet-stream", "firmware_image"),
}

_POC_HINTS = ("poc", "exploit", "payload", "trigger", "repro", "attack", "cve")
_PLAN_HINTS = ("plan",)
_REPORT_HINTS = ("report", "conclusion", "summary", "writeup", "findings")

_MIME_AVAILABLE = shutil.which("file") is not None


def sniff(path: Path, data: Optional[bytes] = None) -> Tuple[str, str, Optional[str]]:
    """Best-effort ``(magic, mime, kind_hint)`` for ``path``.

    Uses the system ``file(1)`` command when available (mirrors what the
    agent container reports), else a built-in signature table.
    """

    if data is None:
        try:
            with open(path, "rb") as fh:
                data = fh.read(4096)
        except OSError:
            return "unreadable", "application/x-empty", None

    if _MIME_AVAILABLE:
        try:  # pragma: no cover - depends on host tooling
            out = subprocess.run(
                ["file", "-b", "--mime-type", str(path)],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            mime = (out.stdout or "").strip() or "application/octet-stream"
            out2 = subprocess.run(
                ["file", "-b", str(path)],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            magic = (out2.stdout or "").strip()
            hint = None
            if "ELF" in magic:
                hint = "elf_binary"
            elif "Squashfs" in magic or "cpio" in magic or "gzip" in magic:
                hint = "archive"
            elif "uImage" in magic or "u-boot" in magic:
                hint = "firmware_image"
            if magic or mime:
                return magic or "data", mime, hint
        except (OSError, ValueError, subprocess.SubprocessError):  # pragma: no cover
            pass

    for prefix, magic, mime, kind in _SIGNATURES:
        if data.startswith(prefix):
            return magic, mime, kind

    head = data[:1024]
    if head and b"\x00" not in head:
        try:
            head.decode("utf-8")
        except UnicodeDecodeError:
            return "data", "application/octet-stream", "other"
        if b"#!/" in head[:64]:
            return "script text executable", "text/x-shellscript", "script"
        return "ASCII text", "text/plain", "other"

    return "data", "application/octet-stream", "other"


def classify_artifact(rel_path: str, magic: str = "", mime: str = "", size: int = 0) -> str:
    """Map an artifact to one of :data:`ARTIFACT_KINDS`."""

    name = rel_path.replace("\\", "/").lower()
    base = name.rsplit("/", 1)[-1]
    ext = os.path.splitext(base)[1]

    # explicit top-level deliverables win over content sniffing
    if base == "plan.md" or base.endswith("/plan.md"):
        return "plan"
    if base == "report.md" or base.endswith("/report.md"):
        return "report"

    if "ELF" in magic or mime in ("application/x-executable", "application/x-sharedlib"):
        return "elf_binary"

    if base.endswith((".pcap", ".cap")) or "pcap" in magic.lower():
        return "pcap"
    if base.endswith(".log") or "/log" in name and "/logs/" in name:
        return "log"
    if "squashfs" in magic.lower() or "cpio" in magic.lower() or mime in (
        "application/gzip",
        "application/zip",
        "application/x-tar",
        "application/x-squashfs",
        "application/x-ubi",
        "application/x-cpio",
    ):
        return "archive"
    if "uImage" in magic or "u-boot" in magic:
        return "firmware_image"

    if ext in _TEXT_EXT:
        _, _, hint = _TEXT_EXT[ext]
        if hint == "other":
            # refine prose deliverables by filename
            if any(h in base for h in _REPORT_HINTS):
                return "report"
            if any(h in base for h in _PLAN_HINTS) and ext in (".md", ".txt"):
                return "plan"
            if any(h in base for h in _POC_HINTS) and ext in (".py", ".sh", ".c", ".js"):
                return "poc"
        return hint

    if any(h in base for h in _POC_HINTS) and ext in (".py", ".sh", ".c", ".js", ".pl"):
        return "poc"
    if any(h in base for h in _REPORT_HINTS) and "text" in mime:
        return "report"
    if any(h in base for h in _PLAN_HINTS) and "text" in mime:
        return "plan"
    if "script" in magic or "text/x-" in mime:
        return "script"
    if size > 512 * 1024 and "octet-stream" in mime:
        return "firmware_image"

    return "other"


def excerpt_for(path: Path, rel_path: str, kind: str, data: Optional[bytes] = None) -> str:
    """Capped excerpt used by the judge (text head/tail, binaries as hexdump)."""

    if data is None:
        try:
            with open(path, "rb") as fh:
                data = fh.read(256 * 1024)
        except OSError:
            return ""

    if not data:
        return ""

    is_binary = kind in ("elf_binary", "firmware_image", "archive") or b"\x00" in data[:4096]
    if is_binary:
        head = data[:256]
        hexdump = " ".join(f"{b:02x}" for b in head)
        magic, mime, _ = sniff(path, data)
        ascii_view = "".join(chr(b) if 32 <= b < 127 else "." for b in head)
        return (
            f"[binary] magic={magic!r} mime={mime} size={len(data)}+ "
            f"head256={hexdump} ascii={ascii_view}"
        )

    text = data.decode("utf-8", errors="replace")
    head, tail = 8192, 4096
    if len(text) <= head + tail:
        return text
    return text[:head] + "\n...[truncated]...\n" + text[-tail:]


# ---------------------------------------------------------------------------
# trace parsing
# ---------------------------------------------------------------------------


def _json_load(text: str) -> Any:
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def _iter_json_candidates(trace_path: Path) -> Iterable[Any]:
    """Yield every trace object found at ``trace_path`` (file or directory)."""

    if trace_path.is_dir():
        names = ("trace.json", "trace.jsonl", "messages.json", "events.jsonl")
        for name in names:
            candidate = trace_path / name
            if candidate.exists():
                yield from _iter_json_candidates(candidate)
        for child in sorted(trace_path.glob("*.json")):
            if child.name not in names:
                yield from _iter_json_candidates(child)
        return

    raw = read_text(trace_path, default="")
    if not raw.strip():
        return

    obj = _json_load(raw)
    if obj is not None:
        yield obj
        return

    # JSONL / streamed JSON objects
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        item = _json_load(line)
        if item is not None:
            yield item


def _extract_messages(obj: Any) -> List[Mapping[str, Any]]:
    """Pull a flat message list out of any supported trace shape."""

    out: List[Mapping[str, Any]] = []
    if isinstance(obj, list):
        for item in obj:
            out.extend(_extract_messages(item))
        return out
    if not isinstance(obj, Mapping):
        return out

    for key in ("messages", "conversation", "history", "steps"):
        value = obj.get(key)
        if isinstance(value, list):
            out.extend([m for m in value if isinstance(m, Mapping)])

    if isinstance(obj.get("events"), list):
        # opencode --format json emits ``{"type": ..., "part": {...}}``
        for event in obj["events"]:
            if isinstance(event, Mapping):
                parts = event.get("messages")
                if isinstance(parts, list):
                    out.extend([m for m in parts if isinstance(m, Mapping)])
                else:
                    out.append(event)

    if obj.get("type") == "message" or "role" in obj or "tool_calls" in obj:
        out.append(obj)

    # nested ``data``/``result`` wrappers
    for key in ("data", "result", "output", "trace"):
        if isinstance(obj.get(key), (Mapping, list)):
            inner = _extract_messages(obj[key])
            if inner:
                out.extend(inner)

    return out


def _tool_calls(message: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    calls: List[Mapping[str, Any]] = []
    raw = message.get("tool_calls")
    if isinstance(raw, list):
        calls.extend([c for c in raw if isinstance(c, Mapping)])
    if isinstance(message.get("tool_call"), Mapping):
        calls.append(message["tool_call"])
    # opencode part shape: {"type": "tool", "tool": "bash", "state": {...}}
    if message.get("type") in ("tool", "tool_use", "tool_result") and (
        "tool" in message or "tool_name" in message
    ):
        state = message.get("state") if isinstance(message.get("state"), Mapping) else {}
        calls.append(
            {
                "name": message.get("tool") or message.get("tool_name"),
                "input": state.get("input", message.get("input")),
                "output": state.get("output", message.get("output")),
            }
        )
    return calls


def _call_name(call: Mapping[str, Any]) -> str:
    fn = call.get("function")
    if isinstance(fn, Mapping):
        return str(fn.get("name") or call.get("name") or "bash")
    return str(call.get("name") or call.get("tool") or "bash")


def _call_input(call: Mapping[str, Any]) -> Dict[str, Any]:
    fn = call.get("function")
    raw = fn.get("arguments") if isinstance(fn, Mapping) else call.get("input")
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        parsed = _json_load(raw)
        if isinstance(parsed, Mapping):
            return dict(parsed)
        return {"command": raw}
    return {}


def _call_output(call: Mapping[str, Any]) -> str:
    out = call.get("output")
    if out is None and isinstance(call.get("function"), Mapping):
        out = call["function"].get("output")
    if isinstance(out, Mapping):
        for key in ("stdout", "output", "text", "content", "stdout_tail"):
            if key in out:
                return str(out[key])
        return json.dumps(out, default=str)
    return str(out) if out is not None else ""


_HTTP_TARGET_RE = re.compile(
    r"(?P<url>(?:https?|ftp)://[^\s'\"<>|;]+|\b\d{1,3}(?:\.\d{1,3}){3}:\d+(?:/[^\s'\"]*)?)"
)
_CURL_STATUS_RE = re.compile(r"HTTP/[\d.]+\s+(?P<code>\d{3})")
_EXIT_RE = re.compile(r"(?:exit(?:ed)?\s*(?:code|status)|returncode)\s*[:=]?\s*(?P<code>-?\d+)")
_PORT_RE = re.compile(r"(?:0\.0\.0\.0|127\.0\.0\.1|\[::\]|\*|::):(?P<port>\d{2,5})")
_SIGNAL_RE = re.compile(
    r"\b(SIG(?:SEGV|ABRT|ILL|BUS|FPE|KILL|TERM|PIPE|TRAP))\b|\b(segmentation fault|core dumped|stack smashing|double free|use-after-free)\b",
    re.IGNORECASE,
)
_TIME_RE = re.compile(r"(?:real|elapsed)\s+(\d+)m([\d.]+)s")
_TIMEOUT_RE = re.compile(r"\b(?:timeout|timed out)\b", re.IGNORECASE)


def _looks_like_bash(name: str, payload: Mapping[str, Any]) -> bool:
    if name.lower() in ("bash", "shell", "run", "exec", "terminal", "sh", "command"):
        return True
    return "command" in payload or "cmd" in payload


def parse_commands(trace_path: Path) -> List[CommandEntry]:
    """Parse a trace into an ordered command timeline."""

    entries: List[CommandEntry] = []
    index = 0
    for obj in _iter_json_candidates(trace_path):
        for message in _extract_messages(obj):
            for call in _tool_calls(message):
                name = _call_name(call)
                payload = _call_input(call)
                text = _call_output(call)
                if not _looks_like_bash(name, payload):
                    # non-shell tools still matter (read/write/glob/grep/webfetch)
                    command = str(
                        payload.get("path")
                        or payload.get("pattern")
                        or payload.get("url")
                        or payload.get("query")
                        or ""
                    )
                    if not command:
                        continue
                    entries.append(
                        CommandEntry(
                            index=index,
                            tool=name,
                            command=f"{name} {command}".strip(),
                            exit_code=None,
                            stdout_tail=text[-2000:],
                        )
                    )
                    index += 1
                    continue

                command = str(payload.get("command") or payload.get("cmd") or "")
                if not command:
                    continue

                statuses = [int(m.group("code")) for m in _CURL_STATUS_RE.finditer(text)]
                targets = [m.group("url") for m in _HTTP_TARGET_RE.finditer(command)]
                signals = [m.group(0) for m in _SIGNAL_RE.finditer(text)]
                ports = [int(m.group("port")) for m in _PORT_RE.finditer(text)]

                exit_code: Optional[int] = None
                explicit = payload.get("exit_code", payload.get("exitCode"))
                if isinstance(explicit, int):
                    exit_code = explicit
                else:
                    match = _EXIT_RE.search(text)
                    if match:
                        try:
                            exit_code = int(match.group("code"))
                        except ValueError:
                            exit_code = None

                entries.append(
                    CommandEntry(
                        index=index,
                        tool=name,
                        command=command,
                        exit_code=exit_code,
                        stdout_tail=text[-2000:],
                        network_targets=targets,
                        http_status=statuses,
                        crash_signal=signals[0] if signals else None,
                        listening_ports=_dedupe_ints(ports)[:8],
                    )
                )
                index += 1
    return entries


def _dedupe_ints(values: Iterable[int]) -> List[int]:
    seen: List[int] = []
    for v in values:
        if v not in seen:
            seen.append(v)
    return seen


def parse_trace(trace_path: Path) -> Dict[str, Any]:
    """Return ``{"commands": [...], "tokens": {...}, "cost": float}``."""

    commands = parse_commands(trace_path)
    tokens: Dict[str, int] = {}
    cost = 0.0
    for obj in _iter_json_candidates(trace_path):
        if not isinstance(obj, Mapping):
            continue
        for key in ("tokens", "usage", "token_usage"):
            usage = obj.get(key)
            if isinstance(usage, Mapping):
                tokens = _merge_tokens(tokens, usage)
        if isinstance(obj.get("cost"), (int, float)):
            cost = max(cost, float(obj["cost"]))
        if isinstance(obj.get("cost_usd"), (int, float)):
            cost = max(cost, float(obj["cost_usd"]))
    return {"commands": commands, "tokens": tokens, "cost": cost}


def _merge_tokens(acc: Dict[str, int], usage: Mapping[str, Any]) -> Dict[str, int]:
    for key in ("input", "output", "reasoning", "cache_read", "cache_write"):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            acc[key] = acc.get(key, 0) + int(value)
        else:
            nested = usage.get(key)
            if isinstance(nested, Mapping):
                inner = nested.get("read") if key == "cache" else None
                if isinstance(inner, (int, float)):
                    acc[key] = acc.get(key, 0) + int(inner)
    cache = usage.get("cache")
    if isinstance(cache, Mapping):
        for sub, dest in (("read", "cache_read"), ("write", "cache_write")):
            if isinstance(cache.get(sub), (int, float)):
                acc[dest] = acc.get(dest, 0) + int(cache[sub])
    return acc


# ---------------------------------------------------------------------------
# provenance chain
# ---------------------------------------------------------------------------


def _looks_like_extraction(command: str) -> bool:
    low = command.lower()
    return any(
        token in low
        for token in (
            "binwalk",
            "unsquashfs",
            "sasquatch",
            "jefferson",
            "ubireader",
            "ubireader_extract_images",
            "uextract",
            "cpio",
            "tar ",
            "unzip",
            "firmware-mod-kit",
            "extract-firmware",
        )
    )


def _executed_binaries(commands: Sequence[CommandEntry]) -> List[str]:
    out: List[str] = []
    for cmd in commands:
        text = cmd.command
        for match in re.finditer(r"(?:^|[\s;|&`(])((?:\.{0,2}/|/)[\w./\-]+)", text):
            token = match.group(1)
            if token.startswith("/dev/") or token in ("/bin/sh", "/bin/bash"):
                continue
            out.append(token)
    return _dedupe_str(out)


def _dedupe_str(values: Iterable[str]) -> List[str]:
    seen: List[str] = []
    for v in values:
        if v not in seen:
            seen.append(v)
    return seen


def build_provenance(
    artifacts: Sequence[ArtifactEntry],
    commands: Sequence[CommandEntry],
    output_dir: Optional[Path] = None,
) -> ProvenanceChain:
    """Derive the firmware -> extraction -> binary -> executed chain."""

    chain = ProvenanceChain()

    firmware = [a for a in artifacts if a.kind == "firmware_image"]
    if firmware:
        # prefer the biggest firmware-looking artifact (the download)
        firmware_sorted = sorted(firmware, key=lambda a: a.size, reverse=True)
        chain.firmware_sha256 = firmware_sorted[0].sha256 or None

    extraction_idx: Optional[int] = None
    for cmd in commands:
        if _looks_like_extraction(cmd.command):
            extraction_idx = cmd.index
            break
    chain.extraction_command_index = extraction_idx

    binaries = [a for a in artifacts if a.kind == "elf_binary"]
    chosen: Optional[ArtifactEntry] = None
    for cmd in commands:
        low = cmd.command.lower()
        if not any(t in low for t in ("qemu", "chroot", "./")):
            continue
        for token in _executed_binaries([cmd]):
            token_norm = token.lstrip("./")
            for art in binaries:
                rel = art.rel_path.lstrip("./")
                if rel == token_norm or rel.endswith("/" + token_norm) or token_norm.endswith("/" + rel):
                    chosen = art
                    break
            if chosen:
                break
        if chosen:
            break
    if chosen is None and binaries:
        # fall back to the largest binary inside an extraction directory
        inside = [a for a in binaries if "/" in a.rel_path]
        chosen = max(inside or binaries, key=lambda a: a.size)

    if chosen is not None:
        chain.binary_sha256 = chosen.sha256 or None
        chain.binary_path = chosen.rel_path
        chain.executed_sha256 = [chosen.sha256] if chosen.sha256 else []
        chain.executed_paths = [chosen.rel_path]

    return chain


# ---------------------------------------------------------------------------
# bundle assembly
# ---------------------------------------------------------------------------


def _read_rel(output_dir: Path, rel: str) -> Optional[Path]:
    candidate = output_dir / rel
    return candidate if candidate.exists() else None


def build_bundle(
    run_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    trace_path: Optional[str | Path] = None,
    run_id: Optional[str] = None,
    cve_id: Optional[str] = None,
    model: Optional[str] = None,
    trial: int = 1,
    extra_meta: Optional[Mapping[str, Any]] = None,
) -> EvidenceBundle:
    """Build an :class:`EvidenceBundle` for one run directory.

    ``output_dir``/``trace_path`` default to ``<run_dir>/output`` and
    ``<run_dir>/trace``.  Missing directories degrade to empty bundles so
    scoring can run on partial/aborted runs.
    """

    run_path = Path(run_dir)
    out_path = Path(output_dir) if output_dir else run_path / "output"
    trace = Path(trace_path) if trace_path else run_path / "trace"

    meta: Dict[str, Any] = {}
    meta_file = run_path / "meta.json"
    if meta_file.exists():
        parsed = _json_load(read_text(meta_file, default=""))
        if isinstance(parsed, Mapping):
            meta = dict(parsed)
    if extra_meta:
        meta.update(dict(extra_meta))

    run_id = run_id or str(meta.get("run_id") or run_path.name)
    cve_id = cve_id or str(meta.get("cve_id") or meta.get("cve") or "")
    model = model or str(meta.get("model") or "")
    trial = int(meta.get("trial", trial) or trial)

    artifacts: List[ArtifactEntry] = []
    if out_path.exists():
        for path in iter_files(out_path):
            try:
                rel = str(path.relative_to(out_path))
            except ValueError:
                rel = path.name
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            digest = ""
            try:
                digest = sha256_file(path)
            except OSError:
                digest = ""
            magic, mime, hint = sniff(path)
            kind = classify_artifact(rel, magic, mime, size)
            artifacts.append(
                ArtifactEntry(
                    path=str(path),
                    rel_path=rel,
                    size=size,
                    sha256=digest,
                    magic=magic,
                    mime=mime,
                    kind=kind,
                )
            )

    commands: List[CommandEntry] = []
    trace_tokens: Dict[str, int] = {}
    trace_cost = 0.0
    if trace.exists():
        parsed = parse_trace(trace)
        commands = list(parsed["commands"])
        trace_tokens = dict(parsed["tokens"])
        trace_cost = float(parsed["cost"])

    provenance = build_provenance(artifacts, commands, out_path)

    def _artifact_text(rel: str) -> str:
        path = _read_rel(out_path, rel)
        if path is None:
            return ""
        return read_text(path, default="")

    plan_text = _artifact_text("plan.md")
    report_text = _artifact_text("report.md")

    # also accept nested deliverables, e.g. ``/output/notes/plan.md``
    if not plan_text:
        for art in artifacts:
            if art.kind == "plan":
                plan_text = _artifact_text(art.rel_path)
                if plan_text:
                    break
    if not report_text:
        for art in artifacts:
            if art.kind == "report":
                report_text = _artifact_text(art.rel_path)
                if report_text:
                    break

    manifest_hash = stable_json(
        sorted((a.rel_path, a.sha256, a.size) for a in artifacts)
    )
    from ..utils.io import sha256_text  # local import keeps module import cheap

    manifest_hash = sha256_text(manifest_hash)

    return EvidenceBundle(
        run_id=run_id,
        cve_id=cve_id,
        model=model,
        trial=trial,
        artifacts=artifacts,
        commands=commands,
        provenance=provenance,
        manifest_hash=manifest_hash,
        trace_tokens=trace_tokens,
        trace_cost_usd=trace_cost,
        has_plan=bool(plan_text.strip()),
        has_report=bool(report_text.strip()),
        infra_failure=bool(meta.get("infrastructure_failure", False)),
        notes=[str(n) for n in meta.get("notes", []) or []],
    )


# Convenience alias: some callers/tests use the paper's wording.
build_evidence_bundle = build_bundle
