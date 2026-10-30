"""Citation existence checking for ReproBench (paper Sec. 4, rubric grounding rule).

Every non-zero credit in the Paper must be backed by a *path that exists*.  The
paper states: "Any non-zero credit without an existing citation path is
invalidated and re-judged, then zeroed."  This module turns an
:class:`~reprobench.schema.EvidenceBundle` into a predicate
(``citation_exists``) that the rubric / plan / attempt scorers can inject, so
those modules stay pure and unit-testable without touching the filesystem.

Citation kinds (``Citation.kind``)::

    artifact_path     -> a file/dir that exists under the run's /output tree
    command_index     -> an index into the parsed command timeline
    plan_line         -> a line number (1-based) in plan.md
    report_line       -> a line number (1-based) in report.md
    provenance_node   -> one of firmware_sha256 / extraction / binary_sha256 /
                         executed_sha256 in the provenance chain

Citations arrive in three shapes:

* :class:`~reprobench.schema.Citation` objects (``kind`` + ``ref``),
* plain strings, optionally prefixed::

      "artifact_path:/output/extract/usr/sbin/httpd"
      "/output/report.md"              # bare path -> artifact_path
      "command_index:12"
      "plan.md:44"                     # -> plan_line
      "report.md:12"                   # -> report_line
      "provenance_node:binary_sha256"

  Anything un-prefixed that looks like a path is treated as ``artifact_path``.

The checker is deliberately lenient about *format* and strict about
*existence*: a citation path is accepted when the referenced receiver exists
(a file on disk, a command index in range, a line inside the text, a provenance
node that is populated).  Malformed citations simply fail closed (return
``False``).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from ..schema import Citation, EvidenceBundle

__all__ = [
    "CITATION_KINDS",
    "CitationIndex",
    "citation_exists_factory",
    "citation_key",
    "citation_ref",
    "is_valid_citation",
    "normalize_citation",
    "provenance_nodes",
    "resolve_citation",
]


CITATION_KINDS: tuple = (
    "artifact_path",
    "command_index",
    "plan_line",
    "report_line",
    "provenance_node",
)

PROVENANCE_NODES: tuple = (
    "firmware_sha256",
    "extraction",
    "binary_sha256",
    "executed_sha256",
)

_PREFIX_RE = re.compile(r"^(artifact_path|command_index|plan_line|report_line|provenance_node)\s*:\s*(.+)$", re.I)
_PLAN_RE = re.compile(r"^(?:.*/)?plan\.md\s*:\s*L?(\d+)", re.I)
_REPORT_RE = re.compile(r"^(?:.*/)?report\.md\s*:\s*L?(\d+)", re.I)
_LINE_SUFFIX_RE = re.compile(r"^(.*?)(?::L?(\d+))$")


# ---------------------------------------------------------------------------
# normalisation helpers
# ---------------------------------------------------------------------------
def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    for attr in ("ref", "path", "value"):
        got = getattr(value, attr, None)
        if isinstance(got, str) and got.strip():
            return got.strip()
    return str(value).strip()


def normalize_citation(citation: Any) -> Citation:
    """Coerce ``citation`` into a :class:`Citation` (best effort, never raises)."""
    if isinstance(citation, Citation):
        return citation
    if isinstance(citation, Mapping):
        kind = str(citation.get("kind") or "artifact_path").strip().lower()
        ref = _as_text(citation.get("ref") or citation.get("path") or citation.get("value"))
        quote = citation.get("quote")
        if kind not in CITATION_KINDS:
            kind = "artifact_path"
        return Citation(kind=kind, ref=ref, quote=quote)  # type: ignore[arg-type]

    text = _as_text(citation)
    if not text:
        return Citation(kind="artifact_path", ref="")

    match = _PLAN_RE.match(text)
    if match:
        return Citation(kind="plan_line", ref=match.group(1))
    match = _REPORT_RE.match(text)
    if match:
        return Citation(kind="report_line", ref=match.group(1))
    match = _PREFIX_RE.match(text)
    if match:
        kind = match.group(1).lower()
        return Citation(kind=kind, ref=match.group(2).strip())  # type: ignore[arg-type]
    return Citation(kind="artifact_path", ref=text)


def citation_key(citation: Any) -> str:
    """Return the stable ``"<kind>:<ref>"`` key for a citation."""
    cit = normalize_citation(citation)
    return f"{cit.kind}:{cit.ref}"


def citation_ref(citation: Any) -> str:
    return normalize_citation(citation).ref


def is_valid_citation(citation: Any) -> bool:
    """Structural validity (kind known and ref non-empty).  Not existence."""
    try:
        cit = normalize_citation(citation)
    except Exception:  # pragma: no cover - defensive
        return False
    return bool(cit.ref) and cit.kind in CITATION_KINDS


def provenance_nodes(bundle: Any) -> Dict[str, bool]:
    """Return the populated-state of the four provenance chain nodes."""
    prov = _get(bundle, "provenance", None)
    nodes: Dict[str, bool] = {}
    fw = _get(prov, "firmware_sha256", None)
    nodes["firmware_sha256"] = bool(fw)
    nodes["extraction"] = bool(
        _get(prov, "extraction_command_index", None) is not None
        or _get(prov, "extraction", None)
        or _get(prov, "binary_path", None)
    )
    nodes["binary_sha256"] = bool(_get(prov, "binary_sha256", None))
    executed = _get(prov, "executed_sha256", None) or []
    if isinstance(executed, str):
        executed = [executed]
    nodes["executed_sha256"] = bool([e for e in executed if e])
    return nodes


# ---------------------------------------------------------------------------
# tolerant accessors (pydantic models or plain dicts)
# ---------------------------------------------------------------------------
def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _artifacts(bundle: Any) -> List[Any]:
    for name in ("artifacts", "manifest"):
        value = _get(bundle, name, None)
        if value:
            return list(value)
    return []


def _commands(bundle: Any) -> List[Any]:
    value = _get(bundle, "commands", None)
    return list(value) if value else []


def _lines(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _strip_number(ref: str) -> str:
    ref = (ref or "").strip()
    ref = ref.lstrip("L#")
    match = re.match(r"^(\d+)", ref)
    return match.group(1) if match else ref


# ---------------------------------------------------------------------------
# citation index
# ---------------------------------------------------------------------------
@dataclass
class CitationIndex:
    """Existence index over one run's evidence bundle.

    Construct with a bundle, or with raw pieces (``paths`` / ``n_lines`` /
    ``n_commands`` / ``provenance``) when a full bundle is not available (unit
    tests, fixtures).
    """

    paths: List[str] = field(default_factory=list)
    n_commands: int = 0
    plan_lines: int = 0
    report_lines: int = 0
    provenance: Dict[str, bool] = field(default_factory=dict)
    output_dir: str = ""

    # -- construction ------------------------------------------------------
    @classmethod
    def from_bundle(cls, bundle: Any) -> "CitationIndex":
        artifact_paths: List[str] = []
        for art in _artifacts(bundle):
            for key in ("path", "rel_path"):
                value = _get(art, key, None)
                if isinstance(value, str) and value:
                    artifact_paths.append(value)
        index = cls(
            paths=artifact_paths,
            n_commands=len(_commands(bundle)),
            plan_lines=_lines(_get(bundle, "plan_text", "") or ""),
            report_lines=_lines(_get(bundle, "report_text", "") or ""),
            provenance=provenance_nodes(bundle),
            output_dir=str(_get(bundle, "output_dir", "") or ""),
        )
        return index

    # -- queries -----------------------------------------------------------
    def _path_exists(self, ref: str) -> bool:
        if not ref:
            return False
        candidates = {ref.strip(), ref.strip().lstrip("./")}

        # relative to the run output directory
        if self.output_dir:
            base = Path(self.output_dir)
            candidates.add(str(base / ref.strip().lstrip("/")))
            candidates.add(str(base / ref.strip().lstrip("./")))

        for path in self.paths:
            norm = path.strip()
            candidates.add(norm)
            # accept either side being the tail of the other
            if norm.endswith(ref.strip()) or ref.strip().endswith(norm):
                return True

        for cand in candidates:
            try:
                if Path(cand).exists():
                    return True
            except OSError:  # pragma: no cover - defensive
                continue
        return False

    def exists_artifact(self, ref: str) -> bool:
        ref = (ref or "").strip()
        if not ref:
            return False
        # allow "path:LINE" citations pointing at an artifact line
        match = _LINE_SUFFIX_RE.match(ref)
        if match and match.group(1) and not Path(match.group(1)).suffix == "":
            if self._path_exists(match.group(1)):
                return True
        return self._path_exists(ref)

    def exists_command(self, ref: str) -> bool:
        value = _strip_number(ref)
        try:
            idx = int(value)
        except (TypeError, ValueError):
            return False
        return 0 <= idx < max(self.n_commands, 0) or (self.n_commands == 0 and idx == 0 and False)

    def exists_plan_line(self, ref: str) -> bool:
        try:
            line = int(_strip_number(ref))
        except (TypeError, ValueError):
            return False
        return 1 <= line <= max(self.plan_lines, 0)

    def exists_report_line(self, ref: str) -> bool:
        try:
            line = int(_strip_number(ref))
        except (TypeError, ValueError):
            return False
        return 1 <= line <= max(self.report_lines, 0)

    def exists_provenance(self, ref: str) -> bool:
        key = (ref or "").strip().lower()
        if key not in PROVENANCE_NODES:
            return False
        return bool(self.provenance.get(key, False))

    # -- unified -----------------------------------------------------------
    def exists(self, citation: Any) -> bool:
        """True when the cited receiver actually exists."""
        try:
            cit = normalize_citation(citation)
        except Exception:  # pragma: no cover - defensive
            return False
        if not cit.ref:
            return False
        kind = cit.kind
        if kind == "artifact_path":
            return self.exists_artifact(cit.ref)
        if kind == "command_index":
            return self.exists_command(cit.ref)
        if kind == "plan_line":
            # "44" but also "plan.md" style refs with no line number
            if cit.ref.isdigit():
                return self.exists_plan_line(cit.ref)
            return self.exists_artifact(cit.ref)
        if kind == "report_line":
            if cit.ref.isdigit():
                return self.exists_report_line(cit.ref)
            return self.exists_artifact(cit.ref)
        if kind == "provenance_node":
            return self.exists_provenance(cit.ref)
        return False

    def any_exists(self, citations: Optional[Iterable[Any]]) -> bool:
        for citation in citations or ():
            if self.exists(citation):
                return True
        return False


def resolve_citation(citation: Any, index: Optional[CitationIndex] = None) -> bool:
    """Resolve one citation against ``index`` (``None`` -> structural check only)."""
    if index is None:
        return is_valid_citation(citation)
    return index.exists(citation)


def citation_exists_factory(bundle: Any = None, index: Optional[CitationIndex] = None) -> Callable[[Any], bool]:
    """Build the ``citation_exists`` predicate injected into the scorers.

    ``bundle`` may be an :class:`EvidenceBundle`, a mapping, or ``None``.  When
    no bundle/index is supplied every structurally valid citation is accepted
    (used by the offline heuristic path and pure-arithmetic unit tests).
    """

    if index is None and bundle is not None:
        try:
            index = CitationIndex.from_bundle(bundle)
        except Exception:  # pragma: no cover - defensive
            index = None

    if index is None:

        def _permissive(citation: Any) -> bool:
            return is_valid_citation(citation)

        return _permissive

    def _check(citation: Any) -> bool:
        try:
            return index.exists(citation)
        except Exception:  # pragma: no cover - defensive
            return False

    return _check


def check_all(citations: Sequence[Any], include: Sequence[str] = CITATION_KINDS) -> Dict[str, Any]:
    """Diagnostic helper: per-existence report for a citation list."""
    index = CitationIndex()
    detail = []
    for citation in citations or ():
        cit = normalize_citation(citation)
        detail.append(
            {
                "key": citation_key(cit),
                "kind": cit.kind,
                "ref": cit.ref,
                "structurally_valid": is_valid_citation(cit),
                "checked": cit.kind in include,
                "exists": index.exists(cit),
            }
        )
    return {"citations": detail, "keys": [d["key"] for d in detail]}
