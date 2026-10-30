"""HTML -> text cleaning and anchored chunking for reference profiles.

This module implements step 2 of the reference-profile skill:

``fetch`` -> **clean** -> ``distill`` -> ``verify``

Responsibilities (paper Appendix / plan section (3)):

* readability-style HTML extraction (stdlib ``html.parser`` based, no network);
* boilerplate stripping (``<script>``, ``<style>``, ``<nav>``, ``<footer>``, ...);
* **preservation of code fences** (``<pre>`` / ``<code>`` blocks) so that
  endpoints, payloads and shell commands survive distillation;
* chunking into <= 32 KB pieces, each carrying a *source anchor*
  (``url#chunk=N``) so distilled facts can cite their origin.

Everything is deterministic and offline: the same fetched text always yields
the same chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List, Optional

__all__ = [
    "CHUNK_SIZE",
    "Chunk",
    "chunk_document",
    "chunk_text",
    "clean_document",
    "html_to_text",
]

#: Maximum chunk size in characters (plan: "chunk <= 32KB with source anchors").
CHUNK_SIZE = 32 * 1024

#: Overlap between consecutive chunks so facts spanning a boundary survive.
CHUNK_OVERLAP = 512

#: Tags whose *content* is dropped entirely.
_DROP_CONTENT_TAGS = {
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "canvas",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    "iframe",
    "object",
    "embed",
}

#: Tags that imply a block break in the rendered text.
_BLOCK_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "main",
    "table",
    "tr",
    "ul",
    "ol",
    "li",
    "dl",
    "dt",
    "dd",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "blockquote",
    "pre",
    "br",
    "hr",
}

#: Tags that mark inline code (kept verbatim, inside a fence).
_CODE_TAGS = {"code", "pre", "kbd", "samp"}

#: Boilerplate lines commonly found on vendor advisories.
_BOILERPLATE_PATTERNS = (
    re.compile(r"^\s*(cookie|accept all|cookies\b)", re.IGNORECASE),
    re.compile(r"^\s*(skip to (main )?content|jump to navigation)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(share|tweet|print|email) (this|on)\b", re.IGNORECASE),
    re.compile(r"^\s*(subscribe|sign in|log in|register)\s*$", re.IGNORECASE),
    re.compile(r"^\s*all rights reserved\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(terms of (use|service)|privacy policy)\s*$", re.IGNORECASE),
)


# --------------------------------------------------------------------------- #
# HTML -> text
# --------------------------------------------------------------------------- #
class _ReadabilityParser(HTMLParser):
    """Very small readability-style extractor built on the stdlib parser.

    It keeps textual content of "content" tags, drops boilerplate containers,
    and turns ``<pre>``/``<code>`` content into fenced code blocks so that
    command lines / payloads are preserved verbatim.
    """

    def __init__(self, base_url: str = "") -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.parts: List[str] = []
        self._skip_depth = 0
        self._code_depth = 0
        self._code_buffer: List[str] = []
        self._title: List[str] = []
        self._in_title = False
        self.anchors: List[Dict[str, str]] = []
        self._current_link: Optional[Dict[str, str]] = None
        self._link_text: List[str] = []

    # -- helpers --------------------------------------------------------- #
    def _flush_code(self) -> None:
        code = "".join(self._code_buffer)
        self._code_buffer = []
        code = code.strip("\n")
        if not code.strip():
            return
        # Normalise indentation but keep everything else verbatim.
        lang = ""
        lines = code.splitlines()
        self.parts.append("\n```%s\n%s\n```\n" % (lang, "\n".join(lines)))

    def _block_break(self) -> None:
        if self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\n")

    # -- HTMLParser hooks ------------------------------------------------ #
    def handle_starttag(self, tag: str, attrs: List[Any]) -> None:  # noqa: D401
        tag = tag.lower()
        attr_map = {str(k).lower(): (v or "") for k, v in attrs}

        if tag in _DROP_CONTENT_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return

        if tag == "title":
            self._in_title = True
            return
        if tag in _CODE_TAGS:
            if self._code_depth == 0:
                self._block_break()
                self._code_buffer = []
            self._code_depth += 1
            return
        if tag == "a":
            href = attr_map.get("href", "")
            self._current_link = {"href": href}
            self._link_text = []
            return
        if tag in _BLOCK_TAGS:
            self._block_break()

    def handle_startendtag(self, tag: str, attrs: List[Any]) -> None:  # noqa: D401
        # ``<br/>`` etc. -- treat as a break, never as a skip boundary.
        if tag.lower() in _BLOCK_TAGS and not self._skip_depth:
            self._block_break()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _DROP_CONTENT_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return

        if tag == "title":
            self._in_title = False
            return
        if tag in _CODE_TAGS:
            self._code_depth = max(0, self._code_depth - 1)
            if self._code_depth == 0:
                self._flush_code()
                self._block_break()
            return
        if tag == "a" and self._current_link is not None:
            text = _collapse("".join(self._link_text)).strip()
            href = self._current_link.get("href", "")
            if text and href:
                self.anchors.append({"text": text, "href": href})
            self._current_link = None
            self._link_text = []
            return
        if tag in _BLOCK_TAGS:
            self._block_break()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self._title.append(data)
            return
        if self._code_depth:
            self._code_buffer.append(data)
            return
        if self._current_link is not None:
            self._link_text.append(data)
        self.parts.append(data)

    # -- result ---------------------------------------------------------- #
    def text(self) -> str:
        if self._code_depth:
            self._flush_code()
        return "".join(self.parts)


def _collapse(text: str) -> str:
    """Collapse runs of spaces/tabs while preserving newlines."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t\f\v]+", " ", line).rstrip() for line in text.split("\n")]
    return "\n".join(lines)


def _strip_boilerplate_lines(text: str) -> str:
    kept: List[str] = []
    in_fence = False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            kept.append(line)
            continue
        if not in_fence and any(p.match(line) for p in _BOILERPLATE_PATTERNS):
            continue
        kept.append(line)
    return "\n".join(kept)


def _squeeze_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text)


def html_to_text(html: str, base_url: str = "") -> str:
    """Convert HTML to readable plain text (deterministic, stdlib only).

    ``base_url`` is only used for diagnostics/anchors; relative links are left
    as-is because the distiller reads the raw text of the advisory.
    """

    if not html:
        return ""
    if "<" not in html:
        # Already plain text (e.g. a raw JSON/API document).
        return _squeeze_blank_lines(_collapse(unescape(html))).strip() + "\n"

    parser = _ReadabilityParser(base_url=base_url)
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # pragma: no cover - malformed markup tolerance
        pass
    text = unescape(parser.text())
    text = _collapse(text)
    text = _strip_boilerplate_lines(text)
    text = _squeeze_blank_lines(text)
    return text.strip() + "\n"


def clean_document(text: str, *, url: str = "", content_type: str = "") -> str:
    """Normalise a fetched document into readable text.

    HTML content types are converted with :func:`html_to_text`; anything else
    is treated as plain text (whitespace normalised, boilerplate stripped).
    """

    if not text:
        return ""
    ct = (content_type or "").lower()
    looks_html = "<html" in text[:2048].lower() or "<!doctype" in text[:512].lower()
    if "html" in ct or looks_html:
        return html_to_text(text, base_url=url)
    return _squeeze_blank_lines(_strip_boilerplate_lines(_collapse(text))).strip() + "\n"


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    """One anchored text chunk ready for map-reduce distillation."""

    text: str
    index: int = 0
    anchor: str = ""
    source_url: str = ""
    source_kind: str = ""
    source_id: str = ""
    start: int = 0
    end: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.text)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "index": self.index,
            "anchor": self.anchor,
            "source_url": self.source_url,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "start": self.start,
            "end": self.end,
            "meta": dict(self.meta),
        }

    def __len__(self) -> int:  # pragma: no cover - convenience
        return len(self.text)


def _preferred_break(text: str, start: int, hard_end: int) -> int:
    """Find a natural break point near ``hard_end`` (never past it)."""

    if hard_end >= len(text):
        return len(text)
    window_start = max(start + int((hard_end - start) * 0.7), start + 1)
    window = text[window_start:hard_end]

    # Prefer a blank line, then a newline, then a sentence end.
    for needle in ("\n\n", "\n", ". "):
        idx = window.rfind(needle)
        if idx != -1:
            return window_start + idx + len(needle)
    return hard_end


def chunk_text(
    text: str,
    *,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    source_url: str = "",
    source_kind: str = "",
    source_id: str = "",
    anchor_prefix: str = "",
) -> List[Chunk]:
    """Split ``text`` into <= ``size`` character chunks with source anchors.

    Anchors have the form ``<url>#chunk=N`` (or ``anchor_prefix#chunk=N`` when
    supplied). When the text already fits in one chunk a single anchor is
    returned so citations stay stable.
    """

    if not text:
        return []
    size = max(256, int(size))
    overlap = max(0, min(int(overlap), size // 2))
    prefix = anchor_prefix or source_url or (source_id or "source")

    chunks: List[Chunk] = []
    start = 0
    index = 0
    total = len(text)
    while start < total:
        hard_end = min(start + size, total)
        end = _preferred_break(text, start, hard_end) if hard_end < total else total
        if end <= start:  # pathological input, guarantee progress
            end = hard_end
        piece = text[start:end]
        chunks.append(
            Chunk(
                text=piece,
                index=index,
                anchor="%s#chunk=%d" % (prefix, index),
                source_url=source_url,
                source_kind=source_kind,
                source_id=source_id,
                start=start,
                end=end,
            )
        )
        index += 1
        if end >= total:
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_document(document: Any, **kwargs: Any) -> List[Chunk]:
    """Chunk a fetched document.

    ``document`` may be a :class:`reprobench.profile.fetch.FetchedSource`, a
    mapping (``text``/``url``/``kind``/``id``) or a plain string. Keyword
    overrides (``size``, ``overlap``, ``source_url``...) win over values read
    from the document.
    """

    text = ""
    url = kwargs.pop("source_url", "")
    kind = kwargs.pop("source_kind", "")
    source_id = kwargs.pop("source_id", "")
    content_type = kwargs.pop("content_type", "")

    if isinstance(document, str):
        text = document
    elif document is None:
        return []
    elif isinstance(document, dict):
        text = str(document.get("text") or document.get("content") or "")
        url = url or str(document.get("url") or "")
        kind = kind or str(document.get("kind") or "")
        source_id = source_id or str(document.get("id") or document.get("source_id") or "")
        content_type = content_type or str(document.get("content_type") or "")
    else:  # attribute-style objects (FetchedSource)
        text = str(getattr(document, "text", "") or "")
        url = url or str(getattr(document, "url", "") or "")
        kind = kind or str(getattr(document, "kind", "") or "")
        source_id = source_id or str(getattr(document, "id", "") or "")
        content_type = content_type or str(getattr(document, "content_type", "") or "")

    if not text:
        return []
    cleaned = clean_document(text, url=url, content_type=content_type)
    return chunk_text(
        cleaned,
        source_url=url,
        source_kind=kind,
        source_id=source_id,
        **kwargs,
    )


def chunks_to_text(chunks: Iterable[Chunk]) -> str:
    """Flatten chunks back into a single anchored text blob (debug helper)."""

    return "\n\n".join("<!-- %s -->\n%s" % (c.anchor, c.text) for c in chunks)
