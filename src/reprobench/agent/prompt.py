"""Agent-facing prompt rendering (paper Fig. 3).

The prompt lives verbatim in ``config/prompt_template.md`` and contains exactly one
Python format placeholder, ``{cve_id}``.  Nothing else about the benchmark (phase
names, rubric items, attempt credit, gating) may leak into the rendered text; the
``test_no_leak.py`` suite enforces this.

Public interface
----------------
``render_prompt(cve_id)``            -> str   (byte-exact Fig. 3 text)
``load_template()``                  -> str
``save_prompt(cve_id, out_dir)``     -> Path  (``out_dir/prompt.txt``)
``check_leakage(text)``              -> list[str]  (offending tokens)
``LEAKAGE_TOKENS``                   -> tuple[str, ...]
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional

from ..utils.io import CONFIG_DIR, read_text, write_text

TEMPLATE_NAME = "prompt_template.md"
PROMPT_FILENAME = "prompt.txt"

#: Tokens that must never appear in the rendered prompt.  Checked case-insensitively
#: and as whole words where sensible; ``phase``/``rubric``/``score`` are checked as
#: substrings because any occurrence would be a leak.
LEAKAGE_TOKENS = (
    "phase",
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6",
    "rubric",
    "checklist",
    "attempt credit",
    "gating",
    "coverage",
    "dependency",
    "fallback",
    "judge",
    "max_points",
    "scoring",
    "simulation substitution",
)

_PLACEHOLDER = "{cve_id}"


def template_path() -> Path:
    """Return the path of the verbatim Fig. 3 template."""
    return Path(CONFIG_DIR) / TEMPLATE_NAME


def load_template() -> str:
    """Read the raw template text (with the ``{cve_id}`` placeholder intact)."""
    return read_text(template_path())


def render_prompt(cve_id: str) -> str:
    """Render the Fig. 3 prompt for ``cve_id``.

    Uses ``str.replace`` rather than ``str.format`` so that any future literal braces
    in the template cannot raise ``KeyError``/``IndexError``; the template contains a
    single placeholder today, so both produce identical output.
    """
    if not cve_id:
        raise ValueError("cve_id must be a non-empty string")
    template = load_template()
    if _PLACEHOLDER not in template:
        raise ValueError(f"{template_path()} is missing the {_PLACEHOLDER} placeholder")
    return template.replace(_PLACEHOLDER, str(cve_id))


def save_prompt(cve_id: str, out_dir: str | Path) -> Path:
    """Render the prompt and persist it as ``<out_dir>/prompt.txt``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return write_text(out / PROMPT_FILENAME, render_prompt(cve_id))


def check_leakage(text: str, tokens: Optional[Iterable[str]] = None) -> List[str]:
    """Return the leakage tokens found in ``text`` (empty list means clean)."""
    found: List[str] = []
    lowered = text.lower()
    for token in tokens or LEAKAGE_TOKENS:
        needle = token.lower()
        if len(needle) <= 3 and needle[0].isalpha():
            # short tokens such as "p1" must match as standalone words
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(needle)}(?![A-Za-z0-9_])", lowered):
                found.append(token)
        elif needle in lowered:
            found.append(token)
    return found


def prompt_metadata(cve_id: str) -> dict:
    """Small helper for run metadata: prompt text, hash and template path."""
    from ..utils.io import sha256_text

    text = render_prompt(cve_id)
    return {
        "cve_id": cve_id,
        "template": str(template_path()),
        "sha256": sha256_text(text),
        "prompt_version": "fig3-v1",
        "chars": len(text),
    }


__all__ = [
    "LEAKAGE_TOKENS",
    "PROMPT_FILENAME",
    "TEMPLATE_NAME",
    "check_leakage",
    "load_template",
    "prompt_metadata",
    "render_prompt",
    "save_prompt",
    "template_path",
]
