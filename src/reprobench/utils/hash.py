"""Hashing helpers shared by the cache, provenance chain and manifest builder.

Thin wrappers over :mod:`reprobench.utils.io` so callers can depend on a
single, well-named hashing module without pulling in the whole I/O layer.
All functions are pure / offline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from .io import iter_files, sha256_bytes, sha256_file, sha256_text, stable_json

__all__ = [
    "dir_manifest",
    "dir_sha256",
    "file_sha256",
    "hash_dict",
    "hash_file",
    "hash_text",
    "sha256_bytes",
    "sha256_file",
    "sha256_text",
    "short_hash",
    "stable_json",
]


def hash_text(text: str) -> str:
    """SHA-256 hex digest of ``text`` (UTF-8)."""
    return sha256_text(text or "")


def hash_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file's contents (streaming)."""
    return sha256_file(path, chunk=chunk)


# Friendly aliases used in different parts of the codebase.
file_sha256 = hash_file


def hash_dict(obj: Any) -> str:
    """Deterministic SHA-256 of a JSON-serialisable object."""
    return sha256_text(stable_json(obj))


def short_hash(value: str, length: int = 12) -> str:
    """Truncated digest for human-readable ids (run ids, cache keys...)."""
    value = value or ""
    # If the input already looks like a hex digest, truncate it; otherwise hash it.
    if len(value) >= length and all(c in "0123456789abcdef" for c in value.lower()):
        return value[:length]
    return sha256_text(value)[:length]


def dir_manifest(root: str | Path) -> Dict[str, Dict[str, Any]]:
    """Return ``{rel_path: {sha256, size}}`` for every file under ``root``."""
    root_path = Path(root)
    manifest: Dict[str, Dict[str, Any]] = {}
    for path in iter_files(root_path):
        try:
            rel = str(path.relative_to(root_path))
        except ValueError:  # pragma: no cover - defensive
            rel = str(path)
        try:
            digest = sha256_file(path)
            size = path.stat().st_size
        except OSError:  # pragma: no cover - unreadable file
            digest, size = "", 0
        manifest[rel] = {"sha256": digest, "size": size}
    return manifest


def dir_sha256(root: str | Path) -> str:
    """Aggregate digest over a directory tree (order independent)."""
    manifest = dir_manifest(root)
    return sha256_text(
        json.dumps(
            {k: v["sha256"] for k, v in sorted(manifest.items())},
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def manifest_hash(entries: Iterable[Mapping[str, Any]]) -> str:
    """Hash a sequence of artifact records (rel_path/sha256/size)."""
    rows = sorted(
        (
            str(entry.get("rel_path") or entry.get("path") or ""),
            str(entry.get("sha256") or ""),
            int(entry.get("size") or 0),
        )
        for entry in entries
    )
    return sha256_text(json.dumps(rows, separators=(",", ":")))


def verify_file(path: str | Path, expected: Optional[str]) -> bool:
    """True when ``path`` exists and its digest equals ``expected``."""
    if not expected:
        return False
    try:
        return sha256_file(path) == expected
    except OSError:
        return False
