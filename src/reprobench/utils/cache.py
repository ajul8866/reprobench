"""Content-addressed, on-disk cache used by profile fetching and the judge.

Implements the plan's *"cache all network fetches content-addressed for offline
replay"* rule.  Two entry points exist:

* :func:`cache_dir` – default cache location (``<project>/.cache``).
* :class:`DiskCache` – tiny content-addressed store with JSON metadata.

The cache is deliberately synchronous and dependency-free so that the whole
offline self-test can run with no third-party packages beyond PyYAML.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .io import ensure_dir, read_json, sha256_bytes, sha256_text, stable_json, write_json
from .io import PROJECT_ROOT, CONFIG_DIR

__all__ = [
    "cache_dir",
    "CACHE_DIR",
    "DiskCache",
    "cache_key",
    "get_cached",
    "put_cached",
]

#: Default cache directory (``<project>/.cache``).
CACHE_DIR: Path = PROJECT_ROOT / ".cache"


def cache_dir(*parts: str, create: bool = True) -> Path:
    """Return ``<project>/.cache/<parts...>`` (optionally created)."""
    path = CACHE_DIR.joinpath(*[str(p) for p in parts]) if parts else CACHE_DIR
    if create:
        ensure_dir(path)
    return path


def cache_key(*parts: Any) -> str:
    """Deterministic cache key from arbitrary JSON-stable parts."""
    return sha256_text(stable_json(list(parts)))


@dataclass
class DiskCache:
    """Content-addressed store for network fetches and judge responses.

    Layout::

        <root>/<namespace>/<key[:2]>/<key>/meta.json
        <root>/<namespace>/<key[:2]>/<key>/payload.bin
        <root>/<namespace>/<key[:2]>/<key>/payload.json   (optional)

    ``key`` is normally a sha256 over the request (url, judge model, rubric
    version, ...).  Reads never raise: missing or malformed entries simply
    report a miss.
    """

    namespace: str = "default"
    root: Path = field(default_factory=lambda: CACHE_DIR)

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    # ------------------------------------------------------------------ paths
    def entry_dir(self, key: str) -> Path:
        key = str(key)
        return self.root / self.namespace / key[:2] / key

    def path_for(self, key: str, name: str = "payload.bin") -> Path:
        return self.entry_dir(key) / name

    # ------------------------------------------------------------------- read
    def has(self, key: str) -> bool:
        return (self.entry_dir(key) / "meta.json").exists()

    def meta(self, key: str) -> Dict[str, Any]:
        data = read_json(self.entry_dir(key) / "meta.json", default=None)
        return data if isinstance(data, dict) else {}

    def get_bytes(self, key: str, name: str = "payload.bin") -> Optional[bytes]:
        path = self.path_for(key, name)
        try:
            return path.read_bytes()
        except OSError:
            return None

    def get_text(self, key: str, name: str = "payload.bin") -> Optional[str]:
        data = self.get_bytes(key, name)
        if data is None:
            return None
        return data.decode("utf-8", errors="replace")

    def get_json(self, key: str, name: str = "payload.json") -> Any:
        return read_json(self.path_for(key, name), default=None)

    # ------------------------------------------------------------------ write
    def put_bytes(
        self,
        key: str,
        data: bytes,
        name: str = "payload.bin",
        meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        entry = ensure_dir(self.entry_dir(key))
        path = entry / name
        path.write_bytes(data)
        record: Dict[str, Any] = {
            "key": str(key),
            "namespace": self.namespace,
            "name": name,
            "sha256": sha256_bytes(data),
            "bytes": len(data),
            "stored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        record.update(meta or {})
        write_json(entry / "meta.json", record)
        return path

    def put_text(
        self,
        key: str,
        text: str,
        name: str = "payload.bin",
        meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        return self.put_bytes(key, text.encode("utf-8"), name=name, meta=meta)

    def put_json(
        self,
        key: str,
        obj: Any,
        name: str = "payload.json",
        meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        ensure_dir(self.entry_dir(key))
        write_json(self.path_for(key, name), obj)
        payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
        record: Dict[str, Any] = {
            "key": str(key),
            "namespace": self.namespace,
            "name": name,
            "sha256": sha256_text(payload),
            "bytes": len(payload.encode("utf-8")),
            "stored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        record.update(meta or {})
        write_json(self.entry_dir(key) / "meta.json", record)
        return self.path_for(key, name)

    # ------------------------------------------------------------- inspection
    def keys(self) -> List[str]:
        base = self.root / self.namespace
        if not base.exists():
            return []
        out: List[str] = []
        for meta in sorted(base.glob("*/*/meta.json")):
            data = read_json(meta, default={}) or {}
            key = data.get("key") if isinstance(data, dict) else None
            if key:
                out.append(str(key))
        return out

    def stats(self) -> Dict[str, Any]:
        keys = self.keys()
        total = 0
        for key in keys:
            meta = self.meta(key)
            try:
                total += int(meta.get("bytes") or 0)
            except (TypeError, ValueError):
                pass
        return {"namespace": self.namespace, "entries": len(keys), "bytes": total}


# --------------------------------------------------------------------------
# Module-level helpers around a default cache instance
# --------------------------------------------------------------------------
def _default() -> DiskCache:
    return DiskCache()


def get_cached(key: str, namespace: str = "default", name: str = "payload.bin") -> Optional[str]:
    """Return cached text for ``key`` or ``None``."""
    return DiskCache(namespace=namespace).get_text(key, name=name)


def put_cached(
    key: str,
    text: str,
    namespace: str = "default",
    name: str = "payload.bin",
    meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Store ``text`` for ``key`` in the default cache root."""
    return DiskCache(namespace=namespace).put_text(key, text, name=name, meta=meta)
