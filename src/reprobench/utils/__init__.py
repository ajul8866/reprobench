"""ReproBench utility layer.

Single import surface for the small, dependency-light helpers used across the
benchmark: filesystem/config I/O, hashing, content-addressed caching, per-host
rate limiting, and logging.

Submodules (all offline-safe, no network at import time):

* :mod:`reprobench.utils.io`       -- YAML/JSON/JSONL I/O, sha256 helpers,
  ``round_half_up`` and crude statistics used by the scoring arithmetic.
* :mod:`reprobench.utils.hash`     -- hashing façade (files, dicts, manifests,
  directory digests, short ids).
* :mod:`reprobench.utils.cache`    -- on-disk content-addressed ``DiskCache``.
* :mod:`reprobench.utils.ratelimit`-- token buckets + exponential backoff.
* :mod:`reprobench.utils.log`      -- shared ``reprobench`` logger namespace.

Only the most frequently used symbols are re-exported here; import the
submodules directly for everything else.
"""

from __future__ import annotations

from .io import (
    CONFIG_DIR,
    DATA_DIR,
    PROJECT_ROOT,
    ensure_dir,
    load_config,
    load_dataset,
    load_yaml,
    read_json,
    read_jsonl,
    read_text,
    relpath,
    round_half_up,
    sha256_bytes,
    sha256_file,
    sha256_text,
    stable_json,
    write_json,
    write_jsonl,
    write_text,
)
from .log import configure as configure_logging
from .log import get_logger, log_event

__all__ = [
    "CONFIG_DIR",
    "DATA_DIR",
    "PROJECT_ROOT",
    "configure_logging",
    "ensure_dir",
    "get_logger",
    "load_config",
    "load_dataset",
    "load_yaml",
    "log_event",
    "read_json",
    "read_jsonl",
    "read_text",
    "relpath",
    "round_half_up",
    "sha256_bytes",
    "sha256_file",
    "sha256_text",
    "stable_json",
    "write_json",
    "write_jsonl",
    "write_text",
]
