"""Filesystem, config-loading and hashing helpers for ReproBench.

Everything that touches disk lives here so that the scoring engine can stay
pure (dict in -> dict out) and therefore trivially unit-testable offline.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

__all__ = [
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "DATA_DIR",
    "load_yaml",
    "load_config",
    "load_dataset",
    "read_text",
    "write_text",
    "read_json",
    "write_json",
    "read_jsonl",
    "write_jsonl",
    "sha256_bytes",
    "sha256_text",
    "sha256_file",
    "stable_json",
    "ensure_dir",
    "round_half_up",
    "clamp",
    "relpath",
]

# --------------------------------------------------------------------------
# Layout discovery
# --------------------------------------------------------------------------
# src/reprobench/utils/io.py -> parents: [utils, reprobench, src, <root>]
_CANDIDATES = Path(__file__).resolve().parents


def _find_project_root() -> Path:
    for parent in _CANDIDATES:
        if (parent / "config").is_dir() and (parent / "data").is_dir():
            return parent
    # fall back to the repository-ish parent of ``src``
    for parent in _CANDIDATES:
        if (parent / "config").is_dir():
            return parent
    return _CANDIDATES[0]


PROJECT_ROOT: Path = _find_project_root()
CONFIG_DIR: Path = PROJECT_ROOT / "config"
DATA_DIR: Path = PROJECT_ROOT / "data"


# --------------------------------------------------------------------------
# YAML / text
# --------------------------------------------------------------------------
def load_yaml(path: str | os.PathLike[str]) -> Any:
    """Load a YAML document.  ``path`` may be absolute or relative to the root."""
    import yaml  # local import: keeps module import cheap

    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_config(name: str) -> Any:
    """Load ``config/<name>`` (``.yaml`` suffix optional)."""
    fname = name if Path(name).suffix else f"{name}.yaml"
    return load_yaml(CONFIG_DIR / fname)


def load_dataset(name: str = "cves.yaml") -> Any:
    return load_yaml(DATA_DIR / name)


def read_text(path: str | os.PathLike[str], default: str | None = None) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        if default is None:
            raise
        return default


def write_text(path: str | os.PathLike[str], text: str) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# JSON / JSONL
# --------------------------------------------------------------------------
def read_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        if default is None:
            raise
        return default


def write_json(path: str | os.PathLike[str], obj: Any, indent: int = 2) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=indent, sort_keys=True) + "\n", encoding="utf-8")
    return p


def read_jsonl(path: str | os.PathLike[str]) -> list[dict]:
    out: list[dict] = []
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    except OSError:
        return out
    return out


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]]) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return p


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | os.PathLike[str], chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def stable_json(obj: Any) -> str:
    """Deterministic JSON used for cache keys / manifest hashes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


# --------------------------------------------------------------------------
# Numeric helpers
# --------------------------------------------------------------------------
def round_half_up(value: float, digits: int = 0) -> float:
    """Round half away from zero (paper's ``round_half_up``)."""
    if value is None:
        return 0.0
    quant = Decimal(1).scaleb(-digits)
    rounded = Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)
    return float(rounded)


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------
def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def relpath(path: str | os.PathLike[str], start: str | os.PathLike[str] | None = None) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(start or PROJECT_ROOT).resolve()))
    except (ValueError, OSError):
        return str(path)


def iter_files(root: str | os.PathLike[str]) -> Iterator[Path]:
    base = Path(root)
    if not base.exists():
        return iter(())
    return (p for p in sorted(base.rglob("*")) if p.is_file())


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    if not den:
        return default
    return num / den


def mean(values: Sequence[float]) -> float:
    vals = [v for v in values if v is not None]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def median(values: Sequence[float]) -> float:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return float(vals[mid])
    return (vals[mid - 1] + vals[mid]) / 2.0


def pct(part: float, total: float, digits: int = 1) -> float:
    if not total:
        return 0.0
    return round_half_up(100.0 * part / total, digits)


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
