"""Structured logging for ReproBench.

A tiny wrapper over the :mod:`logging` stdlib module so every ReproBench
component logs in the same shape and the CLI can switch verbosity once.

Design notes
------------
* Import-safe: no configuration happens at import time beyond obtaining a
  logger namespace.  Call :func:`configure` (or :func:`get_logger`) explicitly.
* Offline: writes to stderr only; no network, no files unless a path is given.
* The trace of *agent* tool calls is NOT here -- that is ``agent/trace.py``.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "DEFAULT_FORMAT",
    "LEVELS",
    "configure",
    "get_logger",
    "log_event",
    "reset",
]

DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
LOG_ENV_VAR = "REPROBENCH_LOG_LEVEL"

LEVELS: Dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_ROOT_NAME = "reprobench"
_configured = False


def _root() -> logging.Logger:
    return logging.getLogger(_ROOT_NAME)


def reset() -> None:
    """Remove all handlers from the ReproBench logger namespace.

    Mostly useful for tests that assert on captured log output.
    """
    global _configured
    logger = _root()
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:  # pragma: no cover - defensive
            handler.close()
        except Exception:
            pass
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    _configured = False


def configure(
    level: Any = "INFO",
    path: Optional[str | Path] = None,
    fmt: str = DEFAULT_FORMAT,
    force: bool = False,
) -> logging.Logger:
    """Configure the ReproBench root logger and return it.

    Parameters
    ----------
    level:
        Either a level name (``"DEBUG"``) or a numeric/logging level.  ``None``
        falls back to ``REPROBENCH_LOG_LEVEL`` and then ``"INFO"``.
    path:
        Optional file to append to *in addition to* stderr.
    fmt:
        Log format string.
    force:
        Re-configure even if a previous call already installed handlers.
    """
    global _configured
    logger = _root()
    if _configured and not force:
        return logger

    if level is None:
        level = os.environ.get(LOG_ENV_VAR, "INFO")
    if isinstance(level, str):
        level = LEVELS.get(level.strip().upper(), logging.INFO)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter(fmt)

    stream = logging.StreamHandler(stream=sys.stderr)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if path is not None:
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(target, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError:  # pragma: no cover - disk/permission issues
            logger.warning("could not open log file %s", target)

    logger.setLevel(level)
    logger.propagate = False
    _configured = True
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child logger under the ``reprobench`` namespace.

    ``get_logger()`` returns the root ReproBench logger; ``get_logger("scoring")``
    returns ``reprobench.scoring``.
    """
    if not _configured:
        configure()
    if not name:
        return _root()
    if name.startswith(_ROOT_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")


def log_event(logger: logging.Logger, event: str, level: Any = "INFO", **fields: Any) -> None:
    """Log a single structured event as ``key=value`` pairs.

    Values containing spaces are quoted so the line stays greppable.
    """
    if isinstance(level, str):
        level_value = LEVELS.get(level.strip().upper(), logging.INFO)
    else:
        level_value = level

    parts: List[str] = [event]
    for key in sorted(fields):
        value = fields[key]
        text = str(value)
        if any(ch.isspace() for ch in text) or text == "":
            text = '"' + text.replace('"', "'") + '"'
        parts.append(f"{key}={text}")
    logger.log(level_value, " ".join(parts))
