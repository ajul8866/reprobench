"""Per-host token-bucket rate limiting and exponential backoff.

The profile builder (``reprobench.profile.fetch``) and the NVD miner make many
requests against a small number of hosts.  Vendor support portals are slow and
frequently rate-limit aggressive clients, so every network call in ReproBench
goes through this module:

* one :class:`TokenBucket` per host (default 1 request / second, burst 2),
* :func:`backoff_delay` for exponential backoff with jitter,
* :class:`RetryPolicy` for a dependency-free retry loop (``tenacity`` is used
  by the NVD miner, but the profile builder must work without it).

Everything here is pure in-process state: no threads are spawned, no I/O is
performed.  ``threading.Lock`` only protects bucket arithmetic so the module is
safe if a caller fans out with a thread pool.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, TypeVar
from urllib.parse import urlparse

__all__ = [
    "DEFAULT_RATE",
    "DEFAULT_BURST",
    "RetryPolicy",
    "RateLimiter",
    "TokenBucket",
    "backoff_delay",
    "host_of",
    "limiter",
]

#: Default sustained request rate per host (requests / second).
DEFAULT_RATE: float = 1.0
#: Default burst capacity per host.
DEFAULT_BURST: float = 2.0

T = TypeVar("T")


def host_of(url: str) -> str:
    """Return the lowercase hostname of ``url`` (``""`` when unparseable)."""
    try:
        parsed = urlparse(str(url))
    except (TypeError, ValueError):
        return ""
    return (parsed.hostname or "").lower()


@dataclass
class TokenBucket:
    """Classic token bucket.

    ``rate`` tokens are added per second up to ``capacity``.  ``acquire``
    blocks (sleeping) until a token is available; ``try_acquire`` never blocks.
    """

    rate: float = DEFAULT_RATE
    capacity: float = DEFAULT_BURST
    tokens: Optional[float] = None
    updated: Optional[float] = None
    clock: Callable[[], float] = time.monotonic
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.rate <= 0:
            self.rate = DEFAULT_RATE
        if self.capacity <= 0:
            self.capacity = max(1.0, self.rate)
        if self.tokens is None:
            self.tokens = float(self.capacity)
        if self.updated is None:
            self.updated = self.clock()

    # -- internals ---------------------------------------------------------
    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - float(self.updated or now))
        if elapsed:
            self.tokens = min(float(self.capacity), float(self.tokens or 0.0) + elapsed * self.rate)
            self.updated = now

    # -- public API --------------------------------------------------------
    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Take ``tokens`` if available; return whether the take succeeded."""
        with self.lock:
            self._refill(self.clock())
            if float(self.tokens or 0.0) >= tokens:
                self.tokens = float(self.tokens or 0.0) - tokens
                return True
            return False

    def acquire(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """Block until ``tokens`` are available (or ``timeout`` expires)."""
        deadline = None if timeout is None else self.clock() + max(0.0, timeout)
        while True:
            with self.lock:
                self._refill(self.clock())
                available = float(self.tokens or 0.0)
                if available >= tokens:
                    self.tokens = available - tokens
                    return True
                needed = tokens - available
                wait = needed / self.rate
            if deadline is not None:
                remaining = deadline - self.clock()
                if remaining <= 0:
                    return False
                wait = min(wait, remaining)
            time.sleep(max(0.001, wait))

    def wait_time(self, tokens: float = 1.0) -> float:
        """Seconds until ``tokens`` would be available (0.0 when ready now)."""
        with self.lock:
            self._refill(self.clock())
            available = float(self.tokens or 0.0)
            if available >= tokens:
                return 0.0
            return (tokens - available) / self.rate


class RateLimiter:
    """A registry of :class:`TokenBucket` objects keyed by hostname."""

    def __init__(self, rate: float = DEFAULT_RATE, burst: float = DEFAULT_BURST) -> None:
        self.rate = float(rate)
        self.burst = float(burst)
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def bucket(self, host: str) -> TokenBucket:
        """Return (creating if needed) the bucket for ``host``."""
        key = (host or "default").lower()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = TokenBucket(rate=self.rate, capacity=self.burst)
                self._buckets[key] = bucket
            return bucket

    def acquire(self, url: str, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """Throttle a request to ``url`` against its host's bucket."""
        return self.bucket(host_of(url)).acquire(tokens=tokens, timeout=timeout)

    def wait(self, url: str, tokens: float = 1.0) -> float:
        """Seconds a caller would wait before requesting ``url``."""
        return self.bucket(host_of(url)).wait_time(tokens=tokens)

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 30.0, jitter: float = 0.25) -> float:
    """Exponential backoff delay for a 1-based ``attempt`` number.

    ``base * 2**(attempt-1)`` capped at ``cap`` with up to ``jitter`` fraction
    of multiplicative noise, so retries are not synchronised across workers.
    """
    attempt = max(1, int(attempt))
    delay = min(float(cap), float(base) * (2.0 ** (attempt - 1)))
    if jitter:
        delay *= 1.0 + random.uniform(-abs(jitter), abs(jitter))
    return max(0.0, delay)


@dataclass
class RetryPolicy:
    """Dependency-free retry loop with exponential backoff."""

    attempts: int = 4
    base: float = 1.0
    cap: float = 30.0
    jitter: float = 0.25
    retry_on: tuple = (Exception,)

    def run(self, fn: Callable[[], T], *, on_retry: Optional[Callable[[int, BaseException], None]] = None) -> T:
        """Call ``fn`` until it succeeds or attempts are exhausted."""
        last: Optional[BaseException] = None
        for attempt in range(1, max(1, int(self.attempts)) + 1):
            try:
                return fn()
            except self.retry_on as exc:  # type: ignore[misc]
                last = exc
                if attempt >= self.attempts:
                    break
                if on_retry is not None:
                    try:
                        on_retry(attempt, exc)
                    except Exception:  # pragma: no cover - observer must not break retry
                        pass
                time.sleep(backoff_delay(attempt, base=self.base, cap=self.cap, jitter=self.jitter))
        assert last is not None
        raise last


#: Process-wide limiter used by the profile fetcher by default.
limiter = RateLimiter()
