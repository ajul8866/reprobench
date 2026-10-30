"""Reference-profile source fetching (plan: ``profile/fetch.py``).

The fetch step is the first of the profile builder's three-step skill::

    fetch   -> cache by sha256, per-host token bucket, exponential backoff
    clean   -> readability HTML -> text
    distill -> LLM map-reduce into ``profile.json``
    verify  -> validate trigger evidence / binaries / endpoints

Design notes
------------
* **Content addressed.**  Every successful fetch is stored in :class:`DiskCache`
  under a key derived from ``(url, etag-independent)`` request identity, and the
  payload is additionally addressable by its sha256.  Re-running the whole
  profile build therefore performs *zero* network calls (offline replay).
* **Polite.**  One :class:`~reprobench.utils.ratelimit.TokenBucket` per host and
  jittered exponential backoff on retryable failures (429/5xx/transport).
* **Offline-first.**  ``offline=True`` never touches the network; a cache miss
  yields a :class:`FetchedSource` with ``ok=False`` and ``status="offline_miss"``
  so the caller can mark the profile ``incomplete`` instead of crashing.

The module is import-safe without ``httpx``/``tenacity`` installed: those are
imported lazily inside :func:`_http_get`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from ..schema import SourceRecord
from ..utils.cache import DiskCache, cache_key
from ..utils.hash import hash_text, short_hash
from ..utils.log import get_logger
from ..utils.ratelimit import RetryPolicy, host_of, limiter

__all__ = [
    "CACHE_NAMESPACE",
    "FetchedSource",
    "fetch_many",
    "fetch_source",
    "fetch_url",
    "fetch_text",
]

LOG = get_logger("profile.fetch")

#: DiskCache namespace for raw source payloads.
CACHE_NAMESPACE = "profile_sources"

#: HTTP statuses worth retrying.
RETRY_STATUS = (408, 425, 429, 500, 502, 503, 504)

DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_BYTES = 16 * 1024 * 1024  # 16 MiB per source is plenty for advisories

USER_AGENT = (
    "ReproBench/1.0 (+https://example.invalid/reprobench) "
    "reference-profile-builder"
)


@dataclass
class FetchedSource:
    """Result of fetching one URL (cached or live)."""

    url: str
    ok: bool = False
    status: str = "ok"
    status_code: Optional[int] = None
    text: str = ""
    sha256: str = ""
    bytes_len: int = 0
    content_type: str = ""
    cache_key: str = ""
    cache_path: Optional[str] = None
    fetched_at: Optional[str] = None
    cached: bool = False
    error: Optional[str] = None
    attempts: int = 0
    redirects: int = 0
    kind: str = "blog"
    title: str = ""

    def record(self, source_id: str = "") -> SourceRecord:
        """Convert into the persisted :class:`~reprobench.schema.SourceRecord`."""
        return SourceRecord(
            url=self.url,
            kind=self.kind,
            sha256=self.sha256 or None,
            fetched_at=self.fetched_at,
            cache_path=self.cache_path,
            title=self.title or source_id or None,
            ok=bool(self.ok),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "ok": self.ok,
            "status": self.status,
            "status_code": self.status_code,
            "sha256": self.sha256,
            "bytes": self.bytes_len,
            "content_type": self.content_type,
            "cache_key": self.cache_key,
            "cache_path": self.cache_path,
            "cached": self.cached,
            "error": self.error,
            "attempts": self.attempts,
            "kind": self.kind,
        }


def _cache(cache: Optional[DiskCache] = None) -> DiskCache:
    return cache if cache is not None else DiskCache(CACHE_NAMESPACE)


def key_for(url: str) -> str:
    """Deterministic cache key for a URL."""
    return cache_key("profile-source-v1", url.strip())


def _lookup(cache: DiskCache, url: str) -> Optional[FetchedSource]:
    """Return a cached fetch for *url* if one exists."""
    key = key_for(url)
    if not cache.has(key):
        return None
    meta = cache.meta(key) or {}
    text = cache.get_text(key, name="payload.txt")
    if text is None:
        # Older entries may have been written as bytes; tolerate both.
        raw = cache.get_bytes(key, name="payload.txt")
        if raw is None:
            return None
        text = raw.decode("utf-8", errors="replace")
    return FetchedSource(
        url=url,
        ok=bool(meta.get("ok", True)),
        status=str(meta.get("status", "ok")),
        status_code=meta.get("status_code"),
        text=text,
        sha256=str(meta.get("sha256") or hash_text(text)),
        bytes_len=int(meta.get("bytes") or len(text.encode("utf-8", "replace"))),
        content_type=str(meta.get("content_type") or ""),
        cache_key=key,
        cache_path=str(cache.entry_dir(key)),
        fetched_at=meta.get("fetched_at"),
        cached=True,
        attempts=int(meta.get("attempts") or 0),
        kind=str(meta.get("kind") or "blog"),
        title=str(meta.get("title") or ""),
    )


def _store(cache: DiskCache, result: FetchedSource) -> FetchedSource:
    """Persist a fetch result and fill in cache/sha fields."""
    key = result.cache_key or key_for(result.url)
    meta = {
        "url": result.url,
        "ok": result.ok,
        "status": result.status,
        "status_code": result.status_code,
        "content_type": result.content_type,
        "sha256": result.sha256 or hash_text(result.text),
        "kind": result.kind,
        "title": result.title,
        "attempts": result.attempts,
        "fetched_at": result.fetched_at,
    }
    path = cache.put_text(key, result.text, name="payload.txt", meta=meta)
    result.cache_key = key
    result.sha256 = meta["sha256"]
    result.bytes_len = len(result.text.encode("utf-8", "replace"))
    result.cache_path = str(path.parent if path.suffix else path)
    return result


def _http_get(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_bytes: int = DEFAULT_MAX_BYTES,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, str, str]:
    """Single HTTP GET.  Returns ``(status_code, text, content_type)``.

    ``httpx`` is imported lazily so the module imports on a bare interpreter.
    Raises on transport errors / non-2xx status (the caller retries).
    """
    import httpx  # noqa: PLC0415  (lazy: optional dependency at import time)

    request_headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if headers:
        request_headers.update(headers)

    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        response = client.get(url, headers=request_headers)
        body = response.content[: max(1, max_bytes)]
        try:
            text = body.decode(response.encoding or "utf-8", errors="replace")
        except (LookupError, TypeError):  # pragma: no cover - defensive
            text = body.decode("utf-8", errors="replace")
        content_type = response.headers.get("content-type", "")
        if response.status_code >= 400:
            raise _HttpStatusError(response.status_code, text, content_type)
        return response.status_code, text, content_type


class _HttpStatusError(RuntimeError):
    """Carries the HTTP status of a failed fetch through the retry policy."""

    def __init__(self, status: int, text: str = "", content_type: str = "") -> None:
        super().__init__(f"HTTP {status}")
        self.status = status
        self.text = text
        self.content_type = content_type


def _retryable(exc: BaseException) -> bool:
    status = getattr(exc, "status", None)
    if status is None:
        return True  # transport error -> retry
    return int(status) in RETRY_STATUS


def fetch_url(
    url: str,
    *,
    offline: bool = False,
    refresh: bool = False,
    cache: Optional[DiskCache] = None,
    kind: str = "blog",
    title: str = "",
    timeout: float = DEFAULT_TIMEOUT,
    max_bytes: int = DEFAULT_MAX_BYTES,
    attempts: int = 4,
    clock: Optional[Callable[[], float]] = None,
) -> FetchedSource:
    """Fetch *url* honouring the cache, the per-host token bucket and backoff.

    Parameters
    ----------
    offline:
        Never perform a network request; return a cache miss instead.
    refresh:
        Ignore an existing cache entry and re-fetch (still writes the cache).
    """
    url = (url or "").strip()
    if not url:
        return FetchedSource(url="", ok=False, status="empty_url", error="empty url")

    store = _cache(cache)
    if not refresh:
        cached = _lookup(store, url)
        if cached is not None:
            cached.kind = cached.kind or kind
            cached.title = cached.title or title
            return cached

    if offline:
        return FetchedSource(
            url=url,
            ok=False,
            status="offline_miss",
            error="offline mode and no cached copy",
            cache_key=key_for(url),
            kind=kind,
            title=title,
        )

    host = host_of(url) or "default"
    try:
        limiter.acquire(url)  # per-host token bucket
    except Exception:  # pragma: no cover - limiter is deterministic, be safe
        LOG.debug("rate limiter failed for %s", host)

    policy = RetryPolicy(
        attempts=max(1, int(attempts)),
        base=1.0,
        cap=30.0,
        jitter=0.25,
        retry_on=(Exception,),
    )
    last_error: Optional[BaseException] = None
    tries = 0

    def _attempt() -> Tuple[int, str, str]:
        nonlocal tries
        tries += 1
        return _http_get(url, timeout=timeout, max_bytes=max_bytes)

    try:
        status_code, text, content_type = policy.run(
            _attempt,
            on_retry=lambda attempt, exc: LOG.warning(
                "retry %s for %s (%s)", attempt, short_hash(url), exc
            ),
        )
    except _HttpStatusError as exc:  # terminal 4xx
        last_error = exc
        result = FetchedSource(
            url=url,
            ok=False,
            status="http_error",
            status_code=exc.status,
            error=f"HTTP {exc.status}",
            attempts=tries,
            kind=kind,
            title=title,
            cache_key=key_for(url),
        )
        # 403/404 are *real* attempt evidence (plan Tier 4: curl->403 = 1.0),
        # so record them in the cache as a failed-but-known outcome.
        return _store(store, result)
    except Exception as exc:  # transport failure after retries
        last_error = exc
        LOG.warning("fetch failed for %s: %s", url, exc)
        return FetchedSource(
            url=url,
            ok=False,
            status="fetch_error",
            error=f"{type(exc).__name__}: {exc}",
            attempts=tries,
            kind=kind,
            title=title,
            cache_key=key_for(url),
        )

    if last_error is not None:  # pragma: no cover - defensive
        return FetchedSource(
            url=url,
            ok=False,
            status="fetch_error",
            error=str(last_error),
            attempts=tries,
            kind=kind,
            title=title,
        )

    result = FetchedSource(
        url=url,
        ok=True,
        status="ok",
        status_code=status_code,
        text=text,
        sha256=hash_text(text),
        bytes_len=len(text.encode("utf-8", "replace")),
        content_type=content_type,
        attempts=tries,
        kind=kind,
        title=title,
        fetched_at=_utc_now(),
        cache_key=key_for(url),
    )
    LOG.info("fetched %s (%s bytes, %s tries)", url, result.bytes_len, tries)
    return _store(store, result)


def fetch_text(url: str, **kwargs: Any) -> str:
    """Convenience wrapper returning just the payload text ('' on failure)."""
    return fetch_url(url, **kwargs).text


def fetch_source(source: Any, *, offline: bool = False, refresh: bool = False,
                 cache: Optional[DiskCache] = None, **kwargs: Any) -> FetchedSource:
    """Fetch a source descriptor (mapping, ``SourceRecord`` or plain string)."""
    if isinstance(source, str):
        return fetch_url(source, offline=offline, refresh=refresh, cache=cache, **kwargs)
    if isinstance(source, dict):
        url = str(source.get("url") or "")
        kind = str(source.get("kind") or kwargs.pop("kind", "blog"))
        title = str(source.get("title") or kwargs.pop("title", ""))
    else:
        url = str(getattr(source, "url", "") or "")
        kind = str(getattr(source, "kind", "blog") or "blog")
        title = str(getattr(source, "title", "") or "")
    return fetch_url(
        url,
        offline=offline,
        refresh=refresh,
        cache=cache,
        kind=kind,
        title=title,
        **kwargs,
    )


def fetch_many(
    sources: Any,
    *,
    offline: bool = False,
    refresh: bool = False,
    cache: Optional[DiskCache] = None,
    **kwargs: Any,
) -> list:
    """Fetch a sequence of sources sequentially (rate limiter handles hosts)."""
    results = []
    for source in sources or []:
        results.append(
            fetch_source(
                source,
                offline=offline,
                refresh=refresh,
                cache=cache,
                **kwargs,
            )
        )
    return results


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
