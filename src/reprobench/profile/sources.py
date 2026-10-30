"""Reference-source discovery for ReproBench profiles (plan component 3).

Given a ``CVE`` dataset entry, this module proposes the *ordered* list of
reference sources to fetch: NVD, CVE Services, vendor advisories, the CISA KEV
catalogue, GitHub PoC repositories, ExploitDB/PacketStorm and generic blogs.
Only metadata is produced here -- nothing is fetched (that is ``fetch.py``).

Ordering follows the profile conflict-resolution precedence documented in the
plan: NVD > CNA > vendor > blog > GitHub.  The list is deterministic so a
content-addressed cache can replay it offline.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import quote, urlparse

from ..utils.io import DATA_DIR, read_text

__all__ = [
    "SOURCE_KINDS",
    "SOURCE_PRECEDENCE",
    "NVD_DETAIL_URL",
    "CVE_SERVICES_URL",
    "KEV_URL",
    "Source",
    "gazetteer_hosts",
    "sources_for_cve",
    "precedence",
    "sort_sources",
    "github_poc_urls",
]

SOURCE_KINDS = (
    "nvd",
    "cve_services",
    "cna",
    "vendor_advisory",
    "cisa_kev",
    "blog",
    "commit",
    "github_poc",
    "exploitdb_packetstorm",
)

#: Conflict-resolution precedence (lower index wins).
SOURCE_PRECEDENCE: Dict[str, int] = {
    "nvd": 0,
    "cve_services": 1,
    "cna": 2,
    "vendor_advisory": 3,
    "cisa_kev": 4,
    "commit": 5,
    "blog": 6,
    "github_poc": 7,
    "exploitdb_packetstorm": 8,
}

NVD_DETAIL_URL = "https://nvd.nist.gov/vuln/detail/{cve_id}"
CVE_SERVICES_URL = "https://cveawg.mitre.org/api/cve/{cve_id}"
KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
GITHUB_SEARCH_URL = "https://api.github.com/search/repositories?q={query}"


class Source(dict):
    """A dict subclass so callers can use ``s["url"]`` or ``s.url``."""

    def __init__(
        self,
        url: str = "",
        kind: str = "blog",
        title: str = "",
        cve_id: str = "",
        priority: int = 99,
        **extra: Any,
    ) -> None:
        super().__init__(
            url=url,
            kind=kind,
            title=title,
            cve_id=cve_id,
            priority=priority,
            **extra,
        )

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - trivial
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(name) from exc

    def to_dict(self) -> Dict[str, Any]:
        return dict(self)


def _read_gazetteer() -> List[str]:
    text = read_text(DATA_DIR / "vendor_gazetteer.txt", default="")
    return [line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def gazetteer_hosts() -> Dict[str, str]:
    """Return ``host -> vendor`` from the pipe-delimited gazetteer HOST records."""
    hosts: Dict[str, str] = {}
    for line in _read_gazetteer():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3 and parts[0].upper() == "HOST":
            hosts[parts[2].lower()] = parts[1]
    return hosts


def _host_vendor(url: str, hosts: Mapping[str, str]) -> Optional[str]:
    try:
        host = urlparse(url).netloc.lower().split(":")[0]
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
    if not host:
        return None
    for known, vendor in hosts.items():
        if host == known or host.endswith("." + known):
            return vendor
    return None


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text or "").strip("-").lower()


def github_poc_urls(entry: Any, limit: int = 3) -> List[str]:
    """Deterministic GitHub search URLs for PoC discovery (never guessed repos)."""
    cve_id = getattr(entry, "id", "") or ""
    vendor = getattr(entry, "vendor", "") or ""
    products = list(getattr(entry, "products", []) or [])
    query = quote(f"{cve_id} poc" if not vendor else f"{cve_id} {vendor} poc")
    urls = [GITHUB_SEARCH_URL.format(query=query)]
    if products:
        urls.append(GITHUB_SEARCH_URL.format(query=quote(f"{cve_id} {products[0]} exploit")))
    return urls[:limit]


def sources_for_cve(entry: Any, *, include_search: bool = True) -> List[Source]:
    """Build the ordered reference-source list for one dataset entry.

    ``entry`` is a :class:`reprobench.dataset.load.CVE` or a plain mapping with
    ``id``/``cve_id``, ``vendor``, ``products`` and ``sources_hint``.
    """
    cve_id = getattr(entry, "id", None) or getattr(entry, "cve_id", "") or ""
    if not cve_id and isinstance(entry, Mapping):
        cve_id = str(entry.get("id") or entry.get("cve_id") or "")
    vendor = getattr(entry, "vendor", "") or ""
    hints = list(getattr(entry, "sources_hint", None) or [])
    advisory_url = getattr(entry, "advisory_url", "") or ""
    if isinstance(entry, Mapping):
        vendor = vendor or str(entry.get("vendor") or "")
        hints = hints or list(entry.get("sources_hint") or [])
        advisory_url = advisory_url or str(entry.get("advisory_url") or "")

    hosts = gazetteer_hosts()
    out: List[Source] = []

    def add(url: str, kind: str, title: str = "") -> None:
        if not url:
            return
        if any(existing["url"] == url for existing in out):
            return
        out.append(
            Source(
                url=url,
                kind=kind,
                title=title or f"{cve_id} {kind}",
                cve_id=cve_id,
                priority=SOURCE_PRECEDENCE.get(kind, 99),
            )
        )

    # Always available, authoritative anchors.
    add(advisory_url or NVD_DETAIL_URL.format(cve_id=cve_id), "nvd", f"{cve_id} NVD record")
    add(CVE_SERVICES_URL.format(cve_id=cve_id), "cve_services", f"{cve_id} CVE Services")

    # Advisory-derived sources first (vendor/CNA), then corroborating sources.
    for hint in hints:
        key = str(hint).strip().lower()
        if key in ("nvd", "cve_services"):
            continue
        if key in ("cisa_kev", "kev"):
            add(KEV_URL, "cisa_kev", "CISA KEV catalogue")
        elif key in ("vendor_advisory", "vendor", "advisory"):
            url = getattr(entry, "advisory_url", "") or ""
            host = _host_vendor(url, hosts)
            if url and host:
                add(url, "vendor_advisory", f"{cve_id} {host} advisory")
        elif key in ("cna",):
            add(CVE_SERVICES_URL.format(cve_id=cve_id), "cna", f"{cve_id} CNA record")
        elif key in ("github_poc", "github", "poc"):
            if include_search:
                for url in github_poc_urls(entry):
                    add(url, "github_poc", f"{cve_id} PoC search")
        elif key in ("exploitdb_packetstorm", "exploitdb", "packetstorm"):
            q = quote(cve_id)
            add(f"https://www.exploit-db.com/search?cve={q}", "exploitdb_packetstorm", f"{cve_id} ExploitDB")
            add(f"https://packetstormsecurity.com/search/?q={q}", "exploitdb_packetstorm", f"{cve_id} PacketStorm")
        elif key in ("blog",):
            add(f"https://www.google.com/search?q={quote(cve_id + ' firmware exploit')}", "blog", f"{cve_id} blog search")
        elif key in ("commit",):
            add(f"https://github.com/search?q={quote(cve_id)}&type=commits", "commit", f"{cve_id} commit search")

    return sort_sources(out)


def precedence(kind: str) -> int:
    """Conflict-resolution rank for a source kind (unknown kinds rank last)."""
    return SOURCE_PRECEDENCE.get(str(kind or "").lower(), 99)


def sort_sources(sources: Iterable[Any]) -> List[Source]:
    """Stable sort by precedence then original insertion order."""
    items = list(sources)
    decorated = []
    for index, src in enumerate(items):
        kind = src.get("kind", "blog") if isinstance(src, Mapping) else getattr(src, "kind", "blog")
        decorated.append((precedence(kind), index, src))
    decorated.sort(key=lambda row: (row[0], row[1]))
    return [row[2] for row in decorated]
