"""Candidate mining for the ReproBench dataset (paper Section 3, Table 3).

The released evaluation set is the 30 CVEs pinned in :mod:`data/cves.yaml`; this
module re-derives the *candidate pool* those 30 were selected from, using the
procedure documented in ``data/cves.yaml`` under ``selection:``:

* **Miner** -- NVD 2.0 API queries by CWE + ``pubStartDate``/``pubEndDate``,
  paginated at ``resultsPerPage=2000`` (``startIndex`` advanced by the page
  size).
* **Enrichment** -- CVE Services (CNA containers), CISA KEV, GitHub PoC
  search and ExploitDB/PacketStorm lookups.
* **Filters** -- F1 firmware URL reachable / known mirror, F2 ``>= 2``
  independent evidence sources including a concrete endpoint/binary, F3 CWE
  maps to exactly one class.  These live in :mod:`.filters`.

Two execution modes are supported:

``offline`` (default in CI)
    Never touches the network.  If ``data/candidate_pool.jsonl`` already exists
    it is replayed verbatim; otherwise a *deterministic synthetic pool* is
    generated from ``data/cves.yaml`` plus a handful of deliberately
    rejected-looking rows so that the selection log exercises accept/reject
    reasons.  Everything is labelled ``"synthetic": true``.

``live``
    Queries NVD 2.0 (``NVD_API_KEY`` optional, raises the rate limit), enriches
    each candidate and writes ``data/candidate_pool.jsonl``.  Network access is
    confined to :func:`_http_get_json`; the rest of the module is pure.

The module is import-safe: ``httpx`` and ``tenacity`` are imported lazily so
``import reprobench.dataset.mine`` works with only the core stack installed.

Public interface
----------------
``mine_candidates(out_path, offline=True, limit=None, year_start=None,
year_end=None, api_key_env="NVD_API_KEY") -> List[Dict[str, Any]]``
    Main entry point used by ``reprobench mine``.  Returns the written rows.

``synthetic_pool(limit=None) -> List[Dict[str, Any]]``
    Deterministic offline pool (released 30 + filler/rejected rows).

``nvd_query_url(...)`` / ``parse_nvd_vulnerability(...)`` / ``enrich_candidate(...)``
    Small, individually testable pieces of the live path.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..utils.io import (
    DATA_DIR,
    load_dataset,
    read_jsonl,
    write_jsonl,
)

__all__ = [
    "CLASS_CWE_SETS",
    "DEFAULT_END_YEAR",
    "DEFAULT_START_YEAR",
    "NVD_ENDPOINT",
    "PAGE_SIZE",
    "enrich_candidate",
    "mine_candidates",
    "nvd_query_url",
    "parse_nvd_vulnerability",
    "synthetic_pool",
]

# ---------------------------------------------------------------------------
# Constants (mirrored from data/cves.yaml `selection:` and `classes:`)
# ---------------------------------------------------------------------------

NVD_ENDPOINT = "https://services.nvd.nist.gov/rest/json/cves/2.0"
PAGE_SIZE = 2000
DEFAULT_START_YEAR = 2020
DEFAULT_END_YEAR = 2026

#: CWE -> class mapping (filter F3).  ``data/cves.yaml`` is authoritative; this
#: copy exists so the miner can run when the dataset file is unavailable.
CLASS_CWE_SETS: Dict[str, Tuple[int, ...]] = {
    "BOF": (120, 121, 122),
    "CMDI": (77, 78),
    "AUTH": (287, 288, 306, 284, 912, 798),
}

#: Source families that count towards F2 ("independent evidence sources").
SOURCE_FAMILIES: Tuple[str, ...] = (
    "nvd",
    "cna",
    "vendor_advisory",
    "cve_services",
    "cisa_kev",
    "github_poc",
    "exploitdb_packetstorm",
    "blog",
    "commit",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classes() -> Dict[str, Dict[str, Any]]:
    """Return the authoritative class -> CWE mapping from ``data/cves.yaml``."""

    try:
        raw = load_dataset("cves.yaml")
    except Exception:  # pragma: no cover - defensive
        return {}
    classes = raw.get("classes") or {}
    out: Dict[str, Dict[str, Any]] = {}
    for name, spec in classes.items():
        if isinstance(spec, dict):
            out[str(name)] = {
                "cwe": [int(c) for c in spec.get("cwe") or []],
                "label": spec.get("label", str(name)),
                "trigger_evidence_type": spec.get("trigger_evidence_type", "unknown"),
            }
    return out


def class_for_cwe(cwe: int, mapping: Optional[Dict[str, Dict[str, Any]]] = None) -> Optional[str]:
    """Map a single CWE id to its class, or ``None`` when it maps to zero/many.

    Filter F3 requires *exactly one* class; a CWE appearing in more than one
    class set (or in none) yields ``None``.
    """

    mapping = mapping if mapping is not None else _classes()
    if not mapping:
        mapping = {k: {"cwe": list(v)} for k, v in CLASS_CWE_SETS.items()}
    hits = [name for name, spec in mapping.items() if int(cwe) in (spec.get("cwe") or [])]
    return hits[0] if len(hits) == 1 else None


def _iso(year: int, month: int, day: int) -> str:
    """NVD 2.0 expects extended ISO-8601 with an explicit offset."""

    return f"{year:04d}-{month:02d}-{day:02d}T00:00:00.000+00:00"


def nvd_query_url(
    cwe: Optional[int] = None,
    keyword: Optional[str] = None,
    year_start: Optional[int] = DEFAULT_START_YEAR,
    year_end: Optional[int] = DEFAULT_END_YEAR,
    start_index: int = 0,
    results_per_page: int = PAGE_SIZE,
) -> str:
    """Build an NVD 2.0 query URL (paginated at 2000 records/page)."""

    params: List[str] = []
    if cwe is not None:
        params.append(f"cweId=CWE-{int(cwe)}")
    if keyword:
        params.append("keywordSearch=" + keyword.replace(" ", "+"))
    if year_start is not None:
        params.append(f"pubStartDate={_iso(int(year_start), 1, 1)}")
    if year_end is not None:
        params.append(f"pubEndDate={_iso(int(year_end), 12, 31)}")
    params.append(f"resultsPerPage={int(results_per_page)}")
    params.append(f"startIndex={int(start_index)}")
    return f"{NVD_ENDPOINT}?{'&'.join(params)}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# NVD parsing
# ---------------------------------------------------------------------------


def _english_description(cve: Dict[str, Any]) -> str:
    for desc in cve.get("descriptions") or []:
        if isinstance(desc, dict) and desc.get("lang") == "en":
            return str(desc.get("value") or "")
    descs = cve.get("descriptions") or []
    if descs and isinstance(descs[0], dict):
        return str(descs[0].get("value") or "")
    return ""


def _cwes_of(cve: Dict[str, Any]) -> List[int]:
    out: List[int] = []
    for weakness in cve.get("weaknesses") or []:
        for desc in (weakness or {}).get("description") or []:
            value = str((desc or {}).get("value") or "")
            if value.upper().startswith("CWE-"):
                try:
                    cwe = int(value.split("-", 1)[1])
                except (IndexError, ValueError):
                    continue
                if cwe not in out:
                    out.append(cwe)
    return out


def _published_year(cve: Dict[str, Any]) -> Optional[int]:
    stamp = str(cve.get("published") or "")
    if len(stamp) >= 4 and stamp[:4].isdigit():
        return int(stamp[:4])
    return None


def parse_nvd_vulnerability(vulnerability: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one NVD 2.0 ``vulnerabilities[i]`` record into a pool row.

    Returns ``None`` when the record is missing a CVE id.  CWE->class mapping
    is *not* applied here (that is filter F3's job) so unmapped records survive
    into the pool with ``class: null`` and an explicit rejection reason later.
    """

    cve = (vulnerability or {}).get("cve") or {}
    cve_id = str(cve.get("id") or "").strip().upper()
    if not cve_id.startswith("CVE-"):
        return None
    cwes = _cwes_of(cve)
    classes = sorted({c for c in (class_for_cwe(cwe) for cwe in cwes) if c})
    description = _english_description(cve)
    metrics = cve.get("metrics") or {}
    kev = bool(cve.get("cisaExploitAdd") or cve.get("cisaVulnerabilityName"))
    references = [
        str((ref or {}).get("url") or "")
        for ref in (cve.get("references") or [])
        if isinstance(ref, dict) and ref.get("url")
    ]
    return {
        "id": cve_id,
        "description": description,
        "cwe": cwes,
        "class": classes[0] if len(classes) == 1 else None,
        "classes": classes,
        "year": _published_year(cve),
        "published": cve.get("published"),
        "last_modified": cve.get("lastModified"),
        "vuln_status": cve.get("vulnStatus"),
        "references": references,
        "metrics": sorted(metrics.keys()) if isinstance(metrics, dict) else [],
        "cisa_kev": kev,
        "sources": ["nvd"] + (["cisa_kev"] if kev else []),
        "firmware_hint": "",
        "vendor": "",
        "products": [],
        "synthetic": False,
        "mined_at": _now_iso(),
    }


# ---------------------------------------------------------------------------
# Enrichment (live path only)
# ---------------------------------------------------------------------------


def enrich_candidate(
    candidate: Dict[str, Any],
    *,
    github_poc: bool = True,
    exploitdb: bool = True,
    timeout: float = 20.0,
    api_key_env: str = "NVD_API_KEY",
) -> Dict[str, Any]:
    """Best-effort enrichment of one pool row.

    Adds ``sources`` entries for CVE Services / GitHub PoC / ExploitDB and a
    ``has_concrete_artifact`` flag used by filter F2.  Every network failure is
    swallowed -- enrichment only ever *adds* evidence, never removes a row.
    """

    row = dict(candidate)
    sources: List[str] = list(row.get("sources") or [])
    text = f"{row.get('description', '')} " + " ".join(row.get("references") or [])

    if github_poc:
        try:
            payload = _http_get_json(
                "https://api.github.com/search/repositories",
                params={"q": f"{row.get('id', '')} PoC"},
                timeout=timeout,
            )
            if int((payload or {}).get("total_count") or 0) > 0:
                sources.append("github_poc")
        except Exception:  # pragma: no cover - network
            pass

    if exploitdb and any(k in text.lower() for k in ("exploit-db", "packetstorm", "exploitdb")):
        sources.append("exploitdb_packetstorm")

    if any(k in text.lower() for k in ("/cgi-bin/", "endpoint", "http request", "parameter")):
        row["has_concrete_artifact"] = True

    seen: List[str] = []
    for src in sources:
        if src not in seen:
            seen.append(src)
    row["sources"] = seen
    row["enriched_at"] = _now_iso()
    return row


# ---------------------------------------------------------------------------
# HTTP plumbing (lazy imports; never imported by the offline path)
# ---------------------------------------------------------------------------


def _http_get_json(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0,
    retries: int = 4,
    api_key_env: str = "NVD_API_KEY",
) -> Any:
    """GET ``url`` and decode JSON, retrying with exponential backoff.

    ``httpx``/``tenacity`` are imported lazily so the module (and therefore the
    CLI) imports cleanly in an offline environment.
    """

    import httpx  # local import: optional at module scope
    from tenacity import (  # local import
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    headers: Dict[str, str] = {"User-Agent": "reprobench-miner/1.0"}
    api_key = os.environ.get(api_key_env)
    if api_key:
        headers["apiKey"] = api_key

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(max(1, retries)),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _get() -> Any:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()

    return _get()


# ---------------------------------------------------------------------------
# Synthetic (offline) pool
# ---------------------------------------------------------------------------


def _released_entries() -> List[Dict[str, Any]]:
    try:
        raw = load_dataset("cves.yaml")
    except Exception:  # pragma: no cover - defensive
        return []
    return list(raw.get("cves") or [])


def _synthetic_extras(year_start: int, year_end: int) -> List[Dict[str, Any]]:
    """Rows deliberately shaped to be rejected by F1/F2/F3.

    They make ``selection_log.jsonl`` meaningful offline (accept *and* reject
    reasons) without claiming any real CVE beyond the released 30.
    """

    span = max(1, int(year_end) - int(year_start))
    rows: List[Dict[str, Any]] = []
    # F3 reject: CWE that maps to no class.
    rows.append(
        {
            "id": "CVE-2021-0001",
            "description": "Synthetic cross-site scripting candidate (no class mapping).",
            "cwe": [79],
            "class": None,
            "classes": [],
            "year": min(year_end, year_start + 1),
            "vendor": "Tenda",
            "products": ["AC6"],
            "sources": ["nvd"],
            "firmware_hint": "",
            "firmware_url": "",
            "has_concrete_artifact": False,
            "synthetic": True,
        }
    )
    # F2 reject: single source, no concrete endpoint/binary.
    rows.append(
        {
            "id": "CVE-2022-0002",
            "description": "Synthetic buffer overflow advisory with prose only.",
            "cwe": [120],
            "class": "BOF",
            "classes": ["BOF"],
            "year": min(year_end, year_start + 2),
            "vendor": "NETGEAR",
            "products": ["R7000"],
            "sources": ["nvd"],
            "firmware_hint": "",
            "firmware_url": "",
            "has_concrete_artifact": False,
            "synthetic": True,
        }
    )
    # F1 reject: plausible CVE but no firmware URL / known mirror.
    rows.append(
        {
            "id": "CVE-2023-0003",
            "description": "Synthetic command injection advisory naming /cgi-bin/luci endpoint.",
            "cwe": [78],
            "class": "CMDI",
            "classes": ["CMDI"],
            "year": min(year_end, year_start + 3),
            "vendor": "Unknown",
            "products": ["Unidentified OEM router"],
            "sources": ["nvd", "github_poc"],
            "firmware_hint": "unidentified OEM router",
            "firmware_url": "",
            "has_concrete_artifact": True,
            "synthetic": True,
        }
    )
    # F3 reject: CWE maps to two classes (constructed overlap).
    rows.append(
        {
            "id": "CVE-2024-0004",
            "description": "Synthetic ambiguous CWE candidate (78 and 287).",
            "cwe": [78, 287],
            "class": None,
            "classes": ["AUTH", "CMDI"],
            "year": min(year_end, year_start + 4),
            "vendor": "D-Link",
            "products": ["DIR-3040"],
            "sources": ["nvd", "vendor_advisory"],
            "firmware_hint": "",
            "firmware_url": "",
            "has_concrete_artifact": False,
            "synthetic": True,
        }
    )
    for idx, row in enumerate(rows, start=1):
        row.setdefault("published", f"{int(row['year']):04d}-0{min(9, idx)}-01T00:00:00.000")
        row.setdefault("references", [])
        row.setdefault("mined_at", _now_iso())
        row.setdefault("selection_hint", f"synthetic_reject_{idx}")
    _ = span
    return rows


def synthetic_pool(
    limit: Optional[int] = None,
    year_start: int = DEFAULT_START_YEAR,
    year_end: int = DEFAULT_END_YEAR,
) -> List[Dict[str, Any]]:
    """Deterministic offline pool: released 30 CVEs + F1/F2/F3 rejects."""

    rows: List[Dict[str, Any]] = []
    for entry in _released_entries():
        rows.append(
            {
                "id": entry.get("id"),
                "description": entry.get("notes")
                or f"{entry.get('vendor')} {', '.join(entry.get('products') or [])} "
                f"{entry.get('class')} ({entry.get('firmware_hint') or 'firmware'})",
                "cwe": list(entry.get("cwe") or []),
                "class": entry.get("class"),
                "classes": [entry.get("class")] if entry.get("class") else [],
                "year": entry.get("year"),
                "vendor": entry.get("vendor"),
                "products": list(entry.get("products") or []),
                "sources": list(entry.get("sources_hint") or ["nvd"]),
                "firmware_hint": entry.get("firmware_hint", ""),
                "firmware_url": "",
                "has_concrete_artifact": True,
                "advisory_url": entry.get("advisory_url", ""),
                "trigger_evidence_type": entry.get("trigger_evidence_type", "unknown"),
                "released": True,
                "synthetic": False,
                "mined_at": _now_iso(),
            }
        )
    rows.extend(_synthetic_extras(year_start, year_end))
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    return rows


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def mine_candidates(
    out_path: str | Path = "data/candidate_pool.jsonl",
    offline: bool = True,
    limit: Optional[int] = None,
    year_start: Optional[int] = DEFAULT_START_YEAR,
    year_end: Optional[int] = DEFAULT_END_YEAR,
    api_key_env: str = "NVD_API_KEY",
) -> List[Dict[str, Any]]:
    """Mine (or replay) the candidate pool and write it to ``out_path``.

    ``offline=True`` never performs network I/O:

    * if ``out_path`` exists it is read back and returned unchanged (replay of
      a previously mined pool);
    * otherwise a deterministic synthetic pool is generated.

    ``offline=False`` performs NVD 2.0 paginated queries (one query per CWE,
    ``resultsPerPage=2000``) and enriches every candidate.
    """

    out = Path(out_path)
    if not out.is_absolute():
        out = DATA_DIR.parent / out if (DATA_DIR.parent / out).exists() else Path(out)

    start = int(year_start) if year_start is not None else DEFAULT_START_YEAR
    end = int(year_end) if year_end is not None else DEFAULT_END_YEAR
    if end < start:
        start, end = end, start

    if offline:
        if out.exists():
            rows = read_jsonl(out)
            if limit is not None:
                rows = rows[: max(0, int(limit))]
            return rows
        rows = synthetic_pool(limit=limit, year_start=start, year_end=end)
        write_jsonl(out, rows)
        return rows

    rows = _live_mine(
        limit=limit,
        year_start=start,
        year_end=end,
        api_key_env=api_key_env,
    )
    write_jsonl(out, rows)
    return rows


def _live_mine(
    *,
    limit: Optional[int] = None,
    year_start: int = DEFAULT_START_YEAR,
    year_end: int = DEFAULT_END_YEAR,
    api_key_env: str = "NVD_API_KEY",
    sleep_seconds: float = 6.0,
) -> List[Dict[str, Any]]:
    """Paginated NVD 2.0 harvest across the three class CWE sets."""

    classes = _classes() or {k: {"cwe": list(v)} for k, v in CLASS_CWE_SETS.items()}
    by_id: Dict[str, Dict[str, Any]] = {}
    for class_name, spec in classes.items():
        for cwe in spec.get("cwe") or []:
            start_index = 0
            while True:
                url = nvd_query_url(
                    cwe=int(cwe),
                    year_start=year_start,
                    year_end=year_end,
                    start_index=start_index,
                    results_per_page=PAGE_SIZE,
                )
                payload = _http_get_json(url, api_key_env=api_key_env)
                vulnerabilities = (payload or {}).get("vulnerabilities") or []
                for item in vulnerabilities:
                    row = parse_nvd_vulnerability(item)
                    if row is None:
                        continue
                    row["cwe_query"] = int(cwe)
                    row["class_query"] = class_name
                    existing = by_id.get(row["id"])
                    if existing is None:
                        by_id[row["id"]] = row
                    else:
                        for c in row["cwe"]:
                            if c not in existing["cwe"]:
                                existing["cwe"].append(c)
                total = int((payload or {}).get("totalResults") or 0)
                start_index += len(vulnerabilities) or PAGE_SIZE
                if not vulnerabilities or start_index >= total:
                    break
                time.sleep(max(0.0, sleep_seconds))
            if limit is not None and len(by_id) >= int(limit):
                break
        if limit is not None and len(by_id) >= int(limit):
            break

    rows: List[Dict[str, Any]] = []
    for row in by_id.values():
        try:
            rows.append(enrich_candidate(row, api_key_env=api_key_env))
        except Exception:  # pragma: no cover - defensive
            rows.append(row)
    rows.sort(key=lambda r: (str(r.get("year") or ""), str(r.get("id") or "")))
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    return rows


def replay_or_mine(pool_path: str | Path = "data/candidate_pool.jsonl", **kwargs: Any) -> List[Dict[str, Any]]:
    """Convenience wrapper: replay ``pool_path`` when present, else mine."""

    kwargs.setdefault("offline", True)
    return mine_candidates(pool_path, **kwargs)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    written = mine_candidates(offline=True)
    print(json.dumps({"candidates": len(written)}, indent=2))
