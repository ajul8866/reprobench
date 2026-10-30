"""Dataset selection filters for ReproBench (paper Section 3, filter F1-F3).

The paper mines a 696-item candidate pool from NVD 2.0 (CWE + publication window
queries), enriches each candidate with CVE Services, CISA KEV, GitHub PoC search
and ExploitDB/PacketStorm, and then applies three filters:

* **F1 firmware_reachable** -- a firmware image URL is reachable or a known
  mirror exists (checked against ``data/vendor_gazetteer.txt`` HOST records).
* **F2 evidence_density** -- at least two *independent* evidence sources, one of
  which names a concrete endpoint or binary.
* **F3 single_class_cwe** -- the CVE's CWE set maps to exactly one of the three
  benchmark classes (BOF / CMDI / AUTH).

Surviving candidates are stratified to the top 10 per class with a logged
accept/reject reason for every candidate (``selection_log.jsonl``).

This module is pure and offline: it never performs network I/O. Reachability is
evaluated against the candidate's recorded ``firmware_urls`` plus the gazetteer's
known-mirror hosts, which is what the paper's F1 check reduces to for a released
dataset. Pass ``check_reachable=True`` and a ``probe`` callable to inject live
HTTP checks (used by ``dataset/mine.py`` in online mode).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..utils.io import (
    DATA_DIR,
    ensure_dir,
    load_dataset,
    read_jsonl,
    relpath,
    write_jsonl,
)

__all__ = [
    "CLASS_CWE",
    "CONCRETE_ARTIFACT_KINDS",
    "CandidateEvaluation",
    "class_for_cwe",
    "classes_cwe_map",
    "evaluate_candidate",
    "gazetteer_hosts",
    "load_gazetteer",
    "select_candidates",
    "stratify",
]


# --------------------------------------------------------------------------- #
# Constants (Table 3 / config/phases.yaml classes block)
# --------------------------------------------------------------------------- #

CLASS_CWE: Dict[str, Tuple[int, ...]] = {
    "BOF": (120, 121, 122),
    "CMDI": (77, 78),
    "AUTH": (287, 288, 306, 284, 912, 798),
}

CLASS_ORDER: Tuple[str, ...] = ("BOF", "CMDI", "AUTH")

#: Evidence source families considered independent of one another. Trusted
#: sources are the ones the paper counts first (Table 6 verification block);
#: corroborating sources are the enrichment families.
TRUSTED_SOURCES: Tuple[str, ...] = (
    "nvd",
    "cna",
    "vendor_advisory",
    "cve_services",
    "cisa_kev",
)
CORROBORATING_SOURCES: Tuple[str, ...] = (
    "github_poc",
    "exploitdb_packetstorm",
    "exploitdb",
    "packetstorm",
    "blog",
    "commit",
    "poc",
)

#: Keys on a candidate record that can supply the "concrete artifact" required
#: by F2 (a concrete endpoint / parameter / binary named by the evidence).
CONCRETE_ARTIFACT_KINDS: Tuple[str, ...] = (
    "endpoint",
    "parameter",
    "binary",
    "concrete_endpoint",
    "concrete_binary",
    "vulnerable_binary",
)

SOURCE_ALIASES: Dict[str, str] = {
    "nvd": "nvd",
    "nvd_2.0": "nvd",
    "cna": "cna",
    "vendor": "vendor_advisory",
    "vendor_advisory": "vendor_advisory",
    "advisory": "vendor_advisory",
    "cve_services": "cve_services",
    "cisa_kev": "cisa_kev",
    "kev": "cisa_kev",
    "github": "github_poc",
    "github_poc": "github_poc",
    "poc": "github_poc",
    "exploitdb": "exploitdb_packetstorm",
    "packetstorm": "exploitdb_packetstorm",
    "exploitdb_packetstorm": "exploitdb_packetstorm",
    "blog": "blog",
    "commit": "commit",
}


# --------------------------------------------------------------------------- #
# Gazetteer
# --------------------------------------------------------------------------- #


def load_gazetteer(path: Optional[Path] = None) -> Dict[str, Any]:
    """Parse ``data/vendor_gazetteer.txt`` into structured maps.

    Returns a dict with keys ``vendors`` (canonical -> aliases/products),
    ``products`` (vendor -> list of product names), ``hosts`` (host -> kind),
    ``aliases`` (lowercased alias -> canonical vendor) and ``classes``
    (class -> list of prose phrases).
    """

    path = Path(path) if path is not None else DATA_DIR / "vendor_gazetteer.txt"
    vendors: Dict[str, Dict[str, Any]] = {}
    products: Dict[str, List[str]] = {}
    hosts: Dict[str, str] = {}
    aliases: Dict[str, str] = {}
    classes: Dict[str, List[str]] = {}

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {
            "vendors": vendors,
            "products": products,
            "hosts": hosts,
            "aliases": aliases,
            "classes": classes,
        }

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if not parts:
            continue
        tag = parts[0].upper()
        if tag == "VENDOR" and len(parts) >= 4:
            canonical = parts[1]
            vendor_aliases = [a.strip() for a in parts[2].split(",") if a.strip()]
            vendor_products = [p.strip() for p in parts[3].split(",") if p.strip()]
            vendors[canonical] = {
                "aliases": vendor_aliases,
                "products": vendor_products,
            }
            aliases[canonical.lower()] = canonical
            for alias in vendor_aliases:
                aliases[alias.lower()] = canonical
            if vendor_products:
                products.setdefault(canonical, []).extend(vendor_products)
        elif tag == "PRODUCT" and len(parts) >= 4:
            vendor, product = parts[1], parts[2]
            products.setdefault(vendor, []).append(product)
            for alias in parts[3].split(","):
                alias = alias.strip()
                if alias:
                    aliases[alias.lower()] = vendor
        elif tag == "HOST" and len(parts) >= 4:
            hosts[parts[2].lower()] = parts[3]
        elif tag == "CLASS_ALIAS" and len(parts) >= 3:
            classes.setdefault(parts[1].upper(), []).append(parts[2])

    return {
        "vendors": vendors,
        "products": products,
        "hosts": hosts,
        "aliases": aliases,
        "classes": classes,
    }


def gazetteer_hosts(path: Optional[Path] = None) -> Dict[str, str]:
    """Known firmware/support host -> kind (official / mirror / legacy_ftp)."""

    return dict(load_gazetteer(path).get("hosts") or {})


# --------------------------------------------------------------------------- #
# Class mapping (filter F3)
# --------------------------------------------------------------------------- #


def classes_cwe_map(cves_config: Optional[Mapping[str, Any]] = None) -> Dict[str, Tuple[int, ...]]:
    """CWE -> class mapping, preferring ``data/cves.yaml`` over built-ins."""

    mapping: Dict[str, Tuple[int, ...]] = {}
    if cves_config:
        for name, spec in (cves_config.get("classes") or {}).items():
            cwes = spec.get("cwe") if isinstance(spec, Mapping) else None
            if cwes:
                mapping[str(name).upper()] = tuple(int(c) for c in cwes)
    if not mapping:
        try:
            loaded = load_dataset("cves.yaml")
            return classes_cwe_map(loaded)
        except Exception:  # pragma: no cover - defensive
            mapping = dict(CLASS_CWE)
    return mapping


def class_for_cwe(cwe: Any, mapping: Optional[Mapping[str, Tuple[int, ...]]] = None) -> Optional[str]:
    """Return the single class a CWE set maps to, or ``None`` if ambiguous.

    ``cwe`` may be an int, a string, an iterable of ints/strings, or a string
    like ``"CWE-78"``/``"78,77"``.
    """

    mapping = mapping or classes_cwe_map()
    values = _normalize_cwe(cwe)
    if not values:
        return None
    matched = {cls for cls, cwes in mapping.items() if any(v in cwes for v in values)}
    if len(matched) == 1:
        return next(iter(matched))
    return None


def _normalize_cwe(cwe: Any) -> List[int]:
    if cwe is None:
        return []
    if isinstance(cwe, int):
        return [cwe]
    if isinstance(cwe, str):
        tokens = re.findall(r"\d+", cwe)
        return [int(t) for t in tokens]
    if isinstance(cwe, Iterable):
        out: List[int] = []
        for item in cwe:
            out.extend(_normalize_cwe(item))
        return out
    return []


# --------------------------------------------------------------------------- #
# Candidate evaluation
# --------------------------------------------------------------------------- #


@dataclass
class CandidateEvaluation:
    """Result of applying F1/F2/F3 to one candidate."""

    cve_id: str
    accepted: bool = False
    cve_class: Optional[str] = None
    vendor: str = ""
    year: Optional[int] = None
    f1_reachable: bool = False
    f1_reason: str = ""
    f2_sources: List[str] = field(default_factory=list)
    f2_concrete: bool = False
    f2_reason: str = ""
    f3_class: Optional[str] = None
    f3_reason: str = ""
    rejected_by: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    evidence_density: float = 0.0
    firmware_hosts: List[str] = field(default_factory=list)

    def as_log_row(self, stage: str = "filter") -> Dict[str, Any]:
        return {
            "cve_id": self.cve_id,
            "stage": stage,
            "accepted": self.accepted,
            "class": self.cve_class,
            "vendor": self.vendor,
            "year": self.year,
            "filters": {
                "F1": {"passed": self.f1_reachable, "reason": self.f1_reason},
                "F2": {
                    "passed": self.f2_concrete and len(self.f2_sources) >= 2,
                    "sources": self.f2_sources,
                    "concrete": self.f2_concrete,
                    "reason": self.f2_reason,
                },
                "F3": {"passed": self.f3_class is not None, "reason": self.f3_reason},
            },
            "rejected_by": self.rejected_by,
            "reasons": self.reasons,
            "evidence_density": self.evidence_density,
        }


def _candidate_id(candidate: Mapping[str, Any]) -> str:
    for key in ("id", "cve_id", "cve", "CVE"):
        value = candidate.get(key)
        if value:
            return str(value).strip().upper()
    return ""


def _candidate_sources(candidate: Mapping[str, Any]) -> List[str]:
    raw = (
        candidate.get("sources")
        or candidate.get("source_families")
        or candidate.get("sources_hint")
        or candidate.get("evidence_sources")
        or []
    )
    if isinstance(raw, str):
        raw = [raw]
    out: List[str] = []
    for item in raw:
        if isinstance(item, Mapping):
            item = item.get("kind") or item.get("type") or item.get("source") or ""
        key = str(item).strip().lower()
        canonical = SOURCE_ALIASES.get(key, key)
        if canonical and canonical not in out:
            out.append(canonical)
    return out


def _candidate_cwe(candidate: Mapping[str, Any]) -> List[int]:
    for key in ("cwe", "cwes", "cwe_ids", "weaknesses"):
        if key in candidate:
            return _normalize_cwe(candidate[key])
    return []


def _candidate_urls(candidate: Mapping[str, Any]) -> List[str]:
    urls: List[str] = []
    for key in ("firmware_urls", "firmware_url", "download_urls", "mirrors", "urls"):
        value = candidate.get(key)
        if not value:
            continue
        if isinstance(value, str):
            urls.append(value)
        elif isinstance(value, Iterable):
            for item in value:
                if isinstance(item, Mapping):
                    item = item.get("url") or item.get("href") or ""
                if item:
                    urls.append(str(item))
    return urls


def _host_of(url: str) -> str:
    match = re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://([^/]+)", url.strip())
    host = match.group(1) if match else url.strip().split("/")[0]
    return host.split("@")[-1].split(":")[0].lower().lstrip("www.") if host else ""


def _has_concrete_artifact(candidate: Mapping[str, Any]) -> bool:
    for key in CONCRETE_ARTIFACT_KINDS:
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)) and list(value):
            return True
    indicators = candidate.get("indicators")
    if isinstance(indicators, Mapping):
        for key in CONCRETE_ARTIFACT_KINDS:
            if indicators.get(key):
                return True
    # a firmware URL that names a concrete image file also counts
    for url in _candidate_urls(candidate):
        if re.search(r"\.(bin|img|trx|chk|zip|tar|gz|squashfs|ubi|fw)\b", url, re.I):
            return True
    return False


def evaluate_candidate(
    candidate: Mapping[str, Any],
    *,
    cwe_map: Optional[Mapping[str, Tuple[int, ...]]] = None,
    hosts: Optional[Mapping[str, str]] = None,
    min_sources: int = 2,
    check_reachable: bool = False,
    probe: Optional[Callable[[str], bool]] = None,
) -> CandidateEvaluation:
    """Apply filters F1, F2 and F3 to a single mined candidate."""

    hosts = hosts if hosts is not None else gazetteer_hosts()
    cwe_map = cwe_map or classes_cwe_map()

    cve_id = _candidate_id(candidate)
    vendor = str(candidate.get("vendor") or candidate.get("vendor_name") or "")
    year = candidate.get("year") or candidate.get("published_year")
    try:
        year = int(year) if year is not None else None
    except (TypeError, ValueError):
        year = None

    evaluation = CandidateEvaluation(cve_id=cve_id, vendor=vendor, year=year)

    # -- F1: firmware image reachable or a known mirror exists ---------------
    urls = _candidate_urls(candidate)
    known_mirror = False
    resolved_hosts: List[str] = []
    for url in urls:
        host = _host_of(url)
        if not host:
            continue
        resolved_hosts.append(host)
        if host in hosts or any(host.endswith("." + h) for h in hosts):
            known_mirror = True
    if urls and check_reachable and probe is not None:
        reachable = any(_safe_probe(probe, url) for url in urls)
    else:
        reachable = bool(urls)
    evaluation.firmware_hosts = sorted(set(resolved_hosts))
    evaluation.f1_reachable = bool(urls) and (reachable or known_mirror)
    if not urls:
        evaluation.f1_reason = "no firmware image URL or known mirror recorded"
    elif known_mirror:
        evaluation.f1_reason = "firmware URL hosted on known vendor/mirror host"
    elif evaluation.f1_reachable:
        evaluation.f1_reason = "firmware URL recorded and reachable"
    else:
        evaluation.f1_reason = "firmware URL present but unreachable and host not in gazetteer"
    if not evaluation.f1_reachable:
        evaluation.rejected_by.append("F1")
        evaluation.reasons.append(f"F1: {evaluation.f1_reason}")

    # -- F2: >= 2 independent sources including a concrete artifact ---------
    sources = _candidate_sources(candidate)
    concrete = _has_concrete_artifact(candidate)
    evaluation.f2_sources = sources
    evaluation.f2_concrete = concrete
    trusted = [s for s in sources if s in TRUSTED_SOURCES]
    corroborating = [s for s in sources if s in CORROBORATING_SOURCES]
    independent = len(trusted) >= min_sources or (bool(trusted) and bool(corroborating)) or (
        len(sources) >= min_sources
    )
    evaluation.evidence_density = round(
        float(len(trusted)) + 0.5 * float(len(corroborating)) + (1.0 if concrete else 0.0), 3
    )
    if len(sources) < min_sources:
        evaluation.f2_reason = f"only {len(sources)} independent source(s), need >= {min_sources}"
    elif not independent:
        evaluation.f2_reason = "sources are not independent families"
    elif not concrete:
        evaluation.f2_reason = "no concrete endpoint or binary named by the evidence"
    else:
        evaluation.f2_reason = f"{len(sources)} independent sources incl. concrete artifact"
    if not (len(sources) >= min_sources and independent and concrete):
        evaluation.rejected_by.append("F2")
        evaluation.reasons.append(f"F2: {evaluation.f2_reason}")

    # -- F3: CWE maps to exactly one class ---------------------------------
    mapping = class_for_cwe(_candidate_cwe(candidate), cwe_map)
    evaluation.f3_class = mapping
    evaluation.cve_class = mapping or (str(candidate.get("class")).upper() if candidate.get("class") else None)
    if mapping:
        evaluation.f3_reason = f"CWE set maps to {mapping}"
    else:
        evaluation.f3_reason = "CWE set is empty or maps to zero/multiple classes"
        evaluation.rejected_by.append("F3")
        evaluation.reasons.append(f"F3: {evaluation.f3_reason}")

    evaluation.accepted = not evaluation.rejected_by and bool(cve_id)
    if not cve_id:
        evaluation.reasons.append("missing CVE id")
    return evaluation


def _safe_probe(probe: Callable[[str], bool], url: str) -> bool:
    try:
        return bool(probe(url))
    except Exception:  # pragma: no cover - network path
        return False


# --------------------------------------------------------------------------- #
# Stratification & selection
# --------------------------------------------------------------------------- #


def _sort_key(evaluation: CandidateEvaluation) -> Tuple[Any, ...]:
    """Paper order: publication year desc, vendor diversity, evidence density."""

    return (
        -(evaluation.year or 0),
        -evaluation.evidence_density,
        evaluation.cve_id,
    )


def stratify(
    evaluations: Sequence[CandidateEvaluation],
    per_class: int = 10,
    *,
    existing_vendors: Optional[Iterable[str]] = None,
) -> List[CandidateEvaluation]:
    """Stratified top-``per_class`` per class with vendor diversity.

    Candidates are ranked by (year desc, evidence density desc). A first pass
    prefers vendors not already represented in the accepted set for that class
    (and not already used by an earlier class) to maximize vendor coverage;
    a second pass fills any remaining slots strictly by rank.
    """

    selected: List[CandidateEvaluation] = []
    used_vendors: List[str] = [str(v) for v in (existing_vendors or [])]

    for cls in CLASS_ORDER:
        pool = [e for e in evaluations if e.accepted and e.cve_class == cls]
        pool.sort(key=_sort_key)
        chosen: List[CandidateEvaluation] = []
        deferred: List[CandidateEvaluation] = []
        for candidate in pool:
            if len(chosen) >= per_class:
                deferred.append(candidate)
                continue
            if candidate.vendor and candidate.vendor in used_vendors:
                deferred.append(candidate)
                continue
            chosen.append(candidate)
            if candidate.vendor:
                used_vendors.append(candidate.vendor)
        for candidate in deferred:
            if len(chosen) >= per_class:
                break
            chosen.append(candidate)
            if candidate.vendor and candidate.vendor not in used_vendors:
                used_vendors.append(candidate.vendor)
        selected.extend(chosen)

    return selected


def select_candidates(
    pool_path: Path,
    log_path: Optional[Path] = None,
    per_class: int = 10,
    out_path: Optional[Path] = None,
    *,
    gazetteer_path: Optional[Path] = None,
    min_sources: int = 2,
    cves_config: Optional[Mapping[str, Any]] = None,
    check_reachable: bool = False,
    probe: Optional[Callable[[str], bool]] = None,
) -> Dict[str, Any]:
    """Run F1/F2/F3 over a mined candidate pool and pin the evaluation set.

    Returns a summary dict with ``accepted`` (number selected), ``evaluated``,
    ``per_class`` counts, the selected CVE ids, and the paths written. The
    selection log records an accept/reject reason for every candidate.
    """

    pool_path = Path(pool_path)
    rows = read_jsonl(pool_path) if pool_path.exists() else []
    if not isinstance(rows, list):
        rows = list(rows)

    gaz = load_gazetteer(gazetteer_path)
    hosts = gaz.get("hosts") or {}
    cwe_map = classes_cwe_map(cves_config)

    evaluations: List[CandidateEvaluation] = []
    seen: set = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        cve_id = _candidate_id(row)
        if cve_id and cve_id in seen:
            evaluations.append(
                CandidateEvaluation(
                    cve_id=cve_id,
                    rejected_by=["duplicate"],
                    reasons=["duplicate CVE id in candidate pool"],
                    cve_class=str(row.get("class") or "").upper() or None,
                )
            )
            continue
        if cve_id:
            seen.add(cve_id)
        evaluations.append(
            evaluate_candidate(
                row,
                cwe_map=cwe_map,
                hosts=hosts,
                min_sources=min_sources,
                check_reachable=check_reachable,
                probe=probe,
            )
        )

    selected = stratify(evaluations, per_class=per_class)
    selected_ids = {e.cve_id for e in selected}

    per_class_counts: Dict[str, int] = {cls: 0 for cls in CLASS_ORDER}
    for evaluation in selected:
        if evaluation.cve_class in per_class_counts:
            per_class_counts[evaluation.cve_class] += 1

    log_rows: List[Dict[str, Any]] = []
    for evaluation in evaluations:
        row = evaluation.as_log_row(stage="select")
        row["selected"] = evaluation.cve_id in selected_ids
        log_rows.append(row)

    if log_path is not None:
        log_path = Path(log_path)
        ensure_dir(log_path.parent)
        write_jsonl(log_path, log_rows)

    if out_path is not None:
        out_path = Path(out_path)
        ensure_dir(out_path.parent)
        write_jsonl(
            out_path,
            [
                {
                    "id": e.cve_id,
                    "class": e.cve_class,
                    "vendor": e.vendor,
                    "year": e.year,
                    "sources": e.f2_sources,
                    "evidence_density": e.evidence_density,
                    "firmware_hosts": e.firmware_hosts,
                }
                for e in selected
            ],
        )

    return {
        "evaluated": len(evaluations),
        "accepted": len(selected),
        "per_class": per_class_counts,
        "cve_ids": [e.cve_id for e in selected],
        "vendors": sorted({e.vendor for e in selected if e.vendor}),
        "log": relpath(log_path) if log_path is not None else None,
        "out": relpath(out_path) if out_path is not None else None,
        "rejected": sum(1 for e in evaluations if not e.accepted),
    }
