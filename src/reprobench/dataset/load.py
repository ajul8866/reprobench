"""Dataset loading and validation for ReproBench.

Loads ``data/cves.yaml`` (paper Table 3, 30 CVEs) into lightweight, attribute
friendly records and validates every published invariant:

* exactly 30 unique CVE ids
* exactly 10 CVEs per class (BOF / CMDI / AUTH)
* exactly 14 distinct vendors
* publication years inside 2020-2026
* filter F3: each CVE's CWE maps to exactly one class
* class-derived ``trigger_evidence_type``
* the statically encoded ``derived`` block agrees with the entries

The loader is deliberately dependency-light (no pydantic, no network) so the
offline self-test and CI can validate the dataset with only PyYAML installed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..utils.io import DATA_DIR, load_dataset

__all__ = [
    "CVE",
    "CWE_TO_CLASS",
    "CLASS_CWE",
    "load_cves",
    "load_dataset_doc",
    "validate_dataset",
    "class_of_cwe",
    "summarize",
]

# ---------------------------------------------------------------------------
# Class definitions (mirrors data/cves.yaml -> classes, config/phases.yaml)
# ---------------------------------------------------------------------------

CLASS_CWE: Dict[str, List[int]] = {
    "BOF": [120, 121, 122],
    "CMDI": [77, 78],
    "AUTH": [287, 288, 306, 284, 912, 798],
}

CWE_TO_CLASS: Dict[int, str] = {
    cwe: cls for cls, cwes in CLASS_CWE.items() for cwe in cwes
}

CLASS_TRIGGER_EVIDENCE: Dict[str, str] = {
    "BOF": "crash",
    "CMDI": "command_output",
    "AUTH": "auth_bypass",
}

_CVE_RE = re.compile(r"^CVE-\d{4}-\d{4,}$")


class CVE:
    """Attribute accessor over one ``data/cves.yaml`` entry.

    Exposes both ``id``/``cve_id`` and ``class``/``cve_class``/``class_name``
    spellings because the CLI, the profile builder and the scorers each use a
    different one.
    """

    __slots__ = (
        "id",
        "vendor",
        "products",
        "year",
        "cwe",
        "trigger_evidence_type",
        "advisory_url",
        "sources_hint",
        "firmware_hint",
        "notes",
        "_class",
        "_raw",
    )

    def __init__(self, raw: Mapping[str, Any]) -> None:
        self._raw = dict(raw)
        self.id = str(raw.get("id") or raw.get("cve_id") or "").strip()
        self._class = str(raw.get("class") or raw.get("cve_class") or "").strip().upper()
        self.vendor = str(raw.get("vendor") or "").strip()
        self.products = [str(p) for p in (raw.get("products") or [])]
        try:
            self.year = int(raw.get("year"))
        except (TypeError, ValueError):
            self.year = 0
        self.cwe = [int(c) for c in (raw.get("cwe") or [])]
        self.trigger_evidence_type = str(raw.get("trigger_evidence_type") or "unknown")
        self.advisory_url = str(raw.get("advisory_url") or "")
        self.sources_hint = [str(s) for s in (raw.get("sources_hint") or [])]
        self.firmware_hint = str(raw.get("firmware_hint") or "")
        self.notes = str(raw.get("notes") or "")

    # -- spelling aliases --------------------------------------------------
    @property
    def cve_id(self) -> str:
        return self.id

    @property
    def cve_class(self) -> str:
        return self._class

    @property
    def class_name(self) -> str:
        return self._class

    def get(self, key: str, default: Any = None) -> Any:
        if key in ("id", "cve_id"):
            return self.id
        if key in ("class", "cve_class", "class_name"):
            return self._class
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "class": self._class,
            "vendor": self.vendor,
            "products": list(self.products),
            "year": self.year,
            "cwe": list(self.cwe),
            "trigger_evidence_type": self.trigger_evidence_type,
            "advisory_url": self.advisory_url,
            "sources_hint": list(self.sources_hint),
            "firmware_hint": self.firmware_hint,
            "notes": self.notes,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CVE({self.id}, {self._class}, {self.vendor})"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_dataset_doc(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load the raw ``cves.yaml`` document as a dict."""
    if path is None:
        doc = load_dataset("cves.yaml")
    else:
        p = Path(path)
        if not p.is_absolute():
            p = DATA_DIR / p
        from ..utils.io import load_yaml

        doc = load_yaml(p)
    return dict(doc or {})


def load_cves(
    path: Optional[str | Path] = None,
    cve_class: Optional[str] = None,
    ids: Optional[Iterable[str]] = None,
) -> List[CVE]:
    """Return the 30 released CVEs as :class:`CVE` records.

    ``cve_class`` filters by BOF/CMDI/AUTH; ``ids`` restricts to a set of ids
    (used by ``reprobench profile``/``run`` for single-CVE work).
    """
    doc = load_dataset_doc(path)
    entries = [CVE(raw) for raw in doc.get("cves", [])]
    if cve_class:
        wanted = cve_class.upper()
        entries = [e for e in entries if e.cve_class == wanted]
    if ids is not None:
        wanted_ids = {str(i) for i in ids}
        entries = [e for e in entries if e.id in wanted_ids]
    return entries


def class_of_cwe(cwe: int) -> Optional[str]:
    """Map a CWE number to its single ReproBench class (filter F3)."""
    return CWE_TO_CLASS.get(int(cwe))


def summarize(entries: Sequence[CVE]) -> Dict[str, Any]:
    """Roll up a list of CVEs into counts used by analysis/tests."""
    per_class: Dict[str, int] = {cls: 0 for cls in CLASS_CWE}
    vendors: Dict[str, int] = {}
    years: Dict[int, int] = {}
    for entry in entries:
        per_class[entry.cve_class] = per_class.get(entry.cve_class, 0) + 1
        vendors[entry.vendor] = vendors.get(entry.vendor, 0) + 1
        years[entry.year] = years.get(entry.year, 0) + 1
    return {
        "total": len(entries),
        "per_class": per_class,
        "vendor_count": len([v for v in vendors if v]),
        "per_vendor": vendors,
        "years": years,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_dataset(
    path: Optional[str | Path] = None,
    expected_total: int = 30,
    expected_per_class: int = 10,
    expected_vendors: int = 14,
    year_min: int = 2020,
    year_max: int = 2026,
) -> List[str]:
    """Validate every published dataset invariant.

    Returns a list of human-readable problems; an empty list means the dataset
    satisfies the paper's Table 3 invariants.
    """
    problems: List[str] = []
    try:
        doc = load_dataset_doc(path)
    except OSError as exc:  # pragma: no cover - missing data file
        return [f"cannot read dataset: {exc}"]

    raw_entries = doc.get("cves") or []
    if not raw_entries:
        return ["dataset contains no 'cves' entries"]

    entries = [CVE(raw) for raw in raw_entries]

    # -- ids ---------------------------------------------------------------
    ids = [e.id for e in entries]
    if len(ids) != expected_total:
        problems.append(f"expected {expected_total} CVEs, found {len(ids)}")
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        problems.append(f"duplicate CVE ids: {dupes}")
    for cve_id in ids:
        if not _CVE_RE.match(cve_id):
            problems.append(f"malformed CVE id: {cve_id!r}")

    # -- classes -----------------------------------------------------------
    per_class: Dict[str, int] = {}
    for entry in entries:
        if entry.cve_class not in CLASS_CWE:
            problems.append(f"{entry.id}: unknown class {entry.cve_class!r}")
            continue
        per_class[entry.cve_class] = per_class.get(entry.cve_class, 0) + 1
    for cls in CLASS_CWE:
        got = per_class.get(cls, 0)
        if got != expected_per_class:
            problems.append(
                f"class {cls}: expected {expected_per_class} CVEs, found {got}"
            )

    # -- vendors -----------------------------------------------------------
    vendors = {e.vendor for e in entries if e.vendor}
    if len(vendors) != expected_vendors:
        problems.append(
            f"expected {expected_vendors} vendors, found {len(vendors)}: {sorted(vendors)}"
        )

    # -- years -------------------------------------------------------------
    for entry in entries:
        if not (year_min <= entry.year <= year_max):
            problems.append(
                f"{entry.id}: year {entry.year} outside {year_min}-{year_max}"
            )

    # -- F3: CWE maps to exactly one class ---------------------------------
    for entry in entries:
        if not entry.cwe:
            problems.append(f"{entry.id}: no CWE listed")
            continue
        for cwe in entry.cwe:
            mapped = class_of_cwe(cwe)
            if mapped is None:
                problems.append(f"{entry.id}: CWE-{cwe} not in any class set")
            elif mapped != entry.cve_class:
                problems.append(
                    f"{entry.id}: CWE-{cwe} maps to {mapped}, entry class {entry.cve_class}"
                )

    # -- trigger evidence type is class-derived ----------------------------
    for entry in entries:
        expected = CLASS_TRIGGER_EVIDENCE.get(entry.cve_class)
        if expected and entry.trigger_evidence_type != expected:
            problems.append(
                f"{entry.id}: trigger_evidence_type {entry.trigger_evidence_type!r} "
                f"!= class-derived {expected!r}"
            )

    # -- derived block self-consistency ------------------------------------
    derived = doc.get("derived") or {}
    summary = summarize(entries)
    if derived:
        if int(derived.get("total", -1)) != summary["total"]:
            problems.append(
                f"derived.total {derived.get('total')} != actual {summary['total']}"
            )
        derived_per_class = derived.get("per_class") or {}
        for cls, count in summary["per_class"].items():
            if int(derived_per_class.get(cls, -1)) != count:
                problems.append(
                    f"derived.per_class[{cls}] {derived_per_class.get(cls)} != {count}"
                )
        if int(derived.get("vendor_count", -1)) != summary["vendor_count"]:
            problems.append(
                f"derived.vendor_count {derived.get('vendor_count')} "
                f"!= actual {summary['vendor_count']}"
            )
        derived_vendors = set(derived.get("vendors") or [])
        if derived_vendors and derived_vendors != vendors:
            missing = sorted(vendors - derived_vendors)
            extra = sorted(derived_vendors - vendors)
            if missing:
                problems.append(f"vendors missing from derived block: {missing}")
            if extra:
                problems.append(f"vendors listed in derived block but unused: {extra}")

    # -- expected block (if present) ---------------------------------------
    expected = doc.get("expected") or {}
    if expected:
        if int(expected.get("total", expected_total)) != summary["total"]:
            problems.append(
                f"expected.total {expected.get('total')} != {summary['total']}"
            )
        exp_per_class = expected.get("per_class") or {}
        for cls, count in exp_per_class.items():
            if summary["per_class"].get(cls, 0) != int(count):
                problems.append(
                    f"expected.per_class[{cls}] {count} != {summary['per_class'].get(cls, 0)}"
                )

    # -- class CWE sets agree with config ----------------------------------
    for cls, defined in (doc.get("classes") or {}).items():
        cwes = [int(c) for c in (defined.get("cwe") or [])]
        if cwes != CLASS_CWE.get(cls):
            problems.append(
                f"classes[{cls}].cwe {cwes} != canonical {CLASS_CWE.get(cls)}"
            )

    return problems
