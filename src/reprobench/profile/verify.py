"""Reference-profile verification (paper Table 6 / plan step (3) "verify").

Responsibilities
----------------
The distiller produces a :class:`~reprobench.schema.ReferenceProfile`.  Before a
profile may be used as ground truth by the rubric it must pass the
``config/trigger_evidence.yaml`` gate:

* ``trigger_evidence_type`` is a member of the closed enum ``types[*].id``;
* it is not one of the sentinel values ``none`` / ``unknown``;
* a concrete endpoint or binary is named;
* the claim is corroborated by at least one source beyond advisory prose
  (``cna`` / ``vendor_advisory`` / ``cve_services`` / ``cisa_kev`` / commit /
  GitHub PoC / ExploitDB ...);
* the affected version or version range is stated.

On success ``concrete_trigger_evidence_verified`` becomes ``True`` and the
profile ``status`` is ``"complete"``.  On failure the profile is marked
``status="incomplete"`` or ``"manual_review"`` (configurable) and
:attr:`ReferenceProfile.excluded` is ``True`` so the run never enters the
benchmark matrix.

All constants live in YAML; the small literal tables below are defensive
fallbacks for offline unit tests only.

Public interface
----------------
``verify_profile``      main entry point (accepts pydantic models or mappings)
``verify_profiles``     batch helper
``verification_rate``   aggregate verification statistics
``VerificationResult``  detailed outcome
``check_trigger_evidence`` single-rule helpers used by the judge/tests
``load_trigger_rules``  cached YAML loader
``MANUAL_REVIEW`` / ``INCOMPLETE`` / ``COMPLETE`` status constants
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..schema import EVIDENCE_TYPE_IDS, ReferenceProfile
from ..utils.io import load_config
from ..utils.log import get_logger

log = get_logger("profile.verify")

__all__ = [
    "COMPLETE",
    "INCOMPLETE",
    "MANUAL_REVIEW",
    "VerificationResult",
    "check_trigger_evidence",
    "load_trigger_rules",
    "tree_has_binary",
    "verification_rate",
    "verify_profile",
    "verify_profiles",
]

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #

CONFIG_NAME = "trigger_evidence.yaml"

COMPLETE = "complete"
INCOMPLETE = "incomplete"
MANUAL_REVIEW = "manual_review"

#: Sentinel evidence types that can never earn credit.
SENTINELS: Tuple[str, ...] = ("none", "unknown", "")

#: Fallback enum (mirrors ``config/trigger_evidence.yaml``).
FALLBACK_TYPES: Tuple[str, ...] = tuple(EVIDENCE_TYPE_IDS)

#: Fallback corroborating kinds beyond advisory prose.
FALLBACK_CORROBORATING: Tuple[str, ...] = (
    "github_poc",
    "exploitdb_packetstorm",
    "exploitdb",
    "packetstorm",
    "commit",
    "blog",
    "poc",
)

#: Fallback "trusted" (advisory) source kinds.
FALLBACK_TRUSTED: Tuple[str, ...] = (
    "nvd",
    "cna",
    "vendor_advisory",
    "cve_services",
    "cisa_kev",
)

_TRIGGER_RULES: Optional[Dict[str, Any]] = None


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def load_trigger_rules(refresh: bool = False) -> Dict[str, Any]:
    """Load ``config/trigger_evidence.yaml`` (cached).

    Returns ``{}`` when the file cannot be read so callers fall back to the
    literal constants above rather than crashing.
    """
    global _TRIGGER_RULES
    if _TRIGGER_RULES is not None and not refresh:
        return _TRIGGER_RULES
    try:
        cfg = load_config(CONFIG_NAME)
        _TRIGGER_RULES = dict(cfg) if isinstance(cfg, Mapping) else {}
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("cannot load %s: %s", CONFIG_NAME, exc)
        _TRIGGER_RULES = {}
    return _TRIGGER_RULES


def evidence_type_ids(rules: Optional[Mapping[str, Any]] = None) -> List[str]:
    """Closed enum of admissible trigger-evidence type ids."""
    cfg = rules if rules is not None else load_trigger_rules()
    types = cfg.get("types") if isinstance(cfg, Mapping) else None
    ids: List[str] = []
    if isinstance(types, Sequence):
        for entry in types:
            if isinstance(entry, Mapping) and entry.get("id"):
                ids.append(str(entry["id"]))
    return ids or list(FALLBACK_TYPES)


def type_weights(rules: Optional[Mapping[str, Any]] = None) -> Dict[str, float]:
    cfg = rules if rules is not None else load_trigger_rules()
    weights = cfg.get("strength_weights") if isinstance(cfg, Mapping) else None
    if isinstance(weights, Mapping):
        return {str(k): _as_float(v, 0.0) for k, v in weights.items()}
    return {"strong": 1.0, "moderate": 0.6, "weak": 0.25, "none": 0.0}


def corroborating_sources(rules: Optional[Mapping[str, Any]] = None) -> Tuple[str, ...]:
    cfg = rules if rules is not None else load_trigger_rules()
    verification = cfg.get("verification") if isinstance(cfg, Mapping) else None
    if isinstance(verification, Mapping):
        kinds = verification.get("corroborating_evidence_sources")
        if isinstance(kinds, Sequence) and not isinstance(kinds, (str, bytes)):
            return tuple(str(k) for k in kinds)
    return FALLBACK_CORROBORATING


def trusted_sources(rules: Optional[Mapping[str, Any]] = None) -> Tuple[str, ...]:
    cfg = rules if rules is not None else load_trigger_rules()
    verification = cfg.get("verification") if isinstance(cfg, Mapping) else None
    if isinstance(verification, Mapping):
        kinds = verification.get("trusted_evidence_sources")
        if isinstance(kinds, Sequence) and not isinstance(kinds, (str, bytes)):
            return tuple(str(k) for k in kinds)
    return FALLBACK_TRUSTED


# --------------------------------------------------------------------------- #
# tolerant accessors over schema models / mappings
# --------------------------------------------------------------------------- #


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [value]


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _indicators(profile: Any) -> Any:
    ind = _get(profile, "technical_indicators")
    if ind is None:
        ind = _get(profile, "indicators")
    return ind if ind is not None else {}


def _indicator_list(profile: Any, *names: str) -> List[str]:
    ind = _indicators(profile)
    out: List[str] = []
    for name in names:
        for item in _as_list(_get(ind, name, [])):
            text = _text(item)
            if text and text not in out:
                out.append(text)
    return out


def _indicator_text(profile: Any, *names: str) -> str:
    ind = _indicators(profile)
    for name in names:
        value = _text(_get(ind, name, ""))
        if value:
            return value
    return ""


def source_kind(source: Any) -> str:
    kind = _get(source, "kind", "") or _get(source, "type", "")
    kind = _text(kind).lower()
    # tolerate the schema's ``SourceRecord`` which stores ``kind`` directly
    return kind


def source_ok(source: Any) -> bool:
    """A source counts as usable evidence when it was fetched successfully."""
    ok = _get(source, "ok", None)
    if ok is not None:
        return bool(ok)
    status = _text(_get(source, "status", "ok")).lower()
    return status in {"", "ok", "cached", "success"}


# --------------------------------------------------------------------------- #
# rule checks
# --------------------------------------------------------------------------- #


def check_trigger_evidence(
    profile: Any,
    rules: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate every ``verification.require_all`` condition.

    Returns ``{"passed": bool, "checks": {name: bool}, "problems": [str],
    "evidence_type": str, "strength": str}``.
    """
    cfg = rules if rules is not None else load_trigger_rules()
    ids = evidence_type_ids(cfg)
    weights = type_weights(cfg)
    corroborating = set(corroborating_sources(cfg))
    trusted = set(trusted_sources(cfg))

    evidence_type = _text(
        _get(profile, "trigger_evidence_type", None)
        or _indicator_text(profile, "trigger_evidence_type")
        or "unknown"
    ).lower()
    checks: Dict[str, bool] = {}
    problems: List[str] = []

    # 1. enum membership
    checks["type_in_enum"] = evidence_type in set(ids)
    if not checks["type_in_enum"]:
        problems.append(
            "trigger_evidence_type %r not in Table 6 enum" % (evidence_type,)
        )

    # 2. not a sentinel
    checks["type_not_sentinel"] = evidence_type not in SENTINELS
    if not checks["type_not_sentinel"]:
        problems.append("trigger_evidence_type is %r (no admissible evidence)" % evidence_type)

    # 3. concrete endpoint or binary
    endpoints = _indicator_list(profile, "endpoint", "endpoints", "url_path")
    binaries = _indicator_list(
        profile, "vulnerable_binary", "binary_paths", "binary", "binaries"
    )
    parameters = _indicator_list(profile, "parameter", "parameters")
    checks["concrete_artifact"] = bool(endpoints or binaries or parameters)
    if not checks["concrete_artifact"]:
        problems.append("no concrete endpoint/binary/parameter named")

    # 4. corroboration beyond advisory prose
    usable = [s for s in _as_list(_get(profile, "sources", [])) if source_ok(s)]
    kinds = [source_kind(s) for s in usable]
    corroborated = any(k in corroborating for k in kinds)
    checks["corroborated"] = bool(corroborated)
    if not corroborated:
        # allow a source whose kind is unknown but whose url is a PoC/exploit host
        urls = " ".join(_text(_get(s, "url", "")) for s in usable).lower()
        checks["corroborated"] = any(
            token in urls for token in ("exploit", "poc", "packetstorm", "github")
        )
        if not checks["corroborated"]:
            problems.append(
                "no corroborating source beyond advisory prose (kinds=%s)" % (kinds or ["none"],)
            )

    # 5. affected version / range stated
    versions = _indicator_list(profile, "affected_versions", "versions", "firmware_version")
    checks["version_stated"] = bool(versions)
    if not checks["version_stated"]:
        problems.append("affected version/range not stated")

    strength_family = ""
    for entry in cfg.get("types", []) if isinstance(cfg, Mapping) else []:
        if isinstance(entry, Mapping) and _text(entry.get("id", "")).lower() == evidence_type:
            strength_family = _text(entry.get("strength", ""))
            break

    passed = all(checks.values())
    return {
        "passed": bool(passed),
        "checks": checks,
        "problems": problems,
        "evidence_type": evidence_type,
        "strength": strength_family,
        "strength_weight": _as_float(weights.get(strength_family, 0.0), 0.0) if strength_family else 0.0,
        "trusted_sources": [k for k in kinds if k in trusted],
        "corroborating_sources": [k for k in kinds if k in corroborating],
    }


def tree_has_binary(path: str, suffixes: Sequence[str] = (".cgi", ".so", ".bin", "httpd")) -> bool:
    """Small helper used by tests/fixtures: does a path mention a binary?"""
    text = _text(path).lower()
    return any(suffix in text for suffix in suffixes)


# --------------------------------------------------------------------------- #
# result container
# --------------------------------------------------------------------------- #


@dataclass
class VerificationResult:
    """Outcome of verifying one reference profile.

    The object proxies unknown attributes to the underlying (possibly updated)
    profile so callers written against "``verify_profile`` returns a profile"
    keep working::

        result = verify_profile(profile)
        result.status                    # VerificationResult.status
        result.concrete_trigger_evidence_verified
        result.cve_id                    # proxied from the profile
    """

    profile: Any = None
    passed: bool = False
    verified: bool = False
    status: str = INCOMPLETE
    evidence_type: str = "unknown"
    strength: str = ""
    checks: Dict[str, bool] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    manual_review: bool = False

    # -- convenience -------------------------------------------------------
    @property
    def ok(self) -> bool:
        return bool(self.passed)

    @property
    def reasons(self) -> List[str]:
        return list(self.problems)

    @property
    def verified_flag(self) -> bool:
        return bool(self.verified)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cve_id": _text(_get(self.profile, "cve_id", "")),
            "passed": self.passed,
            "verified": self.verified,
            "status": self.status,
            "evidence_type": self.evidence_type,
            "strength": self.strength,
            "checks": dict(self.checks),
            "problems": list(self.problems),
            "notes": list(self.notes),
            "manual_review": self.manual_review,
        }

    def to_dict(self) -> Dict[str, Any]:  # alias
        return self.as_dict()

    def __bool__(self) -> bool:
        return bool(self.passed)

    def __getattr__(self, name: str) -> Any:  # proxy to the profile
        if name.startswith("__"):
            raise AttributeError(name)
        profile = self.__dict__.get("profile")
        if profile is None:
            raise AttributeError(name)
        if isinstance(profile, Mapping):
            if name in profile:
                return profile[name]
            raise AttributeError(name)
        return getattr(profile, name)


# --------------------------------------------------------------------------- #
# core API
# --------------------------------------------------------------------------- #


def _set(obj: Any, name: str, value: Any) -> None:
    """Set an attribute on a pydantic model or a mapping."""
    if isinstance(obj, dict):
        obj[name] = value
        return
    try:
        setattr(obj, name, value)
    except Exception:  # pragma: no cover - frozen models
        pass


def verify_profile(
    profile: Any,
    rules: Optional[Mapping[str, Any]] = None,
    *,
    strict: bool = True,
    require_corroboration: bool = True,
) -> VerificationResult:
    """Apply the Table 6 gate to one reference profile.

    Parameters
    ----------
    profile:
        ``ReferenceProfile`` (pydantic) or a plain mapping with the same keys.
        The object is updated in place (``concrete_trigger_evidence_verified``,
        ``status``, ``verification_notes``) and also returned via
        :attr:`VerificationResult.profile`.
    rules:
        Optional pre-loaded ``trigger_evidence.yaml`` mapping.
    strict:
        When ``False`` only the enum/sentinel checks are enforced (used for
        exploratory profiling runs; never for benchmark-grade profiles).
    require_corroboration:
        When ``False`` the corroboration rule is recorded but does not fail the
        profile (documented deviation for advisory-only CVEs).
    """
    cfg = rules if rules is not None else load_trigger_rules()
    detail = check_trigger_evidence(profile, cfg)

    checks = dict(detail["checks"])
    if not strict:
        checks = {
            "type_in_enum": checks.get("type_in_enum", False),
            "type_not_sentinel": checks.get("type_not_sentinel", False),
        }
    if not require_corroboration:
        checks.pop("corroborated", None)

    passed = bool(checks) and all(checks.values())
    problems = list(detail["problems"])
    if not require_corroboration:
        problems = [p for p in problems if "corroborating" not in p]

    verification_cfg = cfg.get("verification") if isinstance(cfg, Mapping) else None
    on_failure = (
        verification_cfg.get("on_failure") if isinstance(verification_cfg, Mapping) else None
    ) or {}
    failure_status = _text(on_failure.get("action", INCOMPLETE)) or INCOMPLETE
    if failure_status not in {INCOMPLETE, MANUAL_REVIEW}:
        failure_status = INCOMPLETE
    manual_review = failure_status == MANUAL_REVIEW

    status = COMPLETE if passed else failure_status
    notes: List[str] = list(problems)
    if passed:
        notes = [
            "trigger evidence %r verified (strength=%s)"
            % (detail["evidence_type"], detail["strength"] or "unknown")
        ]

    # update the profile in place
    _set(profile, "concrete_trigger_evidence_verified", bool(passed))
    _set(profile, "status", status)
    existing_notes = _as_list(_get(profile, "verification_notes", []))
    merged: List[str] = []
    for note in list(existing_notes) + notes:
        text = _text(note)
        if text and text not in merged:
            merged.append(text)
    _set(profile, "verification_notes", merged)
    if detail.get("evidence_type"):
        _set(profile, "trigger_evidence_type", detail["evidence_type"])
    if hasattr(profile, "verified"):
        _set(profile, "verified", bool(passed))

    result = VerificationResult(
        profile=profile,
        passed=passed,
        verified=passed,
        status=status,
        evidence_type=detail["evidence_type"],
        strength=detail.get("strength", ""),
        checks=checks,
        problems=problems,
        notes=merged,
        manual_review=manual_review,
    )
    log.info(
        "verified profile cve=%s passed=%s status=%s type=%s",
        _text(_get(profile, "cve_id", "")),
        passed,
        status,
        detail["evidence_type"],
    )
    return result


def verify_profiles(
    profiles: Iterable[Any],
    rules: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> List[VerificationResult]:
    """Verify a batch of profiles, returning one result per input."""
    return [verify_profile(p, rules, **kwargs) for p in profiles or []]


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #


def _profile_status(profile: Any) -> str:
    status = _text(_get(profile, "status", "")).lower()
    if status:
        return status
    verified = _get(profile, "concrete_trigger_evidence_verified", None)
    return COMPLETE if verified else INCOMPLETE


def verification_rate(profiles: Any = None) -> Dict[str, Any]:
    """Aggregate verification statistics.

    ``profiles`` may be an iterable of profiles, an iterable of
    :class:`VerificationResult` objects, or a mapping ``{cve_id: profile}``.
    Returns ``{"total", "verified", "incomplete", "manual_review",
    "rate_pct", "verified_ids", "excluded_ids"}``.
    """
    if profiles is None:
        profiles = []
    if isinstance(profiles, Mapping):
        items: List[Any] = list(profiles.values())
    else:
        items = list(profiles)

    total = 0
    verified = 0
    incomplete = 0
    manual = 0
    verified_ids: List[str] = []
    excluded_ids: List[str] = []

    for item in items:
        profile = item.profile if isinstance(item, VerificationResult) else item
        if profile is None:
            continue
        total += 1
        cve_id = _text(_get(profile, "cve_id", ""))
        status = _profile_status(profile)
        flag = _get(profile, "concrete_trigger_evidence_verified", None)
        ok = bool(flag) if flag is not None else status == COMPLETE
        if ok:
            verified += 1
            if cve_id:
                verified_ids.append(cve_id)
        else:
            if status == MANUAL_REVIEW:
                manual += 1
            else:
                incomplete += 1
            if cve_id:
                excluded_ids.append(cve_id)

    rate = round(100.0 * verified / total, 1) if total else 0.0
    return {
        "total": total,
        "verified": verified,
        "incomplete": incomplete,
        "manual_review": manual,
        "rate_pct": rate,
        "verified_ids": verified_ids,
        "excluded_ids": excluded_ids,
    }
