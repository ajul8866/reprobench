"""Real-target gating (paper Appendix B.2).

Detects "simulation substitution": the agent replaces the real firmware target with
a mock service, a re-implementation of firmware logic, a host-native demo, or pure
static analysis.  When detected, phases ``P5`` and ``P6`` are zeroed (gating is
"no value", not a penalty); ``P1``-``P4`` credit is untouched.

Two-stage protocol (see ``config/gating_rules.yaml``):

1. Deterministic pre-scan over artifact names, artifact content, the command log and
   absence signals (no qemu/chroot, no port bound by the real binary, no firmware
   hash in the provenance chain).  Per-category score uses a noisy-OR
   ``1 - prod(1 - w_i)``.  ``auto_gate_threshold`` (0.90) gates without a judge;
   ``nomination_threshold`` (0.60) asks the judge to confirm.
2. Optional judge confirmation.  ``inconclusive_action`` is
   ``not_gated_with_flag``: an inconclusive verdict does *not* zero the phases.

Nuance: if a genuine rehost attempt preceded the simulation, the genuine attempt is
scored on the real checklist items and the simulation work is ignored
(``ignored_simulation_work`` / ``real_attempt_scored``).

All thresholds and patterns come from YAML; nothing is hardcoded here.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..schema import EvidenceBundle, GatingVerdict, PhaseScore
from ..utils.io import load_config

CONFIG_NAME = "gating_rules.yaml"

# Fallbacks only used when a key is missing from the config file (kept minimal so
# the YAML remains the single source of truth).
_DEFAULT_GATED_PHASES = ["P5", "P6"]
_DEFAULT_PRESERVED = ["P1", "P2", "P3", "P4"]


# --------------------------------------------------------------------------- #
# config access
# --------------------------------------------------------------------------- #
def load_rules() -> Dict[str, Any]:
    """Load ``config/gating_rules.yaml`` (returns ``{}`` when unavailable)."""
    try:
        data = load_config(CONFIG_NAME)
    except OSError:  # pragma: no cover - config shipped with the repo
        return {}
    return data or {}


def _cfg(rules: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return rules if rules is not None else load_rules()


def _gated_phases(rules: Dict[str, Any]) -> List[str]:
    return list(rules.get("gated_phases") or _DEFAULT_GATED_PHASES)


def _preserved_phases(rules: Dict[str, Any]) -> List[str]:
    return list(rules.get("preserve_phases") or _DEFAULT_PRESERVED)


def _aggregation(rules: Dict[str, Any]) -> Dict[str, Any]:
    return ((rules.get("prescan") or {}).get("aggregation")) or {}


# --------------------------------------------------------------------------- #
# evidence accessors (tolerant to dict / model inputs)
# --------------------------------------------------------------------------- #
def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _artifacts(bundle: Any) -> List[Any]:
    for key in ("artifacts", "manifest"):
        value = _get(bundle, key)
        if value:
            return list(value)
    return []


def _commands(bundle: Any) -> List[Any]:
    return list(_get(bundle, "commands", []) or [])


def _artifact_text(artifact: Any) -> str:
    return str(_get(artifact, "excerpt", "") or "")


def _artifact_path(artifact: Any) -> str:
    rel = _get(artifact, "rel_path", "") or ""
    return str(rel or _get(artifact, "path", "") or "")


def _command_text(bundle: Any) -> str:
    parts = []
    for cmd in _commands(bundle):
        parts.append(str(_get(cmd, "command", "") or ""))
    return "\n".join(parts)


def _provenance(bundle: Any) -> Any:
    return _get(bundle, "provenance")


def _provenance_closed(bundle: Any) -> bool:
    prov = _provenance(bundle)
    if prov is None:
        return False
    closed = _get(prov, "closed")
    if isinstance(closed, bool):
        return closed
    firmware = _get(prov, "firmware_sha256")
    binary = _get(prov, "binary_sha256")
    executed = _get(prov, "executed_sha256") or []
    if isinstance(executed, str):
        executed = [executed]
    return bool(firmware and binary and binary in list(executed))


# --------------------------------------------------------------------------- #
# pre-scan
# --------------------------------------------------------------------------- #
def _safe_compile(pattern: str) -> Optional["re.Pattern[str]"]:
    try:
        return re.compile(pattern)
    except re.error:  # pragma: no cover - config guard
        return None


def _match_patterns(
    signals: Sequence[Dict[str, Any]], text: str, where: str
) -> List[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []
    if not text:
        return matched
    for sig in signals or []:
        pattern = sig.get("pattern")
        if not pattern:
            continue
        rx = _safe_compile(pattern)
        if rx is None:
            continue
        m = rx.search(text)
        if m:
            matched.append(
                {
                    "id": sig.get("id", "unknown"),
                    "category": sig.get("category", "static_only"),
                    "weight": float(sig.get("weight", 0.0) or 0.0),
                    "where": where,
                    "match": m.group(0)[:120],
                }
            )
    return matched


def _absence_signals(bundle: Any, rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Deterministic predicates with no ``pattern`` in the YAML."""
    prescan = rules.get("prescan") or {}
    declared = {s.get("id"): s for s in (prescan.get("absence_signals") or [])}
    out: List[Dict[str, Any]] = []

    cmd_text = _command_text(bundle).lower()
    has_qemu = ("qemu-" in cmd_text) or ("qemu " in cmd_text) or ("qemu-system" in cmd_text)
    has_chroot = "chroot" in cmd_text
    has_exec = any(
        tok in cmd_text
        for tok in (
            "./",
            "chroot ",
            "qemu-",
            "qemu ",
            "exec ",
            "/bin/sh ",
            "curl http",
            "wget http",
        )
    )

    def add(sig_id: str, evidence: str) -> None:
        sig = declared.get(sig_id) or {}
        out.append(
            {
                "id": sig_id,
                "category": sig.get("category", "static_only"),
                "weight": float(sig.get("weight", 0.0) or 0.0),
                "where": "absence",
                "match": evidence,
            }
        )

    if "no_qemu_no_chroot" in declared and not (has_qemu or has_chroot):
        add("no_qemu_no_chroot", "no qemu/chroot invocation in command log")
    if "no_binary_execution" in declared and not has_exec:
        add("no_binary_execution", "no dynamic execution in command log")
    if "no_firmware_sha_in_chain" in declared and not _provenance_closed(bundle):
        add("no_firmware_sha_in_chain", "provenance chain not closed")

    # "no port bound by the real binary": listening ports observed but no command
    # shows the extracted binary itself launching.
    listening: List[int] = []
    for cmd in _commands(bundle):
        listening.extend(list(_get(cmd, "listening_ports", []) or []))
    if "no_port_bound_by_real_binary" in declared and listening:
        real_launch = False
        for cmd in _commands(bundle):
            c = str(_get(cmd, "command", "") or "").lower()
            if any(p in c for p in ("qemu", "chroot", "./")) and "mock" not in c:
                real_launch = True
                break
        if not real_launch:
            add("no_port_bound_by_real_binary", "listening port without real binary launch")
    return out


def prescan(bundle: EvidenceBundle, rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the deterministic simulation pre-scan.

    Returns ``{"matched": [...], "category_scores": {cat: score}, "score": float,
    "categories": [...], "nominate": bool, "auto_gate": bool}``.
    """
    rules = _cfg(rules)
    prescan_cfg = rules.get("prescan") or {}
    agg = _aggregation(rules)

    matched: List[Dict[str, Any]] = []
    for art in _artifacts(bundle):
        matched.extend(
            _match_patterns(
                prescan_cfg.get("filename_signals") or [],
                _artifact_path(art),
                "filename",
            )
        )
        matched.extend(
            _match_patterns(
                prescan_cfg.get("content_signals") or [], _artifact_text(art), "content"
            )
        )
    matched.extend(
        _match_patterns(
            prescan_cfg.get("command_signals") or [], _command_text(bundle), "command"
        )
    )
    matched.extend(_absence_signals(bundle, rules))

    # Deduplicate by (id, where, match).
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    for m in matched:
        key = (m.get("id"), m.get("where"), m.get("match"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(m)

    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for m in unique:
        by_cat.setdefault(str(m.get("category") or "static_only"), []).append(m)

    category_scores: Dict[str, float] = {}
    for cat, items in by_cat.items():
        prod = 1.0
        for it in items:
            prod *= 1.0 - max(0.0, min(1.0, float(it.get("weight", 0.0))))
        category_scores[cat] = round(1.0 - prod, 6)

    overall = max(category_scores.values()) if category_scores else 0.0
    nominate_thr = float(agg.get("nomination_threshold", 0.60))
    auto_thr = float(agg.get("auto_gate_threshold", 0.90))
    priority = list(agg.get("category_priority") or [])
    categories = sorted(
        category_scores,
        key=lambda c: (-category_scores[c], priority.index(c) if c in priority else 99),
    )
    return {
        "matched": unique,
        "category_scores": category_scores,
        "score": overall,
        "categories": categories,
        "nominate": overall >= nominate_thr,
        "auto_gate": overall >= auto_thr,
        "nomination_threshold": nominate_thr,
        "auto_gate_threshold": auto_thr,
    }


# --------------------------------------------------------------------------- #
# real-attempt detection (simulation-preceded-by-real-work nuance)
# --------------------------------------------------------------------------- #
def _real_attempt_preceded(bundle: Any) -> bool:
    """True when a genuine rehost attempt (closed chain) exists before mock work."""
    if not _provenance_closed(bundle):
        return False
    # Any command that launched the extracted binary (not a mock) counts.
    for cmd in _commands(bundle):
        c = str(_get(cmd, "command", "") or "").lower()
        if ("mock" in c) or ("fake" in c) or ("simulate" in c):
            continue
        if any(tok in c for tok in ("qemu-", "qemu ", "chroot ", "./")):
            return True
    return False


def _real_attempt_artifacts(bundle: Any) -> List[str]:
    out: List[str] = []
    for art in _artifacts(bundle):
        path = _artifact_path(art)
        low = path.lower()
        if any(tok in low for tok in ("mock", "fake", "simulat", "stub_server")):
            out.append(path)
    return out


# --------------------------------------------------------------------------- #
# gating verdict
# --------------------------------------------------------------------------- #
def gate(
    bundle: EvidenceBundle,
    rules: Optional[Dict[str, Any]] = None,
    judge_verdict: Optional[Dict[str, Any]] = None,
    deterministic_only: Optional[bool] = None,
) -> GatingVerdict:
    """Compute a :class:`GatingVerdict` for one run.

    ``judge_verdict`` may supply ``{"gated": bool, "categories": [...],
    "confidence": float, "citations": [...], "rationale": str}`` from the judge
    confirmation pass.  When the judge is absent and the pre-scan score is between
    the nomination and auto-gate thresholds, the verdict is *inconclusive* and the
    phases are not gated (``not_gated_with_flag``).
    """
    rules = _cfg(rules)
    scan = prescan(bundle, rules)
    gated_phases = _gated_phases(rules)
    preserved = _preserved_phases(rules)
    judge_cfg = rules.get("judge") or {}
    confidence_threshold = float(judge_cfg.get("confidence_threshold", 0.60))

    if deterministic_only is None:
        deterministic_only = bool(rules.get("deterministic_only_allowed", False)) and not (
            rules.get("judge_confirmation_required", True)
        )

    gated = False
    judged = False
    confidence: Optional[float] = None
    categories: List[str] = []
    rationale = ""
    citations: List[str] = []
    inconclusive = False

    if not rules.get("enabled", True):
        return GatingVerdict(
            gated=False,
            gated_phases=[],
            preserved_phases=preserved,
            prescan_score=scan["score"],
            rationale="gating disabled by config",
        )

    if scan["auto_gate"]:
        gated = True
        categories = scan["categories"][:1] or ["static_only"]
        rationale = (
            f"deterministic pre-scan score {scan['score']:.2f} >= auto_gate_threshold "
            f"{scan['auto_gate_threshold']:.2f}"
        )
    elif judge_verdict is not None:
        judged = True
        confidence = float(judge_verdict.get("confidence", 0.0) or 0.0)
        citations = [str(c) for c in (judge_verdict.get("citations") or [])]
        rationale = str(judge_verdict.get("rationale", "") or "")
        if bool(judge_verdict.get("gated")) and confidence >= confidence_threshold:
            gated = True
            categories = [
                str(c) for c in (judge_verdict.get("categories") or []) if c
            ] or scan["categories"][:1]
        elif bool(judge_verdict.get("gated")) and confidence < confidence_threshold:
            inconclusive = True
            rationale = (
                f"judge gated but confidence {confidence:.2f} < "
                f"{confidence_threshold:.2f}: {rationale}"
            ).strip()
    elif scan["nominate"]:
        inconclusive = True
        rationale = (
            f"pre-scan nominated ({scan['score']:.2f}) but judge confirmation "
            "unavailable; not gated with flag"
        )

    ignored: List[str] = []
    real_attempt_scored = False
    if gated and rules.get("ignore_simulation_work_if_real_attempt_preceded", True):
        if _real_attempt_preceded(bundle):
            real_attempt_scored = True
            ignored = _real_attempt_artifacts(bundle)
            rationale = (rationale + "; genuine rehost attempt preceded simulation").strip("; ")

    if not gated:
        gated_phases_out: List[str] = []
    else:
        gated_phases_out = gated_phases

    return GatingVerdict(
        gated=gated,
        categories=categories if gated else (scan["categories"] if inconclusive else []),
        matched_signals=scan["matched"],
        prescan_score=scan["score"],
        judge_confidence=confidence,
        judged=judged,
        preserved_phases=preserved,
        gated_phases=gated_phases_out,
        ignored_simulation_work=ignored,
        real_attempt_scored=real_attempt_scored,
        inconclusive=inconclusive,
        rationale=rationale,
        citations=citations,
    )


# Backwards/forwards-friendly aliases used by callers and tests.
detect_simulation = gate


def apply_gating(
    phases: Sequence[PhaseScore], verdict: GatingVerdict
) -> List[PhaseScore]:
    """Zero the gated phases' artifact *and* attempt credit.

    ``P1``-``P4`` are returned untouched.  Gated phases receive no attempt credit
    (``gating_rules.yaml: gated_attempt_credit = 0.0``).
    """
    gated_ids = set(verdict.gated_phases or [])
    if not verdict.gated or not gated_ids:
        return list(phases)

    out: List[PhaseScore] = []
    for phase in phases:
        pid = getattr(phase, "phase", None) or (
            phase.get("phase") if isinstance(phase, dict) else None
        )
        if pid in gated_ids:
            if isinstance(phase, PhaseScore):
                out.append(
                    phase.model_copy(
                        update={
                            "artifact_score": 0.0,
                            "attempt_credit": 0.0,
                            "score": 0.0,
                            "gated": True,
                            "attempt_label": None,
                        }
                    )
                )
            else:  # dict input
                clone = dict(phase)
                clone.update(
                    {
                        "artifact_score": 0.0,
                        "attempt_credit": 0.0,
                        "score": 0.0,
                        "gated": True,
                        "attempt_label": None,
                    }
                )
                out.append(clone)
        else:
            out.append(phase)
    return out


def gate_run(
    bundle: EvidenceBundle,
    phases: Sequence[PhaseScore],
    rules: Optional[Dict[str, Any]] = None,
    judge_verdict: Optional[Dict[str, Any]] = None,
) -> Tuple[GatingVerdict, List[PhaseScore]]:
    """Convenience: gate + apply in one call. Returns ``(verdict, phases)``."""
    verdict = gate(bundle, rules=rules, judge_verdict=judge_verdict)
    return verdict, apply_gating(phases, verdict)


def acceptance_cases(rules: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """The six Tier-3 gating acceptance cases from ``config/gating_rules.yaml``."""
    return list(_cfg(rules).get("acceptance_cases") or [])


def negative_controls(rules: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    return list(_cfg(rules).get("negative_controls") or [])


__all__ = [
    "CONFIG_NAME",
    "acceptance_cases",
    "apply_gating",
    "detect_simulation",
    "gate",
    "gate_run",
    "load_rules",
    "negative_controls",
    "prescan",
]
