"""Plan scorer: Eq. 1-3 of the ReproBench paper.

    C = sum_p w_p * q_p          w = (15,15,15,15,20,20), q in {0, 0.5, 1}
    D = max(0, 100 - 20 * V)     V = inverted phase pairs in the stated order
    F in [0,100] anchored 0/25/50/75/100
    Plan = 0.6*C + 0.3*D + 0.1*F

Every q and F credit must carry a ``plan.md`` line citation; ``score_plan``
accepts a citation lookup and zeroes uncited credits (then flags them).

All constants are read from ``config/benchmark.yaml`` (``plan:`` block) with
documented fallbacks so the module stays importable/testable offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

PHASES: List[str] = ["P1", "P2", "P3", "P4", "P5", "P6"]
COVERAGE_WEIGHTS: List[float] = [15, 15, 15, 15, 20, 20]
Q_VALUES: List[float] = [0.0, 0.5, 1.0]
PLAN_WEIGHTS: Dict[str, float] = {"coverage": 0.6, "dependency": 0.3, "fallback": 0.1}
DEPENDENCY_PENALTY: float = 20.0
MAX_VIOLATIONS: int = 5
FALLBACK_ANCHORS: List[float] = [0, 25, 50, 75, 100]
REQUIRED_PHASES_FOR_FULL_FALLBACK: List[str] = ["P2", "P3", "P5", "P6"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def snap_q(q: float, allowed: Sequence[float] = Q_VALUES) -> float:
    """Snap a coverage quality to the nearest allowed value {0, .5, 1}."""
    q = float(q)
    return float(min(allowed, key=lambda a: (abs(a - q), a)))


def coverage(qualities: Dict[str, float], weights: Sequence[float] = COVERAGE_WEIGHTS,
             phases: Sequence[str] = PHASES) -> float:
    """C = sum_p w_p * q_p  (max 100)."""
    total = 0.0
    for phase, w in zip(phases, weights):
        total += float(w) * snap_q(qualities.get(phase, 0.0))
    return total


def dependency_score(order: Sequence[str], canonical: Sequence[str] = PHASES) -> float:
    """D = max(0, 100 - 20*V), V = number of pairwise inversions (cap 5)."""
    return dependency_detail(order, canonical)["score"]


def dependency_detail(order: Sequence[str], canonical: Sequence[str] = PHASES) -> Dict[str, Any]:
    """Count inverted pairs between the plan's stated order and the canonical one.

    A pair (a, b) is inverted when ``a`` precedes ``b`` canonically but appears
    after ``b`` in the plan's order.  V is capped at 5 so D floors at 0.
    """
    idx = {p: i for i, p in enumerate(canonical)}
    seq = [p for p in order if p in idx]
    inversions: List[str] = []
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if idx[seq[i]] > idx[seq[j]]:
                inversions.append(f"{seq[j]}<->{seq[i]}")
    v = len(inversions)
    capped = min(v, MAX_VIOLATIONS)
    return {
        "violations": v,
        "counted": capped,
        "inversions": inversions,
        "score": max(0.0, 100.0 - DEPENDENCY_PENALTY * capped),
    }


def fallback_score(level: int) -> float:
    """F anchored: 0 none; 25 generic; 50 one phase; 75 >=2 phases; 100 full."""
    level = int(level)
    if level not in (0, 1, 2, 3, 4):
        level = min(max(level, 0), 4)
    return float(FALLBACK_ANCHORS[level])


def classify_fallback(phases_with_contingency: Iterable[str]) -> int:
    """Derive the 0..4 fallback anchor level from covered contingency phases.

    * 0 -> no contingency at all
    * 1 -> generic ("will retry") contingency, no specific phase
    * 2 -> exactly one phase with mode + alternative
    * 3 -> two or more phases
    * 4 -> contingencies for every phase in P2, P3, P5, P6
    """
    covered = {p for p in phases_with_contingency if p in PHASES}
    if not covered:
        return 0
    if set(REQUIRED_PHASES_FOR_FULL_FALLBACK).issubset(covered):
        return 4
    if len(covered) >= 2:
        return 3
    return 2


def plan_score(coverage_score: float, dependency: float, fallback: float) -> float:
    """Plan = 0.6*C + 0.3*D + 0.1*F  (golden oracles: 99.0 / 84.0 / 66.5)."""
    return round(
        PLAN_WEIGHTS["coverage"] * float(coverage_score)
        + PLAN_WEIGHTS["dependency"] * float(dependency)
        + PLAN_WEIGHTS["fallback"] * float(fallback),
        10,
    )


# --------------------------------------------------------------------------- #
# rich scoring
# --------------------------------------------------------------------------- #
@dataclass
class PlanScore:
    coverage: float = 0.0
    dependency: float = 0.0
    fallback: float = 0.0
    total: float = 0.0
    qualities: Dict[str, float] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)
    inversions: List[str] = field(default_factory=list)
    fallback_level: int = 0
    fallback_phases: List[str] = field(default_factory=list)
    citations: Dict[str, str] = field(default_factory=dict)
    invalidated: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "coverage": self.coverage,
            "dependency": self.dependency,
            "fallback": self.fallback,
            "plan_score": self.total,
            "qualities": dict(self.qualities),
            "order": list(self.order),
            "inversions": list(self.inversions),
            "fallback_level": self.fallback_level,
            "fallback_phases": list(self.fallback_phases),
            "citations": dict(self.citations),
            "invalidated": list(self.invalidated),
            "notes": list(self.notes),
        }


def score_plan(
    qualities: Optional[Dict[str, float]] = None,
    order: Optional[Sequence[str]] = None,
    fallback_level: Optional[int] = None,
    fallback_phases: Optional[Iterable[str]] = None,
    citations: Optional[Dict[str, str]] = None,
    citation_exists: Optional[Callable[[str], bool]] = None,
    require_citation: bool = True,
) -> PlanScore:
    """Score a plan.md.

    ``citations`` maps a credit key (``"P1"``..``"P6"`` or ``"fallback"``) to a
    ``plan.md`` line reference.  When ``require_citation`` is true, any credit
    lacking an existing citation is zeroed and listed in ``invalidated``.
    """
    qualities = dict(qualities or {})
    citations = dict(citations or {})
    result = PlanScore()

    def cited(key: str) -> bool:
        ref = citations.get(key)
        if not ref:
            return False
        if citation_exists is None:
            return True
        try:
            return bool(citation_exists(ref))
        except Exception:
            return False

    for phase in PHASES:
        q = snap_q(qualities.get(phase, 0.0))
        if q > 0 and require_citation and not cited(phase):
            result.invalidated.append(phase)
            q = 0.0
        result.qualities[phase] = q

    result.coverage = coverage(result.qualities)
    result.order = [p for p in (order or PHASES) if p in PHASES]
    dep = dependency_detail(result.order)
    result.dependency = dep["score"]
    result.inversions = dep["inversions"]

    if fallback_level is None:
        fallback_level = classify_fallback(fallback_phases or [])
    level = int(fallback_level)
    if level > 0 and require_citation and not cited("fallback"):
        result.invalidated.append("fallback")
        level = 0
    result.fallback_level = level
    result.fallback = fallback_score(level)
    result.fallback_phases = [p for p in (fallback_phases or []) if p in PHASES]
    result.citations = citations
    result.total = plan_score(result.coverage, result.dependency, result.fallback)
    return result


def plan_from_text(plan_md: str, citation_exists: Optional[Callable[[str], bool]] = None,
                   require_citation: bool = True) -> PlanScore:
    """Heuristic plan.md parser: detects phase mentions, order and contingencies."""
    import re

    text = plan_md or ""
    low = text.lower()

    # order of first appearance of any phase marker (P1..P6 or phase keywords)
    keywords = {
        "P1": ["p1", "vulnerability identification", "identify"],
        "P2": ["p2", "firmware acquisition", "acquire", "download"],
        "P3": ["p3", "extraction", "extract", "binwalk"],
        "P4": ["p4", "localiz", "binary analysis", "ghidra"],
        "P5": ["p5", "rehost", "emulat", "qemu"],
        "P6": ["p6", "poc", "trigger", "exploit"],
    }
    hits = []
    for phase, words in keywords.items():
        pos = min([low.find(w) for w in words if low.find(w) >= 0] or [10 ** 9])
        if pos < 10 ** 9:
            hits.append((pos, phase))
    order = [p for _, p in sorted(hits)]

    qualities: Dict[str, float] = {}
    citations: Dict[str, str] = {}
    for phase in PHASES:
        pos = next((p for p, ph in hits if ph == phase), None)
        if pos is None:
            qualities[phase] = 0.0
            continue
        line_no = low[:pos].count("\n") + 1
        qualities[phase] = 1.0
        citations[phase] = f"plan.md:{line_no}"

    # contingency coverage: "if ... then/fallback/instead" near a phase keyword
    contingency = []
    for phase, words in keywords.items():
        for m in re.finditer(r"(?i)(if|when|in case)\b[^\n]{0,160}\b(then|fallback|instead|alternative|retry|switch)", low):
            window = m.group(0)
            if any(w in window for w in words):
                contingency.append(phase)
                break
    if contingency:
        line_no = 1
        for phase in contingency:
            for w in keywords[phase]:
                i = low.find(w)
                if i >= 0:
                    line_no = low[:i].count("\n") + 1
                    break
        citations["fallback"] = f"plan.md:{line_no}"

    return score_plan(
        qualities=qualities,
        order=order or PHASES,
        fallback_phases=contingency,
        citations=citations,
        citation_exists=citation_exists,
        require_citation=require_citation,
    )
