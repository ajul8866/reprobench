"""Pydantic v2 models for ReproBench.

All rubric constants live in ``config/*.yaml``; these models only describe the
shape of the data that flows between the dataset miner, the profile builder,
the agent harness, the scoring engine and the analysis layer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

PHASE_IDS: List[str] = ["P1", "P2", "P3", "P4", "P5", "P6"]
PHASE_MAX: List[float] = [15.0, 15.0, 15.0, 15.0, 20.0, 20.0]
ALLOWED_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
ATTEMPT_VALUES = (0.0, 0.5, 1.0)
EVIDENCE_TYPE_IDS = [
    "command_output_reflected",
    "outbound_connection",
    "crash_signal",
    "memory_corruption_observable",
    "auth_bypass_privileged_access",
    "auth_bypass_token_forgery",
    "file_system_effect",
    "config_or_state_change",
    "log_entry",
    "timing_side_channel",
    "http_status_anomaly",
    "service_restart_or_dos",
    "none",
    "unknown",
]


class RBModel(BaseModel):
    """Base model: extra keys are preserved, strings are stripped."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


# --------------------------------------------------------------------------
# Reference profiles (reference side, never mounted into the agent container)
# --------------------------------------------------------------------------
class TechnicalIndicators(RBModel):
    """Concrete, checkable facts distilled from public sources."""

    vendor: Optional[str] = None
    product: Optional[str] = None
    affected_versions: List[str] = Field(default_factory=list)
    endpoint: Optional[str] = None
    parameter: Optional[str] = None
    vulnerable_binary: Optional[str] = None
    vulnerable_function: Optional[str] = None
    architecture: Optional[str] = None
    endianness: Optional[str] = None
    firmware_urls: List[str] = Field(default_factory=list)
    firmware_sha256: List[str] = Field(default_factory=list)
    extraction_hints: List[str] = Field(default_factory=list)
    poc_hints: List[str] = Field(default_factory=list)

    @property
    def has_concrete_artifact(self) -> bool:
        return bool(self.vulnerable_binary or self.endpoint or self.firmware_urls)


class SourceRecord(RBModel):
    url: str
    kind: str = "blog"  # nvd|cna|vendor_advisory|cve_services|cisa_kev|github_poc|...
    sha256: Optional[str] = None
    fetched_at: Optional[str] = None
    cache_path: Optional[str] = None
    title: Optional[str] = None
    ok: bool = True


class ReferenceProfile(RBModel):
    """Verified ground-truth profile for one CVE (``profile.json``)."""

    cve_id: str
    class_name: Literal["BOF", "CMDI", "AUTH"]
    cwe: List[int] = Field(default_factory=list)
    vendor: str
    products: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    description: str = ""
    technical_indicators: TechnicalIndicators = Field(default_factory=TechnicalIndicators)
    trigger_evidence_type: str = "unknown"
    expected_observables: List[str] = Field(default_factory=list)
    sources: List[SourceRecord] = Field(default_factory=list)
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    concrete_trigger_evidence_verified: bool = False
    status: Literal["complete", "incomplete", "manual_review"] = "incomplete"
    verified: bool = False
    verification_notes: List[str] = Field(default_factory=list)
    profile_version: str = "1.0"

    @field_validator("trigger_evidence_type")
    @classmethod
    def _known_evidence_type(cls, v: str) -> str:
        if v not in EVIDENCE_TYPE_IDS:
            raise ValueError(f"unknown trigger_evidence_type: {v!r}")
        return v

    def to_info_text(self) -> str:
        """Flat ``info.txt`` rendering (human/LLM readable, no JSON)."""
        ti = self.technical_indicators
        rows = [
            f"CVE: {self.cve_id}",
            f"Class: {self.class_name}",
            f"CWE: {', '.join('CWE-' + str(c) for c in self.cwe) or 'unknown'}",
            f"Vendor: {self.vendor}",
            f"Products: {', '.join(self.products) or 'unknown'}",
            f"Affected versions: {', '.join(ti.affected_versions) or 'unknown'}",
            f"Endpoint: {ti.endpoint or 'unknown'}",
            f"Parameter: {ti.parameter or 'unknown'}",
            f"Vulnerable binary: {ti.vulnerable_binary or 'unknown'}",
            f"Vulnerable function: {ti.vulnerable_function or 'unknown'}",
            f"Architecture: {ti.architecture or 'unknown'} ({ti.endianness or 'unknown'})",
            f"Firmware URLs: {', '.join(ti.firmware_urls) or 'unknown'}",
            f"Trigger evidence type: {self.trigger_evidence_type}",
            f"Expected observables: {', '.join(self.expected_observables) or 'unknown'}",
            f"Verified: {self.concrete_trigger_evidence_verified}",
            f"Status: {self.status}",
            "Description: " + " ".join(self.description.split()),
        ]
        return "\n".join(rows) + "\n"


# --------------------------------------------------------------------------
# Evidence bundle (agent side)
# --------------------------------------------------------------------------
class ArtifactEntry(RBModel):
    path: str
    rel_path: str
    size: int = 0
    sha256: str = ""
    magic: str = ""
    mime: str = ""
    kind: str = "other"  # firmware_image|archive|rootfs_dir|elf_binary|script|...
    note: Optional[str] = None


class CommandEntry(RBModel):
    index: int
    command: str
    tool: str = "bash"
    exit_code: Optional[int] = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    network_targets: List[str] = Field(default_factory=list)
    http_status: List[int] = Field(default_factory=list)
    crash_signal: Optional[str] = None
    listening_ports: List[int] = Field(default_factory=list)


class ProvenanceChain(RBModel):
    firmware_sha256: Optional[str] = None
    extraction_command_index: Optional[int] = None
    binary_sha256: Optional[str] = None
    binary_path: Optional[str] = None
    executed_sha256: List[str] = Field(default_factory=list)
    executed_paths: List[str] = Field(default_factory=list)

    @property
    def closed(self) -> bool:
        if not (self.firmware_sha256 and self.binary_sha256):
            return False
        return self.binary_sha256 in self.executed_sha256


class EvidenceBundle(RBModel):
    run_id: str
    cve_id: str
    model: str
    trial: int = 1
    artifacts: List[ArtifactEntry] = Field(default_factory=list)
    commands: List[CommandEntry] = Field(default_factory=list)
    provenance: ProvenanceChain = Field(default_factory=ProvenanceChain)
    manifest_hash: str = ""
    trace_tokens: Dict[str, int] = Field(default_factory=dict)
    trace_cost_usd: float = 0.0
    has_report: bool = False
    has_plan: bool = False
    infra_failure: bool = False
    notes: List[str] = Field(default_factory=list)

    def by_kind(self, kind: str) -> List[ArtifactEntry]:
        return [a for a in self.artifacts if a.kind == kind]

    def artifact(self, rel_path: str) -> Optional[ArtifactEntry]:
        for a in self.artifacts:
            if a.rel_path == rel_path or a.path == rel_path:
                return a
        return None


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
class Citation(RBModel):
    kind: Literal["artifact_path", "command_index", "plan_line", "report_line", "provenance_node"]
    ref: str
    quote: Optional[str] = None

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.ref}"


class ItemCredit(RBModel):
    item_id: str
    points: float
    fraction: float = 0.0
    raw_fraction: Optional[float] = None
    credit: float = 0.0
    citations: List[Citation] = Field(default_factory=list)
    rationale: str = ""
    judge: str = "heuristic"  # heuristic|llm|deterministic
    invalidated: bool = False

    @field_validator("fraction")
    @classmethod
    def _fraction_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("fraction must be within [0, 1]")
        return v


class PhaseScore(RBModel):
    phase: str
    max_points: float
    artifact_score: float = 0.0
    attempt_credit: float = 0.0
    score: float = 0.0
    gated: bool = False
    items: List[ItemCredit] = Field(default_factory=list)
    attempt_label: Optional[str] = None
    attempt_rationale: str = ""
    infrastructure_failure: bool = False
    citations: List[Citation] = Field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.score < 0.5 * self.max_points


class PlanScore(RBModel):
    coverage: float = 0.0
    dependency: float = 0.0
    fallback: float = 0.0
    plan: float = 0.0
    q: List[float] = Field(default_factory=lambda: [0.0] * 6)
    violations: int = 0
    stated_order: List[str] = Field(default_factory=list)
    inverted_pairs: List[List[str]] = Field(default_factory=list)
    fallback_anchor: Optional[int] = None
    citations: List[Citation] = Field(default_factory=list)
    rationale: str = ""


class GatingVerdict(RBModel):
    gated: bool = False
    categories: List[str] = Field(default_factory=list)
    matched_signals: List[Dict[str, Any]] = Field(default_factory=list)
    prescan_score: float = 0.0
    judge_confidence: Optional[float] = None
    judged: bool = False
    preserved_phases: List[str] = Field(default_factory=lambda: ["P1", "P2", "P3", "P4"])
    gated_phases: List[str] = Field(default_factory=lambda: ["P5", "P6"])
    ignored_simulation_work: List[str] = Field(default_factory=list)
    real_attempt_scored: bool = False
    inconclusive: bool = False
    rationale: str = ""
    citations: List[Citation] = Field(default_factory=list)


class FailureAttribution(RBModel):
    mode: Optional[str] = None
    family: Optional[str] = None
    phase: Optional[str] = None
    rationale: str = ""
    first_match_rule: Optional[str] = None
    gated: bool = False


class RunScore(RBModel):
    run_id: str
    cve_id: str
    class_name: Optional[str] = None
    model: str
    trial: int = 1
    phases: List[PhaseScore] = Field(default_factory=list)
    plan: PlanScore = Field(default_factory=PlanScore)
    gating: GatingVerdict = Field(default_factory=GatingVerdict)
    failure: Optional[FailureAttribution] = None
    task_score: float = 0.0
    overall: float = 0.0
    infrastructure_failure: bool = False
    refusal_detected: bool = False
    premature_abandonment: bool = False
    terminated_by: Optional[str] = None
    exit_reason: Optional[str] = None
    judge_model: Optional[str] = None
    prompt_version: Optional[str] = None
    rubric_version: str = "rubric-v1"
    created_at: Optional[str] = None

    def phase(self, phase_id: str) -> Optional[PhaseScore]:
        for p in self.phases:
            if p.phase == phase_id:
                return p
        return None

    def phase_scores(self) -> List[float]:
        return [self.phase(p).score if self.phase(p) else 0.0 for p in PHASE_IDS]


class TrialRecord(RBModel):
    """Per-trial score row persisted under ``results/scores``."""

    cve_id: str
    class_name: Optional[str] = None
    model: str
    trial: int
    run_id: str
    plan_score: float = 0.0
    task_score: float = 0.0
    overall: float = 0.0
    phase_scores: List[float] = Field(default_factory=lambda: [0.0] * 6)
    gated: bool = False
    failure_mode: Optional[str] = None
    cost_usd: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_reasoning: int = 0
    tokens_cache_read: int = 0
    tokens_cache_write: int = 0
    duration_seconds: float = 0.0
    traced: bool = True


class ModelSummary(RBModel):
    model: str
    avg_plan: float = 0.0
    avg_task: float = 0.0
    avg_overall: float = 0.0
    max_overall: float = 0.0
    p5_at_least_15: int = 0
    p6_at_least_15: int = 0
    runs: int = 0
    trials: int = 0


__all__ = [
    "PHASE_IDS",
    "PHASE_MAX",
    "ALLOWED_FRACTIONS",
    "ATTEMPT_VALUES",
    "EVIDENCE_TYPE_IDS",
    "RBModel",
    "TechnicalIndicators",
    "SourceRecord",
    "ReferenceProfile",
    "ArtifactEntry",
    "CommandEntry",
    "ProvenanceChain",
    "EvidenceBundle",
    "Citation",
    "ItemCredit",
    "PhaseScore",
    "PlanScore",
    "GatingVerdict",
    "FailureAttribution",
    "RunScore",
    "TrialRecord",
    "ModelSummary",
]
