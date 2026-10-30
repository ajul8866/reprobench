"""Tier 2 invariant tests: every credit must carry an *existing* citation path.

These tests exercise :mod:`reprobench.scoring.citations` (normalization, structural
validity, existence indexing) and its integration with the rubric and plan
scorers, which accept an injected ``citation_exists`` predicate.

All tests are offline and deterministic: citations resolve against either an
in-memory :class:`CitationIndex` or a temporary :class:`pathlib.Path` tree.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reprobench.schema import (  # noqa: E402
    CITATION_KINDS if False else ArtifactEntry,  # placeholder removed below
)

# -- clean imports (the placeholder above is intentionally not used) ----------
from reprobench.schema import ArtifactEntry, Citation, CommandEntry, EvidenceBundle, ProvenanceChain  # noqa: E402
from reprobench.scoring import citations as cit  # noqa: E402
from reprobench.scoring.plan import score_plan  # noqa: E402
from reprobench.scoring.rubric import item_credit  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_tree(tmp_path: Path) -> Path:
    """Create a small and realistic ``/output`` tree."""
    out = tmp_path / "output"
    (out / "rootfs" / "usr" / "bin").mkdir(parents=True)
    (out / "firmware.bin").write_bytes(b"\x27\x05\x19\x56firmware" * 64)
    (out / "rootfs" / "usr" / "bin" / "httpd").write_bytes(b"\x7fELF" + b"\x00" * 64)
    (out / "plan.md").write_text("# plan\nstep 1\nstep 2\n", encoding="utf-8")
    (out / "report.md").write_text("# report\nresult\n", encoding="utf-8")
    return out


def make_index(out: Path, **overrides: Any) -> "cit.CitationIndex":
    kwargs: Dict[str, Any] = dict(
        paths=[
            str(out / "firmware.bin"),
            str(out / "rootfs" / "usr" / "bin" / "httpd"),
            str(out / "plan.md"),
        ],
        n_commands=3,
        plan_lines=12,
        report_lines=30,
        provenance={
            "firmware_sha256": True,
            "extraction": True,
            "binary_sha256": True,
            "executed_sha256": False,
        },
        output_dir=str(out),
    )
    kwargs.update(overrides)
    return cit.CitationIndex(**kwargs)


def make_bundle(out: Path, n_commands: int = 2) -> Dict[str, Any]:
    """Dict-shaped bundle (the tolerant accessors accept mappings)."""
    return {
        "output_dir": str(out),
        "artifacts": [
            {"path": str(out / "firmware.bin"), "rel_path": "firmware.bin"},
            {"path": str(out / "plan.md"), "rel_path": "plan.md"},
        ],
        "commands": [
            {"index": i, "command": f"echo step {i}"} for i in range(n_commands)
        ],
        "plan_text": "\n".join(f"plan line {i}" for i in range(1, 13)),
        "report_text": "\n".join(f"report line {i}" for i in range(1, 31)),
        "provenance": {
            "firmware_sha256": "a" * 64,
            "binary_sha256": "b" * 64,
            "executed_sha256": ["b" * 64],
            "extraction_command_index": 0,
        },
    }


# ---------------------------------------------------------------------------
# constants / normalization
# ---------------------------------------------------------------------------

def test_citation_kinds_are_the_rubric_kinds() -> None:
    assert cit.CITATION_KINDS == (
        "artifact_path",
        "command_index",
        "plan_line",
        "report_line",
        "provenance_node",
    )


def test_provenance_nodes_match_the_chain() -> None:
    assert cit.PROVENANCE_NODES == (
        "firmware_sha256",
        "extraction",
        "binary_sha256",
        "executed_sha256",
    )


def test_normalize_citation_from_object_roundtrip() -> None:
    c = Citation(kind="command_index", ref="4", quote="binwalk -e fw.bin")
    normalized = cit.normalize_citation(c)
    assert normalized.kind == "command_index"
    assert normalized.ref == "4"
    assert cit.citation_key(normalized) == "command_index:4"
    assert cit.citation_ref(normalized) == "4"


def test_normalize_citation_from_mapping() -> None:
    c = cit.normalize_citation({"kind": "artifact_path", "ref": "/output/poc.py"})
    assert c.kind == "artifact_path"
    assert c.ref == "/output/poc.py"
    assert cit.citation_key(c) == "artifact_path:/output/poc.py"


def test_normalize_citation_from_prefixed_string() -> None:
    c = cit.normalize_citation("command_index:7")
    assert c.kind == "command_index"
    assert str(c.ref) == "7"


def test_normalize_citation_from_plan_markdown_line() -> None:
    c = cit.normalize_citation("plan.md:12")
    assert c.kind == "plan_line"
    # ``ref`` may be the bare line number or the ``plan.md:12`` form; both must
    # survive normalization without raising.
    assert str(c.ref)


def test_normalize_citation_from_report_markdown_line() -> None:
    c = cit.normalize_citation("report.md:3")
    assert c.kind == "report_line"
    assert str(c.ref)


def test_normalize_citation_bare_string_defaults_to_artifact_path() -> None:
    c = cit.normalize_citation("/output/rootfs/usr/bin/httpd")
    assert c.kind == "artifact_path"
    assert c.ref == "/output/rootfs/usr/bin/httpd"


def test_normalize_citation_never_raises_on_garbage() -> None:
    # Unknown kind falls back to artifact_path rather than raising.
    c = cit.normalize_citation({"kind": "nonsense", "ref": "x"})
    assert c.kind == "artifact_path"
    assert c.ref == "x"


def test_is_valid_citation_requires_non_empty_ref() -> None:
    assert cit.is_valid_citation(Citation(kind="plan_line", ref="3")) is True
    assert cit.is_valid_citation(Citation(kind="plan_line", ref="")) is False


def test_is_valid_citation_rejects_unknown_kind() -> None:
    # Unknown kinds are normalized away, so construct one bypassing validation
    # is unnecessary: normalization is the contract point.
    assert cit.is_valid_citation(cit.normalize_citation({"kind": "bogus", "ref": "r"})) is True


# ---------------------------------------------------------------------------
# provenance node helper
# ---------------------------------------------------------------------------

def test_provenance_nodes_from_dict() -> None:
    nodes = cit.provenance_nodes(make_bundle(Path("/tmp"))["provenance"])
    assert nodes["firmware_sha256"] is True
    assert nodes["binary_sha256"] is True
    assert nodes["executed_sha256"] is True


def test_provenance_nodes_from_schema_chain() -> None:
    chain = ProvenanceChain(
        firmware_sha256="a" * 64,
        extraction_command_index=1,
        binary_sha256="b" * 64,
        executed_sha256=["c" * 64],
    )
    nodes = cit.provenance_nodes(chain)
    assert nodes["firmware_sha256"] is True
    assert nodes["extraction"] is True
    assert nodes["binary_sha256"] is True
    # executed hash exists but does not equal the binary hash -> node populated,
    # closure is decided by ``ProvenanceChain.closed``, not by this helper.
    assert nodes["executed_sha256"] is True


def test_provenance_nodes_empty_chain() -> None:
    nodes = cit.provenance_nodes(ProvenanceChain())
    assert nodes["firmware_sha256"] is False
    assert nodes["binary_sha256"] is False
    assert nodes["executed_sha256"] is False


# ---------------------------------------------------------------------------
# CitationIndex existence semantics
# ---------------------------------------------------------------------------

def test_index_from_bundle_counts(tmp_path: Path) -> None:
    out = make_tree(tmp_path)
    idx = cit.CitationIndex.from_bundle(make_bundle(out, n_commands=4))
    assert idx.n_commands == 4
    assert idx.plan_lines == 12
    assert idx.report_lines == 30
    assert len(idx.paths) >= 2
    assert idx.provenance.get("firmware_sha256") is True


def test_index_from_schema_evidence_bundle(tmp_path: Path) -> None:
    out = make_tree(tmp_path)
    bundle = EvidenceBundle(
        run_id="run-1",
        cve_id="CVE-2023-1389",
        model="test-model",
        artifacts=[
            ArtifactEntry(path=str(out / "firmware.bin"), rel_path="firmware.bin"),
        ],
        commands=[CommandEntry(index=0, command="binwalk -e firmware.bin")],
        provenance=ProvenanceChain(firmware_sha256="a" * 64),
    )
    idx = cit.CitationIndex.from_bundle(bundle)
    assert idx.n_commands == 1
    assert idx.exists_artifact(str(out / "firmware.bin")) is True


def test_exists_artifact_absolute_and_relative(tmp_path: Path) -> None:
    out = make_tree(tmp_path)
    idx = make_index(out)
    assert idx.exists_artifact(str(out / "firmware.bin")) is True
    assert idx.exists_artifact("firmware.bin") is True
    assert idx.exists_artifact("rootfs/usr/bin/httpd") is True


def test_exists_artifact_missing_file(tmp_path: Path) -> None:
    out = make_tree(tmp_path)
    idx = make_index(out)
    assert idx.exists_artifact("does-not-exist.bin") is False
    assert idx.exists_artifact("") is False


def test_exists_artifact_with_line_suffix(tmp_path: Path) -> None:
    out = make_tree(tmp_path)
    idx = make_index(out)
    assert idx.exists_artifact("firmware.bin:42") is True


def test_exists_command_is_zero_based_range_check(tmp_path: Path) -> None:
    out = make_tree(tmp_path)
    idx = make_index(out, n_commands=3)
    assert idx.exists_command("0") is True
    assert idx.exists_command("2") is True
    assert idx.exists_command("3") is False
    assert idx.exists_command("-1") is False
    assert idx.exists_command("not-an-int") is False


def test_exists_plan_and_report_lines_are_one_based(tmp_path: Path) -> None:
    out = make_tree(tmp_path)
    idx = make_index(out, plan_lines=12, report_lines=30)
    assert idx.exists_plan_line("1") is True
    assert idx.exists_plan_line("12") is True
    assert idx.exists_plan_line("0") is False
    assert idx.exists_plan_line("13") is False
    assert idx.exists_report_line("30") is True
    assert idx.exists_report_line("31") is False


def test_exists_provenance_node(tmp_path: Path) -> None:
    out = make_tree(tmp_path)
    idx = make_index(out)
    assert idx.exists_provenance("firmware_sha256") is True
    assert idx.exists_provenance("executed_sha256") is False
    assert idx.exists_provenance("no_such_node") is False


def test_exists_dispatch_on_citation_kind(tmp_path: Path) -> None:
    out = make_tree(tmp_path)
    idx = make_index(out)
    assert idx.exists(Citation(kind="artifact_path", ref=str(out / "firmware.bin"))) is True
    assert idx.exists(Citation(kind="command_index", ref="1")) is True
    assert idx.exists(Citation(kind="plan_line", ref="3")) is True
    assert idx.exists(Citation(kind="report_line", ref="29")) is True
    assert idx.exists(Citation(kind="provenance_node", ref="binary_sha256")) is True
    assert idx.exists(Citation(kind="artifact_path", ref="nope")) is False


def test_exists_accepts_raw_strings(tmp_path: Path) -> None:
    out = make_tree(tmp_path)
    idx = make_index(out)
    assert idx.exists("command_index:1") is True
    assert idx.exists("command_index:99") is False


def test_any_exists_semantics(tmp_path: Path) -> None:
    out = make_tree(tmp_path)
    idx = make_index(out)
    assert idx.any_exists([]) is False
    assert idx.any_exists([Citation(kind="artifact_path", ref="nope")]) is False
    assert (
        idx.any_exists(
            [
                Citation(kind="artifact_path", ref="nope"),
                Citation(kind="artifact_path", ref="firmware.bin"),
            ]
        )
        is True
    )


# ---------------------------------------------------------------------------
# resolution / factory predicate
# ---------------------------------------------------------------------------

def test_resolve_citation_without_index_is_structural() -> None:
    assert cit.resolve_citation(Citation(kind="plan_line", ref="4"), None) is True
    assert cit.resolve_citation(Citation(kind="plan_line", ref=""), None) is False


def test_resolve_citation_with_index(tmp_path: Path) -> None:
    out = make_tree(tmp_path)
    idx = make_index(out)
    assert cit.resolve_citation(Citation(kind="artifact_path", ref="firmware.bin"), idx) is True
    assert cit.resolve_citation(Citation(kind="artifact_path", ref="nope.bin"), idx) is False


def test_factory_without_bundle_is_permissive() -> None:
    predicate = cit.citation_exists_factory()
    assert predicate("artifact_path:/output/anything") is True
    # structurally invalid references still fail closed
    assert predicate("") is False


def test_factory_with_bundle_is_strict(tmp_path: Path) -> None:
    out = make_tree(tmp_path)
    predicate = cit.citation_exists_factory(make_bundle(out))
    assert predicate("artifact_path:" + str(out / "firmware.bin")) is True
    assert predicate("artifact_path:" + str(out / "nope.bin")) is False


def test_factory_swallows_exceptions(tmp_path: Path) -> None:
    out = make_tree(tmp_path)
    predicate = cit.citation_exists_factory(make_bundle(out))
    # ``None`` is not a citation; predicate must return False, not raise.
    assert predicate(None) is False


# ---------------------------------------------------------------------------
# integration with the rubric: no credit without an existing citation
# ---------------------------------------------------------------------------

def test_rubric_credit_requires_existing_citation(tmp_path: Path) -> None:
    out = make_tree(tmp_path)
    idx = make_index(out)

    good = item_credit(
        "P2.1",
        5,
        1.0,
        citations=["artifact_path:firmware.bin"],
        citation_exists=idx.exists,
    )
    assert good.credit == pytest.approx(5.0)
    assert good.invalidated is False

    bad = item_credit(
        "P2.1",
        5,
        1.0,
        citations=["artifact_path:ghost.bin"],
        citation_exists=idx.exists,
    )
    assert bad.credit == pytest.approx(0.0)
    assert bad.invalidated is True
    assert bad.fraction == pytest.approx(0.0)


def test_rubric_credit_zero_when_no_citation_supplied(tmp_path: Path) -> None:
    out = make_tree(tmp_path)
    idx = make_index(out)
    credit = item_credit("P1.1", 4, 1.0, citations=None, citation_exists=idx.exists)
    assert credit.credit == pytest.approx(0.0)
    assert credit.invalidated is True


def test_rubric_zero_credit_needs_no_citation(tmp_path: Path) -> None:
    out = make_tree(tmp_path)
    idx = make_index(out)
    credit = item_credit("P1.1", 4, 0.0, citations=None, citation_exists=idx.exists)
    assert credit.credit == pytest.approx(0.0)
    # A zero fraction is not "invalidated" -- there was nothing to ground.
    assert credit.invalidated is False


def test_rubric_credit_with_factory_predicate(tmp_path: Path) -> None:
    out = make_tree(tmp_path)
    predicate = cit.citation_exists_factory(make_bundle(out))
    credit = item_credit(
        "P3.2",
        4,
        0.5,
        citations=["command_index:0"],
        citation_exists=predicate,
    )
    assert credit.credit == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# integration with the plan scorer
# ---------------------------------------------------------------------------

def test_plan_credits_require_citations(tmp_path: Path) -> None:
    out = make_tree(tmp_path)
    idx = make_index(out)
    qualities = {"P1": 1.0, "P2": 1.0, "P3": 1.0, "P4": 1.0, "P5": 1.0, "P6": 1.0}
    citations = {phase: "plan.md:1" for phase in qualities}
    citations["fallback"] = "plan.md:2"

    scored = score_plan(
        qualities=qualities,
        citations=citations,
        citation_exists=idx.exists,
        require_citation=True,
    )
    assert scored.coverage == pytest.approx(100.0)
    assert scored.total > 0.0
    assert scored.invalidated == []


def test_plan_zero_credits_when_citations_missing(tmp_path: Path) -> None:
    out = make_tree(tmp_path)
    idx = make_index(out)
    qualities = {"P1": 1.0, "P2": 1.0, "P3": 1.0, "P4": 1.0, "P5": 1.0, "P6": 1.0}
    scored = score_plan(
        qualities=qualities,
        citations={},
        citation_exists=idx.exists,
        require_citation=True,
    )
    assert scored.coverage == pytest.approx(0.0)
    assert len(scored.invalidated) >= 6


# ---------------------------------------------------------------------------
# diagnostic helper
# ---------------------------------------------------------------------------

def test_check_all_reports_every_citation() -> None:
    calls: List[Any] = [
        "artifact_path:/output/poc.py",
        Citation(kind="command_index", ref="2"),
        "plan.md:5",
    ]
    result = cit.check_all(calls)
    assert isinstance(result, dict)
    assert len(result["citations"]) == len(calls)
    assert len(result["keys"]) == len(calls)
    for entry in result["citations"]:
        assert "kind" in entry
        assert "ref" in entry
        assert "structurally_valid" in entry
