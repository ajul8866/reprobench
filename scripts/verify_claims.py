#!/usr/bin/env python3
"""ReproBench Tier-7 claims ledger.

Emits ``results/reports/claims.md`` (and ``claims.json``) marking each paper
claim as one of:

    verified         - a deterministic, offline artifact/test proves it
    partial          - partially supported (e.g. dataset valid but no runs)
    not_verifiable   - blocked: requires unavailable 2026 model endpoints,
                       live vendor firmware, ARM64 host, or a full 450-run matrix
    deviation        - implemented with a documented departure from the paper

The single hard rule (plan Tier 7): the ledger *refuses* to mark a
run-dependent claim "verified" unless a completed run matrix is detected under
``results/scores``.  This script therefore never asserts that the empirical
Table 2 / Figure 4-5 / Table 11 numbers were reproduced.

Usage
-----
    python scripts/verify_claims.py [--reports results/reports]
                                    [--dataset data/cves.yaml]
                                    [--scores results/scores]
                                    [--json] [--strict]

Exit codes
----------
    0  ledger written (offline; always succeeds unless --strict finds a
       run-dependent claim claimed as verified)
    1  --strict violated: a run-dependent claim is marked verified without runs
    2  dataset invariants failed (when --strict)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# import bootstrap: allow running from a source checkout without pip install
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Claims that may only be "verified" when a completed run matrix exists.
RUN_DEPENDENT = frozenset(
    {
        "C12",
        "C13",
        "C14",
        "C15",
        "C16",
        "table2",
        "figure4",
        "figure5",
        "table11",
        "failure_distribution",
        "divergence",
        "classwise",
        "cost",
    }
)

STATUS_ORDER = ("verified", "partial", "deviation", "not_verifiable")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:  # pragma: no cover - defensive
        return str(path)


def _fallback_claims(dataset_ok: bool, runs_detected: bool) -> Dict[str, Any]:
    """Minimal ledger used when ``analysis.tables`` cannot be imported.

    Mirrors the 16-claim structure of ``analysis.tables.claims_ledger`` closely
    enough that the markdown renderer and the strict check stay meaningful.
    """
    def claim(cid: str, text: str, status: str, evidence: str, notes: str = "") -> Dict[str, Any]:
        return {
            "id": cid,
            "claim": text,
            "status": status,
            "evidence": evidence,
            "notes": notes,
            "run_dependent": cid in RUN_DEPENDENT,
        }

    claims: List[Dict[str, Any]] = [
        claim(
            "C1",
            "Dataset contains 30 CVEs (10 buffer overflow / 10 command injection / "
            "10 auth bypass), 14 vendors, 2020-2026.",
            "verified" if dataset_ok else "partial",
            "data/cves.yaml + tests/test_golden_tables.py::test_dataset_*",
            "" if dataset_ok else "dataset invariants did not validate",
        ),
        claim(
            "C2",
            "Phase maxima sum to 100 with item points 15/15/15/15/20/20.",
            "verified",
            "config/phases.yaml + tests/test_golden_tables.py",
        ),
        claim(
            "C3",
            "Plan = 0.6*C + 0.3*D + 0.1*F with weights (15,15,15,15,20,20).",
            "verified",
            "src/reprobench/scoring/plan.py + tests/test_plan_score.py",
        ),
        claim(
            "C4",
            "Overall = 0.2*Plan + 0.8*Task (golden 99.0 / 59.2 / 24.5).",
            "verified",
            "src/reprobench/scoring/aggregate.py + tests/test_golden_tables.py",
        ),
        claim(
            "C5",
            "Every credit requires an existing citation path, else invalidated.",
            "verified",
            "src/reprobench/scoring/citations.py + tests/test_citations.py",
        ),
        claim(
            "C6",
            "Real-target gating zeroes P5/P6 on simulation substitution.",
            "verified",
            "src/reprobench/scoring/gating.py + tests/test_gating.py (6 cases)",
        ),
        claim(
            "C7",
            "Attempt credit {0, 0.5, 1} replaces (never adds to) a zero artifact phase.",
            "verified",
            "src/reprobench/scoring/attempt.py + tests/test_attempt.py",
        ),
        claim(
            "C8",
            "Agent prompt is Fig. 3 verbatim and leaks no phase/rubric tokens.",
            "verified",
            "config/prompt_template.md + tests/test_no_leak.py",
        ),
        claim(
            "C9",
            "Failure taxonomy has 12 modes; paper text says 13 (documented deviation).",
            "deviation",
            "config/failure_taxonomy.yaml + tests/test_taxonomy.py",
            "Implemented 12 modes + residual 'ambiguous' + extension slot.",
        ),
        claim(
            "C10",
            "Published failure percentages sum to 94.0%; 6.0% residual reported explicitly.",
            "deviation",
            "src/reprobench/analysis/failures.py + tests/test_taxonomy.py",
            "Residual is reported, never redistributed.",
        ),
        claim(
            "C11",
            "Best-of-three uses independent per-metric maxima (best_by switch exposed).",
            "deviation",
            "src/reprobench/scoring/aggregate.py + tests/test_aggregate.py",
            "Table 2 'All' row is a column macro-average (paper rounding).",
        ),
        claim(
            "C12",
            "Table 2 values (AvgPlan/AvgTask/AvgOverall/MaxOverall/P5>=15/P6>=15).",
            "verified" if runs_detected else "not_verifiable",
            "results/reports/table2.md",
            "" if runs_detected else "requires the full 450-run matrix",
        ),
        claim(
            "C13",
            "Figure 4 per-phase normalized panels (all/BOF/CMDI/AUTH).",
            "verified" if runs_detected else "not_verifiable",
            "results/reports/fig4.md",
            "" if runs_detected else "requires the full 450-run matrix",
        ),
        claim(
            "C14",
            "Figure 5 stage-level means (P1 13.9/15, P5 3.7/20, P6 2.8/20).",
            "verified" if runs_detected else "not_verifiable",
            "results/reports/fig5.md",
            "" if runs_detected else "requires the full 450-run matrix",
        ),
        claim(
            "C15",
            "Failure distribution (simulation 45.3%, rehosting 20.7%, acquisition 16.7%).",
            "verified" if runs_detected else "not_verifiable",
            "results/reports/failures.md",
            "" if runs_detected else "requires the full 450-run matrix",
        ),
        claim(
            "C16",
            "Table 11 cost accounting (369/450 traced, 1.19B tokens, $329.36, 88.9h).",
            "verified" if runs_detected else "not_verifiable",
            "results/reports/table11.md",
            "" if runs_detected else "requires the full 450-run matrix",
        ),
    ]
    counts = {s: sum(1 for c in claims if c["status"] == s) for s in STATUS_ORDER}
    return {
        "label": "fixture-derived",
        "claims": claims,
        "counts": counts,
        "runs_detected": runs_detected,
        "run_dependent_verified_without_runs": _strict_violations(claims),
    }


def _strict_violations(claims: List[Dict[str, Any]]) -> List[str]:
    bad: List[str] = []
    for c in claims:
        cid = str(c.get("id", ""))
        status = str(c.get("status", "")).lower()
        if status != "verified":
            continue
        if not (cid in RUN_DEPENDENT or c.get("run_dependent")):
            continue
        if c.get("evidence_runs") or c.get("runs"):
            continue
        bad.append(cid)
    return bad


def _load_ledger(dataset: Path, scores: Path) -> Dict[str, Any]:
    """Build the ledger via ``analysis.tables`` when importable."""
    try:
        from reprobench.analysis import tables as tables_mod  # type: ignore
    except Exception as exc:  # pragma: no cover - degraded path
        print(f"[verify_claims] analysis.tables unavailable ({exc}); using fallback", file=sys.stderr)
        return _fallback_claims(_dataset_ok(dataset), _runs_detected(scores))

    ledger: Optional[Dict[str, Any]] = None
    for name in ("claims_ledger", "build_claims", "claim_ledger"):
        fn = getattr(tables_mod, name, None)
        if not callable(fn):
            continue
        for kwargs in (
            {"dataset_path": dataset, "scores_dir": scores},
            {"dataset_path": dataset},
            {},
        ):
            try:
                ledger = fn(**kwargs)
            except TypeError:
                continue
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[verify_claims] {name} failed: {exc}", file=sys.stderr)
                ledger = None
            if isinstance(ledger, dict) and ledger.get("claims"):
                break
        if isinstance(ledger, dict) and ledger.get("claims"):
            break

    if not isinstance(ledger, dict) or not ledger.get("claims"):
        return _fallback_claims(_dataset_ok(dataset), _runs_detected(scores))

    # never let a run-dependent claim be "verified" without detected runs
    runs = _runs_detected(scores)
    ledger.setdefault("runs_detected", runs)
    for c in ledger["claims"]:
        cid = str(c.get("id", ""))
        if (cid in RUN_DEPENDENT or c.get("run_dependent")) and str(
            c.get("status", "")
        ).lower() == "verified" and not runs:
            c["status"] = "not_verifiable"
            note = "requires the full 450-run matrix"
            existing = str(c.get("notes", "") or "").strip()
            c["notes"] = f"{existing}; {note}" if existing and note not in existing else (existing or note)
    counts: Dict[str, int] = {s: 0 for s in STATUS_ORDER}
    for c in ledger["claims"]:
        counts[str(c.get("status", "partial"))] = counts.get(str(c.get("status", "partial")), 0) + 1
    ledger["counts"] = counts
    ledger["run_dependent_verified_without_runs"] = _strict_violations(ledger["claims"])
    return ledger


def _dataset_ok(path: Path) -> bool:
    try:
        from reprobench.dataset.load import validate_dataset  # type: ignore
    except Exception:  # pragma: no cover
        return False
    try:
        return not validate_dataset(path)
    except Exception:
        return False


def _runs_detected(scores: Path) -> bool:
    """A completed matrix needs >= 450 per-trial score JSON files."""
    if not scores.exists():
        return False
    n = 0
    for p in scores.rglob("*.json"):
        if p.name in ("summary.json", "claims.json"):
            continue
        n += 1
        if n >= 450:
            return True
    return False


def _render_markdown(ledger: Dict[str, Any], label: str) -> str:
    claims = list(ledger.get("claims") or [])
    counts = ledger.get("counts") or {}
    lines = [
        "# ReproBench paper-claims ledger",
        "",
        f"Source label: **{label}**. Values labelled `fixture-derived` are produced from",
        "offline fixtures; they are **not** the paper's published 450-run results.",
        "",
        "| Claim | Status | Evidence pointer | Notes |",
        "| --- | --- | --- | --- |",
    ]
    for c in claims:
        cid = str(c.get("id", "?"))
        text = str(c.get("claim", c.get("text", ""))).replace("\n", " ")
        status = str(c.get("status", "?"))
        ev = str(c.get("evidence", c.get("evidence_pointer", ""))).replace("\n", " ")
        note = str(c.get("notes", "") or "").replace("\n", " ")
        lines.append(f"| {cid}: {text} | `{status}` | {ev} | {note} |")
    lines += [
        "",
        "## Counts",
        "",
        "| Status | Count |",
        "| --- | --- |",
    ]
    for s in STATUS_ORDER:
        lines.append(f"| {s} | {int(counts.get(s, 0))} |")
    lines += [
        "",
        f"Run matrix detected: **{'yes' if ledger.get('runs_detected') else 'no'}**.",
        "",
        "Run-dependent claims (Table 2, Figures 4-5, Table 11, divergence, failure",
        "distribution, per-class means) are refused 'verified' status unless a",
        "completed 450-run matrix is present under `results/scores`.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Emit the ReproBench claims ledger.")
    ap.add_argument("--reports", default="results/reports", help="output directory")
    ap.add_argument("--dataset", default="data/cves.yaml", help="dataset YAML")
    ap.add_argument("--scores", default="results/scores", help="score JSON directory")
    ap.add_argument("--label", default="fixture-derived", help="provenance label")
    ap.add_argument("--json", action="store_true", help="also print the ledger to stdout")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero on dataset failure or a run-dependent 'verified' claim",
    )
    args = ap.parse_args(argv)

    dataset = Path(args.dataset)
    if not dataset.is_absolute():
        dataset = REPO_ROOT / dataset
    scores = Path(args.scores)
    if not scores.is_absolute():
        scores = REPO_ROOT / scores
    out_dir = Path(args.reports)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    ledger = _load_ledger(dataset, scores)
    ledger.setdefault("label", args.label)
    out_dir.mkdir(parents=True, exist_ok=True)

    md = _render_markdown(ledger, args.label)
    md_path = out_dir / "claims.md"
    md_path.write_text(md, encoding="utf-8")
    json_path = out_dir / "claims.json"
    json_path.write_text(json.dumps(ledger, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    counts = ledger.get("counts") or {}
    print(f"claims ledger written: {_rel(md_path)} ({len(ledger.get('claims') or [])} claims)")
    print("  " + ", ".join(f"{s}={int(counts.get(s, 0))}" for s in STATUS_ORDER))
    if args.json:
        print(json.dumps(ledger, indent=2, sort_keys=True, default=str))

    if args.strict:
        if not _dataset_ok(dataset):
            print("[verify_claims] dataset invariants FAILED", file=sys.stderr)
            return 2
        bad = ledger.get("run_dependent_verified_without_runs") or _strict_violations(
            list(ledger.get("claims") or [])
        )
        if bad:
            print(
                "[verify_claims] run-dependent claims marked verified without runs: "
                + ", ".join(bad),
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
