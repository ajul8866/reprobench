"""ReproBench command line interface.

Entry point for the whole measurement instrument:

    reprobench mine      -- mine candidate CVEs from NVD (or replay offline pool)
    reprobench select    -- apply filters F1/F2/F3 and stratify the 30-CVE set
    reprobench profile   -- build reference profiles (fetch/clean/distill/verify)
    reprobench run       -- execute one (CVE, model, trial) run in the harness
    reprobench score     -- two-pass evidence-grounded scoring of finished runs
    reprobench analyze   -- emit Table 2 / Figure 4 / Figure 5 / Table 11 reports
    reprobench verify    -- dataset + config invariants and the claims ledger

Every heavy import happens *inside* the command body so that the CLI can be
imported (and `--help` printed) with only typer installed; the offline
self-test therefore never needs network access or model credentials.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="ReproBench: CVE-only IoT firmware vulnerability reproduction benchmark.",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _echo_json(obj) -> None:
    typer.echo(json.dumps(obj, indent=2, sort_keys=True, default=str))


def _fail(message: str, code: int = 2) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


def _ok(message: str) -> None:
    typer.secho(message, fg=typer.colors.GREEN)


# ---------------------------------------------------------------------------
# mine
# ---------------------------------------------------------------------------
@app.command()
def mine(
    out: Path = typer.Option(Path("data/candidate_pool.jsonl"), help="Candidate pool JSONL output."),
    offline: bool = typer.Option(False, help="Replay the pinned offline pool; never touch NVD."),
    limit: Optional[int] = typer.Option(None, help="Maximum candidates to keep."),
    years: str = typer.Option("2020-2026", help="Publication year window (pubStartDate/pubEndDate)."),
    api_key_env: str = typer.Option("NVD_API_KEY", help="Environment variable holding the NVD key."),
) -> None:
    """Mine candidate CVEs (NVD 2.0 API, paginated at 2000/page) by CWE + year."""
    from .dataset.mine import mine_candidates

    start, _, end = years.partition("-")
    try:
        rows = mine_candidates(
            out_path=out,
            offline=offline,
            limit=limit,
            year_start=int(start) if start else None,
            year_end=int(end) if end else None,
            api_key_env=api_key_env,
        )
    except Exception as exc:  # pragma: no cover - network path
        _fail(f"mine failed: {exc}")
    _ok(f"wrote {len(rows)} candidates to {out}")
    _echo_json({"candidates": len(rows), "offline": offline, "out": str(out)})


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------
@app.command()
def select(
    pool: Path = typer.Option(Path("data/candidate_pool.jsonl"), help="Candidate pool JSONL."),
    log: Path = typer.Option(Path("data/candidates/selection_log.jsonl"), help="Selection log JSONL."),
    per_class: int = typer.Option(10, help="Stratified sample size per vulnerability class."),
    out: Optional[Path] = typer.Option(None, help="Optional filtered pool output."),
) -> None:
    """Apply F1/F2/F3 filters and stratify the released 30-CVE evaluation set."""
    from .dataset.filters import select_candidates

    try:
        result = select_candidates(pool_path=pool, log_path=log, per_class=per_class, out_path=out)
    except Exception as exc:
        _fail(f"select failed: {exc}")
    _ok(f"accepted {result.get('accepted', 0)} of {result.get('evaluated', 0)} candidates")
    _echo_json(result)


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------
@app.command()
def profile(
    cves: Path = typer.Option(Path("data/cves.yaml"), help="Dataset YAML (Table 3)."),
    out: Path = typer.Option(Path("data/profiles"), help="Profile directory."),
    offline: bool = typer.Option(False, help="Use the content-addressed cache only."),
    limit: Optional[int] = typer.Option(None, help="Only build the first N CVEs."),
    refresh: bool = typer.Option(False, help="Ignore cached fetches and re-download."),
) -> None:
    """Build reference profiles (never mounted into the agent container)."""
    from .dataset.load import load_cves
    from .profile.distill import build_profile

    entries = load_cves(cves)
    if limit is not None:
        entries = entries[:limit]
    built, excluded = [], []
    for entry in entries:
        try:
            prof = build_profile(entry, out_dir=out, offline=offline, refresh=refresh)
        except Exception as exc:  # pragma: no cover - network path
            excluded.append({"cve_id": getattr(entry, "id", str(entry)), "error": str(exc)})
            continue
        target = built if not prof.excluded else excluded
        target.append(
            {
                "cve_id": prof.cve_id,
                "status": prof.status,
                "verified": prof.concrete_trigger_evidence_verified,
            }
        )
    _echo_json({"built": built, "excluded": excluded, "out": str(out)})
    _ok(f"{len(built)} profiles built, {len(excluded)} excluded")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
@app.command()
def run(
    cve: str = typer.Option(..., help="CVE identifier, e.g. CVE-2023-1389."),
    model: str = typer.Option("glm-5.2", help="Model alias from config/models.yaml."),
    trial: int = typer.Option(1, help="Trial index (1..3)."),
    adapter: str = typer.Option("replay", help="opencode | openai_compat | replay."),
    runs_dir: Path = typer.Option(Path("results/runs"), help="Run output root."),
    fixtures: Optional[Path] = typer.Option(None, help="Replay workspace fixture (tests only)."),
    budget_usd: float = typer.Option(5.0, help="Per-run cost cap in USD."),
    max_hours: float = typer.Option(24.0, help="Per-run wall clock cap."),
) -> None:
    """Run one (CVE, model, trial) trial inside the sandboxed harness."""
    from .agent.runner import RunConfig, run_trial

    cfg = RunConfig(
        cve_id=cve,
        model=model,
        trial=trial,
        adapter=adapter,
        runs_dir=runs_dir,
        fixture_dir=fixtures,
        budget_usd=budget_usd,
        max_hours=max_hours,
    )
    try:
        result = run_trial(cfg)
    except Exception as exc:
        _fail(f"run failed: {exc}")
    _ok(f"run {result.get('run_id')} finished ({result.get('terminated_by')})")
    _echo_json(result)


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------
@app.command()
def score(
    runs: Path = typer.Option(Path("results/runs"), help="Run directory root."),
    out: Path = typer.Option(Path("results/scores"), help="Score JSON output directory."),
    judge: str = typer.Option("heuristic", help="heuristic | llm."),
    judge_model: Optional[str] = typer.Option(None, help="Judge model alias (llm only)."),
    repeats: int = typer.Option(1, help="Judge repeats (median taken when >1)."),
    profiles: Path = typer.Option(Path("data/profiles"), help="Reference profile root."),
    strict_citations: bool = typer.Option(True, help="Invalidate credits without a citation path."),
) -> None:
    """Two-pass scoring: artifacts first, attempts only for zero non-infra phases."""
    from .scoring.judge import score_run_directory

    try:
        summary = score_run_directory(
            runs_dir=runs,
            out_dir=out,
            judge=judge,
            judge_model=judge_model,
            repeats=repeats,
            profiles_dir=profiles,
            strict_citations=strict_citations,
        )
    except Exception as exc:
        _fail(f"score failed: {exc}")
    _ok(f"scored {summary.get('scored', 0)} runs -> {out}")
    _echo_json(summary)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------
@app.command()
def analyze(
    scores: Path = typer.Option(Path("results/scores"), help="Score JSON directory."),
    out: Path = typer.Option(Path("results/reports"), help="Report output directory."),
    label: str = typer.Option("fixture-derived", help="Provenance label stamped on every report."),
    best_by: str = typer.Option("independent", help="independent | same_run best-of-three."),
) -> None:
    """Emit Table 2, Figure 4, Figure 5, failure distribution and Table 11."""
    from .analysis.tables import build_reports

    try:
        written = build_reports(scores_dir=scores, out_dir=out, label=label, best_by=best_by)
    except Exception as exc:
        _fail(f"analyze failed: {exc}")
    _ok(f"wrote {len(written)} report files to {out}")
    _echo_json({"reports": [str(p) for p in written]})


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------
@app.command()
def verify(
    dataset: Path = typer.Option(Path("data/cves.yaml"), help="Dataset YAML to validate."),
    reports: Path = typer.Option(Path("results/reports"), help="Where claims.md is written."),
    scores: Optional[Path] = typer.Option(None, help="Optional scores dir for run-dependent claims."),
) -> None:
    """Validate dataset/config invariants and emit the claims ledger."""
    from .dataset.load import validate_dataset
    from .analysis.tables import write_claims

    problems = validate_dataset(dataset)
    ledger = write_claims(out_dir=reports, dataset_path=dataset, scores_dir=scores)
    if problems:
        _echo_json({"ok": False, "problems": problems})
        _fail("dataset invariants violated", code=1)
    _ok("dataset invariants hold")
    _echo_json({"ok": True, "claims": len(ledger.get("claims", []))})


def main() -> None:  # pragma: no cover - console entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
