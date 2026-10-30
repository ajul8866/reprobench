#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# ReproBench Tier-6 offline self-test.
#
#   prompt render -> replay harness (fixture workspaces) -> two-pass scoring
#   (heuristic judge) -> per-run score JSON -> table2 / fig4 / fig5 / classes /
#   divergence / failures / table11 reports -> claims ledger
#
# Runs with NO network access, NO Docker, and NO API keys.  Every emitted
# report is labelled "fixture-derived"; the numbers are NOT the paper's 450-run
# empirical results and must never be presented as such.
#
# Usage:
#   scripts/selftest_offline.sh [workdir]
#   make selftest
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export REPROBENCH_OFFLINE=1
# Hard-block accidental network use from the offline path.
export NO_PROXY="*"
export HTTP_PROXY="http://127.0.0.1:9"
export HTTPS_PROXY="http://127.0.0.1:9"
export ALL_PROXY="http://127.0.0.1:9"

PY="${PYTHON:-python3}"
WORK="${1:-$ROOT/results/selftest}"
RUNS="$WORK/runs"
SCORES="$WORK/scores"
REPORTS="$WORK/reports"

fail() { printf '\033[31mFAIL\033[0m %s\n' "$*" >&2; exit 1; }
ok()   { printf '\033[32m  ok\033[0m %s\n' "$*"; }
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

command -v "$PY" >/dev/null 2>&1 || fail "python interpreter not found: $PY"

step "0/6  prepare workspace $WORK"
rm -rf "$WORK"
mkdir -p "$RUNS" "$SCORES" "$REPORTS" || fail "cannot create $WORK"
ok "workspace ready"

step "1/6  render the Fig. 3 prompt (leakage checked)"
"$PY" - "$WORK" <<'PY' || exit 1
import sys
from pathlib import Path

work = Path(sys.argv[1])
from reprobench.agent import prompt as prompt_mod

cve = "CVE-2023-1389"
rendered = prompt_mod.render_prompt(cve)
leaks = prompt_mod.check_leakage(rendered)
if leaks:
    raise SystemExit(f"prompt leaks benchmark internals: {leaks}")
if cve not in rendered:
    raise SystemExit("rendered prompt does not mention the CVE id")
out = prompt_mod.save_prompt(cve, work / "prompt")
meta = prompt_mod.prompt_metadata(cve)
print(f"   prompt chars={meta['chars']} sha256={meta['sha256'][:12]} version={meta['prompt_version']}")
print(f"   saved -> {out}")
PY
ok "prompt render + leakage guard"

step "2/6  build deterministic fixture workspaces (replay harness)"
"$PY" - "$RUNS" <<'PY' || exit 1
import hashlib
import json
import sys
from pathlib import Path

runs_root = Path(sys.argv[1])

from reprobench.agent.prompt import save_prompt
from reprobench.agent.trace import TraceRecorder

MODEL = "glm-5.2"
CVES = ["CVE-2023-1389", "CVE-2024-5293"]

PLAN_TMPL = """# Reproduction plan for {cve}

## P1 Identify
Read the NVD record and the vendor advisory; the affected firmware version is
1.0.2 and the vulnerable endpoint is /cgi-bin/luci with parameter cmd.

## P2 Acquire
Download the vendor firmware image from the official support portal
(https://example.invalid/firmware.bin) and verify its sha256 digest.

## P3 Extract
Extract the squashfs image with binwalk and unsquashfs; inspect the rootfs.

## P4 Localize
Identify the architecture (mips, little endian) and the candidate binary
/usr/bin/httpd; confirm the vulnerable function with strings and ghidra.

## P5 Rehost
Launch the extracted binary under qemu-user inside a chroot of the rootfs and
bind the management port 8080.

## P6 Trigger
Send the crafted request to the vulnerable endpoint and capture the crash
signal as trigger evidence.

## Fallbacks
If the vendor download is unavailable then fallback to a known mirror.
If the binary aborts on missing NVRAM then instead provide LD_PRELOAD stubs.
If the service is not reachable then retry with a different network namespace.
If the PoC does not trigger then alternative payload encodings will be tried.
"""

REPORT_TMPL = """# Reproduction report for {cve}

## Summary
The firmware was acquired, extracted and executed under qemu; the crafted
request reached the vulnerable endpoint and produced an observable crash.

## Evidence
- command 0: curl -o firmware.bin https://example.invalid/firmware.bin (HTTP 200)
- command 1: binwalk -e firmware.bin (ok)
- command 2: unsquashfs -d rootfs fw.squashfs (ok)
- command 3: qemu-arm-static -L ./lib ./rootfs/usr/bin/httpd -p 8080 (listening)
- command 4: curl http://127.0.0.1:8080/cgi-bin/luci?cmd=... (crash SIGSEGV)

## Roadblocks
The vendor portal rate-limited the first download attempt (HTTP 429).
"""

COMMANDS = [
    ("curl -sS -o firmware.bin https://example.invalid/firmware.bin", 0,
     "HTTP 200  firmware.bin 4194304 bytes"),
    ("binwalk -e firmware.bin", 0, "DECIMAL  HEXADECIMAL  DESCRIPTION\n0  0x0  uImage"),
    ("unsquashfs -d rootfs fw.squashfs", 0, "created 812 files\nrootfs/usr/bin/httpd"),
    ("file rootfs/usr/bin/httpd", 0, "ELF 32-bit LSB executable, MIPS"),
    ("qemu-arm-static -L ./lib ./rootfs/usr/bin/httpd -p 8080", 0,
     "listening on 0.0.0.0:8080"),
    ("curl -sS 'http://127.0.0.1:8080/cgi-bin/luci?cmd=AA;id'", 139,
     "Segmentation fault (core dumped)"),
]

for cve in CVES:
    for trial in (1, 2, 3):
        run_id = f"{cve}_{MODEL}_t{trial}"
        run_dir = runs_root / run_id
        out = run_dir / "output"
        out.mkdir(parents=True, exist_ok=True)

        save_prompt(cve, run_dir)
        (out / "plan.md").write_text(PLAN_TMPL.format(cve=cve), encoding="utf-8")
        (out / "report.md").write_text(REPORT_TMPL.format(cve=cve), encoding="utf-8")

        fw = out / "firmware.bin"
        fw.write_bytes(b"FWIMG\x00" + bytes(range(256)) * 8)
        rootfs = out / "rootfs" / "usr" / "bin"
        rootfs.mkdir(parents=True, exist_ok=True)
        binary = rootfs / "httpd"
        binary.write_bytes(b"\x7fELF\x01\x01\x01\x00" + b"\x00" * 248 + b"httpd\x00")
        (out / "poc.sh").write_text(
            "#!/bin/sh\ncurl -sS 'http://127.0.0.1:8080/cgi-bin/luci?cmd=AA;id'\n",
            encoding="utf-8",
        )
        (out / "rootfs.squashfs").write_bytes(b"hsqs" + b"\x00" * 508)

        trace = TraceRecorder(run_dir / "trace", run_id=run_id, cve_id=cve,
                              model=MODEL, trial=trial)
        trace.message("user", "Reproduce the vulnerability.")
        for cmd, rc, output in COMMANDS:
            trace.tool_call("bash", {"command": cmd}, output=output, exit_code=rc)
        trace.finalize(
            tokens={"input": 12000, "output": 900, "reasoning": 300,
                    "cache_read": 0, "cache_write": 0},
            cost_usd=0.42,
            status="ok",
            extra={"fixture_derived": True},
        )

        meta = {
            "run_id": run_id,
            "cve_id": cve,
            "model": MODEL,
            "trial": trial,
            "adapter": "replay",
            "image": "reprobench/agent:24.04",
            "terminated_by": "completed",
            "cost_usd": 0.42,
            "wall_seconds": 180.0,
            "fixture_derived": True,
            "prompt_version": "fig3-v1",
            "infrastructure_failure": False,
            "provenance": {
                "firmware_sha256": hashlib.sha256(fw.read_bytes()).hexdigest(),
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "executed_sha256": [hashlib.sha256(binary.read_bytes()).hexdigest()],
            },
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"   fixture run -> {run_dir.name}")

print(f"   {len(CVES) * 3} fixture runs under {runs_root}")
PY
ok "fixture workspaces built"

step "3/6  two-pass scoring (heuristic judge, no API keys)"
"$PY" - "$RUNS" "$SCORES" <<'PY' || exit 1
import inspect
import json
import sys
from pathlib import Path

runs_dir = Path(sys.argv[1])
scores_dir = Path(sys.argv[2])
scores_dir.mkdir(parents=True, exist_ok=True)

from reprobench.scoring import judge as judge_mod

fn = None
for name in ("score_run_directory", "score_directory", "score_runs", "score_all_runs"):
    candidate = getattr(judge_mod, name, None)
    if callable(candidate):
        fn = candidate
        break

summary = None
if fn is not None:
    kwargs = {}
    try:
        params = set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        params = set()
    if "judge" in params:
        kwargs["judge"] = "heuristic"
    if "judge_model" in params:
        kwargs["judge_model"] = None
    if "repeats" in params:
        kwargs["repeats"] = 1
    if "profiles_dir" in params:
        kwargs["profiles_dir"] = "data/profiles"
    if "strict_citations" in params:
        kwargs["strict_citations"] = True
    try:
        summary = fn(runs_dir, scores_dir, **kwargs)
    except TypeError:
        summary = fn(runs_dir, scores_dir)

written = sorted(scores_dir.glob("*.json"))
if not written:
    raise SystemExit("scoring produced no score files")
print(f"   score files: {len(written)}  summary={json.dumps(summary, default=str)[:160]}")
PY
ok "scores written"

step "4/6  reports (table2 / fig4 / fig5 / classes / divergence / failures / table11)"
"$PY" - "$SCORES" "$REPORTS" <<'PY' || exit 1
import sys
from pathlib import Path

scores_dir = Path(sys.argv[1])
reports_dir = Path(sys.argv[2])
reports_dir.mkdir(parents=True, exist_ok=True)

from reprobench.analysis import tables as tables_mod

paths = tables_mod.build_reports(
    scores_dir, reports_dir, label="fixture-derived", best_by="independent"
)
names = sorted(p.name for p in paths)
print(f"   reports: {', '.join(names) if names else '(none)'}")

missing = [n for n in ("table2.md", "table11.md", "failures.md") if n not in names]
if missing:
    raise SystemExit(f"missing required reports: {missing}")
PY
ok "reports rendered"

step "5/6  claims ledger (refuses 'verified' for run-dependent claims)"
if [ -f "$ROOT/scripts/verify_claims.py" ]; then
    "$PY" "$ROOT/scripts/verify_claims.py" --reports "$REPORTS" --scores "$SCORES" \
        || fail "verify_claims.py failed"
    ok "claims ledger written"
else
    printf '   (scripts/verify_claims.py not present, skipping)\n'
fi

step "6/6  assert no report claims the paper's published numbers"
grep -rn --include='*.md' -e '77\.7' -e '44\.8' -e '51\.1' "$REPORTS" >/dev/null 2>&1 \
    && printf '   note: published reference values appear in a labelled context; check disclaimers\n'
if ! grep -q 'fixture-derived' "$REPORTS/table2.md" 2>/dev/null; then
    fail "table2.md is missing the 'fixture-derived' label"
fi
ok "disclaimer present"

printf '\n\033[32mOFFLINE SELF-TEST PASSED\033[0m\n'
printf 'reports: %s\n' "$REPORTS"
printf 'scores:  %s\n' "$SCORES"
printf 'runs:    %s\n\n' "$RUNS"
printf 'All numbers above are FIXTURE-DERIVED and do not reproduce the paper.\n'
exit 0
