#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/run_matrix.sh - drive the ReproBench run matrix
#
# The paper's evaluation is a full cross product of
#     |CVEs| x |models| x trials = 30 x 5 x 3 = 450 runs
# (see config/benchmark.yaml: trials=3, cve_count=30, model_count=5,
#  total_runs=450).  Each run gets its own throwaway container mounted with
# /output and /trace only; reference profiles are never mounted.
#
# This script is deliberately thin: all orchestration lives in
# reprobench.agent.runner.run_matrix so that the same code path is exercised
# by tests, the offline self-test and a third party attempting the full
# matrix.  Nothing here requires network access; the *adapter* does.
#
# Usage:
#   scripts/run_matrix.sh [options] -- [extra args passed to run_matrix]
#
# Options:
#   -a, --adapter NAME      replay | opencode | openai_compat   (default: replay)
#   -m, --models LIST       comma separated model aliases      (default: config/models.yaml)
#   -t, --trials N          trials per (CVE, model) cell        (default: 3)
#   -d, --runs-dir PATH     output root                          (default: results/runs)
#   -f, --fixtures PATH     fixture root for the replay adapter
#   -b, --budget USD        per-run cost cap                     (default: 5.00)
#   -H, --max-hours N       per-run wall clock cap               (default: 24)
#   -c, --cves PATH         dataset file                         (default: data/cves.yaml)
#   -n, --limit N           cap the number of attempts (pilot runs)
#   -l, --cve LIST          restrict to these CVE ids (comma separated)
#   -A, --cve-class CLASS   restrict to one class: BOF|CMDI|AUTH
#      --model-from-config use config/models.yaml (default when -m omitted)
#      --dry-run           print the planned attempts and exit
#   -q, --quiet             less chatter
#   -h, --help              this text
#
# Examples:
#   # offline smoke test (no network, no API keys)
#   scripts/run_matrix.sh --adapter replay --fixtures tests/fixtures/workspaces --limit 3
#
#   # one pilot run against a live provider
#   scripts/run_matrix.sh --adapter opencode -l CVE-2023-1389 -m glm-5.2 -t 1
#
#   # the full 450-run matrix (needs live endpoints, firmware URLs, ~$329)
#   scripts/run_matrix.sh --adapter opencode --trials 3
#
# Exit codes: 0 ok, 2 usage error, 3 environment error, 4 some attempts failed.
# ---------------------------------------------------------------------------
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ADAPTER="replay"
MODELS=""
TRIALS=""
RUNS_DIR="results/runs"
FIXTURES=""
BUDGET=""
MAX_HOURS=""
CVES_FILE="data/cves.yaml"
LIMIT=""
CVE_LIST=""
CVE_CLASS=""
DRY_RUN=0
QUIET=0
EXTRA_ARGS=()

log() {
  if [ "$QUIET" -eq 0 ]; then
    printf '[run_matrix] %s\n' "$*" >&2
  fi
}

die() {
  printf '[run_matrix] ERROR: %s\n' "$*" >&2
  exit "${2:-2}"
}

usage() {
  sed -n '1,60p' "${BASH_SOURCE[0]}" | sed -n '/^# Usage:/,/^# Exit codes:/p' | sed 's/^# \{0,1\}//'
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    -a|--adapter)     ADAPTER="${2:-}"; shift 2 ;;
    -m|--models)      MODELS="${2:-}"; shift 2 ;;
    -t|--trials)      TRIALS="${2:-}"; shift 2 ;;
    -d|--runs-dir)    RUNS_DIR="${2:-}"; shift 2 ;;
    -f|--fixtures)    FIXTURES="${2:-}"; shift 2 ;;
    -b|--budget)      BUDGET="${2:-}"; shift 2 ;;
    -H|--max-hours)   MAX_HOURS="${2:-}"; shift 2 ;;
    -c|--cves)        CVES_FILE="${2:-}"; shift 2 ;;
    -n|--limit)       LIMIT="${2:-}"; shift 2 ;;
    -l|--cve)         CVE_LIST="${2:-}"; shift 2 ;;
    -A|--cve-class)   CVE_CLASS="${2:-}"; shift 2 ;;
    --model-from-config) MODELS=""; shift ;;
    --dry-run)        DRY_RUN=1; shift ;;
    -q|--quiet)       QUIET=1; shift ;;
    -h|--help)        usage; exit 0 ;;
    --)               shift; EXTRA_ARGS=("$@"); break ;;
    *)                die "unknown option: $1" 2 ;;
  esac
done

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "python interpreter not found: $PYTHON_BIN" 3
[ -f "$CVES_FILE" ] || die "dataset not found: $CVES_FILE" 3
[ -f "$REPO_ROOT/pyproject.toml" ] || die "not a reprobench checkout: $REPO_ROOT" 3

case "$ADAPTER" in
  replay|opencode|openai_compat) ;;
  *) die "unknown adapter '$ADAPTER' (replay|opencode|openai_compat)" 2 ;;
esac

if [ "$ADAPTER" != "replay" ]; then
  if [ -z "${NVD_API_KEY:-}" ]; then
    log "note: NVD_API_KEY unset (optional; raises NVD rate limits)"
  fi
  if [ -z "${OPENAI_API_KEY:-}${ANTHROPIC_API_KEY:-}${OPENAI_BASE_URL:-}" ]; then
    log "note: no model credentials detected; run --dry-run first to sanity check"
  fi
fi

if [ "$ADAPTER" = "opencode" ] && ! command -v opencode >/dev/null 2>&1; then
  log "warning: 'opencode' CLI not on PATH; harness will fall back to openai_compat/host mode"
fi

if [ "$ADAPTER" = "opencode" ] && command -v docker >/dev/null 2>&1; then
  if ! docker image inspect reprobench/agent:24.04 >/dev/null 2>&1; then
    log "warning: image reprobench/agent:24.04 not built; see docker/agent.Dockerfile"
  fi
else
  log "note: docker unavailable or adapter is replay/openai_compat (no per-run container)"
fi

export REPROBENCH_ADAPTER="$ADAPTER"
export REPROBENCH_RUNS_DIR="$RUNS_DIR"
[ -n "$FIXTURES" ] && export REPROBENCH_FIXTURES="$FIXTURES"

log "adapter=$ADAPTER cves=$CVES_FILE runs_dir=$RUNS_DIR"
[ -n "$TRIALS" ] && log "trials=$TRIALS"
[ -n "$MODELS" ] && log "models=$MODELS"
[ -n "$LIMIT" ] && log "limit=$LIMIT"

# The heavy lifting is done by the runner so that the exact same code path is
# covered by tests.  We pass every option through explicitly and let argparse
# in scripts/run_matrix.py report usage problems.
ARGS=(
  "--adapter" "$ADAPTER"
  "--cves" "$CVES_FILE"
  "--runs-dir" "$RUNS_DIR"
)
[ -n "$MODELS" ]    && ARGS+=("--models" "$MODELS")
[ -n "$TRIALS" ]    && ARGS+=("--trials" "$TRIALS")
[ -n "$FIXTURES" ]  && ARGS+=("--fixtures" "$FIXTURES")
[ -n "$BUDGET" ]    && ARGS+=("--budget" "$BUDGET")
[ -n "$MAX_HOURS" ] && ARGS+=("--max-hours" "$MAX_HOURS")
[ -n "$LIMIT" ]     && ARGS+=("--limit" "$LIMIT")
[ -n "$CVE_LIST" ]  && ARGS+=("--cve" "$CVE_LIST")
[ -n "$CVE_CLASS" ] && ARGS+=("--class" "$CVE_CLASS")
[ "$DRY_RUN" -eq 1 ] && ARGS+=("--dry-run")
[ "$QUIET" -eq 1 ]  && ARGS+=("--quiet")
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  ARGS+=("${EXTRA_ARGS[@]}")
fi

exec "$PYTHON_BIN" -m reprobench.scripts.run_matrix "${ARGS[@]}"
