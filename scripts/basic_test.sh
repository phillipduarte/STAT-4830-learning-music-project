#!/usr/bin/env bash
# =============================================================================
# run_test.sh — Smoke test runner for train_mlp.py
#
# Usage:
#   chmod +x run_test.sh          # once, to make it executable
#   ./run_test.sh                 # run with defaults
#   ./run_test.sh --smoke-epochs 20 --out-dir /mnt/storage/outputs
#
# Options:
#   --data-dir  DIR    Path to .npy embedding files  (default: /mnt/storage/embeddings)
#   --out-dir   DIR    Path for outputs/logs         (default: /mnt/storage/outputs)
#   --smoke-epochs N   Epochs for the training check (default: 5)
#   --project   DIR    uv project root (default: same dir as this script)
#   --no-log           Print to stdout only, no log file
#   -h, --help         Show this message
# =============================================================================

set -euo pipefail  # -e: exit on error  -u: error on unset vars  -o pipefail: catch pipe errors

# ── Defaults ──────────────────────────────────────────────────────────────────
DATA_DIR="/mnt/storage/embeddings"
OUT_DIR="/mnt/storage/outputs"
SMOKE_EPOCHS=5
PROJECT=""   # uv project root; defaults to SCRIPT_DIR
NO_LOG=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # always relative to this script
LOG_FILE=""   # set after OUT_DIR is resolved

# ── Colours ───────────────────────────────────────────────────────────────────
# Disabled automatically if output is not a terminal (e.g. when piped to a file)
if [[ -t 1 ]]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
    CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; CYAN=''; BOLD=''; RESET=''
fi

# ── Logging ───────────────────────────────────────────────────────────────────
# log()   — always printed and written to log file
# info()  — cyan informational line
# warn()  — yellow warning (non-fatal)
# die()   — red error, then exit 1

log()  { echo -e "$*" | tee -a "${LOG_FILE:-/dev/null}"; }
info() { log "${CYAN}[INFO]${RESET}  $*"; }
warn() { log "${YELLOW}[WARN]${RESET}  $*"; }
die()  { log "${RED}[ERROR]${RESET} $*"; exit 1; }

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)     DATA_DIR="$2";     shift 2 ;;
        --out-dir)      OUT_DIR="$2";      shift 2 ;;
        --smoke-epochs) SMOKE_EPOCHS="$2"; shift 2 ;;
        --project)      PROJECT="$2";      shift 2 ;;
        --no-log)       NO_LOG=true;       shift   ;;
        -h|--help)
            sed -n '2,15p' "$0" | sed 's/^# \?//'  # print the header comment
            exit 0 ;;
        *) die "Unknown argument: $1. Run with --help for usage." ;;
    esac
done

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$OUT_DIR"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
if [[ "$NO_LOG" == false ]]; then
    LOG_FILE="${OUT_DIR}/smoke_test_${TIMESTAMP}.log"
    # Tee all further stdout+stderr into the log file for the rest of the script
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

# ── Header ────────────────────────────────────────────────────────────────────
log ""
log "${BOLD}============================================================${RESET}"
log "${BOLD} Smoke Test Runner${RESET}"
log "${BOLD}============================================================${RESET}"
log "  Timestamp:    ${TIMESTAMP}"
log "  Script dir:   ${SCRIPT_DIR}"
log "  Data dir:     ${DATA_DIR}"
log "  Output dir:   ${OUT_DIR}"
log "  Smoke epochs: ${SMOKE_EPOCHS}"
log "  uv project:   ${PROJECT:-${SCRIPT_DIR} (default)}"
[[ -n "$LOG_FILE" ]] && log "  Log file:     ${LOG_FILE}"
log "${BOLD}============================================================${RESET}"
log ""

# ── Environment checks ────────────────────────────────────────────────────────
info "Checking environment..."

# uv
UV=$(command -v uv || true)
[[ -z "$UV" ]] && die "uv not found. Install it: https://docs.astral.sh/uv/getting-started/installation/"
UV_VERSION=$("$UV" --version 2>&1)
info "uv: ${UV} (${UV_VERSION})"

# Resolve project root — uv run uses this to find pyproject.toml / .venv
PROJECT="${PROJECT:-${SCRIPT_DIR}}"
[[ -f "${PROJECT}/pyproject.toml" ]] || \
    warn "No pyproject.toml found in ${PROJECT}. uv will fall back to any active venv."



# Required files exist
info "Checking data files..."
for f in embeddings_train.npy embeddings_test.npy labels_train.npy labels_test.npy; do
    [[ -f "${DATA_DIR}/${f}" ]] || die "Missing data file: ${DATA_DIR}/${f}"
done
info "Data files: all present"

# Test script exists
TEST_SCRIPT="${SCRIPT_DIR}/test_mlp.py"
[[ -f "$TEST_SCRIPT" ]] || die "test_mlp.py not found at: ${TEST_SCRIPT}"

# Train script exists (test_mlp.py imports from it)
TRAIN_SCRIPT="${SCRIPT_DIR}/train_mlp.py"
[[ -f "$TRAIN_SCRIPT" ]] || die "train_mlp.py not found at: ${TRAIN_SCRIPT}"

info "Test scripts: found"
log ""

# ── Run the smoke test ────────────────────────────────────────────────────────
info "Launching test_mlp.py..."
log ""

START_TIME=$(date +%s)

# Run test_mlp.py. 
# We deliberately do NOT use 'set -e' suppression here — if Python exits non-zero
# we want to capture it cleanly rather than letting bash abort mid-flow.
set +e
"$UV" run --project "$PROJECT" python "$TEST_SCRIPT" \
    --data-dir     "$DATA_DIR" \
    --out-dir      "$OUT_DIR" \
    --smoke-epochs "$SMOKE_EPOCHS"
EXIT_CODE=$?
set -e

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

# ── Result ────────────────────────────────────────────────────────────────────
log ""
log "${BOLD}============================================================${RESET}"
if [[ $EXIT_CODE -eq 0 ]]; then
    log "${GREEN}${BOLD} ALL CHECKS PASSED${RESET} (${ELAPSED}s)"
    log "${GREEN} Safe to launch a full training job.${RESET}"
else
    log "${RED}${BOLD} SMOKE TEST FAILED${RESET} (exit code ${EXIT_CODE}, ${ELAPSED}s)"
    log "${RED} Fix the errors above before launching a full job.${RESET}"
fi
[[ -n "$LOG_FILE" ]] && log " Log saved: ${LOG_FILE}"
log "${BOLD}============================================================${RESET}"
log ""

exit $EXIT_CODE