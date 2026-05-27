#!/usr/bin/env bash
# Run all paper-revision suites sequentially in a single tmux session.
# Each suite runs to completion before the next starts, so all 8 GPUs are
# free for whichever suite is current (vs. all suites racing simultaneously).
#
# Usage:
#   scripts/replicate.sh <mamba-env>
#     e.g. scripts/replicate.sh ml13
#
# Knobs (env vars):
#   WORKERS_PER_GPU=4 scripts/replicate.sh ml13     # back off if OOM
#   SESSION=repl_v2 scripts/replicate.sh ml13       # custom tmux session
#
# Resumable: re-running picks up where wallow left off across all suites.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $(basename "$0") <mamba-env>" >&2
    echo "  e.g. $(basename "$0") ml13" >&2
    exit 2
fi

MAMBA_ENV="$1"
SESSION="${SESSION:-replicate}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-6}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# list of experiments to replicate, in order
experiments=(
  "central"
  "weight_decay_sweep"
  "alpha_sweep"
  "dropout_sweep"
  "lr_sweep"
  "init_scale_sweep"
  "task_add"
  "depth_scaling"
  "heads_sweep"
  "task_mul"
)

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="$LOG_DIR/${SESSION}_${ts}.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already running — attaching."
    exec tmux attach -t "$SESSION"
fi

# Build the in-tmux command: activate env once, then run each suite serially.
# Use `set -e` so a failed suite stops the loop (catch the error before more
# GPU-hours are burned).
ACTIVATE='eval "$(mamba shell hook -s bash)" && mamba activate '"$MAMBA_ENV"
SUITES_QUOTED=$(printf '"%s" ' "${experiments[@]}")
INNER='set -e; '"$ACTIVATE"' && for s in '"$SUITES_QUOTED"'; do echo "===== $(date -u +%FT%TZ) starting $s ====="; gc-dispatch --config "configs/${s}.yaml" --workers-per-gpu '"$WORKERS_PER_GPU"' || { echo "FAILED: $s"; exit 1; }; echo "===== $(date -u +%FT%TZ) finished $s ====="; done; echo "ALL SUITES DONE"; echo; echo "Press Enter to close."; read'

CMD="bash -c '$INNER' 2>&1 | tee -a $LOG_FILE"

tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" "$CMD"
echo "Started tmux session '$SESSION'"
echo "  env:        $MAMBA_ENV"
echo "  workers:    $WORKERS_PER_GPU per GPU"
echo "  log:        $LOG_FILE"
echo "  suites:     ${#experiments[@]} (run sequentially)"
echo
echo "Attach with:   tmux attach -t $SESSION"
echo "Detach with:   Ctrl-b then d"
echo "Kill with:     tmux kill-session -t $SESSION"
