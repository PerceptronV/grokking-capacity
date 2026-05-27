#!/usr/bin/env bash
# Launch a gc-dispatch suite inside a tmux session, with a specific mamba
# environment activated. The session is named after the suite so each one runs
# in its own session and `tmux ls` gives you a roster of in-flight sweeps.
#
# Defaults to 6 workers/GPU = 48 parallel workers on 8x A100 (above the
# conservative auto-allocator default of 3 — these models are tiny enough that
# an A100 is mostly bottlenecked on launch overhead, not compute).
#
# Usage:
#   scripts/run_suite.sh <suite> <mamba-env>
#     e.g. scripts/run_suite.sh central main
#     e.g. scripts/run_suite.sh configs/depth_scaling.yaml main
#   tmux attach -t <suite>                  # reattach later
#   tmux kill-session -t <suite>            # cancel
#
# Knobs (env vars):
#   WORKERS_PER_GPU=4 scripts/run_suite.sh central main     # back off if OOM
#   SESSION=central_v2 scripts/run_suite.sh central main    # custom session
#   FORCE=1 scripts/run_suite.sh central main               # add --force
#
# Resumable: re-running just picks up where wallow left off — already-completed
# combos are skipped.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $(basename "$0") <suite> <mamba-env>" >&2
    echo "  e.g. $(basename "$0") central main" >&2
    echo "  e.g. $(basename "$0") configs/depth_scaling.yaml main" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Resolve the config path — accept either "central" or "configs/central.yaml".
arg="$1"
MAMBA_ENV="$2"
if [[ "$arg" == *.yaml || "$arg" == *.yml ]]; then
    CONFIG="$arg"
else
    CONFIG="configs/${arg}.yaml"
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi

# Session name: derived from the config basename unless the user overrode it.
SESSION="${SESSION:-$(basename "$CONFIG" .yaml)}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-6}"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="$LOG_DIR/${SESSION}_${ts}.log"

# Reattach if the session is already alive (resumes ongoing work).
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' exists — attaching."
    exec tmux attach -t "$SESSION"
fi

EXTRA=()
if [[ "${FORCE:-0}" == "1" ]]; then
    EXTRA+=(--force)
fi

# Activate the requested mamba env inside tmux. tmux spawns a fresh shell
# that sources ~/.bashrc, which can clobber any inherited PATH; explicit
# activation guarantees the right Python is on PATH.
ACTIVATE="eval \"\$(mamba shell hook -s bash)\" && mamba activate '$MAMBA_ENV' && "

CMD="${ACTIVATE}gc-dispatch --config $CONFIG --workers-per-gpu $WORKERS_PER_GPU ${EXTRA[*]} 2>&1 | tee -a $LOG_FILE"

tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" "$CMD; echo; echo 'Done. Press Enter to close.'; read"
echo "Started tmux session '$SESSION'"
echo "  config:    $CONFIG"
echo "  env:       $MAMBA_ENV"
echo "  workers:   $WORKERS_PER_GPU per GPU"
echo "  log:       $LOG_FILE"
echo
echo "Attach with:   tmux attach -t $SESSION"
echo "Detach with:   Ctrl-b then d"
echo "Kill with:     tmux kill-session -t $SESSION"
