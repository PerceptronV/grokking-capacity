#!/usr/bin/env bash
# Run all paper suites, or an explicit subset, sequentially in tmux.
# Usage: scripts/replicate.sh <mamba-env> [suite-or-config ...]
# Examples:
#   scripts/replicate.sh ml13
#   SESSION=learning scripts/replicate.sh ml13 lr_sweep task_mul
# Overrides: SESSION, WORKERS_PER_GPU, GC_WALLOW_DB, LOG_DIR, FORCE=1.
# Keep the default SQLite registry on local disk; merge it after training.
set -euo pipefail

if [[ $# -lt 1 || "$1" == --help || "$1" == -h ]]; then
    echo "Usage: $(basename "$0") <mamba-env> [suite-or-config ...]"
    if [[ $# -lt 1 ]]; then exit 2; fi
    exit 0
fi

MAMBA_ENV="$1"
shift
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
SESSION="${SESSION:-replicate}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-6}"
export GC_WALLOW_DB="${GC_WALLOW_DB:-$HOME/grokking_capacity_runs.db}"

suites=("$@")
if [[ ${#suites[@]} -eq 0 ]]; then
    suites=(central weight_decay_sweep alpha_sweep dropout_sweep lr_sweep
            init_scale_sweep task_add depth_scaling heads_sweep task_mul)
fi
configs=()
for suite in "${suites[@]}"; do
    if [[ "$suite" == *.yaml || "$suite" == *.yml ]]; then
        config="$suite"
    else
        config="configs/${suite}.yaml"
    fi
    if [[ ! -f "$config" ]]; then
        echo "Config not found: $config" >&2
        exit 1
    fi
    configs+=("$config")
done

if tmux has-session -t "$SESSION" 2>/dev/null; then
    exec tmux attach -t "$SESSION"
fi

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${SESSION}_$(date -u +%Y%m%dT%H%M%SZ).log"
printf -v quoted_env '%q' "$MAMBA_ENV"
printf -v quoted_db '%q' "$GC_WALLOW_DB"
inner="set -euo pipefail; export GC_WALLOW_DB=$quoted_db; eval \"\$(mamba shell hook -s bash)\"; mamba activate $quoted_env; "
for config in "${configs[@]}"; do
    dispatch=(gc-dispatch --config "$config" --workers-per-gpu "$WORKERS_PER_GPU")
    if [[ "${FORCE:-0}" == 1 ]]; then dispatch+=(--force); fi
    printf -v job '%q ' "${dispatch[@]}"
    inner+="$job; "
done
printf -v command 'set -o pipefail; bash -c %q 2>&1 | tee -a %q' "$inner" "$LOG_FILE"
tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" "$command"
printf 'Started %s: %s suite(s), %s workers/GPU\nRegistry: %s\nLog: %s\n' \
    "$SESSION" "${#configs[@]}" "$WORKERS_PER_GPU" "$GC_WALLOW_DB" "$LOG_FILE"
printf 'Attach with: tmux attach -t %s\n' "$SESSION"
