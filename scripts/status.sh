#!/usr/bin/env bash
# Snapshot of wallow run state. Safe to run anytime alongside an active dispatcher.
#
# Usage:
#   scripts/status.sh           # summary + recent failures + stuck-running rows
#   scripts/status.sh --watch   # auto-refresh every 30 s
#   STALE_MIN=10 scripts/status.sh   # flag 'running' rows older than 10 min as stuck
#   TG_WALLOW_DB=/path/to/runs.db scripts/status.sh   # override DB location
#
# NFS note: must point at the same DB the dispatcher uses. replicate{,2,3}.sh
# default to $HOME/torch_grokking_runs.db (SQLite WAL can't lock on NFS); we
# match that default here so plain `scripts/status.sh` Just Works.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export TG_WALLOW_DB="${TG_WALLOW_DB:-$HOME/torch_grokking_runs.db}"
STALE_MIN="${STALE_MIN:-30}"

run_once() {
python - <<PY
import datetime as dt
from wallow import F
from torch_grokking.registry import get_store
from torch_grokking.registry.store import default_db_path

store = get_store()  # honours TG_WALLOW_DB / TG_WALLOW_TOML env vars

print(f"\n=== {dt.datetime.now(dt.timezone.utc).isoformat()} ===")
print(f"db: {default_db_path()}")
print(f"total rows: {store.count()}")
for st in ("pending", "running", "completed", "failed"):
    n = store.where(F("status") == st).count()
    print(f"  {st:10} {n:6}")

# Per-experiment-type completion
print("\nby experiment_type (completed):")
for et in ("capacity", "speed", "groks"):
    total = store.where(F("experiment_type") == et).count()
    done = store.where((F("experiment_type") == et) & (F("status") == "completed")).count()
    if total:
        print(f"  {et:9} {done:>5} / {total:<5}  ({100*done/total:.1f}%)")

# Failures with errors
fails = store.where(F("status") == "failed").all()
if fails:
    print(f"\nFAILURES ({len(fails)}):")
    for r in fails[:20]:
        excerpt = (r.error_excerpt or "(no excerpt)").replace("\n", " ")[:120]
        print(f"  [{r.experiment_type}] p={r.p} dim={r.dim} n_samples={r.n_samples} "
              f"wd={r.weight_decay} lr={r.lr} dropout={r.dropout} seed={r.seed}")
        print(f"      {excerpt}")
        print(f"      uuid={r.run_uuid}  npz={r.npz_path}")

# Stuck 'running' rows (worker died without graceful exit)
stale_cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=${STALE_MIN})
running = store.where(F("status") == "running").all()
stuck = []
for r in running:
    started = r.started_at
    if started is None:
        continue
    if started.tzinfo is None:
        started = started.replace(tzinfo=dt.timezone.utc)
    if started < stale_cutoff:
        stuck.append((r, started))

if stuck:
    print(f"\nSTUCK ('running' > ${STALE_MIN}min, likely SIGKILLed): {len(stuck)}")
    for r, started in stuck[:20]:
        age_min = (dt.datetime.now(dt.timezone.utc) - started).total_seconds() / 60
        print(f"  [{r.experiment_type}] p={r.p} dim={r.dim} wd={r.weight_decay} "
              f"started={started.isoformat()} ({age_min:.0f}min ago)")
        print(f"      uuid={r.run_uuid}  host={r.host}")
PY
}

if [[ "${1:-}" == "--watch" ]]; then
    while :; do
        clear
        run_once
        sleep 30
    done
else
    run_once
fi
