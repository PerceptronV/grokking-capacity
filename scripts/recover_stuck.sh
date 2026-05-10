#!/usr/bin/env bash
# Reset stuck 'running' rows that were SIGKILLed (or otherwise died without
# graceful exit), so the dispatcher can pick them up again.
#
# A row is "stuck" when status='running' and started_at is older than
# STALE_MIN minutes. The script reconstructs the identifying tuple from the
# row and calls wallow.register(on_duplicate='overwrite') to flip status.
# This is the same code path lifecycle.fail() would have taken.
#
# Dry-run by default — pass --apply to actually write.
#
# Usage:
#   scripts/recover_stuck.sh                    # dry-run, all stuck rows
#   scripts/recover_stuck.sh --apply            # apply the reset
#   STALE_MIN=60 scripts/recover_stuck.sh       # only rows older than 60 min
#   EXP_TYPE=speed scripts/recover_stuck.sh     # filter by experiment type
#   HOST=147-224-250-73 scripts/recover_stuck.sh   # filter by host
#   RESET_TO=pending scripts/recover_stuck.sh --apply   # reset to pending instead of failed
#
# Defaults:
#   STALE_MIN=30
#   RESET_TO=failed   (records error_excerpt; dispatcher re-picks failed rows)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STALE_MIN="${STALE_MIN:-30}"
RESET_TO="${RESET_TO:-failed}"
EXP_TYPE="${EXP_TYPE:-}"
HOST="${HOST:-}"

APPLY=0
if [[ "${1:-}" == "--apply" ]]; then
    APPLY=1
fi

if [[ "$RESET_TO" != "failed" && "$RESET_TO" != "pending" ]]; then
    echo "RESET_TO must be 'failed' or 'pending' (got: $RESET_TO)" >&2
    exit 2
fi

APPLY=$APPLY STALE_MIN=$STALE_MIN RESET_TO=$RESET_TO EXP_TYPE=$EXP_TYPE HOST=$HOST python - <<'PY'
import datetime as dt
import os

from wallow import F, register
from grokking_capacity.registry import get_store
from grokking_capacity.registry.identifying import IDENTIFYING_FIELDS

apply     = os.environ["APPLY"] == "1"
stale_min = int(os.environ["STALE_MIN"])
reset_to  = os.environ["RESET_TO"]
exp_type  = os.environ.get("EXP_TYPE") or None
host      = os.environ.get("HOST") or None

store = get_store()
now   = dt.datetime.now(dt.timezone.utc)
cutoff = now - dt.timedelta(minutes=stale_min)

q = store.where(F("status") == "running")
if exp_type:
    q = q.where(F("experiment_type") == exp_type)
if host:
    q = q.where(F("host") == host)

stuck = []
for r in q.all():
    started = r.started_at
    if started is None:
        continue
    if started.tzinfo is None:
        started = started.replace(tzinfo=dt.timezone.utc)
    if started < cutoff:
        stuck.append((r, started))

print(f"=== {now.isoformat()} ===")
print(f"mode:       {'APPLY' if apply else 'DRY-RUN (pass --apply to write)'}")
print(f"cutoff:     started_at < {cutoff.isoformat()}  (>{stale_min} min ago)")
print(f"reset_to:   {reset_to}")
filt = []
if exp_type: filt.append(f"experiment_type={exp_type}")
if host:     filt.append(f"host={host}")
print(f"filters:    {', '.join(filt) if filt else '(none)'}")
print(f"stuck rows: {len(stuck)}")
print()

if not stuck:
    print("Nothing to do.")
    raise SystemExit(0)

# Group by experiment_type for a tidy summary
by_type: dict[str, int] = {}
for r, _ in stuck:
    by_type[r.experiment_type] = by_type.get(r.experiment_type, 0) + 1
for et, n in sorted(by_type.items()):
    print(f"  {et:9} {n}")
print()

# Show first few rows
for r, started in stuck[:10]:
    age_min = (now - started).total_seconds() / 60
    print(f"  [{r.experiment_type}] p={r.p} dim={r.dim} wd={r.weight_decay} "
          f"seed={r.seed}  uuid={r.run_uuid}  ({age_min:.0f}min ago)")
if len(stuck) > 10:
    print(f"  ... and {len(stuck) - 10} more")
print()

if not apply:
    print("Dry-run only. Re-run with --apply to write.")
    raise SystemExit(0)

annotating_base = {
    "status":          reset_to,
    "completed_at":    now,
}
if reset_to == "failed":
    annotating_base["error_excerpt"] = (
        f"recover_stuck.sh: zombie 'running' row reset; "
        f"started_at={cutoff.isoformat()} cutoff exceeded "
        f"(no graceful exit, likely SIGKILL/OOM/host-restart)."
    )

ok = 0
errs: list[tuple[str, str]] = []
for r, _ in stuck:
    identifying = {}
    missing = []
    for f in IDENTIFYING_FIELDS:
        v = getattr(r, f, None)
        if v is None:
            missing.append(f)
        else:
            identifying[f] = v
    if missing:
        errs.append((r.run_uuid, f"missing identifying fields: {missing}"))
        continue
    try:
        register(
            store,
            identifying=identifying,
            annotating=annotating_base,
            on_duplicate="overwrite",
        )
        ok += 1
    except Exception as e:
        errs.append((r.run_uuid, f"{type(e).__name__}: {e}"))

print(f"Reset to '{reset_to}': {ok} / {len(stuck)}")
if errs:
    print(f"Errors: {len(errs)}")
    for uuid, msg in errs[:10]:
        print(f"  {uuid}  {msg}")
PY
