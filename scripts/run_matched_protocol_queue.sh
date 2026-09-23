#!/usr/bin/env bash
# Matched-protocol re-run queue, v2 (2026-09-22). Two lanes, one per GPU,
# using gc-dispatch's node partitioning (--num-nodes 2 --node-rank k) so
# the two dispatchers never claim the same row. Lane 0 starts at once on
# GPU 0; lane 1 waits for the gc-speed band loop on GPU 1 to drain. Each
# config gets its own local-disk registry, shared by both lanes.
#   mup_depth_series (340 groks, muP 236, 10 seeds)
#   band_gaps        (210 groks, fixed rate, p = 181/191/193)
#   mup_laws         (168 speed + 168 groks, muP 236)
# Usage: scripts/run_matched_protocol_queue.sh <lane 0|1>
set -u
cd "$(dirname "$0")/.."
L=$1
if [ "$L" = "1" ]; then
  while pgrep -f "run_apex_band_gcspeed.sh" >/dev/null; do sleep 120; done
  echo "[$(date -Is)] band gc-speed loop drained; lane 1 starting"
fi
for cfg in mup_depth_series band_gaps mup_laws; do
  echo "[$(date -Is)] lane $L dispatch $cfg"
  CUDA_VISIBLE_DEVICES=$L GC_WALLOW_DB=$HOME/${cfg}_runs.db \
    gc-dispatch --config configs/$cfg.yaml --no-match-table --num-nodes 2 --node-rank $L
  echo "[$(date -Is)] lane $L done $cfg (exit $?)"
done
echo "[$(date -Is)] lane $L complete"
