#!/usr/bin/env bash
# Generic two-lane dispatcher: runs the given configs in order on one GPU
# lane, partitioned with --num-nodes 2 --node-rank <lane> so the two lanes
# never claim the same row. Each config gets its own local-disk registry.
# Usage: scripts/run_dispatch_lanes.sh <lane 0|1> <config> [<config> ...]
set -u
cd "$(dirname "$0")/.."
L=$1; shift
for cfg in "$@"; do
  echo "[$(date -Is)] lane $L dispatch $cfg"
  CUDA_VISIBLE_DEVICES=$L GC_WALLOW_DB=$HOME/${cfg}_runs.db \
    gc-dispatch --config configs/$cfg.yaml --no-match-table --num-nodes 2 --node-rank $L
  echo "[$(date -Is)] lane $L done $cfg (exit $?)"
done
echo "[$(date -Is)] lane $L complete"
