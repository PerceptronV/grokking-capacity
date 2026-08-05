#!/usr/bin/env bash
# Speedrun grid (speedrun/README.md). Two halves, one per GPU.
# Usage: scripts/run_speedrun_grid.sh <gpu 0|1>
set -u
cd "$(dirname "$0")/.."
G=$1
run() { CUDA_VISIBLE_DEVICES=$G python scripts/speedrun_train.py "$@"; }
if [ "$G" = "0" ]; then
  # A: character-coloured init (baseline kappa=1 shares seeds with B's const arm)
  for k in 1 3 10 30 oracle; do
    for s in 42 43 44; do run --char-boost "$k" --seed "$s" --tag A; done
  done
  # B: decay schedules
  for sch in "1.0:30,0.3" "2.0:30,0.3"; do
    for s in 42 43 44; do run --wd-schedule "$sch" --seed "$s" --tag B; done
  done
else
  # C: de-lazified width
  for d in 236 800; do
    for ls in 1.0 0.1 0.01; do
      for s in 42 43 44; do run --dim "$d" --logit-scale "$ls" --seed "$s" --tag C; done
    done
  done
fi
echo "grid half $G done"
