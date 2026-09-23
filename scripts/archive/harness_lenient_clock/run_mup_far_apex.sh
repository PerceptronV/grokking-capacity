#!/usr/bin/env bash
# Discriminating test for the refit muP forms (mup-apex-locate.md,
# 2026-09-21): p in {449, 907}, wide widths, 8 seeds. New forms predict
# positive delay at 449, none at 907; asymptotic fade predicts
# sub-resolution at both. Gated on the apex-band sweep by ROW COUNT
# (366 = 216 apexMem + 150 apexGen), not pgrep (see deadlock note).
# Usage: scripts/run_mup_far_apex.sh <gpu 0|1>
set -u
cd "$(dirname "$0")/.."
G=$1
while [ "$(grep -c 'apexMem\|apexGen' results/speedrun/runs.jsonl)" -lt 366 ]; do sleep 600; done
if [ "$G" = "0" ]; then P=449; else P=907; fi
for d in 880 1400 2000; do for s in 42 43 44 45 46 47 48 49; do
  CUDA_VISIBLE_DEVICES=$G python scripts/speedrun_train.py --p $P --dim $d \
    --mup-base 236 --seed $s --tag farApex --epochs 1500
done; done
echo "far-apex done (p=$P)"
