#!/usr/bin/env bash
# SGD control cells for the batch-size discriminator (theory-trunk §3).
# Separate registry DB: the identifying tuple has no optimiser column, so
# SGD rows must never share a DB with AdamW rows. All cells: p=113,
# dim=96 (f ≈ 0.13, mid-window), SGD momentum 0.9, weight decay 0
# (decoupled-vs-L2 confound avoided; random-label fitting needs no wd).
# Grid: B in {128, 512, 2048} x lr in {0.1, 0.3} x seeds {52, 53}.
set -u
export GC_WALLOW_DB="$HOME/sgd_control_runs.db"
cd "$(dirname "$0")/.."
for lr in 0.1 0.3; do
  for B in 128 512 2048; do
    for seed in 52 53; do
      echo "=== SGD control: lr=$lr B=$B seed=$seed ==="
      gc-speed --p 113 --seed "$seed" --dim 96 --n-samples 6328 \
        --operation / --train-fraction 0.5 \
        --optimizer sgd --momentum 0.9 --lr "$lr" --weight-decay 0.0 \
        --batch-size "$B" --epochs 5000
    done
  done
done
echo "SGD control done"
