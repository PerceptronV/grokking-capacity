#!/usr/bin/env bash
# Grokking runs with init+final model snapshots for the m₀ selection test
# (analysis/init_overlap.py selection). Separate registry DB so these
# never collide with the canonical rows. (p=113, d=128) sits just past
# the p=113 onset (grokking regime); (p=97, d=128) likewise.
set -u
export GC_WALLOW_DB="$HOME/m0_runs.db"
cd "$(dirname "$0")/.."
for p in 113 97; do
  for seed in 42 43 44 45 46; do
    echo "=== m0 selection run: p=$p dim=128 seed=$seed ==="
    gc-groks --p "$p" --seed "$seed" --dim 128 --operation / \
      --train-fraction 0.5 --weight-decay 1.0 --depth 2 --heads 1 \
      --epochs 5000 --save-model
  done
done
echo "m0 selection runs done"
