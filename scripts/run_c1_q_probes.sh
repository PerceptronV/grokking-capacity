#!/usr/bin/env bash
# C1-breakdown + curvature-continuation probes (theory/principled item 5).
#
#  Arm 1 (C1): speed runs at eta in {1e-4, 3e-4, 1e-3} at small f (d=216)
#    and mid f (d=64), wd=1.0 — mem-gram-projection predicts eta-invariance
#    degrades at small f; mem-dmft-perceptron predicts breakdown when
#    eta*lambda*T ~ 1.
#  Arm 2 (q): long-cap (30k epochs) speed runs at d in {50, 52, 56} —
#    pushes the censoring wall to add uncensored high-f points, testing
#    whether the measured q=+71 keeps steepening (wall precursor, dmft P2)
#    or saturates.
#
# Usage: CUDA_VISIBLE_DEVICES=<gpu> bash scripts/run_c1_q_probes.sh
set -u
export GC_WALLOW_DB="$HOME/probe_runs.db"
cd "$(dirname "$0")/.."

P=113
N=6328   # n_equiv at p=113, alpha=0.5

# Arm 1: eta grid
for dim in 216 64; do
  for lr in 0.0001 0.0003 0.001; do
    for s in 52 53; do
      echo "== C1 probe d=$dim lr=$lr seed=$s =="
      gc-speed --p $P --seed $s --dim $dim --n-samples $N \
          --weight-decay 1.0 --lr $lr --batch-size 512 -e 5000 \
          || echo "FAILED c1 d=$dim lr=$lr s=$s"
    done
  done
done

# Arm 2: long-cap curvature
for dim in 56 52 50; do
  for s in 52 53; do
    echo "== q probe d=$dim cap=30000 seed=$s =="
    gc-speed --p $P --seed $s --dim $dim --n-samples $N \
        --weight-decay 1.0 --lr 0.001 --batch-size 512 -e 30000 \
        || echo "FAILED q d=$dim s=$s"
  done
done
echo "probe suite done"
