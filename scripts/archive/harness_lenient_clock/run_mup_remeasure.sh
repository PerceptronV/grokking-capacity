#!/usr/bin/env bash
# muP re-measurement of both timescale laws (gated: waits for the
# re-entrance runs to drain). 3 primes x 10 widths x 3 seeds, groks + speed,
# per-layer muP (base 236). Refit (gamma, b, beta, a) under muP and re-solve
# the fold -> does the cone survive correct parameterisation?
set -u
cd "$(dirname "$0")/.."
while pgrep -f "tag reentrance" >/dev/null; do sleep 300; done
run() { python scripts/speedrun_train.py --mup-base 236 --epochs 3000 "$@"; }
G0() { CUDA_VISIBLE_DEVICES=0 run "$@"; }
G1() { CUDA_VISIBLE_DEVICES=1 run "$@"; }
DIMS="44 64 92 128 180 236 320 440 620 880"
(
for p in 97 139; do for d in $DIMS; do for s in 42 43 44; do
  G0 --p "$p" --dim "$d" --seed "$s" --tag mupG
  G0 --p "$p" --dim "$d" --seed "$s" --tag mupM --random-labels
done; done; done
) &
(
for p in 113; do for d in $DIMS; do for s in 42 43 44; do
  G1 --p "$p" --dim "$d" --seed "$s" --tag mupG
  G1 --p "$p" --dim "$d" --seed "$s" --tag mupM --random-labels
done; done; done
) &
wait
echo "muP re-measurement done"
