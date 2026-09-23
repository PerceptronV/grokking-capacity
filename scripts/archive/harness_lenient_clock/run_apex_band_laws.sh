#!/usr/bin/env bash
# Empirical T_mem and T_gen across the apex band, fixed-lr central protocol
# (user request 2026-09-20). Fills the measurement map p = 149..197:
#  - T_mem (random labels): primes 151..193 (149 exists in the registry;
#    197 measured 2026-09-20), 8 widths, 3 seeds  -> tag apexMem
#  - T_gen (grokking runs): primes 181, 191, 193 (the 179..197 gap; the
#    rest exist), 10 widths, 5 seeds              -> tag apexGen
# Gated: waits for the muP law-coverage runs to drain before starting.
# Usage: scripts/run_apex_band_laws.sh <gpu 0|1>
set -u
cd "$(dirname "$0")/.."
G=$1
# gate removed 2026-09-21: muP sweep complete (was: sibling-pgrep deadlock)
MEMD="92 124 172 236 300 364 428 556"
GEND="64 92 124 164 216 284 372 488 640 840"
run() { CUDA_VISIBLE_DEVICES=$G python scripts/speedrun_train.py "$@"; }
if [ "$G" = "0" ]; then
  for p in 151 157 163 167 173; do for d in $MEMD; do for s in 42 43 44; do
    run --p "$p" --dim "$d" --seed "$s" --tag apexMem --random-labels --epochs 2500
  done; done; done
  for p in 181 191; do for d in $GEND; do for s in 42 43 44 45 46; do
    run --p "$p" --dim "$d" --seed "$s" --tag apexGen --epochs 2500
  done; done; done
else
  for p in 179 181 191 193; do for d in $MEMD; do for s in 42 43 44; do
    run --p "$p" --dim "$d" --seed "$s" --tag apexMem --random-labels --epochs 2500
  done; done; done
  for p in 193; do for d in $GEND; do for s in 42 43 44 45 46; do
    run --p "$p" --dim "$d" --seed "$s" --tag apexGen --epochs 2500
  done; done; done
fi
echo "apex band laws done (gpu $G)"
