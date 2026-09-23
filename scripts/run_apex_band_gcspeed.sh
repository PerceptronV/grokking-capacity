#!/usr/bin/env bash
# Registered 2026-09-22 (mup-apex-locate.md, "Convention seam"): re-measure
# T_mem across the fixed-lr apex band on the framework's own convention
# (gc-speed: running train-mode accuracy with dropout on, >= 99 %), so the
# race map compares like with like. Primes 151..197, widths 124..556 (92
# omitted: censors at 5000 on this convention), 3 seeds, wd 1.0, dropout 0.2,
# n = n_equiv = p(p-1)/2. Local-disk DB (NFS cannot take WAL locks).
# Usage: scripts/run_apex_band_gcspeed.sh <gpu 0|1>
set -u
cd "$(dirname "$0")/.."
G=$1
export GC_WALLOW_DB=$HOME/apexmem_gcspeed_runs.db
if [ "$G" = "0" ]; then PR="151 163 173"; else PR="181 191 197"; fi
for p in $PR; do n=$(( p * (p - 1) / 2 )); for d in 556 428 364 300 236 172 124; do for s in 42 43 44; do
  CUDA_VISIBLE_DEVICES=$G gc-speed --p $p --seed $s --dim $d --n-samples $n --weight-decay 1.0 --dropout 0.2
done; done; done
echo "apex band gc-speed done (gpu $G)"
