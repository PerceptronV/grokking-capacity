#!/usr/bin/env bash
# RMSNorm-frozen groks runs — the normalisation-clock test
# (theory/principled/gen-solvable-circuit falsifier): with the RMSNorm
# gains frozen at init, the prediction is (i) T_gen–λ slope steepens from
# −0.42 toward −1, and (ii) the width advantage (β) collapses toward its
# λ→0 value. Frozen runs live in a dedicated DB (no freeze column in the
# identifying tuple).
#
# Usage: CUDA_VISIBLE_DEVICES=<gpu> bash scripts/run_normfrozen_groks.sh
set -u
export GC_WALLOW_DB="$HOME/normfrozen_runs.db"
cd "$(dirname "$0")/.."

P=113
SEEDS="42 43 44"

run () {  # dim wd
  local dim=$1 wd=$2 s
  for s in $SEEDS; do
    echo "== normfrozen groks p=$P d=$dim wd=$wd seed=$s =="
    gc-groks --p $P --seed $s --dim $dim --operation / \
        --weight-decay "$wd" --depth 2 --heads 1 \
        --freeze-norm-scales --epochs 5000 \
        || echo "FAILED d=$dim wd=$wd seed=$s"
  done
}

# λ-slope test at d=128
run 128 1.0
run 128 0.3
run 128 0.1
# β test at λ=1.0 (d=128 shared with above)
run 64  1.0
run 256 1.0
echo "normfrozen suite done"
