# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running experiments

```bash
# Run a full experiment suite (dispatches jobs in parallel across GPUs)
python main.py --config configs/weight_decay_sweep.yaml

# Dry-run to see what commands would be issued
python main.py --config configs/weight_decay_sweep.yaml --dry-run

# Re-run even if results already exist
python main.py --config configs/depth_scaling.yaml --force

# Run a single experiment directly
python groks.py --p 113 --seed 42 --dims 64 128 --op / --weight-decay 1.0 --depth 2 --heads 1
python speed.py --p 113 --seed 42 --dims 64 --samples-start 6328 --samples-end 6328 --samples-steps 1 --weight-decay 1.0
python capacity.py --p 113 --seed 42 --dims 10 12 14 --samples-start 1000 --samples-end 9300 --samples-steps 8
```

## Multi-node clusters (e.g. Lambda 2×8 H100)

`main.py` uses `--node-rank` / `--num-nodes` to partition the command list across nodes. Each node runs an interleaved shard (`i % num_nodes == node_rank`) of each experiment-type batch, preserving the `capacity → speed → groks` ordering within each node. The match table is built only on node-rank 0. Both nodes must share a filesystem (NFS) so that file-based existence checks are consistent.

Run simultaneously on each node:
```bash
# Node 0
python main.py --config configs/weight_decay_sweep.yaml --num-nodes 2 --node-rank 0

# Node 1
python main.py --config configs/weight_decay_sweep.yaml --num-nodes 2 --node-rank 1
```

Preview each node's shard before running:
```bash
python main.py --config configs/weight_decay_sweep.yaml --num-nodes 2 --node-rank 0 --dry-run
python main.py --config configs/weight_decay_sweep.yaml --num-nodes 2 --node-rank 1 --dry-run
```

Workers per GPU default to 4 for H100 (based on compute capability). Override with `--workers-per-gpu` if needed:
```bash
python main.py --config configs/weight_decay_sweep.yaml --num-nodes 2 --node-rank 0 --workers-per-gpu 6
```

## Visualisation

```bash
python visualise.py capacity --all --p 127 --save --no-show --curves
python visualise.py primes --p 97 101 103 107 109 113 127 131 137 139 --threshold-val 98 --max-dim 200 --save --no-show --correlation
python visualise.py primes --p 97 101 ... --save --no-show --speed
python visualise.py primes --p 97 101 ... --save --no-show --groks
```

## Data migration

```bash
# Migrate legacy data to current path format (idempotent)
python scripts/migrate_legacy_data.py --standardise-only

# Full migration (legacy → op-in-dir → standardised)
python scripts/migrate_legacy_data.py
```

## Architecture overview

The paper's central claim is that grokking onset is predicted by the intersection of memorisation speed and generalisation speed. Three experiment types generate the data:

- **`capacity.py`** — Measures `C` (bits per parameter). Trains on random-target data to saturation; fits the slope of bits-memorised vs param-count. The constant `C = 2.16` in `consts.py` is the empirical result for the standard architecture.
- **`speed.py`** — Measures `T_mem` (memorisation speed). Counts steps to saturation on random-target data of a fixed size (`n_equiv`), which is the equivalent-complexity dataset computed from the modular arithmetic task.
- **`groks.py`** — Measures `T_gen` (generalisation speed). Trains on modular arithmetic; records the epoch when validation accuracy first crosses the grokking threshold.

**`main.py`** orchestrates suites of these experiments via YAML configs. It expands parameter grids, skips already-existing results, and dispatches jobs in parallel across GPUs. Dependency order within a suite is always `capacity → speed → groks`.

**`matching.py`** pairs groks and speed results by `(param_count, n_equiv)` and writes a `matches.json` table used for the paper's plots.

**`results.py`** (`ResultsIndex`) is a queryable index over all `.meta.json` sidecar files. Each `.npz` result file has a paired `.meta.json` with full provenance (hyperparameters, git hash, timestamps).

**`experiment.py`** (`ExperimentConfig`, `save_run`) defines the standard metadata schema written to every `.meta.json`.

## File/directory naming

Every result lives under `data/{type}/` with:
- **Directory**: `p{p}_op_{op}_seed{seed}[_split{split_type}]`
- **Filename**: `{type}_dim{dim}_depth{depth}_heads{heads}_wd{wd}[_tf{tf}][_samples{n}].npz`

`_op_safe()` maps operation symbols: `/`→`div`, `*`→`mul`, `+`→`add`, `-`→`sub`.

The `_tf{tf}` suffix is only included when `train_fraction != 0.5`. Depth and heads are **always** present in filenames (never omitted when default).

## YAML config structure

Each config has `name`, `defaults`, and `experiments`. Experiments have a `type` field (`groks`, `speed`, `capacity`). List-valued keys (except `seeds`, `primes`, `dims`, `dim_ranges`, `param_count_targets`, `match_by`, `n_samples`, `type`) are Cartesian-producted. `n_samples: auto` resolves to `n_equiv` for the given `(p, operation, train_fraction)`. `match_by: param_count` with `param_count_targets` selects dims by nearest param count rather than by explicit dim list.

## Key constants and defaults

`consts.py` has canonical hyperparameter defaults per experiment type. The groks default `weight_decay=1.0` differs from speed/capacity default `weight_decay=0.01` — this is a known historical confound that the `weight_decay_sweep` config corrects.

## Existence checks (main.py)

Speed has a three-generation legacy chain: standard format → intermediate (pre-standardisation) → pre-migration path. The `_check_speed_exists`, `_check_groks_exists`, and `_check_capacity_exists` functions in `main.py` handle all generations so existing data is never redundantly re-run.
