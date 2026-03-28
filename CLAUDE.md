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

- `**capacity.py**` — Measures `C` (bits per parameter). Trains on random-target data to saturation; fits the slope of bits-memorised vs param-count. The constant `C = 2.16` in `consts.py` is the empirical result for the standard architecture.
- `**speed.py**` — Measures `T_mem` (memorisation speed). Counts steps to saturation on random-target data of a fixed size (`n_equiv`), which is the equivalent-complexity dataset computed from the modular arithmetic task.
- `**groks.py**` — Measures `T_gen` (generalisation speed). Trains on modular arithmetic; records the epoch when validation accuracy first crosses the grokking threshold.

`**main.py**` orchestrates suites of these experiments via YAML configs. It expands parameter grids, skips already-existing results, and dispatches jobs in parallel across GPUs. Dependency order within a suite is always `capacity → speed → groks`.

`**matching.py**` pairs groks and speed results by `(param_count, n_equiv)` and writes a `matches.json` table used for the paper's plots.

`**results.py**` (`ResultsIndex`) is a queryable index over all `.meta.json` sidecar files. Each `.npz` result file has a paired `.meta.json` with full provenance (hyperparameters, git hash, timestamps).

`**experiment.py**` (`ExperimentConfig`, `save_run`) defines the standard metadata schema written to every `.meta.json`.

## File/directory naming

Every result lives under `data/{type}/` with:

- **Directory**: `p{p}_op_{op}_seed{seed}[_split{split_type}]`
- **Filename**: `{type}_dim{dim}_depth{depth}_heads{heads}_wd{wd}[_tf{tf}][_is{is}][_samples{n}].npz`

`_op_safe()` maps operation symbols: `/`→`div`, `*`→`mul`, `+`→`add`, `-`→`sub`.

**Naming convention philosophy:** filenames are human-readable and encode the hyperparameters that identify a run. New hyperparameters follow the *optional suffix with default* pattern: the suffix is **omitted when the value equals the default** and included otherwise. This gives zero migration cost — existing files are untouched, and new runs at non-default values get a distinguishing suffix automatically.

Current optional suffixes (all omitted at default):

- `_tf{train_fraction}` — omitted when `train_fraction == 0.5`
- `_is{init_scale}` — omitted when `init_scale == 1.0`

Depth and heads are **always** present (never omitted even at default), because they define the architecture family rather than a hyperparameter choice.

**Existence check philosophy:** all "does this run exist?" checks query `.meta.json` sidecars via `ResultsIndex.exists()`, not by reconstructing and checking filenames. This means adding a new hyperparameter requires only a one-line change to the query — no new filename generation logic and no migration branch. The `.meta.json` sidecar is the source of truth; the filename is for human readability only.

For hyperparameters added after some runs were already completed (e.g. `init_scale`), legacy sidecars that predate the field are matched using a callable filter that treats `None` as the default:

```python
is_filter = (lambda x: x is None or x == init_scale) if init_scale == 1.0 else init_scale
index.exists(..., init_scale=is_filter)
```

## YAML config structure

Each config has `name`, `defaults`, and `experiments`. Experiments have a `type` field (`groks`, `speed`, `capacity`). List-valued keys (except `seeds`, `primes`, `dims`, `dim_ranges`, `param_count_targets`, `match_by`, `n_samples`, `type`) are Cartesian-producted. `n_samples: auto` resolves to `n_equiv` for the given `(p, operation, train_fraction)`. `match_by: param_count` with `param_count_targets` selects dims by nearest param count rather than by explicit dim list.

## Key constants and defaults

`consts.py` has canonical hyperparameter defaults per experiment type. The groks default `weight_decay=1.0` differs from speed/capacity default `weight_decay=0.01` — this is a known historical confound that the `weight_decay_sweep` config corrects.

## Revision experiments

Experiments are organised across three axes (hyperparameters, tasks, architectures). Priority refers to revision timeline (Week 1 is highest).


| ID     | Name                    | Description                                                                                                     | Priority                 | Config                            |
| ------ | ----------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------ | --------------------------------- |
| **1a** | Weight decay sweep      | λ ∈ {0.1, 0.3, 1.0, 3.0} with matched weight decay between speed and groks — eliminates the historical confound | Week 1 / highest         | `configs/weight_decay_sweep.yaml` |
| **1b** | Learning rate sweep     | η ∈ {3×10⁻⁴, 1×10⁻³, 3×10⁻³} with matched lr between speed and groks                                            | Week 2                   | `configs/lr_sweep.yaml`           |
| **1c** | Initialisation scale    | init_scale ∈ {0.5, 1.0, 2.0} — scales all weights post-init                                                     | Week 3                   | `configs/init_scale_sweep.yaml`   |
| **1d** | Training fraction sweep | α ∈ {0.3, 0.4, 0.5, 0.6, 0.7} at p ∈ {97, 113, 139}                                                             | Week 1                   | `configs/alpha_sweep.yaml`        |
| **2a** | Modular addition        | Full pipeline on `+` (mod p)                                                                                    | Week 1 / high            | `configs/tasks.yaml`              |
| **2b** | Modular multiplication  | Full pipeline on `*` (mod p)                                                                                    | Week 1                   | `configs/tasks.yaml`              |
| **2c** | Modular subtraction     | Full pipeline on `-` (mod p); T_mem reusable from 2a                                                            | Week 3 / low             | `configs/subtraction.yaml`        |
| **2d** | Permutation composition | S₅ group — needs new task implementation                                                                        | Ambitious / out of scope | —                                 |
| **3a** | Depth scaling           | L_depth ∈ {2,3,4,6,8,10} at fixed heads=1, matched by param count                                               | Week 2 / high            | `configs/depth_scaling.yaml`      |
| **3b** | Attention heads         | H ∈ {1,2,4,8} at fixed depth=2; dims must be multiples of 8                                                     | Week 2                   | `configs/heads_sweep.yaml`        |
| **3c** | Architectural variants  | Gated FFN removal, RMSNorm→LayerNorm, RoPE→learned pos emb                                                      | Week 3                   | —                                 |
| **3d** | MLP baseline            | Replace Transformer with MLP; tests framework generality                                                        | Week 3 / ambitious       | —                                 |


