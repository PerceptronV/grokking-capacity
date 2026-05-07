# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Install

```bash
pip install -e .   # editable; pulls wallow from /Users/yiding/Desktop/Research/wallow
```

Run-tracking lives in a SQLite registry (`runs.db`) defined by `wallow.toml`.
Both paths can be overridden via env vars: `GC_WALLOW_DB`, `GC_WALLOW_TOML`,
`GC_DATA_DIR`. The schema auto-creates on first use.

## Running experiments

```bash
# Run a full experiment suite (dispatches jobs in parallel across GPUs)
gc-dispatch --config configs/weight_decay_sweep.yaml

# Dry-run (does not touch the wallow DB)
gc-dispatch --config configs/weight_decay_sweep.yaml --dry-run

# Re-run even if a row already exists with status='completed'
gc-dispatch --config configs/depth_scaling.yaml --force

# Run a single experiment directly
gc-groks    --p 113 --seed 42 --dim 128 --operation / --weight-decay 1.0 --depth 2 --heads 1
gc-speed    --p 113 --seed 42 --dim 64 --n-samples 6328 --weight-decay 1.0
gc-capacity --p 113 --seed 42 --dim 10 --n-samples 9300 --dataset-type random
```

## Multi-node clusters (e.g. Lambda 2×8 H100)

`gc-dispatch` partitions the expanded command list across nodes via
`--node-rank` / `--num-nodes` (interleaved by `i % num_nodes == node_rank`),
preserving the `capacity → speed → groks` ordering within each node. The
match table is built only on node-rank 0. Both nodes must share `runs.db`
(typically over NFS) so the wallow UNIQUE constraint dedups across nodes.

```bash
# Node 0
gc-dispatch --config configs/weight_decay_sweep.yaml --num-nodes 2 --node-rank 0

# Node 1
gc-dispatch --config configs/weight_decay_sweep.yaml --num-nodes 2 --node-rank 1
```

Workers per GPU default to 4 for H100 (based on compute capability). Override
with `--workers-per-gpu` if needed.

## Figures

`gc-figures` reads any config in `configs/` and renders every figure family
straight off the wallow store. No CLI flags for plot selection — drive
everything from the YAML.

```bash
gc-figures --config configs/central.yaml                  # → figures/central/
gc-figures --config configs/weight_decay_sweep.yaml --out /tmp/wd/
gc-figures --all                                           # every config
gc-figures --config configs/central.yaml --only intersection --only stats
```

Output per config: `intersection/p=<P>[__<axis>=<v>].pdf` (mem-vs-gen
crossing, one per prime, suffix-tagged with the swept axes), `capacity/`
(Image 2a + 2b: total memorisation vs dataset size and saturation bits vs
params), `speed/` (Image 3a + 3b: epochs vs 1/(C·P) and vs capacity
fraction), `stats/predictiveness.csv` plus `predicted_vs_empirical.pdf`
and one `error_vs_<axis>.pdf` per swept axis, and `meta.json` recording
the per-arch capacity constants used and their provenance. All figures
are vector PDFs saved with `bbox_inches="tight"`. Intersection panels
carry no on-figure title — the slice identity (e.g. `p=113`) lives in
the filename.

The capacity constant is **per-architecture, measured from each config's
own capacity runs** (not the global `consts.C = 2.16`). When a sweep cell
has no matching capacity runs (e.g. capacity ran at `dropout=0.0` while
speed/groks ran at `dropout=0.2`), `analysis.config_view._resolve_C`
finds the closest-matching capacity-bearing arch group — scoring on
architecture > weight_decay > dropout — and falls back to `consts.C`
only if nothing matches. Provenance for every cell goes into `meta.json`.

## Architecture overview

The paper's central claim is that grokking onset is predicted by the intersection of memorisation speed and generalisation speed. Three experiment types generate the data:

- `**experiments/capacity.py**` — Measures `C` (bits per parameter). Trains on random-target data to saturation; the slope of bits-memorised vs param-count is fit by `analysis.capacity_constant.fit_capacity_slope` (or its wallow-querying wrapper `measure_capacity_constant`). The constant `C = 2.16` in `consts.py` is the empirical result for the standard architecture and is now used only as a fallback when a config lacks matching capacity runs.
- `**experiments/speed.py**` — Measures `T_mem` (memorisation speed). Counts steps to saturation on random-target data of a fixed size (`n_equiv`).
- `**experiments/groks.py**` — Measures `T_gen` (generalisation speed). Trains on modular arithmetic; records the epoch when validation accuracy first crosses the grokking threshold.

`**dispatch/main.py**` (entry point `gc-dispatch`) orchestrates suites via YAML configs. It expands parameter grids and, for each combo, claims a wallow row (`return_existing` → check `status` → skip or dispatch). Workers receive the claimed `--run-uuid` and write artefacts to `data/<exp_type>/<run_uuid>/trace.npz`. Dependency order within a suite is always `capacity → speed → groks`.

`**analysis/**` is the post-hoc layer. Two surfaces:
- Legacy match-table — `analysis/matching.py` pairs completed groks/speed rows by `(p, operation, train_fraction, depth, heads, dropout, init_scale, seed, n_samples ≈ n_equiv, param_count ≈ groks.param_count)` and `dispatch/main.py` writes `data/<suite_name>/matches.json` at the end of every suite.
- Config-driven figure pipeline — `analysis/config_view.py` (parses a YAML, groups completed wallow rows into `ArchGroup`s, resolves a per-group capacity constant), `analysis/aggregate.py` (seed aggregation: mean curves, min-delay, onset detection, intersection finder), `analysis/plots.py` (orchestrators per figure family), `analysis/stats.py` (predictiveness CSV + scatter + per-axis breakdowns), `analysis/_primitives.py` (matplotlib primitives), `analysis/cli.py` (the `gc-figures` entry point).

`**registry/**` is the wallow integration layer:
- `store.py` — cached `Store` and `Schema` accessors.
- `identifying.py` — `build_identifying()` constructs the identifying tuple per experiment type (derives `n_samples` for groks, picks `dataset_type='random'/'modular'`).
- `paths.py` — flat layout helpers: `data/<exp_type>/<run_uuid>/`.
- `provenance.py` — host / gpu_type / git info collector.
- `lifecycle.py` — `claim()` / `run_lifecycle()` helpers wrap the worker's claim → start → finalise/fail flow.

## Run registry (wallow)

`wallow.toml` is the source of truth for the schema. Two field categories:

- **Identifying** — composite UNIQUE constraint. Two runs with the same identifying tuple are the same experiment.
- **Annotating** — recorded *about* a run; freely overwritable. Includes `status`, `run_uuid`, paths, provenance (`host`, `gpu_type`, `git_hash`, `wallclock_seconds`), and per-experiment results.

To inspect a run: `wallow inspect <id>`. To add a hyperparameter: edit `wallow.toml` (and `registry/identifying.IDENTIFYING_FIELDS`), then `wallow migrate generate "..."` and `wallow migrate apply` (Alembic-managed once set up).

## File / directory naming

`data/<experiment_type>/<run_uuid>/trace.npz` (plus any extra artefacts in the same dir, e.g. `model.pt` from `gc-groks --save-model`). The `run_uuid` is a 12-char random hex generated at first claim; reruns of the same identifying tuple reuse it (artefacts are overwritten in place, no orphan dirs).

Filenames carry no semantic information — all hyperparameters live in the wallow row. To find the npz for a run, query the registry:

```python
from wallow import F
from grokking_capacity.registry import get_store
store = get_store()
r = store.where((F("experiment_type")=="speed") & (F("dim")==64) & (F("p")==113)).first()
print(r.npz_path)
```

## YAML config structure

Each config has `name`, `defaults`, and `experiments`. Experiments have a `type` field (`groks`, `speed`, `capacity`). List-valued keys (except `seeds`, `primes`, `dims`, `dim_ranges`, `param_count_targets`, `match_by`, `n_samples`, `type`) are Cartesian-producted. `n_samples: auto` resolves to `n_equiv` for the given `(p, operation, train_fraction)`. `match_by: param_count` with `param_count_targets` selects dims by nearest param count rather than by explicit dim list.

For capacity, `operation: random` (in the experiment block) selects random-target data; any of `+`, `-`, `*`, `/` selects the modular task at that op (subject to `n_samples ≤ p*(p-1)` for `/` or `p*p` otherwise).

## Key constants and defaults

`consts.py` has canonical hyperparameter defaults per experiment type. Two known asymmetries (both also encoded in `wallow.toml` and `dispatch/config.py`'s `_TYPE_DEFAULTS`):

- **Weight decay**: groks default `weight_decay=1.0` vs speed/capacity default `weight_decay=0.01` — historical confound corrected by `weight_decay_sweep`.
- **Dropout**: capacity default `dropout=0.0` vs speed/groks default `dropout=0.2`. The constant `C = 2.16` was measured at `dropout=0.0`; whether C is stable across dropout values is what `dropout_sweep` tests.

## Revision experiments

Experiments are organised across three axes (hyperparameters, tasks, architectures). Priority refers to revision timeline (Week 1 is highest).


| ID      | Name                      | Description                                                                                                     | Priority                 | Config                            |
| ------- | ------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------ | --------------------------------- |
| **1a**  | Weight decay sweep ✅      | λ ∈ {0.1, 0.3, 1.0, 3.0} with matched weight decay between speed and groks — eliminates the historical confound | Week 1 / highest         | `configs/weight_decay_sweep.yaml` |
| **1a*** | Dropout sweep             | do ∈ {0.0, 0.1, 0.2, 0.4} matched across capacity, speed, and groks — tests whether C depends on dropout        | Week 1                   | `configs/dropout_sweep.yaml`      |
| **1b**  | Learning rate sweep       | η ∈ {3×10⁻⁴, 1×10⁻³, 3×10⁻³} with matched lr between speed and groks                                            | Week 2                   | `configs/lr_sweep.yaml`           |
| **1c**  | Initialisation scale      | init_scale ∈ {0.5, 1.0, 2.0} — scales all weights post-init                                                     | Week 3                   | `configs/init_scale_sweep.yaml`   |
| **1d**  | Training fraction sweep ✅ | α ∈ {0.3, 0.4, 0.5, 0.6, 0.7} at p ∈ {97, 113, 139}                                                             | Week 1                   | `configs/alpha_sweep.yaml`        |
| **2a**  | Modular addition ✅        | Full pipeline on `+` (mod p)                                                                                    | Week 1 / high            | `configs/task_add.yaml`           |
| **2b**  | Modular multiplication    | Full pipeline on `*` (mod p)                                                                                    | Week 1                   | `configs/task_mul.yaml`           |
| **2c**  | Modular subtraction       | Full pipeline on `-` (mod p); T_mem reusable from 2a                                                            | Week 3 / low             | `configs/subtraction.yaml`        |
| **2d**  | Permutation composition   | S₅ group — needs new task implementation                                                                        | Ambitious / out of scope | —                                 |
| **3a**  | Depth scaling ✅           | L_depth ∈ {2,3,4,6,8,10} at fixed heads=1, matched by param count                                               | Week 2 / high            | `configs/depth_scaling.yaml`      |
| **3b**  | Attention heads           | H ∈ {1,2,4,8} at fixed depth=2; dims must be multiples of 8                                                     | Week 2                   | `configs/heads_sweep.yaml`        |
| **3c**  | Architectural variants    | Gated FFN removal, RMSNorm→LayerNorm, RoPE→learned pos emb                                                      | Week 3                   | —                                 |
| **3d**  | MLP baseline              | Replace Transformer with MLP; tests framework generality                                                        | Week 3 / ambitious       | —                                 |


