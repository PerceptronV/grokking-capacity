# Repository guidance

This is the canonical guide for coding agents working on this repository.
`CLAUDE.md` is a relative symlink to this file; edit `AGENTS.md` only.

## Purpose and scope

The project studies grokking onset through measured memorisation and
generalisation timescales. Start with [README.md](README.md) for installation
and revised-paper replication, [configs/README.md](configs/README.md) for suite
selection, and [scripts/README.md](scripts/README.md) for workflow commands.

The manuscript is in a separate, ignored Git checkout under
`docs/papers/2026 Grokking Capacity/`. Treat code and manuscript changes as
separate repository changes. Do not alter paper prose, the submission
checklist, or archived submissions as a side effect of a code refactor.

Keep the discrete experiments runnable; they are a separate line of work.
`logs/` is entirely ignored. Do not track its contents or link to internal
working notes from public documentation.

## Setup and validation

Use Python 3.10 or newer and an editable installation:

```bash
python -m pip install -e '.[test]'
```

The `paper` extra adds PDF-validation dependencies. A TeX installation is
separate. The optional shell launchers need Mamba and tmux; Python entry points
do not.

Run focused tests for the changed behaviour, then broader tests when the scope
warrants them. The suite uses temporary stores and synthetic inputs:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    PYTHONPATH=src MPLCONFIGDIR=/tmp/grokking-mpl pytest -q
```

Use `gc-dispatch --config configs/central.yaml --dry-run` and
`gc-workflow --list` for command inspection. Do not launch full training sweeps
or regenerate publication artifacts merely to test documentation or imports.
Analysis modules must remain safe to import without training, reading local
experiment archives, or creating outputs.

## Code organisation

- `experiments/`: individual capacity, speed, and grokking trainers; shared
  arithmetic training; character experiments; and specialised workflows.
- `experiments/campaigns/`: configuration, registration, execution, resume,
  and audit machinery shared by campaign launchers.
- `analysis/`: reusable measurements, matching, aggregation, fits, statistics,
  and rendering. `analysis/paper/` assembles the revised publication analyses.
- `data/`: modular-arithmetic and random-label dataset generation and splits.
- `models/`: the Transformer, discrete model variants, and shared model helpers.
- `dispatch/`: YAML expansion, scheduling, and suite execution.
- `registry/`: wallow schema access, run identity, lifecycle, provenance, and
  registry conversion/merging.
- `utils/`: shared losses, checkpoints, artifact handling, and workflow paths.
- `workflow_catalog.json`: the source of truth for `gc-workflow` commands.
- `configs/`: dispatcher suites; campaign and analysis profiles in subdirectories.
- `scripts/`: two shell launchers and their README. Put implementation in the
  package rather than adding another script.

Package paths above are relative to `src/grokking_capacity/`; `configs/` and
`scripts/` are at the repository root.

Prefer shared implementations with explicit configuration over hardcoded
prime-specific copies. Retain thin compatibility entry points where registered
workflows depend on them. Preserve protocol differences when sharing code:
epoch budgets and update budgets, for example, are not interchangeable.

## Training and configuration

`gc-capacity`, `gc-speed`, and `gc-groks` run individual experiments.
`gc-dispatch --config <yaml>` expands suites in capacity → speed → groks order.
`--workers-per-gpu` and `--max-workers` control concurrency; `--dry-run` must not
create or claim registry rows.

Suite YAMLs define `defaults` and `experiments`, including primes, seeds, width
grids, and swept settings. For speed, `n_samples: auto` uses the task-equivalent
random-label dataset size. Depth comparisons can match parameter counts rather
than embedding widths. See the config parser before changing expansion rules.

The individual trainers have different historical defaults. Reproduce a suite
with its YAML rather than assuming bare CLI defaults match the paper.
`central.yaml` explicitly sets weight decay 1, dropout 0.2, and 5,000 epochs.
The revised paper's matched capacity calibration is a separate campaign,
configured in `configs/campaigns/matched_capacity.yaml`.

Campaigns freeze configurations and source hashes under their output root.
Use a new root for a changed experiment. `--resume` checks registration and
artifacts; it is not an instruction to overwrite a partially trained checkpoint.
Keep `experiments/legacy_v1/` archives and manifests unchanged: they preserve
the source bytes for historical registrations. Current runs must record current
source hashes. Source inventories come from `utils/workflow_paths.py`.

## Registries and artifacts

`wallow.toml` defines the schema. Identifying fields form the unique experiment
key; annotations store outcomes, paths, and provenance. UUIDs are assigned by
wallow. Changes to identifying fields require coordinated schema, identity,
dispatcher, and trainer changes, with an explicit migration where needed.

Environment overrides are `GC_WALLOW_DB`, `GC_WALLOW_TOML`, and `GC_DATA_DIR`.
Set them before using the registry. Store/schema accessors are cached;
tests changing overrides must clear their caches.

Training must write SQLite registries on local disk, not NFS. The root
`runs.db` is the canonical merged analysis registry. Default artifacts are
`data/<experiment_type>/<uuid>/trace.npz`, with checkpoints beside them.
Use registry path helpers rather than reconstructing semantic filenames.
Forcing a rerun reuses the run identity and can overwrite its artifacts.

For multiple nodes, use distinct `--node-rank` values with the same
`--num-nodes`, plus a local registry per node. Jobs are interleaved within each
experiment type. The match table built on node 0 covers its local registry;
rebuild it against the merged data for a complete table.
Independent overlapping dispatchers do not have a shared atomic job lease:
the unique experiment key alone does not prevent duplicate training.

`gc-merge-registries convert` handles legacy schemas;
`gc-merge-registries merge` combines registries into a new output.
Sources remain read-only. Duplicate selection prioritises completed status,
artifact availability, and source order. Preserve the originals: existing
`runs_superseded` history is not imported from source databases.

## Scientific invariants

Preserve model initialisation, random-number consumption, data ordering,
parameter counts, optimiser settings, and stopping rules during refactors.
Use seeded equivalence checks when modifying training or model construction.
Do not silently turn a scientific correction into a behaviour-preserving cleanup.

In particular:

- The current Transformer passes head-first tensors into RoPE, whose sequence
  axis is axis 1. Changing this alters the trained architecture; it is not a
  neutral refactor.
- Nonrandom splits can produce sample counts different from the random-split
  count used in run identity. A correction needs coordinated metadata/identity
  handling. The central suite uses random splits.
- Historical host, hardware, and seed filters define analysis cohorts. Do not
  remove them to fill missing data, or select replacement runs by outcome.
  New replications use explicitly adapted copies of the analysis configs.
- Missing threshold crossings remain unresolved. A training limit is not an
  observed event time, and an unresolved seed is not a confirmed grokking seed.
- The primary onset is the first sampled size with a positive minimum delay
  across the declared seed pool. Any smaller unresolved minimum leaves onset
  unresolved. Later zeros do not invalidate this first onset. Keep left-censored
  bounds distinct from identified point observations. `analysis/aggregate.py`
  implements this; the sustained-tail rule is a separate sensitivity analysis.
- Keep threshold conventions, seed reducers, width domains, and capacity
  normalisation explicit. General suite plots can use historical fallback
  capacity constants; the revised paper uses its saved matched calibration.
- Timescale notation is `T_gen = A p^a P^(-b)` and
  `T_mem = B exp(gamma f)`, with `f = K_mem(p, alpha)/(C_model P)`.
  `alpha` denotes training fraction. Preserve compatibility with historical
  serialized coefficients through `analysis/timescale_laws.py`.

## Analysis and publication

`gc-figures --config <yaml> --db <registry> --out <directory>` renders suite
figures and statistics. `--only` selects families, `--skip-stats` skips tests,
and `meta.json` records capacity provenance. `--all` selects dispatcher suites,
not every config in every subdirectory.

`gc-workflow --list` discovers grouped commands without executing them.
Numerical logic belongs in the package; manuscript-specific formatting and
compilation live beside the TeX.

`gc-paper-analysis` runs crossings, statistics, models, and supplement stages
into `results/paper_revision_20261107/`. Use `--out` for isolated comparisons.
Models and supplement require statistics outputs. The pipeline depends on
archived registries, traces, and frozen law-fit inputs listed in the README;
the general registry environment overrides do not relocate every input.

The manuscript's `code/build_revised_paper.py --reuse-analyses` consumes the
default analysis output, publishes generated tables/figures, builds both PDF
copies, and validates numerical consistency. It does not reconcile changed
results in running prose automatically. Review scientific claims when analysis
changes; do not equate successful compilation with scientific validation.
