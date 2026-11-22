# Experiment and analysis commands

Install with `pip install -e .`. The paper analysis is `gc-paper-analysis`;
manuscript formatting and compilation stay beside the TeX. Other workflows
are exposed through one command:

```bash
gc-workflow --list
gc-workflow capacity analyse_matched_capacity --figures
gc-workflow discrete run_p97_capacity_scaling --help
gc-workflow discrete report_p97_capacity_scaling
```

Use `python -m grokking_capacity.workflows` before reinstalling entry points.
Arguments are forwarded unchanged. Listing commands does not execute them.

| Group | Implementation | Purpose |
| --- | --- | --- |
| `capacity` | `experiments/capacity_workflow/` | Capacity calibration and hardware comparisons |
| `cone` | `experiments/cone_workflow/` | Cone experiments, aggregation, and verification |
| `discrete` | `experiments/discrete_workflow/` and `experiments/campaigns/` | Discrete experiments, reports, and audits |
| `mechanisms` | `analysis/mechanisms/` | Mechanism experiments and analyses |
| `sensitivity` | `analysis/law_sensitivity/` | Timescale fits and estimator sensitivity |

Paths are relative to `src/grokking_capacity/`. The catalogue is
`src/grokking_capacity/workflow_catalog.json`.

## Campaign execution

The capacity-scaling, dataset-scaling, categorical-search, dropout-match,
multilevel, matched-capacity, and dropout-follow-up launchers share
`experiments/campaigns/runner.py`. Campaigns read settings from
`configs/campaigns/` and accept `--config` and `--root`. The two stage launchers
also take `--stage` and a JSON `--jobs` list. Resource slots are configuration
values; repeating a device permits concurrent workers on that device.

```bash
gc-workflow discrete run_p97_dataset_scaling --config configs/campaigns/dataset_scaling.yaml --root results/my-dataset-sweep
gc-workflow discrete run_p97_dataset_scaling --config configs/campaigns/dataset_scaling.yaml --root results/my-dataset-sweep --resume --audit-device cuda:0
gc-workflow capacity run_capacity_h100_matched --dry-run
python -m grokking_capacity.experiments.campaigns status results/my-dataset-sweep
```

`--resume` verifies frozen inputs and completed artifacts, skips successful
training, and retries unfinished audits. It does not overwrite or automatically
resume a partially trained checkpoint. After a failure, pending jobs stop and
already running workers finish. Locks prevent concurrent controllers in a root.

Epoch-based arithmetic training, model construction, and capacity matching
have shared implementations. The dataset-size experiment retains its distinct
update-budget stopping rule. Capacity matching still requires the registered
model family and successful learning/calibration gates. Configuration changes
must respect these protocol constraints; a different experiment needs a new
output root. Some configurations refer to previously measured calibration
artifacts and cannot run without that evidence.

## Historical results

New runs use version-two registrations and output roots. Completed version-one
registrations retain their original hashes. Read-only reports verify these
against exact archived source bytes, and new audits record their current
implementation separately. Current training is never stamped with old hashes.
The fixed-cohort publication reports retain their original statistical scope.

Export the original sources, configurations, and dependencies for inspection or
historical replay:

```bash
python -m grokking_capacity.experiments.campaigns export-legacy /tmp/grokking-original
```

The archive and its SHA-256 manifest live in `experiments/legacy_v1/`. For replay,
use the exported tree as the working directory and put its `src` on `PYTHONPATH`;
provide the original result/calibration paths required by that registration.
The earlier frozen confirmation controllers are also available through
`gc-workflow discrete confirm_frozen <controller> --help`.

## Files in this directory

Only two executable launchers remain:

- `replicate.sh <env> [suite ...]`: launch all paper suites or a selected subset.
- `run_suite.sh <suite> <env>`: launch one suite in its own tmux session.

The third file is this guide. Experiment implementations live in the package.
