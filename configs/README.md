# Experiment configurations

The top-level YAML suites with an `experiments` mapping are consumed by
`gc-dispatch --config <path>` and the `gc-figures` / analysis commands.
`--all` analyzes these dispatcher suites only. Configs in subdirectories are
selected explicitly by path.

- **Main sweep and robustness:** `central.yaml`, `weight_decay_sweep.yaml`,
  `lr_sweep.yaml`, `init_scale_sweep.yaml`, `alpha_sweep.yaml`,
  `depth_scaling.yaml`, `task_add.yaml`, and `task_mul.yaml`.
- **Capacity and mechanism experiments:** the other dispatcher suites here
  retain their original names and measurement settings.
