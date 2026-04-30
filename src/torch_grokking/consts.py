"""Canonical hyperparameter defaults per experiment type.

These mirror the defaults declared in `wallow.toml`. Two known asymmetries:

- **Weight decay**: groks default 1.0 vs speed/capacity default 0.01 — historical
  confound corrected by `weight_decay_sweep`.
- **Dropout**: capacity default 0.0 vs speed/groks default 0.2. The constant
  `C = 2.16` was measured at `dropout=0.0`.
"""

C = 2.16  # bits per parameter (depth=2, heads=1, dropout=0, transformer_gated)

GROKKING_DEFAULTS = {
    'weight_decay': 1.0,
    'dropout': 0.2,
    'lr': 1e-3,
    'batch_size': 512,
    'beta1': 0.9,
    'beta2': 0.98,
    'max_epochs': 200,
}

CAPACITY_DEFAULTS = {
    'weight_decay': 0.01,
    'dropout': 0.0,
    'lr': 1e-3,
    'batch_size': 512,
    'beta1': 0.9,
    'beta2': 0.98,
    'max_epochs': 5000,
}

SPEED_DEFAULTS = {
    'weight_decay': 0.01,
    'dropout': 0.2,
    'lr': 1e-3,
    'batch_size': 512,
    'beta1': 0.9,
    'beta2': 0.98,
    'max_epochs': 5000,
}
