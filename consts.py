C = 2.16  # bits per parameter (depth=2, heads=1, dropout=0, transformer_gated family)

# Canonical hyperparameter defaults per experiment type.
# These are the defaults baked into the CLI argparse blocks. Making them
# explicit here prevents the hidden-default confound from recurring.

GROKKING_DEFAULTS = {
    'weight_decay': 1.0,
    'dropout': 0.2,
    'lr': 1e-3,
    'batch_size': 512,
    'beta1': 0.9,
    'beta2': 0.98,
}

CAPACITY_DEFAULTS = {
    'weight_decay': 0.01,
    'dropout': 0.0,
    'lr': 1e-3,
    'batch_size': 512,
    'beta1': 0.9,
    'beta2': 0.98,
}

# NOTE: weight_decay=0.01 here does NOT match GROKKING_DEFAULTS.
# New speed experiments should use matched weight_decay.
SPEED_DEFAULTS = {
    'weight_decay': 0.01,
    'dropout': 0.2,
    'lr': 1e-3,
    'batch_size': 512,
    'beta1': 0.9,
    'beta2': 0.98,
}
