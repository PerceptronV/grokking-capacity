"""Shared helpers; load training dependencies only when their exports are used."""
from importlib import import_module

_EXPORTS = {
    'get_device': 'device',
    'gpu_type': 'device',
    'compute_dataset_size_bits': 'compute',
    'load_model': 'checkpoints',
    'save_model': 'checkpoints',
    'build_adamw': 'optim',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module('.' + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
