"""YAML config expansion for the dispatcher.

Each YAML file has `name`, optional `defaults`, and `experiments`. Each
experiment has a `type` (`capacity`/`speed`/`groks`). List-valued keys
(except a small reserved set) are Cartesian-producted to enumerate runs.
"""
from __future__ import annotations

import itertools
from typing import Iterator

from .. import consts
from ..analysis.matching import compute_n_equiv, find_dims_for_param_targets

_NON_GRID_KEYS = {
    'seeds', 'primes', 'dims', 'dim_ranges', 'param_count_targets',
    'match_by', 'n_samples', 'type',
}

_TYPE_DEFAULTS = {
    name: {key: values[key] for key in ('weight_decay', 'dropout', 'max_epochs')}
    for name, values in {
        'capacity': consts.CAPACITY_DEFAULTS,
        'speed': consts.SPEED_DEFAULTS,
        'groks': consts.GROKKING_DEFAULTS,
    }.items()
}


def expand_experiment(exp_spec: dict, defaults: dict) -> list[dict]:
    """Cartesian-product all list-valued keys (excluding _NON_GRID_KEYS)."""
    merged = {**defaults, **exp_spec}
    grid_keys = [k for k, v in merged.items()
                 if isinstance(v, list) and k not in _NON_GRID_KEYS]
    scalar = {k: v for k, v in merged.items() if k not in grid_keys}
    if not grid_keys:
        return [merged]
    out = []
    for combo in itertools.product(*[merged[k] for k in grid_keys]):
        out.append({**scalar, **dict(zip(grid_keys, combo))})
    return out


def resolve_dims(cfg: dict) -> list:
    """Return either a list of int dims, or a list of (dim, actual_param_count) tuples."""
    if cfg.get('match_by') == 'param_count':
        targets = cfg.get('param_count_targets', [])
        primes = cfg.get('primes')
        p = primes if isinstance(primes, int) else (primes or [113])[0]
        return find_dims_for_param_targets(
            targets,
            depth=cfg.get('depth', 2),
            heads=cfg.get('heads', 1),
            p=p,
        )
    if 'dim_ranges' in cfg:
        dims: list[int] = []
        for r in cfg['dim_ranges']:
            dims.extend(range(r['start'], r['end'] + 1, r.get('step', 1)))
        return sorted(set(dims))
    if 'dims' in cfg:
        return list(cfg['dims'])
    return []


def resolve_n_samples(cfg: dict, p: int) -> int | None:
    if cfg.get('n_samples') == 'auto':
        op = cfg.get('operation', '/')
        tf = cfg.get('train_fraction', 0.5)
        n_equiv, _ = compute_n_equiv(p, op, tf)
        return n_equiv
    return cfg.get('n_samples')


def _iter_list(cfg: dict, key: str) -> list:
    val = cfg.get(key, [])
    return val if isinstance(val, list) else [val]


def expand_runs(spec: dict) -> Iterator[dict]:
    """Yield identifying-shaped dicts for every run in the suite, in dependency
    order (capacity → speed → groks). Each dict is a complete identifying tuple
    plus enough hyperparameters for the worker subprocess to consume.

    The yielded dict has the same shape as `build_identifying()` accepts: keys
    are field names, values are scalars. `dataset_type` is set per experiment
    type. n_samples for groks is derived inside `build_identifying`.
    """
    defaults = spec.get('defaults', {})
    experiments = spec.get('experiments', {})

    priority = {'capacity': 0, 'speed': 1, 'groks': 2}
    ordered = sorted(experiments.items(),
                     key=lambda item: priority.get(item[1].get('type', item[0]), 3))
    for exp_name, exp_spec in ordered:
        exp_type = exp_spec.get('type', exp_name)
        type_defaults = _TYPE_DEFAULTS.get(exp_type, {})
        configs = expand_experiment(exp_spec, {**type_defaults, **defaults})

        for cfg in configs:
            primes = cfg.get('primes', defaults.get('primes', [113]))
            if not isinstance(primes, list):
                primes = [primes]
            seeds = _iter_list(cfg, 'seeds')

            for p in primes:
                dims_raw = resolve_dims({**cfg, 'primes': p})
                if dims_raw and isinstance(dims_raw[0], tuple):
                    dims = [d for d, _ in dims_raw]
                else:
                    dims = list(dims_raw)
                if not dims:
                    continue

                if exp_type == 'capacity':
                    sizes = _iter_list(cfg, 'n_samples')
                elif exp_type == 'speed':
                    # Only speed resolves 'auto' to task-equivalent size.
                    raw = cfg.get('n_samples')
                    if isinstance(raw, list):
                        sizes = raw
                    else:
                        n = resolve_n_samples(cfg, p)
                        sizes = [n] if n is not None else []
                elif exp_type == 'groks':
                    # Its sample count is derived by build_identifying.
                    sizes = [None]
                else:
                    raise ValueError(f"unknown experiment type {exp_type!r}")

                for seed, dim, n in itertools.product(seeds, dims, sizes):
                    yield _build_run_dict(cfg, experiment_type=exp_type, p=p,
                                          seed=seed, dim=dim, n_samples=n,
                                          exp_name=exp_name)


def _common_hparams(cfg: dict) -> dict:
    return {
        'depth': cfg.get('depth', 2),
        'heads': cfg.get('heads', 1),
        'dropout': cfg.get('dropout', 0.2),
        'init_scale': cfg.get('init_scale', 1.0),
        'lr': cfg.get('lr', 1e-3),
        'weight_decay': cfg.get('weight_decay', 1.0),
        'beta1': cfg.get('beta1', 0.9),
        'beta2': cfg.get('beta2', 0.98),
        'batch_size': cfg.get('batch_size', 512),
        'max_epochs': cfg.get('max_epochs', 5000),
        'mup_base': int(cfg.get('mup_base', 0)),
    }


def _build_run_dict(cfg: dict, *, experiment_type: str, p: int, seed: int,
                    dim: int, n_samples: int | None, exp_name: str) -> dict:
    run = {
        'experiment_type': experiment_type,
        '_exp_name': exp_name,
        'p': p, 'seed': seed, 'dim': dim,
        'operation': cfg.get('operation', '/'),
        'train_fraction': cfg.get('train_fraction', 0.5),
        'split_type': cfg.get('split_type', 'random'),
        'dataset_type': 'modular' if experiment_type == 'groks' else 'random',
        **_common_hparams(cfg),
    }
    if experiment_type != 'groks':
        run['n_samples'] = n_samples
    if experiment_type == 'capacity':
        operation = cfg.get('operation', 'random')
        run.update(_op_raw=operation,
                   operation='/' if operation == 'random' else operation,
                   dataset_type='random' if operation == 'random' else 'modular',
                   train_fraction=0.5, split_type='random')
    if experiment_type == 'groks':
        # Optional, non-identifying diagnostics use worker defaults when absent.
        for key in ('norm_log_every', 'post_grok_epochs'):
            if cfg.get(key) is not None:
                run[key] = cfg[key]
    return run
