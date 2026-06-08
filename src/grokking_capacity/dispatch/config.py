"""YAML config expansion for the dispatcher.

Each YAML file has `name`, optional `defaults`, and `experiments`. Each
experiment has a `type` (`capacity`/`speed`/`groks`). List-valued keys
(except a small reserved set) are Cartesian-producted to enumerate runs.
"""
from __future__ import annotations

import itertools
from typing import Iterator

from ..analysis.matching import compute_n_equiv, find_dims_for_param_targets

_NON_GRID_KEYS = {
    'seeds', 'primes', 'dims', 'dim_ranges', 'param_count_targets',
    'match_by', 'n_samples', 'type',
}

_TYPE_DEFAULTS = {
    # weight_decay / dropout / max_epochs differ per experiment type.
    'capacity': {'weight_decay': 0.01, 'dropout': 0.0, 'max_epochs': 5000},
    'speed':    {'weight_decay': 0.01, 'dropout': 0.2, 'max_epochs': 5000},
    'groks':    {'weight_decay': 1.0,  'dropout': 0.2, 'max_epochs': 200},
}


def _merge(defaults: dict, spec: dict) -> dict:
    return {**defaults, **spec}


def expand_experiment(exp_spec: dict, defaults: dict) -> list[dict]:
    """Cartesian-product all list-valued keys (excluding _NON_GRID_KEYS)."""
    merged = _merge(defaults, exp_spec)
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

    ordered = ['capacity', 'speed', 'groks']
    order = []
    for t in ordered:
        for name, esp in experiments.items():
            if esp.get('type', name) == t and name not in order:
                order.append(name)
    for name in experiments:
        if name not in order:
            order.append(name)

    for exp_name in order:
        if exp_name not in experiments:
            continue
        exp_spec = experiments[exp_name]
        exp_type = exp_spec.get('type', exp_name)
        type_defaults = _TYPE_DEFAULTS.get(exp_type, {})
        configs = expand_experiment(exp_spec, _merge(type_defaults, defaults))

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
                    n_samples_list = _iter_list(cfg, 'n_samples')
                    op_raw = cfg.get('operation', 'random')
                    for seed in seeds:
                        for dim in dims:
                            for n in n_samples_list:
                                yield _build_capacity_dict(cfg, p=p, seed=seed, dim=dim,
                                                            n_samples=n, op_raw=op_raw, exp_name=exp_name)
                elif exp_type == 'speed':
                    n = resolve_n_samples(cfg, p)
                    if n is None:
                        continue
                    for seed in seeds:
                        for dim in dims:
                            yield _build_speed_dict(cfg, p=p, seed=seed, dim=dim,
                                                     n_samples=n, exp_name=exp_name)
                elif exp_type == 'groks':
                    for seed in seeds:
                        for dim in dims:
                            yield _build_groks_dict(cfg, p=p, seed=seed, dim=dim, exp_name=exp_name)
                else:
                    raise ValueError(f"unknown experiment type {exp_type!r}")


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
    }


def _build_capacity_dict(cfg: dict, *, p: int, seed: int, dim: int, n_samples: int,
                          op_raw: str, exp_name: str) -> dict:
    if op_raw == 'random':
        dataset_type, operation = 'random', '/'
    else:
        dataset_type, operation = 'modular', op_raw
    return {
        'experiment_type': 'capacity',
        '_exp_name': exp_name,
        '_op_raw': op_raw,
        'p': p, 'seed': seed, 'dim': dim, 'n_samples': n_samples,
        'operation': operation, 'train_fraction': 0.5, 'split_type': 'random',
        'dataset_type': dataset_type,
        **_common_hparams(cfg),
    }


def _build_speed_dict(cfg: dict, *, p: int, seed: int, dim: int, n_samples: int,
                       exp_name: str) -> dict:
    return {
        'experiment_type': 'speed',
        '_exp_name': exp_name,
        'p': p, 'seed': seed, 'dim': dim, 'n_samples': n_samples,
        'operation': cfg.get('operation', '/'),
        'train_fraction': cfg.get('train_fraction', 0.5),
        'split_type': cfg.get('split_type', 'random'),
        'dataset_type': 'random',
        **_common_hparams(cfg),
    }


def _build_groks_dict(cfg: dict, *, p: int, seed: int, dim: int, exp_name: str) -> dict:
    d = {
        'experiment_type': 'groks',
        '_exp_name': exp_name,
        'p': p, 'seed': seed, 'dim': dim,
        'operation': cfg.get('operation', '/'),
        'train_fraction': cfg.get('train_fraction', 0.5),
        'split_type': cfg.get('split_type', 'random'),
        'dataset_type': 'modular',
        **_common_hparams(cfg),
    }
    # Optional norm-contraction knobs (non-identifying; forwarded to gc-groks
    # when present). Absent in most configs → the gc-groks CLI defaults apply.
    for k in ('norm_log_every', 'post_grok_epochs'):
        if cfg.get(k) is not None:
            d[k] = cfg[k]
    return d
