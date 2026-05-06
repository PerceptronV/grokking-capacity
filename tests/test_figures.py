"""Tests for the figures package: aggregation primitives, capacity fit,
and a smoke render against an isolated wallow store."""
from __future__ import annotations

import numpy as np
import pytest

from torch_grokking.analysis.capacity_constant import fit_capacity_slope
from torch_grokking.analysis import aggregate
from torch_grokking.analysis.config_view import ArchKey, ARCH_KEY_FIELDS


# ---- aggregate.find_grokking_onset -------------------------------------------

def test_onset_first_nonzero_after_last_zero():
    # Three zero-delay runs, then non-zero — onset is the first non-zero pc.
    md = {1000: 0.0, 2000: 0.0, 3000: 0.0, 4000: 5.0, 5000: 12.0}
    assert aggregate.find_grokking_onset(md) == 4000


def test_onset_skips_intermediate_zero():
    # Zero in the middle, non-zero at the end — onset jumps past the last zero.
    md = {1000: 5.0, 2000: 0.0, 3000: 8.0, 4000: 12.0}
    assert aggregate.find_grokking_onset(md) == 3000


def test_onset_all_zero_returns_none():
    assert aggregate.find_grokking_onset({1000: 0.0, 2000: 0.0}) is None


def test_onset_all_nonzero_returns_smallest():
    md = {2000: 5.0, 1000: 8.0, 3000: 2.0}
    assert aggregate.find_grokking_onset(md) == 1000


# ---- aggregate.find_intersection ---------------------------------------------

def test_intersection_on_monotone_curves():
    # speed: epochs decreases with param count (ish). groks: also decreases but
    # slower. They cross around ~3162.
    pcs = np.logspace(2, 4, 30)
    speed = {float(p): float(1e6 / p) for p in pcs}
    groks = {float(p): float(316.0) for p in pcs}  # constant; crosses where 1e6/p = 316
    # 1e6/p = 316 → p ≈ 3164
    pt = aggregate.find_intersection(speed, groks)
    assert pt is not None
    ix, iy = pt
    assert 3000 < ix < 3400
    assert 300 < iy < 332


def test_intersection_disjoint_returns_none():
    speed = {100.0: 50.0, 200.0: 30.0}
    groks = {1000.0: 10.0, 2000.0: 5.0}
    assert aggregate.find_intersection(speed, groks) is None


# ---- aggregate.compute_delays ------------------------------------------------

def test_compute_delays_basic():
    train = np.zeros(100)
    train[10:] = 100  # train reaches 99% at epoch 10
    val = np.zeros(100)
    val[40:] = 100    # val reaches 99% at epoch 40
    delays = aggregate.compute_delays(
        [{"param_count": 5000, "train_acc": train, "val_acc": val}],
    )
    assert delays == [(5000.0, 30.0)]


def test_compute_delays_train_never_saturates_dropped():
    train = np.zeros(100)
    val = np.zeros(100)
    val[10:] = 100
    assert aggregate.compute_delays(
        [{"param_count": 5000, "train_acc": train, "val_acc": val}],
    ) == []


def test_compute_delays_val_never_saturates_uses_run_length():
    train = np.zeros(100)
    train[10:] = 100
    val = np.zeros(100)  # never reaches threshold
    delays = aggregate.compute_delays(
        [{"param_count": 5000, "train_acc": train, "val_acc": val}],
    )
    # Falls back to len(val) - train_epoch = 100 - 10 = 90.
    assert delays == [(5000.0, 90.0)]


# ---- aggregate.mean_over_seeds -----------------------------------------------

def test_mean_over_seeds_averages_within_param_count():
    rows = [
        {"param_count": 100, "y": 10.0},
        {"param_count": 100, "y": 20.0},
        {"param_count": 200, "y": 5.0},
    ]
    out = aggregate.mean_over_seeds(rows, y_field="y")
    assert out == {100.0: 15.0, 200.0: 5.0}


# ---- capacity slope fit ------------------------------------------------------

def test_fit_capacity_slope_takes_max_at_each_param_count():
    # At pc=1000 we have two values; the saturation is the max.
    rows = [
        {"param_count": 1000, "total_bits_memorized": 1000.0},
        {"param_count": 1000, "total_bits_memorized": 2150.0},  # max wins
        {"param_count": 2000, "total_bits_memorized": 4300.0},
        {"param_count": 5000, "total_bits_memorized": 10750.0},
    ]
    slope = fit_capacity_slope(rows)
    assert slope is not None
    # bits = 2.15 * params (no intercept) → slope ≈ 2.15.
    assert abs(slope - 2.15) < 0.01


def test_fit_capacity_slope_too_few_points_returns_none():
    assert fit_capacity_slope([]) is None
    assert fit_capacity_slope([{"param_count": 1000, "total_bits_memorized": 2000}]) is None


# ---- ArchKey -----------------------------------------------------------------

def test_arch_key_defaults_match_wallow_toml():
    """If a YAML omits a field that wallow.toml defaults, ArchKey.from_dict
    should fill the same default — otherwise rows the dispatcher inserted
    won't match the filter the figures package builds."""
    minimal = {"p": 113}
    key = ArchKey.from_dict(minimal)
    assert key.p == 113
    assert key.operation == "/"          # wallow default
    assert key.train_fraction == 0.5
    assert key.depth == 2
    assert key.heads == 1
    assert key.architecture_family == "transformer_gated"


def test_arch_key_fields_cover_all_identifying_minus_seed_dim_n_samples():
    excluded = {"experiment_type", "seed", "dim", "n_samples",
                "dataset_type", "max_epochs"}
    from torch_grokking.registry.identifying import IDENTIFYING_FIELDS
    expected = set(IDENTIFYING_FIELDS) - excluded
    assert set(ARCH_KEY_FIELDS) == expected


# ---- end-to-end smoke render against an isolated wallow store ----------------

SMOKE_YAML = """
name: figures_smoke
defaults:
  primes: [13]
  seeds: [42]
  weight_decay: 1.0
experiments:
  capacity:
    type: capacity
    dims: [16]
    n_samples: [40, 80]
    weight_decay: 0.01
    dropout: 0.0
"""


def test_config_view_groups_runs_by_arch(isolated_repo, tmp_path):
    """ConfigView.from_yaml returns one ArchGroup per architecture cell.

    With only capacity runs in the DB, the speed/groks lists stay empty, the
    capacity list reflects the inserted rows, and the capacity_constant
    falls back to consts.C since fit needs ≥2 distinct param counts (the
    smoke config only has one dim).
    """
    import yaml as _yaml
    from wallow import register
    from torch_grokking.consts import C as DEFAULT_C
    from torch_grokking.dispatch.config import expand_runs
    from torch_grokking.analysis import ConfigView
    from torch_grokking.registry import build_identifying, get_store

    cfg = tmp_path / "smoke.yaml"
    cfg.write_text(SMOKE_YAML)

    store = get_store()
    spec = _yaml.safe_load(SMOKE_YAML)
    for run in expand_runs(spec):
        ident = build_identifying(**{k: v for k, v in run.items() if not k.startswith("_")})
        register(
            store,
            identifying=ident,
            annotating={
                "status": "completed",
                "run_uuid": f"smoke_{ident['n_samples']}",
                "param_count": 12000,
                "total_bits_memorized": float(ident["n_samples"]) * 5.0,
            },
            on_duplicate="overwrite",
        )

    view = ConfigView.from_yaml(cfg)
    assert len(view.groups) == 1
    g = view.groups[0]
    assert len(g.capacity_runs) == 2
    assert g.speed_runs == []
    assert g.groks_runs == []
    # Only one distinct param_count → fit returns None → fallback to consts.C.
    assert g.capacity_constant == DEFAULT_C
    assert g.capacity_constant_source == "fallback:consts.C"
