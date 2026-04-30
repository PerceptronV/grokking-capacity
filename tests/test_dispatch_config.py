"""Verify YAML expansion produces the expected identifying tuples."""
import yaml

from torch_grokking.dispatch.config import expand_runs


SMOKE_YAML = """
name: smoke_test_inline
defaults:
  primes: [13]
  seeds: [42]
  weight_decay: 1.0

experiments:
  capacity:
    type: capacity
    dims: [16]
    n_samples: [100, 200]
    weight_decay: 0.01
    dropout: 0.0
  speed:
    type: speed
    dims: [16]
    n_samples: auto
  groks:
    type: groks
    dims: [16]
"""


def test_expand_runs_orders_capacity_speed_groks():
    spec = yaml.safe_load(SMOKE_YAML)
    runs = list(expand_runs(spec))
    types = [r["experiment_type"] for r in runs]
    # 2 capacity + 1 speed + 1 groks
    assert types.count("capacity") == 2
    assert types.count("speed") == 1
    assert types.count("groks") == 1
    # Capacity entries come first.
    assert types[:2] == ["capacity", "capacity"]


def test_groks_run_omits_n_samples_in_dispatch_dict():
    spec = yaml.safe_load(SMOKE_YAML)
    groks = [r for r in expand_runs(spec) if r["experiment_type"] == "groks"]
    assert len(groks) == 1
    g = groks[0]
    # The dispatcher-internal dict for groks doesn't carry n_samples;
    # build_identifying derives it from (p, op, train_fraction).
    assert "n_samples" not in g
    assert g["operation"] == "/"
    assert g["train_fraction"] == 0.5
    assert g["dataset_type"] == "modular"


def test_speed_n_samples_auto_resolves():
    spec = yaml.safe_load(SMOKE_YAML)
    speed = [r for r in expand_runs(spec) if r["experiment_type"] == "speed"]
    assert len(speed) == 1
    # p=13, op='/', tf=0.5 → n_equiv = 13 * 12 * 0.5 = 78
    assert speed[0]["n_samples"] == 78


def test_default_per_type_overrides():
    """Capacity gets weight_decay=0.01 default; groks/speed inherit suite default 1.0."""
    spec = yaml.safe_load(SMOKE_YAML)
    runs = list(expand_runs(spec))
    cap = next(r for r in runs if r["experiment_type"] == "capacity")
    groks = next(r for r in runs if r["experiment_type"] == "groks")
    assert cap["weight_decay"] == 0.01
    assert groks["weight_decay"] == 1.0
    # Capacity dropout default 0.0 in the YAML overrides.
    assert cap["dropout"] == 0.0
