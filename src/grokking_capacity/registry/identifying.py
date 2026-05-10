"""Build the identifying dict for a run from CLI args / config combos.

The set of identifying fields is the contract between the dispatcher (which
claims a row) and the worker (which finalises it). Both must produce the same
dict from the same hyperparameters, otherwise dedup splits silently.
"""
from __future__ import annotations

from typing import Any

from ..utils.compute import compute_dataset_size_bits


# Authoritative list of identifying-field names. Mirrors wallow.toml. The
# dispatcher and workers both pull from this so any future schema bump only
# needs to change one place.
IDENTIFYING_FIELDS: tuple[str, ...] = (
    "experiment_type",
    "p",
    "operation",
    "train_fraction",
    "split_type",
    "dataset_type",
    "n_samples",
    "dim",
    "depth",
    "heads",
    "dropout",
    "init_scale",
    "lr",
    "weight_decay",
    "beta1",
    "beta2",
    "batch_size",
    "max_epochs",
    "seed",
    "architecture_family",
)


def derive_n_samples(p: int, operation: str, train_fraction: float) -> int:
    """Number of training samples for a modular task at the given split."""
    n_samples, _ = compute_dataset_size_bits(p, operation, train_fraction)
    return int(n_samples)


def build_identifying(*, experiment_type: str, **kwargs: Any) -> dict[str, Any]:
    """Build the identifying dict for a run.

    Required: experiment_type ('capacity' | 'speed' | 'groks'), p, dim, seed.
    Other identifying fields use TOML defaults if not supplied. n_samples and
    dataset_type are derived per experiment_type when not explicitly given.
    """
    if experiment_type not in {"capacity", "speed", "groks"}:
        raise ValueError(f"unknown experiment_type {experiment_type!r}")

    out: dict[str, Any] = {"experiment_type": experiment_type}
    for k in IDENTIFYING_FIELDS:
        if k == "experiment_type":
            continue
        if k in kwargs and kwargs[k] is not None:
            out[k] = kwargs[k]

    # Derive dataset_type if not supplied: groks -> 'modular', speed -> 'random',
    # capacity -> defer to caller (they explicitly choose 'random' vs the modular op).
    if "dataset_type" not in out:
        if experiment_type == "groks":
            out["dataset_type"] = "modular"
        elif experiment_type == "speed":
            out["dataset_type"] = "random"
        # else capacity: rely on TOML default ('random')

    # Derive n_samples for groks runs from (p, operation, train_fraction).
    if experiment_type == "groks" and "n_samples" not in out:
        if "p" not in out:
            raise ValueError("groks identifying must include p")
        operation = out.get("operation", "/")
        train_fraction = out.get("train_fraction", 0.5)
        out["n_samples"] = derive_n_samples(out["p"], operation, train_fraction)

    if "n_samples" not in out:
        raise ValueError(
            f"{experiment_type} identifying must include n_samples"
        )
    return out
