"""Parse a YAML config, group completed wallow rows by sweep cell.

A "cell" (`ArchGroup`) is one architecture × task setting that holds across
its capacity / speed / groks runs. Within a group the only varying fields
are `seed`, `dim`, and `n_samples` — those drive the seed/param/dataset axes
of every plot.

Each `ArchGroup` carries its own `capacity_constant`, fitted from its own
capacity runs (or `consts.C` when no capacity runs exist) — so that
sweep-specific normalisations (e.g. `f = S/(C·P)` and the inverse-capacity
plots) use the C that was actually measured at that architecture.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import yaml
from wallow import F

from .capacity_constant import fit_capacity_slope
from ..consts import C as DEFAULT_C
from ..registry import get_store, npz_path_for

# `..dispatch.config` imports back from this package (matching utilities),
# so `expand_runs` is imported lazily inside `ConfigView.from_yaml` to break
# the cycle.


# Identifying fields that pin one architecture/task setting. Excluded from
# the key (because they vary within a group): `seed`, `dim`, `n_samples`,
# `experiment_type`, `dataset_type`, `max_epochs` (capacity runs an order
# of magnitude longer than groks; not part of the architecture).
ARCH_KEY_FIELDS: tuple[str, ...] = (
    "p", "operation", "train_fraction", "split_type",
    "depth", "heads", "dropout", "init_scale",
    "lr", "weight_decay", "beta1", "beta2", "batch_size",
    "architecture_family",
)


# The wallow.toml defaults the dispatcher relies on when a field is not in
# the config — duplicated here so we can rebuild ArchKey from a wallow row
# whose absent fields default to these values.
_TOML_DEFAULTS: dict[str, Any] = {
    "operation": "/",
    "train_fraction": 0.5,
    "split_type": "random",
    "depth": 2,
    "heads": 1,
    "dropout": 0.2,
    "init_scale": 1.0,
    "lr": 0.001,
    "weight_decay": 1.0,
    "beta1": 0.9,
    "beta2": 0.98,
    "batch_size": 512,
    "architecture_family": "transformer_gated",
}


@dataclass(frozen=True)
class ArchKey:
    p: int
    operation: str
    train_fraction: float
    split_type: str
    depth: int
    heads: int
    dropout: float
    init_scale: float
    lr: float
    weight_decay: float
    beta1: float
    beta2: float
    batch_size: int
    architecture_family: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArchKey":
        return cls(**{f: d.get(f, _TOML_DEFAULTS.get(f)) for f in ARCH_KEY_FIELDS})

    def as_filter(self):
        """Wallow filter expression that matches rows with this architecture."""
        expr = F("p") == self.p
        for f in ARCH_KEY_FIELDS:
            if f == "p":
                continue
            expr = expr & (F(f) == getattr(self, f))
        return expr

    def short_label(self, swept_axes: Iterable[str]) -> str:
        """Filename-friendly suffix using only the swept fields."""
        parts = [f"{ax}={getattr(self, ax)}" for ax in swept_axes if hasattr(self, ax)]
        return "__".join(parts)


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Pull all fields a plot might want off a wallow Row."""
    fields = (
        "experiment_type", "p", "operation", "train_fraction", "split_type",
        "dataset_type", "n_samples", "dim", "depth", "heads", "dropout",
        "init_scale", "lr", "weight_decay", "beta1", "beta2", "batch_size",
        "max_epochs", "seed", "architecture_family",
        "param_count", "run_uuid", "npz_path", "n_equiv", "dataset_bits",
        "saturation_step", "saturation_epoch", "saturated", "final_acc",
        "final_loss", "final_train_acc", "final_val_acc", "grokking_epoch",
        "total_bits_memorized", "final_bits_per_example",
    )
    return {f: getattr(row, f, None) for f in fields}


@dataclass
class ArchGroup:
    key: ArchKey
    capacity_runs: list[dict[str, Any]] = field(default_factory=list)
    speed_runs: list[dict[str, Any]] = field(default_factory=list)
    groks_runs: list[dict[str, Any]] = field(default_factory=list)
    capacity_constant: float = DEFAULT_C
    capacity_constant_source: str = "fallback:consts.C"

    def has_data(self) -> bool:
        return bool(self.capacity_runs or self.speed_runs or self.groks_runs)


@dataclass
class ConfigView:
    config_name: str
    config_path: Path
    groups: list[ArchGroup]
    swept_axes: list[str]

    @classmethod
    def from_yaml(cls, path: str | Path, *, db_path: Optional[str] = None) -> "ConfigView":
        path = Path(path)
        with open(path) as f:
            spec = yaml.safe_load(f)

        # Lazy import: `dispatch.config` imports back from this package.
        from ..dispatch.config import expand_runs

        # Walk the config to discover every (arch, experiment_type) the
        # dispatcher would have produced. We don't query wallow per dispatched
        # row — one wallow query per (arch_key, experiment_type) is enough,
        # because the dispatcher uniquely defines the run.
        seen: set[ArchKey] = set()
        for run in expand_runs(spec):
            seen.add(ArchKey.from_dict(run))

        store = get_store(db_path)
        groups: list[ArchGroup] = []
        for key in sorted(seen, key=_key_sort):
            cap = [_row_to_dict(r) for r in store.where(
                (F("status") == "completed")
                & (F("experiment_type") == "capacity")
                & key.as_filter()
            ).all()]
            spd = [_row_to_dict(r) for r in store.where(
                (F("status") == "completed")
                & (F("experiment_type") == "speed")
                & key.as_filter()
            ).all()]
            grk = [_row_to_dict(r) for r in store.where(
                (F("status") == "completed")
                & (F("experiment_type") == "groks")
                & key.as_filter()
            ).all()]
            groups.append(ArchGroup(
                key=key,
                capacity_runs=cap, speed_runs=spd, groks_runs=grk,
            ))

        # Resolve capacity constants. Each group's C comes from its own
        # capacity rows when available; otherwise from the closest-matching
        # capacity-bearing group in the view. Fallback to consts.C.
        for g in groups:
            g.capacity_constant, g.capacity_constant_source = _resolve_C(g, groups)

        swept = _detect_swept_axes(groups)
        name = spec.get("name") or path.stem
        return cls(config_name=name, config_path=path,
                   groups=groups, swept_axes=swept)

    def iter_groups(self) -> Iterator[ArchGroup]:
        return iter(self.groups)


def _key_sort(k: ArchKey) -> tuple:
    return tuple(getattr(k, f) for f in ARCH_KEY_FIELDS)


# Architecture-only fields (excluding wd and dropout, which are the two axes
# that conventionally drift between capacity and speed/groks defaults).
_ARCH_ONLY_FIELDS = tuple(
    f for f in ARCH_KEY_FIELDS if f not in ("weight_decay", "dropout")
)


def _resolve_C(group: "ArchGroup", all_groups: list["ArchGroup"]) -> tuple[float, str]:
    """Choose a capacity constant for `group`.

    Order of preference:
      1. Group's own capacity runs.
      2. Closest-matching other group with capacity runs — score on
         architecture > weight_decay > dropout.
      3. `consts.C`.
    """
    own = fit_capacity_slope(group.capacity_runs) if group.capacity_runs else None
    if own is not None:
        return own, f"measured(own group, n={len(group.capacity_runs)} rows)"

    candidates = [g for g in all_groups if g is not group and g.capacity_runs]
    if not candidates:
        return DEFAULT_C, "fallback:consts.C"

    def score(g: "ArchGroup") -> tuple[int, int, int]:
        arch_match = sum(
            1 for f in _ARCH_ONLY_FIELDS
            if getattr(group.key, f) == getattr(g.key, f)
        )
        wd_match = int(group.key.weight_decay == g.key.weight_decay)
        do_match = int(group.key.dropout == g.key.dropout)
        return (arch_match, wd_match, do_match)

    best = max(candidates, key=score)
    slope = fit_capacity_slope(best.capacity_runs)
    if slope is None:
        return DEFAULT_C, "fallback:consts.C"
    label = f"wd={best.key.weight_decay}, dropout={best.key.dropout}"
    return slope, f"measured(matched group: {label})"


def _detect_swept_axes(groups: list[ArchGroup]) -> list[str]:
    """Identifying fields whose value differs across groups. `p` is excluded
    because the per-prime intersection plots already split on it."""
    if not groups:
        return []
    out = []
    for ax in ARCH_KEY_FIELDS:
        if ax == "p":
            continue
        values = {getattr(g.key, ax) for g in groups}
        if len(values) > 1:
            out.append(ax)
    return out


def load_npz(row: dict[str, Any]):
    """Load the trace.npz for a wallow row dict.

    Prefers the `npz_path` annotation on the row; falls back to the canonical
    path computed from `experiment_type` + `run_uuid`. Returns the npz file
    object (caller is responsible for `.close()` if they care).
    """
    import numpy as np

    explicit = row.get("npz_path")
    if explicit:
        p = Path(explicit)
        if p.exists():
            return np.load(p, allow_pickle=False)
    p = npz_path_for(row["experiment_type"], row["run_uuid"])
    return np.load(p, allow_pickle=False)
