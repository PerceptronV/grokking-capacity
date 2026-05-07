"""Seed-aggregation primitives. Pure numpy, no I/O, no plotting.

Used by `figures/plots.py` and `figures/stats.py` to turn a raw set of
wallow rows into the curves and onset/intersection scalars the figures
and predictiveness analysis report on.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from scipy.interpolate import interp1d


def mean_over_seeds(
    rows: Iterable[dict],
    *,
    x_field: str = "param_count",
    y_field: str,
) -> dict[float, float]:
    """Group rows by `x_field`, return arithmetic mean of `y_field` per x.

    Rows missing either field are dropped. Returns a sorted dict by key.
    """
    bins: dict[float, list[float]] = {}
    for r in rows:
        x = r.get(x_field)
        y = r.get(y_field)
        if x is None or y is None:
            continue
        bins.setdefault(float(x), []).append(float(y))
    return {k: float(np.mean(v)) for k, v in sorted(bins.items())}


def min_delay_curve(
    delays: Iterable[tuple[float, float]],
) -> dict[float, float]:
    """Reduce (param_count, delay) pairs to {param_count: min_delay} across seeds.

    The "min across compatible seeds" rule from visualise.py: at each param
    count, the smallest delay observed is the most generous estimate of when
    grokking *first* gets a foothold.
    """
    out: dict[float, float] = {}
    for pc, d in delays:
        if pc is None or d is None:
            continue
        pc_f = float(pc)
        d_f = float(d)
        if pc_f not in out or d_f < out[pc_f]:
            out[pc_f] = d_f
    return dict(sorted(out.items()))


def find_grokking_onset(min_delay: dict[float, float]) -> Optional[float]:
    """Smallest param count strictly *after* the last zero-delay point.

    Replicates the visualise.py rule: walk from the right, find the last
    zero-delay entry, and return the next param count up. If every point is
    non-zero, return the smallest. If every point is zero, return None.
    """
    if not min_delay:
        return None
    items = sorted(min_delay.items())
    last_zero_idx = -1
    for i in range(len(items) - 1, -1, -1):
        if items[i][1] == 0:
            last_zero_idx = i
            break
    if last_zero_idx == -1:
        return items[0][0]
    if last_zero_idx + 1 >= len(items):
        return None
    return items[last_zero_idx + 1][0]


def find_intersection(
    speed_curve: dict[float, float],
    groks_curve: dict[float, float],
    *,
    n_grid: int = 1000,
) -> Optional[tuple[float, float]]:
    """Return (param_count, epochs) where the two log-y curves cross.

    Both curves are interpolated linearly in (param_count, log(epochs))
    space on a shared log-spaced grid; the crossing point is taken at the
    grid index that minimises |log(speed) - log(groks)|. Returns None if
    the curves don't overlap on the param-count axis.
    """
    if not speed_curve or not groks_curve:
        return None

    speed_x = np.array(sorted(speed_curve.keys()), dtype=float)
    speed_y = np.array([speed_curve[x] for x in speed_x], dtype=float)
    groks_x = np.array(sorted(groks_curve.keys()), dtype=float)
    groks_y = np.array([groks_curve[x] for x in groks_x], dtype=float)

    if len(speed_x) < 2 or len(groks_x) < 2:
        return None

    x_min = max(speed_x.min(), groks_x.min())
    x_max = min(speed_x.max(), groks_x.max())
    if x_min >= x_max:
        return None

    f_speed = interp1d(speed_x, speed_y, kind="linear", fill_value="extrapolate")
    f_groks = interp1d(groks_x, groks_y, kind="linear", fill_value="extrapolate")

    grid = np.logspace(np.log10(x_min), np.log10(x_max), n_grid)
    y_speed = f_speed(grid)
    y_groks = f_groks(grid)

    # Drop grid points where either curve is non-positive: log() blows up.
    valid = (y_speed > 0) & (y_groks > 0)
    if not valid.any():
        return None
    diff = np.abs(np.log(y_speed[valid]) - np.log(y_groks[valid]))
    idx = int(np.argmin(diff))
    grid_v = grid[valid]
    y_speed_v = y_speed[valid]
    y_groks_v = y_groks[valid]
    return float(grid_v[idx]), float((y_speed_v[idx] + y_groks_v[idx]) / 2)


def compute_delays(
    groks_npz_records: Iterable[dict],
    *,
    x_field: str = "param_count",
    threshold_train: float = 99.0,
    threshold_val: float = 99.0,
) -> list[tuple[float, float]]:
    """For each per-seed groks record, return (x_value, delay).

    Records must contain `train_acc`, `val_acc` (in percent) and the field
    named by `x_field` (default `param_count` for the canonical figure).
    Records that never reach `threshold_train` are dropped; records that
    reach train but never val get delay = epochs_trained - train_epoch
    (i.e. the delay is at least that long, not None — matches the user's
    "non-zero delay" notion for the right tail of Image #1).
    """
    out: list[tuple[float, float]] = []
    for rec in groks_npz_records:
        x = rec.get(x_field)
        train = np.asarray(rec.get("train_acc"))
        val = np.asarray(rec.get("val_acc"))
        if x is None or train.size == 0 or val.size == 0:
            continue
        train_above = np.where(train >= threshold_train)[0]
        if train_above.size == 0:
            continue
        train_epoch = int(train_above[0])
        val_above = np.where(val >= threshold_val)[0]
        if val_above.size > 0:
            val_epoch = int(val_above[0])
            delay = max(0, val_epoch - train_epoch)
        else:
            # Train saturated but val never did within the run window — the
            # delay is at least (epochs_trained - train_epoch).
            delay = max(0, len(val) - train_epoch)
        out.append((float(x), float(delay)))
    return out
