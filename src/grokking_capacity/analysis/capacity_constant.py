"""Fit the capacity constant C from completed capacity rows."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from wallow import F

from ..registry import get_store


def fit_capacity_slope(rows: Iterable) -> Optional[float]:
    """Linear-fit total_bits_memorized ≈ C · param_count over already-fetched rows.

    Each row must expose `param_count` and `total_bits_memorized` (either as
    attributes on a wallow Row or as keys on a dict). At each param_count we
    keep the *maximum* observed bits — the saturation point. Returns None
    when fewer than two distinct param counts are available; callers fall
    back to `consts.C`.
    """
    by_param: dict[int, float] = {}
    for r in rows:
        if isinstance(r, dict):
            pc = r.get("param_count")
            bits = r.get("total_bits_memorized")
        else:
            pc = getattr(r, "param_count", None)
            bits = getattr(r, "total_bits_memorized", None)
        if pc is None or bits is None:
            continue
        pc_i = int(pc)
        bits_f = float(bits)
        if pc_i not in by_param or bits_f > by_param[pc_i]:
            by_param[pc_i] = bits_f

    if len(by_param) < 2:
        return None

    points = sorted(by_param.items())
    xs = np.array([p for p, _ in points], dtype=float)
    ys = np.array([b for _, b in points], dtype=float)
    slope, _intercept = np.polyfit(xs, ys, 1)
    return float(slope)


def measure_capacity_constant(
    *,
    db_path: str | None = None,
    depth: int = 2,
    heads: int = 1,
    weight_decay: float = 1.0,
    dropout: float = 0.0,
    init_scale: float = 1.0,
) -> Optional[float]:
    """Query wallow for completed capacity rows matching the architecture, fit C.

    Thin wrapper over `fit_capacity_slope`. Returns None when there aren't
    enough rows — callers fall back to `consts.C`.
    """
    store = get_store(db_path)
    rows = store.where(
        (F("status") == "completed")
        & (F("experiment_type") == "capacity")
        & (F("depth") == depth)
        & (F("heads") == heads)
        & (F("weight_decay") == weight_decay)
        & (F("dropout") == dropout)
        & (F("init_scale") == init_scale)
    ).all()
    return fit_capacity_slope(rows)
