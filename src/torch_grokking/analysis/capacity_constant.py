"""Fit the capacity constant C from completed capacity rows."""
from __future__ import annotations

from typing import Optional

import numpy as np
from wallow import F

from ..registry import get_store


def measure_capacity_constant(
    *,
    db_path: str | None = None,
    depth: int = 2,
    heads: int = 1,
    weight_decay: float = 1.0,
    dropout: float = 0.0,
    init_scale: float = 1.0,
) -> Optional[float]:
    """Linear-fit total_bits_memorized ≈ C * param_count over completed capacity rows.

    Returns None when fewer than two distinct param counts are available — the
    caller should fall back to the global `consts.C` constant.
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
    if not rows:
        return None

    by_param: dict[int, float] = {}
    for r in rows:
        pc = getattr(r, "param_count", None)
        bits = getattr(r, "total_bits_memorized", None)
        if pc is None or bits is None:
            continue
        if pc not in by_param or bits > by_param[pc]:
            by_param[pc] = float(bits)

    if len(by_param) < 2:
        return None

    points = sorted(by_param.items())
    xs = np.array([p for p, _ in points], dtype=float)
    ys = np.array([b for _, b in points], dtype=float)
    # bits = C * params  (no intercept). Fit via least squares of y/x.
    slope, _intercept = np.polyfit(xs, ys, 1)
    return float(slope)
