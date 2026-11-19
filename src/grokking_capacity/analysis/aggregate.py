"""Seed-aggregation primitives. Pure numpy, no I/O, no plotting.

Used by `figures/plots.py` and `figures/stats.py` to turn a raw set of
wallow rows into the curves and onset/intersection scalars the figures
and predictiveness analysis report on.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional

import numpy as np
from scipy.interpolate import interp1d


# Strings are serialisable in YAML; Python callers may supply a scalar callable.
# Quantiles use NumPy's linear interpolation convention, including the even-n median.
Reducer = str | Callable[[np.ndarray], float]


def reducer_name(metric: Reducer) -> str:
    return metric if isinstance(metric, str) else getattr(metric, "__name__", "callable")


def reduce_values(values: Iterable[float], metric: Reducer = "mean") -> float:
    """Reduce finite observations. Missing values must be handled by the caller.

    Built-ins: min, max, mean, median, geometric_mean, quantile:<q> (0 <= q <= 1).
    A callable receives a one-dimensional NumPy array and must return a finite scalar.
    """
    x = np.asarray(list(values), dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("reduction requires a non-empty finite vector")
    if callable(metric):
        value = float(metric(x))
    elif metric in {"min", "max", "mean", "median"}:
        value = float(getattr(np, metric)(x))
    elif metric == "geometric_mean":
        if (x <= 0).any():
            raise ValueError("geometric_mean requires strictly positive values")
        value = float(np.exp(np.mean(np.log(x))))
    elif metric.startswith("quantile:"):
        q = float(metric.split(":", 1)[1])
        if not 0 <= q <= 1:
            raise ValueError("quantile must be between zero and one")
        value = float(np.quantile(x, q))
    else:
        raise ValueError(f"unknown aggregation metric: {metric!r}")
    if not np.isfinite(value):
        raise ValueError("aggregation must return a finite scalar")
    return value


def _extended_reduce(values: list[float], metric: str) -> float:
    """Monotone reducers on non-negative extended reals (avoids 0 * infinity)."""
    x = sorted(values)
    if metric == "min":
        return x[0]
    if metric == "max":
        return x[-1]
    if metric == "mean":
        return float(np.mean(x))
    if metric == "median" or metric.startswith("quantile:"):
        q = .5 if metric == "median" else float(metric.split(":", 1)[1])
        if not 0 <= q <= 1:
            raise ValueError("quantile must be between zero and one")
        h = (len(x) - 1) * q
        lo, hi = int(np.floor(h)), int(np.ceil(h))
        if lo == hi or x[lo] == x[hi]:
            return x[lo]
        return (1 - (h - lo)) * x[lo] + (h - lo) * x[hi]
    raise ValueError(f"no censoring bounds registered for {metric!r}")


def summarize_delays(values: Iterable[Optional[float]], metric: Reducer = "min") -> dict:
    """Aggregate non-negative delays while retaining unresolved seed outcomes.

    Unknown delays range from zero to infinity. Elapsed follow-up is NEVER
    substituted for an observed validation crossing. Built-in monotone reducers
    propagate these bounds; a custom callable is evaluated only on complete data.
    `value` is supplied only when the aggregate is identified exactly. `sign`
    may be identified earlier, but is not evidence that unresolved seeds validated.
    """
    vals = list(values)
    observed = [float(v) for v in vals if v is not None]
    if any(not np.isfinite(v) or v < 0 for v in observed):
        raise ValueError("delays must be finite and non-negative, or None")
    n_missing = len(vals) - len(observed)
    result = {"metric": reducer_name(metric), "n_seeds": len(vals),
              "n_observed": len(observed), "n_unresolved": n_missing,
              "value": None, "lower": None, "upper": None, "upper_unbounded": True,
              "state": "unresolved", "sign": "unresolved"}
    if not vals:
        return result
    if not n_missing:
        value = reduce_values(observed, metric)
        if value < 0:
            raise ValueError("delay reducer must return a non-negative value")
        lower = upper = value
    elif callable(metric):
        return result
    else:
        # Unknown reducer names are errors even on incomplete data.
        lower = _extended_reduce([0. if v is None else float(v) for v in vals], metric)
        upper = _extended_reduce([float("inf") if v is None else float(v) for v in vals], metric)
    exact = lower == upper and np.isfinite(upper)
    result.update(value=float(lower) if exact else None, lower=float(lower),
                  upper=float(upper) if np.isfinite(upper) else None,
                  upper_unbounded=not np.isfinite(upper),
                  state=("positive" if lower > 0 else "zero") if exact else "unresolved",
                  sign="positive" if lower > 0 else "zero" if upper == 0 else "unresolved")
    return result


def aggregate_over_seeds(rows: Iterable[dict], *, x_field: str = "param_count",
                         y_field: str, metric: Reducer = "mean") -> dict[float, float]:
    """Group observed values by x, explicitly conditional on observing y.

    This is NOT a censoring correction. Delay observations should instead use
    `delay_curve`, which preserves missing seed outcomes.
    """
    bins: dict[float, list[float]] = {}
    for r in rows:
        x, y = r.get(x_field), r.get(y_field)
        if x is not None and y is not None:
            bins.setdefault(float(x), []).append(float(y))
    return {k: reduce_values(v, metric) for k, v in sorted(bins.items())}


def mean_over_seeds(rows: Iterable[dict], *, x_field: str = "param_count",
                    y_field: str) -> dict[float, float]:
    """Compatibility wrapper for the arithmetic seed mean of observed y values."""
    return aggregate_over_seeds(rows, x_field=x_field, y_field=y_field, metric="mean")


def delay_curve(delays: Iterable[tuple[float, Optional[float]]],
                metric: Reducer = "min") -> dict[float, Optional[float]]:
    """Aggregate per-seed delays by parameter count, keeping unknown aggregates."""
    bins: dict[float, list[Optional[float]]] = {}
    for pc, delay in delays:
        if pc is not None:
            bins.setdefault(float(pc), []).append(delay)
    return {pc: summarize_delays(v, metric)["value"] for pc, v in sorted(bins.items())}


def min_delay_curve(delays: Iterable[tuple[float, Optional[float]]]
                    ) -> dict[float, Optional[float]]:
    """Compatibility name for `delay_curve(..., metric="min")`."""
    return delay_curve(delays, metric="min")


def onset_summary(curve: dict[float, Optional[float]], *, threshold: float = 0.) -> dict:
    """First positive sampled aggregate, only when no smaller size is unresolved.

    Later zeros or unresolved sizes do not invalidate an identified first onset.
    An earlier unknown aggregate leaves the onset unresolved, even if a larger
    size is confirmed positive. Bounds and the first confirmed size are retained
    as metadata, never substituted for an identified onset in point statistics.
    """
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("delay threshold must be finite and non-negative")
    if any(v is not None and (not np.isfinite(v) or v < 0) for v in curve.values()):
        raise ValueError("onset curve values must be non-negative finite delays or None")
    items = sorted(curve.items())
    confirmed = next((x for x, v in items if v is not None and v > threshold), None)
    possible = next((x for x, v in items if v is None or v > threshold), None)
    earlier = [x for x, v in items if v is None and (confirmed is None or x < confirmed)]
    if not items:
        status = "empty"
    elif earlier:
        status = "unresolved"
    elif confirmed is None:
        status = "no_positive_observed"
    else:
        status = "left_censored" if confirmed == items[0][0] else "observed"
    return {"value": confirmed if status in ("observed", "left_censored") else None,
            "status": status, "earliest": possible, "latest": confirmed,
            "first_confirmed": confirmed, "earlier_unresolved": len(earlier),
            "earlier_unresolved_coordinates": earlier,
            "rule": "first_positive_with_smaller_sizes_resolved"}


def sustained_onset_summary(curve: dict[float, Optional[float]], *, threshold: float = 0.) -> dict:
    """Last-zero onset on a fixed grid, with uncertainty and edge status explicit.

    This is a sustained-positive-tail rule, NOT the first positive cell. Unknown
    aggregates are considered in both ways; a scalar onset is returned only if
    its location is invariant. Positive at the narrowest width is left censored.
    """
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("delay threshold must be finite and non-negative")
    if not curve:
        return {"value": None, "status": "empty", "earliest": None, "latest": None}
    if any(v is not None and (not np.isfinite(v) or v < 0) for v in curve.values()):
        raise ValueError("onset curve values must be non-negative finite delays or None")
    items = sorted(curve.items())
    def locate(unknown_positive):
        positive = [(unknown_positive if v is None else v > threshold) for _, v in items]
        zeros = [i for i, yes in enumerate(positive) if not yes]
        if not zeros:
            return items[0][0], "left_censored"
        if zeros[-1] == len(items) - 1:
            return None, "no_positive_tail"
        return items[zeros[-1] + 1][0], "observed"
    early, e_status = locate(True)
    late, l_status = locate(False)
    identified = early == late and e_status == l_status
    return {"value": early if identified else None,
            "status": e_status if identified else "unresolved",
            "earliest": early, "latest": late}


def find_grokking_onset(delay: dict[float, Optional[float]], *, threshold: float = 0.
                       ) -> Optional[float]:
    """First identified positive point on the sampled grid; see onset_summary."""
    return onset_summary(delay, threshold=threshold)["value"]


def intersection_summary(
    speed_curve: dict[float, float],
    groks_curve: dict[float, float],
    *,
    direction: str = "any",
) -> dict:
    """Actual roots of piecewise-linear log-time curves in log x.

    Interpolate only over the shared observed x domain. Union knots give
    every linear segment exactly, without a search grid or nearest-point
    fallback. ``decreasing`` selects the lower-clock transition from
    T_mem > T_gen to T_mem < T_gen; ``any`` accepts either direction.
    Coincident intervals have no identifiable single crossing.
    """
    if direction not in {"any", "decreasing", "increasing"}:
        raise ValueError(f"unknown crossing direction: {direction}")
    result = {"value": None, "time": None, "status": "insufficient_points",
              "roots": [], "interpolation": "linear log(time) in log(x)",
              "direction": direction, "domain": None}
    if len(speed_curve) < 2 or len(groks_curve) < 2:
        return result
    xs = np.array(sorted(speed_curve), dtype=float)
    xg = np.array(sorted(groks_curve), dtype=float)
    ys = np.array([speed_curve[x] for x in xs], dtype=float)
    yg = np.array([groks_curve[x] for x in xg], dtype=float)
    if any(not np.isfinite(a).all() or (a <= 0).any() for a in (xs, xg, ys, yg)):
        raise ValueError("crossing coordinates and times must be finite and positive")
    left, right = max(xs[0], xg[0]), min(xs[-1], xg[-1])
    if left >= right:
        return {**result, "status": "no_overlap"}
    result["domain"] = [float(left), float(right)]
    knots = np.unique(np.r_[xs[(xs >= left) & (xs <= right)],
                            xg[(xg >= left) & (xg <= right)], left, right])
    u = np.log(knots)
    sm = np.interp(u, np.log(xs), np.log(ys))
    gm = np.interp(u, np.log(xg), np.log(yg))
    gap = sm - gm
    zero = np.isclose(gap, 0, rtol=0, atol=1e-12)
    if np.any(zero[:-1] & zero[1:]):
        return {**result, "status": "coincident_interval"}
    candidates = []
    for i in np.flatnonzero(zero):
        before = gap[max(0, i - 1)]
        after = gap[min(len(gap) - 1, i + 1)]
        trend = "decreasing" if before > after else "increasing" if before < after else "touch"
        candidates.append((float(u[i]), trend))
    for i in range(len(u) - 1):
        if zero[i] or zero[i + 1] or gap[i] * gap[i + 1] >= 0:
            continue
        root = u[i] - gap[i] * (u[i + 1] - u[i]) / (gap[i + 1] - gap[i])
        candidates.append((float(root), "decreasing" if gap[i] > 0 else "increasing"))
    roots = [{"x": float(np.exp(q)), "time": float(np.exp(np.interp(q, u, sm))),
              "direction": trend} for q, trend in sorted(candidates)]
    result["roots"] = roots
    selected = [r for r in roots if direction == "any" or r["direction"] == direction]
    if selected:
        return {**result, "value": selected[0]["x"], "time": selected[0]["time"],
                "status": "observed"}
    status = ("mem_slower_throughout" if (gap > 0).all() else
              "mem_faster_throughout" if (gap < 0).all() else "no_selected_crossing")
    return {**result, "status": status}


def find_intersection(
    speed_curve: dict[float, float],
    groks_curve: dict[float, float],
    *,
    n_grid: int = 1000,
    direction: str = "any",
) -> Optional[tuple[float, float]]:
    """Return the first actual log-log crossing in the shared observed domain.

    ``n_grid`` is retained for call compatibility; exact segment roots do
    not use a grid. No crossing or a coincident interval returns None.
    Use ``intersection_summary`` for the status and all detected roots.
    """
    result = intersection_summary(speed_curve, groks_curve, direction=direction)
    if result["value"] is None:
        return None
    return result["value"], result["time"]


def compute_delays(
    groks_npz_records: Iterable[dict],
    *,
    x_field: str = "param_count",
    threshold_train: float = 99.0,
    threshold_val: float = 99.0,
    censoring: str = "unresolved",
) -> list[tuple[float, Optional[float]]]:
    """Per-seed delays. Default: unfinished validation remains None.

    `legacy_lower_bound` reproduces archived analyses: drop unobserved training
    crossings and replace unfinished validation with elapsed follow-up. It is
    for reproducibility audits only, not evidence of eventual generalisation.
    """
    if censoring not in {"unresolved", "legacy_lower_bound"}:
        raise ValueError(f"unknown censoring policy: {censoring}")
    out: list[tuple[float, Optional[float]]] = []
    for rec in groks_npz_records:
        x = rec.get(x_field)
        train = np.asarray(rec.get("train_acc"))
        val = np.asarray(rec.get("val_acc"))
        if x is None or train.size == 0 or val.size == 0:
            continue
        train_above = np.where(train >= threshold_train)[0]
        val_above = np.where(val >= threshold_val)[0]
        if censoring == "unresolved":
            if val_above.size == 0:
                out.append((float(x), None))
            elif train_above.size == 0:
                out.append((float(x), 0.))
            else:
                out.append((float(x), float(max(0, val_above[0] - train_above[0]))))
            continue
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
