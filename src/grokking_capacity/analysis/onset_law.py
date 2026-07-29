"""Onset capacity fraction law.

The central quantity is the *onset capacity fraction*

    f_onset(p, alpha) = K_mem(p, alpha) / (C * P_onset(p, alpha))

where K_mem is the dataset memorisation load in bits
(`matching.compute_n_equiv`), C is the capacity constant in bits/param
(`consts.C`), and P_onset is the empirical grokking-onset parameter count —
located exactly the way `stats._empirical_onset` does (min-delay over seeds,
smallest grid point past the last zero-delay one). If grokking onset is set
by the model's capacity relative to the dataset, f_onset should be a
dimensionless constant f* — flat in p, flat in train fraction alpha, and
(up to the small K ratio) flat across operations.

Two-stage CLI, because different registries resolve their npz artefacts
under different data roots (set GC_DATA_DIR before the interpreter starts):

  # Stage 1 — per (config, db): dump one onset cell per (arch group, prime)
  python -m grokking_capacity.analysis.onset_law extract \
      --config configs/central.yaml --db /path/to/runs.db \
      --out results/onset_law/cells_central.json

  # Stage 2 — pure arithmetic over the extracted cell tables
  python -m grokking_capacity.analysis.onset_law report \
      --central ... --alpha ... --div ... --add ... --mul ... \
      --out-dir results/onset_law --md-out docs/.../a2_onset_law.md

Stage 2 runs four analyses:
  * prime flatness   — spread of f_onset across primes at alpha=0.5, with the
                       width-grid quantisation floor made explicit;
  * alpha flatness   — paired residual comparison (fixed-f* vs intersection
                       prediction) across train fractions;
  * extrapolation    — calibrate f* on the small primes, predict P_onset for
                       the large ones, against a log-log power-law null;
  * operation ratios — P_onset shifts between /, + and * on matched hardware,
                       against the capacity-ratio and timescale-gap predictions.
plus a mechanical usability check on pre-registered criteria.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..consts import C as CAPACITY_C
from .config_view import ARCH_KEY_FIELDS, ConfigView
from .matching import compute_n_equiv
from .plots import _delay_records_for_slice, _passes_filters, _slice_values
from .stats import _empirical_onset, _predicted_onset

# Calibration / test split for the extrapolation analysis (central primes).
CALIBRATION_PRIMES = (97, 101, 103, 107, 109)
TEST_PRIMES = (113, 127, 131, 137, 139, 149)

# Predicted log10(P_onset ratio) between + or * and / coming from the
# T_gen gap between operations (timescale account), in dex.
TIMESCALE_GAP_DEX = 0.24


# ---------------------------------------------------------------------------
# Stage 1 — extraction
# ---------------------------------------------------------------------------


def extract_onset_cells(
    config_path: str | Path,
    db_path: str,
    *,
    figure_index: int = 0,
    group_filters: Optional[dict[str, Any]] = None,
    group_excludes: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """One onset cell per (arch group, slice value) of one intersection figure.

    Each cell carries the empirical onset P_onset, the intersection-predicted
    onset P_cross, the dataset load K_mem, the capacity fraction at both, and
    the local width-grid geometry (dex gaps to the adjacent grid parameter
    counts, and whether the onset sits within one grid step of the dim cap).
    """
    view = ConfigView.from_yaml(config_path, db_path=db_path)
    figure = view.intersection_figures[figure_index]

    groups = list(view.iter_groups())
    if group_filters:
        groups = [g for g in groups
                  if all(getattr(g.key, k) == v for k, v in group_filters.items())]
    if group_excludes:
        groups = [g for g in groups
                  if not all(getattr(g.key, k) == v for k, v in group_excludes.items())]

    cells: list[dict[str, Any]] = []
    for group in groups:
        for sv in _slice_values(group, figure):
            p = int(sv)
            records = _delay_records_for_slice(group, figure, sv)
            if not records:
                continue
            grid = sorted({float(r["x"]) for r in records})
            x_to_dim = {float(r["x"]): float(r["colour"]) for r in records}

            p_onset = _empirical_onset(group, figure, sv)
            p_cross = _predicted_onset(group, figure, sv)
            n_equiv, k_mem = compute_n_equiv(
                p, group.key.operation, float(group.key.train_fraction))

            gap_below = gap_above = None
            at_cap = None
            dim_onset = None
            if p_onset is not None:
                idx = int(np.argmin(np.abs(np.asarray(grid) - p_onset)))
                if idx > 0:
                    gap_below = math.log10(grid[idx] / grid[idx - 1])
                if idx + 1 < len(grid):
                    gap_above = math.log10(grid[idx + 1] / grid[idx])
                # Onset detection is right-censored when it lands within one
                # grid step of the widest width the sweep ran (the dim cap):
                # the true onset may lie beyond the grid.
                at_cap = idx >= len(grid) - 2
                dim_onset = x_to_dim.get(grid[idx])

            seeds = {r.get("seed") for r in group.groks_runs
                     if r.get(figure.slice_field) == sv
                     and _passes_filters(r, figure)
                     and r.get("seed") is not None}

            cell: dict[str, Any] = {
                "config": view.config_name,
                "p": p,
                "n_seeds": len(seeds),
                "P_onset": _f(p_onset),
                "P_cross": _f(p_cross),
                "dim_onset": _f(dim_onset),
                "grid_n": len(grid),
                "grid_min": grid[0],
                "grid_max": grid[-1],
                "gap_below_dex": _f(gap_below),
                "gap_above_dex": _f(gap_above),
                "at_cap": at_cap,
                "n_equiv": int(n_equiv),
                "K_mem_bits": float(k_mem),
                "C": float(CAPACITY_C),
                "group_capacity_constant": float(group.capacity_constant),
                "capacity_constant_source": group.capacity_constant_source,
                "f_onset": (float(k_mem / (CAPACITY_C * p_onset))
                            if p_onset else None),
                "f_cross": (float(k_mem / (CAPACITY_C * p_cross))
                            if p_cross else None),
            }
            for fld in ARCH_KEY_FIELDS:
                cell[fld] = getattr(group.key, fld)
            cells.append(cell)

    return {
        "config": view.config_name,
        "config_path": str(config_path),
        "db_path": str(db_path),
        "figure": figure.name,
        "slice_field": figure.slice_field,
        "x_field": figure.x_field,
        "max_dim": figure.max_dim,
        "delay_train_threshold": figure.delay_train_threshold,
        "delay_val_threshold": figure.delay_val_threshold,
        "cells": cells,
    }


def _f(v) -> Optional[float]:
    return None if v is None else float(v)


# ---------------------------------------------------------------------------
# Small statistics helpers
# ---------------------------------------------------------------------------


def _ols_slope_ci(x: np.ndarray, y: np.ndarray, level: float = 0.95) -> dict[str, Any]:
    """OLS y ~ a + b*x with a t-based CI on b."""
    from scipy import stats as sps

    n = len(x)
    if n < 3 or len(set(map(float, x))) < 2:
        return {"slope": None, "intercept": None, "se": None,
                "ci_low": None, "ci_high": None, "n": int(n),
                "skipped": "n<3_or_constant_x"}
    X = np.column_stack([np.ones(n), x])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = n - 2
    sigma2 = float(resid @ resid) / df if df > 0 else float("nan")
    xtx_inv = np.linalg.inv(X.T @ X)
    se = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    tcrit = float(sps.t.ppf(0.5 + level / 2, df)) if df > 0 else float("nan")
    b = float(beta[1])
    return {"slope": b, "intercept": float(beta[0]), "se": se,
            "ci_low": b - tcrit * se, "ci_high": b + tcrit * se, "n": int(n)}


def _bootstrap_slope_ci(
    x: np.ndarray, y: np.ndarray, *,
    n_resamples: int = 5000, seed: int = 0, level: float = 0.95,
) -> dict[str, Any]:
    """Percentile CI on the OLS slope, resampling observation units (here:
    primes) with replacement. Degenerate resamples (a single distinct x)
    are skipped."""
    rng = np.random.default_rng(seed)
    n = len(x)
    slopes = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if len(set(map(float, xb))) < 2:
            continue
        X = np.column_stack([np.ones(n), xb])
        beta, _, _, _ = np.linalg.lstsq(X, yb, rcond=None)
        slopes.append(float(beta[1]))
    if len(slopes) < 2:
        return {"ci_low": None, "ci_high": None, "n_resamples": 0}
    lo, hi = np.quantile(slopes, [0.5 - level / 2, 0.5 + level / 2])
    return {"ci_low": float(lo), "ci_high": float(hi),
            "n_resamples": int(len(slopes))}


def _median(vals: list[float]) -> Optional[float]:
    return float(np.median(vals)) if vals else None


# ---------------------------------------------------------------------------
# Stage 2 — analyses
# ---------------------------------------------------------------------------


def analyze_prime_flatness(cells: list[dict], *, n_bootstrap: int = 5000) -> dict[str, Any]:
    """Spread of f_onset across primes at one setting, plus flatness-in-p.

    The onset lives on a discrete width grid, so per-prime f_onset carries a
    quantisation floor: the dex gap between P_onset and the previous grid
    point bounds how precisely the onset is located. When the observed
    spread of log10 f_onset is at or below the median gap, the CV is
    reported as resolution-limited rather than as real scatter.

    The empirical onset is a seed-minimum order statistic, so no seed-level
    resampling is done here; the flatness slope CI comes from a prime-level
    bootstrap instead.
    """
    ok = sorted((c for c in cells if c.get("f_onset")), key=lambda c: c["p"])
    per_prime = [{
        "p": c["p"], "P_onset": c["P_onset"], "P_cross": c["P_cross"],
        "f_onset": c["f_onset"], "f_cross": c["f_cross"],
        "dim_onset": c["dim_onset"], "n_seeds": c["n_seeds"],
        "gap_below_dex": c["gap_below_dex"], "gap_above_dex": c["gap_above_dex"],
        "at_cap": c["at_cap"],
    } for c in ok]

    f = np.array([c["f_onset"] for c in ok], dtype=float)
    logf = np.log10(f)
    logp = np.log10(np.array([c["p"] for c in ok], dtype=float))
    gaps = [c["gap_below_dex"] for c in ok if c["gap_below_dex"] is not None]

    mean = float(f.mean())
    sd = float(f.std(ddof=1)) if len(f) > 1 else float("nan")
    cv = sd / mean if mean else float("nan")
    sd_log10 = float(logf.std(ddof=1)) if len(logf) > 1 else float("nan")
    median_gap = _median(gaps)
    resolution_limited = (median_gap is not None and np.isfinite(sd_log10)
                          and sd_log10 <= median_gap)

    ols = _ols_slope_ci(logp, logf)
    boot = _bootstrap_slope_ci(logp, logf, n_resamples=n_bootstrap)

    return {
        "n_primes": len(ok),
        "per_prime": per_prime,
        "f_onset_mean": mean,
        "f_onset_median": _median(list(f)),
        "f_onset_cv": cv,
        "f_onset_sd_log10_dex": sd_log10,
        "median_grid_gap_dex": median_gap,
        "cv_resolution_limited": bool(resolution_limited),
        "f_cross_median": _median([c["f_cross"] for c in ok if c["f_cross"]]),
        "flatness_ols_log10f_vs_log10p": ols,
        "flatness_slope_prime_bootstrap_ci": boot,
    }


def _residuals_for_cells(cells: list[dict], f_star: float) -> list[dict]:
    """Per-cell residuals of the two onset predictors, in dex.

    r_fstar — fixed capacity fraction: P_pred = K_mem / (C * f_star)
    r_int   — the speed/groks intersection: P_pred = P_cross
    Cells missing either onset are dropped; right-censored (at_cap) cells
    are kept but flagged for the caller to exclude.
    """
    out = []
    for c in cells:
        if not c.get("P_onset") or not c.get("P_cross"):
            continue
        r_fstar = math.log10(c["P_onset"] * c["C"] * f_star / c["K_mem_bits"])
        r_int = math.log10(c["P_onset"] / c["P_cross"])
        out.append({
            "train_fraction": c["train_fraction"], "p": c["p"],
            "config": c["config"], "at_cap": bool(c["at_cap"]),
            "n_seeds": c["n_seeds"],
            "r_fstar_dex": r_fstar, "r_int_dex": r_int,
            "r_diff_dex": r_fstar - r_int,
        })
    return out


def analyze_alpha_flatness(cells: list[dict], *, f_star: float) -> dict[str, Any]:
    """Is the fixed-f* predictor as alpha-stable as the intersection?

    Regresses the paired per-cell residual difference (r_fstar - r_int) on
    train fraction; each residual's own slope on alpha is reported for
    context. Cells whose onset is right-censored at the dim cap are
    excluded from every fit (their P_onset is a lower bound, not a
    measurement). The alpha=0.5 cells come from a much larger prime/seed
    pool than the other alphas, so all slopes are reported both with and
    without alpha=0.5.
    """
    res = _residuals_for_cells(cells, f_star)
    usable = [r for r in res if not r["at_cap"]]
    excluded = [r for r in res if r["at_cap"]]

    def fits(rows: list[dict]) -> dict[str, Any]:
        a = np.array([r["train_fraction"] for r in rows], dtype=float)
        return {
            "n_cells": len(rows),
            "paired_diff_slope": _ols_slope_ci(
                a, np.array([r["r_diff_dex"] for r in rows])),
            "r_fstar_slope": _ols_slope_ci(
                a, np.array([r["r_fstar_dex"] for r in rows])),
            "r_int_slope": _ols_slope_ci(
                a, np.array([r["r_int_dex"] for r in rows])),
        }

    return {
        "f_star": f_star,
        "cells": res,
        "n_excluded_at_cap": len(excluded),
        "excluded_cells": [{"train_fraction": r["train_fraction"], "p": r["p"]}
                           for r in excluded],
        "with_alpha_0.5": fits(usable),
        "without_alpha_0.5": fits(
            [r for r in usable if r["train_fraction"] != 0.5]),
    }


def analyze_extrapolation(
    central_cells: list[dict],
    alpha_cells: list[dict],
    *,
    f_star_all_primes: float,
    n_bootstrap: int = 5000,
) -> dict[str, Any]:
    """Calibrate f* on the small primes, predict P_onset for the large ones.

    Null baseline: a log-log power law P_onset ~ p fitted on the same
    calibration primes and extrapolated. The comparison is paired per
    prime on |residual| in dex. Cross-axis variant: the alpha!=0.5 cells
    can be predicted by f* (K_mem changes with alpha) but not by the
    p-only power law, so its MAE there is reported without a baseline.
    """
    by_p = {c["p"]: c for c in central_cells if c.get("f_onset")}
    calib = [by_p[p] for p in CALIBRATION_PRIMES if p in by_p]
    test = [by_p[p] for p in TEST_PRIMES if p in by_p]

    f_star_calib = _median([c["f_onset"] for c in calib])

    # Power-law null on the calibration primes.
    logp_c = np.log10(np.array([c["p"] for c in calib], dtype=float))
    logP_c = np.log10(np.array([c["P_onset"] for c in calib], dtype=float))
    null = _ols_slope_ci(logp_c, logP_c)

    per_prime = []
    for c in test:
        pred_f = c["K_mem_bits"] / (c["C"] * f_star_calib)
        res_f = math.log10(c["P_onset"] / pred_f)
        pred_null = 10 ** (null["intercept"] + null["slope"] * math.log10(c["p"]))
        res_null = math.log10(c["P_onset"] / pred_null)
        per_prime.append({
            "p": c["p"], "P_onset": c["P_onset"],
            "P_pred_fstar": pred_f, "residual_fstar_dex": res_f,
            "P_pred_null": pred_null, "residual_null_dex": res_null,
            "fstar_closer": abs(res_f) < abs(res_null),
            "gap_below_dex": c["gap_below_dex"],
        })

    abs_f = [abs(r["residual_fstar_dex"]) for r in per_prime]
    abs_n = [abs(r["residual_null_dex"]) for r in per_prime]

    # Cross-axis: predict every alpha != 0.5 cell from the alpha=0.5 f*.
    cross = []
    for c in alpha_cells:
        if c["train_fraction"] == 0.5 or not c.get("P_onset"):
            continue
        pred = c["K_mem_bits"] / (c["C"] * f_star_all_primes)
        cross.append({
            "train_fraction": c["train_fraction"], "p": c["p"],
            "P_onset": c["P_onset"], "P_pred_fstar": pred,
            "residual_dex": math.log10(c["P_onset"] / pred),
            "at_cap": bool(c["at_cap"]),
        })
    cross_ok = [r for r in cross if not r["at_cap"]]

    return {
        "calibration_primes": [c["p"] for c in calib],
        "test_primes": [c["p"] for c in test],
        "f_star_calibrated": f_star_calib,
        "null_powerlaw": null,
        "per_prime": per_prime,
        "mae_fstar_dex": float(np.mean(abs_f)) if abs_f else None,
        "mae_null_dex": float(np.mean(abs_n)) if abs_n else None,
        "median_abs_residual_fstar_dex": _median(abs_f),
        "median_grid_gap_test_primes_dex": _median(
            [r["gap_below_dex"] for r in per_prime
             if r["gap_below_dex"] is not None]),
        "n_primes_fstar_closer": sum(r["fstar_closer"] for r in per_prime),
        "n_test_primes": len(per_prime),
        "cross_axis": {
            "f_star_used": f_star_all_primes,
            "cells": cross,
            "n_cells": len(cross_ok),
            "n_excluded_at_cap": len(cross) - len(cross_ok),
            "mae_dex": (float(np.mean([abs(r["residual_dex"]) for r in cross_ok]))
                        if cross_ok else None),
            "note": ("the p-only power-law null has no dependence on "
                     "train_fraction and cannot make this prediction"),
        },
    }


def analyze_operation_ratios(
    div_cells: list[dict],
    add_cells: list[dict],
    mul_cells: list[dict],
    *,
    timescale_gap_dex: float = TIMESCALE_GAP_DEX,
) -> dict[str, Any]:
    """P_onset shifts between operations on matched hardware and seeds.

    Two candidate accounts for a + or * vs / shift:
      * capacity-only — K_mem grows by p/(p-1) (full table p^2 vs p(p-1)),
        predicting log10(p/(p-1)) ~ +0.004 dex;
      * timescale — the T_gen gap between operations predicts ~ +0.24 dex.
    The width grid quantises onsets at ~0.05-0.16 dex, so the capacity-only
    shift is below the measurement floor by construction: the data can be
    consistent with the timescale gap or unresolved, but can never positively
    confirm the capacity-only shift.
    """
    div_by_p = {c["p"]: c for c in div_cells if c.get("P_onset")}

    def ratios(op_cells: list[dict]) -> list[dict]:
        out = []
        for c in sorted(op_cells, key=lambda c: c["p"]):
            base = div_by_p.get(c["p"])
            if base is None or not c.get("P_onset"):
                continue
            gaps = [g for g in (c["gap_below_dex"], base["gap_below_dex"])
                    if g is not None]
            out.append({
                "p": c["p"],
                "P_onset_op": c["P_onset"], "P_onset_div": base["P_onset"],
                "delta_dex": math.log10(c["P_onset"] / base["P_onset"]),
                "capacity_pred_dex": math.log10(c["p"] / (c["p"] - 1)),
                "grid_gap_dex": max(gaps) if gaps else None,
                "at_cap_either": bool(c["at_cap"] or base["at_cap"]),
            })
        return out

    def summarise(rows: list[dict], label: str) -> dict[str, Any]:
        deltas = [r["delta_dex"] for r in rows]
        gaps = [r["grid_gap_dex"] for r in rows if r["grid_gap_dex"] is not None]
        mean_delta = float(np.mean(deltas)) if deltas else None
        floor = _median(gaps)
        if mean_delta is None or floor is None:
            outcome = "unresolved"
        elif abs(mean_delta - timescale_gap_dex) <= floor:
            outcome = "timescale-gap consistent"
        else:
            outcome = "unresolved"
        return {
            "operation": label,
            "per_prime": rows,
            "mean_delta_dex": mean_delta,
            "median_delta_dex": _median(deltas),
            "median_grid_gap_dex": floor,
            "timescale_pred_dex": timescale_gap_dex,
            "capacity_pred_below_floor": True,
            "outcome": outcome,
        }

    def f_star_of(cells: list[dict]) -> Optional[float]:
        return _median([c["f_onset"] for c in cells if c.get("f_onset")])

    return {
        "add_vs_div": summarise(ratios(add_cells), "+"),
        "mul_vs_div": summarise(ratios(mul_cells), "*"),
        "f_star_per_operation": {
            "/": f_star_of(div_cells),
            "+": f_star_of(add_cells),
            "*": f_star_of(mul_cells),
        },
    }


def evaluate_usability(alpha_result: dict, extrap_result: dict) -> dict[str, Any]:
    """Mechanical, pre-registered usability check for f*.

    f* is 'usable' iff all three hold:
      1. the alpha paired-difference slope CI includes 0 (flatness in alpha);
      2. the extrapolation paired comparison favours f* on >= 4 of 6 primes;
      3. the median extrapolation |residual| is below the median grid gap.
    """
    paired = alpha_result["with_alpha_0.5"]["paired_diff_slope"]
    ci_includes_zero = (
        paired.get("ci_low") is not None and paired.get("ci_high") is not None
        and paired["ci_low"] <= 0.0 <= paired["ci_high"]
    )
    n_closer = extrap_result["n_primes_fstar_closer"]
    n_test = extrap_result["n_test_primes"]
    favours = n_closer >= 4
    med_res = extrap_result["median_abs_residual_fstar_dex"]
    med_gap = extrap_result["median_grid_gap_test_primes_dex"]
    below_gap = (med_res is not None and med_gap is not None
                 and med_res < med_gap)
    return {
        "criterion_1_alpha_slope_ci_includes_0": bool(ci_includes_zero),
        "alpha_paired_slope": paired.get("slope"),
        "alpha_paired_slope_ci": [paired.get("ci_low"), paired.get("ci_high")],
        "criterion_2_extrapolation_favours_fstar": bool(favours),
        "n_primes_fstar_closer": f"{n_closer}/{n_test}",
        "criterion_3_median_residual_below_grid_gap": bool(below_gap),
        "median_abs_residual_dex": med_res,
        "median_grid_gap_dex": med_gap,
        "usable": bool(ci_includes_zero and favours and below_gap),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_AQUA = "#1baf7a"
_INK = "#0b0b0b"
_MUTED = "#52514e"


def _style_ax(ax) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(_MUTED)


def _prime_log_xticks(ax, p: np.ndarray) -> None:
    """Plain decimal ticks on a log prime axis — the range is narrow, so
    default log ticks collide and read poorly."""
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

    ticks = [t for t in (100, 110, 120, 130, 140, 150)
             if p.min() * 0.97 <= t <= p.max() * 1.03]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())


def plot_prime_flatness(result: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = result["per_prime"]
    p = np.array([r["p"] for r in rows], dtype=float)
    f = np.array([r["f_onset"] for r in rows], dtype=float)
    med = result["f_onset_median"]
    gap = result["median_grid_gap_dex"] or 0.0

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.axhspan(med * 10 ** (-gap), med * 10 ** gap, color=_MUTED, alpha=0.12,
               lw=0, label="median ± grid gap")
    ax.axhline(med, color=_MUTED, lw=1, ls="--")
    ax.plot(p, f, "o", color=_BLUE, ms=7, mec="white", mew=0.5,
            label=r"$f_\mathrm{onset}$ per prime")
    ols = result["flatness_ols_log10f_vs_log10p"]
    boot = result["flatness_slope_prime_bootstrap_ci"]
    if ols.get("slope") is not None:
        grid = np.linspace(p.min(), p.max(), 50)
        ax.plot(grid, 10 ** (ols["intercept"] + ols["slope"] * np.log10(grid)),
                color=_ORANGE, lw=2, label="OLS fit")
        ax.text(0.03, 0.05,
                f"slope = {ols['slope']:.2f} "
                f"[{boot['ci_low']:.2f}, {boot['ci_high']:.2f}] "
                "(prime bootstrap 95%)",
                transform=ax.transAxes, fontsize=8.5, color=_INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _prime_log_xticks(ax, p)
    ax.set_xlabel("Prime p", fontsize=10, color=_INK)
    ax.set_ylabel(r"Onset capacity fraction  $f_\mathrm{onset}$",
                  fontsize=10, color=_INK)
    _style_ax(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_alpha_residuals(result: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    cells = result["cells"]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    jitter = {0.3: -0.006, 0.4: -0.006, 0.5: -0.006, 0.6: -0.006, 0.7: -0.006}
    for r in cells:
        a = r["train_fraction"]
        face_f = _BLUE if not r["at_cap"] else "none"
        face_i = _ORANGE if not r["at_cap"] else "none"
        ax.plot(a + jitter.get(a, 0), r["r_fstar_dex"], "o", ms=6,
                mfc=face_f, mec=_BLUE, mew=1.1, alpha=0.85)
        ax.plot(a - jitter.get(a, 0), r["r_int_dex"], "s", ms=6,
                mfc=face_i, mec=_ORANGE, mew=1.1, alpha=0.85)
    ax.axhline(0, color=_MUTED, lw=1, ls="--")
    fits = result["with_alpha_0.5"]
    txt = (f"paired (f* − int.) slope = "
           f"{fits['paired_diff_slope']['slope']:.2f} dex/α  "
           f"[{fits['paired_diff_slope']['ci_low']:.2f}, "
           f"{fits['paired_diff_slope']['ci_high']:.2f}]")
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top",
            fontsize=8.5, color=_INK)
    ax.plot([], [], "o", mfc=_BLUE, mec=_BLUE, label=r"$r_{f^*}$ (fixed fraction)")
    ax.plot([], [], "s", mfc=_ORANGE, mec=_ORANGE, label=r"$r_\mathrm{int}$ (intersection)")
    ax.plot([], [], "o", mfc="none", mec=_MUTED, label="at dim cap (excluded)")
    ax.set_xlabel("Train fraction α", fontsize=10, color=_INK)
    ax.set_ylabel("Onset residual (dex)", fontsize=10, color=_INK)
    _style_ax(ax)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_extrapolation(result: dict, central_rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    p_all = np.array([r["p"] for r in central_rows], dtype=float)
    P_all = np.array([r["P_onset"] for r in central_rows], dtype=float)
    kmem = {r["p"]: r for r in central_rows}

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    calib_mask = np.isin(p_all, result["calibration_primes"])
    ax.plot(p_all[calib_mask], P_all[calib_mask], "o", color=_BLUE, ms=7,
            mec="white", mew=0.5, label="empirical (calibration)")
    ax.plot(p_all[~calib_mask], P_all[~calib_mask], "D", color=_INK, ms=6,
            mec="white", mew=0.5, label="empirical (held out)")

    grid_p = np.linspace(p_all.min() * 0.98, p_all.max() * 1.02, 200)
    fstar = result["f_star_calibrated"]
    # K_mem(p) for division at alpha=0.5: 0.5 * p(p-1) * log2(p+2).
    K = 0.5 * grid_p * (grid_p - 1) * np.log2(grid_p + 2)
    C = central_rows[0].get("C", 2.16) if central_rows else 2.16
    ax.plot(grid_p, K / (C * fstar), color=_AQUA, lw=2,
            label=r"$K_\mathrm{mem}/(C f^*)$")
    null = result["null_powerlaw"]
    ax.plot(grid_p, 10 ** (null["intercept"] + null["slope"] * np.log10(grid_p)),
            color=_ORANGE, lw=2, ls="--", label="power-law null")
    ax.axvline(111, color=_MUTED, lw=0.8, ls=":")
    ax.text(0.02, 0.97,
            f"MAE f* = {result['mae_fstar_dex']:.3f} dex   "
            f"MAE null = {result['mae_null_dex']:.3f} dex\n"
            f"f* closer on {result['n_primes_fstar_closer']}"
            f"/{result['n_test_primes']} held-out primes",
            transform=ax.transAxes, va="top", fontsize=8.5, color=_INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _prime_log_xticks(ax, p_all)
    ax.set_xlabel("Prime p", fontsize=10, color=_INK)
    ax.set_ylabel(r"Onset parameter count  $P_\mathrm{onset}$",
                  fontsize=10, color=_INK)
    _style_ax(ax)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    _ = kmem  # (per-prime K held for potential annotation; grid form used)


def plot_operation_ratios(result: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    add = result["add_vs_div"]["per_prime"]
    mul = result["mul_vs_div"]["per_prime"]
    # Small horizontal offset so coincident deltas (both often exactly 0
    # at the same prime) stay individually visible.
    for rows, color, marker, label, dx in (
        (add, _BLUE, "o", r"$\log_{10}(P^{+}/P^{/})$", -0.7),
        (mul, _ORANGE, "s", r"$\log_{10}(P^{*}/P^{/})$", +0.7),
    ):
        p = [r["p"] + dx for r in rows]
        d = [r["delta_dex"] for r in rows]
        ax.plot(p, d, marker, color=color, ms=7, mec="white", mew=0.5,
                label=label)
    floor = result["add_vs_div"]["median_grid_gap_dex"] or 0.0
    ax.axhspan(-floor, floor, color=_MUTED, alpha=0.12, lw=0)
    ax.axhline(0, color=_MUTED, lw=0.8)
    ts = result["add_vs_div"]["timescale_pred_dex"]
    ax.axhline(ts, color=_AQUA, lw=1.6, ls="--")
    ax.text(0.99, ts + 0.01, "timescale prediction", ha="right",
            fontsize=8, color=_AQUA, transform=ax.get_yaxis_transform())
    cap = float(np.mean([r["capacity_pred_dex"] for r in add + mul])) if add + mul else 0.004
    ax.axhline(cap, color=_MUTED, lw=1.2, ls=":")
    ax.text(0.99, cap + 0.01, "capacity prediction (below grid floor)",
            ha="right", fontsize=8, color=_MUTED,
            transform=ax.get_yaxis_transform())
    ax.set_xlabel("Prime p", fontsize=10, color=_INK)
    ax.set_ylabel("Onset shift vs division (dex)", fontsize=10, color=_INK)
    _style_ax(ax)
    ax.legend(fontsize=8, frameon=False, loc="center right")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------


def _fmt(v, spec: str = ".3f") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return format(v, spec)


def format_summary_markdown(results: dict[str, Any]) -> str:
    a = results["prime_flatness"]
    b = results["alpha_flatness"]
    c = results["extrapolation"]
    e = results["operation_ratios"]
    v = results["usability"]

    L: list[str] = []
    L.append("# Onset capacity fraction law — f_onset = K_mem / (C · P_onset)")
    L.append("")
    L.append("C = 2.16 bits/param (`consts.C`); P_onset located as in "
             "`stats._empirical_onset` (min-delay over seeds, first grid "
             "point past the last zero-delay one, dim ≤ 256, thresholds "
             "train 99 / val 98); P_cross from `stats._predicted_onset`.")
    L.append("")

    L.append("## A2a — f_onset across the central primes (division, α = 0.5)")
    L.append("")
    L.append("| p | P_onset | P_cross | f_onset | f_cross | gap below (dex) | n seeds |")
    L.append("|---|---|---|---|---|---|---|")
    for r in a["per_prime"]:
        L.append(f"| {r['p']} | {_fmt(r['P_onset'], ',.0f')} | "
                 f"{_fmt(r['P_cross'], ',.0f')} | {_fmt(r['f_onset'])} | "
                 f"{_fmt(r['f_cross'])} | {_fmt(r['gap_below_dex'])} | "
                 f"{r['n_seeds']} |")
    L.append("")
    cv_label = (f"{a['f_onset_cv']:.3f} (≤ floor — spread is at the grid "
                "resolution, not resolved scatter)"
                if a["cv_resolution_limited"] else f"{a['f_onset_cv']:.3f}")
    L.append(f"- f_onset: mean = {_fmt(a['f_onset_mean'])}, "
             f"median = {_fmt(a['f_onset_median'])}, CV = {cv_label}")
    L.append(f"- spread of log10 f_onset = {_fmt(a['f_onset_sd_log10_dex'])} dex "
             f"vs width-grid quantisation floor (median dex gap between "
             f"P_onset and the adjacent grid point) = "
             f"{_fmt(a['median_grid_gap_dex'])} dex")
    ols = a["flatness_ols_log10f_vs_log10p"]
    boot = a["flatness_slope_prime_bootstrap_ci"]
    L.append(f"- flatness in p: OLS slope of log10 f_onset on log10 p = "
             f"{_fmt(ols['slope'])} (95% prime-bootstrap CI "
             f"[{_fmt(boot['ci_low'])}, {_fmt(boot['ci_high'])}], "
             f"{boot['n_resamples']} resamples). No seed-level bootstrap is "
             f"run: the onset is a seed-min order statistic.")
    L.append("")

    L.append("## A2b — flatness in train fraction α")
    L.append("")
    L.append(f"f* = A2a median f_onset = {_fmt(b['f_star'])}. Residuals per "
             f"(α, p) cell: r_f* = log10(P_onset · C · f*/K_mem), "
             f"r_int = log10(P_onset/P_cross). "
             f"{b['n_excluded_at_cap']} cell(s) excluded with P_onset at the "
             f"dim-cap boundary (dim = 256): "
             + ", ".join(f"(α={x['train_fraction']}, p={x['p']})"
                         for x in b["excluded_cells"]) + ".")
    L.append("")
    L.append("| subset | n | paired (r_f* − r_int) slope on α [95% CI] | r_f* slope | r_int slope |")
    L.append("|---|---|---|---|---|")
    for label, key in (("with α=0.5", "with_alpha_0.5"),
                       ("without α=0.5", "without_alpha_0.5")):
        f = b[key]
        pd_, rf, ri = f["paired_diff_slope"], f["r_fstar_slope"], f["r_int_slope"]
        L.append(f"| {label} | {f['n_cells']} | {_fmt(pd_['slope'])} "
                 f"[{_fmt(pd_['ci_low'])}, {_fmt(pd_['ci_high'])}] | "
                 f"{_fmt(rf['slope'])} [{_fmt(rf['ci_low'])}, {_fmt(rf['ci_high'])}] | "
                 f"{_fmt(ri['slope'])} [{_fmt(ri['ci_low'])}, {_fmt(ri['ci_high'])}] |")
    L.append("")
    L.append("- α = 0.5 carries most of the leverage (11 primes × 10 seeds "
             "vs 5 primes × 4 seeds at the other α), hence both rows.")
    L.append("")

    L.append("## A2c — extrapolation across primes")
    L.append("")
    L.append(f"f* calibrated on the 5 smallest central primes "
             f"{list(c['calibration_primes'])}: f*_calib = "
             f"{_fmt(c['f_star_calibrated'])}. Null baseline: log-log power "
             f"law on the same 5 primes (slope = "
             f"{_fmt(c['null_powerlaw']['slope'])}).")
    L.append("")
    L.append("| p | P_onset | f* residual (dex) | null residual (dex) | f* closer |")
    L.append("|---|---|---|---|---|")
    for r in c["per_prime"]:
        L.append(f"| {r['p']} | {_fmt(r['P_onset'], ',.0f')} | "
                 f"{_fmt(r['residual_fstar_dex'])} | "
                 f"{_fmt(r['residual_null_dex'])} | "
                 f"{'yes' if r['fstar_closer'] else 'no'} |")
    L.append("")
    L.append(f"- MAE: f* = {_fmt(c['mae_fstar_dex'])} dex, "
             f"null = {_fmt(c['mae_null_dex'])} dex; paired comparison "
             f"favours f* on {c['n_primes_fstar_closer']}/"
             f"{c['n_test_primes']} held-out primes.")
    x = c["cross_axis"]
    L.append(f"- Cross-axis: predicting every (α ≠ 0.5, p) cell with the "
             f"α = 0.5 f* gives MAE = {_fmt(x['mae_dex'])} dex over "
             f"{x['n_cells']} cells ({x['n_excluded_at_cap']} dim-cap cell(s) "
             f"excluded). The p-only power-law null cannot make this "
             f"prediction, so its MAE stands alone with no f* comparison.")
    L.append("")

    L.append("## A2e — operation ratios (matched hardware and seeds)")
    L.append("")
    for key, label in (("add_vs_div", "+ vs /"), ("mul_vs_div", "* vs /")):
        s = e[key]
        per = ", ".join(f"p={r['p']}: {_fmt(r['delta_dex'], '+.3f')}"
                        for r in s["per_prime"])
        L.append(f"- **{label}**: per-prime onset shift (dex): {per}. "
                 f"Mean = {_fmt(s['mean_delta_dex'], '+.3f')} dex, median "
                 f"grid floor = {_fmt(s['median_grid_gap_dex'])} dex. "
                 f"Timescale prediction ≈ +{s['timescale_pred_dex']:.2f} dex; "
                 f"capacity-only prediction ≈ "
                 f"+{s['per_prime'][0]['capacity_pred_dex']:.4f} dex — below "
                 f"the grid floor, hence unmeasurable in principle here. "
                 f"Outcome: **{s['outcome']}**.")
    fso = e["f_star_per_operation"]
    L.append(f"- f* per operation (context): "
             f"'/' (B) = {_fmt(fso.get('/'))}, '+' = {_fmt(fso.get('+'))}, "
             f"'*' = {_fmt(fso.get('*'))}"
             + (f", '/' (central, A) = {_fmt(fso.get('central_div'))}"
                if fso.get("central_div") else "") + ".")
    L.append("")

    L.append("## Usability check (pre-registered, mechanical)")
    L.append("")
    L.append(f"1. α paired slope CI includes 0: "
             f"**{'yes' if v['criterion_1_alpha_slope_ci_includes_0'] else 'no'}** "
             f"(slope = {_fmt(v['alpha_paired_slope'])}, CI = "
             f"[{_fmt(v['alpha_paired_slope_ci'][0])}, "
             f"{_fmt(v['alpha_paired_slope_ci'][1])}])")
    L.append(f"2. extrapolation favours f* on ≥ 4/6 primes: "
             f"**{'yes' if v['criterion_2_extrapolation_favours_fstar'] else 'no'}** "
             f"({v['n_primes_fstar_closer']})")
    L.append(f"3. median extrapolation |residual| < median grid gap: "
             f"**{'yes' if v['criterion_3_median_residual_below_grid_gap'] else 'no'}** "
             f"({_fmt(v['median_abs_residual_dex'])} vs "
             f"{_fmt(v['median_grid_gap_dex'])} dex)")
    L.append("")
    L.append(f"**A4 VERDICT: {'usable' if v['usable'] else 'not-usable'}**")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_kv(pairs: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs or []:
        k, _, raw = pair.partition("=")
        try:
            v: Any = int(raw)
        except ValueError:
            try:
                v = float(raw)
            except ValueError:
                v = raw
        out[k] = v
    return out


def _cmd_extract(args: argparse.Namespace) -> None:
    table = extract_onset_cells(
        args.config, args.db,
        figure_index=args.figure_index,
        group_filters=_parse_kv(args.group_filter),
        group_excludes=_parse_kv(args.group_exclude),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(table, f, indent=2)
    print(f"wrote {len(table['cells'])} cells -> {out}")


def _load_cells(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)["cells"]


def _cmd_report(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    central = _load_cells(args.central)
    alpha = _load_cells(args.alpha) + [c for c in central
                                       if c["train_fraction"] == 0.5]
    div_b = [c for c in _load_cells(args.div)
             if c["operation"] == "/" and c["init_scale"] == 1.0]
    add_b = [c for c in _load_cells(args.add) if c["operation"] == "+"]
    mul_b = [c for c in _load_cells(args.mul) if c["operation"] == "*"]

    a2a = analyze_prime_flatness(central, n_bootstrap=args.n_bootstrap)
    f_star = a2a["f_onset_median"]
    a2b = analyze_alpha_flatness(alpha, f_star=f_star)
    a2c = analyze_extrapolation(central, alpha, f_star_all_primes=f_star,
                                n_bootstrap=args.n_bootstrap)
    a2e = analyze_operation_ratios(div_b, add_b, mul_b,
                                   timescale_gap_dex=args.timescale_gap_dex)
    a2e["f_star_per_operation"]["central_div"] = f_star
    verdict = evaluate_usability(a2b, a2c)

    results = {
        "capacity_constant": float(CAPACITY_C),
        "f_star": f_star,
        "prime_flatness": a2a,
        "alpha_flatness": a2b,
        "extrapolation": a2c,
        "operation_ratios": a2e,
        "usability": verdict,
    }
    with open(out_dir / "onset_law_results.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_prime_flatness(a2a, out_dir / "a2a_f_onset_vs_p.pdf")
    plot_alpha_residuals(a2b, out_dir / "a2b_residuals_vs_alpha.pdf")
    plot_extrapolation(a2c, a2a["per_prime"], out_dir / "a2c_extrapolation.pdf")
    plot_operation_ratios(a2e, out_dir / "a2e_operation_ratios.pdf")

    md = format_summary_markdown(results)
    if args.md_out:
        md_path = Path(args.md_out)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md)
    print(md)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="dump per-(group, prime) onset cells")
    ex.add_argument("--config", required=True)
    ex.add_argument("--db", required=True)
    ex.add_argument("--figure-index", type=int, default=0)
    ex.add_argument("--group-filter", action="append", default=[],
                    metavar="FIELD=VALUE",
                    help="keep only groups whose key matches (repeatable)")
    ex.add_argument("--group-exclude", action="append", default=[],
                    metavar="FIELD=VALUE",
                    help="drop groups whose key matches (repeatable)")
    ex.add_argument("--out", required=True)
    ex.set_defaults(func=_cmd_extract)

    rp = sub.add_parser("report", help="combine cell tables into the analyses")
    rp.add_argument("--central", required=True, help="central division cells")
    rp.add_argument("--alpha", required=True, help="train-fraction sweep cells")
    rp.add_argument("--div", required=True, help="division baseline cells (matched hw)")
    rp.add_argument("--add", required=True, help="addition cells")
    rp.add_argument("--mul", required=True, help="multiplication cells")
    rp.add_argument("--out-dir", required=True)
    rp.add_argument("--md-out", default=None)
    rp.add_argument("--n-bootstrap", type=int, default=5000)
    rp.add_argument("--timescale-gap-dex", type=float, default=TIMESCALE_GAP_DEX)
    rp.set_defaults(func=_cmd_report)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
