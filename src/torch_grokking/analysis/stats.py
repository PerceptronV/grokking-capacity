"""Predictiveness analysis: does the speed/groks intersection actually
predict where grokking onset happens empirically?

Per (ArchGroup × slice value) cell, for each declared intersection figure,
we compute:
  - predicted_onset_x: x-coordinate of the speed/groks intersection
  - empirical_onset_x: smallest x past the last zero-delay run, with
    delays taken as min over compatible seeds (matches the visualise.py
    convention)
And report log-ratio = log10(empirical / predicted) — zero is perfect, the
sign tells which side missed.

The "x" here is whatever `IntersectionFigure.x_field` says (param_count
for the canonical figure; dataset_bits or p when the figure swaps roles).

In addition to the descriptive scatter/CSV, this module runs a formal
hypothesis-test suite per intersection figure (see
`.claude/plans/statistics.md` for the design). Output is
`hypothesis_tests.{json,md}` next to `predictiveness.csv`.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from . import aggregate
from .config_view import ArchGroup, ConfigView, IntersectionFigure, StatsConfig
from .plots import (
    DELAY_TRAIN_THRESHOLD,
    DELAY_VAL_THRESHOLD,
    _delay_records_for_slice,
    _curve_for_slice,
    _slice_values,
)


def _empirical_onset(group: ArchGroup, figure: IntersectionFigure, slice_value) -> Optional[float]:
    records = _delay_records_for_slice(group, figure, slice_value)
    if not records:
        return None
    pairs = [(r["x"], r["delay"]) for r in records]
    return aggregate.find_grokking_onset(aggregate.min_delay_curve(pairs))


def _predicted_onset(group: ArchGroup, figure: IntersectionFigure, slice_value) -> Optional[float]:
    speed = _curve_for_slice(group.speed_runs, figure, slice_value, "saturation_epoch")
    groks = _curve_for_slice(group.groks_runs, figure, slice_value, "grokking_epoch")
    pt = aggregate.find_intersection(speed, groks)
    return pt[0] if pt is not None else None


def compute_predictiveness(view: ConfigView, figure: IntersectionFigure) -> pd.DataFrame:
    """One row per (group, slice value) cell of one figure family.

    Columns: config, figure, slice_field, slice_value, x_field,
    predicted_onset_x, empirical_onset_x, log_ratio, capacity_constant,
    capacity_source, n_seeds_groks, n_seeds_speed, plus every swept-axis
    identifying field.
    """
    rows: list[dict] = []
    for group in view.iter_groups():
        for sv in _slice_values(group, figure):
            predicted = _predicted_onset(group, figure, sv)
            empirical = _empirical_onset(group, figure, sv)
            n_groks = sum(1 for r in group.groks_runs
                          if r.get(figure.slice_field) == sv)
            n_speed = sum(1 for r in group.speed_runs
                          if r.get(figure.slice_field) == sv)
            log_ratio = (
                float(np.log10(empirical / predicted))
                if (empirical and predicted and predicted > 0)
                else None
            )
            row = {
                "config": view.config_name,
                "figure": figure.name,
                "slice_field": figure.slice_field,
                "slice_value": sv,
                "x_field": figure.x_field,
                "predicted_onset_x": predicted,
                "empirical_onset_x": empirical,
                "log_ratio": log_ratio,
                "capacity_constant": group.capacity_constant,
                "capacity_source": group.capacity_constant_source,
                "n_seeds_groks": n_groks,
                "n_seeds_speed": n_speed,
            }
            for ax in view.swept_axes:
                row[ax] = getattr(group.key, ax, None)
            rows.append(row)
    return pd.DataFrame(rows)


def save_predictiveness_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot_predicted_vs_empirical(df: pd.DataFrame, save_path: Path) -> Optional[Path]:
    """Log-log scatter, identity line, R² + MAE in log-space corner box."""
    if not {"predicted_onset_x", "empirical_onset_x"}.issubset(df.columns):
        return None
    valid = df.dropna(subset=["predicted_onset_x", "empirical_onset_x"])
    if len(valid) < 2:
        return None
    x = valid["predicted_onset_x"].to_numpy(dtype=float)
    y = valid["empirical_onset_x"].to_numpy(dtype=float)
    keep = (x > 0) & (y > 0)
    x, y = x[keep], y[keep]
    if len(x) < 2:
        return None

    log_x, log_y = np.log10(x), np.log10(y)
    residuals = log_y - log_x
    mae_log = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    x_field = str(valid["x_field"].iloc[0]) if "x_field" in valid.columns else "x"

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, s=80, alpha=0.7, edgecolors="black", linewidths=0.5)
    lo, hi = float(min(x.min(), y.min())) * 0.7, float(max(x.max(), y.max())) * 1.3
    ax.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.7, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"Predicted onset (intersection {x_field})", fontsize=13)
    ax.set_ylabel(f"Empirical onset (smallest non-zero-delay {x_field})", fontsize=13)
    ax.text(
        0.05, 0.95,
        f"N = {len(x)}\nR² (log-log vs y=x) = {r2:.3f}\nMAE (log10) = {mae_log:.3f}",
        transform=ax.transAxes, fontsize=11, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
    )
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    return save_path


def plot_error_vs_axis(df: pd.DataFrame, axis: str, save_path: Path) -> Optional[Path]:
    """Per-axis breakdown: log_ratio vs sweep value, coloured by slice."""
    if axis not in df.columns or "log_ratio" not in df.columns:
        return None
    valid = df.dropna(subset=["log_ratio", axis])
    if valid.empty:
        return None

    slice_field = str(valid["slice_field"].iloc[0]) if "slice_field" in valid.columns else "slice"
    fig, ax = plt.subplots(figsize=(10, 6))
    slice_values = sorted(valid["slice_value"].unique())
    palette = plt.cm.viridis(np.linspace(0, 1, max(len(slice_values), 1)))
    for color, sv in zip(palette, slice_values):
        sub = valid[valid["slice_value"] == sv]
        ax.scatter(
            sub[axis], sub["log_ratio"],
            s=70, alpha=0.8, color=color, edgecolors="black", linewidths=0.5,
            label=f"{slice_field}={sv}",
        )
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel(axis, fontsize=13)
    ax.set_ylabel("log10(empirical / predicted)", fontsize=13)
    ax.legend(title=slice_field.title(), fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    return save_path


def render_stats(view: ConfigView, out_dir: Path) -> dict[str, Path]:
    """End-to-end per intersection figure: CSV, scatter, per-axis plots,
    plus the formal hypothesis-test suite (`hypothesis_tests.{json,md}`)
    when `view.stats.enabled`.

    Each figure's outputs go to `out_dir/<figure.name>/` so a config with
    multiple intersection figures gets one self-contained subdir each.
    """
    out_dir = Path(out_dir)
    paths: dict[str, Path] = {}
    for figure in view.intersection_figures:
        sub = out_dir / figure.name
        df = compute_predictiveness(view, figure)
        csv_path = sub / "predictiveness.csv"
        save_predictiveness_csv(df, csv_path)
        paths[f"{figure.name}/csv"] = csv_path
        scatter = plot_predicted_vs_empirical(df, sub / "predicted_vs_empirical.png")
        if scatter is not None:
            paths[f"{figure.name}/scatter"] = scatter
        for axis in view.swept_axes:
            p = plot_error_vs_axis(df, axis, sub / f"error_vs_{axis}.png")
            if p is not None:
                paths[f"{figure.name}/axis:{axis}"] = p
        if view.stats.enabled:
            test_paths = render_hypothesis_tests(view, figure, df, sub)
            for label, p in test_paths.items():
                paths[f"{figure.name}/{label}"] = p
    return paths


# ---------------------------------------------------------------------------
# Formal hypothesis-test suite (see .claude/plans/statistics.md).
# ---------------------------------------------------------------------------


def _valid_log_pairs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (log10 predicted, log10 empirical, valid_df) for cells where
    both onsets are positive and finite. Other tests build off the same
    valid mask so every test is run on the same N. An empty df (no
    completed runs in this config yet) returns three empty containers
    instead of raising."""
    required = {"predicted_onset_x", "empirical_onset_x"}
    if not required.issubset(df.columns):
        return np.empty(0), np.empty(0), df.iloc[0:0].copy()
    valid = df.dropna(subset=list(required)).copy()
    if valid.empty:
        return np.empty(0), np.empty(0), valid
    p = valid["predicted_onset_x"].to_numpy(dtype=float)
    e = valid["empirical_onset_x"].to_numpy(dtype=float)
    keep = np.isfinite(p) & np.isfinite(e) & (p > 0) & (e > 0)
    valid = valid.iloc[keep].reset_index(drop=True)
    return np.log10(p[keep]), np.log10(e[keep]), valid


def _spearman_and_permutation(
    log_p: np.ndarray, log_e: np.ndarray, *, n_permutations: int, seed: int = 0,
) -> dict[str, Any]:
    """Spearman ρ with both analytic and permutation p-values.

    Bails early if either input is constant (ρ undefined): scipy emits a
    warning and returns nan in that case, and the permutation loop would
    spam the warning n_permutations times to no informational purpose."""
    if len(log_p) < 3:
        return {"rho": None, "p_analytic": None, "p_permutation": None,
                "n_permutations": 0, "n": int(len(log_p)),
                "skipped": "n<3"}
    if np.std(log_p) == 0 or np.std(log_e) == 0:
        return {"rho": None, "p_analytic": None, "p_permutation": None,
                "n_permutations": 0, "n": int(len(log_p)),
                "skipped": "constant_input"}
    # scipy may still raise ConstantInputWarning when ranks tie out under a
    # specific permutation even though the underlying values aren't constant.
    # Silenced narrowly here (and in _kendall): the std==0 short-circuit
    # already covers the genuinely-degenerate case.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*[Cc]onstant.*")
        rho, p_an = scipy_stats.spearmanr(log_p, log_e)
        if not np.isfinite(rho):
            return {"rho": None, "p_analytic": None, "p_permutation": None,
                    "n_permutations": 0, "n": int(len(log_p)),
                    "skipped": "rho_nan"}
        rng = np.random.default_rng(seed)
        obs = float(rho)
        n_ge = 0
        perm = log_e.copy()
        for _ in range(n_permutations):
            rng.shuffle(perm)
            r, _ = scipy_stats.spearmanr(log_p, perm)
            if np.isfinite(r) and abs(r) >= abs(obs):
                n_ge += 1
    p_perm = (n_ge + 1) / (n_permutations + 1)  # +1 smoothing
    return {"rho": float(obs), "p_analytic": float(p_an),
            "p_permutation": float(p_perm), "n_permutations": int(n_permutations),
            "n": int(len(log_p))}


def _kendall(log_p: np.ndarray, log_e: np.ndarray) -> dict[str, Any]:
    if len(log_p) < 3:
        return {"tau": None, "p_analytic": None, "skipped": "n<3"}
    if np.std(log_p) == 0 or np.std(log_e) == 0:
        return {"tau": None, "p_analytic": None, "skipped": "constant_input"}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*[Cc]onstant.*")
        tau, p_an = scipy_stats.kendalltau(log_p, log_e)
    if not np.isfinite(tau):
        return {"tau": None, "p_analytic": None, "skipped": "tau_nan"}
    return {"tau": float(tau), "p_analytic": float(p_an)}


def _ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient. Penalises both
    decorrelation and offset from the y=x line."""
    if len(x) < 2:
        return float("nan")
    mx, my = float(np.mean(x)), float(np.mean(y))
    vx, vy = float(np.var(x)), float(np.var(y))
    cov = float(np.mean((x - mx) * (y - my)))
    denom = vx + vy + (mx - my) ** 2
    if denom <= 0:
        return float("nan")
    return 2.0 * cov / denom


def _ccc_with_bootstrap(
    log_p: np.ndarray, log_e: np.ndarray, *, n_bootstrap: int, seed: int = 0,
) -> dict[str, Any]:
    if len(log_p) < 3:
        return {"value": None, "ci_low": None, "ci_high": None,
                "n_bootstrap": 0, "skipped": "n<3"}
    obs = _ccc(log_p, log_e)
    rng = np.random.default_rng(seed)
    n = len(log_p)
    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[i] = _ccc(log_p[idx], log_e[idx])
    boot = boot[np.isfinite(boot)]
    if boot.size < 2:
        return {"value": float(obs), "ci_low": None, "ci_high": None,
                "n_bootstrap": int(n_bootstrap), "skipped": "bootstrap_degenerate"}
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"value": float(obs), "ci_low": float(lo), "ci_high": float(hi),
            "n_bootstrap": int(n_bootstrap)}


def _ols_fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Plain OLS via least-squares. Returns (β̂, RSS, df_residual)."""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid)
    df = int(len(y) - X.shape[1])
    return beta, rss, df


def _joint_calibration_ftest(log_p: np.ndarray, log_e: np.ndarray) -> dict[str, Any]:
    """Joint F-test on H0: (a, b) = (0, 1) in `log_e = a + b·log_p`."""
    n = len(log_p)
    if n < 4:
        return {"slope": None, "intercept": None,
                "f": None, "df_num": None, "df_den": None, "p": None,
                "rss_full": None, "rss_restricted": None,
                "skipped": "n<4"}
    X = np.column_stack([np.ones(n), log_p])
    beta, rss_full, df = _ols_fit(X, log_e)
    rss_restricted = float(np.sum((log_e - log_p) ** 2))
    if df <= 0 or rss_full <= 0:
        return {"slope": float(beta[1]), "intercept": float(beta[0]),
                "f": None, "df_num": 2, "df_den": int(df), "p": None,
                "rss_full": float(rss_full),
                "rss_restricted": float(rss_restricted),
                "skipped": "rss_full=0_or_df<=0"}
    f = ((rss_restricted - rss_full) / 2.0) / (rss_full / df)
    f = max(f, 0.0)
    p = float(scipy_stats.f.sf(f, 2, df))
    return {"slope": float(beta[1]), "intercept": float(beta[0]),
            "f": float(f), "df_num": 2, "df_den": int(df), "p": float(p),
            "rss_full": float(rss_full),
            "rss_restricted": float(rss_restricted)}


def _wilcoxon_log_ratio(log_ratio: np.ndarray) -> dict[str, Any]:
    """Wilcoxon signed-rank on log_ratio vs 0 — symmetric-bias detector."""
    finite = log_ratio[np.isfinite(log_ratio)]
    if finite.size < 6:
        return {"median_log_ratio": float(np.median(finite)) if finite.size else None,
                "w": None, "p": None, "n": int(finite.size),
                "skipped": "n<6"}
    # Drop exact zeros — wilcoxon's default zero_method="wilcox" already
    # does, but be explicit so the reported n matches the test n.
    nz = finite[finite != 0]
    if nz.size < 6:
        return {"median_log_ratio": float(np.median(finite)),
                "w": None, "p": None, "n": int(nz.size),
                "skipped": "n_nonzero<6"}
    res = scipy_stats.wilcoxon(nz, alternative="two-sided",
                               zero_method="wilcox", correction=False,
                               mode="auto")
    return {"median_log_ratio": float(np.median(finite)),
            "w": float(res.statistic), "p": float(res.pvalue),
            "n": int(nz.size)}


def _design_matrix_for_axes(
    valid: pd.DataFrame, axes: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Build a design matrix for the sceptic baseline. Numeric columns go
    in as-is; non-numeric/categorical columns get one-hot encoded with the
    first category dropped (intercept absorbs it). Returns (X without
    intercept, predictor_names) — caller prepends the intercept column."""
    cols: list[np.ndarray] = []
    names: list[str] = []
    for ax in axes:
        if ax not in valid.columns:
            continue
        s = valid[ax]
        if s.isna().all():
            continue
        # Numeric? Cast and use directly.
        try:
            num = pd.to_numeric(s)
            if num.notna().all():
                cols.append(num.to_numpy(dtype=float))
                names.append(ax)
                continue
        except (ValueError, TypeError):
            pass
        # Otherwise one-hot encode (drop first level).
        levels = sorted(s.dropna().unique())
        if len(levels) <= 1:
            continue
        for lvl in levels[1:]:
            cols.append((s == lvl).to_numpy(dtype=float))
            names.append(f"{ax}={lvl}")
    if not cols:
        return np.empty((len(valid), 0)), names
    return np.column_stack(cols), names


def _baseline_lr_table(
    log_p: np.ndarray, log_e: np.ndarray, valid: pd.DataFrame,
    *, axes: list[str],
) -> dict[str, Any]:
    """Nested OLS comparisons: M0 (null), M1 (intersection), M2 (axes),
    M3 (combined). Returns model fits and the three nested F-tests."""
    n = len(log_e)
    if n < 4:
        return {"models": {}, "ftests": {}, "skipped": "n<4"}

    X_axes, axis_names = _design_matrix_for_axes(valid, axes)

    def fit(X: np.ndarray) -> dict[str, Any]:
        beta, rss, df = _ols_fit(X, log_e)
        ss_tot = float(np.sum((log_e - log_e.mean()) ** 2))
        r2 = 1.0 - rss / ss_tot if ss_tot > 0 else 0.0
        k = X.shape[1] - 1  # predictors excluding intercept
        r2_adj = (1 - (1 - r2) * (n - 1) / (n - 1 - k)) if (n - 1 - k) > 0 else None
        return {"rss": float(rss), "df": int(df),
                "r2": float(r2),
                "r2_adj": float(r2_adj) if r2_adj is not None else None,
                "n_predictors": int(k), "beta": beta}

    intercept = np.ones((n, 1))
    X0 = intercept
    X1 = np.column_stack([intercept, log_p])
    X2 = np.column_stack([intercept, X_axes]) if X_axes.shape[1] > 0 else None
    X3 = (np.column_stack([intercept, log_p, X_axes])
          if X_axes.shape[1] > 0 else None)

    fits: dict[str, dict[str, Any]] = {"M0": fit(X0), "M1": fit(X1)}
    fits["M0"]["predictors"] = []
    fits["M1"]["predictors"] = ["log10(predicted_onset_x)"]

    skipped: list[str] = []
    # M2 needs at least one residual df; otherwise it's saturated/degenerate.
    if X2 is not None and X2.shape[1] < n:
        fits["M2"] = fit(X2)
        fits["M2"]["predictors"] = list(axis_names)
    else:
        skipped.append("M2")
    if X3 is not None and X3.shape[1] < n:
        fits["M3"] = fit(X3)
        fits["M3"]["predictors"] = ["log10(predicted_onset_x)", *axis_names]
    else:
        skipped.append("M3")

    def f_test(small: str, big: str) -> Optional[dict[str, Any]]:
        if small not in fits or big not in fits:
            return None
        s, b = fits[small], fits[big]
        df_num = s["df"] - b["df"]
        df_den = b["df"]
        if df_num <= 0 or df_den <= 0 or b["rss"] <= 0:
            return {"f": None, "df_num": int(df_num), "df_den": int(df_den),
                    "p": None, "skipped": "df<=0_or_rss=0"}
        f = ((s["rss"] - b["rss"]) / df_num) / (b["rss"] / df_den)
        f = max(f, 0.0)
        return {"f": float(f), "df_num": int(df_num), "df_den": int(df_den),
                "p": float(scipy_stats.f.sf(f, df_num, df_den))}

    ftests = {
        "M1_vs_M0": f_test("M0", "M1"),
        "M3_vs_M2": f_test("M2", "M3"),
        "M3_vs_M1": f_test("M1", "M3"),
    }

    # Strip beta coefficients (numpy arrays) before serialisation.
    out_models: dict[str, dict[str, Any]] = {}
    for k, v in fits.items():
        out_models[k] = {kk: vv for kk, vv in v.items() if kk != "beta"}

    out: dict[str, Any] = {"models": out_models,
                           "ftests": {k: v for k, v in ftests.items() if v is not None}}
    if skipped:
        out["skipped_models"] = skipped
        out["skipped_reason"] = "more predictors than observations"
    return out


def _holm_correction(p_raw: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values for a list of raw p's."""
    k = len(p_raw)
    if k == 0:
        return []
    order = sorted(range(k), key=lambda i: p_raw[i])
    adj = [0.0] * k
    running_max = 0.0
    for rank, i in enumerate(order):
        # i-th smallest (0-indexed) compares against alpha / (k - rank).
        v = min(1.0, p_raw[i] * (k - rank))
        running_max = max(running_max, v)
        adj[i] = running_max
    return adj


def _axis_residual_test(
    valid: pd.DataFrame, axis: str, log_ratio: np.ndarray,
) -> Optional[dict[str, Any]]:
    """Single-axis regression of log_ratio on axis. Numeric → slope t-test;
    categorical → F-test on dummy variables. Returns None if the axis has
    fewer than two distinct values."""
    if axis not in valid.columns:
        return None
    s = valid[axis]
    finite_mask = np.isfinite(log_ratio) & s.notna()
    if finite_mask.sum() < 4:
        return None
    s_f = s[finite_mask]
    y = log_ratio[finite_mask.to_numpy()]
    n = len(y)

    # Numeric path.
    try:
        num = pd.to_numeric(s_f)
    except (ValueError, TypeError):
        num = None
    if num is not None and num.notna().all() and num.nunique() >= 2:
        x = num.to_numpy(dtype=float)
        X = np.column_stack([np.ones(n), x])
        beta, rss, df = _ols_fit(X, y)
        if df <= 0 or rss <= 0:
            return None
        # Var(slope) = σ² · (XᵀX)⁻¹[1,1].
        sigma2 = rss / df
        try:
            xtx_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            return None
        var_slope = float(sigma2 * xtx_inv[1, 1])
        if var_slope <= 0:
            return None
        se = float(np.sqrt(var_slope))
        t = float(beta[1] / se)
        p = float(2 * scipy_stats.t.sf(abs(t), df))
        return {"kind": "numeric", "slope": float(beta[1]), "se": se,
                "t": t, "p_raw": p, "n": n,
                "n_levels": int(num.nunique())}

    # Categorical path: F-test on dummy fit vs intercept-only.
    levels = sorted(s_f.dropna().unique())
    if len(levels) < 2:
        return None
    cols = [(s_f == lvl).to_numpy(dtype=float) for lvl in levels[1:]]
    X = np.column_stack([np.ones(n), *cols])
    _, rss_full, df_full = _ols_fit(X, y)
    rss_null = float(np.sum((y - y.mean()) ** 2))
    df_num = X.shape[1] - 1
    if df_full <= 0 or rss_full <= 0 or df_num <= 0:
        return None
    f = ((rss_null - rss_full) / df_num) / (rss_full / df_full)
    f = max(f, 0.0)
    p = float(scipy_stats.f.sf(f, df_num, df_full))
    return {"kind": "categorical", "f": float(f),
            "df_num": int(df_num), "df_den": int(df_full),
            "p_raw": float(p), "n": int(n), "n_levels": int(len(levels))}


def _robustness_axes(
    valid: pd.DataFrame, log_ratio: np.ndarray, *, axes: list[str],
    correction: str,
) -> dict[str, Any]:
    per_axis: dict[str, dict[str, Any]] = {}
    raw_p: list[float] = []
    keys: list[str] = []
    for ax in axes:
        res = _axis_residual_test(valid, ax, log_ratio)
        if res is None:
            continue
        per_axis[ax] = res
        raw_p.append(res["p_raw"])
        keys.append(ax)

    if correction == "holm":
        adj = _holm_correction(raw_p)
        for k, p_adj in zip(keys, adj):
            per_axis[k]["p_holm"] = float(p_adj)
    else:
        # Field is future-proof; today only "holm" is supported. Fall back
        # to raw with a noted absence of correction.
        for k in keys:
            per_axis[k]["p_holm"] = None

    return {"correction": correction, "axes": per_axis}


def compute_hypothesis_tests(
    view: ConfigView, figure: IntersectionFigure, df: pd.DataFrame,
) -> dict[str, Any]:
    """Run the full test suite for one figure's predictiveness table.

    Returns a nested dict ready for JSON serialisation; `format_hypothesis_tests_markdown`
    turns the same dict into a human-readable report.
    """
    cfg: StatsConfig = view.stats
    log_p, log_e, valid = _valid_log_pairs(df)
    log_ratio = log_e - log_p

    out: dict[str, Any] = {
        "figure": figure.name,
        "config": view.config_name,
        "n_cells": int(len(df)),
        "n_valid": int(len(valid)),
        "alpha": float(cfg.alpha),
    }

    if len(valid) < 3:
        out["skipped"] = f"n_valid={len(valid)} < 3 cells; tests omitted"
        return out

    out["predictiveness"] = {
        "spearman": _spearman_and_permutation(
            log_p, log_e, n_permutations=cfg.n_permutations,
        ),
        "kendall": _kendall(log_p, log_e),
    }
    out["calibration"] = {
        "ccc": _ccc_with_bootstrap(log_p, log_e, n_bootstrap=cfg.n_bootstrap),
        "joint_ftest": _joint_calibration_ftest(log_p, log_e),
        "wilcoxon": _wilcoxon_log_ratio(log_ratio),
    }

    if cfg.baseline_predictors == "auto":
        axes = list(view.swept_axes) + (
            [figure.slice_field] if figure.slice_field not in view.swept_axes else []
        )
    else:
        axes = list(cfg.baseline_predictors)
    out["sufficiency"] = _baseline_lr_table(log_p, log_e, valid, axes=axes)

    out["robustness"] = _robustness_axes(
        valid, log_ratio, axes=axes, correction=cfg.multiple_comparisons,
    )

    return out


def _verdict(p: Optional[float], alpha: float, *, want_reject: bool) -> str:
    """Tiny helper for the markdown summary's verdict glyph."""
    if p is None:
        return "—"
    if want_reject:
        return "✓" if p < alpha else "✗"
    return "✓" if p >= alpha else "✗"


def _fmt(v: Any, spec: str = ".3g") -> str:
    """Format a number, returning '—' for None/NaN so the markdown report
    doesn't blow up on missing fields when a test was skipped."""
    if v is None:
        return "—"
    try:
        if isinstance(v, float) and not np.isfinite(v):
            return "—"
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def format_hypothesis_tests_markdown(results: dict[str, Any]) -> str:
    """Compact human summary of the JSON results. Each section ends with a
    verdict glyph: ✓ when the sub-claim survives, ✗ when it doesn't.

    The verdicts are deliberately rough — read the numbers, don't argue
    with the glyphs. They're there to make the report scannable."""
    lines: list[str] = []
    lines.append(f"# Hypothesis tests — `{results.get('figure', '?')}`")
    lines.append("")
    lines.append(f"- config: `{results.get('config', '?')}`")
    lines.append(f"- N cells (total / valid): "
                 f"{results.get('n_cells')} / {results.get('n_valid')}")
    alpha = results.get("alpha", 0.05)
    lines.append(f"- α = {alpha}")
    lines.append("")

    if "skipped" in results:
        lines.append(f"_Skipped: {results['skipped']}_")
        return "\n".join(lines) + "\n"

    pred = results.get("predictiveness", {})
    sp = pred.get("spearman", {})
    kd = pred.get("kendall", {})
    sp_p = sp.get("p_permutation") if sp.get("p_permutation") is not None else sp.get("p_analytic")
    lines.append(f"## (1) Predictiveness  {_verdict(sp_p, alpha, want_reject=True)}")
    lines.append(f"- Spearman ρ = {_fmt(sp.get('rho'), '.3f')}, "
                 f"p_analytic = {_fmt(sp.get('p_analytic'))}, "
                 f"p_permutation = {_fmt(sp.get('p_permutation'))} "
                 f"(N_perm = {sp.get('n_permutations')})")
    lines.append(f"- Kendall τ = {_fmt(kd.get('tau'), '.3f')}, p = {_fmt(kd.get('p_analytic'))}")
    lines.append("")

    cal = results.get("calibration", {})
    ccc = cal.get("ccc", {})
    jt = cal.get("joint_ftest", {})
    wx = cal.get("wilcoxon", {})
    cal_p = jt.get("p")
    lines.append(f"## (2) Calibration  {_verdict(cal_p, alpha, want_reject=False)}")
    lines.append(f"- Lin's CCC = {_fmt(ccc.get('value'), '.3f')} "
                 f"(95% CI [{_fmt(ccc.get('ci_low'), '.3f')}, "
                 f"{_fmt(ccc.get('ci_high'), '.3f')}], "
                 f"N_boot = {ccc.get('n_bootstrap')})")
    lines.append(f"- log_e ~ a + b·log_p̂: a = {_fmt(jt.get('intercept'), '.3f')}, "
                 f"b = {_fmt(jt.get('slope'), '.3f')}; "
                 f"joint F-test (a=0, b=1) F = {_fmt(jt.get('f'))}, "
                 f"p = {_fmt(jt.get('p'))}")
    lines.append(f"- Wilcoxon signed-rank on log_ratio: "
                 f"median = {_fmt(wx.get('median_log_ratio'), '.3f')}, "
                 f"p = {_fmt(wx.get('p'))}")
    lines.append("")

    suf = results.get("sufficiency", {})
    models = suf.get("models", {})
    ftests = suf.get("ftests", {})
    lines.append("## (3) Sufficiency vs baselines")
    lines.append("")
    if models:
        lines.append("| Model | Predictors | RSS | df | R² | adj R² |")
        lines.append("|---|---|---|---|---|---|")
        for k in ("M0", "M1", "M2", "M3"):
            m = models.get(k)
            if not m:
                continue
            preds = ", ".join(m.get("predictors", [])) or "(intercept only)"
            lines.append(f"| {k} | {preds} | {_fmt(m.get('rss'))} | "
                         f"{m.get('df')} | {_fmt(m.get('r2'), '.3f')} | "
                         f"{_fmt(m.get('r2_adj'), '.3f')} |")
        lines.append("")
    if ftests:
        m1_vs_m0 = ftests.get("M1_vs_M0", {})
        m3_vs_m2 = ftests.get("M3_vs_M2", {})
        m3_vs_m1 = ftests.get("M3_vs_M1", {})
        lines.append(f"- **M1 vs M0** (intersection has signal): "
                     f"F = {_fmt(m1_vs_m0.get('f'))}, p = {_fmt(m1_vs_m0.get('p'))}  "
                     f"{_verdict(m1_vs_m0.get('p'), alpha, want_reject=True)}")
        lines.append(f"- **M3 vs M2** (intersection adds over hyperparams): "
                     f"F = {_fmt(m3_vs_m2.get('f'))}, p = {_fmt(m3_vs_m2.get('p'))}  "
                     f"{_verdict(m3_vs_m2.get('p'), alpha, want_reject=True)}")
        lines.append(f"- **M3 vs M1** (hyperparams add over intersection — sceptic-friendly): "
                     f"F = {_fmt(m3_vs_m1.get('f'))}, p = {_fmt(m3_vs_m1.get('p'))}  "
                     f"{_verdict(m3_vs_m1.get('p'), alpha, want_reject=False)}")
        lines.append("")
    if "skipped_models" in suf:
        lines.append(f"_Skipped: {', '.join(suf['skipped_models'])} — "
                     f"{suf.get('skipped_reason', '')}._")
        lines.append("")

    rob = results.get("robustness", {})
    axes = rob.get("axes", {})
    if axes:
        lines.append(f"## (4) Robustness across axes "
                     f"(correction: {rob.get('correction')})")
        lines.append("")
        lines.append("| Axis | Kind | Slope/F | p_raw | p_holm | Verdict |")
        lines.append("|---|---|---|---|---|---|")
        for ax, r in axes.items():
            stat = (f"slope={_fmt(r.get('slope'))}" if r["kind"] == "numeric"
                    else f"F={_fmt(r.get('f'))}")
            v = _verdict(r.get("p_holm"), alpha, want_reject=False)
            lines.append(f"| {ax} | {r['kind']} | {stat} | "
                         f"{_fmt(r.get('p_raw'))} | "
                         f"{_fmt(r.get('p_holm'))} | {v} |")
        lines.append("")

    return "\n".join(lines) + "\n"


def render_hypothesis_tests(
    view: ConfigView, figure: IntersectionFigure, df: pd.DataFrame, out_dir: Path,
) -> dict[str, Path]:
    """Run + serialise the test suite for one figure. Returns a dict of
    saved paths (relative labels: `tests/json`, `tests/md`)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = compute_hypothesis_tests(view, figure, df)
    json_path = out_dir / "hypothesis_tests.json"
    md_path = out_dir / "hypothesis_tests.md"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    md_path.write_text(format_hypothesis_tests_markdown(results))
    return {"tests/json": json_path, "tests/md": md_path}
