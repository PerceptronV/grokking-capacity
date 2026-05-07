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
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import aggregate
from .config_view import ArchGroup, ConfigView, IntersectionFigure
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
    if axis not in df.columns:
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
    """End-to-end per intersection figure: CSV, scatter, per-axis plots.

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
    return paths
