"""Rendering primitives consumed by `grokking_capacity.figures`.

Trimmed from a much larger legacy file. All call sites now live inside the
`figures` package; nothing else in the repo imports from here.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .aggregate import find_intersection


def calculate_grokking_delay(
    train_acc, val_acc,
    threshold_train: float = 99.0, threshold_val: float = 99.0,
):
    """(train_epoch, val_epoch, delay) — delay clamped to >= 0.

    Returns (None, None, None) when train never reaches threshold;
    (None, val_epoch, 0) when val reaches before train. Mirrors the original
    legacy behaviour because the rest of the plotting code depends on it.
    """
    train_epoch = next(
        (i for i, a in enumerate(train_acc) if a >= threshold_train), None
    )
    val_epoch = next(
        (i for i, a in enumerate(val_acc) if a >= threshold_val), None
    )
    if train_epoch is not None and val_epoch is not None:
        return train_epoch, val_epoch, max(0, val_epoch - train_epoch)
    if train_epoch is None and val_epoch is not None:
        return None, val_epoch, 0
    return None, None, None


def plot_grokking_delay_with_speed(
    delay_records: List[Dict],
    speed_curve: Dict[float, float],
    groks_curve: Dict[float, float],
    *,
    mem_curve_threshold: float = 99.0,
    gen_curve_threshold: float = 99.0,
    threshold_train: float = 99.0,
    threshold_val: float = 99.0,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    x_label: str = "Parameter count",
    colour_label: str = "Dimension",
) -> Optional[Tuple[float, float]]:
    """The intersection plot (Image #1).

    Args:
        delay_records: per-seed delay scatter. List of dicts with `x`,
            `colour`, `delay` (the caller decides which row fields those
            map to — for the canonical figure, x=param_count and
            colour=dim).
        speed_curve: {x: mean epochs to memorise (train ≥ mem_curve_threshold)}.
        groks_curve: {x: mean epochs to generalise (val ≥ gen_curve_threshold)}.
        mem_curve_threshold: train accuracy threshold (in %) the speed
            experiment used when storing `saturation_epoch`. Drives the
            mem-curve legend label only — does not recompute anything.
        gen_curve_threshold: val accuracy threshold (in %) the groks
            experiment used when storing `grokking_epoch`. Drives the
            gen-curve legend label only.
        threshold_train / threshold_val: thresholds used to *recompute*
            per-seed delay from the npz traces for the scatter. May
            differ from the curve thresholds.

    Returns the intersection (x, epochs) or None.
    """
    if not delay_records or not speed_curve or not groks_curve:
        return None

    delay_records = sorted(delay_records, key=lambda r: r["x"])
    xs = np.array([r["x"] for r in delay_records], dtype=float)
    colours = np.array([r["colour"] for r in delay_records], dtype=float)
    delays = np.array([r["delay"] for r in delay_records], dtype=float)

    fig, ax1 = plt.subplots(figsize=(10, 7))

    crest = sns.color_palette("crest", as_cmap=True)
    scatter = ax1.scatter(
        xs, delays, c=colours, cmap=crest, s=80, alpha=0.7, edgecolors="none",
        label=f"Generalisation delay (val≥{threshold_val:.0f}% − train≥{threshold_train:.0f}%)",
    )
    ax1.set_xlabel(x_label, fontsize=14)
    ax1.set_ylabel("Generalisation delay (epochs)", fontsize=14, color="#1b7a3d")
    ax1.set_xscale("log")
    ax1.tick_params(axis="y", labelcolor="#1b7a3d")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax1.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax1, label=colour_label, pad=0.12)
    cbar.ax.tick_params(labelsize=10)

    ax2 = ax1.twinx()
    speed_x = np.array(sorted(speed_curve.keys()), dtype=float)
    speed_y = np.array([speed_curve[x] for x in speed_x], dtype=float)
    groks_x = np.array(sorted(groks_curve.keys()), dtype=float)
    groks_y = np.array([groks_curve[x] for x in groks_x], dtype=float)
    ax2.plot(
        groks_x, groks_y, "-", color="#d95f02", linewidth=2, alpha=0.85,
        label=f"Epochs to generalise (val≥{gen_curve_threshold:.0f}%)",
    )
    ax2.plot(
        speed_x, speed_y, "-", color="#7570b3", linewidth=2, alpha=0.85,
        label=f"Epochs to memorise (train≥{mem_curve_threshold:.0f}%)",
    )
    ax2.set_ylabel("Epochs", fontsize=14, color="#d95f02")
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor="#d95f02")

    intersection = find_intersection(speed_curve, groks_curve)
    if intersection is not None:
        ix, iy = intersection
        ax2.axvline(x=ix, color="#8B0000", linestyle=":", linewidth=1.5, alpha=0.5)
        ax2.axhline(y=iy, color="#8B0000", linestyle=":", linewidth=1.5, alpha=0.5)
        ax1.annotate(
            f"{ix:,.0f}", xy=(ix, ax1.get_ylim()[0]),
            xytext=(0, -16), textcoords="offset points",
            ha="center", va="bottom", fontsize=11, color="#8B0000",
        )
        ax2.annotate(
            f"{iy:,.0f}", xy=(ax2.get_xlim()[1], iy),
            xytext=(-5, 5), textcoords="offset points",
            ha="right", va="bottom", fontsize=11, color="#8B0000",
        )

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=11)
    if title:
        ax1.set_title(title, fontsize=15)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
    return intersection


def plot_capacity_curves(
    by_dim: Dict[int, List[Dict]],
    p: int,
    *,
    save_path: Optional[str] = None,
    show: bool = False,
) -> List[Tuple[int, float]]:
    """Image #2(a): total memorised bits vs dataset size, per dim.

    Each value in `by_dim` is a list of dicts with `n_samples`,
    `total_bits_memorized`, `param_count`. Returns the saturation
    (param_count, max_bits) points used by `plot_capacity_estimation`.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xscale("log")
    ax.set_yscale("log")

    dims = sorted(by_dim.keys())
    colors = sns.color_palette("crest", n_colors=len(dims))
    saturation_points: List[Tuple[int, float]] = []

    for color, dim in zip(colors, dims):
        rows = sorted(by_dim[dim], key=lambda r: r["n_samples"])
        if not rows:
            continue
        ns = [r["n_samples"] for r in rows]
        bits = [r["total_bits_memorized"] for r in rows]
        pc = rows[0]["param_count"]
        if pc >= 1e6:
            label = f"{pc/1e6:.1f}M"
        elif pc >= 1e3:
            label = f"{pc/1e3:.0f}K"
        else:
            label = str(pc)
        ax.plot(ns, bits, marker="o", markersize=6, linewidth=2, color=color, label=label)
        saturation_points.append((int(pc), float(max(bits))))

    all_sizes = [r["n_samples"] for dim in dims for r in by_dim[dim]]
    if all_sizes:
        x_min, x_max = min(all_sizes) * 0.5, max(all_sizes) * 2
        x_range = np.logspace(np.log10(x_min), np.log10(x_max), 50)
        bits_per_example = np.log2(p + 2)
        ax.plot(x_range, x_range * bits_per_example, "--", color="gray", alpha=0.5,
                label="Dataset complexity")

    ax.set_xlabel("Dataset size (number of datapoints)", fontsize=14)
    ax.set_ylabel("Memorisation (bits)", fontsize=14)
    ax.legend(title="Parameters", loc="upper left", fontsize=11, title_fontsize=12)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
    return saturation_points


def plot_capacity_estimation(
    saturation_points: List[Tuple[int, float]],
    *,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[float, float, float]:
    """Image #2(b): saturation memorisation vs param count, with linear fit.

    Returns (C, intercept, r_squared) where bits ≈ C * params + intercept.
    """
    if len(saturation_points) < 2:
        return 0.0, 0.0, 0.0

    params = np.array([p for p, _ in saturation_points], dtype=float)
    bits = np.array([b for _, b in saturation_points], dtype=float)
    C, intercept = np.polyfit(params, bits, 1)
    y_pred = C * params + intercept
    ss_res = float(np.sum((bits - y_pred) ** 2))
    ss_tot = float(np.sum((bits - bits.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(params, bits, s=100, zorder=5, label="Data")
    x_fit = np.linspace(0, params.max() * 1.1, 100)
    sign, abs_int = ("+", intercept) if intercept >= 0 else ("−", -intercept)
    ax.plot(x_fit, C * x_fit + intercept, "--", linewidth=2, color="C1",
            label=f"Fit: bits = {C:.2f} × params {sign} {abs_int:.0f}")
    ax.set_xlabel("Number of parameters", fontsize=14)
    ax.set_ylabel("Saturation memorisation (bits)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.05, 0.95,
        f"C: {C:.2f} bits/param\nIntercept: {intercept:.0f} bits\nR²: {r_squared:.3f}",
        transform=ax.transAxes, fontsize=11, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
    return float(C), float(intercept), float(r_squared)


def plot_saturation_epochs_vs_inverse_capacity(
    by_dim: Dict[Tuple[int, int] | int, List[Dict]],
    C: float,
    *,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Image #3(a): epochs to saturation vs 1/(C·P), coloured by dataset size.

    Keys may be `dim` (single-prime) or `(p, dim)` (multi-prime). Values are
    dicts with `n_samples`, `param_count`, `saturation_epoch`, optional
    `saturated`, optional `p`.
    """
    if not by_dim:
        return
    is_multi_prime = isinstance(next(iter(by_dim.keys())), tuple)

    by_size: dict[int, list[dict]] = {}
    primes_by_size: dict[int, set[int]] = {}
    for key, rows in by_dim.items():
        p_key = key[0] if is_multi_prime else None
        for r in rows:
            if not r.get("saturated", True):
                continue
            n = r["n_samples"]
            by_size.setdefault(n, []).append(r)
            if p_key is not None:
                primes_by_size.setdefault(n, set()).add(p_key)
            elif "p" in r:
                primes_by_size.setdefault(n, set()).add(r["p"])

    fig, ax = plt.subplots(figsize=(12, 8))
    sizes = sorted(by_size.keys())
    colors = sns.color_palette("crest", n_colors=len(sizes))
    for color, n in zip(colors, sizes):
        rows = sorted(by_size[n], key=lambda r: 1.0 / (C * r["param_count"]))
        xs = [1.0 / (C * r["param_count"]) for r in rows]
        ys = [r["saturation_epoch"] for r in rows]
        if is_multi_prime and n in primes_by_size:
            primes_str = ", ".join(map(str, sorted(primes_by_size[n])))
            label = f"{n} samples (p={primes_str})"
        else:
            label = f"{n} samples"
        ax.plot(xs, ys, marker="o", markersize=8, linewidth=2, color=color, label=label)

    ax.set_xlabel("1 / (Model capacity) [1/bits]", fontsize=14)
    ax.set_ylabel("Epochs to saturation", fontsize=14)
    ax.set_yscale("log")
    ax.legend(title="Dataset size", loc="best", fontsize=10, title_fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def plot_saturation_time_vs_capacity_fraction(
    by_dim: Dict[Tuple[int, int] | int, List[Dict]],
    C: float,
    *,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[float, float, float]:
    """Image #3(b): epochs to saturation vs capacity fraction f = S/(C·P).

    Returns (b, a, r²) for the global fit `epochs = a · exp(b · f)`.
    """
    if not by_dim:
        return 0.0, 0.0, 0.0
    is_multi_prime = isinstance(next(iter(by_dim.keys())), tuple)

    f_vals, epochs, dims, primes = [], [], [], []
    for key, rows in by_dim.items():
        p_key, dim_key = (key if is_multi_prime else (None, key))
        for r in rows:
            if not r.get("saturated", True):
                continue
            P = r["param_count"]
            S = r.get("dataset_bits")
            if P is None or S is None:
                continue
            f_vals.append(S / (C * P))
            epochs.append(r["saturation_epoch"])
            dims.append(dim_key)
            primes.append(p_key if p_key is not None else r.get("p", 0))

    if len(f_vals) < 2:
        return 0.0, 0.0, 0.0

    f_arr = np.array(f_vals)
    epochs_arr = np.array(epochs, dtype=float)
    primes_arr = np.array(primes)
    dims_arr = np.array(dims)

    log_e = np.log(epochs_arr)
    b, log_a = np.polyfit(f_arr, log_e, 1)
    a = float(np.exp(log_a))
    pred = b * f_arr + log_a
    ss_res = float(np.sum((log_e - pred) ** 2))
    ss_tot = float(np.sum((log_e - log_e.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    fig, ax = plt.subplots(figsize=(12, 8))
    if is_multi_prime:
        unique = sorted(set(primes_arr))
        palette = sns.color_palette("tab20" if len(unique) <= 20 else "husl", n_colors=len(unique))
        for color, p in zip(palette, unique):
            mask = primes_arr == p
            label = f"p={p}"
            if mask.sum() >= 2:
                bp, log_ap = np.polyfit(f_arr[mask], np.log(epochs_arr[mask]), 1)
                pred_p = bp * f_arr[mask] + log_ap
                ssr = float(np.sum((np.log(epochs_arr[mask]) - pred_p) ** 2))
                sst = float(np.sum((np.log(epochs_arr[mask]) - np.log(epochs_arr[mask]).mean()) ** 2))
                r2_p = 1 - ssr / sst if sst > 0 else 0.0
                label = f"p={p} (R²={r2_p:.3f})"
                xfit = np.linspace(f_arr[mask].min() * 0.95, f_arr[mask].max() * 1.05, 100)
                ax.plot(xfit, np.exp(log_ap) * np.exp(bp * xfit), "--", color=color,
                        linewidth=2, alpha=0.8)
            ax.scatter(f_arr[mask], epochs_arr[mask], c=[color], s=100, alpha=0.7,
                       edgecolors="black", linewidths=1, label=label)
    else:
        unique = sorted(set(dims_arr))
        palette = sns.color_palette("crest", n_colors=len(unique))
        for color, d in zip(palette, unique):
            mask = dims_arr == d
            ax.scatter(f_arr[mask], epochs_arr[mask], c=[color], s=100, alpha=0.7,
                       edgecolors="black", linewidths=1, label=f"dim={d}")
        xfit = np.linspace(f_arr.min() * 0.95, f_arr.max() * 1.05, 100)
        ax.plot(xfit, a * np.exp(b * xfit), "--", color="red", linewidth=2, alpha=0.7,
                label=f"Fit: epochs = {a:.1f} × exp({b:.2f} × f)")

    ax.set_xlabel("Capacity fraction", fontsize=14)
    ax.set_ylabel("Epochs to saturation", fontsize=14)
    ax.set_yscale("log")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3, which="both")
    ax.text(
        0.05, 0.95,
        f"Overall fit: epochs = {a:.1f} × exp({b:.2f} × f)\n"
        f"Exponent: {b:.2f}\nR²: {r_squared:.3f}\nC = {C:.2f} bits/param",
        transform=ax.transAxes, fontsize=11, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
    return float(b), float(a), float(r_squared)
