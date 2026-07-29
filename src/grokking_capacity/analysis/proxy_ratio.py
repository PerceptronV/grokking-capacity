"""Proxy-ratio analysis: how far apart are the two "time to fit the training set" clocks?

Two quantities both measure how long the model takes to fit its training data:

  * ``T_mem``  — epochs to saturate a *random-label* dataset of matched size
    (speed runs; the ``saturation_epoch`` annotation). Runs with ``saturated``
    false never reached saturation within ``max_epochs`` — for those the true
    ``T_mem`` exceeds ``max_epochs``, so they are **censored** (a lower bound),
    counted but excluded from means.
  * ``E_train`` — epochs for the *modular task's* training accuracy to first
    reach >= 99%, recomputed from each groks run's stored per-epoch
    ``train_acc`` trace (1-indexed, matching the ``grokking_epoch`` convention).
    Runs that never reach 99% are likewise censored.

The ratio ``T_mem / E_train`` per matched (depth, prime, dim) cell quantifies
how much slower random-label memorisation is than fitting the structured task,
and how that gap moves with depth. Matching is exact: a cell exists only where
a speed row and a groks row share the same (depth, p, dim) within one
architecture family, and their ``n_samples`` agree.

Aggregation is a seed-mean on both sides over the *same* seed set (the
intersection of seeds present on both sides), with censored seeds excluded
from the mean but counted.

Also reported: the pooled median ``log10(T_mem / E_train)`` at depth 2 (the
proxy gap in dex) and the crossing displacement it implies given the local
slope gap between the two curves vs ``log10 P`` at the crossing
(``T_mem`` slope ~ -2.3, ``T_gen`` slope ~ -1.15, i.e. a gap of ~1.15 dex/dex).
This sits next to the published crossing offset of -0.16 dex, with the caveat
that the published offset uses seed-min aggregation and a 98% *validation*
threshold, while this ratio uses seed-mean and a 99% *training* threshold.

Scenario verdict (decided mechanically when more than one depth is analysed):

  Scenario A  iff  the pooled depth-2 median ratio is in [1, 1.5]
              AND  the per-depth median ratio increases with depth
                   (Spearman correlation of (depth, median ratio) > 0,
                   medians over uncensored cells)
              AND  at least half of the depth >= 6 cells are uncensored on
                   the T_mem side.
  Scenario B  otherwise. If most deep cells are T_mem-censored the trend is
  censoring-dominated and the verdict is B by default.

Usage::

    python -m grokking_capacity.analysis.proxy_ratio \
        --db /path/to/runs.db --data-root /path/to/data \
        --out results/depth_proxy/depth_trend --depths 2,4,6,8,10
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np


# Architecture family under study (the '/' modular-division pipeline defaults).
ARCH_FILTER: dict[str, Any] = {
    "operation": "/",
    "train_fraction": 0.5,
    "lr": 0.001,
    "init_scale": 1.0,
    "weight_decay": 1.0,
    "dropout": 0.2,
    "heads": 1,
}

TRAIN_ACC_PCT = 99.0          # E_train := first epoch with train_acc >= this (percent)
EXPECTED_SEEDS = {42, 43, 44, 45}

# Local slopes of the two curves vs log10 P at the memorisation/generalisation
# crossing at depth 2: d log10 T_mem / d log10 P ~ -2.3 and
# d log10 T_gen / d log10 P ~ -1.15, so a vertical offset of g dex between the
# proxy and the true curve displaces the crossing by -g / 1.15 dex in log10 P.
DEPTH2_SLOPE_GAP_DEX_PER_DEX = 1.15
PUBLISHED_CROSSING_OFFSET_DEX = -0.16   # seed-min aggregation, 98% validation threshold

CONVENTIONS_NOTE = (
    "T_mem/E_train here uses seed-MEAN aggregation and a 99% TRAINING-accuracy "
    "threshold; the published crossing offset of -0.16 dex uses seed-MIN "
    "aggregation and a 98% VALIDATION-accuracy threshold. The two are related "
    "but not identical conventions."
)

# Fixed categorical hue order (assigned to sorted primes in slot order, never cycled).
PRIME_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                "#008300", "#4a3aa7", "#e34948"]
INK_MUTED = "#898781"
GRID = "#e1e0d9"


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
def _fetch_rows(db: Path, experiment_type: str, depths: list[int],
                max_dim: Optional[int]) -> list[dict[str, Any]]:
    where = ["experiment_type = ?", "status = 'completed'"]
    params: list[Any] = [experiment_type]
    for k, v in ARCH_FILTER.items():
        where.append(f"{k} = ?")
        params.append(v)
    where.append(f"depth IN ({','.join('?' * len(depths))})")
    params.extend(depths)
    if max_dim is not None:
        where.append("dim <= ?")
        params.append(max_dim)
    cols = ("uuid, depth, p, dim, seed, n_samples, param_count, max_epochs, "
            "saturation_epoch, saturated, grokking_epoch")
    con = sqlite3.connect(str(db))
    con.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in con.execute(
            f"SELECT {cols} FROM runs WHERE {' AND '.join(where)}", params)]
    finally:
        con.close()
    return rows


def _e_train_from_npz(npz_path: Path) -> tuple[Optional[int], Optional[str]]:
    """First 1-indexed epoch with train_acc >= 99% (None if never reached)."""
    if not npz_path.exists():
        return None, "missing_npz"
    with np.load(npz_path) as z:
        if "train_acc" not in z.files:
            return None, "no_train_acc"
        acc = np.asarray(z["train_acc"], dtype=float)
    if acc.size and acc.max() <= 1.5:            # stored as fraction, not percent
        acc = acc * 100.0
    above = np.where(acc >= TRAIN_ACC_PCT)[0]
    if above.size == 0:
        return None, "censored"                  # never reached 99% within the trace
    return int(above[0]) + 1, None


# --------------------------------------------------------------------------- #
# Cell construction
# --------------------------------------------------------------------------- #
def build_cells(db: Path, data_root: Path, depths: list[int],
                max_dim: Optional[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    speed = _fetch_rows(db, "speed", depths, max_dim)
    groks = _fetch_rows(db, "groks", depths, max_dim)

    by_cell_speed: dict[tuple, list[dict]] = defaultdict(list)
    by_cell_groks: dict[tuple, list[dict]] = defaultdict(list)
    for r in speed:
        by_cell_speed[(r["depth"], r["p"], r["dim"])].append(r)
    for r in groks:
        by_cell_groks[(r["depth"], r["p"], r["dim"])].append(r)

    matched_keys = sorted(set(by_cell_speed) & set(by_cell_groks))
    diag: dict[str, Any] = {
        "n_speed_rows": len(speed), "n_groks_rows": len(groks),
        "n_speed_only_cells": len(set(by_cell_speed) - set(by_cell_groks)),
        "n_groks_only_cells": len(set(by_cell_groks) - set(by_cell_speed)),
        "n_samples_mismatches": [], "missing_npz": [], "no_train_acc": [],
        "seed_sets_observed": {},
    }

    cells: list[dict[str, Any]] = []
    for key in matched_keys:
        depth, p, dim = key
        srows = {r["seed"]: r for r in by_cell_speed[key]}
        grows = {r["seed"]: r for r in by_cell_groks[key]}

        # n_samples must agree exactly between the two sides of the cell.
        s_ns = {r["n_samples"] for r in srows.values()}
        g_ns = {r["n_samples"] for r in grows.values()}
        if len(s_ns) != 1 or len(g_ns) != 1 or s_ns != g_ns:
            diag["n_samples_mismatches"].append(
                {"cell": key, "speed_n_samples": sorted(s_ns),
                 "groks_n_samples": sorted(g_ns)})
            continue
        n_samples = next(iter(s_ns))

        seeds_common = sorted(set(srows) & set(grows))
        if not seeds_common:
            continue

        t_mem_by_seed: dict[int, Optional[float]] = {}
        max_epochs_speed = None
        for s in seeds_common:
            r = srows[s]
            max_epochs_speed = r["max_epochs"]
            if r["saturated"] and r["saturation_epoch"] is not None:
                t_mem_by_seed[s] = float(r["saturation_epoch"])
            else:
                t_mem_by_seed[s] = None          # censored: T_mem > max_epochs

        e_train_by_seed: dict[int, Optional[float]] = {}
        for s in seeds_common:
            r = grows[s]
            npz = data_root / "groks" / r["uuid"] / "trace.npz"
            e, err = _e_train_from_npz(npz)
            if err == "missing_npz":
                diag["missing_npz"].append(str(npz))
            elif err == "no_train_acc":
                diag["no_train_acc"].append(str(npz))
            e_train_by_seed[s] = float(e) if e is not None else None

        t_vals = [v for v in t_mem_by_seed.values() if v is not None]
        e_vals = [v for v in e_train_by_seed.values() if v is not None]
        n_t_cens = sum(v is None for v in t_mem_by_seed.values())
        n_e_cens = sum(v is None for v in e_train_by_seed.values())

        t_mean = float(np.mean(t_vals)) if t_vals else None
        e_mean = float(np.mean(e_vals)) if e_vals else None
        ratio = (t_mean / e_mean) if (t_mean is not None and e_mean is not None) else None
        # When every T_mem seed is censored, max_epochs / E_train_mean is a
        # lower bound on the true ratio.
        ratio_lb = (float(max_epochs_speed) / e_mean
                    if (t_mean is None and e_mean is not None and max_epochs_speed)
                    else None)

        seed_key = ",".join(map(str, seeds_common))
        diag["seed_sets_observed"][seed_key] = diag["seed_sets_observed"].get(seed_key, 0) + 1

        cells.append({
            "depth": depth, "p": p, "dim": dim, "n_samples": n_samples,
            "param_count": grows[seeds_common[0]]["param_count"],
            "seeds": seeds_common, "n_seeds": len(seeds_common),
            "n_t_mem_censored": n_t_cens, "n_e_train_censored": n_e_cens,
            "t_mem_censored": n_t_cens > 0, "e_train_censored": n_e_cens > 0,
            "t_mem_mean": t_mean, "e_train_mean": e_mean,
            "ratio": ratio,
            "log10_ratio": float(np.log10(ratio)) if ratio else None,
            "ratio_lower_bound": ratio_lb,
            "t_mem_by_seed": t_mem_by_seed, "e_train_by_seed": e_train_by_seed,
        })
    return cells, diag


# --------------------------------------------------------------------------- #
# Aggregation, gap, verdict
# --------------------------------------------------------------------------- #
def _median(vals: list[float]) -> Optional[float]:
    return float(np.median(vals)) if vals else None


def _uncensored(cells: list[dict]) -> list[dict]:
    return [c for c in cells if not c["t_mem_censored"] and not c["e_train_censored"]
            and c["ratio"] is not None]


def summarise(cells: list[dict]) -> dict[str, Any]:
    per_depth: dict[int, dict[str, Any]] = {}
    for depth in sorted({c["depth"] for c in cells}):
        dc = [c for c in cells if c["depth"] == depth]
        unc = _uncensored(dc)
        avail = [c for c in dc if c["ratio"] is not None]
        per_prime = {}
        for p in sorted({c["p"] for c in dc}):
            pc = [c for c in dc if c["p"] == p]
            punc = _uncensored(pc)
            per_prime[p] = {
                "n_cells": len(pc), "n_uncensored": len(punc),
                "median_ratio_uncensored": _median([c["ratio"] for c in punc]),
                "median_ratio_all_available": _median(
                    [c["ratio"] for c in pc if c["ratio"] is not None]),
            }
        per_depth[depth] = {
            "n_cells": len(dc),
            "n_uncensored": len(unc),
            "n_t_mem_censored_cells": sum(c["t_mem_censored"] for c in dc),
            "n_t_mem_fully_censored_cells": sum(c["t_mem_mean"] is None for c in dc),
            "n_e_train_censored_cells": sum(c["e_train_censored"] for c in dc),
            "n_seed_level_t_mem_censored": sum(c["n_t_mem_censored"] for c in dc),
            "n_seed_level_e_train_censored": sum(c["n_e_train_censored"] for c in dc),
            "median_ratio_uncensored": _median([c["ratio"] for c in unc]),
            "median_ratio_all_available": _median([c["ratio"] for c in avail]),
            "per_prime": per_prime,
        }
    return per_depth


def depth2_gap(cells: list[dict]) -> dict[str, Any]:
    d2 = [c for c in cells if c["depth"] == 2]
    unc = _uncensored(d2)
    gap = _median([c["log10_ratio"] for c in unc])
    gap_all = _median([c["log10_ratio"] for c in d2 if c["log10_ratio"] is not None])
    out = {
        "n_cells_pooled": len(unc),
        "median_log10_ratio_uncensored": gap,
        "median_log10_ratio_all_available": gap_all,
        "median_ratio_uncensored": 10 ** gap if gap is not None else None,
        "slope_gap_dex_per_dex": DEPTH2_SLOPE_GAP_DEX_PER_DEX,
        "implied_crossing_displacement_dex":
            (-gap / DEPTH2_SLOPE_GAP_DEX_PER_DEX) if gap is not None else None,
        "published_crossing_offset_dex": PUBLISHED_CROSSING_OFFSET_DEX,
        "conventions_note": CONVENTIONS_NOTE,
    }
    return out


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _spearman(x: list[float], y: list[float]) -> Optional[float]:
    if len(x) < 2:
        return None
    rx, ry = _rankdata(np.asarray(x, float)), _rankdata(np.asarray(y, float))
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def scenario_verdict(cells: list[dict], per_depth: dict[int, dict]) -> dict[str, Any]:
    depths = sorted(per_depth)
    d2_median = per_depth.get(2, {}).get("median_ratio_uncensored")
    c1 = d2_median is not None and 1.0 <= d2_median <= 1.5

    pairs = [(d, per_depth[d]["median_ratio_uncensored"]) for d in depths
             if per_depth[d]["median_ratio_uncensored"] is not None]
    rho = _spearman([p[0] for p in pairs], [p[1] for p in pairs])
    c2 = rho is not None and rho > 0

    # cell-level Spearman over uncensored cells, for reference
    unc = _uncensored(cells)
    rho_cells = _spearman([c["depth"] for c in unc], [c["ratio"] for c in unc])

    deep = [c for c in cells if c["depth"] >= 6]
    deep_unc_tmem = [c for c in deep if not c["t_mem_censored"]]
    frac_deep_unc = (len(deep_unc_tmem) / len(deep)) if deep else None
    c3 = frac_deep_unc is not None and frac_deep_unc >= 0.5
    censoring_dominated = frac_deep_unc is not None and frac_deep_unc < 0.5

    verdict = "A" if (c1 and c2 and c3) else "B"
    return {
        "scenario": verdict,
        "criterion_1_depth2_median_in_1_to_1p5": c1,
        "depth2_median_ratio": d2_median,
        "criterion_2_ratio_increases_with_depth": c2,
        "spearman_depth_vs_median_ratio": rho,
        "spearman_depth_vs_ratio_cell_level": rho_cells,
        "criterion_3_half_deep_cells_uncensored_t_mem": c3,
        "frac_depth_ge6_cells_uncensored_t_mem": frac_deep_unc,
        "n_depth_ge6_cells": len(deep),
        "censoring_dominated": censoring_dominated,
        "note": ("Depth trend is censoring-dominated (most depth>=6 cells are "
                 "T_mem-censored); verdict is B by default."
                 if censoring_dominated else None),
    }


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def _style_axes(ax) -> None:
    ax.grid(True, which="both", color=GRID, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def plot_ratio(cells: list[dict], per_depth: dict[int, dict], path: Path,
               title: str) -> None:
    depths = sorted(per_depth)
    primes = sorted({c["p"] for c in cells})
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    single_depth = len(depths) == 1
    if single_depth:
        # One depth: spread cells along the prime axis instead.
        color = PRIME_COLORS[0]
        for c in cells:
            y = c["ratio"] if c["ratio"] is not None else c["ratio_lower_bound"]
            if y is None:
                continue
            if c["t_mem_censored"]:
                ax.scatter(c["p"], y, marker="^", facecolors="none",
                           edgecolors=color, s=30, lw=1.1, zorder=3)
            elif c["e_train_censored"]:
                ax.scatter(c["p"], y, marker="v", facecolors="none",
                           edgecolors=color, s=30, lw=1.1, zorder=3)
            else:
                ax.scatter(c["p"], y, marker="o", color=color, s=16,
                           alpha=0.65, lw=0, zorder=3)
        d = depths[0]
        med_x = [p for p in primes
                 if per_depth[d]["per_prime"][p]["median_ratio_uncensored"] is not None]
        med_y = [per_depth[d]["per_prime"][p]["median_ratio_uncensored"] for p in med_x]
        ax.plot(med_x, med_y, "-", color="#0b0b0b", lw=1.6, zorder=4,
                label="per-prime median (uncensored)")
        ax.set_xlabel("prime $p$", fontsize=9)
    else:
        n = len(primes)
        for i, p in enumerate(primes):
            color = PRIME_COLORS[i % len(PRIME_COLORS)]
            dx = (i - (n - 1) / 2) * 0.14
            pc = [c for c in cells if c["p"] == p]
            for c in pc:
                y = c["ratio"] if c["ratio"] is not None else c["ratio_lower_bound"]
                if y is None:
                    continue
                x = c["depth"] + dx
                if c["t_mem_censored"]:
                    ax.scatter(x, y, marker="^", facecolors="none",
                               edgecolors=color, s=26, lw=1.0, zorder=3)
                elif c["e_train_censored"]:
                    ax.scatter(x, y, marker="v", facecolors="none",
                               edgecolors=color, s=26, lw=1.0, zorder=3)
                else:
                    ax.scatter(x, y, marker="o", color=color, s=14,
                               alpha=0.6, lw=0, zorder=3)
            med_x, med_y = [], []
            for d in depths:
                m = per_depth[d]["per_prime"].get(p, {}).get("median_ratio_uncensored")
                if m is not None:
                    med_x.append(d + dx)
                    med_y.append(m)
            ax.plot(med_x, med_y, "-", color=color, lw=1.8, zorder=4,
                    label=f"$p={p}$")
        ax.set_xticks(depths)
        ax.set_xlabel("depth", fontsize=9)

    ax.axhline(1.0, color=INK_MUTED, lw=1.0, ls="--", zorder=2)
    ax.set_yscale("log")
    ax.set_ylabel(r"$T_{\mathrm{mem}}\,/\,E_{\mathrm{train}}$", fontsize=9)
    ax.set_title(title, fontsize=9, color="#0b0b0b")
    _style_axes(ax)

    handles, labels = ax.get_legend_handles_labels()
    handles += [
        plt.Line2D([], [], marker="^", ls="", markerfacecolor="none",
                   markeredgecolor="#0b0b0b", markersize=6,
                   label=r"$T_{\mathrm{mem}}$-censored (lower bound)"),
        plt.Line2D([], [], marker="v", ls="", markerfacecolor="none",
                   markeredgecolor="#0b0b0b", markersize=6,
                   label=r"$E_{\mathrm{train}}$-censored"),
    ]
    ax.legend(handles=handles, fontsize=7, frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def run(db: Path, data_root: Path, out_dir: Path, depths: list[int],
        max_dim: Optional[int], label: str) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cells, diag = build_cells(db, data_root, depths, max_dim)
    per_depth = summarise(cells)
    gap = depth2_gap(cells)
    verdict = scenario_verdict(cells, per_depth) if len(per_depth) > 1 else None

    non_expected = [c for c in cells if set(c["seeds"]) != EXPECTED_SEEDS]
    payload = {
        "label": label,
        "db": str(db), "data_root": str(data_root),
        "arch_filter": ARCH_FILTER, "depths": depths, "max_dim": max_dim,
        "train_acc_threshold_pct": TRAIN_ACC_PCT,
        "n_cells": len(cells),
        "n_cells_seed_set_not_42_45": len(non_expected),
        "diagnostics": diag,
        "per_depth_summary": per_depth,
        "depth2_proxy_gap": gap,
        "scenario_verdict": verdict,
        "cells": cells,
    }
    with open(out_dir / "proxy_ratio.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    plot_ratio(cells, per_depth, out_dir / "proxy_ratio.pdf",
               title=f"{label}: random-label memorisation time vs task train-acc "
                     f"convergence")

    print(f"[proxy-ratio] {label}: {len(cells)} matched cells "
          f"({diag['n_speed_rows']} speed rows, {diag['n_groks_rows']} groks rows)")
    if diag["n_samples_mismatches"]:
        print(f"  n_samples mismatches: {diag['n_samples_mismatches']}")
    for d in sorted(per_depth):
        s = per_depth[d]
        m = s["median_ratio_uncensored"]
        print(f"  depth={d:<3d} cells={s['n_cells']:<3d} uncensored={s['n_uncensored']:<3d} "
              f"T_mem-cens={s['n_t_mem_censored_cells']:<3d} "
              f"E_train-cens={s['n_e_train_censored_cells']:<3d} "
              f"median ratio={m:.3f}" if m is not None else
              f"  depth={d:<3d} cells={s['n_cells']:<3d} median ratio=n/a (all censored)")
    g = gap["median_log10_ratio_uncensored"]
    if g is not None:
        print(f"  depth-2 proxy gap: {g:+.3f} dex "
              f"(implied crossing displacement {gap['implied_crossing_displacement_dex']:+.3f} dex; "
              f"published offset {PUBLISHED_CROSSING_OFFSET_DEX:+.2f} dex)")
    if verdict:
        print(f"  SCENARIO: {verdict['scenario']}  "
              f"(c1 depth2-median-in-[1,1.5]={verdict['criterion_1_depth2_median_in_1_to_1p5']}, "
              f"c2 spearman>0={verdict['criterion_2_ratio_increases_with_depth']} "
              f"(rho={verdict['spearman_depth_vs_median_ratio']}), "
              f"c3 deep-uncensored>=0.5={verdict['criterion_3_half_deep_cells_uncensored_t_mem']} "
              f"(frac={verdict['frac_depth_ge6_cells_uncensored_t_mem']}))")
        if verdict["censoring_dominated"]:
            print("  note: depth trend is censoring-dominated; verdict B by default.")
    return payload


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="gc-proxy-ratio",
        description="Per-(depth, prime, dim) ratio of random-label memorisation "
                    "time (T_mem) to the modular task's 99% train-accuracy epoch "
                    "(E_train), with censoring accounting and a depth trend.")
    ap.add_argument("--db", type=Path, required=True, help="Path to the runs SQLite DB")
    ap.add_argument("--data-root", type=Path, required=True,
                    help="Artefacts root containing groks/<uuid>/trace.npz")
    ap.add_argument("--out", type=Path, required=True, help="Output directory")
    ap.add_argument("--depths", default="2,4,6,8,10",
                    help="Comma-separated depths to analyse (default 2,4,6,8,10)")
    ap.add_argument("--max-dim", type=int, default=None,
                    help="Optional upper bound on dim (inclusive)")
    ap.add_argument("--label", default="proxy-ratio",
                    help="Dataset label for printouts/plot title")
    args = ap.parse_args(argv)

    depths = sorted({int(d) for d in str(args.depths).split(",") if d.strip()})
    run(args.db, args.data_root, args.out, depths, args.max_dim, args.label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
