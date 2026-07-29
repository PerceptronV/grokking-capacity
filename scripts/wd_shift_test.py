#!/usr/bin/env python3
"""Angle E discriminator on archived data: does the crossing algebra track
the onset shift when T_mem is reshaped by weight decay?

For each weight-decay group in configs/weight_decay_sweep.yaml, fit the two
timescale laws on the pre-registered window (f in [0.08, 1], epochs), solve
the implicit crossing per prime, and compare the per-lambda median onset
residual and the measured-vs-predicted onset shift across lambda.

Usage:  python scripts/wd_shift_test.py
Output: results/theory/wd_shift_test.json + stdout table.
"""
import json
import math
import os

import numpy as np

from grokking_capacity.analysis import aggregate
from grokking_capacity.analysis import forecast_onset as fo
from grokking_capacity.analysis.config_view import ConfigView, load_npz
from grokking_capacity.analysis.matching import compute_n_equiv
from grokking_capacity.analysis.plots import _passes_filters

F_LO, F_HI = fo.MEM_F_LO, fo.MEM_F_HI


def group_data(group, figure):
    C = float(group.capacity_constant)
    op, tf = group.key.operation, float(group.key.train_fraction)
    K = {}

    def K_of(p):
        if p not in K:
            K[p] = float(compute_n_equiv(p, op, tf)[1])
        return K[p]

    speed = []
    for r in group.speed_runs:
        if (not _passes_filters(r, figure) or r.get("saturated") is False
                or r.get("saturation_epoch") is None
                or not r.get("param_count") or r.get("p") is None):
            continue
        f = K_of(int(r["p"])) / (C * float(r["param_count"]))
        if F_LO <= f <= F_HI:
            speed.append((f, math.log(float(r["saturation_epoch"]))))

    gen = []
    primes = sorted({r.get("p") for r in group.groks_runs if r.get("p")})
    onsets = {}
    for p in primes:
        curve = aggregate.mean_over_seeds(
            (r for r in group.groks_runs
             if r.get("p") == p and _passes_filters(r, figure)
             and r.get("grokking_epoch") is not None
             and r.get("saturated") is not False),
            x_field="param_count", y_field="grokking_epoch")
        for pc, tg in curve.items():
            f = K_of(p) / (C * float(pc))
            if F_LO <= f <= F_HI:
                gen.append((p, float(pc), float(tg)))
        # empirical onset from per-seed delays
        pairs = []
        for r in group.groks_runs:
            if r.get("p") != p or not _passes_filters(r, figure):
                continue
            try:
                with load_npz(r) as npz:
                    d = aggregate.compute_delays(
                        [{"param_count": r.get("param_count"),
                          "train_acc": npz["train_acc"],
                          "val_acc": npz["val_acc"]}],
                        threshold_train=figure.delay_train_threshold,
                        threshold_val=figure.delay_val_threshold)
            except (FileNotFoundError, KeyError):
                continue
            if d:
                pairs.append(d[0])
        onsets[p] = aggregate.find_grokking_onset(
            aggregate.min_delay_curve(pairs))
    return C, K_of, speed, gen, onsets


def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    r2 = 1 - (r @ r) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    return b, r2


def main():
    view = ConfigView.from_yaml("configs/weight_decay_sweep.yaml",
                                db_path="runs.db")
    figure = view.intersection_figures[0]
    out = []
    for group in view.iter_groups():
        if not (group.speed_runs and group.groks_runs):
            continue
        lam = group.key.weight_decay
        C, K_of, speed, gen, onsets = group_data(group, figure)
        if len(speed) < 8 or len(gen) < 8:
            out.append({"lambda": lam, "note": "insufficient",
                        "n_speed": len(speed), "n_gen": len(gen)})
            continue
        f = np.array([s[0] for s in speed])
        (lnb, gamma), r2m = ols(np.column_stack([np.ones_like(f), f]),
                                np.array([s[1] for s in speed]))
        lp = np.log([g[0] for g in gen])
        lP = np.log([g[1] for g in gen])
        y = np.log([g[2] for g in gen])
        (lnA, a, nb), r2g = ols(np.column_stack([np.ones_like(lp), lp, lP]), y)
        beta = -nb
        mem = {"ln_b": float(lnb), "gamma": float(gamma)}
        genl = {"lnA": float(lnA), "a": float(a), "beta": float(beta)}
        resid = []
        per_prime = {}
        for p, on in onsets.items():
            L = fo.solve_crossing(int(p), K_of(int(p)), C, mem, genl)
            per_prime[int(p)] = {
                "onset": on, "log10_cross": L,
                "resid_dex": (math.log10(on) - L) if (on and L) else None}
            if on and L:
                resid.append(math.log10(on) - L)
        out.append({
            "lambda": lam, "gamma": float(gamma), "b": float(math.exp(lnb)),
            "r2_mem": float(r2m), "beta": float(beta), "a": float(a),
            "r2_gen": float(r2g), "n_speed": len(speed), "n_gen": len(gen),
            "median_resid_dex": (float(np.median(resid)) if resid else None),
            "n_crossings": len(resid), "per_prime": per_prime})
        print(f"lambda={lam}: gamma={gamma:.1f} b={math.exp(lnb):.1f} "
              f"(R2 {r2m:.2f})  beta={beta:.2f} a={a:.2f} (R2 {r2g:.2f})  "
              f"crossings={len(resid)}  median resid="
              f"{np.median(resid) if resid else float('nan'):.3f} dex")
    os.makedirs("results/theory", exist_ok=True)
    json.dump(out, open("results/theory/wd_shift_test.json", "w"), indent=2)


if __name__ == "__main__":
    main()
