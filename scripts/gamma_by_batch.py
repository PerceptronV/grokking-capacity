#!/usr/bin/env python3
"""Batch-sweep adjudication: gamma per batch size (theory-trunk §3, C3).

Fits ln T_mem = ln b + gamma_B * f separately per batch size B on the
pre-registered window f in [0.08, 1.0], in the pre-specified adjudication
unit (optimiser STEPS: T_steps = T_epochs * ceil(n/B)) and in epochs
alongside. Censored cells (saturated = False) are excluded from the OLS
fit; a censoring-aware Tobit MLE (right-censored at the epoch cap) is
reported alongside, which lets the heavily-censored B=128 arm yield an
estimate instead of "insufficient" (see the censoring lemma in
docs/papers/2026 Grokking Capacity/theory/derivations.tex: OLS-on-uncensored is biased low at high f,
so gamma_tobit >= gamma_ols is expected). Run bootstrap CIs. Thermal
barrier: gamma_B falls with B (temperature ~ 1/B); geometric: gamma_B
flat.

Usage:  python scripts/gamma_by_batch.py [--db ~/batch_sweep_runs.db]
"""
import argparse
import json
import math
import os
import sqlite3

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

F_LO, F_HI = 0.08, 1.0
C = 2.16
N_BOOT = 4000


def fit(f, y):
    X = np.column_stack([np.ones_like(f), f])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    r2 = 1 - (r @ r) / np.sum((y - y.mean()) ** 2)
    rng = np.random.default_rng(0)
    n = len(y)
    boots = []
    for _ in range(N_BOOT):
        s = rng.integers(0, n, n)
        c, *_ = np.linalg.lstsq(X[s], y[s], rcond=None)
        boots.append(c[1])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(b[1]), float(lo), float(hi), float(b[0]), r2, n


def tobit_fit(f, y, censored, n_boot=800):
    """Right-censored Tobit MLE of y = ln_b + gamma*f + N(0, sigma^2).

    For censored rows, y is the censoring level (ln of the epoch-cap
    time) and the likelihood term is P(Y > y). Returns gamma, its
    bootstrap 95% CI, ln_b, sigma, n_obs, n_cens.
    """
    f, y, censored = map(np.asarray, (f, y, censored))

    def nll(theta, f, y, cens):
        lnb, g, lnsig = theta
        sig = math.exp(lnsig)
        mu = lnb + g * f
        z = (y - mu) / sig
        ll = np.where(cens, norm.logsf(z), norm.logpdf(z) - lnsig)
        return -ll.sum()

    def solve(f, y, cens):
        obs = ~cens
        if obs.sum() >= 2 and np.ptp(f[obs]) > 0:
            g0, b0 = np.polyfit(f[obs], y[obs], 1)
        else:
            g0, b0 = 30.0, float(np.median(y) - 30.0 * np.median(f))
        best = None
        for s0 in (0.3, 1.0):
            r = minimize(nll, [b0, g0, math.log(s0)], args=(f, y, cens),
                         method="Nelder-Mead",
                         options={"maxiter": 4000, "xatol": 1e-6,
                                  "fatol": 1e-8})
            if best is None or r.fun < best.fun:
                best = r
        return best.x

    lnb, g, lnsig = solve(f, y, censored)
    rng = np.random.default_rng(0)
    n = len(y)
    boots = []
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        if (~censored[s]).sum() < 2:
            continue
        boots.append(solve(f[s], y[s], censored[s])[1])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"gamma": float(g), "ci95": [float(lo), float(hi)],
            "ln_b": float(lnb), "sigma": float(math.exp(lnsig)),
            "n_obs": int((~censored).sum()), "n_cens": int(censored.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="~/batch_sweep_runs.db")
    ap.add_argument("--out", default="results/theory/gamma_by_batch.json")
    args = ap.parse_args()

    con = sqlite3.connect(os.path.expanduser(args.db))
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "select p, dim, seed, batch_size, n_samples, param_count, "
        "saturation_epoch, saturated, epochs_trained, max_epochs "
        "from runs where status='completed'")]

    out = {"window_f": [F_LO, F_HI], "per_batch": {}}
    for B in sorted({r["batch_size"] for r in rows}):
        sel, cens = [], []
        for r in rows:
            if r["batch_size"] != B or not r["param_count"]:
                continue
            f = r["n_samples"] * math.log2(r["p"] + 2) / (C * r["param_count"])
            if not (F_LO <= f <= F_HI):
                continue
            spe = math.ceil(r["n_samples"] / B)
            # sqlite returns 0/1 ints for the saturated flag, not booleans
            if r["saturated"] and r["saturation_epoch"] is not None:
                sel.append((f, math.log(r["saturation_epoch"]),
                            math.log(r["saturation_epoch"] * spe)))
            else:
                cap = r["epochs_trained"] or r["max_epochs"]
                cens.append((f, math.log(cap), math.log(cap * spe)))
        res = {}
        if len(sel) < 8:
            res["note"] = f"insufficient uncensored (n={len(sel)}) for OLS"
        else:
            f_u = np.array([s[0] for s in sel])
            for unit, idx in (("epochs", 1), ("steps", 2)):
                g, lo, hi, lnb, r2, n = fit(
                    f_u, np.array([s[idx] for s in sel]))
                res[unit] = {"gamma": g, "ci95": [lo, hi], "ln_b": lnb,
                             "r2": r2, "n": n}
            print(f"B={B}: n={res['steps']['n']}  "
                  f"gamma_steps={res['steps']['gamma']:.1f} "
                  f"[{res['steps']['ci95'][0]:.1f}, "
                  f"{res['steps']['ci95'][1]:.1f}]  "
                  f"gamma_epochs={res['epochs']['gamma']:.1f} "
                  f"[{res['epochs']['ci95'][0]:.1f}, "
                  f"{res['epochs']['ci95'][1]:.1f}]")
        # Tobit refit on all in-window rows, censored included
        allrows = sel + cens
        if len(allrows) >= 8 and len(sel) >= 2:
            f_a = np.array([s[0] for s in allrows])
            flag = np.array([False] * len(sel) + [True] * len(cens))
            for unit, idx in (("epochs", 1), ("steps", 2)):
                y = np.array([s[idx] for s in allrows])
                res[f"tobit_{unit}"] = tobit_fit(f_a, y, flag)
            t = res["tobit_steps"]
            print(f"B={B} tobit: n_obs={t['n_obs']} n_cens={t['n_cens']}  "
                  f"gamma_steps={t['gamma']:.1f} "
                  f"[{t['ci95'][0]:.1f}, {t['ci95'][1]:.1f}]  "
                  f"gamma_epochs={res['tobit_epochs']['gamma']:.1f}")
        out["per_batch"][B] = res

    # thermal-vs-geometric verdict on the steps-unit gammas
    for key, label in (("steps", "gamma_vs_B_loglog_slope"),
                       ("tobit_steps", "gamma_vs_B_loglog_slope_tobit")):
        gs = {B: v[key]["gamma"] for B, v in out["per_batch"].items()
              if key in v}
        if len(gs) >= 3:
            Bs = sorted(gs)
            lgB = np.log([float(b) for b in Bs])
            gam = np.array([gs[b] for b in Bs])
            slope = np.polyfit(lgB, np.log(gam), 1)[0]
            out[label] = float(slope)
            print(f"d ln gamma_{key} / d ln B = {slope:+.3f}  "
                  "(thermal ~ -1; geometric ~ 0)")

    # interference-model form gamma(B) = gamma_inf + c'/sqrt(B), on the
    # tobit gammas (needs all three arms)
    gs = {B: v["tobit_steps"]["gamma"] for B, v in out["per_batch"].items()
          if "tobit_steps" in v}
    if len(gs) >= 3:
        Bs = sorted(gs)
        X = np.column_stack([np.ones(len(Bs)),
                             [1 / math.sqrt(float(b)) for b in Bs]])
        gam = np.array([gs[b] for b in Bs])
        (g_inf, cprime), *_ = np.linalg.lstsq(X, gam, rcond=None)
        resid = gam - X @ np.array([g_inf, cprime])
        out["sqrtB_form"] = {"gamma_inf": float(g_inf),
                             "c_prime": float(cprime),
                             "resid": resid.tolist()}
        print(f"gamma(B) = gamma_inf + c'/sqrt(B): gamma_inf={g_inf:.1f}, "
              f"c'={cprime:.0f}, resid={np.round(resid, 2).tolist()}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
