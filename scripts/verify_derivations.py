#!/usr/bin/env python3
"""Numerical verification of every claim in consequences.tex (né derivations.tex).

Run from the repo root:  python scripts/verify_derivations.py
Reads the frozen fits (results/theory/forecast_onset.json) and the registry.
Each block prints the derived quantity next to its independent numerical check.
"""
import json
import math

import numpy as np
from scipy import integrate
from scipy.optimize import brentq
from scipy.stats import norm

RES = json.load(open("results/theory/forecast_onset.json"))
MEM, GEN = RES["mem_law"], RES["gen_law"]
G, LNB = MEM["gamma"], MEM["ln_b"]
LNA, A_EXP, BETA = GEN["lnA"], GEN["a"], GEN["beta"]
C = 2.16


def K(p):
    return 0.5 * p * (p - 1) * math.log2(p + 2)


def R(p):
    return (LNA + A_EXP * math.log(p) - LNB) - BETA * math.log(G * K(p) / C)


def kappa(p, h=1e-4):
    return (math.log(K(p * math.exp(h))) - math.log(K(p * math.exp(-h)))) / (2 * h)


def x_roots(p):
    Rp = R(p)
    Rstar = BETA * (1 - math.log(BETA))
    if Rp < Rstar:
        return None
    F = lambda x: x - BETA * math.log(x) - Rp
    return brentq(F, BETA, 1e3), brentq(F, 1e-9, BETA)


def main():
    Rstar = BETA * (1 - math.log(BETA))

    print("== Prop 2 (fold): closed-form p* vs pipeline ==")
    pstar = brentq(lambda p: R(p) - Rstar, 100, 400)
    print(f"  p* = {pstar:.3f}   (pipeline grid search: "
          f"{RES['divergence']['p_star_point']:.3f})")
    print(f"  f_fold = beta/gamma = {BETA/G:.4f}   "
          f"P_fold = {G*K(pstar)/(C*BETA):.3e}")

    print("== Prop 1 (normal form): analytic roots vs pipeline crossings ==")
    for p in (97, 101, 103, 107, 109, 113, 127, 131, 137, 139):
        xp, xm = x_roots(p)
        lp = math.log10(G * K(p) / (C * xp))
        pipe = RES["per_prime"][str(p)]["log10_P_cross_fit"]
        print(f"  p={p}: analytic {lp:.3f}  pipeline {pipe:.3f}  "
              f"| re-entrant P = {G*K(p)/(C*xm):.2e}")

    print("== Prop 3 (s_eff) vs finite differences; measured onset slope 3.31 ==")
    def lnP_on(p):
        r = x_roots(p)
        return None if r is None else math.log(G * K(p) / (C * r[0]))
    for p in (105, 113, 125, 139, 155, 163):
        xp, _ = x_roots(p)
        s = kappa(p) + (BETA * kappa(p) - A_EXP) / (xp - BETA)
        h = 0.005
        s_num = (lnP_on(p * math.exp(h)) - lnP_on(p * math.exp(-h))) / (2 * h)
        print(f"  p={p}: s_eff formula {s:.3f}  finite-diff {s_num:.3f}")
    ps = np.array([97, 101, 103, 107, 109, 113, 127, 131, 137, 139], float)
    lp = np.array([lnP_on(p) for p in ps]) / math.log(10)
    slope = np.polyfit(np.log10(ps), lp, 1)[0]
    print(f"  OLS slope of model onset curve over calibration primes: {slope:.3f}")

    print("== Eq (sqrt-scaling) near the fold ==")
    for p in (160, 165, 167):
        xp, _ = x_roots(p)
        approx = math.sqrt(2 * BETA * (R(p) - Rstar))
        print(f"  p={p}: x+ - beta = {xp-BETA:.4f}  sqrt approx = {approx:.4f}")

    print("== Prop 4 (Gardner RS entropy = counting bound at small load) ==")
    def s_rs(alpha):
        def s_of(q):
            a = math.sqrt(q / (1 - q))
            I, _ = integrate.quad(
                lambda t: norm.pdf(t) * math.log(max(norm.sf(a * t), 1e-300)),
                -12, 12)
            return 0.5 * math.log(1 - q) + q / (2 * (1 - q)) + alpha * I
        ds = lambda q: (s_of(q + 1e-6) - s_of(q - 1e-6)) / 2e-6
        q = brentq(ds, 1e-6, 0.9)
        return s_of(q), q
    for alpha in (0.01, 0.032, 0.06):
        s, q = s_rs(alpha)
        print(f"  alpha={alpha}: s={s:.5f}  -alpha ln2={-alpha*math.log(2):.5f}  "
              f"ratio {s/(-alpha*math.log(2)):.3f}  q*={q:.4f}")
    print(f"  extensive exponent P*|s| at P=1e5, alpha=0.032: "
          f"{1e5*abs(s_rs(0.032)[0]):.0f} nats  vs measured gamma*f ~ 3 nats")

    print("== Cor (C3): residual P-dependence at fixed f (registry regression) ==")
    from grokking_capacity.analysis import forecast_onset as fo
    from grokking_capacity.analysis.config_view import ConfigView
    view = ConfigView.from_yaml("configs/central.yaml", db_path="runs.db")
    data = fo.collect(view)
    for label, lo in (("window f>=0.08", 0.08), ("full range", 0.0)):
        pts = [q for q in data["speed_pts"]
               if q["p"] in fo.CENTRAL_PRIMES and q["f"] >= lo]
        f = np.array([q["f"] for q in pts])
        lP = np.log([q["param_count"] for q in pts])
        y = np.log([q["t_mem"] for q in pts])
        X = np.column_stack([np.ones_like(f), f, lP])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b
        se = np.sqrt(np.diag((r @ r / (len(y) - 3)) * np.linalg.inv(X.T @ X)))
        print(f"  {label}: bP = {b[2]:+.3f} +- {se[2]:.3f} nats/e-fold "
              f"(extensive requires ~ln T ~ {np.mean(y):.1f})")


if __name__ == "__main__":
    main()
