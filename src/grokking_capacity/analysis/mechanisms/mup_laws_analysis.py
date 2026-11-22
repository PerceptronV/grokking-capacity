#!/usr/bin/env python3
"""Matched-protocol muP laws (configs/mup_laws.yaml, registry
~/mup_laws_runs.db). Everything on the canonical clocks:
  T_mem: gc-speed saturation epoch (train-mode acc >= 99 %), censored at
         max_epochs when unsaturated.
  T_gen: gc-groks grokking epoch (val >= 99 %), censored at cap.
  delay: canonical (val-98 minus train-mode train-99, clipped), from traces.
Fits (same windows as the fixed-rate pipeline, f = K/(C P) >= 0.08, P the
registry param_count, K = n_equiv * log2(p+2), C = 2.16):
  ln T_mem = ln B + gamma f          (saturated rows; censored count reported;
                                      a Tobit fit including censored rows too)
  ln T_gen = ln A + a ln p - b ln P
Fold: x - b ln x = R(p), R(p) = lnA - ln B + a ln p - b ln(gamma K/C);
crossing exists iff R >= b (1 - ln b); p* solves equality. Seed
bootstrap CI on p*. Also: muP race map per cell, and the p = 197 onset
(seed-min canonical delay vs P) for the scorecard row.
Writes results/theory/mup_laws_matched.json and ..._racemap.csv."""
import os, sqlite3, math, json, csv
from collections import defaultdict
import numpy as np
from scipy.optimize import least_squares, brentq, minimize
from scipy.stats import norm

C = 2.16
K = lambda p: 0.5 * p * (p - 1) * math.log2(p + 2)


def fit_mem(rows, window=0.08):
    f = np.array([K(p) / (C * P) for p, d, P, T, cz in rows]); T = np.array([T for *_, T, cz in rows], float); cz = np.array([cz for *_, cz in rows])
    sel = f >= window
    ok = sel & ~cz
    X = np.column_stack([np.ones(ok.sum()), f[ok]]); cf = np.linalg.lstsq(X, np.log(T[ok]), rcond=None)[0]
    # Tobit (right-censored at log cap) on all window rows
    y = np.log(T[sel]); fs = f[sel]; cen = cz[sel]
    def nll(th):
        mu = th[0] + th[1] * fs; s = math.exp(th[2])
        ll = np.where(cen, norm.logsf((y - mu) / s), norm.logpdf((y - mu) / s) - math.log(s))
        return -ll.sum()
    tb = minimize(nll, [cf[0], cf[1], 0.0], method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-8})
    return dict(ln_B=float(cf[0]), gamma=float(cf[1]), n_ok=int(ok.sum()), n_window=int(sel.sum()), n_censored_window=int(cen.sum()),
                tobit_ln_B=float(tb.x[0]), tobit_gamma=float(tb.x[1]))


def fit_gen(rows, window=0.08):
    f = np.array([K(p) / (C * P) for p, d, P, T, cz, _ in rows]); sel = (f >= window) & ~np.array([cz for *_, cz, _ in rows])
    lp = np.log([p for p, *_ in rows]); lP = np.log([P for _, _, P, *_ in rows]); y = np.log([T for _, _, _, T, _, _ in rows])
    X = np.column_stack([np.ones(sel.sum()), lp[sel], lP[sel]]); cf = np.linalg.lstsq(X, y[sel], rcond=None)[0]
    return dict(lnA=float(cf[0]), a=float(cf[1]), b=float(-cf[2]), n=int(sel.sum()))


def fold(mf, gf, key_B="ln_B", key_g="gamma"):
    lnB, g, A, a, b = mf[key_B], mf[key_g], gf["lnA"], gf["a"], gf["b"]
    R = lambda p: A - lnB + a * math.log(p) - b * math.log(g * K(p) / C)
    thr = b * (1 - math.log(b))
    try:
        return brentq(lambda p: R(p) - thr, 50, 5000)
    except ValueError:
        return None


def main():
    db = os.path.expanduser("~/mup_laws_runs.db")
    c = sqlite3.connect(db); c.row_factory = sqlite3.Row

    mem = []   # (p, d, P, T, censored)
    for r in c.execute("select p, dim, param_count, saturation_epoch, saturated, max_epochs from runs where experiment_type='speed' and status='completed'"):
        mem.append((r["p"], r["dim"], r["param_count"], r["saturation_epoch"] if r["saturated"] else r["max_epochs"], not r["saturated"]))
    gen = []   # (p, d, P, T_gen, censored, delay98)
    for r in c.execute("select p, dim, param_count, grokking_epoch, max_epochs, npz_path from runs where experiment_type='groks' and status='completed'"):
        z = np.load(r["npz_path"]); tr, va = z["train_acc"], z["val_acc"]
        ti = np.nonzero(tr >= 99)[0]; vi = np.nonzero(va >= 98)[0]
        d98 = None
        if ti.size:
            te = ti[0] + 1; d98 = max(0, vi[0] + 1 - te) if vi.size else max(0, len(va) - te)
        gen.append((r["p"], r["dim"], r["param_count"], r["grokking_epoch"] or r["max_epochs"], r["grokking_epoch"] is None, d98))
    print(f"speed rows {len(mem)} (censored {sum(m[4] for m in mem)}), groks rows {len(gen)} (censored {sum(g[4] for g in gen)})")




    mf = fit_mem(mem); gf = fit_gen(gen)
    # Under muP on the canonical clock, almost every cell with f >= 0.08 censors at the
    # 5000-epoch cap, so the fixed-rate window leaves the OLS fit unconstrained. Report
    # a lower window (f >= 0.03), labelled as such, alongside the registered one.
    mf_lo = fit_mem(mem, window=0.03)
    print("T_mem, lower window f >= 0.03: ln B = %.2f, gamma = %.1f (OLS on %d saturated rows); Tobit incl. %d censored: ln B = %.2f, gamma = %.1f"
          % (mf_lo["ln_B"], mf_lo["gamma"], mf_lo["n_ok"], mf_lo["n_censored_window"], mf_lo["tobit_ln_B"], mf_lo["tobit_gamma"]))
    print("T_mem: ln B = %.2f, gamma = %.1f  (OLS on %d saturated window rows; Tobit incl. %d censored: ln B = %.2f, gamma = %.1f)" % (mf["ln_B"], mf["gamma"], mf["n_ok"], mf["n_censored_window"], mf["tobit_ln_B"], mf["tobit_gamma"]))
    print("T_gen: lnA = %.2f, a = %+.2f, b = %.2f  (n = %d window rows)" % (gf["lnA"], gf["a"], gf["b"], gf["n"]))
    ps = fold(mf, gf); ps_t = fold(mf, gf, "tobit_ln_B", "tobit_gamma")
    ps_lo = fold(mf_lo, gf); ps_lo_t = fold(mf_lo, gf, "tobit_ln_B", "tobit_gamma")
    fmt = lambda v: None if v is None else round(v)
    print("fold p*: window 0.08 OLS %s, Tobit %s | window 0.03 OLS %s, Tobit %s" % (fmt(ps), fmt(ps_t), fmt(ps_lo), fmt(ps_lo_t)))
    rng = np.random.default_rng(0); boots = []
    for _ in range(300):
        mb = [mem[i] for i in rng.integers(0, len(mem), len(mem))]; gb = [gen[i] for i in rng.integers(0, len(gen), len(gen))]
        try:
            v = fold(fit_mem(mb), fit_gen(gb))
            if v: boots.append(v)
        except Exception: pass
    if boots: print("bootstrap p*: median %.0f, 95%% [%.0f, %.0f], %d/300 with a fold" % (np.median(boots), np.percentile(boots, 2.5), np.percentile(boots, 97.5), len(boots)))

    # race map
    cells_m = defaultdict(list); cells_g = defaultdict(list)
    for p, d, P, T, cz in mem: cells_m[(p, d)].append(T)
    for p, d, P, T, cz, d98 in gen: cells_g[(p, d)].append((T, d98))
    rows = []
    for k in sorted(set(cells_m) & set(cells_g)):
        Tm = float(np.exp(np.mean(np.log(cells_m[k])))); Tg = float(np.exp(np.mean(np.log([t for t, _ in cells_g[k]]))))
        ds = [x for _, x in cells_g[k] if x is not None]
        rows.append(dict(p=k[0], dim=k[1], T_mem=round(Tm, 1), mem_censored=int(max(cells_m[k]) >= 5000), T_gen=round(Tg, 1), ratio=round(Tm / Tg, 2),
                         delay_mean=round(float(np.mean(ds)), 2) if ds else None, delay_min=int(min(ds)) if ds else None))
    with open("results/theory/mup_laws_matched_racemap.csv", "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("\nmuP race map (p d T_mem T_gen ratio delay_mean delay_min):")
    for r in rows: print("  ", r["p"], r["dim"], r["T_mem"], ("c" if r["mem_censored"] else ""), r["T_gen"], r["ratio"], r["delay_mean"], r["delay_min"])
    agree = [((r["ratio"] < 1) == (r["delay_min"] > 0)) for r in rows if r["delay_min"] is not None]
    print("cell-level agreement (mem faster <-> seed-min delay > 0): %.2f over %d cells" % (np.mean(agree), len(agree)))

    # p = 197 onset on the seed-min canonical delay
    o = sorted([(r["dim"], r["delay_min"]) for r in rows if r["p"] == 197 and r["delay_min"] is not None])
    Pd = {d: P for p, d, P, *_ in gen if p == 197}
    onset = None
    zeros = [d for d, dm in o if dm == 0]
    if zeros:
        last0 = max(zeros); above = [d for d, dm in o if d > last0]
        onset = above[0] if above else None
    print("p=197 muP seed-min delay by width:", o, "-> onset width", onset, "log10 P_onset", None if onset is None else round(math.log10(Pd[onset]), 2))
    json.dump(dict(mem=mf, mem_lo=mf_lo, gen=gf, p_star=ps, p_star_tobit=ps_t, p_star_lo=ps_lo, p_star_lo_tobit=ps_lo_t, boot=boots, onset197=dict(width=onset, log10P=None if onset is None else math.log10(Pd[onset]))),
              open("results/theory/mup_laws_matched.json", "w"), indent=1)


if __name__ == "__main__":
    main()
