#!/usr/bin/env python3
"""Matched-protocol muP depth series (configs/mup_depth_series.yaml).
Canonical delay per run: val-98 epoch minus train-mode train-99 epoch,
clipped at zero; runs that never reach train-99 are dropped (estimator
gate). Per (p, d): seed-mean with s.e., seed-min, fraction zero.
Registered adjudication (mup-apex-locate.md 2026-09-19, re-applied on the
matched data): fit exponential fade D = A e^{-r p} and fold closing
D = A sqrt(ln(p*/p)) (zero beyond p*) to the seed-mean at each width over
the primes with nonzero mean plus the first zero; report AIC and the
fold's p*. Also report the empirical apex: first prime at which the
seed-mean is exactly zero with all later primes zero too.
Writes results/theory/mup_depth_matched.csv, figures/theory/mup_depth_matched.pdf."""
import os, sqlite3, csv, math, shutil
from collections import defaultdict
import numpy as np, matplotlib, matplotlib.ticker
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import least_squares

c = sqlite3.connect(os.path.expanduser("~/mup_depth_series_runs.db")); c.row_factory = sqlite3.Row
cells = defaultdict(list); dropped = 0
for r in c.execute("select p, dim, seed, grokking_epoch, npz_path from runs where status='completed'"):
    z = np.load(r["npz_path"]); tr, va = z["train_acc"], z["val_acc"]
    ti = np.nonzero(tr >= 99)[0]; vi = np.nonzero(va >= 98)[0]
    if not ti.size: dropped += 1; continue
    te = ti[0] + 1
    cells[(r["p"], r["dim"])].append(max(0, vi[0] + 1 - te) if vi.size else max(0, len(va) - te))
rows = []
for (p, d), v in sorted(cells.items()):
    v = np.array(v, float)
    rows.append(dict(p=p, dim=d, n=len(v), mean=round(v.mean(), 2), se=round(v.std(ddof=1) / math.sqrt(len(v)), 2) if len(v) > 1 else 0,
                     min=int(v.min()), max=int(v.max()), frac_zero=round(float((v == 0).mean()), 2)))
os.makedirs("results/theory", exist_ok=True)
with open("results/theory/mup_depth_matched.csv", "w") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"runs dropped by estimator gate: {dropped}")
print(f"{'p':>4} | {'d=880 mean (se)':>18} min  zero% | {'d=2000 mean (se)':>18} min  zero%")
ps = sorted({r["p"] for r in rows})
for p in ps:
    a = next((r for r in rows if r["p"] == p and r["dim"] == 880), None); b = next((r for r in rows if r["p"] == p and r["dim"] == 2000), None)
    f = lambda r: f"{r['mean']:>8.1f} ({r['se']:>4.1f}) {r['min']:>4} {int(100*r['frac_zero']):>5}" if r else " " * 28
    print(f"{p:>4} | {f(a)} | {f(b)}")

def aic(res, n, k): return n * math.log(max(res, 1e-12) / n) + 2 * k
for d in (880, 2000):
    sub = sorted([(r["p"], r["mean"]) for r in rows if r["dim"] == d])
    P = np.array([x[0] for x in sub], float); D = np.array([x[1] for x in sub])
    # empirical apex: first prime from which every later seed-mean is zero
    apex = next((int(P[i]) for i in range(len(P)) if all(D[i:] == 0)), None)
    # fit on primes with D>0 plus the first zero (so the closing is constrained)
    k = int(np.argmax(D == 0)) if (D == 0).any() else len(D) - 1
    Pf, Df = P[:k + 1], D[:k + 1]
    e = least_squares(lambda th: th[0] * np.exp(-th[1] * (Pf - Pf[0])) - Df, [Df[0], 0.02])
    def fold(th):
        pst = math.exp(th[1]); return np.where(Pf < pst, th[0] * np.sqrt(np.log(np.maximum(pst / Pf, 1.0))), 0.0) - Df
    fo = least_squares(fold, [Df[0], math.log(P[k] if apex else 400)], max_nfev=40000)
    n = len(Pf)
    print(f"\nd={d}: empirical apex (seed-mean exactly 0 from here on) = {apex}; fit on p<={int(Pf[-1])} ({n} primes):")
    print(f"   exponential: rate {e.x[1]:.4f}/prime (e-fold {1/e.x[1]:.0f}), AIC {aic(2*e.cost, n, 2):.1f}")
    print(f"   fold sqrt(ln(p*/p)): p* = {math.exp(fo.x[1]):.0f}, AIC {aic(2*fo.cost, n, 2):.1f}")

fig, ax = plt.subplots(figsize=(6, 4))
for d, col, mk in ((880, "C0", "o"), (2000, "C1", "s")):
    sub = sorted([(r["p"], r["mean"], r["se"]) for r in rows if r["dim"] == d])
    ax.errorbar([x[0] for x in sub], [max(x[1], 0.05) for x in sub], yerr=[x[2] for x in sub], fmt=mk + "-", color=col, ms=4, lw=1, capsize=2, label=f"$d={d}$")
ax.axhline(0.1, color="grey", lw=0.6, ls=":"); ax.text(100, 0.11, "0.1 epoch (ten-seed resolution)", fontsize=7, color="grey")
ax.set(xscale="log", yscale="log", xlabel="prime $p$", ylabel="seed-mean canonical delay (epochs)")
ax.set_xticks([97, 139, 197, 251, 311, 383, 449]); ax.set_xticklabels([97, 139, 197, 251, 311, 383, 449], fontsize=8)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig("figures/theory/mup_depth_matched.pdf", bbox_inches="tight"); fig.savefig("/tmp/mup_depth_matched.png", dpi=80)
dst = "docs/papers/2026 Grokking Capacity/media/theory/mup_depth_matched.pdf"
if os.path.isdir(os.path.dirname(dst)): shutil.copy("figures/theory/mup_depth_matched.pdf", dst)
