#!/usr/bin/env python3
"""The grokking region on the canonical protocol, all registry data.
Per cell (protocol, p, d): seed-minimum canonical delay (val-98 epoch minus
train-mode train-99 epoch, clipped at zero; runs never reaching train-99
dropped). Filled marker: seed-min > 0 (grokking at every seed); open marker:
seed-min = 0. Colour: seed-mean delay. Left: fixed rate (central runs.db at
wd 1.0, cone fill, band gaps, p197 column, crossing_fixed, lr_wall_check);
right: per-layer scaling base 236 (mup_depth_series, mup_laws, crossing_mup,
lr_wall_check, open_top). Fixed-rate panel overlays the law-derived onset
crossing and p* = 167.5. Writes figures/theory/cone_seedmin.pdf (+ media)."""
import os, sys, sqlite3, json, math, shutil
from collections import defaultdict
import numpy as np, matplotlib, matplotlib.ticker
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
H = os.path.expanduser("~")
BASE = ("experiment_type='groks' and status='completed' and operation='/' and train_fraction=0.5 and dropout=0.2 "
        "and weight_decay=1.0 and lr=0.001 and batch_size=512 and depth=2 and heads=1")
SRC = {0: [("runs.db", BASE + " and mup_base=0"), (f"{H}/cone_fill_runs.db", BASE), (f"{H}/band_gaps_runs.db", BASE),
           (f"{H}/p197_runs.db", BASE), (f"{H}/crossing_fixed_runs.db", BASE), (f"{H}/lr_wall_check_runs.db", BASE + " and mup_base=0")],
       236: [(f"{H}/mup_depth_series_runs.db", BASE), (f"{H}/mup_laws_runs.db", BASE), (f"{H}/crossing_mup_runs.db", BASE),
             (f"{H}/lr_wall_check_runs.db", BASE + " and mup_base=236"), (f"{H}/open_top_runs.db", BASE)]}
def delays(db, where):
    if not os.path.exists(db): return
    c = sqlite3.connect(db); c.row_factory = sqlite3.Row
    for r in c.execute(f"select p, dim, param_count, npz_path from runs where {where}"):
        path = r["npz_path"]
        if not os.path.exists(path): path = os.path.join("data", "groks", os.path.basename(os.path.dirname(path)), "trace.npz")
        if not os.path.exists(path): continue
        try: z = np.load(path); tr, va = z["train_acc"], z["val_acc"]
        except Exception: continue
        ti = np.nonzero(tr >= 99)[0]
        if not ti.size: continue
        vi = np.nonzero(va >= 98)[0]; te = ti[0] + 1
        yield r["p"], r["dim"], r["param_count"], (max(0, vi[0] + 1 - te) if vi.size else max(0, len(va) - te))
cells = {0: defaultdict(list), 236: defaultdict(list)}; Pof = {}
for mb, srcs in SRC.items():
    for db, where in srcs:
        for p, d, P, dl in delays(db, where):
            cells[mb][(p, d)].append(dl); Pof[(p, d)] = P
R = json.load(open("results/theory/forecast_onset.json")); C = 2.16
g, lnb, lnA, a, beta = R["mem_law"]["gamma"], R["mem_law"]["ln_b"], R["gen_law"]["lnA"], R["gen_law"]["a"], R["gen_law"]["beta"]
K = lambda p: (p * (p - 1) // 2) * math.log2(p + 2)
def law_onset_P(p):
    # onset root x+ of x - beta ln x = R(p); P = gamma K / (C x)
    Rp = lnA - lnb + a * math.log(p) - beta * math.log(g * K(p) / C)
    if Rp < beta * (1 - math.log(beta)): return None
    from scipy.optimize import brentq
    x = brentq(lambda x: x - beta * math.log(x) - Rp, beta, 1e3)
    return g * K(p) / (C * x)
FIXED_ONLY = "--fixed-only" in sys.argv
if FIXED_ONLY:
    fig, ax0 = plt.subplots(1, 1, figsize=(5.0, 4.0)); axs = np.array([ax0]); panels = ((ax0, 0, "fixed learning rate"),)
else:
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True); panels = ((axs[0], 0, "fixed learning rate"), (axs[1], 236, "per-layer scaling (base 236)"))
for ax, mb, title in panels:
    cs = cells[mb]; ps = [k[0] for k in cs]; Ps = [Pof[k] for k in cs]
    mn = np.array([min(v) for v in cs.values()]); mean = np.array([np.mean(v) for v in cs.values()])
    grok = mn > 0
    ms = 16 if FIXED_ONLY else 28
    sc = ax.scatter(np.array(ps)[grok], np.array(Ps)[grok], c=np.clip(mean[grok], 1, None), cmap="viridis", norm=LogNorm(1, 1500), s=ms, edgecolors="k", linewidths=.3, label="seed-min delay > 0 (grokking)")
    ax.scatter(np.array(ps)[~grok], np.array(Ps)[~grok], facecolors="none", edgecolors="grey", s=ms*0.8, linewidths=.6, label="seed-min delay = 0")
    ax.set(yscale="log", xlabel="prime $p$", title=f"{title}: {len(cs)} cells, {sum(len(v) for v in cs.values())} runs")
    ax.set_xticks(sorted(set(ps))[::2]); ax.tick_params(axis="x", labelsize=8 if FIXED_ONLY else 7, rotation=90)
    if FIXED_ONLY:
        ax.xaxis.label.set_size(12); ax.set_title(""); ax.tick_params(axis="y", labelsize=10)
        ax.set_xticks([97, 113, 139, 149, 163, 179, 197]); ax.tick_params(axis="x", labelsize=10, rotation=0)
    if mb == 0:
        pp = np.arange(97, 168); Pl = [law_onset_P(p) for p in pp]
        ax.plot([p for p, P in zip(pp, Pl) if P], [P for P in Pl if P], "r-", lw=1.5, label="law-derived onset crossing")
        ax.axvline(167.5, color="r", ls="--", lw=1, label="$p^* = 167.5$")
    ax.legend(fontsize=9 if FIXED_ONLY else 7, loc="upper left")
axs.flat[0].set_ylabel("parameters $P$", fontsize=13 if FIXED_ONLY else 10)
cb = fig.colorbar(sc, ax=axs, fraction=.04 if FIXED_ONLY else .03, pad=.02); cb.set_label("seed-mean delay (epochs)", fontsize=10 if FIXED_ONLY else 10)
name = "cone_seedmin_fixed" if FIXED_ONLY else "cone_seedmin"
fig.savefig(f"figures/theory/{name}.pdf", bbox_inches="tight"); fig.savefig(f"/tmp/{name}.png", dpi=90, bbox_inches="tight")
dst = f"docs/papers/2026 Grokking Capacity/media/theory/{name}.pdf"
if os.path.isdir(os.path.dirname(dst)): shutil.copy(f"figures/theory/{name}.pdf", dst)
for mb in (0, 236):
    print(f"mup_base={mb}: primes {sorted(set(k[0] for k in cells[mb]))}")
