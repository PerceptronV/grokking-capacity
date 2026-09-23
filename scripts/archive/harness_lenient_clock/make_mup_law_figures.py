#!/usr/bin/env python3
"""Empirical timescale laws re-measured under per-layer muP, one figure per
law (style of laws_T_vs_P). Reads mupM/mupG rows from runs.jsonl; overlays
the fixed-lr laws for contrast. Rerun at sweep completion."""
import json, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
C = 2.16
K = lambda p: 0.5 * p * (p - 1) * math.log2(p + 2)
Pc = lambda p, d: 32 * d * d + 2 * (p + 2) * d
R = json.load(open("results/speedrun/../theory/forecast_onset.json")) if False else \
    json.load(open("results/theory/forecast_onset.json"))
rows = [json.loads(l) for l in open("results/speedrun/runs.jsonl")]
M = [r for r in rows if r.get("tag") == "mupM" and r["t_train"]]
G = [r for r in rows if r.get("tag") == "mupG" and r["t_gen"]]
n_note = f"interim: {len(M)}M/{len(G)}G rows"

# ---- T_mem under muP -------------------------------------------------------
f = np.array([K(r["p"]) / (C * Pc(r["p"], r["dim"])) for r in M])
t = np.array([r["t_train"] for r in M], float)
pp = np.array([r["p"] for r in M])
w = f >= 0.08
b_, g_ = np.polyfit(f[w], np.log(t[w]), 1)[::-1]
fig, ax = plt.subplots(figsize=(6, 4.2))
sc = ax.scatter(f, t, c=pp, cmap="viridis", s=22, alpha=.8, lw=0)
plt.colorbar(sc, label="prime $p$")
xs = np.linspace(f.min(), f.max(), 200)
ax.plot(xs, math.exp(b_) * np.exp(g_ * xs), color=INK, lw=2,
        label=r"$\mu$P fit (window): $\gamma=%.1f$, $b=%.1f$" % (g_, math.exp(b_)))
mm = R["mem_law"]
ax.plot(xs, mm["b"] * np.exp(mm["gamma"] * xs), color=ORANGE, lw=1.6, ls="--",
        label=r"fixed-lr law: $\gamma=42.3$, $b=3.0$")
ax.set(yscale="log", xlabel=r"capacity fraction $f=K/(C\,P)$",
       ylabel=r"$T_{\rm mem}$ (epochs)",
       title="Memorisation law under per-layer $\\mu$P (%s)" % n_note)
ax.legend(fontsize=8); ax.grid(alpha=.25)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig("figures/theory/mup_law_mem.pdf", bbox_inches="tight"); plt.close(fig)

# ---- T_gen under muP -------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.2))
lp = np.log([r["p"] for r in G]); lP = np.log([float(Pc(r["p"], r["dim"])) for r in G])
y = np.log([r["t_gen"] for r in G])
fw = np.array([K(r["p"]) / (C * Pc(r["p"], r["dim"])) for r in G]) >= 0.08
X = np.column_stack([np.ones(fw.sum()), lp[fw], lP[fw]])
cf, *_ = np.linalg.lstsq(X, y[fw], rcond=None)
lnA_, a_, beta_ = float(cf[0]), float(cf[1]), float(-cf[2])
gn = R["gen_law"]
for p, col in ((97, BLUE), (113, ORANGE), (139, "#1baf7a")):
    sel = [r for r in G if r["p"] == p]
    Ps = np.array([float(Pc(p, r["dim"])) for r in sel])
    ax.plot(Ps, [r["t_gen"] for r in sel], "o", color=col, ms=5, alpha=.8,
            label=f"p={p} ($\\mu$P)")
    Pg = np.geomspace(6e4, 3e7, 100)
    ax.plot(Pg, np.exp(lnA_ + a_ * math.log(p) - beta_ * np.log(Pg)),
            color=col, lw=1.6)
    ax.plot(Pg, np.exp(gn["lnA"] + gn["a"] * math.log(p) - gn["beta"] * np.log(Pg)),
            color=col, lw=1, ls="--", alpha=.5)
ax.set(xscale="log", yscale="log", xlabel="parameters $P$",
       ylabel=r"$T_{\rm gen}$ (epochs)",
       title="Generalisation law under $\\mu$P: $\\beta=%.2f$, $a=%.2f$ "
             "(fixed-lr: 1.11, $-0.45$; dashed) — %s" % (beta_, a_, n_note))
ax.legend(fontsize=8); ax.grid(alpha=.25, which="both")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig("figures/theory/mup_law_gen.pdf", bbox_inches="tight"); plt.close(fig)
print("gamma_muP=%.1f b_muP=%.1f | beta_muP=%.2f a_muP=%.2f" %
      (g_, math.exp(b_), beta_, a_))
