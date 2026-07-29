#!/usr/bin/env python3
"""Visual explainer for the theory program -> figures/theory/*.pdf
1 laws_T_vs_P    - the two timescale laws + rival forms at p=113
2 gamma_knobs    - what sets gamma: measured vs thermal/geometric predictions
3 dome           - the grokking dome in (p,P); onsets, re-entrance, p=197 verdict
4 tgen_lambda    - T_gen ~ lambda^-1/2 vs rejected 1/lambda prediction
"""
import json, math, sqlite3
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BLUE, ORANGE, AQUA, INK, MUTED, RED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#c23b3b"
OUT = Path("figures/theory"); OUT.mkdir(parents=True, exist_ok=True)
R = json.load(open("results/theory/forecast_onset.json"))
mem, gen = R["mem_law"], R["gen_law"]
C = 2.16
K = lambda p: 0.5*p*(p-1)*math.log2(p+2)

def st(ax):
    ax.grid(alpha=.25, lw=.6); ax.spines[["top","right"]].set_visible(False)

# ---- 1: the two laws and rival forms (p=113) --------------------------------
p = 113; Pg = np.logspace(4.7, 7, 200); f = K(p)/(C*Pg)
Tm = mem["b"]*np.exp(mem["gamma"]*f)
Tg = np.exp(gen["lnA"] + gen["a"]*math.log(p) - gen["beta"]*np.log(Pg))
fig, ax = plt.subplots(figsize=(6,4.2))
ax.plot(Pg, Tm, color=BLUE, lw=2.2, label=r"$T_{\rm mem}=b\,e^{\gamma K/CP}$ (fit)")
ax.plot(Pg, Tg, color=ORANGE, lw=2.2, label=r"$T_{\rm gen}=A\,p^a P^{-\beta}$ (fit)")
# rival T_mem forms (fit elsewhere, shown rejected): power law & critical
ax.plot(Pg, 3e3*(f/0.1)**2.2, color=BLUE, lw=1.2, ls=":", label=r"rival: power law (rejected, $\Delta$AIC$>$4000)")
Lc = R["per_prime"]["113"]["log10_P_cross_fit"]
ax.axvline(10**Lc, color=MUTED, lw=1, ls="--")
ax.annotate("crossing\n(onset $\\approx$ here $\\times 10^{-0.22}$)", (10**Lc, 2e3), fontsize=8, color=INK)
Pon = R["per_prime"]["113"]["P_onset"]
ax.plot([Pon],[np.interp(Pon,Pg,Tg)],"*",ms=14,color=RED,label="measured onset")
ax.set(xscale="log", yscale="log", xlabel="parameters $P$", ylabel="epochs",
       title="p=113: grokking begins where memorisation overtakes generalisation")
ax.legend(fontsize=7.5); st(ax); fig.savefig(OUT/"laws_T_vs_P.pdf", bbox_inches="tight"); plt.close(fig)

# ---- 2: what sets gamma ------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6,3.6))
B = np.array([512, 2048]); g = np.array([45.5, 23.8])
ge = np.array([[45.5-33.6, 55.4-45.5],[23.8-17.1, 27.8-23.8]]).T
a1.errorbar(B, g, yerr=ge, fmt="o", ms=8, color=INK, capsize=4, label="measured $\\gamma_B$ (steps)")
Bg = np.geomspace(400, 2600, 50)
a1.plot(Bg, 45.5*(Bg/512), color=RED, ls="--", lw=1.6, label="thermal: $\\gamma\\propto B$ (rejected)")
a1.plot(Bg, np.full_like(Bg,45.5), color=AQUA, ls=":", lw=1.6, label="geometric: flat (rejected)")
a1.plot(Bg, 45.5*(Bg/512)**-.47, color=BLUE, lw=2, label="measured: $\\gamma\\propto B^{-1/2}$")
a1.set(xscale="log", yscale="log", xlabel="batch size $B$", ylabel=r"$\gamma$",
       title="noise knob: noise-impeded storage")
a1.legend(fontsize=7); st(a1)
lam = np.array([0.0,.01,.03,.1,.3,1.0]); gl = np.array([20.3,20.9,22.9,24.9,34.0,42.4])
a2.plot(lam, gl, "o-", color=BLUE, ms=7)
a2.set(xlabel=r"weight decay $\lambda$", ylabel=r"$\gamma_\lambda$",
       title=r"decay knob: $\gamma$ doubles, $\lambda=0\to1$")
st(a2); fig.savefig(OUT/"gamma_knobs.pdf", bbox_inches="tight"); plt.close(fig)

# ---- 3: the dome -------------------------------------------------------------
from grokking_capacity.analysis import forecast_onset as fo
def roots(pp):
    Rp = (gen["lnA"]+gen["a"]*math.log(pp)-mem["ln_b"]) - gen["beta"]*math.log(mem["gamma"]*K(pp)/C)
    Rs = gen["beta"]*(1-math.log(gen["beta"]))
    if Rp < Rs: return None
    from scipy.optimize import brentq
    F = lambda x: x-gen["beta"]*math.log(x)-Rp
    return brentq(F, gen["beta"], 1e3), brentq(F, 1e-9, gen["beta"])
pgr = np.linspace(96, 167.4, 250)
lo, hi = [], []
for pp in pgr:
    r = roots(pp)
    lo.append(mem["gamma"]*K(pp)/(C*r[0]) if r else np.nan)
    hi.append(mem["gamma"]*K(pp)/(C*r[1]) if r else np.nan)
fig, ax = plt.subplots(figsize=(6.4,4.6))
ax.fill_between(pgr, lo, hi, color=ORANGE, alpha=.15, lw=0)
ax.plot(pgr, lo, color=ORANGE, lw=2, label="onset branch (derived)")
ax.plot(pgr, hi, color=ORANGE, lw=2, ls="--", label="re-entrant branch (derived)")
ons = {int(k): v["P_onset"] for k, v in R["per_prime"].items()}
ax.plot(list(ons), list(ons.values()), "o", color=BLUE, ms=7, mec="w", zorder=5, label="measured onsets")
wd = json.load(open("results/theory/wide_dim_delays.json"))
zz = [(139, pc) for dim, pc, mind, n in wd["139"] if mind == 0]
ax.plot([z[0] for z in zz], [z[1] for z in zz], "s", color=AQUA, ms=5, label="p=139 zero-delay (re-entrance seen)")
ax.axvline(167.5, color=MUTED, ls=":", lw=1); ax.text(168, 2e5, "$p^*=167.5$", fontsize=9, color=MUTED)
p197P = [10**x for x in np.linspace(5.03, 7.0, 18)]
ax.plot([197]*18, p197P, "x", color=RED, ms=6, mew=1.6,
        label="p=197: zero delay everywhere (verdict A)")
ax.set(xlabel="prime $p$", ylabel="parameters $P$", yscale="log",
       title="The grokking dome: bounded region, closing at $p^*$")
ax.legend(fontsize=7.5, loc="upper left"); st(ax)
fig.savefig(OUT/"dome.pdf", bbox_inches="tight"); plt.close(fig)

# ---- 4: T_gen vs lambda ------------------------------------------------------
con = sqlite3.connect("runs.db"); con.row_factory = sqlite3.Row
cells = {}
for r in con.execute("select weight_decay w, dim, grokking_epoch g from runs where experiment_type='groks' and status='completed' and p=113 and grokking_epoch is not null and dropout=0.2 and lr=0.001 and weight_decay in (0.01,0.03,0.1,0.3,1.0)"):
    cells.setdefault((r["w"], r["dim"]), []).append(r["g"])
fig, ax = plt.subplots(figsize=(5.6,4))
lams = [0.01,0.03,0.1,0.3,1.0]
for dim, col in ((56,BLUE),(112,ORANGE),(224,AQUA)):
    T = [np.mean(cells[(l,dim)]) for l in lams]
    ax.plot(lams, T, "o-", color=col, ms=6, label=f"d={dim} (measured)")
xg = np.geomspace(.01,1,50)
ax.plot(xg, 700*(xg/0.1)**-.42, color=INK, lw=2, ls="-", alpha=.6, label=r"$\lambda^{-0.42}$ (fit)")
ax.plot(xg, 700*(xg/0.1)**-1, color=RED, lw=1.4, ls="--", label=r"norm-min. $1/\lambda$ (rejected)")
ax.set(xscale="log", yscale="log", xlabel=r"weight decay $\lambda$", ylabel=r"$T_{\rm gen}$ (epochs)",
       title=r"$T_{\rm gen}\propto\lambda^{-1/2}$, not $1/\lambda$  (p=113)")
ax.legend(fontsize=7.5); st(ax)
fig.savefig(OUT/"tgen_lambda.pdf", bbox_inches="tight"); plt.close(fig)
print("wrote 4 figures to", OUT)

# sync into the writeup repo (docs/ is a sparse checkout of writings.git)
import shutil
media = Path("docs/papers/2026 Grokking Capacity/media/theory")
if media.is_dir():
    for f in OUT.glob("*.pdf"):
        shutil.copy2(f, media/f.name)
    print("synced figures to", media)
