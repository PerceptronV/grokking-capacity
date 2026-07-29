#!/usr/bin/env python3
"""Figures for the theory comments / revision, from the frozen Day-0 fits.

  fig 1  mem_law.pdf       — ln T_mem vs f collapse, full-range + window fits
  fig 2  onset_vs_p.pdf    — the money figure: measured onsets, implicit
                             crossing law (diverging at p*), power-law null,
                             p = 197 decision region

Reads results/theory/forecast_onset.json and the registry.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grokking_capacity.analysis import forecast_onset as fo
from grokking_capacity.analysis.config_view import ConfigView

OUT = Path("results/theory/figures")
OUT.mkdir(parents=True, exist_ok=True)

res = json.load(open("results/theory/forecast_onset.json"))
view = ConfigView.from_yaml("configs/central.yaml", db_path="runs.db")
data = fo.collect(view)
C = data["C"]
mem, memf = res["mem_law"], res["mem_law_full_range"]
gen = res["gen_law"]

# ---------------------------------------------------------------- fig 1
pts = [q for q in data["speed_pts"] if q["p"] in fo.CENTRAL_PRIMES]
f = np.array([q["f"] for q in pts])
t = np.array([q["t_mem"] for q in pts])
pp = np.array([q["p"] for q in pts])

fig, ax = plt.subplots(figsize=(6.0, 4.2))
sc = ax.scatter(f, t, c=pp, cmap="viridis", s=10, alpha=0.5, lw=0)
plt.colorbar(sc, label="prime $p$")
xs = np.linspace(f.min(), f.max(), 300)
ax.plot(xs, memf["b"] * np.exp(memf["gamma"] * xs), "k-", lw=1.8,
        label=(r"full range: $\gamma=%.1f$, $b=%.1f$ ($R^2=%.2f$)"
               % (memf["gamma"], memf["b"], memf["r2"])))
xw = np.linspace(fo.MEM_F_LO, f.max(), 100)
ax.plot(xw, mem["b"] * np.exp(mem["gamma"] * xw), "r--", lw=1.8,
        label=(r"window $f\geq %.2f$: $\gamma=%.1f$, $b=%.1f$"
               % (fo.MEM_F_LO, mem["gamma"], mem["b"])))
ax.axvspan(fo.MEM_F_LO, f.max(), color="0.85", alpha=0.4, lw=0)
ax.set_yscale("log")
ax.set_xlabel(r"capacity fraction  $f = K/(C\,P)$   ($C=2.16$)")
ax.set_ylabel(r"$T_{\rm mem}$  (epochs)")
ax.legend(fontsize=8, loc="upper left")
ax.set_title("Memorisation law: single exponential, steepening with load",
             fontsize=10)
ax.grid(alpha=0.3, which="both")
fig.savefig(OUT / "mem_law.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- fig 2
onsets = {p: fo.empirical_onset(data["delay_records"][p])
          for p in fo.CENTRAL_PRIMES + (fo.HELD_OUT_PRIME,)}
dh = res["delta_hat_dex"]

def crossing_curve(p_grid):
    out = []
    for p in p_grid:
        K = 0.5 * p * (p - 1) * math.log2(p + 2)
        L = fo.solve_crossing(int(round(p)), K, C, mem, gen)
        out.append(np.nan if L is None else 10 ** (L + dh))
    return np.array(out)

pg = np.linspace(95, 200, 300)
cross = crossing_curve(pg)
null = res["null_powerlaw"]

fig, ax = plt.subplots(figsize=(6.4, 4.6))
pc = sorted(fo.CENTRAL_PRIMES)
ax.plot([p for p in pc], [onsets[p] for p in pc], "o", color="#2a78d6",
        ms=7, mec="white", mew=0.5, zorder=5, label="measured onset (fit set)")
ax.plot([149], [onsets[149]], "D", color="#0b0b0b", ms=7, mec="white",
        zorder=5, label="p = 149 (held out)")
ax.plot(pg, cross, "-", color="#eb6834", lw=2,
        label=r"crossing law + $\hat\Delta$ (diverges at $p^*$)")
ax.plot(pg, 10 ** (null["intercept"] + null["slope"] * np.log10(pg)),
        "--", color="#1baf7a", lw=2,
        label=r"power-law null ($s=%.2f$)" % null["slope"])
ps, ci = res["divergence"]["p_star_point"], res["divergence"]["p_star_ci95"]
ax.axvline(ps, color="#eb6834", lw=1, ls=":")
ax.axvspan(ci[0], ci[1], color="#eb6834", alpha=0.10, lw=0)
ax.text(ps + 1, 6e4, r"$p^*=%.0f$" % ps, color="#eb6834", fontsize=9)
lo, hi = res["null_powerlaw"]["p197"]["total_band_log10"]
ax.plot([197, 197], [10 ** lo, 10 ** hi], "-", color="#1baf7a", lw=5,
        alpha=0.6, solid_capstyle="round")
ax.plot([197], [10 ** res["null_powerlaw"]["p197"]["log10_P_onset"]], "s",
        color="#1baf7a", ms=6, mec="white",
        label="null prediction at 197 (95% band)")
ax.annotate("crossing model:\nno onset at 197", (197, 2.0e7),
            ha="right", fontsize=8.5, color="#eb6834")
ax.set_xscale("log")
ax.set_yscale("log")
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
ax.xaxis.set_major_locator(FixedLocator([100, 120, 140, 160, 180, 200]))
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("prime $p$")
ax.set_ylabel(r"onset parameter count  $P_{\rm onset}$")
ax.set_ylim(5e4, 4e7)
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3, which="both")
fig.savefig(OUT / "onset_vs_p.pdf", bbox_inches="tight")
plt.close(fig)

print("wrote", OUT / "mem_law.pdf", "and", OUT / "onset_vs_p.pdf")
