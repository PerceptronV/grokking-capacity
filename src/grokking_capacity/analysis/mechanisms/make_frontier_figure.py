#!/usr/bin/env python3
"""Regenerate the cone figure with the apex-band data (cone-fill.md).
Writes figures/theory/frontier.pdf and syncs to the paper media dir."""
from grokking_capacity.analysis.timescale_laws import normalise_timescale_laws
import json, math, shutil
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from grokking_capacity.analysis import forecast_onset as fo

BLUE, ORANGE, AQUA, INK, MUTED, RED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#c23b3b"


def main():
    R = normalise_timescale_laws(json.load(open("results/theory/forecast_onset.json")))
    mem, gen = R["mem_law"], R["gen_law"]
    C = 2.16
    K = lambda p: 0.5 * p * (p - 1) * math.log2(p + 2)

    def roots(pp):
        Rp = (gen["lnA"] + gen["a"]*math.log(pp) - mem["ln_B"]) - gen["b"]*math.log(mem["gamma"]*K(pp)/C)
        Rs = gen["b"]*(1 - math.log(gen["b"]))
        if Rp < Rs: return None
        from scipy.optimize import brentq
        F = lambda x: x - gen["b"]*math.log(x) - Rp
        return brentq(F, gen["b"], 1e3), brentq(F, 1e-9, gen["b"])

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    pg = np.linspace(96, 167.4, 250)
    lo = [mem["gamma"]*K(p)/(C*roots(p)[0]) if roots(p) else np.nan for p in pg]
    hi = [mem["gamma"]*K(p)/(C*roots(p)[1]) if roots(p) else np.nan for p in pg]
    ax.fill_between(pg, lo, [5e7]*len(pg), color=ORANGE, alpha=.10, lw=0)
    ax.plot(pg, lo, color=ORANGE, lw=2, label="onset branch (derived)")
    ax.plot(pg, hi, color=MUTED, lw=1.2, ls=":",
            label=r"upper root $x_-$ — EXCLUDED by $\mu$P test 2026-08-10 (cone open-topped)")

    # delay-depth shading: apex-band cells coloured by seed-mean delay
    cf = json.load(open("results/theory/cone_fill_curves.json"))
    px, py, pc_ = [], [], []
    for p, d in cf.items():
        for pc, mn, mean, n in d["curve_meanmin"]:
            px.append(int(p)); py.append(pc); pc_.append(mean)
    sc = ax.scatter(px, py, c=np.clip(pc_, 0, 40), cmap="Greens", s=14, alpha=.75,
                    lw=0, zorder=3)
    plt.colorbar(sc, label="seed-mean delay (ep, clipped at 40)")

    ons = {int(k): v["P_onset"] for k, v in R["per_prime"].items()}
    ax.plot(list(ons), list(ons.values()), "o", color=BLUE, ms=7, mec="w",
            zorder=5, label="measured onsets (central)")
    apex = {149: 5.51, 151: 5.92, 157: 5.92, 163: 5.92, 167: 6.49, 173: 6.49}
    ax.plot(list(apex), [10**v for v in apex.values()], "D", color=INK, ms=7,
            mec="w", zorder=6, label="apex-band onsets (domain rule); p=179: none")
    ax.plot([197]*18, [10**x for x in np.linspace(5.03, 7.0, 18)], "x",
            color=RED, ms=6, mew=1.5, label="p=197: zero delay (verdict A)")
    ax.axvline(167.5, color=MUTED, ls=":", lw=1)
    ax.axvspan(147.8, 191.6, color=MUTED, alpha=.06, lw=0)
    ax.text(168.3, 1.4e5, "$p^*=167.5$", color=MUTED, fontsize=9)
    ax.set(xlabel="prime $p$", ylabel="parameters $P$", yscale="log",
           ylim=(6e4, 5e7), title="The grokking frontier: onsets bend upward and diverge at $p^*$;\n"
           "grokking magnitude fades to the measurement floor at the apex")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=.25, lw=.6); ax.spines[["top", "right"]].set_visible(False)
    fig.savefig("figures/theory/frontier.pdf", bbox_inches="tight")
    shutil.copy("figures/theory/frontier.pdf",
                "docs/papers/2026 Grokking Capacity/media/theory/frontier.pdf")
    print("wrote + synced frontier.pdf")


if __name__ == "__main__":
    main()
