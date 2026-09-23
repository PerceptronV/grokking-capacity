#!/usr/bin/env python3
"""Crossing programme analysis, as registered (theory/registrations.md R1).
Per prime: seed-geometric-mean ln T_mem and ln T_gen vs ln P (registry
param_count; censored cells at the cap), local linear fit to each over the
grid, intersection = crossing, 95 % CI from 300 seed-bootstrap resamples;
'none' if no intersection inside the grid or memorisation slower everywhere.
Onset: canonical estimator (smallest width above the last zero of the
per-seed-minimum canonical delay). Mechanism score: agreement between
'mem faster' and 'seed-min delay > 0' per cell. Fixed-rate law crossing
from results/theory/forecast_onset.json for comparison.
Usage: python scripts/crossing_analysis.py fixed|mup
Writes results/theory/crossing_<which>.csv and figures/theory/crossing_<which>.pdf."""
import os, sys, sqlite3, math, csv, json, shutil
from collections import defaultdict
import numpy as np, matplotlib, matplotlib.ticker
matplotlib.use("Agg"); import matplotlib.pyplot as plt

which = sys.argv[1] if len(sys.argv) > 1 else "fixed"
db = os.path.expanduser(f"~/crossing_{which}_runs.db")
c = sqlite3.connect(db); c.row_factory = sqlite3.Row
R = json.load(open("results/theory/forecast_onset.json")); C = 2.16
g, lnb, lnA, a, beta = R["mem_law"]["gamma"], R["mem_law"]["ln_b"], R["gen_law"]["lnA"], R["gen_law"]["a"], R["gen_law"]["beta"]
K = lambda p: (p * (p - 1) // 2) * math.log2(p + 2)

mem = defaultdict(list); gen = defaultdict(list); Pof = {}
for r in c.execute("select p, dim, param_count, saturation_epoch, saturated, max_epochs from runs where experiment_type='speed' and status='completed'"):
    mem[(r["p"], r["dim"])].append(r["saturation_epoch"] if r["saturated"] else r["max_epochs"]); Pof[(r["p"], r["dim"])] = r["param_count"]
for r in c.execute("select p, dim, param_count, grokking_epoch, max_epochs, npz_path from runs where experiment_type='groks' and status='completed'"):
    z = np.load(r["npz_path"]); tr, va = z["train_acc"], z["val_acc"]
    ti = np.nonzero(tr >= 99)[0]; vi = np.nonzero(va >= 98)[0]
    d98 = None
    if ti.size:
        te = ti[0] + 1; d98 = max(0, vi[0] + 1 - te) if vi.size else max(0, len(va) - te)
    gen[(r["p"], r["dim"])].append((r["grokking_epoch"] or r["max_epochs"], d98)); Pof[(r["p"], r["dim"])] = r["param_count"]

def crossing(lP, lm, lg):
    """Local estimator (registration R1: 'local linear fit'): root of the
    piecewise-linear interpolant of gap = ln T_mem - ln T_gen in log10 P at
    the first downward sign change (memorisation slower -> faster). The
    first implementation fitted one straight line through the whole grid;
    that is not local and was replaced before scoring (see registrations.md)."""
    gap = lm - lg
    if np.all(gap > 0): return "none (mem slower everywhere)"
    if np.all(gap < 0): return "none (mem faster everywhere)"
    for i in range(len(gap) - 1):
        if gap[i] > 0 and gap[i + 1] <= 0:
            return float(lP[i] + (lP[i + 1] - lP[i]) * gap[i] / (gap[i] - gap[i + 1]))
    return None

def onset_domain(dmin, lP):
    """Canonical onset with the registered estimator-domain rule (addendum,
    2026-07): W = largest width such that no two consecutive widths >= W both
    have zero seed-min delay; walk on widths < W; onset = width after the
    last zero in that domain. Returns log10 P or a string."""
    n = len(dmin); W = n
    for i in range(n - 1, 0, -1):
        if dmin[i] == 0 and dmin[i - 1] == 0: W = i - 1
        else: break
    dom = list(range(W))
    if not dom: return "none (zero everywhere)"
    zeros = [i for i in dom if dmin[i] == 0]
    if not zeros: return f"<= {lP[0]:.2f} (positive at narrowest)"
    if zeros[-1] + 1 < n: return float(lP[zeros[-1] + 1])
    return "none"

rows = []
primes = sorted({p for p, _ in gen} & {p for p, _ in mem})
for p in primes:
    ds = sorted(d for pp, d in gen if pp == p and (pp, d) in mem)
    if len(ds) < 4: continue
    lP = np.log10([Pof[(p, d)] for d in ds])
    lm = np.array([np.mean(np.log(mem[(p, d)])) for d in ds]); lg = np.array([np.mean(np.log([t for t, _ in gen[(p, d)]])) for d in ds])
    x = crossing(lP, lm, lg)
    boots = []
    rng = np.random.default_rng(0)
    for _ in range(300):
        bm = np.array([np.mean(np.log(rng.choice(mem[(p, d)], len(mem[(p, d)])))) for d in ds])
        bg = np.array([np.mean(np.log(rng.choice([t for t, _ in gen[(p, d)]], len(gen[(p, d)])))) for d in ds])
        v = crossing(lP, bm, bg)
        if isinstance(v, float): boots.append(v)
    ci = (np.percentile(boots, 2.5), np.percentile(boots, 97.5)) if len(boots) > 30 else (None, None)
    dmin = [min([x_ for _, x_ in gen[(p, d)] if x_ is not None], default=None) for d in ds]
    onset = onset_domain([0 if v is None else v for v in dmin], lP)
    graze = float(np.mean([abs(lm[i] - lg[i]) <= math.log(1.3) for i in range(len(ds) // 2, len(ds))]))
    # law crossing (fixed-rate constants)
    lawx = None
    if which == "fixed":
        gaps = [(lnb + g * K(p) / (C * Pof[(p, d)])) - (lnA + a * math.log(p) - beta * math.log(Pof[(p, d)])) for d in ds]
        for i in range(len(ds) - 1):
            if gaps[i] > 0 and gaps[i + 1] <= 0:
                lawx = lP[i] + (lP[i + 1] - lP[i]) * gaps[i] / (gaps[i] - gaps[i + 1]); break
    agree = np.mean([(lm[i] < lg[i]) == (dmin[i] is not None and dmin[i] > 0) for i in range(len(ds))])
    ratio_wide = [round(float(np.exp(lm[i] - lg[i])), 2) for i in range(len(ds) // 2, len(ds))]
    rows.append(dict(p=p, n_widths=len(ds), crossing_log10P=(round(x, 2) if isinstance(x, float) else x), ci_lo=None if ci[0] is None else round(ci[0], 2),
                     ci_hi=None if ci[1] is None else round(ci[1], 2), onset_log10P=(round(onset, 2) if isinstance(onset, float) else onset),
                     law_crossing_log10P=None if lawx is None else round(float(lawx), 2), mech_agreement=round(float(agree), 2), graze_frac_wide=round(graze, 2), ratio_wide_half=ratio_wide,
                     seedmin_delay_by_width=dmin))
if not rows:
    sys.exit(f"no prime with >= 4 matched widths yet ({len(mem)} mem cells, {len(gen)} gen cells)")
os.makedirs("results/theory", exist_ok=True)
with open(f"results/theory/crossing_{which}.csv", "w") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{'p':>4} {'crossing':>28} {'95% CI':>14} {'onset':>30} {'law':>6} {'agree':>6} {'graze':>6}  ratio (wide half)")
for r in rows:
    print(f"{r['p']:>4} {str(r['crossing_log10P']):>28} {str((r['ci_lo'], r['ci_hi'])):>14} {str(r['onset_log10P']):>30} {str(r['law_crossing_log10P']):>6} {r['mech_agreement']:>6} {r['graze_frac_wide']:>6}  {r['ratio_wide_half']}")

n = len(rows); cols = 4; rws = int(np.ceil(n / cols))
fig, axs = plt.subplots(rws, cols, figsize=(3.0 * cols, 2.5 * rws), squeeze=False, sharey=True)
for ax, r in zip(axs.flat, rows):
    p = r["p"]; ds = sorted(d for pp, d in gen if pp == p and (pp, d) in mem)
    P = [Pof[(p, d)] for d in ds]
    ax.plot(P, [np.exp(np.mean(np.log(mem[(p, d)]))) for d in ds], "s-", c="C3", ms=3, label=r"$T_{\rm mem}$")
    ax.plot(P, [np.exp(np.mean(np.log([t for t, _ in gen[(p, d)]]))) for d in ds], "o-", c="C0", ms=3, label=r"$T_{\rm gen}$")
    if isinstance(r["crossing_log10P"], float): ax.axvline(10 ** r["crossing_log10P"], color="k", lw=0.8, ls="--")
    if isinstance(r["onset_log10P"], float): ax.axvline(10 ** r["onset_log10P"], color="C2", lw=0.8, ls=":")
    if r["law_crossing_log10P"]: ax.axvline(10 ** r["law_crossing_log10P"], color="C1", lw=0.8, ls="-.")
    ax.set(xscale="log", yscale="log", title=f"$p={p}$"); ax.tick_params(labelsize=7); ax.title.set_fontsize(9)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=4))
    ax.xaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation(base=10))
for ax in axs.flat[n:]: ax.axis("off")
axs.flat[0].legend(fontsize=6)
for ax in axs[-1]: ax.set_xlabel("parameters $P$", fontsize=8)
for ax in axs[:, 0]: ax.set_ylabel("epochs", fontsize=8)
fig.tight_layout(); fig.savefig(f"figures/theory/crossing_{which}.pdf", bbox_inches="tight"); fig.savefig(f"/tmp/crossing_{which}.png", dpi=80, bbox_inches="tight")
dst = f"docs/papers/2026 Grokking Capacity/media/theory/crossing_{which}.pdf"
if os.path.isdir(os.path.dirname(dst)): shutil.copy(f"figures/theory/crossing_{which}.pdf", dst)
