#!/usr/bin/env python3
"""Fixed-lr race map across the apex band, p = 149..197 (registered in
mup-apex-locate.md, 2026-09-21). Conventions, fixed before analysis:
  T_mem  : random-label saturation epoch (train acc >= 99 %), n = n_equiv
           (alpha = 0.5). Sources: registry speed rows (p = 149, wd = 1.0,
           censored at 5000), apexMem rows (151..193), fixedLrMem197 (197).
  T_gen  : first epoch with val acc >= 99 % on the modular task. Sources:
           registry / cone-fill groks rows (149..179), apexGen (181,191,193),
           p197 sweep (197). Censored at the run's cap when absent.
  t_train: first epoch with train acc >= 99 % in the *task* run (trace).
  delay  : T_gen - t_train per run; mean over seeds.
Matching: per prime, seed-mean log T_mem is interpolated (linear in log d)
onto every task width inside the T_mem width range; ratio = T_mem / T_gen.
Writes results/speedrun/fixed_lr_race_map.csv, figures/theory/race_map_fixed_lr.pdf
and syncs the figure to the paper media dir.
"""
import json, os, sqlite3, csv, shutil
from collections import defaultdict
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

H = os.path.expanduser("~")
rows = [json.loads(l) for l in open("results/speedrun/runs.jsonl")]

mem = defaultdict(list)          # (p, d) -> [T_mem]
for r in rows:
    if r["tag"] in ("apexMem", "fixedLrMem197"):
        mem[(r["p"], r["dim"])].append(r["t_train"] if r["t_train"] else 2500)
c = sqlite3.connect("runs.db")
for d, se, sat in c.execute("select dim, saturation_epoch, saturated from runs where "
                            "experiment_type='speed' and p=149 and weight_decay=1.0 and status='completed'"):
    mem[(149, d)].append(se if sat else 5000)

gen = defaultdict(list)          # (p, d) -> [(t_train, t_gen, censored)]
for r in rows:
    if r["tag"] == "apexGen":
        gen[(r["p"], r["dim"])].append((r["t_train"] or 2500, r["t_gen"] or 2500, r["t_gen"] is None, None))

def load_groks(db, where=""):
    cc = sqlite3.connect(db); cc.row_factory = sqlite3.Row
    for r in cc.execute("select p, dim, grokking_epoch, max_epochs, npz_path from runs where "
                        "experiment_type='groks' and status='completed' " + where):
        path = r["npz_path"]
        if not os.path.exists(path):
            path = os.path.join("data", "groks", os.path.basename(os.path.dirname(path)), "trace.npz")
        if not os.path.exists(path): continue
        z = np.load(path); tr, va = z["train_acc"], z["val_acc"]
        hit = np.nonzero(tr >= 99.0)[0]
        t_train = int(hit[0]) + 1 if hit.size else r["max_epochs"]
        cens = r["grokking_epoch"] is None
        v98 = np.nonzero(va >= 98.0)[0]
        d98 = (int(v98[0]) + 1 - t_train) if (v98.size and hit.size) else None
        gen[(r["p"], r["dim"])].append((t_train, r["grokking_epoch"] or r["max_epochs"], cens, d98))
load_groks("runs.db", "and p=149 and weight_decay=1.0")
load_groks(f"{H}/cone_fill_runs.db", "and p between 149 and 179")
load_groks(f"{H}/p197_runs.db", "and p=197 and train_fraction=0.5")

primes = sorted({p for p, _ in gen} & {p for p, _ in mem})
out = []
for p in primes:
    md = sorted(d for pp, d in mem if pp == p)
    lm = np.array([np.mean(np.log(mem[(p, d)])) for d in md])
    for d in sorted(dd for pp, dd in gen if pp == p):
        if not (md[0] <= d <= md[-1]): continue
        Tm = float(np.exp(np.interp(np.log(d), np.log(md), lm)))
        g = gen[(p, d)]
        Tg = float(np.exp(np.mean([np.log(t[1]) for t in g])))
        delay = float(np.mean([t[1] - t[0] for t in g]))
        frac = float(np.mean([not t[2] for t in g]))
        d98 = [t[3] for t in g if t[3] is not None]
        delay98 = round(float(np.mean(d98)), 2) if d98 else None
        out.append(dict(p=p, dim=d, n_seeds=len(g), T_mem=round(Tm, 1), T_gen=round(Tg, 1),
                        ratio=round(Tm / Tg, 2), mean_delay=round(delay, 2), delay98=delay98, grok_frac=round(frac, 2),
                        mem_censored=int((p, d) in mem and max(mem[(p, d)]) >= 2500)))
with open("results/speedrun/fixed_lr_race_map.csv", "w") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

# summary per prime
print(f"{'p':>4} {'cells':>5} {'mem<gen':>8} {'grok>0':>7} {'min ratio':>9} {'max ratio':>9}  ratio-vs-delay agreement")
for p in primes:
    o = [r for r in out if r["p"] == p]
    faster = [r["ratio"] < 1 for r in o]; grok = [r["mean_delay"] > 0.5 and r["grok_frac"] > 0.5 for r in o]
    agree = np.mean([f == g for f, g in zip(faster, grok)])
    print(f"{p:>4} {len(o):>5} {sum(faster):>8} {sum(grok):>7} {min(r['ratio'] for r in o):>9.2f} {max(r['ratio'] for r in o):>9.2f}  {agree:.2f}")
print("\ncells (p, d, T_mem, T_gen, ratio, mean delay, grok frac):")
for r in out: print("  ", r["p"], r["dim"], r["T_mem"], r["T_gen"], r["ratio"], r["mean_delay"], r["delay98"], r["grok_frac"])

# figure: one panel per prime, T_mem and T_gen vs P (~32 d^2), delay as marker size
n = len(primes); cols = 5; rws = int(np.ceil(n / cols))
fig, axs = plt.subplots(rws, cols, figsize=(3.0 * cols, 2.5 * rws), sharey=True)
for ax, p in zip(axs.flat, primes):
    md = sorted(d for pp, d in mem if pp == p)
    ax.plot([32 * d * d for d in md], [np.exp(np.mean(np.log(mem[(p, d)]))) for d in md], "s-", c="C3", ms=3, label=r"$T_{\rm mem}$")
    gd = sorted(dd for pp, dd in gen if pp == p)
    ax.plot([32 * d * d for d in gd], [np.exp(np.mean([np.log(t[1]) for t in gen[(p, d)]])) for d in gd], "o-", c="C0", ms=3, label=r"$T_{\rm gen}$")
    ax.plot([32 * d * d for d in gd], [np.mean([t[0] for t in gen[(p, d)]]) for d in gd], ":", c="C0", lw=1, label=r"$t_{\rm train}$ (task)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"$p={p}$", fontsize=9)
    ax.tick_params(labelsize=7); ax.set_ylim(5, 6000)
for ax in axs.flat[n:]: ax.axis("off")
axs.flat[0].legend(fontsize=6, loc="upper right")
for ax in axs[-1] if rws > 1 else axs: ax.set_xlabel("parameters $P$", fontsize=8)
for ax in (axs[:, 0] if rws > 1 else [axs[0]]): ax.set_ylabel("epochs", fontsize=8)
fig.tight_layout(); os.makedirs("figures/theory", exist_ok=True)
fig.savefig("figures/theory/race_map_fixed_lr.pdf", bbox_inches="tight"); fig.savefig("/tmp/race_map_fixed_lr.png", dpi=80, bbox_inches="tight")
dst = "docs/papers/2026 Grokking Capacity/media/theory/race_map_fixed_lr.pdf"
if os.path.isdir(os.path.dirname(dst)): shutil.copy("figures/theory/race_map_fixed_lr.pdf", dst)
