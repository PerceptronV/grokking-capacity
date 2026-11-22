#!/usr/bin/env python3
"""Fixed-rate race map on the canonical protocol (registry rows only).
T_mem: gc-speed saturation epoch (train-mode running acc >= 99 %, random
labels at n_equiv), censored at max_epochs when unsaturated. Sources:
repo runs.db (p = 149, wd 1.0) and ~/apexmem_gcspeed_runs.db (151..197).
T_gen: gc-groks grokking_epoch (val >= 99 %, eval mode), censored at cap.
Sources: runs.db (149), ~/cone_fill_runs.db (149..179), ~/band_gaps_runs.db
(181/191/193, when present), ~/p197_runs.db (197, alpha 0.5).
Delay: canonical, val-98 minus train-mode train-99 from the trace, clipped
at zero, seed-mean and seed-min. Memorisation interpolated in log d onto
task widths inside its range. Writes results/theory/race_map_registry.csv
and figures/theory/race_map_registry.pdf (+ media copy).
"""
import os, sqlite3, csv, shutil
from collections import defaultdict
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt


def main():
    H = os.path.expanduser("~")

    def rows(db, sql):
        if not os.path.exists(db): return []
        c = sqlite3.connect(db); c.row_factory = sqlite3.Row
        return list(c.execute(sql))

    mem = defaultdict(list)
    for r in rows("runs.db", "select p, dim, saturation_epoch, saturated, max_epochs from runs where experiment_type='speed' and p=149 and weight_decay=1.0 and status='completed'"):
        mem[(r["p"], r["dim"])].append(r["saturation_epoch"] if r["saturated"] else r["max_epochs"])
    for r in rows(f"{H}/apexmem_gcspeed_runs.db", "select p, dim, saturation_epoch, saturated, max_epochs from runs where experiment_type='speed' and status='completed'"):
        mem[(r["p"], r["dim"])].append(r["saturation_epoch"] if r["saturated"] else r["max_epochs"])

    gen = defaultdict(list)   # (p,d) -> [(t_gen99, delay98, censored)]
    def load_groks(db, where):
        for r in rows(db, "select p, dim, grokking_epoch, max_epochs, npz_path from runs where experiment_type='groks' and status='completed' " + where):
            path = r["npz_path"]
            if not os.path.exists(path):
                path = os.path.join("data", "groks", os.path.basename(os.path.dirname(path)), "trace.npz")
            if not os.path.exists(path): continue
            z = np.load(path); tr, va = z["train_acc"], z["val_acc"]
            ti = np.nonzero(tr >= 99)[0]; vi = np.nonzero(va >= 98)[0]
            if not ti.size: continue                      # estimator gate: never fit the training set
            te = ti[0] + 1
            d98 = max(0, (vi[0] + 1 - te)) if vi.size else max(0, len(va) - te)
            gen[(r["p"], r["dim"])].append((r["grokking_epoch"] or r["max_epochs"], d98, r["grokking_epoch"] is None))
    load_groks("runs.db", "and p=149 and weight_decay=1.0")
    load_groks(f"{H}/cone_fill_runs.db", "and p between 149 and 179")
    load_groks(f"{H}/band_gaps_runs.db", "and p in (181,191,193)")
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
            Tg = float(np.exp(np.mean([np.log(t[0]) for t in g])))
            out.append(dict(p=p, dim=d, n_seeds=len(g), T_mem=round(Tm, 1), T_gen=round(Tg, 1), ratio=round(Tm / Tg, 2),
                            mem_censored=int((p, d) in mem and max(mem[(p, d)]) >= 5000),
                            delay_mean=round(float(np.mean([t[1] for t in g])), 2), delay_min=int(min(t[1] for t in g)),
                            grok_frac=round(float(np.mean([not t[2] for t in g])), 2)))
    os.makedirs("results/theory", exist_ok=True)
    with open("results/theory/race_map_registry.csv", "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

    print(f"{'p':>4} {'cells':>5} {'mem slower':>10} {'mean delay>0.5':>14} {'min delay>0':>11}  crossing width (ratio=1)")
    for p in primes:
        o = [r for r in out if r["p"] == p]
        slower = sum(r["ratio"] > 1 for r in o); dm = sum(r["delay_mean"] > 0.5 for r in o); dn = sum(r["delay_min"] > 0 for r in o)
        xs = [(r["dim"], r["ratio"]) for r in o]; cross = "none (mem slower everywhere)" if all(x[1] > 1 for x in xs) else ("none (mem faster everywhere)" if all(x[1] < 1 for x in xs) else "d≈%d" % next(d for d, rt in xs if rt < 1))
        print(f"{p:>4} {len(o):>5} {slower:>10} {dm:>14} {dn:>11}  {cross}")
    print("\np d T_mem T_gen ratio delay_mean delay_min grok_frac")
    for r in out: print(" ", r["p"], r["dim"], r["T_mem"], r["T_gen"], r["ratio"], r["delay_mean"], r["delay_min"], r["grok_frac"])

    n = len(primes); cols = 4; rws = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rws, cols, figsize=(3.0 * cols, 2.5 * rws), sharey=True, squeeze=False)
    for ax, p in zip(axs.flat, primes):
        md = sorted(d for pp, d in mem if pp == p)
        ax.plot([32 * d * d for d in md], [np.exp(np.mean(np.log(mem[(p, d)]))) for d in md], "s-", c="C3", ms=3, label=r"$T_{\rm mem}$ (gc-speed)")
        gd = sorted(dd for pp, dd in gen if pp == p)
        ax.plot([32 * d * d for d in gd], [np.exp(np.mean([np.log(t[0]) for t in gen[(p, d)]])) for d in gd], "o-", c="C0", ms=3, label=r"$T_{\rm gen}$ (gc-groks)")
        ax2 = ax.twinx(); ax2.bar([32 * d * d for d in gd], [np.mean([t[1] for t in gen[(p, d)]]) for d in gd], width=[0.25 * 32 * d * d for d in gd], color="C2", alpha=.25)
        ax2.set_ylim(0, 40); ax2.tick_params(labelsize=6, colors="C2")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"$p={p}$", fontsize=9); ax.tick_params(labelsize=7); ax.set_ylim(5, 6000)
    for ax in axs.flat[n:]: ax.axis("off")
    axs.flat[0].legend(fontsize=6, loc="upper right")
    for ax in axs[-1]: ax.set_xlabel("parameters $P$", fontsize=8)
    for ax in axs[:, 0]: ax.set_ylabel("epochs", fontsize=8)
    fig.tight_layout(); os.makedirs("figures/theory", exist_ok=True)
    fig.savefig("figures/theory/race_map_registry.pdf", bbox_inches="tight"); fig.savefig("/tmp/race_map_registry.png", dpi=80, bbox_inches="tight")
    dst = "docs/papers/2026 Grokking Capacity/media/theory/race_map_registry.pdf"
    if os.path.isdir(os.path.dirname(dst)): shutil.copy("figures/theory/race_map_registry.pdf", dst)


if __name__ == "__main__":
    main()
