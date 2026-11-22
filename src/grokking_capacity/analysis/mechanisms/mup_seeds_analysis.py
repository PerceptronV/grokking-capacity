#!/usr/bin/env python3
"""R2: per-seed canonical delay distribution at muP wide cells beyond p~250
(configs/mup_seeds.yaml, ~/mup_seeds_runs.db). Per (p, d): n seeds, delays
sorted, fraction with delay >= 5, max, mean, and the task train-99 / val-98
epochs; plus a histogram figure. Writes results/theory/mup_seeds.csv and
figures/theory/mup_seeds_hist.pdf (+ media copy)."""
import os, sqlite3, csv, shutil
from collections import defaultdict
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
c = sqlite3.connect(os.path.expanduser("~/mup_seeds_runs.db")); c.row_factory = sqlite3.Row
cells = defaultdict(list)
for r in c.execute("select p, dim, seed, grokking_epoch, max_epochs, npz_path from runs where status='completed'"):
    z = np.load(r["npz_path"]); tr, va = z["train_acc"], z["val_acc"]
    ti = np.nonzero(tr >= 99)[0]; vi = np.nonzero(va >= 98)[0]
    if not ti.size: cells[(r["p"], r["dim"])].append((None, None, None, r["seed"])); continue
    te = ti[0] + 1; ve = vi[0] + 1 if vi.size else None
    cells[(r["p"], r["dim"])].append((max(0, ve - te) if ve else max(0, len(va) - te), te, ve, r["seed"]))
rows = []
print("p     d | n | delays sorted | frac>=5 | max | mean | mean train-99 | mean val-98")
for k in sorted(cells):
    g = [x for x in cells[k] if x[0] is not None]; ds = sorted(x[0] for x in g)
    frac5 = float(np.mean([d >= 5 for d in ds])) if ds else float("nan")
    rows.append(dict(p=k[0], dim=k[1], n=len(g), n_nofit=len(cells[k]) - len(g), frac_ge5=round(frac5, 3), max=max(ds) if ds else None,
                     mean=round(float(np.mean(ds)), 2) if ds else None, frac_zero=round(float(np.mean([d == 0 for d in ds])), 3) if ds else None,
                     mean_train99=round(float(np.mean([x[1] for x in g])), 1) if g else None, mean_val98=round(float(np.mean([x[2] for x in g if x[2]])), 1) if g else None,
                     delays=" ".join(map(str, ds))))
    print(f"{k[0]:>3} {k[1]:>5} | {len(g):>2} | {ds} | {frac5:.2f} | {max(ds) if ds else None} | {np.mean(ds) if ds else float('nan'):.1f} | {rows[-1]['mean_train99']} | {rows[-1]['mean_val98']}")
os.makedirs("results/theory", exist_ok=True)
with open("results/theory/mup_seeds.csv", "w") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
fig, axs = plt.subplots(1, len(rows), figsize=(2.6 * len(rows), 2.6), sharey=True, squeeze=False)
for ax, r in zip(axs.flat, rows):
    ds = [int(x) for x in r["delays"].split()] if r["delays"] else []
    ax.hist(ds, bins=np.arange(-0.5, max(ds + [5]) + 1.5, 1), color="C0", alpha=.8)
    ax.set_title(f"$p={r['p']}$, $d={r['dim']}$", fontsize=9); ax.set_xlabel("canonical delay (epochs)", fontsize=8); ax.tick_params(labelsize=7)
axs.flat[0].set_ylabel("seeds", fontsize=8)
fig.tight_layout(); fig.savefig("figures/theory/mup_seeds_hist.pdf", bbox_inches="tight"); fig.savefig("/tmp/mup_seeds_hist.png", dpi=90, bbox_inches="tight")
dst = "docs/papers/2026 Grokking Capacity/media/theory/mup_seeds_hist.pdf"
if os.path.isdir(os.path.dirname(dst)): shutil.copy("figures/theory/mup_seeds_hist.pdf", dst)
