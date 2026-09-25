#!/usr/bin/env python3
"""muP cells where random-label memorisation is faster than generalisation
yet the task run shows no delay: task train (train-mode) and validation
accuracy per epoch for every seed, with a dotted vertical line at the epoch
by which the same model memorised a random-label set of equivalent size
(seed-geometric-mean of the gc-speed saturation epoch at that (p, d)).
Fourth panel: a cell one prime earlier where the same ordering still yields
a delay. Writes figures/theory/mem_faster_unused.pdf (+ media copy)."""
import os, sqlite3, shutil, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
c = sqlite3.connect(os.path.expanduser("~/crossing_mup_runs.db")); c.row_factory = sqlite3.Row
cells = [(223, 792), (251, 1000), (281, 1000), (311, 1000)]
fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
for ax, (p, d) in zip(axs, cells):
    tm = [r["saturation_epoch"] if r["saturated"] else r["max_epochs"] for r in c.execute(
        "select saturation_epoch, saturated, max_epochs from runs where experiment_type='speed' and p=? and dim=? and status='completed'", (p, d))]
    Tm = float(np.exp(np.mean(np.log(tm))))
    for i, r in enumerate(c.execute("select npz_path from runs where experiment_type='groks' and p=? and dim=? and status='completed' order by seed", (p, d))):
        z = np.load(r["npz_path"]); ep = np.arange(1, len(z["train_acc"]) + 1)
        ax.plot(ep, z["train_acc"], color="C0", alpha=.75, lw=1.2, label="task: train accuracy" if i == 0 else None)
        ax.plot(ep, z["val_acc"], color="C2", alpha=.75, lw=1.2, label="task: validation accuracy" if i == 0 else None)
    ax.axvline(Tm, color="k", ls=":", lw=1.4, label=r"$T_{\rm mem}$: random labels of equal size memorised" if p == cells[0][0] else None)
    ax.axhline(99, color="grey", lw=.5, ls="--")
    ax.set_xlim(0, 45); ax.set_ylim(0, 102); ax.set_xlabel("epoch"); ax.set_title(f"$p={p}$, $d={d}$, $T_{{\\rm mem}}={Tm:.0f}$", fontsize=10)
axs[0].set_ylabel("accuracy (%)"); axs[0].legend(fontsize=7, loc="center right")
fig.suptitle("Per-layer scaling: memorisation is the faster clock in every panel; delay survives only in the first", fontsize=10)
fig.tight_layout(); fig.savefig("figures/theory/mem_faster_unused.pdf", bbox_inches="tight"); fig.savefig("/tmp/mem_faster_unused.png", dpi=90, bbox_inches="tight")
dst = "docs/papers/2026 Grokking Capacity/media/theory/mem_faster_unused.pdf"
if os.path.isdir(os.path.dirname(dst)): shutil.copy("figures/theory/mem_faster_unused.pdf", dst)
