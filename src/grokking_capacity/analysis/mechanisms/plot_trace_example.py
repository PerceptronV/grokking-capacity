#!/usr/bin/env python3
"""Per-epoch accuracy traces for two muP cells from the crossing programme:
p=311 d=1000 (memorisation faster, no delay) and p=197 d=660 (memorisation
faster, delay present). Task train and validation accuracy per seed, plus the
random-label memorisation accuracy trace from the matching gc-speed run.
Writes figures/theory/trace_example.pdf (+ media copy)."""
import os, sqlite3, shutil, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
db = os.path.expanduser("~/crossing_mup_runs.db"); c = sqlite3.connect(db); c.row_factory = sqlite3.Row
cells = [(311, 1000), (197, 660)]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, (p, d) in zip(axs, cells):
    for i, r in enumerate(c.execute("select seed, npz_path from runs where experiment_type='groks' and p=? and dim=? and status='completed' order by seed", (p, d))):
        z = np.load(r["npz_path"]); ep = np.arange(1, len(z["train_acc"]) + 1)
        ax.plot(ep, z["train_acc"], color="C0", alpha=.7, lw=1.2, label="task: train (train-mode)" if i == 0 else None)
        ax.plot(ep, z["val_acc"], color="C2", alpha=.7, lw=1.2, label="task: validation" if i == 0 else None)
    for i, r in enumerate(c.execute("select seed, npz_path from runs where experiment_type='speed' and p=? and dim=? and status='completed' order by seed", (p, d))):
        z = np.load(r["npz_path"]); tr = z["train_acc_trace"] if "train_acc_trace" in z.files else z["train_acc"]
        ep = np.arange(1, len(tr) + 1)
        ax.plot(ep, tr, color="C3", alpha=.7, lw=1.2, ls="--", label="random labels: train (memorisation clock)" if i == 0 else None)
    ax.axhline(99, color="k", lw=.6, ls=":"); ax.set_xlim(0, 40); ax.set_ylim(0, 102)
    ax.set_title(f"$p={p}$, $d={d}$, per-layer scaling", fontsize=10); ax.set_xlabel("epoch"); ax.set_ylabel("accuracy (%)")
axs[0].legend(fontsize=7, loc="lower right")
fig.tight_layout(); os.makedirs("figures/theory", exist_ok=True)
fig.savefig("figures/theory/trace_example.pdf", bbox_inches="tight"); fig.savefig("/tmp/trace_example.png", dpi=90, bbox_inches="tight")
dst = "docs/papers/2026 Grokking Capacity/media/theory/trace_example.pdf"
if os.path.isdir(os.path.dirname(dst)): shutil.copy("figures/theory/trace_example.pdf", dst)
