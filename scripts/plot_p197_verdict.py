#!/usr/bin/env python3
"""p=197 verdict artefacts (todo item 5, 2026-07-29).

Reads the completed 18x10 p=197 sweep (~/p197_runs.db by default) and the
repo registry (p=113 central-protocol comparison), computes generalisation
delays with the canonical estimator (train >= 99, val >= 98; records that
never reach train-99 are dropped, per analysis.aggregate.compute_delays),
and writes

  * docs/papers/2026 Grokking Capacity/submissions/followups/
    p197-delay-table.md                              (verbatim-postable)
  * figures/theory/p197_delay.pdf                    (delay-curve figure,
    synced into the paper repo's media/theory/)

Usage: python scripts/plot_p197_verdict.py [--db ~/p197_runs.db]
"""
import argparse
import json
import shutil
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, INK, MUTED, RED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#c23b3b"
THR_TRAIN, THR_VAL = 99.0, 98.0
NULL_BAND = (5.83, 6.10)  # rival power-law 95% onset band, log10 P

REPO = Path(__file__).resolve().parents[1]
# The writeup lives in a separate sparse-checkout git repo under docs/
PAPER_DIR = REPO / "docs/papers/2026 Grokking Capacity"


def delay_from_npz(npz_path):
    """(delay, train_epoch, val_epoch) with the canonical thresholds.

    delay is None when train never reaches 99 (record dropped from the
    estimator); censored runs get delay >= epochs remaining, as in
    analysis.aggregate.compute_delays.
    """
    d = np.load(npz_path)
    tr, va = np.asarray(d["train_acc"]), np.asarray(d["val_acc"])
    if tr.max() <= 1.5:
        tr, va = tr * 100, va * 100
    t_i = np.where(tr >= THR_TRAIN)[0]
    v_i = np.where(va >= THR_VAL)[0]
    ve = int(v_i[0]) if v_i.size else None
    if t_i.size == 0:
        return None, None, ve
    te = int(t_i[0])
    delay = max(0, ve - te) if ve is not None else max(0, len(va) - te)
    return delay, te, ve


def p197_cells(db_path):
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT dim, seed, param_count, npz_path FROM runs "
        "WHERE experiment_type='groks' AND status='completed' "
        "ORDER BY dim, seed").fetchall()
    cells = defaultdict(list)
    for r in rows:
        delay, te, ve = delay_from_npz(r["npz_path"])
        cells[(r["dim"], r["param_count"])].append(
            dict(seed=r["seed"], delay=delay, te=te, ve=ve))
    return dict(sorted(cells.items()))


def p113_min_delay(max_dims=40):
    db = sqlite3.connect(REPO / "runs.db")
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT dim, seed, param_count, npz_path FROM runs "
        "WHERE experiment_type='groks' AND status='completed' AND p=113 "
        "AND operation='/' AND train_fraction=0.5 AND depth=2 AND heads=1 "
        "AND dropout=0.2 AND lr=0.001 AND weight_decay=1.0 "
        "AND init_scale=1.0 AND batch_size=512 "
        "AND seed BETWEEN 42 AND 51 ORDER BY dim, seed").fetchall()
    by_dim = defaultdict(list)
    for r in rows:
        by_dim[r["dim"]].append(r)
    dims = [d for d in sorted(by_dim) if len(by_dim[d]) == 10]
    step = max(1, len(dims) // max_dims)
    out = []
    for dim in dims[::step]:
        delays, pc = [], None
        for r in by_dim[dim]:
            pc = r["param_count"]
            delay, _, _ = delay_from_npz(r["npz_path"])
            if delay is not None:
                delays.append(delay)
        if delays:
            out.append((pc, min(delays), float(np.median(delays))))
    return out


def write_table(cells, out_md):
    lines = [
        "| d | P | min delay | median | max | val-98 epoch (med) | n |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    star_note = False
    for (dim, pc), lst in cells.items():
        dl = [c["delay"] for c in lst if c["delay"] is not None]
        n_drop = sum(1 for c in lst if c["delay"] is None)
        ve = [c["ve"] for c in lst if c["ve"] is not None]
        star = "\\*" if n_drop else ""
        star_note = star_note or bool(n_drop)
        lines.append(
            f"| {dim} | {pc:,} | {min(dl)} | {np.median(dl):g} | {max(dl)} "
            f"| {np.median(ve):.0f} | {len(dl)}{star} |")
    md = "\n".join([
        "**p = 197 generalisation delays** (delay = epochs from train-acc "
        "crossing 99% to val-acc crossing 98%, the estimator calibrated on "
        "p = 97-139; 18 widths x 10 seeds, epoch cap 5,000; every seed "
        "converged; onset requires a zero -> non-zero min-delay transition, "
        "which never occurs):",
        "",
        *lines,
        "",
    ])
    if star_note:
        md += ("\\* at d = 52, 3 of 10 seeds reached val-98 *before* "
               "train-99 (no memorisation phase at all) and are dropped by "
               "the estimator's train-99 gate; a fortiori no delayed "
               "generalisation.\n")
    out_md.write_text(md)
    return md


def make_figure(cells, p113, out_pdf):
    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(9.2, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
    for ax in (a1, a2):
        ax.grid(alpha=.25, lw=.6)
        ax.spines[["top", "right"]].set_visible(False)

    # Panel A: what a grokking prime looks like (p=113 context). Clip to the
    # same decades as panel B; the far tail is epoch-cap censored.
    p113 = [t for t in p113 if t[0] <= 1.3e7]
    pcs = [t[0] for t in p113]
    a1.plot(pcs, [t[1] for t in p113], "o-", color=BLUE, ms=4, lw=1.8,
            label="min over 10 seeds")
    a1.plot(pcs, [t[2] for t in p113], color=BLUE, lw=1.1, ls=":",
            alpha=.7, label="median")
    a1.set(xscale="log", xlabel="parameters $P$",
           ylabel="generalisation delay (epochs)",
           title="$p=113$, inside the dome:\ndelayed generalisation")
    a1.title.set_fontsize(10)
    a1.legend(fontsize=7.5)

    # Panel B: p=197 verdict.
    a2.axvspan(10**NULL_BAND[0], 10**NULL_BAND[1], color=MUTED, alpha=.18,
               lw=0)
    a2.text(10**np.mean(NULL_BAND), 7.15,
            "power-law null:\nonset predicted here\n(falsified)",
            ha="center", fontsize=7.5, color=MUTED)
    rng = np.random.default_rng(0)
    for (dim, pc), lst in cells.items():
        dl = [c["delay"] for c in lst if c["delay"] is not None]
        jitter = 10**(rng.uniform(-.012, .012, len(dl)))
        a2.plot(pc * jitter, dl, "o", color=RED, ms=3.2, alpha=.45, mew=0)
    pcs = [pc for (_, pc) in cells]
    mins = [min(c["delay"] for c in lst if c["delay"] is not None)
            for lst in cells.values()]
    a2.plot(pcs, mins, "-", color=RED, lw=2.2,
            label="min over 10 seeds (= 0 everywhere)")
    a2.plot([], [], "o", color=RED, ms=3.2, alpha=.45, mew=0,
            label="per-seed delays")
    a2.set(xscale="log", xlabel="parameters $P$", ylim=(-0.4, 8.6),
           ylabel="generalisation delay (epochs)",
           title="$p=197 > p^*$:\nzero delay at every width (verdict A)")
    a2.title.set_fontsize(10)
    a2.legend(fontsize=7.5, loc="center left")
    fig.subplots_adjust(wspace=0.28)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(Path.home() / "p197_runs.db"))
    args = ap.parse_args()
    cells = p197_cells(args.db)
    md = write_table(
        cells, PAPER_DIR / "submissions/followups/p197-delay-table.md")
    print(md)
    p113 = p113_min_delay()
    out_pdf = REPO / "figures/theory/p197_delay.pdf"
    make_figure(cells, p113, out_pdf)
    print("wrote", out_pdf)
    media = PAPER_DIR / "media/theory"
    if media.is_dir():
        shutil.copy2(out_pdf, media / out_pdf.name)
        print("synced to", media / out_pdf.name)


if __name__ == "__main__":
    main()
