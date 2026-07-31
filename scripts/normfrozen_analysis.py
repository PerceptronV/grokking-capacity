#!/usr/bin/env python3
"""Normalisation-clock test analysis (principled/gen-solvable-circuit
falsifier). Compares RMSNorm-frozen groks runs (~/normfrozen_runs.db)
against matched unfrozen cells in the repo registry.

Predictions under the spherical-motion-clock reading of the λ^(−1/2) law:
  (i) frozen λ-slope d ln T_gen/d ln λ steepens from ≈−0.42 toward −1;
  (ii) frozen width-slope β collapses toward the λ→0 value (~0.2).

Usage: python scripts/normfrozen_analysis.py [--db ~/normfrozen_runs.db]
Output: results/theory/normfrozen_analysis.json
"""
import argparse
import json
import math
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

P = 113
LAMBDAS = (0.1, 0.3, 1.0)
DIMS_BETA = (64, 128, 256)
SEEDS = (42, 43, 44)


def cells_from(db_path, *, where_extra=""):
    con = sqlite3.connect(os.path.expanduser(db_path))
    con.row_factory = sqlite3.Row
    q = ("select dim, weight_decay wd, seed, param_count, grokking_epoch g "
         "from runs where experiment_type='groks' and status='completed' "
         f"and p={P} and operation='/' and train_fraction=0.5 and depth=2 "
         "and heads=1 and dropout=0.2 and lr=0.001 and init_scale=1.0 "
         "and batch_size=512 and grokking_epoch is not null " + where_extra)
    out = defaultdict(list)
    pc = {}
    for r in con.execute(q):
        out[(int(r["dim"]), float(r["wd"]))].append(float(r["g"]))
        pc[int(r["dim"])] = float(r["param_count"])
    return out, pc


def slope(xs, ys):
    b, a = np.polyfit(np.log(xs), np.log(ys), 1)
    return float(b)


def analyse(cells, pc, label):
    res = {"label": label, "cells": {f"d{d}_wd{w:g}": {
        "n": len(v), "mean_T": float(np.mean(v)), "T": v}
        for (d, w), v in sorted(cells.items())}}
    lam_pts = [(w, np.mean(cells[(128, w)])) for w in LAMBDAS
               if (128, w) in cells and cells[(128, w)]]
    if len(lam_pts) >= 2:
        res["lambda_slope_d128"] = slope([x for x, _ in lam_pts],
                                         [y for _, y in lam_pts])
    wid_pts = [(pc[d], np.mean(cells[(d, 1.0)])) for d in DIMS_BETA
               if (d, 1.0) in cells and cells[(d, 1.0)] and d in pc]
    if len(wid_pts) >= 2:
        res["beta_lambda1"] = -slope([x for x, _ in wid_pts],
                                     [y for _, y in wid_pts])
    print(f"[{label}] lambda-slope(d=128): "
          f"{res.get('lambda_slope_d128', float('nan')):+.3f}   "
          f"beta(wd=1): {res.get('beta_lambda1', float('nan')):+.3f}")
    for k, v in res["cells"].items():
        print(f"   {k}: n={v['n']} T_gen mean {v['mean_T']:.0f}  {v['T']}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="~/normfrozen_runs.db")
    ap.add_argument("--out", default="results/theory/normfrozen_analysis.json")
    args = ap.parse_args()

    frozen, pc_f = cells_from(args.db)
    seeds = ",".join(str(s) for s in SEEDS)
    unfrozen, pc_u = cells_from("runs.db",
                                where_extra=f"and seed in ({seeds})")
    out = {"frozen": analyse(frozen, pc_f, "frozen"),
           "unfrozen_baseline": analyse(unfrozen, pc_u, "unfrozen"),
           "prediction": {"lambda_slope": "frozen steepens toward -1 "
                          "(unfrozen ~ -0.42)",
                          "beta": "frozen collapses toward ~0.2 "
                          "(unfrozen ~ 1.1)"}}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("saved", args.out)


if __name__ == "__main__":
    main()
