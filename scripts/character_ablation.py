#!/usr/bin/env python3
"""Causal character-ablation test (restricted/excluded loss transposed to
multiplicative characters) — the check the character stack owed to the
Nanda et al. standard (tutorials/04 §4).

For each saved grokking model (runs with model.pt in --db), in the
discrete-log-reordered spectral basis of the unembedding (and optionally
the embedding):

  * excluded  — zero the winner characters' spectral content (winners =
    final unembedding share > 3x uniform), keep everything else;
  * restricted — keep ONLY the winner characters (+ the row-mean/DC and
    the residue-0 and special-token rows), zero the other characters;
  * random-k controls — exclude k random non-winner characters, k matched
    to the winner count (N_CTRL draws).

Accuracy is evaluated on the full division task (all p*(p-1) pairs) —
grokked models sit at ~100%, so the drop is the causal signal.

Usage (from repo root):
  python scripts/character_ablation.py --db ~/m0_runs.db \
      --out results/theory/character_ablation.json
"""
import argparse
import json
import math
import os
import sqlite3
from pathlib import Path

import numpy as np
import torch

from grokking_capacity.models import TransformerTorch

THRESH_X_UNIFORM = 3.0
N_CTRL = 20


def primitive_root(p):
    fac, n, d = [], p - 1, 2
    while d * d <= n:
        if n % d == 0:
            fac.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        fac.append(n)
    return next(g for g in range(2, p)
                if all(pow(g, (p - 1) // q, p) != 1 for q in fac))


def spectral_surgery(W: np.ndarray, p: int, order: list[int],
                     keep: np.ndarray | None = None,
                     drop: np.ndarray | None = None) -> np.ndarray:
    """Return a copy of W with character bins dropped/kept on the nonzero-
    residue rows. Bins are 1..(p-1)//2 (conjugate pairs; symmetric bin
    handled by rfft/irfft). Row mean (DC), residue-0 row and special-token
    rows are always preserved."""
    W2 = W.copy()
    X = W[order, :].astype(np.float64)
    mean = X.mean(0, keepdims=True)
    C = np.fft.rfft(X - mean, axis=0)
    mask = np.zeros(C.shape[0], dtype=bool)   # True = zero this bin
    if drop is not None:
        mask[drop] = True
    if keep is not None:
        mask[:] = True
        mask[keep] = False
        mask[0] = False                        # never touch DC (already 0)
    C[mask, :] = 0.0
    Xr = np.fft.irfft(C, n=p - 1, axis=0) + mean
    W2[order, :] = Xr
    return W2


def full_division_eval(model, p: int, device: str) -> float:
    xs, ys = np.meshgrid(np.arange(p), np.arange(1, p), indexing="ij")
    xs, ys = xs.ravel(), ys.ravel()
    inv = np.array([pow(int(y), p - 2, p) for y in range(1, p)])
    zs = (xs * inv[ys - 1]) % p
    # sequence layout [x, op, y, =] as in grokking_data
    X = np.stack([xs, np.full_like(xs, p), ys, np.full_like(xs, p + 1)], 1)
    X = torch.tensor(X, dtype=torch.long, device=device)
    T = torch.tensor(zs, dtype=torch.long, device=device)
    accs = []
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            logits = model(X[i:i + 8192])
            accs.append((logits.argmax(-1) == T[i:i + 8192]).float())
    return float(torch.cat(accs).mean().item()) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(Path.home() / "m0_runs.db"))
    ap.add_argument("--out", default="results/theory/character_ablation.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()

    con = sqlite3.connect(os.path.expanduser(args.db))
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "select uuid, p, dim, seed, artefacts_dir, dropout, depth, heads, "
        "init_scale from runs where experiment_type='groks' and "
        "status='completed'").fetchall()

    rng = np.random.default_rng(0)
    out_runs = []
    for r in rows:
        mp = Path(r["artefacts_dir"]) / "model.pt"
        if not mp.exists():
            continue
        p, dim = int(r["p"]), int(r["dim"])
        g = primitive_root(p)
        order = [pow(g, m, p) for m in range(p - 1)]
        n_bins = (p - 1) // 2 + 1              # rfft bins 0..(p-1)/2

        sd = torch.load(mp, map_location="cpu", weights_only=False)
        sd = sd.get("model_state_dict", sd)
        model = TransformerTorch(depth=int(r["depth"] or 2), dim=dim,
                                 heads=int(r["heads"] or 1), n_tokens=p + 2,
                                 seq_len=4, dropout=0.0,
                                 init_scale=1.0).to(args.device)
        model.load_state_dict(sd)
        model.eval()

        U = model.state_dict()["out.weight"].cpu().numpy()
        X = U[order, :].astype(np.float64)
        C = np.fft.rfft(X - X.mean(0, keepdims=True), axis=0)
        pw = (np.abs(C[1:, :]) ** 2).sum(1)
        share = pw / pw.sum()
        m = len(share)
        winners = np.where(share > THRESH_X_UNIFORM / m)[0] + 1  # bin idx
        losers = np.setdiff1d(np.arange(1, n_bins), winners)
        if winners.size == 0 or winners.size >= m - 1:
            continue

        base = full_division_eval(model, p, args.device)

        def eval_with(U_new):
            model.load_state_dict({**sd, "out.weight":
                                   torch.tensor(U_new, dtype=torch.float32)})
            return full_division_eval(model, p, args.device)

        acc_excl = eval_with(spectral_surgery(U, p, order, drop=winners))
        acc_restr = eval_with(spectral_surgery(U, p, order, keep=winners))
        ctrl = []
        for _ in range(N_CTRL):
            pick = rng.choice(losers, size=winners.size, replace=False)
            ctrl.append(eval_with(spectral_surgery(U, p, order, drop=pick)))
        model.load_state_dict(sd)

        rec = {"uuid": r["uuid"], "p": p, "dim": dim, "seed": int(r["seed"]),
               "n_winners": int(winners.size),
               "winner_share": float(share[winners - 1].sum()),
               "acc_base": base, "acc_excluded": acc_excl,
               "acc_restricted": acc_restr,
               "acc_ctrl_mean": float(np.mean(ctrl)),
               "acc_ctrl_min": float(np.min(ctrl)),
               "acc_ctrl_sd": float(np.std(ctrl, ddof=1))}
        out_runs.append(rec)
        print(f"p={p} d={dim} s={rec['seed']}: base {base:.1f} | "
              f"excl {acc_excl:.1f} | restr {acc_restr:.1f} | "
              f"ctrl {rec['acc_ctrl_mean']:.1f}±{rec['acc_ctrl_sd']:.1f} "
              f"(k={winners.size})", flush=True)

    summary = {}
    if out_runs:
        summary = {
            "n_runs": len(out_runs),
            "median_drop_excluded": float(np.median(
                [r["acc_base"] - r["acc_excluded"] for r in out_runs])),
            "median_drop_ctrl": float(np.median(
                [r["acc_base"] - r["acc_ctrl_mean"] for r in out_runs])),
            "median_acc_restricted": float(np.median(
                [r["acc_restricted"] for r in out_runs])),
        }
        print("SUMMARY:", json.dumps(summary))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"runs": out_runs, "summary": summary,
         "threshold_x_uniform": THRESH_X_UNIFORM, "n_ctrl": N_CTRL}, indent=1))
    print("saved", args.out)


if __name__ == "__main__":
    main()
