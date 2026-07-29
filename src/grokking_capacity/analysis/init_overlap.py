"""m₀ — Fourier structure of token embeddings at initialisation.

Implements the respecced m₀ measurement of theory-trunk §0.7:

(a) ``null`` — the **max init-overlap** statistic. At random init no
    frequency is distinguished, so the well-posed quantity is
    F_max(p, d, seed) = max_k F_k over the (p−1)/2 candidate frequencies,
    where F_k is frequency k's share of the numeric-token embedding's
    non-DC Fourier power. Computed on real model inits across a (p, d)
    grid and compared to a matched i.i.d.-Gaussian Monte-Carlo null
    (extreme-value enhancement over the uniform share 2/(p−1)).

(b) ``selection`` — the causal test. For trained grokking runs that saved
    both ``model_init.pt`` and ``model.pt`` (the exact weights training
    started from and ended at), identify the run's key frequencies from
    the *final* embedding and ask whether those frequencies were already
    over-represented at *init* relative to the non-selected population.
    Positive ⇒ frequency selection is initialisation-seeded (saddle-escape
    support); null ⇒ selection is dynamical and g(p) stays
    phenomenological.

Usage:
  python -m grokking_capacity.analysis.init_overlap null \
      --out results/theory/init_overlap_null.json
  python -m grokking_capacity.analysis.init_overlap selection \
      --db ~/m0_runs.db --out results/theory/init_overlap_selection.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

SELECT_THRESHOLD_X_UNIFORM = 3.0   # F_k > 3× uniform share ⇒ "key" frequency
NULL_MC = 200                      # Gaussian MC draws per (p, d) cell


def fourier_power_fractions(W: np.ndarray, p: int) -> np.ndarray:
    """Non-DC Fourier power shares F_k, k = 1..(p−1)//2, of the numeric-token
    embedding rows (tokens 0..p−1), summed over embedding dimensions."""
    X = W[:p, :].astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)      # drop DC explicitly
    C = np.fft.rfft(X, axis=0)                 # (p//2+1, d) complex
    power = (np.abs(C[1:(p - 1) // 2 + 1, :]) ** 2).sum(axis=1)
    total = power.sum()
    return power / total if total > 0 else power


def _model_init_embedding(p: int, d: int, seed: int) -> np.ndarray:
    import torch
    from grokking_capacity.models import build_model
    np.random.seed(seed)
    torch.manual_seed(seed)
    m = build_model(depth=2, dim=d, heads=1, p=p, dropout=0.2)
    return m.embedding.weight.detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# (a) max-overlap null scaling
# --------------------------------------------------------------------------- #

def run_null(out: Path, *, primes=(97, 113, 139, 197),
             dims=(32, 64, 128, 256), seeds=tuple(range(42, 52))) -> dict:
    rng = np.random.default_rng(0)
    cells = []
    for p in primes:
        n_freq = (p - 1) // 2
        uniform = 1.0 / n_freq
        for d in dims:
            fmax_real = []
            argmax_real = []
            for s in seeds:
                F = fourier_power_fractions(_model_init_embedding(p, d, s), p)
                fmax_real.append(float(F.max()))
                argmax_real.append(int(F.argmax()) + 1)
            fmax_null = []
            for _ in range(NULL_MC):
                W = rng.standard_normal((p, d))
                F = fourier_power_fractions(W, p)
                fmax_null.append(float(F.max()))
            cells.append({
                "p": p, "d": d, "n_freq": n_freq, "uniform_share": uniform,
                "fmax_real_mean": float(np.mean(fmax_real)),
                "fmax_real_sd": float(np.std(fmax_real, ddof=1)),
                "fmax_null_mean": float(np.mean(fmax_null)),
                "fmax_null_sd": float(np.std(fmax_null, ddof=1)),
                "enhancement_real": float(np.mean(fmax_real) / uniform),
                "enhancement_null": float(np.mean(fmax_null) / uniform),
                "ev_prediction_1p_sqrt": float(
                    1.0 + math.sqrt(2.0 * math.log(n_freq) / d)),
                "argmax_freqs": argmax_real,
            })
    res = {"cells": cells,
           "note": ("enhancement = E[F_max]/(1/n_freq); the extreme-value "
                    "null for chi^2_{2d} shares predicts ~1+sqrt(2 ln "
                    "n_freq / d); real-init vs Gaussian-null agreement "
                    "means the init carries no frequency structure beyond "
                    "extreme-value fluctuation")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    return res


# --------------------------------------------------------------------------- #
# (b) selection test
# --------------------------------------------------------------------------- #

def _load_embedding(path: Path) -> Optional[np.ndarray]:
    import torch
    if not path.exists():
        return None
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    for k in ("embedding.weight",):
        if k in sd:
            return sd[k].numpy()
    return None


def run_selection(db: str, out: Path, *, data_root: Optional[str] = None) -> dict:
    import sqlite3
    con = sqlite3.connect(os.path.expanduser(db))
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "select uuid, p, dim, seed, artefacts_dir from runs "
        "where experiment_type='groks' and status='completed'").fetchall()

    runs_out = []
    for r in rows:
        adir = Path(r["artefacts_dir"])
        if data_root is not None and not adir.exists():
            adir = Path(data_root) / "groks" / r["uuid"]
        W_fin = _load_embedding(adir / "model.pt")
        W_ini = _load_embedding(adir / "model_init.pt")
        if W_fin is None or W_ini is None:
            continue
        p = int(r["p"])
        n_freq = (p - 1) // 2
        F_fin = fourier_power_fractions(W_fin, p)
        F_ini = fourier_power_fractions(W_ini, p)
        selected = np.where(F_fin > SELECT_THRESHOLD_X_UNIFORM / n_freq)[0]
        if selected.size == 0 or selected.size == n_freq:
            continue
        non = np.setdiff1d(np.arange(n_freq), selected)
        init_sel = float(F_ini[selected].mean())
        init_non = float(F_ini[non].mean())
        # percentile rank of each selected freq's init power in the full
        # init distribution (0.5 expected under the dynamical-selection null)
        ranks = [float((F_ini < F_ini[k]).mean()) for k in selected]
        runs_out.append({
            "uuid": r["uuid"], "p": p, "dim": int(r["dim"]),
            "seed": int(r["seed"]),
            "n_selected": int(selected.size),
            "selected_freqs": (selected + 1).tolist(),
            "final_power_in_selected": float(F_fin[selected].sum()),
            "init_share_selected_mean": init_sel,
            "init_share_nonselected_mean": init_non,
            "log_ratio": math.log(init_sel / init_non) if init_non > 0 else None,
            "init_rank_of_selected": ranks,
        })

    ratios = [r["log_ratio"] for r in runs_out if r["log_ratio"] is not None]
    ranks_all = [x for r in runs_out for x in r["init_rank_of_selected"]]
    verdict: dict[str, Any] = {"n_runs": len(runs_out)}
    if len(ratios) >= 5:
        from scipy import stats as sps
        w = sps.wilcoxon(ratios)
        verdict.update({
            "median_log_ratio": float(np.median(ratios)),
            "wilcoxon_p": float(w.pvalue),
            "mean_init_rank_of_selected": float(np.mean(ranks_all)),
            "interpretation": (
                "selected frequencies were over-represented at init"
                if np.median(ratios) > 0 and w.pvalue < 0.05 else
                "no evidence that selection is initialisation-seeded"),
        })
    res = {"selection_threshold_x_uniform": SELECT_THRESHOLD_X_UNIFORM,
           "runs": runs_out, "verdict": verdict}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("null")
    n.add_argument("--out", default="results/theory/init_overlap_null.json")
    s = sub.add_parser("selection")
    s.add_argument("--db", required=True)
    s.add_argument("--data-root", default=None)
    s.add_argument("--out", default="results/theory/init_overlap_selection.json")
    args = ap.parse_args(argv)
    if args.cmd == "null":
        res = run_null(Path(args.out))
        for c in res["cells"]:
            print(f"p={c['p']} d={c['d']}: enh real {c['enhancement_real']:.2f} "
                  f"null {c['enhancement_null']:.2f} "
                  f"EV-pred {c['ev_prediction_1p_sqrt']:.2f}")
    else:
        res = run_selection(args.db, Path(args.out), data_root=args.data_root)
        print(json.dumps(res["verdict"], indent=2))


if __name__ == "__main__":
    main()
