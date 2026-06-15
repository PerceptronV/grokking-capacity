"""`gc-kappa-stability` — is the contraction rate κ a stable architecture constant?

Stage 2 of the norm-contraction synthesis. The closed-form delay law calibrates one
κ on a cell and transfers it; this asks whether κ is stable enough to be that single
constant. Reuses ``gc-contraction``'s per-trajectory κ (and the Kosson κ_kos) and
decomposes its variance:

  - **within-cell** CV — seed-to-seed spread at fixed ``(p, dim, λ, η)``;
  - **across-cell** drift — whether per-cell median κ trends systematically with the
    parameter count ``P`` (or the prime ``p``) at fixed ``λ, η``.

STABLE ⟹ pin ``κ̂`` = pooled median. STRUCTURED ⟹ carry ``κ̂(P)``. UNSTABLE ⟹ fall
back to the Kosson rate ``κ_kos`` (tighter because it removes the AdamW-asymptote
fitting bias), or STOP if even that is noise.

Pure post-hoc: reuses ``gc-contraction``'s window/fits. Config-driven like ``gc-figures``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .config_view import ConfigView, load_npz
from .contraction import analyze_run, collect_groks_runs


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO_ROOT / "configs"
DEFAULT_OUT_ROOT = REPO_ROOT / "results"

MIN_SEEDS = 3            # cells with fewer valid seeds are dropped
STABLE_CV = 0.20        # within-cell median CV at/under this ⇒ "tight"
KOSSON_STABLE_CV = 0.15  # κ_kos within-cell CV for the UNSTABLE→Kosson rescue
ALPHA = 0.05            # κ–logP slope permutation-test significance
N_PERMUTATIONS = 10_000


def _cv(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    m = float(np.mean(x))
    return float(np.std(x) / m) if m else float("nan")


def _linfit(x: np.ndarray, y: np.ndarray):
    """OLS ``y = slope·x + intercept``. Returns ``(slope, intercept, r2)``."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ np.array([slope, intercept])
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def _perm_pvalue(x: np.ndarray, y: np.ndarray, slope_obs: float) -> float:
    rng = np.random.default_rng(0)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 3:
        return float("nan")
    count = 0
    for _ in range(N_PERMUTATIONS):
        s, _, _ = _linfit(x, rng.permutation(y))
        if abs(s) >= abs(slope_obs):
            count += 1
    return (count + 1) / (N_PERMUTATIONS + 1)


# --------------------------------------------------------------------------- #
# Collect Stage-1 records and group into cells
# --------------------------------------------------------------------------- #
def _records(view: ConfigView) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    for row in collect_groks_runs(view):
        npz = load_npz(row)
        try:
            rec = analyze_run(row, npz)
        finally:
            npz.close()
        if rec is not None and np.isfinite(rec["kappa"]):
            recs.append(rec)
    return recs


def _cells(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group records into ``(p, dim, λ, η)`` cells with ≥ MIN_SEEDS valid seeds."""
    groups: dict[tuple, list[dict]] = {}
    for r in recs:
        key = (r["p"], r["dim"], r["weight_decay"], r["lr"])
        groups.setdefault(key, []).append(r)

    cells: list[dict[str, Any]] = []
    for (p, dim, wd, lr), rs in sorted(groups.items(), key=lambda kv: kv[0]):
        if len(rs) < MIN_SEEDS:
            continue
        kappa = np.array([r["kappa"] for r in rs], float)
        kkos = np.array([r["kappa_kosson"] for r in rs
                         if np.isfinite(r["kappa_kosson"])], float)
        pc = rs[0]["param_count"]
        cells.append({
            "p": p, "dim": dim, "weight_decay": wd, "lr": lr,
            "param_count": pc, "n_seeds": len(rs),
            "median_kappa": float(np.median(kappa)), "cv_kappa": _cv(kappa),
            "median_kappa_kosson": float(np.median(kkos)) if kkos.size else float("nan"),
            "cv_kappa_kosson": _cv(kkos) if kkos.size else float("nan"),
            "f_window": (float(np.median(kappa)) / float(np.median(kkos))
                         if kkos.size and np.median(kkos) else float("nan")),
        })
    return cells


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def analyse(view: ConfigView) -> dict[str, Any]:
    recs = _records(view)
    cells = _cells(recs)
    out: dict[str, Any] = {"config_name": view.config_name,
                           "n_records": len(recs), "n_cells": len(cells)}
    if len(cells) < 2:
        out["verdict"] = "no_data"
        out["note"] = (f"{len(cells)} cell(s) with ≥{MIN_SEEDS} seeds — need ≥2 to "
                       "assess across-cell drift. Add seeds or dims.")
        out["_cells"] = cells
        return out

    within_cv = float(np.nanmedian([c["cv_kappa"] for c in cells]))
    within_cv_kos = float(np.nanmedian([c["cv_kappa_kosson"] for c in cells]))
    all_kappa = np.array([r["kappa"] for r in recs], float)
    pooled_median = float(np.median(all_kappa))
    pooled_cv = _cv(all_kappa)

    # κ vs log P (at fixed λ, η) and κ vs p.
    logP = np.log(np.array([c["param_count"] for c in cells], float))
    medk = np.array([c["median_kappa"] for c in cells], float)
    sl_P, _, r2_P = _linfit(logP, medk)
    p_P = _perm_pvalue(logP, medk, sl_P)

    primes = np.array([c["p"] for c in cells], float)
    sl_p, _, r2_p = _linfit(primes, medk)
    p_p = _perm_pvalue(primes, medk, sl_p) if len(set(primes.tolist())) > 1 else float("nan")

    drift_significant = np.isfinite(p_P) and p_P < ALPHA

    if within_cv <= STABLE_CV and not drift_significant:
        verdict, rate = "stable", "kappa"
        kappa_hat: Any = pooled_median
    elif within_cv <= STABLE_CV and drift_significant:
        verdict, rate = "structured", "kappa(P)"
        kappa_hat = {"slope_logP": sl_P, "at_pooled_median": pooled_median}
    else:
        # UNSTABLE: try the Kosson rate.
        if np.isfinite(within_cv_kos) and within_cv_kos <= KOSSON_STABLE_CV:
            verdict, rate = "unstable_kosson_rescue", "kappa_kosson"
            kappa_hat = float(np.nanmedian([c["median_kappa_kosson"] for c in cells]))
        else:
            verdict, rate = "unstable", None
            kappa_hat = None

    routing = {
        "stable": f"STABLE — pin κ̂={pooled_median:.3f} (pooled median). Go to gc-bridge.",
        "structured": "STRUCTURED — carry κ̂(P); expect structured Stage-4 residuals. "
                      "Go to gc-bridge.",
        "unstable_kosson_rescue": "UNSTABLE on κ_LL but κ_kos is tight — adopt κ_kos as "
                                  "the rate. Go to gc-bridge.",
        "unstable": "UNSTABLE — neither κ_LL nor κ_kos is a usable constant; the delay "
                    "law cannot be calibrated. Report this bound (STOP).",
    }[verdict]

    out.update({
        "verdict": verdict,
        "rate_used": rate,
        "kappa_hat": kappa_hat,
        "within_cell_median_cv": within_cv,
        "within_cell_median_cv_kosson": within_cv_kos,
        "pooled_median_kappa": pooled_median,
        "pooled_cv_kappa": pooled_cv,
        "kappa_vs_logP": {"slope": sl_P, "r2": r2_P, "perm_pvalue": p_P,
                          "significant": bool(drift_significant)},
        "kappa_vs_p": {"slope": sl_p, "r2": r2_p, "perm_pvalue": p_p},
        "median_f_window": float(np.nanmedian([c["f_window"] for c in cells])),
        "routing": routing,
        "_cells": cells,
    })
    return out


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def run_kappa_stability(view: ConfigView, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    res = analyse(view)
    payload = {k: v for k, v in res.items() if not k.startswith("_")}
    payload["cells"] = res.get("_cells", [])
    with open(out_dir / "kappa_stability.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    _plot(res, out_dir / "kappa_stability.pdf")
    return res


def _plot(res: dict[str, Any], path: Path) -> None:
    cells = res.get("_cells", [])
    if len(cells) < 2:
        return
    primes = sorted({c["p"] for c in cells})
    palette = sns.color_palette("crest", n_colors=max(len(primes), 1))
    colour = {p: palette[i] for i, p in enumerate(primes)}

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for c in cells:
        pc = c["param_count"]
        cl = colour.get(c["p"], "0.5")
        kap, cv = c["median_kappa"], c["cv_kappa"]
        ax.errorbar(pc, kap, yerr=(abs(kap) * cv if np.isfinite(cv) else 0.0),
                    fmt="o", color=cl, capsize=3, ms=6)
        if np.isfinite(c["median_kappa_kosson"]):
            ax.scatter(pc, c["median_kappa_kosson"], marker="x", color=cl, alpha=0.7)
    ax.set_xscale("log")
    ax.axhline(res.get("pooled_median_kappa", np.nan), color="0.4", ls="--", lw=1,
               label=f"pooled median κ={res.get('pooled_median_kappa', float('nan')):.3f}")
    ax.set_xlabel("parameter count $P$")
    ax.set_ylabel(r"$\kappa$ (● $\kappa_{LL}$, ✕ $\kappa_{kos}$)")
    ax.set_title(f"κ stability — {res.get('verdict', '')} "
                 f"(within-cell CV={res.get('within_cell_median_cv', float('nan')):.2f})")
    handles = [plt.Line2D([], [], marker="o", ls="", color=colour[p], label=f"p={p}")
               for p in primes]
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0], fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _render_one(config_path: Path, out_dir: Path, db_path: Optional[str]) -> None:
    print(f"[gc-kappa-stability] {config_path.name} → {out_dir}")
    view = ConfigView.from_yaml(config_path, db_path=db_path)
    if not view.groups:
        print("  (no runs found in this config — skipping)")
        return
    res = run_kappa_stability(view, out_dir)
    print(f"  verdict: {res['verdict']}  "
          f"(n_cells={res['n_cells']}, n_records={res['n_records']})")
    if res.get("note"):
        print(f"  note: {res['note']}")
    if "within_cell_median_cv" in res:
        kp = res["kappa_vs_logP"]
        print(f"  within-cell CV={res['within_cell_median_cv']:.2f}  "
              f"pooled κ={res['pooled_median_kappa']:.3f}  "
              f"κ–logP slope={kp['slope']:.3g} (p={kp['perm_pvalue']:.3f})")
        print(f"  → {res['routing']}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="gc-kappa-stability",
        description="Variance-decompose the contraction rate κ across seeds and cells "
                    "to test whether it is a stable architecture-level constant.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=Path, help="Path to a single configs/*.yaml")
    src.add_argument("--all", action="store_true", help="Iterate every yaml in configs/")
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir (default: results/<config_name>/kappa_stability/)")
    p.add_argument("--db", default=None,
                   help="Path to runs.db (default: GC_WALLOW_DB env or repo runs.db)")
    args = p.parse_args(argv)

    if args.all:
        if args.out:
            print("[gc-kappa-stability] --out is ignored with --all", file=sys.stderr)
        for cfg in sorted(CONFIGS_DIR.glob("*.yaml")):
            _render_one(cfg, DEFAULT_OUT_ROOT / cfg.stem / "kappa_stability", args.db)
        return 0

    cfg = args.config
    out = args.out or (DEFAULT_OUT_ROOT / cfg.stem / "kappa_stability")
    _render_one(cfg, out, args.db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
