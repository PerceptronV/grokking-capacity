"""`gc-threshold-invariance` — does the accuracy threshold enter T_mem only through b(τ)?

The mean-field hazard model of memorisation predicts
``T_mem(f, τ) ≈ b(τ)·e^{a·f}`` with ``b(τ) = (1/r0)·log(ε₀/(1-τ))`` and ``ε₀ ≈ 1``. Two
sharp, training-free consequences fall out of re-thresholding the existing random-label
``train_acc(E)`` curves at several ``τ``:

  (a) **exponent invariance** — fitting ``log T_mem(f) = log b(τ) + a·f`` separately at each
      ``τ`` must give a ``τ``-independent slope ``a`` (overlapping CIs);
  (b) **prefactor law** — at fixed ``f``, ``T_mem(τ) ∝ -log(1-τ)`` (linear, ~zero intercept).

Pure post-hoc: recomputes ``T_mem(τ)`` from each run's stored ``train_acc_trace`` (speed
runs and random-target capacity runs) — no new training. Config-driven like ``gc-figures``.
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
from .error_decay import collect_random_runs, _capacity_fraction


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO_ROOT / "configs"
DEFAULT_OUT_ROOT = REPO_ROOT / "results"

THRESHOLDS: tuple[float, ...] = (0.90, 0.95, 0.99, 0.999)
MIN_RUNS = 3              # per-τ fit needs at least this many runs
INTERCEPT_TOL = 0.30     # |intercept| / (slope·mean_x) below this -> "near-zero intercept"
GOOD_R2 = 0.70           # prefactor regression must clear this (median over runs)


def _norm_acc(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    return a / 100.0 if a.size and a.max() > 1.5 else a


def _t_mem(acc: np.ndarray, tau: float) -> Optional[int]:
    """1-indexed epoch where acc first reaches τ, else None (never reached)."""
    above = np.where(acc >= tau)[0]
    return int(above[0]) + 1 if above.size else None


def _ols(x: np.ndarray, y: np.ndarray):
    """OLS with the analytic slope standard error. Returns
    ``(slope, intercept, se_slope, r2)`` (NaNs if under-determined)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    if n < 3 or np.allclose(x, x[0]):
        return (float("nan"),) * 4
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ np.array([slope, intercept])
    resid = y - yhat
    dof = n - 2
    sxx = float(np.sum((x - x.mean()) ** 2))
    sigma2 = float(np.sum(resid ** 2)) / dof if dof > 0 else float("nan")
    se = float(np.sqrt(sigma2 / sxx)) if sxx > 0 and np.isfinite(sigma2) else float("nan")
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), se, r2


# --------------------------------------------------------------------------- #
# Collect per-run T_mem(τ) and f
# --------------------------------------------------------------------------- #
def collect(view: ConfigView) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for row in collect_random_runs(view):
        npz = load_npz(row)
        try:
            if "train_acc_trace" not in npz.files:
                continue
            acc = _norm_acc(npz["train_acc_trace"])
        finally:
            npz.close()
        f = _capacity_fraction(row)
        if f is None or f <= 0:
            continue
        tmem = {tau: _t_mem(acc, tau) for tau in THRESHOLDS}
        if any(v is None for v in tmem.values()):    # run must reach the strictest τ
            continue
        runs.append({"uuid": row.get("uuid"), "p": row.get("p"), "dim": row.get("dim"),
                     "f": float(f), "tmem": {tau: int(tmem[tau]) for tau in THRESHOLDS}})
    return runs


def analyse(view: ConfigView) -> dict[str, Any]:
    runs = collect(view)
    out: dict[str, Any] = {"config_name": view.config_name, "n_runs": len(runs),
                           "thresholds": list(THRESHOLDS)}
    if len(runs) < MIN_RUNS:
        out["verdict"] = "no_data"
        out["_runs"] = runs
        return out

    f = np.array([r["f"] for r in runs])

    # (a) exponent invariance: a(τ) from log T_mem(f) = log b(τ) + a·f per threshold.
    per_tau = {}
    for tau in THRESHOLDS:
        t = np.array([r["tmem"][tau] for r in runs], float)
        slope, intercept, se, r2 = _ols(f, np.log(t))
        ci = [slope - 1.96 * se, slope + 1.96 * se] if np.isfinite(se) else [float("nan")] * 2
        per_tau[tau] = {"a": slope, "log_b": intercept, "se_a": se, "r2": r2, "ci95": ci}

    cis = [per_tau[tau]["ci95"] for tau in THRESHOLDS
           if all(np.isfinite(per_tau[tau]["ci95"]))]
    overlap = bool(cis) and (max(c[0] for c in cis) <= min(c[1] for c in cis))

    # (b) prefactor law: per run (fixed f) regress T_mem(τ) on x = -log(1-τ).
    x_tau = np.array([-np.log(1.0 - tau) for tau in THRESHOLDS])
    slopes, intercepts, r2s, ratios = [], [], [], []
    for r in runs:
        y = np.array([r["tmem"][tau] for tau in THRESHOLDS], float)
        s, b, _, r2 = _ols(x_tau, y)
        if not np.isfinite(s):
            continue
        slopes.append(s)
        intercepts.append(b)
        r2s.append(r2)
        denom = s * float(x_tau.mean())
        ratios.append(abs(b) / abs(denom) if denom else float("inf"))

    med_intercept_ratio = float(np.median(ratios)) if ratios else float("nan")
    med_r2_pref = float(np.median(r2s)) if r2s else float("nan")
    near_zero = np.isfinite(med_intercept_ratio) and med_intercept_ratio < INTERCEPT_TOL
    linear = np.isfinite(med_r2_pref) and med_r2_pref >= GOOD_R2

    verdict = "invariant" if (overlap and near_zero and linear) else "not_invariant"

    out.update({
        "verdict": verdict,
        "exponent_invariance": {
            "per_threshold": {str(tau): per_tau[tau] for tau in THRESHOLDS},
            "cis_overlap": overlap,
        },
        "prefactor_law": {
            "median_slope": float(np.median(slopes)) if slopes else float("nan"),
            "median_intercept": float(np.median(intercepts)) if intercepts else float("nan"),
            "median_intercept_ratio": med_intercept_ratio,
            "median_r2": med_r2_pref,
            "near_zero_intercept": bool(near_zero),
        },
        "routing": (
            "INVARIANT — hazard+interference functional form supported; "
            "go to gc-interference (optional) then the T_gen diagnosis."
            if verdict == "invariant" else
            "NOT INVARIANT — hazard functional form falsified; e^{af} survives as an "
            "empirical fit only. Proceed to the T_gen diagnosis regardless."),
        "_runs": runs, "_per_tau": per_tau, "_x_tau": x_tau,
    })
    return out


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def run_threshold_invariance(view: ConfigView, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    res = analyse(view)
    _write_json(res, out_dir)
    _plot(res, out_dir / "threshold_invariance.pdf")
    return res


def _write_json(res: dict[str, Any], out_dir: Path) -> None:
    payload = {k: v for k, v in res.items() if not k.startswith("_")}
    payload["runs"] = [
        {"uuid": r["uuid"], "p": r["p"], "dim": r["dim"], "f": r["f"],
         "tmem": {str(k): v for k, v in r["tmem"].items()}}
        for r in res.get("_runs", [])
    ]
    with open(out_dir / "threshold_invariance.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _plot(res: dict[str, Any], path: Path) -> None:
    runs = res.get("_runs", [])
    if len(runs) < MIN_RUNS or "_per_tau" not in res:
        return
    per_tau = res["_per_tau"]
    x_tau = res["_x_tau"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    # (a) a(τ) with 95% CI — flat line expected.
    taus = list(THRESHOLDS)
    a = [per_tau[t]["a"] for t in taus]
    err = [1.96 * per_tau[t]["se_a"] if np.isfinite(per_tau[t]["se_a"]) else 0.0
           for t in taus]
    ax1.errorbar(range(len(taus)), a, yerr=err, fmt="o", capsize=4,
                 color=sns.color_palette("crest")[2])
    ax1.set_xticks(range(len(taus)))
    ax1.set_xticklabels([f"{t:g}" for t in taus])
    ax1.set_xlabel(r"threshold $\tau$")
    ax1.set_ylabel(r"exponent $a(\tau)$")
    ax1.set_title(f"Exponent invariance (CIs overlap: "
                  f"{res['exponent_invariance']['cis_overlap']})")
    ax1.grid(True, alpha=0.3)

    # (b) T_mem(τ) vs -log(1-τ) — linear through ~origin expected.
    primes = sorted({r["p"] for r in runs if r["p"] is not None})
    palette = sns.color_palette("flare", n_colors=max(len(primes), 1))
    colour = {p: palette[i] for i, p in enumerate(primes)}
    for r in runs:
        y = [r["tmem"][t] for t in taus]
        ax2.plot(x_tau, y, color=colour.get(r["p"], "0.5"), alpha=0.3, lw=1, marker=".")
    ax2.set_xlabel(r"$-\log(1-\tau)$")
    ax2.set_ylabel(r"$T_{\rm mem}(\tau)$  (epochs)")
    ax2.set_title(f"Prefactor law (median $R^2$="
                  f"{res['prefactor_law']['median_r2']:.2f})")
    ax2.grid(True, alpha=0.3)
    handles = [plt.Line2D([], [], color=colour[p], label=f"p={p}") for p in primes]
    if handles:
        ax2.legend(handles=handles, fontsize=7, title="prime")

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _render_one(config_path: Path, out_dir: Path, db_path: Optional[str]) -> None:
    print(f"[gc-threshold-invariance] {config_path.name} → {out_dir}")
    view = ConfigView.from_yaml(config_path, db_path=db_path)
    if not view.groups:
        print("  (no runs found in this config — skipping)")
        return
    res = run_threshold_invariance(view, out_dir)
    print(f"  verdict: {res['verdict']}  (n_runs={res['n_runs']})")
    if "prefactor_law" in res:
        ei = res["exponent_invariance"]
        pf = res["prefactor_law"]
        print(f"  exponent CIs overlap={ei['cis_overlap']}  "
              f"prefactor median R²={pf['median_r2']:.2f}  "
              f"intercept ratio={pf['median_intercept_ratio']:.2f}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="gc-threshold-invariance",
        description="Test whether the accuracy threshold enters T_mem only through b(τ): "
                    "exponent invariance + the -log(1-τ) prefactor law, off existing "
                    "random-label train_acc traces.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=Path, help="Path to a single configs/*.yaml")
    src.add_argument("--all", action="store_true", help="Iterate every yaml in configs/")
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir (default: results/<config_name>/threshold_invariance/)")
    p.add_argument("--db", default=None,
                   help="Path to runs.db (default: GC_WALLOW_DB env or repo runs.db)")
    args = p.parse_args(argv)

    if args.all:
        if args.out:
            print("[gc-threshold-invariance] --out is ignored with --all", file=sys.stderr)
        for cfg in sorted(CONFIGS_DIR.glob("*.yaml")):
            _render_one(cfg, DEFAULT_OUT_ROOT / cfg.stem / "threshold_invariance", args.db)
        return 0

    cfg = args.config
    out = args.out or (DEFAULT_OUT_ROOT / cfg.stem / "threshold_invariance")
    _render_one(cfg, out, args.db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
