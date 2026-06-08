"""`gc-error-decay` — late-stage training-error decay shape on random-label runs.

On runs trained on random labels (pure memorisation, no generalisation), let
``ε(E) = 1 − train_acc(E)``. A constant-hazard (mean-field) model of memorisation,
``dε/dE = −r(f)·ε``, predicts the late-stage error decays **exponentially**: ``log ε`` is
linear in the epoch ``E``. The competing accounts predict a **power-law** or a
**stretched-exponential** decay. This module fits all three on the late window of each
trajectory and reports which family wins.

Pure post-hoc analysis: it re-reads the per-epoch ``train_acc_trace`` already persisted in
each run's ``trace.npz`` (speed runs, always random-label; capacity runs with
``dataset_type='random'``). It never trains, loads a model, or writes to the registry.

Config-driven, mirroring ``gc-figures``: ``--config <yaml>`` (or ``--all``) scopes the runs
via :class:`ConfigView`.
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
from scipy.optimize import curve_fit

from ..consts import C as DEFAULT_C
from .config_view import ConfigView, load_npz


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO_ROOT / "configs"
DEFAULT_OUT_ROOT = REPO_ROOT / "results"

# Window / validity thresholds. Fixed here, before looking at any aggregates.
MID_ACC = 0.5            # window starts at the first epoch with acc >= this
MEM_ACC = 0.99           # window ends at the memorisation epoch (first acc >= this)
MIN_WINDOW_POINTS = 15   # fewer than this -> under-resolved, excluded from headline stats

# Gate thresholds for the plain-language verdict.
EXP_R2_MEDIAN = 0.90
EXP_MAJORITY = 0.60


# --------------------------------------------------------------------------- #
# Run selection
# --------------------------------------------------------------------------- #
def collect_random_runs(view: ConfigView) -> list[dict[str, Any]]:
    """All completed random-label runs in the view: every speed run plus the
    capacity runs trained on random targets."""
    runs: list[dict[str, Any]] = []
    for group in view.iter_groups():
        runs.extend(group.speed_runs)
        runs.extend(r for r in group.capacity_runs if r.get("dataset_type") == "random")
    return runs


# --------------------------------------------------------------------------- #
# Window + fits
# --------------------------------------------------------------------------- #
def _error_window(train_acc: np.ndarray, n_samples: int):
    """Return ``(E_idx, eps)`` over the late memorisation window, or ``None`` when the
    window holds fewer than ``MIN_WINDOW_POINTS`` epochs.

    ``E_idx`` is the (0-based) epoch index; ``eps = 1 − acc`` floored at ``1/n_samples``
    so ``log ε`` is finite. The window runs from the first epoch with acc >= ``MID_ACC``
    up to (and excluding) the memorisation epoch — the first epoch with acc >= ``MEM_ACC``
    — recomputed from the trace so the threshold stays adjustable for downstream work.
    """
    acc = np.asarray(train_acc, dtype=float)
    if acc.size == 0:
        return None
    if acc.max() > 1.5:           # speed traces are in %, capacity in fraction
        acc = acc / 100.0

    start = next((i for i, a in enumerate(acc) if a >= MID_ACC), None)
    if start is None:
        return None
    mem = next((i for i, a in enumerate(acc) if a >= MEM_ACC), None)
    end = mem if mem is not None else acc.size   # exclusive
    if end - start < MIN_WINDOW_POINTS:
        return None

    floor = 1.0 / max(int(n_samples), 1)
    E = np.arange(start, end, dtype=float)
    eps = np.clip(1.0 - acc[start:end], floor, None)
    return E, eps


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_exponential(E: np.ndarray, eps: np.ndarray):
    """Linear fit of ``log ε`` on ``E``. Returns ``(decay_rate, r2)`` with
    ``decay_rate = −slope`` (the per-epoch hazard ``r``)."""
    y = np.log(eps)
    A = np.vstack([E, np.ones_like(E)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ np.array([slope, intercept])
    return float(-slope), _r2(y, yhat)


def fit_power_law(E: np.ndarray, eps: np.ndarray) -> float:
    """Linear fit of ``log ε`` on ``log E`` (E shifted to start at 1). Returns ``r2``."""
    x = np.log(E - E[0] + 1.0)
    y = np.log(eps)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return _r2(y, A @ coef)


def fit_stretched(E: np.ndarray, eps: np.ndarray) -> float:
    """Stretched-exponential ``log ε = log ε₀ − (ΔE/τ)^β`` (τ, β > 0). Returns ``r2`` in
    log space, or NaN if the fit fails."""
    x = E - E[0]
    y = np.log(eps)

    def model(xx, log_eps0, tau, beta):
        return log_eps0 - (xx / tau) ** beta

    try:
        span = max(x[-1], 1.0)
        (popt, _) = curve_fit(
            model, x, y,
            p0=[y[0], span, 1.0],
            bounds=([-np.inf, 1e-6, 1e-3], [np.inf, np.inf, 10.0]),
            maxfev=20000,
        )
        return _r2(y, model(x, *popt))
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
# Per-run analysis
# --------------------------------------------------------------------------- #
def _capacity_fraction(row: dict[str, Any]) -> Optional[float]:
    """``f = dataset_bits / (C·P)``; fall back to ``n·log2(p+2)/(C·P)`` when the row
    lacks ``dataset_bits`` (capacity rows don't store it)."""
    pc = row.get("param_count")
    if not pc:
        return None
    bits = row.get("dataset_bits")
    if bits is None:
        n, p = row.get("n_samples"), row.get("p")
        if n is None or p is None:
            return None
        bits = float(n) * float(np.log2(int(p) + 2))
    return float(bits) / (DEFAULT_C * float(pc))


def analyze_run(row: dict[str, Any], npz) -> Optional[dict[str, Any]]:
    """Per-run record, or ``None`` if the run is under-resolved (window too short or no
    ``train_acc_trace``)."""
    if "train_acc_trace" not in npz.files:
        return None
    n_samples = int(row.get("n_samples") or 0)
    win = _error_window(npz["train_acc_trace"], n_samples)
    if win is None:
        return None
    E, eps = win
    decay_rate, r2_exp = fit_exponential(E, eps)
    return {
        "uuid": row.get("uuid"),
        "experiment_type": row.get("experiment_type"),
        "p": row.get("p"),
        "dim": row.get("dim"),
        "depth": row.get("depth"),
        "heads": row.get("heads"),
        "n_samples": n_samples,
        "param_count": row.get("param_count"),
        "f": _capacity_fraction(row),
        "n_window": int(E.size),
        "E_start": int(E[0]),
        "decay_rate": decay_rate,
        "r2_exp": r2_exp,
        "r2_power": fit_power_law(E, eps),
        "r2_stretched": fit_stretched(E, eps),
        # kept for plotting; not serialised
        "_E": E,
        "_eps": eps,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_error_decay(view: ConfigView, out_dir: Path) -> dict[str, Any]:
    """Fit every random-label run, write JSON + two figures, return the summary."""
    out_dir.mkdir(parents=True, exist_ok=True)

    valid: list[dict[str, Any]] = []
    under_resolved: list[dict[str, Any]] = []
    for row in collect_random_runs(view):
        npz = load_npz(row)
        try:
            rec = analyze_run(row, npz)
        finally:
            npz.close()
        if rec is None:
            under_resolved.append({k: row.get(k) for k in
                                   ("uuid", "experiment_type", "p", "dim", "n_samples")})
        else:
            valid.append(rec)

    summary = _summarise(valid, under_resolved)

    _write_json(valid, under_resolved, summary, view.config_name, out_dir)
    _plot_trajectories(valid, out_dir / "error_decay_trajectories.pdf")
    _plot_r2_hist(valid, out_dir / "error_decay_r2_hist.pdf")
    return summary


def _summarise(valid: list[dict], under_resolved: list[dict]) -> dict[str, Any]:
    if not valid:
        return {"verdict": "no_data", "n_valid": 0,
                "n_under_resolved": len(under_resolved)}
    r2 = np.array([r["r2_exp"] for r in valid], dtype=float)
    rate = np.array([r["decay_rate"] for r in valid], dtype=float)
    exp_beats_power = np.array(
        [r["r2_exp"] > r["r2_power"] for r in valid], dtype=bool
    )
    median_r2 = float(np.median(r2))
    majority = float(exp_beats_power.mean())
    verdict = ("exponential"
               if median_r2 > EXP_R2_MEDIAN and majority > EXP_MAJORITY
               else "non_exponential")
    return {
        "verdict": verdict,
        "n_valid": len(valid),
        "n_under_resolved": len(under_resolved),
        "median_r2_exp": median_r2,
        "iqr_r2_exp": [float(np.percentile(r2, 25)), float(np.percentile(r2, 75))],
        "median_decay_rate": float(np.median(rate)),
        "frac_exp_beats_power": majority,
    }


def _write_json(valid, under_resolved, summary, config_name, out_dir: Path) -> None:
    payload = {
        "config_name": config_name,
        "summary": summary,
        "runs": [{k: v for k, v in r.items() if not k.startswith("_")} for r in valid],
        "under_resolved": under_resolved,
    }
    with open(out_dir / "error_decay.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _plot_trajectories(valid: list[dict], path: Path) -> None:
    if not valid:
        return
    primes = sorted({r["p"] for r in valid if r["p"] is not None})
    palette = sns.color_palette("crest", n_colors=max(len(primes), 1))
    colour = {p: palette[i] for i, p in enumerate(primes)}

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for r in valid:
        E, eps = r["_E"], r["_eps"]
        dt = E - E[0]
        c = colour.get(r["p"], "0.5")
        ax.plot(dt, eps / eps[0], color=c, alpha=0.35, lw=1)
        # exponential fit overlay
        yhat = np.exp(np.log(eps[0]) - r["decay_rate"] * dt)
        ax.plot(dt, yhat / eps[0], color=c, alpha=0.9, lw=1, ls="--")

    ax.set_yscale("log")
    ax.set_xlabel(r"epochs since acc $\geq 0.5$  ($E - E_{\rm start}$)")
    ax.set_ylabel(r"$\varepsilon(E) / \varepsilon_{\rm start}$")
    ax.set_title("Late-stage training-error decay (random labels)")
    ax.grid(True, alpha=0.3, which="both")
    handles = [plt.Line2D([], [], color=colour[p], label=f"p={p}") for p in primes]
    if handles:
        ax.legend(handles=handles, fontsize=8, title="prime")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()


def _plot_r2_hist(valid: list[dict], path: Path) -> None:
    if not valid:
        return
    r2 = [r["r2_exp"] for r in valid]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(r2, bins=20, range=(min(0.0, min(r2)), 1.0),
            color=sns.color_palette("crest")[3], edgecolor="white")
    for x, ls in ((0.90, "--"), (0.95, ":")):
        ax.axvline(x, color="0.3", ls=ls, lw=1, label=f"$R^2$={x:g}")
    ax.set_xlabel(r"per-trajectory exponential-fit $R^2$")
    ax.set_ylabel("count")
    ax.set_title("Exponential-fit quality across trajectories")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _render_one(config_path: Path, out_dir: Path, db_path: Optional[str]) -> None:
    print(f"[gc-error-decay] {config_path.name} → {out_dir}")
    view = ConfigView.from_yaml(config_path, db_path=db_path)
    if not view.groups:
        print("  (no runs found in this config — skipping)")
        return
    summary = run_error_decay(view, out_dir)
    print(f"  verdict: {summary['verdict']}  "
          f"(n_valid={summary['n_valid']}, "
          f"n_under_resolved={summary['n_under_resolved']})")
    if summary.get("n_valid"):
        print(f"  median R²_exp={summary['median_r2_exp']:.3f}  "
              f"frac(exp>power)={summary['frac_exp_beats_power']:.2f}  "
              f"median rate={summary['median_decay_rate']:.4g}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="gc-error-decay",
        description="Fit late-stage training-error decay (exp vs power-law vs stretched) "
                    "on random-label runs for a grokking_capacity YAML config.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=Path, help="Path to a single configs/*.yaml")
    src.add_argument("--all", action="store_true", help="Iterate every yaml in configs/")
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory (default: results/<config_name>/error_decay/)")
    p.add_argument("--db", default=None,
                   help="Path to runs.db (default: GC_WALLOW_DB env or repo runs.db)")
    args = p.parse_args(argv)

    if args.all:
        if args.out:
            print("[gc-error-decay] --out is ignored with --all", file=sys.stderr)
        for cfg in sorted(CONFIGS_DIR.glob("*.yaml")):
            _render_one(cfg, DEFAULT_OUT_ROOT / cfg.stem / "error_decay", args.db)
        return 0

    cfg = args.config
    out = args.out or (DEFAULT_OUT_ROOT / cfg.stem / "error_decay")
    _render_one(cfg, out, args.db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
