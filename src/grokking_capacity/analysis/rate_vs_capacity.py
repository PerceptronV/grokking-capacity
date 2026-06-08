"""`gc-rate-vs-capacity` — does the memorisation hazard rate decay with capacity fraction?

The mean-field jamming account of ``T_mem`` models late memorisation as a constant-hazard
process whose per-epoch error-removal rate ``r(f)`` falls exponentially in the capacity
fraction ``f = K/(C·P) = dataset_bits/(C·P)``:

    log r(f) = log r0 - a·f.

The exponent ``a`` recovered from the **dynamics** (per-trajectory rates, fit by
``gc-error-decay``) must match the exponent of the **aggregate** memorisation-time law
``T_mem(f) ∝ e^{a·f}`` (one ``(f, T_mem)`` point per run). They are opposite in sign by
construction (``r ∝ e^{-af}``, ``T_mem ∝ e^{+af}``); this script reports both ``a`` values,
their fit quality, and their relative disagreement.

Pure post-hoc: it reuses ``gc-error-decay``'s window and exponential fit for the rates
(so the rates here ARE those rates), and recomputes ``T_mem(0.99)`` from the same
``train_acc_trace`` for the aggregate fit. Never trains. Config-driven like ``gc-figures``.
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
from .error_decay import analyze_run, collect_random_runs, _capacity_fraction


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO_ROOT / "configs"
DEFAULT_OUT_ROOT = REPO_ROOT / "results"

MEM_ACC = 0.99            # T_mem := first epoch with train_acc >= this
CONSISTENT_TOL = 0.20     # |a_rate - a_agg| / a_agg below this -> CONSISTENT
GOOD_R2 = 0.50            # both fits must clear this for a confident verdict
N_PERMUTATIONS = 10_000


def _norm_acc(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    return a / 100.0 if a.size and a.max() > 1.5 else a


def _t_mem_epochs(train_acc: np.ndarray) -> Optional[int]:
    """1-indexed epoch where train_acc first reaches MEM_ACC, else None."""
    acc = _norm_acc(train_acc)
    above = np.where(acc >= MEM_ACC)[0]
    return int(above[0]) + 1 if above.size else None


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _linfit(x: np.ndarray, y: np.ndarray):
    """OLS ``y = slope·x + intercept``. Returns ``(slope, intercept, r2)``."""
    A = np.vstack([np.asarray(x, float), np.ones_like(x, dtype=float)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, np.asarray(y, float), rcond=None)
    return float(slope), float(intercept), _r2(np.asarray(y, float), A @ [slope, intercept])


def _perm_pvalue(x: np.ndarray, y: np.ndarray, slope_obs: float) -> float:
    """Two-sided permutation p-value for the slope (shuffle y against x)."""
    rng = np.random.default_rng(0)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    count = 0
    for _ in range(N_PERMUTATIONS):
        s, _, _ = _linfit(x, rng.permutation(y))
        if abs(s) >= abs(slope_obs):
            count += 1
    return (count + 1) / (N_PERMUTATIONS + 1)


# --------------------------------------------------------------------------- #
# Collect (f, r) and (f, T_mem) from the same random-label runs
# --------------------------------------------------------------------------- #
def collect(view: ConfigView):
    rate_pts: list[tuple[float, float, dict]] = []   # (f, r, meta)
    tmem_pts: list[tuple[float, float]] = []         # (f, T_mem epochs)
    for row in collect_random_runs(view):
        npz = load_npz(row)
        try:
            rec = analyze_run(row, npz)
            tm = (_t_mem_epochs(npz["train_acc_trace"])
                  if "train_acc_trace" in npz.files else None)
        finally:
            npz.close()
        f = _capacity_fraction(row)
        if rec is not None and rec["decay_rate"] > 0 and rec["f"] and rec["f"] > 0:
            rate_pts.append((rec["f"], rec["decay_rate"],
                             {"uuid": row.get("uuid"), "p": row.get("p"),
                              "dim": row.get("dim"), "n_samples": row.get("n_samples")}))
        if tm is not None and f is not None and f > 0:
            tmem_pts.append((f, float(tm)))
    return rate_pts, tmem_pts


def analyse(view: ConfigView) -> dict[str, Any]:
    rate_pts, tmem_pts = collect(view)
    out: dict[str, Any] = {
        "config_name": view.config_name,
        "n_rate_points": len(rate_pts),
        "n_tmem_points": len(tmem_pts),
    }
    if len(rate_pts) < 3 or len(tmem_pts) < 3:
        out["verdict"] = "no_data"
        out["_rate_pts"] = rate_pts
        out["_tmem_pts"] = tmem_pts
        return out

    f_r = np.array([p[0] for p in rate_pts])
    r = np.array([p[1] for p in rate_pts])
    slope_r, intercept_r, r2_rate = _linfit(f_r, np.log(r))   # log r = log r0 - a·f
    a_rate = -slope_r
    r0 = float(np.exp(intercept_r))
    p_rate = _perm_pvalue(f_r, np.log(r), slope_r)

    f_t = np.array([p[0] for p in tmem_pts])
    t = np.array([p[1] for p in tmem_pts])
    slope_t, intercept_t, r2_agg = _linfit(f_t, np.log(t))    # log T_mem = log b + a·f
    a_agg = slope_t

    rel = abs(a_rate - a_agg) / abs(a_agg) if a_agg else float("inf")
    if a_rate <= 0 or a_agg <= 0:
        verdict = "inconsistent"
    elif rel <= CONSISTENT_TOL and r2_rate >= GOOD_R2 and r2_agg >= GOOD_R2:
        verdict = "consistent"
    else:
        verdict = "inconsistent"

    out.update({
        "verdict": verdict,
        "a_rate": a_rate, "r0": r0, "r2_rate": r2_rate, "perm_pvalue_rate": p_rate,
        "a_aggregate": a_agg, "b_aggregate": float(np.exp(intercept_t)), "r2_aggregate": r2_agg,
        "relative_disagreement": rel,
        "routing": (
            "CONSISTENT — aggregate e^{af} law grounded in the per-trajectory rate; "
            "go to gc-threshold-invariance."
            if verdict == "consistent" else
            "INCONSISTENT — re-check the gc-error-decay late-window/ε₀; if still off, "
            "downgrade: 'e^{af} holds, hazard-rate derivation unvalidated'."),
        "_slope_rate": slope_r, "_intercept_rate": intercept_r,
        "_slope_t": slope_t, "_intercept_t": intercept_t,
        "_rate_pts": rate_pts, "_tmem_pts": tmem_pts,
    })
    return out


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def run_rate_vs_capacity(view: ConfigView, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    res = analyse(view)
    _write_json(res, out_dir)
    _plot(res, out_dir / "rate_vs_capacity.pdf")
    return res


def _write_json(res: dict[str, Any], out_dir: Path) -> None:
    rate_pts = res.get("_rate_pts", [])
    payload = {k: v for k, v in res.items() if not k.startswith("_")}
    payload["rate_runs"] = [
        {"f": f, "decay_rate": r, **meta} for f, r, meta in rate_pts
    ]
    with open(out_dir / "rate_vs_capacity.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _plot(res: dict[str, Any], path: Path) -> None:
    rate_pts = res.get("_rate_pts", [])
    if len(rate_pts) < 3 or "a_rate" not in res:
        return
    f = np.array([p[0] for p in rate_pts])
    r = np.array([p[1] for p in rate_pts])
    primes = sorted({p[2].get("p") for p in rate_pts if p[2].get("p") is not None})
    palette = sns.color_palette("flare", n_colors=max(len(primes), 1))
    colour = {p: palette[i] for i, p in enumerate(primes)}

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for fi, ri, meta in rate_pts:
        ax.scatter(fi, ri, color=colour.get(meta.get("p"), "0.5"), alpha=0.7, s=20)
    xs = np.linspace(float(f.min()), float(f.max()), 100)
    ax.plot(xs, np.exp(res["_intercept_rate"] + res["_slope_rate"] * xs),
            "k-", lw=1.6, label=fr"dynamical: $a_r$={res['a_rate']:.2f} ($R^2$={res['r2_rate']:.2f})")
    # aggregate-a slope, anchored at the rate fit's intercept for visual comparison
    ax.plot(xs, np.exp(res["_intercept_rate"] - res["a_aggregate"] * xs),
            color="0.4", ls="--", lw=1.4,
            label=fr"aggregate slope: $a$={res['a_aggregate']:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel(r"capacity fraction $f = K/(C\,P)$")
    ax.set_ylabel(r"error-removal rate $r(f)$  (per epoch)")
    ax.set_title("Hazard rate vs capacity fraction")
    ax.grid(True, alpha=0.3, which="both")
    handles = [plt.Line2D([], [], marker="o", ls="", color=colour[p], label=f"p={p}")
               for p in primes]
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0], fontsize=7)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _render_one(config_path: Path, out_dir: Path, db_path: Optional[str]) -> None:
    print(f"[gc-rate-vs-capacity] {config_path.name} → {out_dir}")
    view = ConfigView.from_yaml(config_path, db_path=db_path)
    if not view.groups:
        print("  (no runs found in this config — skipping)")
        return
    res = run_rate_vs_capacity(view, out_dir)
    print(f"  verdict: {res['verdict']}  "
          f"(n_rate={res['n_rate_points']}, n_tmem={res['n_tmem_points']})")
    if "a_rate" in res:
        print(f"  a_rate={res['a_rate']:.3f} (R²={res['r2_rate']:.2f})  "
              f"a_aggregate={res['a_aggregate']:.3f} (R²={res['r2_aggregate']:.2f})  "
              f"rel.disagree={res['relative_disagreement']:.2f}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="gc-rate-vs-capacity",
        description="Compare the per-trajectory error-removal rate's f-dependence to the "
                    "aggregate T_mem(f) exponent on random-label runs.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=Path, help="Path to a single configs/*.yaml")
    src.add_argument("--all", action="store_true", help="Iterate every yaml in configs/")
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir (default: results/<config_name>/rate_vs_capacity/)")
    p.add_argument("--db", default=None,
                   help="Path to runs.db (default: GC_WALLOW_DB env or repo runs.db)")
    args = p.parse_args(argv)

    if args.all:
        if args.out:
            print("[gc-rate-vs-capacity] --out is ignored with --all", file=sys.stderr)
        for cfg in sorted(CONFIGS_DIR.glob("*.yaml")):
            _render_one(cfg, DEFAULT_OUT_ROOT / cfg.stem / "rate_vs_capacity", args.db)
        return 0

    cfg = args.config
    out = args.out or (DEFAULT_OUT_ROOT / cfg.stem / "rate_vs_capacity")
    _render_one(cfg, out, args.db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
