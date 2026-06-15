"""`gc-mem-law` — fortify the empirical memorisation law T_mem ∝ e^{a·f}.

Three things a sceptic asks of the law, beyond "R² is high":

  1. **Form.** Is it *exponential* in the capacity fraction ``f = dataset_bits/(C·P)``,
     or could a power law ``f^k`` or stretched exponential ``e^{a√f}`` fit as well? We fit
     all three (each linear in a transformed x) and compare by AIC.
  2. **Universality in f (the collapse).** Does ``T_mem`` depend on ``f`` *alone*, or
     separately on the width ``P`` / dataset size ``N``? With both axes swept (so ``f`` and
     ``log P`` decorrelate), we regress ``log T_mem ~ a·f + b·log P`` and test whether the
     residual ``P``-dependence ``b`` is non-zero. ``b ≈ 0`` ⟹ the law is genuinely about ``f``.
  3. **Precision.** A bootstrap CI on the exponent ``a`` (resampling runs), not a bare point.

Pure post-hoc on random-label runs (speed always; capacity with ``dataset_type='random'``);
recomputes ``T_mem`` from the stored ``train_acc_trace``. Config-driven like ``gc-figures``.
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

MEM_ACC = 0.99          # T_mem := first epoch with train_acc >= this
MIN_POINTS = 5
N_BOOTSTRAP = 4000
DELTA_AIC = 2.0         # exp must beat rivals by this much in AIC to "win"
COLLINEAR_R = 0.98      # |corr(f, logP)| above this ⇒ collapse test underpowered


def _norm_acc(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    return a / 100.0 if a.size and a.max() > 1.5 else a


def _t_mem(train_acc: np.ndarray) -> Optional[int]:
    acc = _norm_acc(train_acc)
    above = np.where(acc >= MEM_ACC)[0]
    return int(above[0]) + 1 if above.size else None


def _ols(X: np.ndarray, y: np.ndarray):
    """Return ``(coef, r2, rss)`` for ``y ≈ X·coef`` (X includes the intercept col)."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    rss = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / ss_tot if ss_tot > 0 else float("nan")
    return coef, r2, rss


def _aic(rss: float, n: int, k: int) -> float:
    return n * np.log(rss / n) + 2 * k if rss > 0 and n > 0 else float("nan")


def _bootstrap_ci(X: np.ndarray, y: np.ndarray, idx: int):
    rng = np.random.default_rng(0)
    n = len(y)
    coefs = []
    for _ in range(N_BOOTSTRAP):
        s = rng.integers(0, n, n)
        c, *_ = np.linalg.lstsq(X[s], y[s], rcond=None)
        coefs.append(c[idx])
    lo, hi = np.percentile(coefs, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Collect (f, T_mem, P, N) points
# --------------------------------------------------------------------------- #
def collect(view: ConfigView) -> list[dict[str, Any]]:
    pts: list[dict[str, Any]] = []
    for row in collect_random_runs(view):
        npz = load_npz(row)
        try:
            if "train_acc_trace" not in npz.files:
                continue
            tm = _t_mem(npz["train_acc_trace"])
        finally:
            npz.close()
        f = _capacity_fraction(row)
        pc = row.get("param_count")
        n = row.get("n_samples")
        if tm is None or f is None or f <= 0 or not pc or not n:
            continue
        pts.append({"f": float(f), "t_mem": float(tm), "param_count": float(pc),
                    "n_samples": float(n), "p": row.get("p"), "dim": row.get("dim"),
                    "uuid": row.get("uuid")})
    return pts


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def analyse(view: ConfigView) -> dict[str, Any]:
    pts = collect(view)
    out: dict[str, Any] = {"config_name": view.config_name, "n_points": len(pts)}
    if len(pts) < MIN_POINTS:
        out["verdict"] = "no_data"
        out["note"] = f"{len(pts)} (f, T_mem) points — need ≥{MIN_POINTS}."
        out["_pts"] = pts
        return out

    f = np.array([p["f"] for p in pts])
    y = np.log(np.array([p["t_mem"] for p in pts]))
    logP = np.log(np.array([p["param_count"] for p in pts]))
    logN = np.log(np.array([p["n_samples"] for p in pts]))
    ones = np.ones_like(f)

    # (1) primary exponential fit  log T = a·f + c
    Xexp = np.vstack([f, ones]).T
    coef_e, r2_e, rss_e = _ols(Xexp, y)
    a = float(coef_e[0])
    a_lo, a_hi = _bootstrap_ci(Xexp, y, 0)

    # (1b) functional-form comparison (each OLS of log T on a transform of f)
    models = {"exponential": f, "power_law": np.log(f), "stretched_exp": np.sqrt(f)}
    n = len(y)
    comparison: dict[str, Any] = {}
    for name, x in models.items():
        _, r2, rss = _ols(np.vstack([x, ones]).T, y)
        comparison[name] = {"r2": r2, "aic": _aic(rss, n, 2), "rss": rss}
    best = min(comparison, key=lambda k: comparison[k]["aic"])
    second = sorted(comparison, key=lambda k: comparison[k]["aic"])[1]
    exp_margin = comparison[second]["aic"] - comparison["exponential"]["aic"]
    form = "exponential" if best == "exponential" and exp_margin >= DELTA_AIC else best

    # (2) collapse / universality: log T = a·f + b·log P + c  (and + b·log N)
    corr_fP = float(np.corrcoef(f, logP)[0, 1])
    decorrelated = abs(corr_fP) < COLLINEAR_R
    XcP = np.vstack([f, logP, ones]).T
    cP, r2_cP, _ = _ols(XcP, y)
    bP, bP_lo, bP_hi = float(cP[1]), *_bootstrap_ci(XcP, y, 1)
    XcN = np.vstack([f, logN, ones]).T
    cN, _, _ = _ols(XcN, y)
    bN, bN_lo, bN_hi = float(cN[1]), *_bootstrap_ci(XcN, y, 1)

    if not decorrelated:
        collapse = "underpowered"     # f and log P collinear — need the N-sweep
    elif bP_lo <= 0 <= bP_hi:
        collapse = "f_sufficient"     # no residual P-dependence beyond f
    else:
        collapse = "residual_P_dependence"

    out.update({
        "verdict": form,
        "a": a, "a_ci95": [a_lo, a_hi], "r2_exponential": r2_e,
        "form_winner": best, "exp_aic_margin": float(exp_margin),
        "model_comparison": comparison,
        "collapse": collapse,
        "f_vs_logP_corr": corr_fP, "axes_decorrelated": bool(decorrelated),
        "residual_P_coef": {"b": bP, "ci95": [bP_lo, bP_hi]},
        "residual_N_coef": {"b": bN, "ci95": [bN_lo, bN_hi]},
        "f_range": [float(f.min()), float(f.max())],
        "routing": _routing(form, collapse, decorrelated),
        "_pts": pts, "_a": a, "_c": float(coef_e[1]), "_comparison_coef": _fit_curves(f, y),
    })
    return out


def _fit_curves(f: np.ndarray, y: np.ndarray) -> dict[str, list[float]]:
    ones = np.ones_like(f)
    out = {}
    for name, x in {"exponential": f, "power_law": np.log(f),
                    "stretched_exp": np.sqrt(f)}.items():
        coef, *_ = _ols(np.vstack([x, ones]).T, y)
        out[name] = [float(coef[0]), float(coef[1])]
    return out


def _routing(form: str, collapse: str, decorrelated: bool) -> str:
    parts = []
    if form == "exponential":
        parts.append("FORM: exponential wins on AIC — e^{af} is the right functional form.")
    else:
        parts.append(f"FORM: {form} fits better than exponential — re-examine the law.")
    if collapse == "f_sufficient":
        parts.append("COLLAPSE: no residual P-dependence — T_mem depends on f alone "
                     "(universality in the capacity fraction holds).")
    elif collapse == "residual_P_dependence":
        parts.append("COLLAPSE: residual P-dependence beyond f — the law is not purely "
                     "about f; report the extra term.")
    else:
        parts.append("COLLAPSE: underpowered — f and log P are collinear; add the "
                     "orthogonal n_samples sweep (speed_nsweep) to decorrelate them.")
    return "  ".join(parts)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def run_mem_law(view: ConfigView, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    res = analyse(view)
    payload = {k: v for k, v in res.items() if not k.startswith("_")}
    with open(out_dir / "mem_law.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    _plot_fit(res, out_dir / "mem_law_fit.pdf")
    _plot_collapse(res, out_dir / "mem_law_collapse.pdf")
    return res


def _plot_fit(res: dict[str, Any], path: Path) -> None:
    pts = res.get("_pts", [])
    if len(pts) < MIN_POINTS or "_comparison_coef" not in res:
        return
    f = np.array([p["f"] for p in pts])
    t = np.array([p["t_mem"] for p in pts])
    pc = np.array([p["param_count"] for p in pts])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sc = ax.scatter(f, t, c=np.log10(pc), cmap="viridis", s=24, alpha=0.8)
    plt.colorbar(sc, label=r"$\log_{10} P$")
    xs = np.linspace(float(f.min()), float(f.max()), 200)
    styles = {"exponential": ("k-", 1.8), "power_law": ("--", 1.2),
              "stretched_exp": (":", 1.2)}
    cc = res["_comparison_coef"]
    for name, (ls, lw) in styles.items():
        a, c = cc[name]
        if name == "exponential":
            yy = np.exp(a * xs + c)
        elif name == "power_law":
            yy = np.exp(a * np.log(xs) + c)
        else:
            yy = np.exp(a * np.sqrt(xs) + c)
        aic = res["model_comparison"][name]["aic"]
        ax.plot(xs, yy, ls if name != "exponential" else "k-", lw=lw, color=None,
                label=f"{name} (AIC={aic:.0f})")
    ax.set_yscale("log")
    ax.set_xlabel(r"capacity fraction $f = \mathrm{bits}/(C\,P)$")
    ax.set_ylabel(r"$T_{\rm mem}$  (epochs)")
    ai = res["a_ci95"]
    ax.set_title(fr"$T_{{\rm mem}}\propto e^{{a f}}$:  $a$={res['a']:.1f} "
                 fr"[{ai[0]:.1f}, {ai[1]:.1f}]  ($R^2$={res['r2_exponential']:.3f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _plot_collapse(res: dict[str, Any], path: Path) -> None:
    pts = res.get("_pts", [])
    if len(pts) < MIN_POINTS or "_a" not in res:
        return
    f = np.array([p["f"] for p in pts])
    y = np.log(np.array([p["t_mem"] for p in pts]))
    logP = np.log(np.array([p["param_count"] for p in pts]))
    resid = y - (res["_a"] * f + res["_c"])      # residual of the f-only exp fit

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(logP, resid, s=24, alpha=0.8, color=sns.color_palette("flare")[2])
    b = res["residual_P_coef"]["b"]
    xs = np.array([logP.min(), logP.max()])
    ax.plot(xs, b * (xs - logP.mean()), "k--", lw=1.3,
            label=fr"residual slope $b$={b:.3f}")
    ax.axhline(0, color="0.6", lw=1)
    ax.set_xlabel(r"$\log P$")
    ax.set_ylabel(r"residual of $e^{af}$ fit  ($\log T_{\rm mem}$)")
    ax.set_title(f"Collapse test: {res['collapse']}  "
                 f"(decorrelated={res['axes_decorrelated']})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _render_one(config_path: Path, out_dir: Path, db_path: Optional[str]) -> None:
    print(f"[gc-mem-law] {config_path.name} → {out_dir}")
    view = ConfigView.from_yaml(config_path, db_path=db_path)
    if not view.groups:
        print("  (no runs found in this config — skipping)")
        return
    res = run_mem_law(view, out_dir)
    print(f"  form: {res['verdict']}  (n_points={res['n_points']})")
    if res.get("note"):
        print(f"  note: {res['note']}")
    if "a" in res:
        ai = res["a_ci95"]
        print(f"  a={res['a']:.2f} [{ai[0]:.2f}, {ai[1]:.2f}]  "
              f"R²={res['r2_exponential']:.3f}  f∈[{res['f_range'][0]:.2f},{res['f_range'][1]:.2f}]")
        print(f"  collapse: {res['collapse']}  "
              f"(b_P={res['residual_P_coef']['b']:.3f}, corr(f,logP)={res['f_vs_logP_corr']:.2f})")
        print(f"  → {res['routing']}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="gc-mem-law",
        description="Fortify T_mem ∝ e^{a·f}: bootstrap CI on a, exp-vs-power-vs-stretched "
                    "model comparison, and the f-collapse universality test.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=Path, help="Path to a single configs/*.yaml")
    src.add_argument("--all", action="store_true", help="Iterate every yaml in configs/")
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir (default: results/<config_name>/mem_law/)")
    p.add_argument("--db", default=None,
                   help="Path to runs.db (default: GC_WALLOW_DB env or repo runs.db)")
    args = p.parse_args(argv)

    if args.all:
        if args.out:
            print("[gc-mem-law] --out is ignored with --all", file=sys.stderr)
        for cfg in sorted(CONFIGS_DIR.glob("*.yaml")):
            _render_one(cfg, DEFAULT_OUT_ROOT / cfg.stem / "mem_law", args.db)
        return 0

    cfg = args.config
    out = args.out or (DEFAULT_OUT_ROOT / cfg.stem / "mem_law")
    _render_one(cfg, out, args.db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
