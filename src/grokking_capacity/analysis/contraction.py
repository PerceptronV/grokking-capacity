"""`gc-contraction` — post-memorisation parameter-norm contraction on modular runs.

Tests whether the squared parameter norm ``V_t = ‖θ‖²`` contracts **log-linearly**
between memorisation and generalisation — the norm first-passage mechanism of Truong et
al. (arXiv:2605.18845). Under decoupled weight decay ``θ_{t+1} = (1-ηλ)θ_t - η∇L`` the
post-memorisation small-gradient regime gives ``log V_t ≈ log V_mem - 2ηλ(t-T_mem)``, i.e.
``log V_t`` linear in the step ``t`` with slope ``-2ηλ``. We never assume the rate: per
trajectory we fit ``r = -slope`` and the AdamW-corrected ``κ = r/(2ηλ) ∈ (0,1]`` (AdamW's
adaptive term drives ``V_t`` to a non-zero asymptote, so a plain log-linear fit recovers an
effective rate below the clean ``2ηλ``; the Kosson refit ``V_t = V_∞ + A·e^{-r(t-t₀)}``
removes that bias).

Reads the per-step ``norm_values`` / ``norm_steps`` channel persisted by ``gc-groks``
(needs ``--norm-log-every > 0``) together with the per-epoch accuracy traces. For each
grokking trajectory it fits ``log V_t`` on the window ``[T_mem + δ, T_grok - δ]`` and
records ``r, R², κ, κ_kos, V_mem, V_post, V_*`` plus ``ρ = log(V_mem/V_post)``.

Pure post-hoc: never trains or writes to the registry. Config-driven like ``gc-figures``
(``--config <yaml>`` or ``--all``); scopes runs via :class:`ConfigView`.

Mechanism check, not a result we want true — the gate has an honest negative outcome
(``does_not_apply``) that is itself a finding. Thresholds are fixed here, before looking at
any aggregate.
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

from .config_view import ConfigView, load_npz


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO_ROOT / "configs"
DEFAULT_OUT_ROOT = REPO_ROOT / "results"

# --- definitions (§1.5) — fixed before viewing aggregates ----------------- #
TRAIN_MEM_ACC = 0.99      # T_mem  := first epoch with train_acc >= this
VAL_GROK_ACC = 0.99       # T_grok := first epoch with val_acc   >= this
VAL_MID_ACC = 0.50        # V_*     := V_t at the epoch val_acc first crosses this
MIN_STEP_DELTA = 20       # δ = max(MIN_STEP_DELTA, DELTA_FRAC·Δt) steps trimmed each end
DELTA_FRAC = 0.05
MIN_WINDOW_POINTS = 25    # fewer logged V_t points in-window -> under-resolved, excluded
TOP_TERCILE = 1.0 / 3.0   # headline set: largest-Δt third per prime (§1.6)

# --- gate thresholds (§2.3) ----------------------------------------------- #
APPLIES_N = 20            # min headline trajectories
APPLIES_R2 = 0.90         # median per-traj log-linear R²
NEG_R2 = 0.70             # below this (and Kosson below NEG_KOSSON_R2) -> not exponential
NEG_KOSSON_R2 = 0.80
VMEM_GT_VPOST_FRAC = 0.90  # fraction of runs with V_mem > V_post required to APPLY


# --------------------------------------------------------------------------- #
# Run selection
# --------------------------------------------------------------------------- #
def collect_groks_runs(view: ConfigView) -> list[dict[str, Any]]:
    """Every completed groks run in the view (modular-task trajectories)."""
    runs: list[dict[str, Any]] = []
    for group in view.iter_groups():
        runs.extend(group.groks_runs)
    return runs


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _norm_acc(arr: np.ndarray) -> np.ndarray:
    """Accuracy trace as a fraction in [0,1] (groks stores it in %)."""
    a = np.asarray(arr, dtype=float)
    return a / 100.0 if a.size and a.max() > 1.5 else a


def _first_cross(acc: np.ndarray, thr: float) -> Optional[int]:
    """0-based epoch index where ``acc`` first reaches ``thr``, else ``None``."""
    above = np.where(acc >= thr)[0]
    return int(above[0]) if above.size else None


def _v_at_step(steps: np.ndarray, vals: np.ndarray, target: float) -> float:
    """``V_t`` at ``target`` step by log-linear interpolation of the logged points."""
    if steps.size == 0:
        return float("nan")
    return float(np.exp(np.interp(target, steps, np.log(vals))))


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Fits (§2.1)
# --------------------------------------------------------------------------- #
def fit_loglinear(steps: np.ndarray, V: np.ndarray):
    """Linear fit ``log V = intercept + slope·t``. Returns ``(slope, intercept, r2)``;
    the contraction rate is ``r = -slope`` (per step)."""
    s = np.asarray(steps, float)
    y = np.log(np.asarray(V, float))
    A = np.vstack([s, np.ones_like(s)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ np.array([slope, intercept])
    return float(slope), float(intercept), _r2(y, yhat)


def fit_kosson(steps: np.ndarray, V: np.ndarray):
    """Kosson refit ``V_t = V_∞ + A·exp(-r_kos·(t-t₀))`` (removes the AdamW asymptote
    bias). Returns ``(V_inf, A, r_kos, r2)``; raises if the fit fails."""
    s = np.asarray(steps, float)
    v = np.asarray(V, float)
    t0 = s[0]
    (Vi, Aa, rk), _ = curve_fit(
        lambda t, Vi, Aa, rk: Vi + Aa * np.exp(-rk * (t - t0)),
        s, v, p0=[v.min(), v[0] - v.min(), 1e-4],
        bounds=([0, 0, 0], [v.min() * 1.5 + 1e-9, np.inf, 1.0]), maxfev=20000,
    )
    vhat = Vi + Aa * np.exp(-rk * (s - t0))
    return float(Vi), float(Aa), float(rk), _r2(v, vhat)


# --------------------------------------------------------------------------- #
# Per-run analysis
# --------------------------------------------------------------------------- #
def analyze_run(row: dict[str, Any], npz) -> Optional[dict[str, Any]]:
    """Per-trajectory contraction record, or ``None`` when the run lacks the norm
    channel, never groks, or the in-window logged points fall below MIN_WINDOW_POINTS."""
    files = set(npz.files)
    if not {"norm_values", "norm_steps", "train_acc", "val_acc"} <= files:
        return None
    spe = int(npz["steps_per_epoch"]) if "steps_per_epoch" in files else 0
    if spe <= 0:
        return None

    train_acc = _norm_acc(npz["train_acc"])
    val_acc = _norm_acc(npz["val_acc"])
    mem_e = _first_cross(train_acc, TRAIN_MEM_ACC)
    grok_e = _first_cross(val_acc, VAL_GROK_ACC)
    half_e = _first_cross(val_acc, VAL_MID_ACC)
    if mem_e is None or grok_e is None:
        return None

    # Per-epoch threshold epoch e (0-based) settles at global step (e+1)·spe.
    t_mem = (mem_e + 1) * spe
    t_grok = (grok_e + 1) * spe
    dt = t_grok - t_mem
    if dt <= 0:                       # immediate generaliser — no delay to fit
        return None
    delta = max(MIN_STEP_DELTA, DELTA_FRAC * dt)
    lo, hi = t_mem + delta, t_grok - delta

    steps = np.asarray(npz["norm_steps"], float)
    vals = np.asarray(npz["norm_values"], float)
    order = np.argsort(steps)
    steps, vals = steps[order], vals[order]
    mask = (steps >= lo) & (steps <= hi) & (vals > 0)
    if int(mask.sum()) < MIN_WINDOW_POINTS:
        return None
    sw, vw = steps[mask], vals[mask]

    eta = float(row.get("lr") or 0.0)
    lam = float(row.get("weight_decay") or 0.0)
    two_etalam = 2.0 * eta * lam

    slope, _, r2 = fit_loglinear(sw, vw)
    r = -slope
    kappa = r / two_etalam if two_etalam > 0 else float("nan")
    try:
        _, _, r_kos, r2_kos = fit_kosson(sw, vw)
        kappa_kos = r_kos / two_etalam if two_etalam > 0 else float("nan")
    except Exception:
        r_kos = r2_kos = kappa_kos = float("nan")

    v_mem = _v_at_step(steps, vals, t_mem)
    v_star = _v_at_step(steps, vals, (half_e + 1) * spe) if half_e is not None else float("nan")
    post = vals[steps >= t_grok]
    v_post = float(np.median(post[-5:])) if post.size else float(vals[-1])
    rho = float(np.log(v_mem / v_post)) if v_mem > 0 and v_post > 0 else float("nan")

    return {
        "uuid": row.get("uuid"),
        "p": row.get("p"),
        "dim": row.get("dim"),
        "depth": row.get("depth"),
        "heads": row.get("heads"),
        "weight_decay": lam,
        "lr": eta,
        "param_count": row.get("param_count"),
        "T_mem": int(t_mem),
        "T_grok": int(t_grok),
        "delta_t": int(dt),
        "n_window": int(sw.size),
        "r": float(r),
        "r2_loglinear": float(r2),
        "kappa": float(kappa),
        "r_kosson": float(r_kos),
        "r2_kosson": float(r2_kos),
        "kappa_kosson": float(kappa_kos),
        "V_mem": float(v_mem),
        "V_post": float(v_post),
        "V_star": float(v_star),
        "rho": rho,
        "vmem_gt_vpost": bool(v_mem > v_post),
        # plotting only; not serialised
        "_sw": sw, "_vw": vw, "_v_mem": v_mem, "_t_mem": t_mem, "_slope": slope,
    }


def _select_headline(valid: list[dict]) -> list[dict]:
    """Largest-Δt tercile per prime — the runs §1.6 says to trust (high-λ short
    windows are noisy). Falls back to the full pool if the tercile is too thin."""
    headline: list[dict] = []
    primes = sorted({r["p"] for r in valid if r["p"] is not None})
    for p in primes:
        cell = sorted((r for r in valid if r["p"] == p),
                      key=lambda r: r["delta_t"], reverse=True)
        k = max(1, int(round(len(cell) * TOP_TERCILE)))
        headline.extend(cell[:k])
    if len(headline) < APPLIES_N and len(valid) >= len(headline):
        return valid                  # too few in the tercile — use everything, flagged
    return headline


# --------------------------------------------------------------------------- #
# Gate (§2.3)
# --------------------------------------------------------------------------- #
def _gate(headline: list[dict], n_valid_full: int, n_excluded: int) -> dict[str, Any]:
    if not headline:
        return {"verdict": "no_data", "n_headline": 0,
                "n_valid_full": n_valid_full, "n_excluded": n_excluded}

    r2 = np.array([r["r2_loglinear"] for r in headline], float)
    r2_kos = np.array([r["r2_kosson"] for r in headline], float)
    rate = np.array([r["r"] for r in headline], float)
    kappa = np.array([r["kappa"] for r in headline], float)
    frac_vmem = float(np.mean([r["vmem_gt_vpost"] for r in headline]))

    med_r2 = float(np.median(r2))
    med_r2_kos = float(np.nanmedian(r2_kos)) if np.isfinite(r2_kos).any() else float("nan")
    med_slope = float(np.median(-rate))      # slope of log V_t; <0 means contraction
    med_kappa = float(np.nanmedian(kappa))
    n = len(headline)

    not_exponential = (med_r2 < NEG_R2 and
                       (not np.isfinite(med_r2_kos) or med_r2_kos < NEG_KOSSON_R2))
    if med_slope >= 0 or frac_vmem < 0.5 or not_exponential:
        verdict = "does_not_apply"
    elif (n >= APPLIES_N and med_r2 > APPLIES_R2
          and 0 < med_kappa <= 1 and frac_vmem >= VMEM_GT_VPOST_FRAC):
        verdict = "applies"
    else:
        verdict = "inconclusive"

    return {
        "verdict": verdict,
        "n_headline": n,
        "n_valid_full": n_valid_full,
        "n_excluded": n_excluded,
        "median_r2_loglinear": med_r2,
        "iqr_r2_loglinear": [float(np.percentile(r2, 25)), float(np.percentile(r2, 75))],
        "median_r2_kosson": med_r2_kos,
        "median_slope_logV": med_slope,
        "median_kappa": med_kappa,
        "cv_kappa": float(np.nanstd(kappa) / np.nanmean(kappa))
                    if np.isfinite(kappa).any() and np.nanmean(kappa) else float("nan"),
        "median_kappa_kosson": float(np.nanmedian([r["kappa_kosson"] for r in headline])),
        "frac_vmem_gt_vpost": frac_vmem,
        "median_rho": float(np.nanmedian([r["rho"] for r in headline])),
        "routing": _ROUTING.get(verdict, ""),
    }


_ROUTING = {
    "applies": "Go to Stage 2 (gc-kappa-stability). Persist per-run κ, V_mem, V_post, V_*.",
    "does_not_apply": "STOP the contraction track — T_gen is not contraction-driven; "
                      "revert to the jamming/interpolation account (Branch J).",
    "inconclusive": "Amplification sweep: low λ ∈ {0.1, 0.3} at p∈{97,113,139}, 1-2 large "
                    "d past crossover, re-run; carry a 'compressed-at-high-λ' flag.",
    "no_data": "No instrumented grokking trajectories — re-run gc-groks with "
               "--norm-log-every>0 (and --post-grok-epochs>0 for a settled V_post).",
}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_contraction(view: ConfigView, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    valid: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in collect_groks_runs(view):
        npz = load_npz(row)
        try:
            rec = analyze_run(row, npz)
        finally:
            npz.close()
        if rec is None:
            excluded.append({k: row.get(k) for k in ("uuid", "p", "dim", "seed")})
        else:
            valid.append(rec)

    headline = _select_headline(valid)
    headline_ids = {r["uuid"] for r in headline}
    for r in valid:
        r["headline"] = r["uuid"] in headline_ids
    summary = _gate(headline, len(valid), len(excluded))

    _write_json(valid, excluded, summary, view.config_name, out_dir)
    _plot_collapse(headline, out_dir / "contraction_collapse.pdf")
    _plot_r2_hist(headline, out_dir / "contraction_r2_hist.pdf")
    _plot_kappa(headline, out_dir / "contraction_kappa.pdf")
    return summary


def _write_json(valid, excluded, summary, config_name, out_dir: Path) -> None:
    payload = {
        "config_name": config_name,
        "summary": summary,
        "runs": [{k: v for k, v in r.items() if not k.startswith("_")} for r in valid],
        "excluded": excluded,
    }
    with open(out_dir / "contraction.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)


# --------------------------------------------------------------------------- #
# Figures (A: collapse, B: R² hist, C: κ vs p)
# --------------------------------------------------------------------------- #
def _prime_palette(records: list[dict]):
    primes = sorted({r["p"] for r in records if r["p"] is not None})
    palette = sns.color_palette("crest", n_colors=max(len(primes), 1))
    return primes, {p: palette[i] for i, p in enumerate(primes)}


def _plot_collapse(records: list[dict], path: Path) -> None:
    if not records:
        return
    primes, colour = _prime_palette(records)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for r in records:
        sw, vw = r["_sw"], r["_vw"]
        dt = sw - r["_t_mem"]
        c = colour.get(r["p"], "0.5")
        ax.plot(dt, vw / r["_v_mem"], color=c, alpha=0.35, lw=1)
        yhat = np.exp(np.log(vw[0]) + r["_slope"] * (sw - sw[0]))
        ax.plot(dt, yhat / r["_v_mem"], color=c, alpha=0.9, lw=1, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel(r"steps since $T_{\rm mem}$  ($t - T_{\rm mem}$)")
    ax.set_ylabel(r"$V_t / V_{\rm mem}$")
    ax.set_title("Post-memorisation norm contraction")
    ax.grid(True, alpha=0.3, which="both")
    handles = [plt.Line2D([], [], color=colour[p], label=f"p={p}") for p in primes]
    if handles:
        ax.legend(handles=handles, fontsize=8, title="prime")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _plot_r2_hist(records: list[dict], path: Path) -> None:
    if not records:
        return
    r2 = [r["r2_loglinear"] for r in records]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(r2, bins=20, range=(min(0.0, min(r2)), 1.0),
            color=sns.color_palette("crest")[3], edgecolor="white")
    for x, ls in ((0.90, "--"), (0.95, ":")):
        ax.axvline(x, color="0.3", ls=ls, lw=1, label=f"$R^2$={x:g}")
    ax.set_xlabel(r"per-trajectory log-linear $R^2$")
    ax.set_ylabel("count")
    ax.set_title("Contraction-fit quality across trajectories")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _plot_kappa(records: list[dict], path: Path) -> None:
    if not records:
        return
    primes, colour = _prime_palette(records)
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in records:
        if r["p"] is None or not np.isfinite(r["kappa"]):
            continue
        ax.scatter(r["p"], r["kappa"], color=colour.get(r["p"], "0.5"),
                   alpha=0.6, s=18)
    ax.axhspan(0.18, 0.37, color="0.85", alpha=0.5, zorder=0,
               label="paper κ range")
    ax.set_xlabel("prime $p$")
    ax.set_ylabel(r"$\kappa = r / (2\eta\lambda)$")
    ax.set_title("AdamW-corrected contraction rate per prime")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _render_one(config_path: Path, out_dir: Path, db_path: Optional[str]) -> None:
    print(f"[gc-contraction] {config_path.name} → {out_dir}")
    view = ConfigView.from_yaml(config_path, db_path=db_path)
    if not view.groups:
        print("  (no runs found in this config — skipping)")
        return
    s = run_contraction(view, out_dir)
    print(f"  verdict: {s['verdict']}  "
          f"(n_headline={s.get('n_headline', 0)}, "
          f"n_valid={s.get('n_valid_full', 0)}, n_excluded={s.get('n_excluded', 0)})")
    if s.get("n_headline"):
        print(f"  median R²={s['median_r2_loglinear']:.3f}  "
              f"median κ={s['median_kappa']:.3f}  "
              f"frac(V_mem>V_post)={s['frac_vmem_gt_vpost']:.2f}")
    if s.get("routing"):
        print(f"  → {s['routing']}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="gc-contraction",
        description="Fit post-memorisation parameter-norm contraction (log-linear + Kosson) "
                    "on grokking trajectories for a grokking_capacity YAML config.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=Path, help="Path to a single configs/*.yaml")
    src.add_argument("--all", action="store_true", help="Iterate every yaml in configs/")
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory (default: results/<config_name>/contraction/)")
    p.add_argument("--db", default=None,
                   help="Path to runs.db (default: GC_WALLOW_DB env or repo runs.db)")
    args = p.parse_args(argv)

    if args.all:
        if args.out:
            print("[gc-contraction] --out is ignored with --all", file=sys.stderr)
        for cfg in sorted(CONFIGS_DIR.glob("*.yaml")):
            _render_one(cfg, DEFAULT_OUT_ROOT / cfg.stem / "contraction", args.db)
        return 0

    cfg = args.config
    out = args.out or (DEFAULT_OUT_ROOT / cfg.stem / "contraction")
    _render_one(cfg, out, args.db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
