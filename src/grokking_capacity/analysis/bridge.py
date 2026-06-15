"""`gc-bridge` — does the speed crossover coincide with the contraction norm threshold?

Stage 3 of the norm-contraction synthesis, and the conceptual keystone. Our framework
defines a speed crossover ``P_cross(p)`` where the memorisation- and generalisation-speed
curves meet; the contraction paper defines a critical norm ``V_*``. Are they the **same
boundary**? The unified piecewise delay

    ΔE(P) ≈ 𝟙[P ≥ P_cross] · (2κηλ)^{-1} · log( V_mem(P) / V_* )

is continuous at the boundary (ΔE → 0 as P → P_cross) only if ``log(V_mem(P_cross)/V_*) → 0``,
i.e. **V_mem(P_cross) ≈ V_***. So the bridge is exactly the consistency condition that lets
the two frameworks compose into a single ΔE(P).

Two readings, per prime:
  - ``g(p) = log10( V_mem(P_cross(p)) / V_*(p) )`` — ≈ 0 under the bridge;
  - the norm-crossover ``P_cross^V`` where ``V_mem(P) = V_*``, vs the speed ``P_cross``:
    ``Δ_log = log10(P_cross^V / P_cross)``.

**Proxy caveat.** ``P_cross`` is built from ``T_mem`` measured on *random* labels, while
``V_mem(P)`` is the *modular* memorising solution's norm. A systematic, roughly constant
gap is expected and is the source of the baseline offset ``δ₀`` — so a constant non-zero
``g(p)`` is BRIDGE OFFSET, not failure.

Pure post-hoc: reuses ``gc-contraction`` for V_mem/V_* and the intersection machinery for
``P_cross``. Config-driven like ``gc-figures``.
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

from . import aggregate
from .config_view import ArchGroup, ConfigView, load_npz
from .contraction import analyze_run


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO_ROOT / "configs"
DEFAULT_OUT_ROOT = REPO_ROOT / "results"

HOLDS_DEX = 0.10        # |mean g| at/under this with small scatter ⇒ bridge holds
SCATTER_DEX = 0.10      # std(g) at/under this ⇒ a clean (constant) offset
FAIL_SCATTER_DEX = 0.15  # std(g) over this ⇒ not the same boundary


def _seed_median_curve(recs: list[dict], field: str) -> dict[float, float]:
    """``{param_count: seed-median field}`` over per-run contraction records."""
    bins: dict[float, list[float]] = {}
    for r in recs:
        pc, v = r.get("param_count"), r.get(field)
        if pc is None or v is None or not np.isfinite(v) or v <= 0:
            continue
        bins.setdefault(float(pc), []).append(float(v))
    return {k: float(np.median(v)) for k, v in sorted(bins.items())}


def _loglog_interp(curve: dict[float, float], x: float) -> Optional[float]:
    """Interpolate ``curve`` (param_count → value) at ``x`` in log-log space."""
    if len(curve) < 2 or x <= 0:
        return None
    xs = np.array(sorted(curve), float)
    ys = np.array([curve[k] for k in xs], float)
    return float(np.exp(np.interp(np.log(x), np.log(xs), np.log(ys))))


def _invert_loglog(curve: dict[float, float], y_target: float) -> Optional[float]:
    """Find ``x`` where ``curve(x) = y_target`` in log-log space. Assumes the
    curve is monotone in log-log (V_mem grows with P); returns None if y_target
    is outside the curve's range."""
    if len(curve) < 2 or y_target <= 0:
        return None
    xs = np.array(sorted(curve), float)
    ys = np.array([curve[k] for k in xs], float)
    lx, ly = np.log(xs), np.log(ys)
    order = np.argsort(ly)
    ly_s, lx_s = ly[order], lx[order]
    lt = np.log(y_target)
    if lt < ly_s.min() or lt > ly_s.max():
        return None
    return float(np.exp(np.interp(lt, ly_s, lx_s)))


def _speed_crossover(group: ArchGroup, prime: Any) -> Optional[float]:
    """``P_cross(p)`` = param count where the mem- and gen-speed curves cross."""
    speed_curve = aggregate.mean_over_seeds(
        (r for r in group.speed_runs
         if r.get("p") == prime and r.get("param_count") is not None
         and r.get("saturation_epoch") is not None and r.get("saturated") is not False),
        x_field="param_count", y_field="saturation_epoch")
    groks_curve = aggregate.mean_over_seeds(
        (r for r in group.groks_runs
         if r.get("p") == prime and r.get("param_count") is not None
         and r.get("grokking_epoch") is not None),
        x_field="param_count", y_field="grokking_epoch")
    hit = aggregate.find_intersection(speed_curve, groks_curve)
    return float(hit[0]) if hit else None


# --------------------------------------------------------------------------- #
# Per-prime bridge record
# --------------------------------------------------------------------------- #
def _analyze_prime(group: ArchGroup, prime: Any,
                   recs: list[dict]) -> Optional[dict[str, Any]]:
    rp = [r for r in recs if r.get("p") == prime]
    v_mem = _seed_median_curve(rp, "V_mem")
    v_star = _seed_median_curve(rp, "V_star")
    if len(v_mem) < 2 or not v_star:
        return None
    v_star_rep = float(np.median(list(v_star.values())))   # V_* ~ constant per prime
    p_cross = _speed_crossover(group, prime)
    if p_cross is None:
        return None

    v_mem_at_cross = _loglog_interp(v_mem, p_cross)
    g = (float(np.log10(v_mem_at_cross / v_star_rep))
         if v_mem_at_cross and v_star_rep > 0 else None)
    p_cross_v = _invert_loglog(v_mem, v_star_rep)
    d_log = (float(np.log10(p_cross_v / p_cross))
             if p_cross_v and p_cross > 0 else None)

    return {
        "p": prime,
        "P_cross": p_cross,
        "P_cross_norm": p_cross_v,
        "V_star": v_star_rep,
        "V_mem_at_P_cross": v_mem_at_cross,
        "V_star_over_V_mem_cv": _cv(list(v_star.values())) if len(v_star) > 1 else float("nan"),
        "g": g,
        "delta_log": d_log,
        "_v_mem": v_mem, "_v_star": v_star,
    }


def _cv(x) -> float:
    x = np.asarray(x, float)
    m = float(np.mean(x))
    return float(np.std(x) / m) if m else float("nan")


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def analyse(view: ConfigView) -> dict[str, Any]:
    per_prime: list[dict[str, Any]] = []
    for group in view.iter_groups():
        if not group.groks_runs:
            continue
        recs = []
        for row in group.groks_runs:
            npz = load_npz(row)
            try:
                rec = analyze_run(row, npz)
            finally:
                npz.close()
            if rec is not None:
                recs.append(rec)
        for prime in sorted({r["p"] for r in recs if r["p"] is not None}):
            rec = _analyze_prime(group, prime, recs)
            if rec is not None:
                per_prime.append(rec)

    out: dict[str, Any] = {"config_name": view.config_name, "n_primes": len(per_prime)}
    gvals = np.array([r["g"] for r in per_prime if r["g"] is not None], float)
    if gvals.size < 1:
        out["verdict"] = "no_data"
        out["note"] = ("no prime yielded both a speed crossover P_cross and a V_mem(P) "
                       "curve — need ≥2 dims of grokking trajectories and a speed/groks "
                       "crossover per prime.")
        out["_per_prime"] = per_prime
        return out

    mean_g = float(np.mean(gvals))
    std_g = float(np.std(gvals)) if gvals.size > 1 else float("nan")
    scatter = std_g if np.isfinite(std_g) else 0.0

    if scatter > FAIL_SCATTER_DEX:
        verdict = "bridge_fails"
    elif abs(mean_g) <= HOLDS_DEX and scatter <= SCATTER_DEX:
        verdict = "bridge_holds"
    elif scatter <= SCATTER_DEX:
        verdict = "bridge_offset"
    else:
        verdict = "bridge_fails"

    routing = {
        "bridge_holds": "BRIDGE HOLDS — the crossover and the norm threshold are the same "
                        "boundary; the unified single ΔE(P) is licensed. Go to gc-delay "
                        "(Stage 4); set δ₀ ≈ mean g.",
        "bridge_offset": "BRIDGE OFFSET — constant non-zero g(p), consistent with the "
                         "random-vs-modular proxy gap. Record δ₀ = mean g (≈ the −0.16 "
                         "baseline). Go to Stage 4 expecting a constant offset.",
        "bridge_fails": "BRIDGE FAILS — g(p) scattered / p-dependent; crossover and V_* are "
                        "NOT the same boundary. Run Stage 4 as a STANDALONE delay test only; "
                        "do not claim unification.",
        "no_data": "",
    }[verdict]

    if not np.isfinite(std_g):
        routing += ("  [WARNING: only 1 prime with a usable bridge — scatter is "
                    "undefined; add primes before trusting the verdict.]")
    elif len(per_prime) < 3:
        routing += (f"  [CAUTION: scatter estimated from {len(per_prime)} primes — "
                    "weak; add primes to firm up the gate.]")

    out.update({
        "verdict": verdict,
        "mean_g_dex": mean_g,
        "std_g_dex": std_g,
        "delta_0": mean_g,            # the baseline offset proxy
        "mean_delta_log": float(np.nanmean(
            [r["delta_log"] for r in per_prime if r["delta_log"] is not None])
            if any(r["delta_log"] is not None for r in per_prime) else np.nan),
        "per_prime": [{k: v for k, v in r.items() if not k.startswith("_")}
                      for r in per_prime],
        "routing": routing,
        "_per_prime": per_prime,
    })
    return out


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def run_bridge(view: ConfigView, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    res = analyse(view)
    payload = {k: v for k, v in res.items() if not k.startswith("_")}
    with open(out_dir / "bridge.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    _plot(res, out_dir / "bridge.pdf")
    return res


def _plot(res: dict[str, Any], path: Path) -> None:
    per_prime = res.get("_per_prime", [])
    plotn = [r for r in per_prime if r.get("_v_mem")]
    if not plotn:
        return
    n = len(plotn)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.3), squeeze=False)
    palette = sns.color_palette("crest", n_colors=max(n, 1))
    for ax, r, col in zip(axes[0], plotn, palette):
        vm = r["_v_mem"]
        xs = np.array(sorted(vm), float)
        ys = np.array([vm[k] for k in xs], float)
        ax.plot(xs, ys, "o-", color=col, label=r"$V_{\rm mem}(P)$")
        ax.axhline(r["V_star"], color="0.4", ls="--", lw=1.2, label=r"$V_*$")
        if r["P_cross"]:
            ax.axvline(r["P_cross"], color="crimson", ls=":", lw=1.4,
                       label=r"$P_{\rm cross}$ (speed)")
        if r["P_cross_norm"]:
            ax.axvline(r["P_cross_norm"], color="navy", ls="-.", lw=1.0,
                       label=r"$P_{\rm cross}^{V}$ (norm)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("parameter count $P$")
        ax.set_ylabel(r"$\|\theta\|^2$")
        gtxt = f"{r['g']:.2f}" if r["g"] is not None else "n/a"
        ax.set_title(f"p={r['p']}  g={gtxt} dex")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, which="both")
    fig.suptitle(f"Bridge: {res.get('verdict', '')}  "
                 f"(mean g={res.get('mean_g_dex', float('nan')):.2f} dex)", y=1.02)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _render_one(config_path: Path, out_dir: Path, db_path: Optional[str]) -> None:
    print(f"[gc-bridge] {config_path.name} → {out_dir}")
    view = ConfigView.from_yaml(config_path, db_path=db_path)
    if not view.groups:
        print("  (no runs found in this config — skipping)")
        return
    res = run_bridge(view, out_dir)
    print(f"  verdict: {res['verdict']}  (n_primes={res['n_primes']})")
    if res.get("note"):
        print(f"  note: {res['note']}")
    if "mean_g_dex" in res:
        print(f"  mean g={res['mean_g_dex']:.3f} dex  std={res['std_g_dex']:.3f}  "
              f"δ₀={res['delta_0']:.3f}")
        print(f"  → {res['routing']}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="gc-bridge",
        description="Test V_mem(P_cross) ≈ V_*: whether the speed crossover and the "
                    "contraction norm threshold are the same boundary (the unification "
                    "consistency condition).",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=Path, help="Path to a single configs/*.yaml")
    src.add_argument("--all", action="store_true", help="Iterate every yaml in configs/")
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir (default: results/<config_name>/bridge/)")
    p.add_argument("--db", default=None,
                   help="Path to runs.db (default: GC_WALLOW_DB env or repo runs.db)")
    args = p.parse_args(argv)

    if args.all:
        if args.out:
            print("[gc-bridge] --out is ignored with --all", file=sys.stderr)
        for cfg in sorted(CONFIGS_DIR.glob("*.yaml")):
            _render_one(cfg, DEFAULT_OUT_ROOT / cfg.stem / "bridge", args.db)
        return 0

    cfg = args.config
    out = args.out or (DEFAULT_OUT_ROOT / cfg.stem / "bridge")
    _render_one(cfg, out, args.db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
