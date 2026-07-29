"""Day-0 timescale-law fits and the pre-registered p=197 onset forecast.

Implements, verbatim, the pre-committed procedure of
``docs/papers/2026 Grokking Capacity/theory/theory-trunk.md`` §0 (written and mtime-stamped before this
module existed):

  * memorisation law  ln T_mem = ln b + γ·f,  f = K/(C·P), pooled OLS over
    the ten central primes' uncensored speed runs; run-level bootstrap CI;
    AIC comparison vs power-law / stretched-exp / critical forms; quadratic
    curvature check;
  * generalisation law  ln T_gen = ln A + a·ln p − β·ln P, pooled and
    per-prime (β_p spread + drift-in-p check); window: uncensored seed-mean
    cells with P ≥ K/C and dim ≤ max_dim;
  * implicit crossing  γK/(C·P) + β·ln P = ln(A p^a / b), solved per prime,
    calibration offset Δ̂ = median log10(P_onset/P_cross) over the ten
    primes; forecasts for p ∈ {149 (held-out), 197, 251} as
    log10 P_onset = log10 P_cross + Δ̂;
  * uncertainty: prime-level bootstrap of the whole pipeline (fit band) ⊕
    seed-subsample (8-of-10) bootstrap of the empirical-onset estimator
    (noise floor), in quadrature; β-sensitivity and window-sensitivity
    reported separately;
  * epoch-budget pre-check for the planned p=197 width grid.

Everything is derived from the canonical merged registry through the same
``ConfigView`` / ``aggregate`` layer the published figures use.

Usage:
  python -m grokking_capacity.analysis.forecast_onset \
      --config configs/central.yaml --db runs.db --out results/theory
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from . import aggregate
from .config_view import ConfigView, load_npz
from .matching import compute_n_equiv
from .plots import _passes_filters

CENTRAL_PRIMES = (97, 101, 103, 107, 109, 113, 127, 131, 137, 139)
HELD_OUT_PRIME = 149
FORECAST_PRIMES = (197, 251)

SEED_SUBSAMPLE_M = 8          # m-of-n subsample for the onset noise floor
N_SEED_DRAWS = 500
N_PRIME_BOOT = 2000
N_RUN_BOOT = 4000
BETA_PERTURB = 0.1
BAND_GATE_DEX = 0.30          # pre-registration gate on the total 95% band
EPOCH_CAP = 5000
PROXY_RATIO = 0.70            # measured T_mem/E_train at depth 2 (p02 log)

# Amendments 1+2 (theory-trunk §0b/§0c): both timescale laws are fit on a
# common dimensionless window in capacity fraction — the branch on which
# every observed crossing lies (f_cross ≈ 0.10–0.17) — in epochs.
GEN_F_LO = 0.08
GEN_F_HI = 1.0
MEM_F_LO = 0.08
MEM_F_HI = 1.0
BATCH = 512                   # steps/epoch = ceil(n_train / BATCH)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def _pick_group(view: ConfigView):
    """The central config has one arch group carrying both speed and groks
    rows (capacity rows live in their own dropout=0 group)."""
    candidates = [g for g in view.iter_groups() if g.groks_runs and g.speed_runs]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one speed+groks arch group, found {len(candidates)}")
    return candidates[0]


def collect(view: ConfigView) -> dict[str, Any]:
    figure = view.intersection_figures[0]
    group = _pick_group(view)
    C = float(group.capacity_constant)
    op = group.key.operation
    tf = float(group.key.train_fraction)

    k_mem: dict[int, float] = {}

    def K_of(p: int) -> float:
        if p not in k_mem:
            k_mem[p] = float(compute_n_equiv(p, op, tf)[1])
        return k_mem[p]

    # Speed runs → per-run (f, T_mem) points, uncensored only.
    speed_pts: list[dict] = []
    for r in group.speed_runs:
        if (not _passes_filters(r, figure) or r.get("saturated") is False
                or r.get("saturation_epoch") is None
                or r.get("param_count") is None or r.get("p") is None):
            continue
        p, pc = int(r["p"]), float(r["param_count"])
        bits = r.get("dataset_bits")
        bits = float(bits) if bits is not None else float(
            r["n_samples"]) * math.log2(p + 2)
        speed_pts.append({
            "p": p, "param_count": pc, "dim": r.get("dim"),
            "seed": r.get("seed"), "t_mem": float(r["saturation_epoch"]),
            "f": bits / (C * pc),
        })

    # Groks runs → seed-mean T_gen cells per (p, param_count).
    tgen_cells: list[dict] = []
    for p in sorted({pt["p"] for pt in speed_pts} | set(CENTRAL_PRIMES) | {HELD_OUT_PRIME}):
        curve = aggregate.mean_over_seeds(
            (r for r in group.groks_runs
             if r.get("p") == p and _passes_filters(r, figure)
             and r.get("grokking_epoch") is not None
             and r.get("saturated") is not False),
            x_field="param_count", y_field="grokking_epoch")
        for pc, tg in curve.items():
            f_cell = K_of(p) / (C * float(pc))
            tgen_cells.append({"p": p, "param_count": float(pc), "t_gen": float(tg),
                               "f": f_cell,
                               "in_window": GEN_F_LO <= f_cell <= GEN_F_HI})

    # Per-seed delay records (npz reads) for onset + estimator bootstrap.
    delay_records: dict[int, list[dict]] = {}
    for p in sorted({c["p"] for c in tgen_cells}):
        recs: list[dict] = []
        for r in group.groks_runs:
            if r.get("p") != p or not _passes_filters(r, figure):
                continue
            try:
                with load_npz(r) as npz:
                    train, val = npz["train_acc"], npz["val_acc"]
            except (FileNotFoundError, KeyError):
                continue
            delays = aggregate.compute_delays(
                [{"param_count": r.get("param_count"),
                  "train_acc": train, "val_acc": val}],
                x_field="param_count",
                threshold_train=figure.delay_train_threshold,
                threshold_val=figure.delay_val_threshold)
            if delays:
                recs.append({"x": delays[0][0], "delay": delays[0][1],
                             "seed": r.get("seed"), "dim": r.get("dim")})
        delay_records[p] = recs

    # Published-pipeline empirical curve intersections, for the cross-check.
    from .stats import _predicted_onset
    pipeline_cross = {p: _predicted_onset(group, figure, p)
                      for p in sorted(delay_records)}

    return {"C": C, "operation": op, "train_fraction": tf, "K_of": K_of,
            "speed_pts": speed_pts, "tgen_cells": tgen_cells,
            "delay_records": delay_records, "pipeline_cross": pipeline_cross,
            "figure": figure, "group": group}


# --------------------------------------------------------------------------- #
# Fits
# --------------------------------------------------------------------------- #

def _ols(X: np.ndarray, y: np.ndarray):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    rss = float(resid @ resid)
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - rss / tss if tss > 0 else float("nan")
    return coef, r2, rss


def _aic(rss: float, n: int, k: int) -> float:
    return n * math.log(rss / n) + 2 * k


def fit_mem_law(pts: list[dict], primes: Iterable[int], *,
                light: bool = False, windowed: bool = True) -> dict[str, Any]:
    by_p: dict[int, list[dict]] = {}
    for q in pts:
        if not windowed or MEM_F_LO <= q["f"] <= MEM_F_HI:
            by_p.setdefault(q["p"], []).append(q)
    sel = [q for p in primes for q in by_p.get(p, [])]  # multiplicity respected
    f = np.array([q["f"] for q in sel])
    y = np.log(np.array([q["t_mem"] for q in sel]))
    ones = np.ones_like(f)

    coef, r2, rss = _ols(np.vstack([ones, f]).T, y)
    lnb, gamma = float(coef[0]), float(coef[1])
    n = len(y)
    if light:
        return {"gamma": gamma, "ln_b": lnb, "b": float(math.exp(lnb)),
                "r2": r2, "n_runs": n}

    forms = {
        "exponential": f,
        "power_law": np.log(f),
        "stretched_exp": np.sqrt(f),
        "critical": -np.log(1.0 - np.sqrt(np.clip(f, 0, 0.9999))),
    }
    comparison = {}
    for name, x in forms.items():
        _, r2_m, rss_m = _ols(np.vstack([ones, x]).T, y)
        comparison[name] = {"r2": r2_m, "aic": _aic(rss_m, n, 2)}

    cq, _, _ = _ols(np.vstack([ones, f, f ** 2]).T, y)

    rng = np.random.default_rng(0)
    boots_g, boots_q = [], []
    X1 = np.vstack([ones, f]).T
    X2 = np.vstack([ones, f, f ** 2]).T
    for _ in range(N_RUN_BOOT):
        s = rng.integers(0, n, n)
        c1, *_ = np.linalg.lstsq(X1[s], y[s], rcond=None)
        c2, *_ = np.linalg.lstsq(X2[s], y[s], rcond=None)
        boots_g.append(c1[1])
        boots_q.append(c2[2])
    g_lo, g_hi = np.percentile(boots_g, [2.5, 97.5])
    q_lo, q_hi = np.percentile(boots_q, [2.5, 97.5])

    span = float(np.exp(y.max()) / np.exp(y.min()))
    return {"gamma": gamma, "gamma_ci95": [float(g_lo), float(g_hi)],
            "ln_b": lnb, "b": float(math.exp(lnb)), "r2": r2, "n_runs": n,
            "f_range": [float(f.min()), float(f.max())],
            "t_mem_span": span, "form_comparison": comparison,
            "curvature_q": float(cq[2]), "curvature_q_ci95": [float(q_lo), float(q_hi)]}


def fit_gen_law(cells: list[dict], primes: Iterable[int], *,
                windowed: bool = True) -> dict[str, Any]:
    primes = tuple(primes)
    pset = set(primes)
    by_p: dict[int, list[dict]] = {}
    for c in cells:
        if c["in_window"] or not windowed:
            by_p.setdefault(c["p"], []).append(c)
    sel = [c for p in primes for c in by_p.get(p, [])]  # multiplicity respected
    lp = np.log(np.array([c["p"] for c in sel], dtype=float))
    lP = np.log(np.array([c["param_count"] for c in sel]))
    y = np.log(np.array([c["t_gen"] for c in sel]))
    ones = np.ones_like(y)

    coef, r2, _ = _ols(np.vstack([ones, lp, lP]).T, y)
    lnA, a, beta = float(coef[0]), float(coef[1]), float(-coef[2])

    per_prime = {}
    for p in sorted(pset):
        pc = [c for c in sel if c["p"] == p]
        if len(pc) < 3:
            continue
        x = np.log(np.array([c["param_count"] for c in pc]))
        yy = np.log(np.array([c["t_gen"] for c in pc]))
        cf, r2p, rssp = _ols(np.vstack([np.ones_like(x), x]).T, yy)
        df = len(pc) - 2
        sig2 = rssp / df if df > 0 else float("nan")
        xtx = np.linalg.inv(np.vstack([np.ones_like(x), x]).T.T @
                            np.vstack([np.ones_like(x), x]).T)
        se = float(np.sqrt(sig2 * xtx[1, 1]))
        per_prime[p] = {"beta": float(-cf[1]), "se": se, "r2": r2p, "n": len(pc)}

    drift = None
    if len(per_prime) >= 3:
        bp = np.array([v["beta"] for v in per_prime.values()])
        lpp = np.log10(np.array(sorted(per_prime.keys()), dtype=float))
        cd, _, rssd = _ols(np.vstack([np.ones_like(lpp), lpp]).T, bp)
        dfd = len(bp) - 2
        sig2 = rssd / dfd if dfd > 0 else float("nan")
        X = np.vstack([np.ones_like(lpp), lpp]).T
        se_d = float(np.sqrt(sig2 * np.linalg.inv(X.T @ X)[1, 1]))
        drift = {"slope_beta_per_dex_p": float(cd[1]), "se": se_d,
                 "beta_spread_sd": float(bp.std(ddof=1))}

    # Steps-unit consistency check: steps/epoch = ceil(n_train/BATCH) with
    # n_train = K/log2(p+2) = 0.5·p·(p−1)·tf; a_steps should sit near
    # a_epochs + 2 (the p² steps-per-epoch factor), i.e. near 0.
    p_arr = np.array([c["p"] for c in sel], dtype=float)
    spe = np.ceil(0.5 * p_arr * (p_arr - 1) / BATCH)
    coef_s, r2_s, _ = _ols(np.vstack([ones, lp, lP]).T, y + np.log(spe))
    steps_check = {"a_steps": float(coef_s[1]), "beta_steps": float(-coef_s[2]),
                   "r2": r2_s}

    return {"lnA": lnA, "a": a, "beta": beta, "r2": r2, "n_cells": len(sel),
            "window_f": [GEN_F_LO, GEN_F_HI],
            "per_prime_beta": per_prime, "beta_drift": drift,
            "steps_unit_check": steps_check}


def fit_gen_law_fixed_beta(cells, primes, beta_fixed: float) -> dict[str, Any]:
    pset = set(primes)
    sel = [c for c in cells if c["p"] in pset and c["in_window"]]
    lp = np.log(np.array([c["p"] for c in sel], dtype=float))
    lP = np.log(np.array([c["param_count"] for c in sel]))
    y = np.log(np.array([c["t_gen"] for c in sel])) + beta_fixed * lP
    coef, _, _ = _ols(np.vstack([np.ones_like(y), lp]).T, y)
    return {"lnA": float(coef[0]), "a": float(coef[1]), "beta": beta_fixed}


# --------------------------------------------------------------------------- #
# Crossing / forecast
# --------------------------------------------------------------------------- #

def solve_crossing(p: int, K: float, C: float, mem: dict, gen: dict
                   ) -> Optional[float]:
    """First root (in log10 P) of ln T_mem(P) = ln T_gen(P)."""
    lnb, gamma = mem["ln_b"], mem["gamma"]
    lnA, a, beta = gen["lnA"], gen["a"], gen["beta"]

    def h(L: float) -> float:
        P = 10.0 ** L
        return (lnb + gamma * K / (C * P)) - (lnA + a * math.log(p) - beta * math.log(P))

    grid = np.arange(3.0, 9.0, 0.01)
    Pg = 10.0 ** grid
    vals = (lnb + gamma * K / (C * Pg)) - (lnA + a * math.log(p)
                                           - beta * np.log(Pg))
    sign_change = np.where((vals[:-1] > 0) & (vals[1:] <= 0))[0]
    if sign_change.size == 0:
        return None
    i = int(sign_change[0])
    from scipy.optimize import brentq
    return float(brentq(h, grid[i], grid[i + 1], xtol=1e-6))


def empirical_onset(records: list[dict], seeds: Optional[set] = None
                    ) -> Optional[float]:
    pairs = [(r["x"], r["delay"]) for r in records
             if seeds is None or r["seed"] in seeds]
    return aggregate.find_grokking_onset(aggregate.min_delay_curve(pairs))


def full_pipeline(data: dict, primes: tuple, forecast_ps: tuple, *,
                  light: bool = False) -> Optional[dict[str, Any]]:
    """Fit both laws on `primes`, calibrate Δ̂, forecast `forecast_ps`.
    Returns None when a bootstrap replicate is degenerate."""
    try:
        mem = fit_mem_law(data["speed_pts"], primes, light=light)
        gen = fit_gen_law(data["tgen_cells"], primes)
    except (np.linalg.LinAlgError, ValueError):
        return None
    C, K_of = data["C"], data["K_of"]

    resid = []
    per_prime = {}
    for p in primes:
        Lc = solve_crossing(p, K_of(p), C, mem, gen)
        Pon = empirical_onset(data["delay_records"].get(p, []))
        if Lc is None or Pon is None:
            continue
        r = math.log10(Pon) - Lc
        resid.append(r)
        per_prime[p] = {"log10_P_cross_fit": Lc, "P_onset": Pon, "resid_dex": r}
    if len(resid) < 3:
        return None
    delta_hat = float(np.median(resid))

    fc = {}
    for p in forecast_ps:
        Lc = solve_crossing(p, K_of(p), C, mem, gen)
        fc[p] = None if Lc is None else Lc + delta_hat
    return {"mem": mem, "gen": gen, "delta_hat": delta_hat,
            "per_prime": per_prime, "forecast_log10": fc}


# --------------------------------------------------------------------------- #
# Uncertainty
# --------------------------------------------------------------------------- #

def prime_bootstrap(data: dict, forecast_ps: tuple) -> dict[str, Any]:
    rng = np.random.default_rng(1)
    primes = np.array(CENTRAL_PRIMES)
    draws: dict[int, list[float]] = {p: [] for p in forecast_ps}
    beta_draws, gamma_draws, delta_draws = [], [], []
    n_degenerate = 0
    for _ in range(N_PRIME_BOOT):
        sample = tuple(rng.choice(primes, size=len(primes), replace=True))
        res = full_pipeline(data, sample, forecast_ps, light=True)
        if res is None:
            n_degenerate += 1
            continue
        for p in forecast_ps:
            if res["forecast_log10"][p] is not None:
                draws[p].append(res["forecast_log10"][p])
        beta_draws.append(res["gen"]["beta"])
        gamma_draws.append(res["mem"]["gamma"])
        delta_draws.append(res["delta_hat"])
    def _pct(vals):
        return list(np.percentile(vals, [2.5, 97.5])) if len(vals) else None

    out = {"n_boot": N_PRIME_BOOT, "n_degenerate": n_degenerate,
           "beta_ci95": _pct(beta_draws),
           "gamma_ci95": _pct(gamma_draws),
           "delta_hat_ci95": _pct(delta_draws)}
    for p in forecast_ps:
        d = np.array(draws[p])
        out[f"p{p}"] = {"n": len(d), "ci95_log10": _pct(d),
                        "sd_log10": float(d.std(ddof=1)) if len(d) > 1 else None}
    return out


def seed_noise_floor(data: dict) -> dict[str, Any]:
    rng = np.random.default_rng(2)
    per_prime = {}
    for p, recs in data["delay_records"].items():
        if p not in CENTRAL_PRIMES:
            continue
        seeds = sorted({r["seed"] for r in recs if r["seed"] is not None})
        if len(seeds) <= SEED_SUBSAMPLE_M:
            continue
        vals, n_none = [], 0
        for _ in range(N_SEED_DRAWS):
            sub = set(rng.choice(seeds, size=SEED_SUBSAMPLE_M, replace=False))
            on = empirical_onset(recs, seeds=sub)
            if on is None:
                n_none += 1
            else:
                vals.append(math.log10(on))
        per_prime[p] = {"sd_log10": float(np.std(vals, ddof=1)) if len(vals) > 1
                        else None,
                        "n_none": n_none, "n_seeds": len(seeds)}
    sds = [v["sd_log10"] for v in per_prime.values() if v["sd_log10"] is not None]
    return {"per_prime": per_prime,
            "median_sd_log10": float(np.median(sds)) if sds else None,
            "subsample_m": SEED_SUBSAMPLE_M, "n_draws": N_SEED_DRAWS}


# --------------------------------------------------------------------------- #
# Width grid / epoch budget for the forecast sweep
# --------------------------------------------------------------------------- #

def param_count_table(p: int, dims: Iterable[int]) -> dict[int, int]:
    from grokking_capacity.models import build_model, count_parameters
    out = {}
    for d in dims:
        m = build_model(depth=2, dim=int(d), heads=1, p=p, dropout=0.2)
        out[int(d)] = int(count_parameters(m))
    return out


def epoch_budget_check(p: int, dims: list[int], mem: dict, gen: dict,
                       C: float, K: float, pred_onset_log10: float
                       ) -> list[dict]:
    """Sub-onset widths must certify ΔE = 0, which needs val to saturate:
    the binding quantity is T_gen. Super-onset widths must certify
    ΔE > 0, which needs only train to saturate (a val-censored run still
    counts as non-zero delay): the binding quantity is E_train ≈
    T_mem / 0.70. Both must clear the epoch cap with 2× margin."""
    pcs = param_count_table(p, dims)
    rows = []
    for d, P in pcs.items():
        t_gen = math.exp(gen["lnA"] + gen["a"] * math.log(p)
                         - gen["beta"] * math.log(P))
        t_mem = math.exp(mem["ln_b"] + mem["gamma"] * K / (C * P))
        e_train = t_mem / PROXY_RATIO
        sub_onset = math.log10(P) < pred_onset_log10
        binding = t_gen if sub_onset else e_train
        rows.append({"dim": d, "param_count": P,
                     "regime": "sub-onset" if sub_onset else "super-onset",
                     "pred_T_gen": round(t_gen), "pred_T_mem": round(t_mem),
                     "pred_E_train": round(e_train),
                     "binding_epochs": round(binding),
                     "clears_cap_2x": binding * 2 < EPOCH_CAP})
    return rows


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run(config: str, db: str, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    view = ConfigView.from_yaml(config, db_path=db)
    data = collect(view)

    point = full_pipeline(data, CENTRAL_PRIMES,
                          (HELD_OUT_PRIME,) + FORECAST_PRIMES)
    if point is None:
        raise RuntimeError("point estimate pipeline returned degenerate result")

    # Window sensitivity + β sensitivity (pre-committed reporting).
    gen_nowin = fit_gen_law(data["tgen_cells"], CENTRAL_PRIMES, windowed=False)
    sens = {}
    for sign in (+1, -1):
        beta_p = point["gen"]["beta"] + sign * BETA_PERTURB
        gen_p = fit_gen_law_fixed_beta(data["tgen_cells"], CENTRAL_PRIMES, beta_p)
        resid = []
        for p in CENTRAL_PRIMES:
            Lc = solve_crossing(p, data["K_of"](p), data["C"], point["mem"], gen_p)
            Pon = (data["delay_records"].get(p) and
                   empirical_onset(data["delay_records"][p]))
            if Lc is not None and Pon:
                resid.append(math.log10(Pon) - Lc)
        dh = float(np.median(resid))
        L197 = solve_crossing(197, data["K_of"](197), data["C"], point["mem"], gen_p)
        sens[f"beta_{'+' if sign > 0 else '-'}{BETA_PERTURB}"] = (
            None if L197 is None else L197 + dh)
    base197 = point["forecast_log10"][197]  # None ⇒ predicted divergence
    beta_sensitivity = {
        "d_log10P197_per_+0.1beta": (None if sens.get("beta_+0.1") is None
                                     else sens["beta_+0.1"] - base197),
        "d_log10P197_per_-0.1beta": (None if sens.get("beta_-0.1") is None
                                     else sens["beta_-0.1"] - base197),
        "raw": sens}

    boot = prime_bootstrap(data, (HELD_OUT_PRIME,) + FORECAST_PRIMES)
    noise = seed_noise_floor(data)
    sigma_est = noise["median_sd_log10"] or 0.0

    # Divergence prime p*: h_min(p) = 0, i.e. the prime beyond which the
    # fitted crossing ceases to exist — the crossing model's prediction is
    # that grokking onset diverges there. Bootstrap CI over primes.
    def _pstar(mem, gen) -> Optional[float]:
        def hmin(pp: float) -> float:
            K = data["K_of"](int(round(pp)))
            L = np.arange(3.5, 8.5, 0.01)
            P = 10.0 ** L
            h = (mem["ln_b"] + mem["gamma"] * K / (data["C"] * P)) \
                - (gen["lnA"] + gen["a"] * math.log(pp) - gen["beta"] * np.log(P))
            return float(h.min())
        from scipy.optimize import brentq
        lo, hi = 140.0, 600.0
        try:
            if hmin(lo) < 0 < hmin(hi):
                return float(brentq(hmin, lo, hi, xtol=0.5))
        except ValueError:
            pass
        return None

    pstar_point = _pstar(point["mem"], point["gen"])
    rng = np.random.default_rng(3)
    primes_arr = np.array(CENTRAL_PRIMES)
    pstar_draws, finite197, finite251, nb = [], 0, 0, 0
    for _ in range(400):
        sample = tuple(rng.choice(primes_arr, size=len(primes_arr), replace=True))
        res = full_pipeline(data, sample, (197, 251), light=True)
        if res is None:
            continue
        nb += 1
        ps = _pstar(res["mem"], res["gen"])
        if ps is not None:
            pstar_draws.append(ps)
        finite197 += res["forecast_log10"][197] is not None
        finite251 += res["forecast_log10"][251] is not None
    divergence = {
        "p_star_point": pstar_point,
        "p_star_ci95": (list(np.percentile(pstar_draws, [2.5, 97.5]))
                        if len(pstar_draws) > 10 else None),
        "p_star_n_draws": len(pstar_draws), "n_boot": nb,
        "P_finite_onset_197": finite197 / nb if nb else None,
        "P_finite_onset_251": finite251 / nb if nb else None,
    }

    # Rival model: pure power law log10 P_onset = c0 + s·log10 p fitted on
    # the ten central onsets, extrapolated to 149/197/251 (prime bootstrap).
    def _null_fit(primes):
        xs, ys = [], []
        for p in primes:
            on = empirical_onset(data["delay_records"].get(p, []))
            if on:
                xs.append(math.log10(p)); ys.append(math.log10(on))
        if len(set(xs)) < 2:
            return None
        cf, r2n, _ = _ols(np.vstack([np.ones(len(xs)), xs]).T,
                          np.array(ys))
        return float(cf[0]), float(cf[1]), r2n

    c0, s, r2n = _null_fit(CENTRAL_PRIMES)
    null_draws = {149: [], 197: [], 251: []}
    for _ in range(2000):
        sample = tuple(rng.choice(primes_arr, size=len(primes_arr), replace=True))
        nf = _null_fit(sample)
        if nf is None:
            continue
        for p in null_draws:
            null_draws[p].append(nf[0] + nf[1] * math.log10(p))
    null_model = {"slope": s, "intercept": c0, "r2": r2n}
    for p, d in null_draws.items():
        arr = np.array(d)
        half_fit = float(np.diff(np.percentile(arr, [2.5, 97.5]))[0]) / 2
        half_total = math.sqrt(half_fit ** 2 + (1.96 * sigma_est) ** 2)
        centre = c0 + s * math.log10(p)
        null_model[f"p{p}"] = {
            "log10_P_onset": centre, "P_onset": 10 ** centre,
            "fit_ci95_log10": list(np.percentile(arr, [2.5, 97.5])),
            "total_band_log10": [centre - half_total, centre + half_total],
            "total_half_band_dex": half_total}

    band = {"point_log10": base197,
            "crossing_model_197": ("diverged" if base197 is None
                                   else {"P_onset_pred": 10 ** base197}),
            "estimator_sd_dex": sigma_est,
            "note": ("crossing model predicts no finite onset at p=197; "
                     "discrimination vs the power-law null replaces the "
                     "point-band gate"),
            "gate_dex": BAND_GATE_DEX,
            "null_total_half_band_dex":
                null_model["p197"]["total_half_band_dex"],
            "gate_pass_null_band":
                null_model["p197"]["total_half_band_dex"] <= BAND_GATE_DEX}

    # Held-out p=149 dry run of the frozen pipeline.
    Pon149 = empirical_onset(data["delay_records"].get(HELD_OUT_PRIME, []))
    check149 = {"P_onset_empirical": Pon149,
                "pred_log10": point["forecast_log10"][HELD_OUT_PRIME],
                "resid_dex": (None if (Pon149 is None or
                                       point["forecast_log10"][HELD_OUT_PRIME] is None)
                              else math.log10(Pon149)
                              - point["forecast_log10"][HELD_OUT_PRIME]),
                "n_records": len(data["delay_records"].get(HELD_OUT_PRIME, []))}

    # Pipeline (empirical-curve) crossings vs fitted-law crossings.
    crosscheck = {p: {"pipeline_P_cross": data["pipeline_cross"].get(p),
                      "fitted_log10_P_cross":
                          point["per_prime"].get(p, {}).get("log10_P_cross_fit")}
                  for p in CENTRAL_PRIMES}

    # Planned p=197 width grid: the crossing model predicts no onset, the
    # null predicts one near its centre — so the grid must cover the null's
    # band densely and extend far above it (to ~8× the null's upper edge)
    # so an absence-of-grokking verdict is meaningful. From ~0.3× the
    # null's lower band edge up to ~8× its upper edge, 16 log-spaced dims
    # on the d % 4 == 0 lattice.
    null_lo, null_hi = null_model["p197"]["total_band_log10"]
    pcs_all = param_count_table(197, range(16, 1001, 4))
    lo_P, hi_P = 10 ** (null_lo - 0.5), 10 ** (null_hi + 0.9)
    dims_in = [d for d, P in pcs_all.items() if lo_P <= P <= hi_P]
    grid = sorted(set(np.array(dims_in)[
        np.unique(np.round(np.linspace(0, len(dims_in) - 1, 16)).astype(int))]
        .tolist())) if dims_in else []
    budget = epoch_budget_check(197, grid, point["mem"], point["gen"],
                                data["C"], data["K_of"](197),
                                null_model["p197"]["log10_P_onset"])

    mem_full = fit_mem_law(data["speed_pts"], CENTRAL_PRIMES, windowed=False)

    results = {
        "pre_committed": "docs/papers/2026 Grokking Capacity/theory/theory-trunk.md §0 (+ amendments §0b, §0c)",
        "C_convention": data["C"],
        "gamma_per_bit_per_param": point["mem"]["gamma"] / data["C"],
        "mem_law": point["mem"],
        "mem_law_full_range": mem_full,
        "gen_law": point["gen"],
        "gen_law_no_window": {k: gen_nowin[k] for k in
                              ("lnA", "a", "beta", "r2", "n_cells")},
        "delta_hat_dex": point["delta_hat"],
        "per_prime": point["per_prime"],
        "crosscheck_pipeline_vs_fitted": crosscheck,
        "forecast_log10": point["forecast_log10"],
        "beta_sensitivity": beta_sensitivity,
        "prime_bootstrap": boot,
        "seed_noise_floor": noise,
        "divergence": divergence,
        "null_powerlaw": null_model,
        "band_p197": band,
        "held_out_p149": check149,
        "p197_width_grid": grid,
        "p197_epoch_budget": budget,
    }
    with open(out_dir / "forecast_onset.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    return results


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/central.yaml")
    ap.add_argument("--db", default="runs.db")
    ap.add_argument("--out", default="results/theory")
    args = ap.parse_args(argv)
    res = run(args.config, args.db, Path(args.out))
    m, g = res["mem_law"], res["gen_law"]
    print(f"gamma = {m['gamma']:.2f} [{m['gamma_ci95'][0]:.2f}, "
          f"{m['gamma_ci95'][1]:.2f}] (C={res['C_convention']});  "
          f"gamma/C = {res['gamma_per_bit_per_param']:.2f} nat/bpp;  "
          f"b = {m['b']:.1f};  R² = {m['r2']:.3f}")
    print(f"beta = {g['beta']:.3f};  a = {g['a']:.2f};  R² = {g['r2']:.3f};  "
          f"drift = {g['beta_drift']}")
    print(f"delta_hat = {res['delta_hat_dex']:.3f} dex")
    d = res["divergence"]
    print(f"p* = {d['p_star_point']} ci95 {d['p_star_ci95']}  "
          f"P(finite onset | 197) = {d['P_finite_onset_197']}")
    n = res["null_powerlaw"]
    print(f"null: slope {n['slope']:.2f}, p197 onset "
          f"{n['p197']['P_onset']:,.0f} band(log10) "
          f"{[round(v, 3) for v in n['p197']['total_band_log10']]}")
    print(f"band gate (null-band): {res['band_p197']['gate_pass_null_band']} "
          f"(σ_est = {res['band_p197']['estimator_sd_dex']:.3f})")
    print(f"held-out p=149: {res['held_out_p149']}")
    print(f"p197 grid: {res['p197_width_grid']}")
    for row in res["p197_epoch_budget"]:
        print("  ", row)


if __name__ == "__main__":
    main()
