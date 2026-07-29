"""Threshold-sensitivity analysis of the pipeline quantities.

The pipeline's three scalar outputs per prime — T_mem(P), T_gen(P), and the
grokking onset P_onset — all depend on accuracy thresholds. This tool
recomputes every quantity directly from the raw traces under a grid of
threshold conventions (tau_train; tau_gen; tau_delay_val) and reports how the
headline statistic, median log10(P_onset / P_cross), moves.

Three analyses:

* E1 — threshold grid: recompute T_mem, T_gen, min-delay onset, and the
  mem/gen intersection P_cross per prime for each threshold triple.
* E2 — estimator variants: at the anchor thresholds, swap the per-cell
  seed reduction of the delay (min / median / mean) before onset detection.
* E3 — units: at the anchor thresholds, fit log(T_mem) = a*f + log(b)
  pooled over primes with T_mem measured in epochs vs in optimiser steps.

All first-crossing epochs are extracted from the .npz traces in a single
pass (one read per file, vectorised over all thresholds) and cached
in-memory; ``--cache`` optionally persists the extraction table so repeated
invocations skip the trace scan.

Usage:
    python -m grokking_capacity.analysis.threshold_grid \
        --db /path/to/runs.db --data-root /path/to/data --out results/threshold_grid
"""
from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

from .. import consts
from .aggregate import find_grokking_onset, find_intersection, min_delay_curve
from .matching import compute_n_equiv

# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------

# Accuracy thresholds (percent) any grid row may reference. Crossing epochs
# for all of these are extracted in the single trace pass.
THRESHOLDS: tuple[float, ...] = (95.0, 98.0, 99.0)

# (tau_train, tau_gen, tau_delay_val) triples. First row is the published
# anchor convention.
GRID: tuple[tuple[float, float, float], ...] = (
    (99.0, 99.0, 98.0),  # anchor
    (95.0, 95.0, 95.0),
    (98.0, 98.0, 98.0),
    (99.0, 99.0, 99.0),
    (99.0, 99.0, 95.0),
)

ANCHOR = GRID[0]

BASELINE_FILTER = (
    "operation = '/' AND train_fraction = 0.5 AND weight_decay = 1.0 "
    "AND dropout = 0.2 AND lr = 0.001 AND init_scale = 1.0 "
    "AND depth = 2 AND heads = 1 AND status = 'completed'"
)


# ---------------------------------------------------------------------------
# Trace extraction (one pass over all npz files)
# ---------------------------------------------------------------------------

def _first_crossings(acc: np.ndarray) -> dict[float, Optional[int]]:
    """1-indexed first epoch with acc >= tau, for every tau in THRESHOLDS."""
    taus = np.asarray(THRESHOLDS)
    mask = acc[None, :] >= taus[:, None]
    reached = mask.any(axis=1)
    idx = mask.argmax(axis=1) + 1  # 1-indexed
    return {
        float(t): (int(i) if r else None)
        for t, r, i in zip(THRESHOLDS, reached, idx)
    }


def _extract_one(args: tuple[str, str, dict]) -> Optional[dict]:
    """Worker: read one trace.npz, return per-seed crossing epochs."""
    exp_type, npz_path, meta = args
    try:
        with np.load(npz_path) as d:
            if exp_type == "speed":
                acc = np.asarray(d["train_acc_trace"], dtype=float)
                steps = np.asarray(d["steps_trace"], dtype=np.int64)
                firsts = _first_crossings(acc)
                steps_at = {
                    t: (int(steps[e - 1]) if e is not None else None)
                    for t, e in firsts.items()
                }
                return {**meta, "mem_first": firsts, "mem_steps": steps_at,
                        "n_epochs": int(acc.size)}
            else:  # groks
                train = np.asarray(d["train_acc"], dtype=float)
                val = np.asarray(d["val_acc"], dtype=float)
                return {**meta,
                        "train_first": _first_crossings(train),
                        "val_first": _first_crossings(val),
                        "n_epochs": int(val.size)}
    except (FileNotFoundError, OSError, KeyError):
        return None


def load_records(
    db: Path, data_root: Path, *, max_dim: int, workers: int,
    cache: Optional[Path] = None,
) -> tuple[list[dict], list[dict], int]:
    """Return (speed_records, groks_records, n_missing) for the baseline arch."""
    if cache is not None and cache.exists():
        with open(cache, "rb") as fh:
            payload = pickle.load(fh)
        return payload["speed"], payload["groks"], payload["n_missing"]

    con = sqlite3.connect(str(db))
    con.row_factory = sqlite3.Row
    jobs: list[tuple[str, str, dict]] = []
    for et in ("speed", "groks"):
        rows = con.execute(
            f"SELECT uuid, p, dim, seed, param_count, dataset_bits, "
            f"saturation_epoch, saturated, grokking_epoch "
            f"FROM runs WHERE experiment_type = ? AND dim <= ? AND {BASELINE_FILTER}",
            (et, max_dim),
        ).fetchall()
        for r in rows:
            meta = {
                "p": int(r["p"]), "dim": int(r["dim"]), "seed": int(r["seed"]),
                "param_count": int(r["param_count"]),
                "dataset_bits": r["dataset_bits"],
                "saturation_epoch": r["saturation_epoch"],
                "saturated": r["saturated"],
                "grokking_epoch": r["grokking_epoch"],
            }
            path = data_root / et / r["uuid"] / "trace.npz"
            jobs.append((et, str(path), meta))
    con.close()

    speed_recs: list[dict] = []
    groks_recs: list[dict] = []
    n_missing = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for job, rec in zip(jobs, pool.map(_extract_one, jobs, chunksize=64)):
            if rec is None:
                n_missing += 1
            elif "mem_first" in rec:
                speed_recs.append(rec)
            else:
                groks_recs.append(rec)

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as fh:
            pickle.dump({"speed": speed_recs, "groks": groks_recs,
                         "n_missing": n_missing}, fh)
    return speed_recs, groks_recs, n_missing


# ---------------------------------------------------------------------------
# Curve construction
# ---------------------------------------------------------------------------

def _seed_mean_curve(
    recs: list[dict], key: str, tau: float,
) -> tuple[dict[float, float], int]:
    """{param_count: seed-mean first-crossing epoch}; censored seeds dropped.

    Returns (curve, n_censored_seeds). Cells whose every seed is censored
    contribute no point.
    """
    bins: dict[float, list[float]] = {}
    censored = 0
    for r in recs:
        e = r[key][tau]
        if e is None:
            censored += 1
            continue
        bins.setdefault(float(r["param_count"]), []).append(float(e))
    curve = {k: float(np.mean(v)) for k, v in sorted(bins.items())}
    return curve, censored


def _delays(
    groks: list[dict], tau_train: float, tau_val: float,
) -> tuple[list[tuple[float, float]], int, int]:
    """Per-seed (param_count, delay) pairs, mirroring `compute_delays`.

    Seeds whose train accuracy never reaches tau_train are dropped
    (counted). Seeds whose val accuracy never reaches tau_val get the floor
    delay n_epochs - train_epoch (0-indexed convention, as in
    `compute_delays`); these are counted separately.
    """
    out: list[tuple[float, float]] = []
    train_censored = 0
    val_floor = 0
    for r in groks:
        t = r["train_first"][tau_train]
        if t is None:
            train_censored += 1
            continue
        v = r["val_first"][tau_val]
        if v is not None:
            delay = max(0, v - t)
        else:
            val_floor += 1
            delay = max(0, r["n_epochs"] - (t - 1))
        out.append((float(r["param_count"]), float(delay)))
    return out, train_censored, val_floor


def _reduce_delays(
    delays: list[tuple[float, float]], how: str,
) -> dict[float, float]:
    """Per-param-count seed reduction of delays: min / median / mean."""
    if how == "min":
        return min_delay_curve(delays)
    bins: dict[float, list[float]] = {}
    for pc, d in delays:
        bins.setdefault(float(pc), []).append(float(d))
    fn = np.median if how == "median" else np.mean
    return {k: float(fn(v)) for k, v in sorted(bins.items())}


def _per_prime(
    speed: list[dict], groks: list[dict],
    tau_train: float, tau_gen: float, tau_delay_val: float,
    *, delay_reduction: str = "min",
) -> dict[int, dict]:
    """Per-prime P_cross / P_onset / f_onset for one threshold triple."""
    primes = sorted({r["p"] for r in groks} | {r["p"] for r in speed})
    out: dict[int, dict] = {}
    for p in primes:
        sp = [r for r in speed if r["p"] == p]
        gk = [r for r in groks if r["p"] == p]
        mem_curve, mem_cens = _seed_mean_curve(sp, "mem_first", tau_train)
        gen_curve, gen_cens = _seed_mean_curve(gk, "val_first", tau_gen)
        delays, dtr_cens, dval_floor = _delays(gk, tau_train, tau_delay_val)
        onset = find_grokking_onset(_reduce_delays(delays, delay_reduction))
        cross = find_intersection(mem_curve, gen_curve)
        _, k_mem = compute_n_equiv(p, "/", 0.5)
        out[p] = {
            "P_cross": None if cross is None else cross[0],
            "cross_epochs": None if cross is None else cross[1],
            "P_onset": onset,
            "f_onset": None if onset is None else k_mem / (consts.C * onset),
            "n_cells_mem": len(mem_curve),
            "n_cells_gen": len(gen_curve),
            "censored_mem_seeds": mem_cens,
            "censored_gen_seeds": gen_cens,
            "censored_delay_train_seeds": dtr_cens,
            "delay_val_floor_seeds": dval_floor,
        }
    return out


def _summarise(per_prime: dict[int, dict]) -> dict:
    ratios = [
        np.log10(v["P_onset"] / v["P_cross"])
        for v in per_prime.values()
        if v["P_onset"] is not None and v["P_cross"] is not None
    ]
    f_onsets = [v["f_onset"] for v in per_prime.values() if v["f_onset"] is not None]
    return {
        "median_log10_onset_over_cross": float(np.median(ratios)) if ratios else None,
        "n_primes_with_both": len(ratios),
        "mean_f_onset": float(np.mean(f_onsets)) if f_onsets else None,
        "n_cells_mem": int(sum(v["n_cells_mem"] for v in per_prime.values())),
        "n_cells_gen": int(sum(v["n_cells_gen"] for v in per_prime.values())),
        "censored_mem_seeds": int(sum(v["censored_mem_seeds"] for v in per_prime.values())),
        "censored_gen_seeds": int(sum(v["censored_gen_seeds"] for v in per_prime.values())),
        "censored_delay_train_seeds": int(
            sum(v["censored_delay_train_seeds"] for v in per_prime.values())),
        "delay_val_floor_seeds": int(
            sum(v["delay_val_floor_seeds"] for v in per_prime.values())),
    }


# ---------------------------------------------------------------------------
# E1 — threshold grid
# ---------------------------------------------------------------------------

def _sanity_vs_annotations(speed: list[dict], groks: list[dict]) -> dict:
    """Median |recomputed - stored annotation| at the anchor thresholds."""
    tau_train, tau_gen, _ = ANCHOR
    mem_diffs = [
        abs(r["mem_first"][tau_train] - float(r["saturation_epoch"]))
        for r in speed
        if r["saturated"] and r["saturation_epoch"] is not None
        and r["mem_first"][tau_train] is not None
    ]
    gen_diffs = [
        abs(r["val_first"][tau_gen] - float(r["grokking_epoch"]))
        for r in groks
        if r["grokking_epoch"] is not None and r["val_first"][tau_gen] is not None
    ]
    return {
        "median_abs_Tmem_minus_saturation_epoch":
            float(np.median(mem_diffs)) if mem_diffs else None,
        "max_abs_Tmem_minus_saturation_epoch":
            float(np.max(mem_diffs)) if mem_diffs else None,
        "n_speed_compared": len(mem_diffs),
        "median_abs_Tgen_minus_grokking_epoch":
            float(np.median(gen_diffs)) if gen_diffs else None,
        "max_abs_Tgen_minus_grokking_epoch":
            float(np.max(gen_diffs)) if gen_diffs else None,
        "n_groks_compared": len(gen_diffs),
    }


def run_e1(speed: list[dict], groks: list[dict]) -> dict:
    rows = []
    for tau_train, tau_gen, tau_delay_val in GRID:
        pp = _per_prime(speed, groks, tau_train, tau_gen, tau_delay_val)
        rows.append({
            "tau_train": tau_train, "tau_gen": tau_gen,
            "tau_delay_val": tau_delay_val,
            "anchor": (tau_train, tau_gen, tau_delay_val) == ANCHOR,
            **_summarise(pp),
            "per_prime": {str(p): v for p, v in pp.items()},
        })
    return {"grid": rows, "sanity": _sanity_vs_annotations(speed, groks)}


# ---------------------------------------------------------------------------
# E2 — estimator variants (seed reduction of the delay)
# ---------------------------------------------------------------------------

def run_e2(speed: list[dict], groks: list[dict]) -> dict:
    tau_train, tau_gen, tau_delay_val = ANCHOR
    out = {}
    for how in ("min", "median", "mean"):
        pp = _per_prime(speed, groks, tau_train, tau_gen, tau_delay_val,
                        delay_reduction=how)
        out[how] = {
            "median_log10_onset_over_cross":
                _summarise(pp)["median_log10_onset_over_cross"],
            "n_primes_with_both": _summarise(pp)["n_primes_with_both"],
            "per_prime_P_onset": {str(p): v["P_onset"] for p, v in pp.items()},
        }
    return {"anchor_thresholds": list(ANCHOR), "seed_reduction": out}


# ---------------------------------------------------------------------------
# E3 — units: T_mem in epochs vs steps
# ---------------------------------------------------------------------------

def _fit_log_linear(f: np.ndarray, y: np.ndarray) -> dict:
    """Fit ln(y) = a*f + ln(b); return a, b, R^2."""
    ln_y = np.log(y)
    a, ln_b = np.polyfit(f, ln_y, 1)
    resid = ln_y - (a * f + ln_b)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((ln_y - ln_y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"a": float(a), "b": float(np.exp(ln_b)), "r2": float(r2),
            "n_points": int(y.size)}


def run_e3(speed: list[dict]) -> dict:
    tau_train = ANCHOR[0]
    # Seed-mean T_mem per (p, param_count) cell, in both units.
    cells: dict[tuple[int, float], dict[str, list[float]]] = {}
    censored = 0
    for r in speed:
        e = r["mem_first"][tau_train]
        if e is None or r["dataset_bits"] is None:
            censored += 1
            continue
        key = (r["p"], float(r["param_count"]))
        c = cells.setdefault(key, {"epochs": [], "steps": [],
                                   "bits": float(r["dataset_bits"])})
        c["epochs"].append(float(e))
        c["steps"].append(float(r["mem_steps"][tau_train]))

    p_arr, f_arr, ep_arr, st_arr = [], [], [], []
    for (p, pc), c in sorted(cells.items()):
        p_arr.append(p)
        f_arr.append(c["bits"] / (consts.C * pc))
        ep_arr.append(np.mean(c["epochs"]))
        st_arr.append(np.mean(c["steps"]))
    p_arr = np.array(p_arr)
    f_arr = np.array(f_arr)
    ep_arr = np.array(ep_arr)
    st_arr = np.array(st_arr)

    result = {
        "tau_train": tau_train,
        "capacity_constant": consts.C,
        "censored_or_missing_seeds": censored,
        "pooled": {
            "epochs": _fit_log_linear(f_arr, ep_arr),
            "steps": _fit_log_linear(f_arr, st_arr),
        },
        "per_prime": {},
    }
    for unit, y in (("epochs", ep_arr), ("steps", st_arr)):
        r2s = {}
        for p in sorted(set(p_arr.tolist())):
            m = p_arr == p
            if m.sum() >= 3:
                r2s[str(int(p))] = _fit_log_linear(f_arr[m], y[m])["r2"]
        result["per_prime"][unit] = {
            "r2": r2s,
            "r2_min": float(min(r2s.values())),
            "r2_max": float(max(r2s.values())),
        }
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _summary_table(e1: dict, e2: dict, e3: dict) -> str:
    lines = [
        "analysis,setting,median_log10_onset_over_cross,mean_f_onset,"
        "n_primes,n_cells_mem,n_cells_gen,censored_mem,censored_gen,"
        "censored_delay_train",
    ]
    for row in e1["grid"]:
        tag = f"tau=({row['tau_train']:g};{row['tau_gen']:g};{row['tau_delay_val']:g})"
        if row["anchor"]:
            tag += "[anchor]"
        lines.append(
            f"E1,{tag},{row['median_log10_onset_over_cross']:.4f},"
            f"{row['mean_f_onset']:.4f},{row['n_primes_with_both']},"
            f"{row['n_cells_mem']},{row['n_cells_gen']},"
            f"{row['censored_mem_seeds']},{row['censored_gen_seeds']},"
            f"{row['censored_delay_train_seeds']}"
        )
    for how, v in e2["seed_reduction"].items():
        lines.append(
            f"E2,delta-seed-{how},{v['median_log10_onset_over_cross']:.4f},,"
            f"{v['n_primes_with_both']},,,,,"
        )
    for unit in ("epochs", "steps"):
        fit = e3["pooled"][unit]
        pr = e3["per_prime"][unit]
        lines.append(
            f"E3,Tmem-{unit} pooled R2={fit['r2']:.4f} "
            f"(per-prime {pr['r2_min']:.4f}..{pr['r2_max']:.4f}),,,,,,,,"
        )
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", type=Path, required=True,
                    help="SQLite registry with the `runs` table")
    ap.add_argument("--data-root", type=Path, required=True,
                    help="Artefacts root: <root>/<experiment_type>/<uuid>/trace.npz")
    ap.add_argument("--out", type=Path, default=Path("results/threshold_grid"))
    ap.add_argument("--max-dim", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--cache", type=Path, default=None,
                    help="Optional pickle path for the trace-extraction table")
    args = ap.parse_args(argv)

    speed, groks, n_missing = load_records(
        args.db, args.data_root, max_dim=args.max_dim,
        workers=args.workers, cache=args.cache,
    )
    print(f"loaded {len(speed)} speed traces, {len(groks)} groks traces "
          f"({n_missing} missing/unreadable)")

    e1 = run_e1(speed, groks)
    e2 = run_e2(speed, groks)
    e3 = run_e3(speed)

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "n_speed_traces": len(speed), "n_groks_traces": len(groks),
        "n_missing": n_missing, "max_dim": args.max_dim,
        "baseline_filter": BASELINE_FILTER, "grid": [list(g) for g in GRID],
    }
    for name, payload in (("e1_threshold_grid", {**meta, **e1}),
                          ("e2_estimator_variants", {**meta, **e2}),
                          ("e3_units", {**meta, **e3})):
        with open(args.out / f"{name}.json", "w") as fh:
            json.dump(payload, fh, indent=2)
    (args.out / "summary.csv").write_text(_summary_table(e1, e2, e3))
    print(f"wrote {args.out}/e1_threshold_grid.json, e2_estimator_variants.json, "
          f"e3_units.json, summary.csv")


if __name__ == "__main__":
    main()
