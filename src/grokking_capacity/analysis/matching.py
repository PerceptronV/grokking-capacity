"""Pair groks runs with their corresponding speed runs via wallow queries.

Replaces the legacy ResultsIndex-based matching: every query now uses the
wallow DSL so field names are validated and the SQLite engine narrows results
before Python sees them.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any, Optional

from wallow import F

from ..models import TransformerTorch
from ..registry import get_store
from ..utils.compute import compute_dataset_size_bits


def compute_n_equiv(p: int, operation: str, train_fraction: float) -> tuple[int, float]:
    """Return (n_equiv, K_mem_bits)."""
    return compute_dataset_size_bits(p, operation, train_fraction)


def get_param_count(dim: int, depth: int, heads: int, p: int, dropout: float = 0.0) -> int:
    model = TransformerTorch(
        depth=depth, dim=dim, heads=heads,
        n_tokens=p + 2, seq_len=4, dropout=dropout,
    )
    return sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)


def find_dims_for_param_targets(
    targets: list,
    *,
    depth: int,
    heads: int,
    p: int,
    dropout: float = 0.0,
    tolerance: float = 0.15,
    dim_search_range=range(8, 512, 2),
) -> list[tuple[int, int]]:
    dim_to_params = {d: get_param_count(d, depth, heads, p, dropout) for d in dim_search_range}
    out: list[tuple[int, int]] = []
    for target in targets:
        best_dim, best_params = min(dim_to_params.items(), key=lambda kv: abs(kv[1] - target))
        if abs(best_params - target) / target <= tolerance:
            out.append((best_dim, best_params))
    return out


@dataclass
class ExperimentMatch:
    groks_run_uuid: str
    speed_run_uuid: str
    capacity_run_uuid: Optional[str]

    param_count_groks: int
    param_count_speed: int
    dataset_bits: float
    n_equiv: int
    capacity_constant: float
    capacity_fraction: float

    match_type: str               # "exact" | "param_matched"
    param_count_mismatch: float

    p: int
    operation: str
    train_fraction: float
    weight_decay_groks: float
    weight_decay_speed: float
    architecture_family: str

    groks_npz_path: str
    speed_npz_path: str


def _row_to_dict(row: Any, extra: tuple[str, ...] = ()) -> dict[str, Any]:
    """Pull everything we need off a SQLAlchemy Run row."""
    fields = (
        "experiment_type", "p", "operation", "train_fraction", "split_type",
        "dataset_type", "n_samples", "dim", "depth", "heads", "dropout",
        "init_scale", "lr", "weight_decay", "seed", "architecture_family",
        "param_count", "uuid", "npz_path", "n_equiv", "dataset_bits",
        "saturation_step", "saturation_epoch", "saturated", "final_acc",
        "final_loss", "final_train_acc", "final_val_acc", "grokking_epoch",
        "total_bits_memorized", "final_bits_per_example",
    ) + extra
    return {f: getattr(row, f, None) for f in fields}


def _query_completed(store, **filters) -> list[dict[str, Any]]:
    expr = F("status") == "completed"
    for k, v in filters.items():
        expr = expr & (F(k) == v)
    return [_row_to_dict(r) for r in store.where(expr).all()]


def build_match_table(
    *,
    db_path: str | None = None,
    primes: list[int] | None = None,
    weight_decays: list[float] | None = None,
    depths: list[int] | None = None,
    heads: list[int] | None = None,
    dropouts: list[float] | None = None,
    init_scales: list[float] | None = None,
    capacity_constant: float,
    measure_capacity: bool = True,
    capacity_source: str = "",
    param_tolerance: float = 0.05,
    n_samples_tolerance: int = 2,
) -> list[ExperimentMatch]:
    """Pair completed groks runs with completed speed runs.

    Pairs by (p, operation, train_fraction, depth, heads, dropout, init_scale,
    seed, n_samples ≈ n_equiv, param_count ≈ groks.param_count). weight_decay
    is deliberately NOT used in the join — the values for the two sides are
    recorded separately so the confound is auditable.
    """
    from .capacity_constant import measure_capacity_constant  # local import: avoids cycle

    store = get_store(db_path)

    base = F("status") == "completed"
    if primes:
        base = base & F("p").in_(primes)

    groks_rows = store.where(base & (F("experiment_type") == "groks")).all()
    speed_rows = store.where(base & (F("experiment_type") == "speed")).all()

    groks = [_row_to_dict(r) for r in groks_rows]
    speed = [_row_to_dict(r) for r in speed_rows]

    # Index speed rows by n_samples for quick neighbourhood lookup.
    speed_by_n: dict[int, list[dict]] = {}
    for s in speed:
        n = s.get("n_samples")
        if n is None:
            continue
        speed_by_n.setdefault(int(n), []).append(s)

    c_cache: dict[tuple, Optional[float]] = {}

    matches: list[ExperimentMatch] = []
    for g in groks:
        gp = g.get("param_count")
        if gp is None:
            continue
        n_equiv, K_mem = compute_n_equiv(g["p"], g["operation"], g["train_fraction"])

        cond = (
            int(g.get("depth") or 2),
            int(g.get("heads") or 1),
            float(g.get("weight_decay") or 1.0),
            float(g.get("dropout") or 0.2),
            float(g.get("init_scale") or 1.0),
        )
        if measure_capacity:
            if cond not in c_cache:
                c_cache[cond] = measure_capacity_constant(
                    db_path=db_path,
                    depth=cond[0], heads=cond[1],
                    weight_decay=cond[2], dropout=cond[3], init_scale=cond[4],
                )
            C = c_cache[cond] or capacity_constant
        else:
            C = capacity_constant

        candidates: list[dict] = []
        for delta in range(-n_samples_tolerance, n_samples_tolerance + 1):
            candidates.extend(speed_by_n.get(n_equiv + delta, []))

        for s in candidates:
            sp = s.get("param_count")
            if sp is None:
                continue
            mismatch = abs(gp - sp) / gp if gp > 0 else 0.0
            if mismatch > param_tolerance:
                continue
            if g.get("operation") != s.get("operation"):
                continue
            if g.get("train_fraction") != s.get("train_fraction"):
                continue
            if g.get("seed") != s.get("seed"):
                continue
            if g.get("depth") != s.get("depth") or g.get("heads") != s.get("heads"):
                continue
            if g.get("dropout") != s.get("dropout"):
                continue
            if g.get("init_scale") != s.get("init_scale"):
                continue
            cap_frac = K_mem / (C * gp) if gp > 0 else 0.0
            matches.append(ExperimentMatch(
                groks_run_uuid=g["uuid"] or "",
                speed_run_uuid=s["uuid"] or "",
                capacity_run_uuid=None,
                param_count_groks=int(gp),
                param_count_speed=int(sp),
                dataset_bits=float(K_mem),
                n_equiv=int(n_equiv),
                capacity_constant=float(C),
                capacity_fraction=float(cap_frac),
                match_type="exact" if mismatch == 0.0 else "param_matched",
                param_count_mismatch=float(mismatch),
                p=int(g["p"]),
                operation=str(g["operation"]),
                train_fraction=float(g["train_fraction"]),
                weight_decay_groks=float(g.get("weight_decay") or 0.0),
                weight_decay_speed=float(s.get("weight_decay") or 0.0),
                architecture_family=str(g.get("architecture_family") or "transformer_gated"),
                groks_npz_path=str(g.get("npz_path") or ""),
                speed_npz_path=str(s.get("npz_path") or ""),
            ))
    return matches


def save_match_table(matches: list[ExperimentMatch], path: str) -> None:
    with open(path, 'w') as f:
        json.dump([dataclasses.asdict(m) for m in matches], f, indent=2)


def load_match_table(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)
