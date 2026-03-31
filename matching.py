import dataclasses
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np

from utils import compute_dataset_size_bits


def compute_n_equiv(p: int, operation: str, train_fraction: float) -> tuple:
    """Return (n_equiv, K_mem_bits). Wraps utils.compute_dataset_size_bits."""
    return compute_dataset_size_bits(p, operation, train_fraction)


def get_param_count(dim: int, depth: int, heads: int, p: int, dropout: float = 0.0) -> int:
    """Compute parameter count by instantiating a CPU model."""
    from models import TransformerTorch
    model = TransformerTorch(
        depth=depth, dim=dim, heads=heads,
        n_tokens=p + 2, seq_len=4, dropout=dropout,
    )
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def find_dims_for_param_targets(
    targets: list,
    depth: int,
    heads: int,
    p: int,
    dropout: float = 0.0,
    tolerance: float = 0.15,
    dim_search_range=range(8, 512, 2),
) -> list:
    """Find (dim, actual_param_count) pairs closest to each target param count.

    Returns a list of (dim, actual_param_count) tuples, one per target that
    has a match within the given fractional tolerance.
    """
    dim_to_params = {d: get_param_count(d, depth, heads, p, dropout) for d in dim_search_range}

    results = []
    for target in targets:
        best_dim, best_params = min(dim_to_params.items(), key=lambda kv: abs(kv[1] - target))
        if abs(best_params - target) / target <= tolerance:
            results.append((best_dim, best_params))
    return results


@dataclass
class ExperimentMatch:
    """Records why two experiments (one groks, one speed) are comparable.

    This is the fundamental unit of the paper's analysis: a paired observation
    of (T_mem, T_gen) at a given model size and dataset complexity, with
    explicit provenance for how the pairing was made.
    """
    groks_run_id: str
    speed_run_id: str
    capacity_run_id: Optional[str]

    param_count_groks: int
    param_count_speed: int
    dataset_bits: float
    n_equiv: int
    capacity_constant: float
    capacity_fraction: float      # K_mem / (C_model * P)

    match_type: str               # "exact" or "param_matched"
    param_count_mismatch: float   # |P_groks - P_speed| / P_groks

    p: int
    operation: str
    train_fraction: float
    weight_decay_groks: float     # separate fields make the confound explicit
    weight_decay_speed: float
    architecture_family: str

    groks_npz_path: str
    speed_npz_path: str


def measure_capacity_constant(
    index,
    depth: int = 2,
    heads: int = 1,
    weight_decay: float = 1.0,
    dropout: float = 0.0,
    init_scale: float = 1.0,
) -> Optional[float]:
    """Fit C from capacity runs matching the given condition.

    Queries the ResultsIndex for capacity entries at the specified hyperparameters,
    collects the saturation point (max bits memorised) per param count, and fits
    the slope bits = C * params via linear regression.

    Returns None if fewer than 2 distinct param counts are found — the caller
    should fall back to a global constant in that case.
    """
    from plotting import estimate_capacity

    # Legacy runs predate init_scale — treat None as the default 1.0
    is_filter = (lambda x: x is None or x == init_scale) if init_scale == 1.0 else init_scale

    entries = index.query(
        experiment_type='capacity',
        depth=depth,
        heads=heads,
        weight_decay=weight_decay,
        dropout=dropout,
        init_scale=is_filter,
    )
    if not entries:
        return None

    by_param: dict = {}
    for entry in entries:
        pc = entry.get('model', {}).get('param_count')
        if pc is None:
            continue
        traces = index.load_traces(entry)
        bits = traces.get('total_bits_memorized') or traces.get('total_bits')
        if bits is None:
            continue
        max_bits = float(np.asarray(bits).max())
        if pc not in by_param or max_bits > by_param[pc]:
            by_param[pc] = max_bits

    saturation_points = sorted(by_param.items())
    if len(saturation_points) < 2:
        return None

    C, _intercept, _r2 = estimate_capacity(saturation_points)
    return C


def build_match_table(
    groks_entries: list,
    speed_entries: list,
    capacity_constant: float,
    capacity_index=None,
    capacity_source: str = "",
    param_tolerance: float = 0.05,
    n_samples_tolerance: int = 2,
) -> list:
    """Pair grokking and speed entries by (param_count, n_equiv).

    groks_entries and speed_entries are metadata dicts from ResultsIndex.
    For each groks entry, finds speed entries with matching n_samples and
    param_count within param_tolerance. Returns a list of ExperimentMatch.

    n_samples_tolerance: allow speed n_samples to differ from the computed
    n_equiv by up to this many samples. Needed for legacy data where a
    rounding discrepancy of ±1 sample exists in some primes.

    Note: matching ignores weight_decay — the mismatch is deliberately recorded
    in weight_decay_groks vs weight_decay_speed so it is auditable.
    """
    # Index speed entries by n_samples for fast lookup
    speed_by_n: dict = {}
    for se in speed_entries:
        n = se['data'].get('n_samples')
        if n is not None:
            speed_by_n.setdefault(n, []).append(se)

    # Cache per-condition C lookups to avoid re-querying the index repeatedly
    _c_cache: dict = {}

    matches = []
    for ge in groks_entries:
        gp = ge['model']['param_count']
        if gp is None:
            continue
        g_n_equiv, g_K_mem = compute_n_equiv(
            ge['data']['p'], ge['data']['operation'], ge['data']['train_fraction']
        )

        # Resolve capacity constant: per-condition lookup when index is available
        if capacity_index is not None:
            cond = (
                ge['model'].get('depth', 2),
                ge['model'].get('heads', 1),
                ge['optimizer'].get('weight_decay', 1.0),
                ge['model'].get('dropout', 0.2),
                ge['model'].get('init_scale', 1.0) or 1.0,
            )
            if cond not in _c_cache:
                _c_cache[cond] = measure_capacity_constant(
                    capacity_index,
                    depth=cond[0], heads=cond[1],
                    weight_decay=cond[2], dropout=cond[3], init_scale=cond[4],
                )
            C = _c_cache[cond] or capacity_constant
        else:
            C = capacity_constant

        # Collect candidates within n_samples_tolerance of the computed n_equiv
        candidates = []
        for delta in range(-n_samples_tolerance, n_samples_tolerance + 1):
            candidates.extend(speed_by_n.get(g_n_equiv + delta, []))
        for se in candidates:
            sp = se['model']['param_count']
            if sp is None:
                continue
            mismatch = abs(gp - sp) / gp if gp > 0 else 0.0
            if mismatch > param_tolerance:
                continue
            match_type = "exact" if mismatch == 0.0 else "param_matched"
            cap_frac = g_K_mem / (C * gp) if gp > 0 else 0.0
            matches.append(ExperimentMatch(
                groks_run_id=ge['run_id'],
                speed_run_id=se['run_id'],
                capacity_run_id=None,
                param_count_groks=gp,
                param_count_speed=sp,
                dataset_bits=g_K_mem,
                n_equiv=g_n_equiv,
                capacity_constant=C,
                capacity_fraction=cap_frac,
                match_type=match_type,
                param_count_mismatch=mismatch,
                p=ge['data']['p'],
                operation=ge['data']['operation'],
                train_fraction=ge['data']['train_fraction'],
                weight_decay_groks=ge['optimizer']['weight_decay'],
                weight_decay_speed=se['optimizer']['weight_decay'],
                architecture_family=ge['model']['architecture_family'],
                groks_npz_path=ge['_npz_path'],
                speed_npz_path=se['_npz_path'],
            ))
    return matches


def save_match_table(matches: list, path: str) -> None:
    """Serialise a list of ExperimentMatch to JSON."""
    with open(path, 'w') as f:
        json.dump([dataclasses.asdict(m) for m in matches], f, indent=2)


def load_match_table(path: str) -> list:
    """Load a match table JSON as plain dicts."""
    with open(path) as f:
        return json.load(f)
