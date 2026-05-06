"""Orchestrators that turn one `ArchGroup` into one figure family on disk.

These functions are thin: they pull the right curves out of the wallow rows
the group already holds, then hand off to the rendering primitives in
`plotting`. None of them re-query wallow.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from ._primitives import (
    plot_capacity_curves,
    plot_capacity_estimation,
    plot_grokking_delay_with_speed,
    plot_saturation_epochs_vs_inverse_capacity,
    plot_saturation_time_vs_capacity_fraction,
)

from . import aggregate
from .config_view import ArchGroup, ConfigView, load_npz


SATURATION_THRESHOLD = 99.0
DELAY_TRAIN_THRESHOLD = 99.0
DELAY_VAL_THRESHOLD = 99.0


def _delay_records_for_prime(group: ArchGroup, p: int) -> list[dict]:
    """Per-seed delay scatter records for the intersection plot."""
    records: list[dict] = []
    for row in group.groks_runs:
        if row.get("p") != p:
            continue
        try:
            with load_npz(row) as npz:
                train = npz["train_acc"]
                val = npz["val_acc"]
        except (FileNotFoundError, KeyError):
            continue
        delays = aggregate.compute_delays(
            [{"param_count": row.get("param_count"),
              "train_acc": train, "val_acc": val}],
            threshold_train=DELAY_TRAIN_THRESHOLD,
            threshold_val=DELAY_VAL_THRESHOLD,
        )
        if not delays:
            continue
        pc, delay = delays[0]
        records.append({
            "param_count": pc,
            "dim": int(row.get("dim") or 0),
            "delay": delay,
        })
    return records


def _curve_for_prime(rows: Iterable[dict], p: int, y_field: str) -> dict[float, float]:
    return aggregate.mean_over_seeds(
        (r for r in rows if r.get("p") == p and r.get(y_field) is not None),
        x_field="param_count", y_field=y_field,
    )


def render_intersection(group: ArchGroup, save_dir: Path, suffix: str = "") -> list[Path]:
    """One PNG per prime. Skips primes with no groks/speed overlap."""
    primes = sorted({r.get("p") for r in group.groks_runs if r.get("p") is not None})
    out: list[Path] = []
    save_dir.mkdir(parents=True, exist_ok=True)
    for p in primes:
        records = _delay_records_for_prime(group, p)
        groks_curve = _curve_for_prime(group.groks_runs, p, "grokking_epoch")
        speed_curve = _curve_for_prime(group.speed_runs, p, "saturation_epoch")
        if not records or not groks_curve or not speed_curve:
            continue
        name = f"p={p}{('__' + suffix) if suffix else ''}.png"
        path = save_dir / name
        plot_grokking_delay_with_speed(
            records, speed_curve=speed_curve, groks_curve=groks_curve,
            saturation_threshold=SATURATION_THRESHOLD,
            threshold_train=DELAY_TRAIN_THRESHOLD,
            threshold_val=DELAY_VAL_THRESHOLD,
            title=f"p = {p}", save_path=str(path),
        )
        out.append(path)
    return out


def render_capacity(group: ArchGroup, save_dir: Path, suffix: str = "") -> list[Path]:
    """Image #2(a) + #2(b)."""
    if not group.capacity_runs:
        return []
    save_dir.mkdir(parents=True, exist_ok=True)
    by_dim: dict[int, list[dict]] = {}
    for r in group.capacity_runs:
        by_dim.setdefault(int(r["dim"]), []).append(r)
    primary_p = next(iter({r.get("p") for r in group.capacity_runs if r.get("p")}), group.key.p)

    suffix_part = f"__{suffix}" if suffix else ""
    curves_path = save_dir / f"M_T_vs_dataset_size{suffix_part}.png"
    fit_path = save_dir / f"saturation_bits_vs_params{suffix_part}.png"
    saturation_points = plot_capacity_curves(
        by_dim, p=int(primary_p), save_path=str(curves_path),
    )
    plot_capacity_estimation(saturation_points, save_path=str(fit_path))
    return [curves_path, fit_path]


def render_speed(group: ArchGroup, save_dir: Path, suffix: str = "") -> list[Path]:
    """Image #3(a) + #3(b)."""
    if not group.speed_runs:
        return []
    save_dir.mkdir(parents=True, exist_ok=True)
    primes = {r.get("p") for r in group.speed_runs if r.get("p") is not None}
    multi_prime = len(primes) > 1

    by_key: dict = {}
    for r in group.speed_runs:
        if multi_prime:
            k = (int(r["p"]), int(r["dim"]))
        else:
            k = int(r["dim"])
        by_key.setdefault(k, []).append(r)

    suffix_part = f"__{suffix}" if suffix else ""
    inv_path = save_dir / f"epochs_vs_inverse_capacity{suffix_part}.png"
    frac_path = save_dir / f"epochs_vs_capacity_fraction{suffix_part}.png"
    plot_saturation_epochs_vs_inverse_capacity(
        by_key, C=group.capacity_constant, save_path=str(inv_path),
    )
    plot_saturation_time_vs_capacity_fraction(
        by_key, C=group.capacity_constant, save_path=str(frac_path),
    )
    return [inv_path, frac_path]


def write_meta(view: ConfigView, out_dir: Path) -> Path:
    """Per-config provenance: capacity constants, run counts, swept axes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "config_name": view.config_name,
        "config_path": str(view.config_path),
        "swept_axes": view.swept_axes,
        "groups": [
            {
                "arch_key": {f: getattr(g.key, f) for f in g.key.__dataclass_fields__},
                "capacity_constant": g.capacity_constant,
                "capacity_constant_source": g.capacity_constant_source,
                "n_capacity_runs": len(g.capacity_runs),
                "n_speed_runs": len(g.speed_runs),
                "n_groks_runs": len(g.groks_runs),
            }
            for g in view.groups
        ],
    }
    path = out_dir / "meta.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return path


def render_all(
    view: ConfigView,
    out_dir: Path,
    *,
    only: Optional[set[str]] = None,
) -> dict[str, list[Path]]:
    """Render every figure family for every group in the config view."""
    out_dir = Path(out_dir)
    only = only or {"intersection", "capacity", "speed"}
    rendered: dict[str, list[Path]] = {"intersection": [], "capacity": [], "speed": []}
    for group in view.iter_groups():
        if not group.has_data():
            continue
        suffix = group.key.short_label(view.swept_axes)
        if "intersection" in only:
            rendered["intersection"].extend(
                render_intersection(group, out_dir / "intersection", suffix=suffix)
            )
        if "capacity" in only:
            rendered["capacity"].extend(
                render_capacity(group, out_dir / "capacity", suffix=suffix)
            )
        if "speed" in only:
            rendered["speed"].extend(
                render_speed(group, out_dir / "speed", suffix=suffix)
            )
    return rendered
