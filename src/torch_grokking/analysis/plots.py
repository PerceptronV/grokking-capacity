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
from .config_view import ArchGroup, ConfigView, IntersectionFigure, load_npz


import warnings

SATURATION_THRESHOLD = 99.0
DELAY_TRAIN_THRESHOLD = 99.0
DELAY_VAL_THRESHOLD = 99.0


def _warn_threshold_mismatch(rows, threshold_field: str, configured: float,
                              context: str) -> None:
    """If wallow rows annotate the threshold they used, warn when it differs
    from the figure's configured curve threshold.

    Currently only `grokking_threshold` is annotated (by groks.py); speed
    runs don't store theirs yet, so this no-ops for the mem curve until
    that annotation lands.
    """
    observed = {r.get(threshold_field) for r in rows
                if r.get(threshold_field) is not None}
    if not observed:
        return
    # Stored threshold is in [0, 1] (e.g. 0.99); configured is in % (99).
    bad = {t for t in observed if abs(float(t) * 100.0 - configured) > 0.5}
    if bad:
        warnings.warn(
            f"{context}: stored {threshold_field} values {sorted(bad)} "
            f"(×100) don't match the configured curve threshold "
            f"{configured}. The legend will say {configured}% even though "
            f"the underlying data was computed at a different threshold.",
            stacklevel=2,
        )


def _passes_filters(row: dict, figure: IntersectionFigure) -> bool:
    """Apply every figure-level row filter (currently just `max_dim`)."""
    if figure.max_dim is not None:
        dim = row.get("dim")
        if dim is None or dim > figure.max_dim:
            return False
    return True


def _delay_records_for_slice(
    group: ArchGroup,
    figure: IntersectionFigure,
    slice_value,
) -> list[dict]:
    """Min-across-seeds delay scatter records for one slice value.

    Returns one dict per `(x, colour)` cell, holding the minimum delay
    observed across compatible seeds — the same convention
    `aggregate.find_grokking_onset` uses when locating the empirical
    onset. The keys `x`, `colour`, `delay` are generic so the rendering
    primitive stays axis-agnostic; the figure spec decides which row
    fields those map to. Delay is recomputed from the npz traces using
    the figure spec's train/val thresholds.
    """
    per_cell: dict[tuple, dict] = {}
    for row in group.groks_runs:
        if row.get(figure.slice_field) != slice_value:
            continue
        if not _passes_filters(row, figure):
            continue
        try:
            with load_npz(row) as npz:
                train = npz["train_acc"]
                val = npz["val_acc"]
        except (FileNotFoundError, KeyError):
            continue
        delays = aggregate.compute_delays(
            [{figure.x_field: row.get(figure.x_field),
              "train_acc": train, "val_acc": val}],
            x_field=figure.x_field,
            threshold_train=figure.delay_train_threshold,
            threshold_val=figure.delay_val_threshold,
        )
        if not delays:
            continue
        x_value, delay = delays[0]
        colour_raw = row.get(figure.colour_field)
        colour = float(colour_raw if colour_raw is not None else 0)
        key = (float(x_value), colour)
        existing = per_cell.get(key)
        if existing is None or delay < existing["delay"]:
            per_cell[key] = {
                "x": float(x_value),
                "colour": colour,
                "delay": float(delay),
            }
    return list(per_cell.values())


def _curve_for_slice(
    rows: Iterable[dict],
    figure: IntersectionFigure,
    slice_value,
    y_field: str,
) -> dict[float, float]:
    return aggregate.mean_over_seeds(
        (r for r in rows
         if r.get(figure.slice_field) == slice_value
         and _passes_filters(r, figure)
         and r.get(figure.x_field) is not None
         and r.get(y_field) is not None),
        x_field=figure.x_field, y_field=y_field,
    )


def _slice_values(group: ArchGroup, figure: IntersectionFigure) -> list:
    """Distinct, sorted slice values that survive the figure's row filters."""
    vals = {r.get(figure.slice_field) for r in group.groks_runs
            if r.get(figure.slice_field) is not None
            and _passes_filters(r, figure)}
    return sorted(vals)


def render_intersection(
    group: ArchGroup,
    save_dir: Path,
    figure: IntersectionFigure = None,
    suffix: str = "",
) -> list[Path]:
    """One PNG per slice value. Skips slices with no groks/speed overlap."""
    if figure is None:
        # Default for ad-hoc callers / tests: today's per-prime figure.
        from .config_view import _DEFAULT_INTERSECTION_FIGURE
        figure = _DEFAULT_INTERSECTION_FIGURE
    out: list[Path] = []
    save_dir.mkdir(parents=True, exist_ok=True)
    for slice_value in _slice_values(group, figure):
        records = _delay_records_for_slice(group, figure, slice_value)
        groks_curve = _curve_for_slice(group.groks_runs, figure, slice_value,
                                        "grokking_epoch")
        speed_curve = _curve_for_slice(group.speed_runs, figure, slice_value,
                                        "saturation_epoch")
        if not records or not groks_curve or not speed_curve:
            continue
        slice_label = f"{figure.slice_field}={slice_value}"
        fname = f"{slice_label}{('__' + suffix) if suffix else ''}.png"
        path = save_dir / fname
        # Surface mismatches between the figure's declared curve thresholds
        # and the threshold each row was actually computed at (groks rows
        # carry `grokking_threshold`; speed rows don't yet annotate theirs).
        contributing_groks = [r for r in group.groks_runs
                              if r.get(figure.slice_field) == slice_value
                              and _passes_filters(r, figure)]
        _warn_threshold_mismatch(contributing_groks, "grokking_threshold",
                                  figure.gen_curve_threshold,
                                  context=f"{figure.name}/{slice_label}")
        # Surface degenerate panels — curves with <2 distinct x values can't
        # form a visible line or yield an intersection.
        if len(speed_curve) < 2 or len(groks_curve) < 2:
            warnings.warn(
                f"{figure.name}/{slice_label}: degenerate curves — "
                f"speed has {len(speed_curve)} point(s), "
                f"groks has {len(groks_curve)} point(s). "
                f"Scatter records: {len(records)}. "
                f"This panel will render but the curves and intersection "
                f"won't be visible.",
                stacklevel=2,
            )
        plot_grokking_delay_with_speed(
            records, speed_curve=speed_curve, groks_curve=groks_curve,
            mem_curve_threshold=figure.mem_curve_threshold,
            gen_curve_threshold=figure.gen_curve_threshold,
            threshold_train=figure.delay_train_threshold,
            threshold_val=figure.delay_val_threshold,
            title=slice_label.replace("=", " = "),
            save_path=str(path),
            x_label=figure.x_label,
            colour_label=figure.colour_label,
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
    primary_p = next(iter({r.get("p") for r in group.capacity_runs if r.get("p")}), 113)

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
    """Render every figure family for every group in the config view.

    When `only` includes `"intersection"`, every figure declared in
    `view.intersection_figures` is rendered (each into `out_dir/<name>/`).
    Each figure's paths land in their own bucket of the returned dict so
    the CLI can report file counts per figure.
    """
    out_dir = Path(out_dir)
    only = only or {"intersection", "capacity", "speed"}
    rendered: dict[str, list[Path]] = {"capacity": [], "speed": []}
    if "intersection" in only:
        for figure in view.intersection_figures:
            rendered.setdefault(figure.name, [])
    for group in view.iter_groups():
        if not group.has_data():
            continue
        suffix = group.key.short_label(view.swept_axes)
        if "intersection" in only:
            for figure in view.intersection_figures:
                rendered[figure.name].extend(
                    render_intersection(group, out_dir / figure.name,
                                        figure=figure, suffix=suffix)
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
