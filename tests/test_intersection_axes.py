"""Smoke tests for the per-figure intersection rendering.

Builds a tiny synthetic ArchGroup with three primes × three dims of mock
speed/groks rows and checks that:
  - the default per-prime figure renders one PNG per prime,
  - a swapped per-dim figure (x = dataset_bits) renders one PNG per dim,
  - render_all writes each figure into its own subdir,
  - the predictiveness CSV gets the new column shape.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from torch_grokking.analysis.config_view import (
    ArchGroup, ArchKey, ConfigView, IntersectionFigure,
)
from torch_grokking.analysis.plots import render_all, render_intersection
from torch_grokking.analysis.stats import compute_predictiveness, render_stats


def _arch_key() -> ArchKey:
    return ArchKey.from_dict({"p": 113})


def _row(*, exp_type, p, dim, seed, param_count, n_samples, dataset_bits,
         epoch_field, epoch_value, npz_path=None):
    """Build a mock wallow row dict with all the fields the analysis layer reads."""
    return {
        "experiment_type": exp_type,
        "p": p, "operation": "/", "train_fraction": 0.5, "split_type": "random",
        "depth": 2, "heads": 1, "dropout": 0.2, "init_scale": 1.0,
        "lr": 0.001, "weight_decay": 1.0, "beta1": 0.9, "beta2": 0.98,
        "batch_size": 512, "architecture_family": "transformer_gated",
        "max_epochs": 5000, "seed": seed, "dim": dim, "n_samples": n_samples,
        "param_count": param_count, "n_equiv": n_samples,
        "dataset_bits": dataset_bits,
        "saturated": True,
        "npz_path": str(npz_path) if npz_path is not None else None,
        epoch_field: epoch_value,
    }


def _write_groks_npz(path: Path, *, train_epoch: int, grok_epoch: int,
                     total_epochs: int = 2000):
    """Synthetic train_acc/val_acc traces that hit 99% at the given epochs."""
    assert grok_epoch < total_epochs, (grok_epoch, total_epochs)
    train_acc = np.zeros(total_epochs)
    train_acc[train_epoch:] = 100.0
    val_acc = np.zeros(total_epochs)
    val_acc[grok_epoch:] = 100.0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, train_acc=train_acc, val_acc=val_acc)


@pytest.fixture
def synth_group(tmp_path):
    """Three primes × three dims × one seed of mock speed + groks rows.

    Mem epochs decrease with capacity (param_count); gen epochs decrease
    faster — they cross within the panel for most slices, which is enough
    to exercise the intersection finder.
    """
    primes = [97, 113, 139]
    dims = [32, 64, 128]
    speed_rows, groks_rows = [], []
    npz_dir = tmp_path / "npz"
    for p in primes:
        for dim in dims:
            param_count = dim * dim * 8 + p * dim * 2
            n_samples = (p * (p - 1)) // 2
            dataset_bits = float(n_samples * np.log2(p + 2))
            mem_epoch = int(50.0 + 1e6 / param_count)
            grok_epoch = int(100.0 + 1e7 / param_count)
            speed_rows.append(_row(
                exp_type="speed", p=p, dim=dim, seed=42,
                param_count=param_count, n_samples=n_samples,
                dataset_bits=dataset_bits,
                epoch_field="saturation_epoch",
                epoch_value=float(mem_epoch),
            ))
            npz_path = npz_dir / f"groks_p{p}_dim{dim}.npz"
            _write_groks_npz(npz_path, train_epoch=mem_epoch,
                             grok_epoch=max(grok_epoch, mem_epoch + 1))
            groks_rows.append(_row(
                exp_type="groks", p=p, dim=dim, seed=42,
                param_count=param_count, n_samples=n_samples,
                dataset_bits=dataset_bits,
                epoch_field="grokking_epoch",
                epoch_value=float(grok_epoch),
                npz_path=npz_path,
            ))
    return ArchGroup(key=_arch_key(), capacity_runs=[],
                     speed_runs=speed_rows, groks_runs=groks_rows)


def test_default_figure_renders_one_png_per_prime(synth_group, tmp_path):
    figure = IntersectionFigure(
        name="intersection", slice_field="p", x_field="param_count",
        x_label="Parameter count", colour_field="dim", colour_label="Dim",
    )
    out = render_intersection(synth_group, tmp_path / "intersection",
                              figure=figure)
    names = sorted(p.name for p in out)
    assert names == ["p=113.png", "p=139.png", "p=97.png"]


def test_swapped_figure_renders_one_png_per_dim(synth_group, tmp_path):
    figure = IntersectionFigure(
        name="intersection_by_prime", slice_field="dim",
        x_field="dataset_bits", x_label="Dataset bits",
        colour_field="p", colour_label="Prime",
    )
    out = render_intersection(synth_group, tmp_path / "by_prime",
                              figure=figure)
    names = sorted(p.name for p in out)
    assert names == ["dim=128.png", "dim=32.png", "dim=64.png"]


def test_render_all_uses_per_figure_subdirs(synth_group, tmp_path):
    figures = [
        IntersectionFigure(name="intersection", slice_field="p",
                           x_field="param_count", x_label="Parameter count",
                           colour_field="dim", colour_label="Dim"),
        IntersectionFigure(name="intersection_by_prime", slice_field="dim",
                           x_field="dataset_bits", x_label="Dataset bits",
                           colour_field="p", colour_label="Prime"),
    ]
    view = ConfigView(
        config_name="synth", config_path=tmp_path / "synth.yaml",
        groups=[synth_group], swept_axes=[],
        intersection_figures=figures,
    )
    rendered = render_all(view, tmp_path / "out", only={"intersection"})
    assert "intersection" in rendered and "intersection_by_prime" in rendered
    assert all((tmp_path / "out" / "intersection" / n).exists()
               for n in ["p=97.png", "p=113.png", "p=139.png"])
    assert all((tmp_path / "out" / "intersection_by_prime" / n).exists()
               for n in ["dim=32.png", "dim=64.png", "dim=128.png"])


def test_predictiveness_csv_has_renamed_columns(synth_group, tmp_path):
    figure = IntersectionFigure(
        name="intersection", slice_field="p", x_field="param_count",
        x_label="Parameter count", colour_field="dim", colour_label="Dim",
    )
    view = ConfigView(
        config_name="synth", config_path=tmp_path / "synth.yaml",
        groups=[synth_group], swept_axes=[],
        intersection_figures=[figure],
    )
    df = compute_predictiveness(view, figure)
    assert {"slice_field", "slice_value", "x_field",
            "predicted_onset_x", "empirical_onset_x"}.issubset(df.columns)
    assert set(df["slice_value"].tolist()) == {97, 113, 139}
    assert (df["x_field"] == "param_count").all()


def test_max_dim_excludes_wide_runs_from_curve(synth_group, tmp_path):
    """A figure with max_dim=64 should drop the dim=128 row from every
    pathway (curves, scatter, slice enumeration)."""
    from torch_grokking.analysis.plots import (
        _curve_for_slice, _slice_values, _delay_records_for_slice,
    )
    figure = IntersectionFigure(
        name="capped", slice_field="p", x_field="param_count",
        x_label="Parameter count", colour_field="dim", colour_label="Dim",
        max_dim=64,
    )
    speed_curve = _curve_for_slice(synth_group.speed_runs, figure, 113,
                                    "saturation_epoch")
    assert len(speed_curve) == 2  # only dim=32 and dim=64 survive
    records = _delay_records_for_slice(synth_group, figure, 113)
    assert len(records) == 2

    figure_by_dim = IntersectionFigure(
        name="capped_by_dim", slice_field="dim", x_field="dataset_bits",
        x_label="Dataset bits", colour_field="p", colour_label="Prime",
        max_dim=64,
    )
    assert _slice_values(synth_group, figure_by_dim) == [32, 64]


def test_delay_thresholds_are_wired_through(synth_group, tmp_path):
    """Two checks that the figure's thresholds actually drive compute_delays:

    (1) train threshold above what the npz ever reaches drops every
        record (compute_delays' first guard is 'train never crossed').
    (2) val threshold of 98 vs 99 selects different val epochs when the
        trace plateaus between the two — produces different delays.
    """
    from torch_grokking.analysis.plots import _delay_records_for_slice
    figure_unreachable_train = IntersectionFigure(
        name="ur_train", slice_field="p", x_field="param_count",
        x_label="Parameter count", colour_field="dim", colour_label="Dim",
        delay_train_threshold=101.0,  # synthetic train maxes at 100 → drop all
    )
    assert _delay_records_for_slice(synth_group, figure_unreachable_train, 113) == []

    # Carve out a single row whose npz val plateaus at 98.5 between epochs
    # 200..299 then jumps to 100 at 300. Train hits 100 immediately at 100.
    plateau_path = tmp_path / "npz" / "plateau.npz"
    plateau_path.parent.mkdir(parents=True, exist_ok=True)
    train = np.zeros(500); train[100:] = 100.0
    val = np.zeros(500); val[200:300] = 98.5; val[300:] = 100.0
    np.savez(plateau_path, train_acc=train, val_acc=val)
    plateau_row = _row(
        exp_type="groks", p=999, dim=64, seed=42,
        param_count=10000, n_samples=500_000, dataset_bits=5_000_000.0,
        epoch_field="grokking_epoch", epoch_value=300.0,
        npz_path=plateau_path,
    )
    plateau_group = ArchGroup(
        key=_arch_key(), capacity_runs=[],
        speed_runs=[], groks_runs=[plateau_row],
    )
    fig_98 = IntersectionFigure(
        name="v98", slice_field="p", x_field="param_count",
        x_label="Parameter count", colour_field="dim", colour_label="Dim",
        delay_val_threshold=98.0,
    )
    fig_99 = IntersectionFigure(
        name="v99", slice_field="p", x_field="param_count",
        x_label="Parameter count", colour_field="dim", colour_label="Dim",
        delay_val_threshold=99.0,
    )
    [r98] = _delay_records_for_slice(plateau_group, fig_98, 999)
    [r99] = _delay_records_for_slice(plateau_group, fig_99, 999)
    assert r98["delay"] == 200 - 100  # val crosses 98 at 200
    assert r99["delay"] == 300 - 100  # val crosses 99 at 300


def test_yaml_defaults_for_thresholds_and_cap():
    """A figure entry without max_dim / threshold keys defaults to no cap
    and 99/99 thresholds (preserving the pre-cap behaviour)."""
    from torch_grokking.analysis.config_view import _parse_intersection_figures
    spec = {"analysis": {"intersection_figures": [
        {"name": "f", "slice_field": "p", "x_field": "param_count"},
    ]}}
    figs = _parse_intersection_figures(spec)
    assert figs[0].max_dim is None
    assert figs[0].delay_train_threshold == 99.0
    assert figs[0].delay_val_threshold == 99.0
    assert figs[0].mem_curve_threshold == 99.0
    assert figs[0].gen_curve_threshold == 99.0


def test_curve_thresholds_label_the_legend(synth_group, tmp_path):
    """The gen/mem curve thresholds drive the legend strings in the PNG.

    We can't easily assert against rendered pixels, but we can spy on the
    primitive's keyword args via a wrapper.
    """
    import torch_grokking.analysis.plots as plots_mod
    captured = {}
    real = plots_mod.plot_grokking_delay_with_speed
    def wrapper(*a, **kw):
        captured.update(kw)
        return real(*a, **kw)
    plots_mod.plot_grokking_delay_with_speed = wrapper
    try:
        figure = IntersectionFigure(
            name="thr_check", slice_field="p", x_field="param_count",
            x_label="Parameter count", colour_field="dim", colour_label="Dim",
            mem_curve_threshold=97.5, gen_curve_threshold=98.5,
        )
        plots_mod.render_intersection(synth_group, tmp_path / "out", figure=figure)
    finally:
        plots_mod.plot_grokking_delay_with_speed = real
    assert captured["mem_curve_threshold"] == 97.5
    assert captured["gen_curve_threshold"] == 98.5


def test_scatter_takes_min_delay_across_seeds(tmp_path):
    """Three seeds at the same dim/prime produce three rows with three
    delays; the scatter shows ONE dot at that x with the minimum delay."""
    from torch_grokking.analysis.plots import _delay_records_for_slice
    npz_dir = tmp_path / "npz"
    rows = []
    for seed, grok_epoch in [(42, 600), (43, 300), (44, 450)]:
        npz = npz_dir / f"seed{seed}.npz"
        _write_groks_npz(npz, train_epoch=100,
                         grok_epoch=grok_epoch, total_epochs=1000)
        rows.append(_row(
            exp_type="groks", p=113, dim=64, seed=seed,
            param_count=50000, n_samples=6328, dataset_bits=5e6,
            epoch_field="grokking_epoch", epoch_value=float(grok_epoch),
            npz_path=npz,
        ))
    group = ArchGroup(key=_arch_key(), capacity_runs=[],
                      speed_runs=[], groks_runs=rows)
    figure = IntersectionFigure(
        name="min_check", slice_field="p", x_field="param_count",
        x_label="Parameter count", colour_field="dim", colour_label="Dim",
    )
    records = _delay_records_for_slice(group, figure, 113)
    assert len(records) == 1
    assert records[0]["x"] == 50000
    # Min delay = grok_epoch=300 minus train_epoch=100 = 200.
    assert records[0]["delay"] == 200


def test_warn_on_stored_threshold_mismatch(synth_group, tmp_path):
    """If a groks row carries `grokking_threshold` and it disagrees with
    the figure's `gen_curve_threshold`, the renderer warns."""
    import warnings
    # Annotate the stored threshold on every groks row at 0.99.
    for r in synth_group.groks_runs:
        r["grokking_threshold"] = 0.99
    figure = IntersectionFigure(
        name="warn_check", slice_field="p", x_field="param_count",
        x_label="Parameter count", colour_field="dim", colour_label="Dim",
        gen_curve_threshold=95.0,  # disagrees with stored 99
    )
    from torch_grokking.analysis.plots import render_intersection
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        render_intersection(synth_group, tmp_path / "out", figure=figure)
    msgs = [str(w.message) for w in caught]
    assert any("grokking_threshold" in m and "95.0" in m for m in msgs), msgs


def test_render_stats_writes_per_figure_subdirs(synth_group, tmp_path):
    figures = [
        IntersectionFigure(name="intersection", slice_field="p",
                           x_field="param_count", x_label="Parameter count",
                           colour_field="dim", colour_label="Dim"),
        IntersectionFigure(name="intersection_by_prime", slice_field="dim",
                           x_field="dataset_bits", x_label="Dataset bits",
                           colour_field="p", colour_label="Prime"),
    ]
    view = ConfigView(
        config_name="synth", config_path=tmp_path / "synth.yaml",
        groups=[synth_group], swept_axes=[],
        intersection_figures=figures,
    )
    paths = render_stats(view, tmp_path / "out")
    assert (tmp_path / "out" / "intersection" / "predictiveness.csv").exists()
    assert (tmp_path / "out" / "intersection_by_prime" / "predictiveness.csv").exists()
    # render_stats reports per-figure CSV paths.
    assert any(k.endswith("/csv") for k in paths)
