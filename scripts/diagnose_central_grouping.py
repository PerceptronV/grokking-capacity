"""One-shot diagnostic: how many rows match the central.yaml ArchGroups?

Run on the GPU box from the repo root:

    python scripts/diagnose_central_grouping.py

For each ArchGroup in `configs/central.yaml`, prints the per-experiment-type
row counts and a breakdown of (prime, dim) coverage that should appear in
the `intersection_by_prime/dim={dim}` panels. If the swap figures look
empty even after the prime-pooling fix, the rows the analysis layer
actually sees are the smoking gun.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from wallow import F

from grokking_capacity.analysis.config_view import ConfigView
from grokking_capacity.registry import get_store


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "central.yaml"


def main() -> None:
    store = get_store()
    view = ConfigView.from_yaml(CONFIG)

    for group in view.iter_groups():
        speed = group.speed_runs
        groks = group.groks_runs
        cap = group.capacity_runs
        print()
        print(f"ArchGroup key: {group.key}")
        print(f"  capacity rows: {len(cap)}")
        print(f"  speed rows:    {len(speed)}")
        print(f"  groks rows:    {len(groks)}")

        if not groks:
            continue
        # Per-prime breakdown, focused on a few representative dims.
        primes = sorted({r.get("p") for r in groks if r.get("p") is not None})
        print(f"  primes seen in groks_runs: {primes}")

        for dim in (24, 56, 100, 124, 200):
            grok_at_dim = [r for r in groks if r.get("dim") == dim]
            speed_at_dim = [r for r in speed if r.get("dim") == dim]
            grok_with_epoch = [r for r in grok_at_dim
                               if r.get("grokking_epoch") is not None]
            speed_with_epoch = [r for r in speed_at_dim
                                if r.get("saturation_epoch") is not None]
            grok_primes = Counter(r.get("p") for r in grok_at_dim)
            grok_groked_primes = Counter(r.get("p") for r in grok_with_epoch)
            print(f"  dim={dim}: groks rows={len(grok_at_dim)} "
                  f"(non-None grokking_epoch={len(grok_with_epoch)}); "
                  f"speed rows={len(speed_at_dim)} "
                  f"(non-None saturation_epoch={len(speed_with_epoch)})")
            print(f"          groks rows by prime: {dict(grok_primes)}")
            print(f"          groks-with-epoch by prime: {dict(grok_groked_primes)}")

    # Also a raw tally directly from the DB, ignoring ArchKey filters, so we
    # can see whether the rows just aren't there vs. they're there but the
    # filter doesn't match.
    print()
    print("Raw DB tallies (no ArchKey filter):")
    for et in ("capacity", "speed", "groks"):
        n = store.where(
            (F("status") == "completed") & (F("experiment_type") == et)
        ).count()
        print(f"  {et}: {n} completed rows total")

    # Simulate the per-panel rendering path for one dim slice of
    # intersection_by_prime so we can see what _curve_for_slice and
    # _delay_records_for_slice actually return.
    from grokking_capacity.analysis.plots import (
        _curve_for_slice, _delay_records_for_slice, _slice_values,
    )
    speed_group = next((g for g in view.iter_groups() if g.groks_runs), None)
    if speed_group is None:
        return
    figure = next((f for f in view.intersection_figures
                   if f.name == "intersection_by_prime"), None)
    if figure is None:
        print("\nNo intersection_by_prime figure declared; skipping panel sim.")
        return
    print()
    print(f"Panel simulation for figure={figure.name!r}:")
    for dim in (24, 56, 100, 124):
        groks_curve = _curve_for_slice(speed_group.groks_runs, figure, dim,
                                        "grokking_epoch")
        speed_curve = _curve_for_slice(speed_group.speed_runs, figure, dim,
                                        "saturation_epoch")
        records = _delay_records_for_slice(speed_group, figure, dim)
        print(f"  dim={dim}: groks_curve points={len(groks_curve)}, "
              f"speed_curve points={len(speed_curve)}, "
              f"scatter records={len(records)}")
        if groks_curve:
            xs = sorted(groks_curve.keys())
            print(f"          groks_curve x range: "
                  f"{xs[0]:.0f} → {xs[-1]:.0f}; sample y[0]={groks_curve[xs[0]]:.1f}")
        if records:
            xs = sorted(r["x"] for r in records)
            cs = sorted(r["colour"] for r in records)
            print(f"          scatter unique x: {xs}")
            print(f"          scatter unique colour (prime): {cs}")


if __name__ == "__main__":
    main()
