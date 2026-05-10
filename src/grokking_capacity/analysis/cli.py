"""`gc-figures` — generate every figure family for one config (or all)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config_view import ConfigView
from .plots import render_all, write_meta
from .stats import render_stats


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO_ROOT / "configs"
DEFAULT_OUT_ROOT = REPO_ROOT / "figures"


def _render_one(config_path: Path, out_dir: Path, only: set[str] | None,
                db_path: str | None, skip_stats: bool) -> None:
    print(f"[gc-figures] {config_path.name} → {out_dir}")
    view = ConfigView.from_yaml(config_path, db_path=db_path)
    if not view.groups:
        print("  (no runs found in this config — skipping)")
        return
    print(f"  {len(view.groups)} arch group(s); swept axes: {view.swept_axes or '(none)'}")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_only = only and (only - {"stats"})
    rendered = render_all(view, out_dir, only=plot_only)
    for kind, paths in rendered.items():
        print(f"  {kind}: {len(paths)} file(s)")
    if not skip_stats and (only is None or "stats" in only):
        # Stats live alongside the figures they describe — one subdir per
        # IntersectionFigure under out_dir, e.g. out_dir/intersection/.
        stats_paths = render_stats(view, out_dir)
        print(f"  stats: {len(stats_paths)} file(s)")
    write_meta(view, out_dir)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="gc-figures",
        description="Render figures for a grokking_capacity YAML config.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=Path, help="Path to a single configs/*.yaml")
    src.add_argument("--all", action="store_true",
                     help="Iterate every yaml in configs/")
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory (default: figures/<config_name>/)")
    p.add_argument("--db", default=None,
                   help="Path to runs.db (default: GC_WALLOW_DB env or repo runs.db)")
    p.add_argument("--skip-stats", action="store_true",
                   help="Skip the predictiveness CSV/plots")
    p.add_argument("--only", choices=["intersection", "capacity", "speed", "stats"],
                   action="append", default=None,
                   help="Render only these families (repeat for multiple)")
    args = p.parse_args(argv)

    only = set(args.only) if args.only else None

    if args.all:
        if args.out:
            print("[gc-figures] --out is ignored with --all (each config gets its own folder)",
                  file=sys.stderr)
        for cfg in sorted(CONFIGS_DIR.glob("*.yaml")):
            _render_one(cfg, DEFAULT_OUT_ROOT / cfg.stem, only, args.db, args.skip_stats)
        return 0

    cfg = args.config
    out = args.out or (DEFAULT_OUT_ROOT / cfg.stem)
    _render_one(cfg, out, only, args.db, args.skip_stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
