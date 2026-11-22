"""`gc-figures` — generate every figure family for one config (or all)."""
from __future__ import annotations

import sys
from pathlib import Path

from ._cli import CONFIGS_DIR, REPO_ROOT, analysis_parser, config_jobs
from .config_view import ConfigView
from .plots import render_all, write_meta
from .stats import render_stats


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
    plot_only = None if only is None else only - {"stats"}
    rendered = {} if plot_only == set() else render_all(view, out_dir, only=plot_only)
    for kind, paths in rendered.items():
        print(f"  {kind}: {len(paths)} file(s)")
    if not skip_stats and (only is None or "stats" in only):
        # Stats live alongside the figures they describe — one subdir per
        # IntersectionFigure under out_dir, e.g. out_dir/intersection/.
        stats_paths = render_stats(view, out_dir)
        print(f"  stats: {len(stats_paths)} file(s)")
    write_meta(view, out_dir)


def main(argv: list[str] | None = None) -> int:
    p = analysis_parser(
        "gc-figures", "Render figures for a grokking_capacity YAML config.",
        output_dir="figures",
    )
    p.add_argument("--skip-stats", action="store_true",
                   help="Skip the predictiveness CSV/plots")
    p.add_argument("--only", choices=["intersection", "capacity", "speed", "stats"],
                   action="append", default=None,
                   help="Render only these families (repeat for multiple)")
    args = p.parse_args(argv)

    only = set(args.only) if args.only else None

    for cfg, out in config_jobs(
        args, configs_dir=CONFIGS_DIR, output_root=DEFAULT_OUT_ROOT, prog=p.prog,
    ):
        _render_one(cfg, out, only, args.db, args.skip_stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
