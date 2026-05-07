"""Post-hoc analysis of completed runs: matching, capacity-constant fits,
config-driven figure rendering, and predictiveness stats.

The CLI lives in `cli.py` and is registered as `gc-figures`.
"""
# Legacy matching utilities (still used by dispatch/main.py to emit matches.json
# and by dispatch/config.py for compute_n_equiv / find_dims_for_param_targets).
from .matching import (
    ExperimentMatch,
    build_match_table,
    save_match_table,
    load_match_table,
    compute_n_equiv,
    find_dims_for_param_targets,
    get_param_count,
)
from .capacity_constant import fit_capacity_slope, measure_capacity_constant

# Config-driven analysis pipeline.
from .config_view import ArchKey, ArchGroup, ConfigView, IntersectionFigure, load_npz
from .plots import (
    render_all,
    render_intersection,
    render_capacity,
    render_speed,
    write_meta,
)
from .stats import (
    compute_predictiveness,
    plot_predicted_vs_empirical,
    plot_error_vs_axis,
    render_stats,
    save_predictiveness_csv,
)

__all__ = [
    # legacy matching
    "ExperimentMatch",
    "build_match_table", "save_match_table", "load_match_table",
    "compute_n_equiv", "find_dims_for_param_targets", "get_param_count",
    # capacity constant
    "fit_capacity_slope", "measure_capacity_constant",
    # config-driven pipeline
    "ArchKey", "ArchGroup", "ConfigView", "IntersectionFigure", "load_npz",
    "render_all", "render_intersection", "render_capacity", "render_speed",
    "write_meta",
    "compute_predictiveness", "plot_predicted_vs_empirical",
    "plot_error_vs_axis", "render_stats", "save_predictiveness_csv",
]
