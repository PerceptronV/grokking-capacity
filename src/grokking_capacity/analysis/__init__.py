"""Post-hoc analysis, with public exports loaded only when requested.

Array-only helpers can be imported without loading plotting or registry code.
Existing ``from grokking_capacity.analysis import ...`` calls remain supported.
"""
from importlib import import_module

_EXPORTS = {
    'ExperimentMatch': ('matching', 'ExperimentMatch'),
    'build_match_table': ('matching', 'build_match_table'),
    'save_match_table': ('matching', 'save_match_table'),
    'load_match_table': ('matching', 'load_match_table'),
    'compute_n_equiv': ('matching', 'compute_n_equiv'),
    'find_dims_for_param_targets': ('matching', 'find_dims_for_param_targets'),
    'get_param_count': ('matching', 'get_param_count'),
    'fit_capacity_slope': ('capacity_constant', 'fit_capacity_slope'),
    'measure_capacity_constant': ('capacity_constant', 'measure_capacity_constant'),
    'ArchKey': ('config_view', 'ArchKey'),
    'ArchGroup': ('config_view', 'ArchGroup'),
    'ConfigView': ('config_view', 'ConfigView'),
    'IntersectionFigure': ('config_view', 'IntersectionFigure'),
    'load_npz': ('config_view', 'load_npz'),
    'render_all': ('plots', 'render_all'),
    'render_intersection': ('plots', 'render_intersection'),
    'render_capacity': ('plots', 'render_capacity'),
    'render_speed': ('plots', 'render_speed'),
    'write_meta': ('plots', 'write_meta'),
    'compute_predictiveness': ('stats', 'compute_predictiveness'),
    'plot_predicted_vs_empirical': ('stats', 'plot_predicted_vs_empirical'),
    'plot_error_vs_axis': ('stats', 'plot_error_vs_axis'),
    'render_stats': ('stats', 'render_stats'),
    'save_predictiveness_csv': ('stats', 'save_predictiveness_csv'),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module('.' + module, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
