"""Parse a YAML config, group completed wallow rows by sweep cell.

A "cell" (`ArchGroup`) is one architecture × task setting that holds across
its capacity / speed / groks runs. Within a group the only varying fields
are `seed`, `dim`, and `n_samples` — those drive the seed/param/dataset axes
of every plot.

Each `ArchGroup` carries its own `capacity_constant`, fitted from its own
capacity runs (or `consts.C` when no capacity runs exist) — so that
sweep-specific normalisations (e.g. `f = S/(C·P)` and the inverse-capacity
plots) use the C that was actually measured at that architecture.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import yaml
from wallow import F

from .capacity_constant import fit_capacity_slope
from ..consts import C as DEFAULT_C
from ..registry import get_store, npz_path_for_row

# `..dispatch.config` imports back from this package (matching utilities),
# so `expand_runs` is imported lazily inside `ConfigView.from_yaml` to break
# the cycle.


# Identifying fields that pin one architecture/task setting. Excluded from
# the key (because they vary within a group): `seed`, `dim`, `n_samples`,
# `experiment_type`, `dataset_type`, `max_epochs` (capacity runs an order
# of magnitude longer than groks; not part of the architecture), and `p`
# (treated as an *input* the figures vary along — figures keyed on prime
# slice the group's rows by `p` themselves; figures keyed on dim need rows
# from all primes pooled into one group to form a curve).
ARCH_KEY_FIELDS: tuple[str, ...] = (
    "operation", "train_fraction", "split_type",
    "depth", "heads", "dropout", "init_scale",
    "lr", "weight_decay", "beta1", "beta2", "batch_size",
    "architecture_family",
)


# The wallow.toml defaults the dispatcher relies on when a field is not in
# the config — duplicated here so we can rebuild ArchKey from a wallow row
# whose absent fields default to these values.
_TOML_DEFAULTS: dict[str, Any] = {
    "operation": "/",
    "train_fraction": 0.5,
    "split_type": "random",
    "depth": 2,
    "heads": 1,
    "dropout": 0.2,
    "init_scale": 1.0,
    "lr": 0.001,
    "weight_decay": 1.0,
    "beta1": 0.9,
    "beta2": 0.98,
    "batch_size": 512,
    "architecture_family": "transformer_gated",
}


_OPERATION_FILENAME_LABELS: dict[str, str] = {
    "+": "add", "-": "sub", "*": "mul", "/": "div",
}


def _safe_for_filename(value: Any) -> str:
    """Render a sweep value as a filename-safe token. Operation symbols
    get readable names (`/` → `div` in particular — leaving it would split
    the suffix across a directory boundary)."""
    s = str(value)
    if s in _OPERATION_FILENAME_LABELS:
        return _OPERATION_FILENAME_LABELS[s]
    for ch in '/\\:*?"<>| ':
        s = s.replace(ch, "_")
    return s


@dataclass(frozen=True)
class ArchKey:
    operation: str
    train_fraction: float
    split_type: str
    depth: int
    heads: int
    dropout: float
    init_scale: float
    lr: float
    weight_decay: float
    beta1: float
    beta2: float
    batch_size: int
    architecture_family: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArchKey":
        return cls(**{f: d.get(f, _TOML_DEFAULTS.get(f)) for f in ARCH_KEY_FIELDS})

    def as_filter(self):
        """Wallow filter expression that matches rows with this architecture.

        Does not constrain `p` — an `ArchGroup` pools rows across every
        prime so figures sliced by prime AND figures sliced by dim both
        have multi-prime data to draw from.
        """
        expr = None
        for f in ARCH_KEY_FIELDS:
            term = F(f) == getattr(self, f)
            expr = term if expr is None else expr & term
        return expr

    def short_label(self, swept_axes: Iterable[str]) -> str:
        """Filename-friendly suffix using only the swept fields.

        Operation symbols (`+`, `-`, `*`, `/`) are mapped to their readable
        names (`add`, `sub`, `mul`, `div`) — `/` in particular would
        otherwise be parsed as a path separator and matplotlib would then
        try to save into a nonexistent directory. Other path-unsafe
        characters fall back to `_`."""
        parts = [f"{ax}={_safe_for_filename(getattr(self, ax))}"
                 for ax in swept_axes if hasattr(self, ax)]
        return "__".join(parts)


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Pull all fields a plot might want off a wallow Row.

    Fills in `n_equiv` and `dataset_bits` from (p, operation, train_fraction)
    when the row didn't annotate them — groks runs don't store these (only
    speed/capacity do), but figures keyed on `dataset_bits` need them on
    every row for the curves to align on the same x-axis.
    """
    fields = (
        "experiment_type", "p", "operation", "train_fraction", "split_type",
        "dataset_type", "n_samples", "dim", "depth", "heads", "dropout",
        "init_scale", "lr", "weight_decay", "beta1", "beta2", "batch_size",
        "max_epochs", "seed", "architecture_family",
        "param_count", "uuid", "npz_path", "n_equiv", "dataset_bits", "host",
        "saturation_step", "saturation_epoch", "saturated", "final_acc",
        "final_loss", "final_train_acc", "final_val_acc", "grokking_epoch",
        "total_bits_memorized", "final_bits_per_example",
    )
    out = {f: getattr(row, f, None) for f in fields}
    if (out["n_equiv"] is None or out["dataset_bits"] is None) and (
        out["p"] is not None and out["operation"] is not None
        and out["train_fraction"] is not None
    ):
        from .matching import compute_n_equiv
        n_eq, bits = compute_n_equiv(int(out["p"]), out["operation"],
                                     float(out["train_fraction"]))
        if out["n_equiv"] is None:
            out["n_equiv"] = n_eq
        if out["dataset_bits"] is None:
            out["dataset_bits"] = bits
    return out


@dataclass
class ArchGroup:
    key: ArchKey
    capacity_runs: list[dict[str, Any]] = field(default_factory=list)
    speed_runs: list[dict[str, Any]] = field(default_factory=list)
    groks_runs: list[dict[str, Any]] = field(default_factory=list)
    capacity_constant: float = DEFAULT_C
    capacity_constant_source: str = "fallback:consts.C"

    def has_data(self) -> bool:
        return bool(self.capacity_runs or self.speed_runs or self.groks_runs)


@dataclass(frozen=True)
class IntersectionFigure:
    """One mem/gen intersection figure family.

    Each figure family produces one PNG per `slice_field` value in each
    `ArchGroup`, with `x_field` along the x-axis and `colour_field` driving
    the scatter colour-bar. The default family (slice_field=p,
    x_field=param_count) is the per-prime intersection plot the paper
    centres on; other families swap the roles to test the same theory along
    e.g. dataset complexity.

    Filters and thresholds:
      - `max_dim`: drop rows with `dim > max_dim` from every input to this
        figure (curves, scatter, intersection, slice-value enumeration). Set
        from the YAML when wide-dim runs aren't trustworthy on the right
        tail of the x-axis. None ⇒ no cap.
      - `delay_train_threshold` / `delay_val_threshold`: train/val accuracy
        thresholds (in percent) used when re-computing per-seed delay from
        the npz traces for the scatter and the empirical onset.
      - `mem_curve_threshold` / `gen_curve_threshold`: thresholds the
        underlying *experiments* used when storing `saturation_epoch` /
        `grokking_epoch` (default 99 in both experiment runners). These
        do not recompute anything — they only label the curves so the
        legend reflects what the stored data actually represents. If you
        re-ran the experiments at a non-default early-stopping threshold,
        set these to match.
    """
    name: str
    slice_field: str
    x_field: str
    x_label: str
    colour_field: str
    colour_label: str
    max_dim: Optional[int] = None
    delay_train_threshold: float = 99.0
    delay_val_threshold: float = 99.0
    mem_curve_threshold: float = 99.0
    gen_curve_threshold: float = 99.0


_DEFAULT_INTERSECTION_FIGURE = IntersectionFigure(
    name="intersection",
    slice_field="p",
    x_field="param_count",
    x_label="Parameter count",
    colour_field="dim",
    colour_label="Dimension",
)


@dataclass(frozen=True)
class StatsConfig:
    """Knobs for the formal hypothesis-test suite written by `stats.py`.

    The defaults match the central experiment. `enabled=False` keeps the
    descriptive predictiveness CSV/scatter/per-axis plots but suppresses
    `hypothesis_tests.{json,md}` — useful for smoke configs where there
    aren't enough cells for the tests to be meaningful.

    `baseline_predictors="auto"` selects `view.swept_axes ∪ {slice_field}`
    at test time. A list overrides the auto choice — handy when only a
    subset of the swept axes belong in the sceptic baseline.
    """
    enabled: bool = True
    alpha: float = 0.05
    n_permutations: int = 10_000
    n_bootstrap: int = 10_000
    multiple_comparisons: str = "holm"
    baseline_predictors: Any = "auto"  # "auto" | list[str]


_DEFAULT_STATS_CONFIG = StatsConfig()


def _parse_stats_config(spec: dict[str, Any]) -> StatsConfig:
    raw = (spec.get("analysis") or {}).get("statistics")
    if not raw:
        return _DEFAULT_STATS_CONFIG
    return StatsConfig(
        enabled=bool(raw.get("enabled", True)),
        alpha=float(raw.get("alpha", 0.05)),
        n_permutations=int(raw.get("n_permutations", 10_000)),
        n_bootstrap=int(raw.get("n_bootstrap", 10_000)),
        multiple_comparisons=str(raw.get("multiple_comparisons", "holm")),
        baseline_predictors=raw.get("baseline_predictors", "auto"),
    )


def _parse_intersection_figures(spec: dict[str, Any]) -> list[IntersectionFigure]:
    """Read the optional `analysis.intersection_figures` block.

    Missing/empty ⇒ a one-element list with the default per-prime figure
    (today's behaviour). Each entry must supply `name`, `slice_field`, and
    `x_field`; labels and colour fields default sensibly.
    """
    raw = (spec.get("analysis") or {}).get("intersection_figures")
    if not raw:
        return [_DEFAULT_INTERSECTION_FIGURE]
    out: list[IntersectionFigure] = []
    for entry in raw:
        slice_field = entry["slice_field"]
        x_field = entry["x_field"]
        out.append(IntersectionFigure(
            name=entry.get("name", f"intersection_{slice_field}_x_{x_field}"),
            slice_field=slice_field,
            x_field=x_field,
            x_label=entry.get("x_label", x_field),
            colour_field=entry.get("colour_field", "dim"),
            colour_label=entry.get("colour_label",
                                   entry.get("colour_field", "dim").title()),
            max_dim=entry.get("max_dim"),
            delay_train_threshold=float(entry.get("delay_train_threshold", 99.0)),
            delay_val_threshold=float(entry.get("delay_val_threshold", 99.0)),
            mem_curve_threshold=float(entry.get("mem_curve_threshold", 99.0)),
            gen_curve_threshold=float(entry.get("gen_curve_threshold", 99.0)),
        ))
    return out


@dataclass
class ConfigView:
    config_name: str
    config_path: Path
    groups: list[ArchGroup]
    swept_axes: list[str]
    intersection_figures: list[IntersectionFigure] = field(
        default_factory=lambda: [_DEFAULT_INTERSECTION_FIGURE]
    )
    stats: StatsConfig = field(default_factory=lambda: _DEFAULT_STATS_CONFIG)
    # When true, every group's `capacity_constant` is pinned to `consts.C`
    # regardless of whether the config has capacity rows. Used when the
    # per-config capacity fits are too noisy to trust (see central.yaml's
    # capacity figure: small-dim curves sag below dataset complexity, so
    # the fitted slope is unreliable).
    force_default_capacity: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path, *, db_path: Optional[str] = None) -> "ConfigView":
        path = Path(path)
        with open(path) as f:
            spec = yaml.safe_load(f)

        # Lazy import: `dispatch.config` imports back from this package.
        from ..dispatch.config import expand_runs

        # Optional row pinning. A registry merged from several machines can
        # hold more rows per arch cell than the suite that defined the cell
        # (extra seeds, replicas from another host). Seed-sensitive
        # aggregates (the onset detector takes a per-cell *minimum* over
        # seeds) are only comparable across cells when every cell draws from
        # the same seed pool, so a config can pin exactly the rows its
        # figures are defined over. Applies to speed/groks rows; capacity
        # rows (single-seed by design) are left untouched.
        raw_filters = (spec.get("analysis") or {}).get("row_filters") or {}
        pin_seeds = {int(s) for s in (raw_filters.get("seeds") or [])}
        pin_hosts = {str(h) for h in (raw_filters.get("hosts") or [])}

        def _pinned(rows: list[dict]) -> list[dict]:
            if pin_seeds:
                rows = [r for r in rows if int(r.get("seed", -1)) in pin_seeds]
            if pin_hosts:
                rows = [r for r in rows if str(r.get("host")) in pin_hosts]
            return rows

        # Walk the config to discover every (arch, experiment_type) the
        # dispatcher would have produced. We don't query wallow per dispatched
        # row — one wallow query per (arch_key, experiment_type) is enough,
        # because the dispatcher uniquely defines the run.
        seen: set[ArchKey] = set()
        for run in expand_runs(spec):
            seen.add(ArchKey.from_dict(run))

        store = get_store(db_path)
        groups: list[ArchGroup] = []
        for key in sorted(seen, key=_key_sort):
            cap = [_row_to_dict(r) for r in store.where(
                (F("status") == "completed")
                & (F("experiment_type") == "capacity")
                & key.as_filter()
            ).all()]
            spd = _pinned([_row_to_dict(r) for r in store.where(
                (F("status") == "completed")
                & (F("experiment_type") == "speed")
                & key.as_filter()
            ).all()])
            grk = _pinned([_row_to_dict(r) for r in store.where(
                (F("status") == "completed")
                & (F("experiment_type") == "groks")
                & key.as_filter()
            ).all()])
            groups.append(ArchGroup(
                key=key,
                capacity_runs=cap, speed_runs=spd, groks_runs=grk,
            ))

        # Resolve capacity constants. Each group's C comes from its own
        # capacity rows when available; otherwise from the closest-matching
        # capacity-bearing group in the view. Fallback to consts.C.
        # When `analysis.use_default_capacity_constant: true`, pin every
        # group to `consts.C` and skip the fit entirely.
        force_default = bool(
            (spec.get("analysis") or {}).get("use_default_capacity_constant", False)
        )
        for g in groups:
            if force_default:
                g.capacity_constant = DEFAULT_C
                g.capacity_constant_source = "fallback:consts.C (forced by config)"
            else:
                g.capacity_constant, g.capacity_constant_source = _resolve_C(g, groups)

        swept = _detect_swept_axes(groups)
        figures = _parse_intersection_figures(spec)
        stats_cfg = _parse_stats_config(spec)
        name = spec.get("name") or path.stem
        return cls(config_name=name, config_path=path,
                   groups=groups, swept_axes=swept,
                   intersection_figures=figures,
                   stats=stats_cfg,
                   force_default_capacity=force_default)

    def iter_groups(self) -> Iterator[ArchGroup]:
        return iter(self.groups)


def _key_sort(k: ArchKey) -> tuple:
    return tuple(getattr(k, f) for f in ARCH_KEY_FIELDS)


# Architecture-only fields (excluding wd and dropout, which are the two axes
# that conventionally drift between capacity and speed/groks defaults).
_ARCH_ONLY_FIELDS = tuple(
    f for f in ARCH_KEY_FIELDS if f not in ("weight_decay", "dropout")
)


def _resolve_C(group: "ArchGroup", all_groups: list["ArchGroup"]) -> tuple[float, str]:
    """Choose a capacity constant for `group`.

    Order of preference:
      1. Group's own capacity runs.
      2. Closest-matching other group with capacity runs — score on
         architecture > weight_decay > dropout.
      3. `consts.C`.
    """
    own = fit_capacity_slope(group.capacity_runs) if group.capacity_runs else None
    if own is not None:
        return own, f"measured(own group, n={len(group.capacity_runs)} rows)"

    candidates = [g for g in all_groups if g is not group and g.capacity_runs]
    if not candidates:
        return DEFAULT_C, "fallback:consts.C"

    def score(g: "ArchGroup") -> tuple[int, int, int]:
        arch_match = sum(
            1 for f in _ARCH_ONLY_FIELDS
            if getattr(group.key, f) == getattr(g.key, f)
        )
        wd_match = int(group.key.weight_decay == g.key.weight_decay)
        do_match = int(group.key.dropout == g.key.dropout)
        return (arch_match, wd_match, do_match)

    best = max(candidates, key=score)
    slope = fit_capacity_slope(best.capacity_runs)
    if slope is None:
        return DEFAULT_C, "fallback:consts.C"
    label = f"wd={best.key.weight_decay}, dropout={best.key.dropout}"
    return slope, f"measured(matched group: {label})"


def _detect_swept_axes(groups: list[ArchGroup]) -> list[str]:
    """Identifying fields whose value differs across groups. `p` is not in
    `ARCH_KEY_FIELDS` so it's never a swept axis here — figures slice on
    prime themselves via `IntersectionFigure.slice_field`."""
    if not groups:
        return []
    out = []
    for ax in ARCH_KEY_FIELDS:
        values = {getattr(g.key, ax) for g in groups}
        if len(values) > 1:
            out.append(ax)
    return out


def load_npz(row: dict[str, Any]):
    """Load the trace.npz for a wallow row dict.

    Prefers the `npz_path` annotation on the row; falls back to the canonical
    path computed from `experiment_type` + `uuid`. Returns the npz file
    object (caller is responsible for `.close()` if they care).
    """
    import numpy as np

    explicit = row.get("npz_path")
    if explicit:
        p = Path(explicit)
        if p.exists():
            return np.load(p, allow_pickle=False)
    p = npz_path_for_row(row)
    return np.load(p, allow_pickle=False)
