#!/usr/bin/env python3
"""Refit BOTH clocks for each seed reducer and draw isolated cone comparisons.

The default primary cohort requires ten observed crossings per timescale cell.
The observed cohort is a selection sensitivity. Neither is a censoring model.
Canonical paper figures, coefficients, and the latest-cone pointer are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from check_law_complete_ten_seeds import K, crossing, fit, fold
from cone_data import aggregate
from grokking_capacity.analysis.aggregate import onset_summary, reduce_values

DEFAULT_OUT = ROOT / 'results/cone_matched_aggregations_20261106'
INPUT = ROOT / 'results/law_complete_ten_seed_sensitivity_20261106/selected_runs.csv'
RUNS = ROOT / 'results/cone_ten_seeds_h100_20261106/figure/runs.json'
PRIOR = ROOT / 'results/seed_aggregation_sensitivity_20261106/laws/comparison.json'
CENTRAL = (97, 101, 103, 107, 109, 113, 127, 131, 137, 139)
C_OLD = 2.16


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def save_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def fit_clock_pair(data, metric, cohort):
    """Same reducer, selection and cell weights for both clocks; no cap values."""
    points = {}
    for kind in ('mem', 'gen'):
        chosen = []
        for (p, P), group in data[data.kind == kind].groupby(['p', 'P']):
            assert len(group) == 10 and set(group.seed) == set(range(42, 52))
            complete = bool(group.observed.all())
            assert bool(group.included_complete_ten_fit.all()) == complete
            if cohort == 'complete' and not complete:
                continue
            observed = group[group.observed]
            if observed.empty:
                continue
            chosen.append({'p': int(p), 'P': int(P),
                           'f': float(group.f_original.iloc[0]),
                           't': reduce_values(observed.event_epoch, metric),
                           'n_observed': len(observed)})
        points[kind] = chosen
    mem, gen = fit('mem', points['mem']), fit('gen', points['gen'])
    return {'cohort': cohort, 'metric': metric, 'mem': mem, 'gen': gen,
            'fold': fold(mem, gen, C_OLD),
            'f_original_at_fold': gen['beta'] / mem['gamma_original']}, points


def branches(p, law):
    """Solve both roots in log-parameter space, including a double root at fold."""
    m, g = law['mem'], law['gen']
    qk = m['gamma_original'] / C_OLD * K(p)
    beta = g['beta']
    if beta <= 0 or qk <= 0:
        return None, None
    pivot = math.log(qk / beta)
    def gap(logP):
        return g['lnA'] + g['a'] * math.log(p) - beta * logP - m['ln_b'] - qk * math.exp(-logP)
    maximum = gap(pivot)
    if maximum < -1e-10:
        return None, None
    if abs(maximum) <= 1e-10:
        return math.exp(pivot), math.exp(pivot)
    lower = math.exp(brentq(gap, pivot - 50, pivot, xtol=1e-13))
    upper = math.exp(brentq(gap, pivot, pivot + 500, xtol=1e-13))
    return lower, upper


def get_onsets(cells, limit=256):
    result = {}
    for p in sorted({c['p'] for c in cells}):
        curve = {c['param_count']: c['aggregate_delay'] for c in cells
                 if c['p'] == p and (limit is None or c['dim'] <= limit)}
        result[p] = onset_summary(curve)
    return result


def agreement(law, onsets):
    observed = np.array([onsets[p]['value'] for p in CENTRAL], dtype=float)
    predicted = np.array([crossing(p, law['mem'], law['gen'], C_OLD) for p in CENTRAL])
    assert np.isfinite(observed).all() and np.isfinite(predicted).all()
    x, y = np.log10(predicted), np.log10(observed)
    r = y - x
    delta = float(np.median(r))
    covariance = np.mean((x - x.mean()) * (y - y.mean()))
    ccc = 2 * covariance / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)
    return {'n': len(CENTRAL), 'spearman_rho': float(spearmanr(x, y).statistic),
            'raw_CCC': float(ccc), 'raw_MAE_dex': float(np.mean(abs(r))),
            'calibration_offset_dex': delta,
            'in_sample_calibrated_MAE_dex': float(np.mean(abs(r - delta))),
            'offset_sd_dex': float(np.std(r, ddof=1))}


def verify_curve_domains(fig):
    fitted = extrapolated = 0
    for ax in fig.axes:
        assert ax.get_title() == ''
        for line in ax.lines:
            if line.get_gid() == 'fitted_prime_curve':
                assert max(line.get_xdata()) <= max(CENTRAL)
                fitted += 1
            elif line.get_gid() == 'extrapolated_prime_curve':
                assert min(line.get_xdata()) == max(CENTRAL)
                assert line.get_linestyle() == ':'
                extrapolated += 1
    assert fitted and extrapolated


def draw_cones(out, laws, cells, metrics, cohort, zoom=False):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from grokking_capacity.analysis._figure_style import save_paper_figure

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.1 * len(metrics), 5.4),
                             sharex=True, sharey=True, squeeze=False, layout='constrained')
    axes = list(axes.flat)
    cmap, norm = plt.get_cmap('viridis'), LogNorm(.1, 1500)
    right = 201 if zoom else max(205, math.ceil(max(v['fold'] for v in laws.values()) / 10) * 10)
    for ax, metric in zip(axes, metrics):
        cc, law = cells[metric], laws[f'{cohort}_{metric}']
        zero = [c for c in cc if c['state'] == 'zero']
        positive = [c for c in cc if c['state'] == 'positive']
        unknown = [c for c in cc if c['unresolved_seeds']]
        ax.axvspan(97, 139, color='#eef4fa', zorder=0)
        ax.axvline(139, color='#6e7c8c', lw=.9, zorder=1)
        if right > 201:
            ax.axvspan(199, right, color='#edf0f4', zorder=0)
        ax.scatter([c['p'] for c in zero], [c['param_count'] for c in zero],
                   s=12, facecolors='none', edgecolors='#979fa9', linewidths=.5, alpha=.7, zorder=2)
        ax.scatter([c['p'] for c in positive], [c['param_count'] for c in positive],
                   c=[c['aggregate_delay'] for c in positive], cmap=cmap, norm=norm,
                   s=16, edgecolors='#243943', linewidths=.25, zorder=3)
        ax.scatter([c['p'] for c in unknown], [c['param_count'] for c in unknown],
                   color='#b76814', marker='x', s=23, linewidths=.75, zorder=4)
        ps = np.unique(np.r_[np.linspace(97, law['fold'], 650), 139, 199])
        ps = ps[ps <= law['fold']]
        yy = np.array([branches(p, law)[0] for p in ps])
        # The prime fit ends at 139, even though measurements continue to 199.
        measured = ps <= 139
        ax.plot(ps[measured], yy[measured], color='#c7353b', lw=1.65, zorder=5, gid='fitted_prime_curve')
        extrapolated = ps >= 139
        ax.plot(ps[extrapolated], yy[extrapolated], color='#c7353b', ls=':', lw=1.9, zorder=5, gid='extrapolated_prime_curve')
        ax.axvline(law['fold'], color='#c7353b', ls='--', lw=.85, alpha=.7, zorder=2)
        ax.set(yscale='log', xlim=(95, right), ylim=(1.1e4, 2.1e8),
               xlabel=f'Prime $p$\n{"Minimum" if metric == "min" else metric.capitalize()} | $p^* = {law["fold"]:.1f}$')
        ax.set_xticks([97, 113, 139, 163, 179, 199] if zoom else [97, 139, 163, 199, 227, 243])
        ax.grid(axis='y', alpha=.25, linewidth=.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel('Parameters $P$')
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, fraction=.017, pad=.015)
    cb.set_label('Named delay aggregate (epochs)')
    handles = [Line2D([], [], marker='o', ls='', color=cmap(.5), label='Positive aggregate'),
               Line2D([], [], marker='o', ls='', markerfacecolor='none', markeredgecolor='#979fa9', label='Zero aggregate'),
               Line2D([], [], marker='x', ls='', color='#b76814', label='At least one unresolved seed'),
               Line2D([], [], color='#c7353b', lw=1.6, label='Crossing over fitted prime range'),
               Line2D([], [], color='#c7353b', ls=':', lw=1.9, label='Extrapolation beyond 139'),
               Line2D([], [], color='#c7353b', ls='--', label='Fitted terminal prime $p^*$')]
    fig.legend(handles=handles, fontsize=8.5)
    example = laws[f'{cohort}_{metrics[0]}']
    eligibility = 'All ten crossings observed per timescale cell' if cohort == 'complete' else 'Available observed crossings per timescale cell'
    notes = ['Fit primes: 97, 101, 103, 107, 109, 113, 127, 131, 137, 139 (blue region). Boundary: 139; first held-out prime: 149. Dotted curves are extrapolations.',
             f'Each panel applies its named reducer to BOTH timescales and to the seed delays; each timescale cell has equal fit weight.',
             f'{eligibility}: {example["mem"]["n_observations"]} memorisation and {example["gen"]["n_observations"]} generalisation cells; A100 fits, primes 97–139.',
             'Same 1,736 ten-seed cells in every panel: 1,407 A100 and 329 H100. Orange crosses retain all 102 cells with unresolved seeds.',
             'Raw lower crossing, without onset-offset calibration; the upper branch is in the companion boundary figure. Fits use 99%; delays use training 99%, validation 98%.',
             ('Measurements end at 199. ' if zoom else 'Measurements end at 199; grey space has no observations. ') + 'Fold locations extrapolate below the fitted capacity-fraction window.']
    stem = f'cone_{cohort}_matched' + ('_measured_range' if zoom else '')
    verify_curve_domains(fig)
    save_paper_figure(fig, out / f'{stem}.pdf', notes=[fig.text(0, 0, n, fontsize=8.5) for n in notes], max_columns=3)
    plt.close(fig)


def draw_boundaries(out, laws, metrics, cohorts):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from grokking_capacity.analysis._figure_style import save_paper_figure
    palette = ['#264b7d', '#07847a', '#d37516', '#a3468b']
    fig, axes = plt.subplots(1, len(cohorts), figsize=(6.5 * len(cohorts), 5),
                             squeeze=False, sharey=True, layout='constrained')
    for ax, cohort in zip(axes.flat, cohorts):
        ax.axvspan(97, 139, color='#eef4fa', zorder=0)
        ax.axvline(139, color='#6e7c8c', lw=.9, zorder=1)
        ax.axvspan(199, 250, color='#edf0f4', zorder=0)
        for index, metric in enumerate(metrics):
            law = laws[f'{cohort}_{metric}']; color = palette[index % len(palette)]
            pp = np.unique(np.r_[np.linspace(97, law['fold'], 900), 139])
            bb = np.array([branches(p, law) for p in pp])
            inside, outside = pp <= 139, pp >= 139
            ax.plot(pp[inside], bb[inside, 0], color=color, lw=1.65, gid='fitted_prime_curve')
            ax.plot(pp[inside], bb[inside, 1], color=color, lw=1.35, ls='--', gid='fitted_prime_curve')
            ax.plot(pp[outside], bb[outside, 0], color=color, lw=1.9, ls=':', gid='extrapolated_prime_curve')
            ax.plot(pp[outside], bb[outside, 1], color=color, lw=1.9, ls=':', gid='extrapolated_prime_curve')
            ax.scatter([law['fold']], [bb[-1, 0]], color=color, s=27, zorder=3)
        ax.set(yscale='log', ylim=(8e4, 2e9), xlim=(95, 250),
               xlabel='Prime $p$\n' + ('Complete ten-crossing cells' if cohort == 'complete' else 'Available observed crossings'))
        ax.set_xticks([97, 139, 163, 199, 227, 243]); ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes.flat[0].set_ylabel('Crossing parameters $P$')
    handles = [Line2D([], [], color=palette[i % len(palette)], lw=2, label=('Minimum' if m == 'min' else m.capitalize())) for i, m in enumerate(metrics)]
    handles += [Line2D([], [], color='#555555', label='Lower / onset branch'),
                Line2D([], [], color='#555555', ls='--', label='Upper / re-entry branch'),
                Line2D([], [], color='#555555', ls=':', lw=1.9, label='Both branches beyond fitted primes'),
                Line2D([], [], color='#555555', ls='', marker='o', label='Branches meet at $p^*$')]
    fig.legend(handles=handles, fontsize=9)
    notes = ['Fit primes: 97, 101, 103, 107, 109, 113, 127, 131, 137, 139 (blue region). First held-out prime: 149; all curves beyond 139 are dotted.',
             'Both roots of each matched refit are shown. The fitted generalisation time exceeds memorisation time between the two branches.',
             'These are model predictions, not a proof that the corresponding seed-delay aggregate is positive. No empirical onset calibration is applied.',
             'Fits use primes 97–139 and original capacity fraction 0.08–1. All folds and upper branches extrapolate below that fraction window; grey space is beyond measured prime 199.']
    verify_curve_domains(fig)
    save_paper_figure(fig, out / 'both_crossing_branches.pdf', notes=[fig.text(0, 0, n, fontsize=8.5) for n in notes], max_columns=3)
    plt.close(fig)



def write_report(out, summary, predictions, high_prime_rows):
    laws = summary['laws']
    lines = [
        'Matched aggregation of BOTH timescales: cone reanalysis, 2026-11-06',
        '',
        'The previous three-panel delay comparisons held the law fixed. These new figures instead',
        'refit both clocks separately for every delay statistic. Each panel uses the same reducer',
        'for memorisation time, generalisation time, and the plotted ten-seed delay.',
        '',
        'METHOD AND SCOPE',
        'For statistic S, reduce times across seeds within each (prime, parameter-count) cell,',
        'then fit ln S(T_mem) = ln b + gamma f and ln S(T_gen) = ln A + a ln p - beta ln P.',
        'The reducer is applied BEFORE taking the logarithm. Both regressions give each cell',
        'equal weight. min, median and mean therefore use identical cells within each cohort.',
        'Exact fit primes: 97,101,103,107,109,113,127,131,137,139. The largest is139;',
        '149 is the first held-out prime. All new figures mark139 and use dotted curves',
        'beyond139. This identifies the extrapolation test separately from the fitting range.',
        'Fits use A100 runs, original 99% events, width <=256, and',
        'the frozen original capacity-fraction window 0.08–1 (C=2.16 convention).',
        'The current H100 capacity constant rescales gamma and C together; gamma/C, crossings',
        'and terminal primes are unchanged by this re-expression.',
        '',
        'Complete cohort: each timescale separately requires all ten seeds 42–51 to cross;',
        '67 memorisation cells and 269 generalisation cells. This conditions on finishing',
        'before the cap. Observed cohort: use only observed events in 70 and 326 cells,',
        'respectively, with variable numbers of observed seeds. Neither treats a cap as an',
        'observed event, and neither is a correction for censoring or completion selection.',
        '',
        'Every cone retains the same 1,736 cells and ten planned/run seeds per cell. There are',
        '1,407 A100 cells and 329 H100 cells, with no hardware pooling within a cell. All 102',
        'cells containing unresolved outcomes retain orange crosses. Delays retain training99',
        'versus validation98 thresholds. Zero delay does not mean failed generalisation.',
        '',
        'FOLDS AND FITS',
        'cohort/statistic | terminal prime [95% prime-cluster bootstrap] | mem R2 | gen R2 | beta',
    ]
    for name, v in laws.items():
        ci = v['fold_prime_bootstrap_95']
        lines.append(f"{name} | {v['fold']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | "
                     f"{v['mem']['r2']:.4f} | {v['gen']['r2']:.4f} | {v['gen']['beta']:.6f}")
    lines += ['', 'Bootstrap resamples the ten prime clusters with the same draws for every fit;',
              'eligibility is held fixed. Intervals exclude seed resampling, censoring-model',
              'uncertainty and uncertainty in the validity of the extrapolated functional form.',
              'Paired fold contrasts (first statistic minus second):']
    for row in summary['paired_fold_differences']:
        lo, hi = row['paired_prime_bootstrap_95']
        lines.append(f"{row['cohort']} {row['contrast']}: {row['fold_difference']:.3f}, 95% [{lo:.3f}, {hi:.3f}]")
    lines += [
        '', 'CROSSING SHAPE',
        'Let q=gamma/C and K(p)=p(p-1)log2(p+2)/2. Setting the clocks equal gives',
        'x - beta ln(x) = ln A - ln b + a ln p - beta ln(qK), where x=qK/P.',
        'There are two roots before the terminal prime, one double root at x=beta, and no',
        'roots afterwards. The predicted positive clock gap lies BETWEEN the roots.',
        'The main comparison keeps the usual lower/onset branch. both_crossing_branches.pdf',
        'also displays the upper/re-entry branch, showing that it moves substantially.',
        '',
        'At the fold, f=beta/gamma. These six values range from about 0.012 to 0.026 in the',
        'original convention, all below the fitted lower bound 0.08. Every upper branch lies',
        'below that bound too. Curves beyond prime139 also extrapolate in prime. Measurements',
        'end at prime199; the complete-cohort terminal primes203–243 are not measured endpoints.',
        '',
        'The new lower crossings at p149 and p197, in raw parameter count:',
    ]
    for r in predictions:
        if r['p'] not in (149, 197):
            continue
        val = 'no real crossing' if r['lower_crossing_P'] is None else f"{r['lower_crossing_P']:,.0f}"
        lines.append(f"{r['cohort']}/{r['metric']} p={r['p']}: {val}")
    lines += [
        '', 'DISCREPANCIES THAT REMAIN',
        'At p197, using the FULL raw predicted interval between both branches:',
    ]
    for r in high_prime_rows:
        if r['p'] == 197:
            lines.append(f"{r['fit']}: {r['predicted_positive_cells']} measured cells inside the predicted cone; "
                         f"{r['predicted_positive_observed_positive']} have positive aggregates, "
                         f"{r['predicted_positive_observed_zero']} zero, "
                         f"{r['predicted_positive_unresolved_aggregate']} unresolved. "
                         f"Across all widths: {r['observed_positive_cells']} positive cells.")
    lines += [
        'Thus complete-ten minimum refitting predicts a nonempty p197 cone that is absent',
        'under the all-ten-positive criterion. Mean looks better inside its p197 predicted',
        'interval, but additional positive cells occur outside that interval. This does not',
        'establish an exact classification boundary. At p197 the identified median delays',
        'are at most1 epoch and means at most1.4 epochs; unresolved means remain unbounded.',
        '',
        'Refitting the same statistic is a consistent comparison, but it is NOT an identity:',
        'S(max(T_val-T_train,0)) generally differs from max(S(T_val)-S(T_train),0).',
        'For two seeds with train times[1,100] and validation times[2,99], delays are[1,0].',
        'Their minimum is0 although min(validation)-min(train)=1. Their mean is0.5 although',
        'mean(validation)-mean(train)=0. Median has the same failure in this two-seed example.',
        'Our memorisation and generalisation clocks also come from different experiments.',
        'A rigorous probability/all-seeds cone needs the joint seed-level timing distribution',
        'or a direct calibrated model of positive-delay probability, not marginal reducers alone.',
        '', 'ONSET AGREEMENT AND CALIBRATION',
        'The following compares the GLOBAL FITTED lower crossing against the empirical onset',
        'on the fixed width<=256 domain at the ten central primes. These are descriptive',
        'fit/calibration diagnostics, NOT replacements for the paper main test, which compares',
        'crossings of measured time curves. No confirmatory p-values are claimed here.',
        'cohort/statistic | rank rho | raw MAE(dex) | refitted offset(dex) | calibrated MAE(dex)',
    ]
    for name, v in laws.items():
        r = v['central_onset_agreement']
        lines.append(f"{name} | {r['spearman_rho']:.4f} | {r['raw_MAE_dex']:.4f} | "
                     f"{r['calibration_offset_dex']:.4f} | {r['in_sample_calibrated_MAE_dex']:.4f}")
    lines += [
        'Each offset is refitted using the named onset statistic, without p149. It is not',
        'the old minimum-delay offset reused for a new metric. Primary cones show raw crossings.',
        'This offset moves P vertically but does not change the terminal prime. predictions.csv',
        'also provides the calibrated crossings and p149 residuals:',
    ]
    for r in predictions:
        if r['p'] == 149:
            lines.append(f"{r['cohort']}/{r['metric']}: empirical onset {r['observed_onset_dim_le_256']:,.0f}, "
                         f"calibrated prediction {r['calibrated_lower_P']:,.0f}, "
                         f"residual {r['calibrated_residual_dex']:+.4f} dex")
    lines += [
        'Median is closest at this one held-out prime after recalibration. Selecting it using',
        'this result would be exploratory. It does not justify a general superiority claim.',
        'The onset depends on the prespecified width domain: on the expanded p197 grid, later',
        'zero-delay cells remove the mean positive tail. Do not replace a tail onset with the',
        'first isolated positive cell or silently change domains.',
        '', 'INTERPRETATION AND VALIDATION',
        'The original published law did not use a minimum reducer for both clocks: it used',
        'per-run log regression for memorisation and log of cell arithmetic means for',
        'generalisation. The new fits intentionally use equal-weight cell summaries for both.',
        'Comparing the three new metrics holds that weighting fixed. The original asymmetric',
        'complete-cell endpoint203.845 becomes203.316 for the consistent mean/mean fit;',
        'the corresponding broader-cohort endpoints are167.512 and167.886.',
        '',
        'Both timescales, both roots, the terminal prime and the optional calibration now',
        'vary with the named statistic. Mean/median do not automatically fix the cone or',
        'establish better statistical performance. Cohort selection changes the conclusion',
        'substantially and must be reported alongside aggregation choice.',
        'The script reproduces prior coefficients, checks them with NumPy least squares,',
        'verifies equality of the two clocks at both roots, and preserves all48 protected',
        'publication artifacts by hash. Figure titles are absent; explanations are below.',
        'All figures are exported as vector PDF plus PNG. No training or publication edits.',
        '', 'Reproduce: python scripts/plot_cone_matched_aggregations.py',
        'Optional --metrics min median mean (or supported monotone quantile:<q>),',
        '--cohort complete|observed|both, --bootstrap N, --out PATH.',
        '',
    ]
    (out / 'report.txt').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', nargs='+', default=['min', 'median', 'mean'])
    parser.add_argument('--cohort', choices=['both', 'complete', 'observed'], default='both')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error('--bootstrap must be positive')
    out = args.out.resolve()
    assert out != ROOT / 'results/cone_ten_seeds_h100_20261106/figure'
    out.mkdir(parents=True, exist_ok=True)
    protected_manifest = json.loads((ROOT / 'results/seed_aggregation_sensitivity_20261106/protected_before.json').read_text())
    protected = {p: sha(ROOT / p) for p in protected_manifest}
    data = pd.read_csv(INPUT)
    assert set(data.p) == set(CENTRAL) and set(data.gpu_type) == {'A100'}
    assert data.f_original.between(.08, 1).all()
    runs = json.loads(RUNS.read_text())
    variants = {m: aggregate(runs, m) for m in args.metrics}
    for cc in variants.values():
        assert len(cc) == 1736 and all(c['n_seeds'] == 10 and c['seeds'] == list(range(42, 52)) for c in cc)
        assert sum(c['unresolved_seeds'] > 0 for c in cc) == 102
    cohorts = ['complete', 'observed'] if args.cohort == 'both' else [args.cohort]
    prior = json.loads(PRIOR.read_text())['unified_cell_reducers']
    C_current = json.loads((ROOT / 'results/capacity_h100_matched_20261103/forecast_coefficients.json').read_text())['C_convention']
    laws, points, prediction_rows, curve_rows, score_rows = {}, {}, [], [], []
    high_prime_rows = []
    max_root_residual = 0.
    for cohort in cohorts:
        for metric in args.metrics:
            name = f'{cohort}_{metric}'
            law, pp = fit_clock_pair(data, metric, cohort)
            laws[name], points[name] = law, pp
            # Independently check the OLS coefficients with NumPy least squares.
            for kind, rr in pp.items():
                X = np.array([[1., r['f']] if kind == 'mem' else
                              [1., math.log(r['p']), -math.log(r['P'])] for r in rr])
                coefficients = np.linalg.lstsq(X, np.log([r['t'] for r in rr]), rcond=None)[0]
                keys = ('ln_b', 'gamma_original') if kind == 'mem' else ('lnA', 'a', 'beta')
                assert np.allclose(coefficients, [law[kind][k] for k in keys], rtol=1e-11, atol=1e-11)
            for prime in (149, 193, 197, 199):
                cc = [c for c in variants[metric] if c['p'] == prime]
                m, g = law['mem'], law['gen']
                inside = [c for c in cc if g['lnA'] + g['a'] * math.log(prime)
                          - g['beta'] * math.log(c['param_count']) - m['ln_b']
                          - m['gamma_original'] / C_OLD * K(prime) / c['param_count'] > 0]
                counts = Counter(c['state'] for c in inside)
                high_prime_rows.append({'fit': name, 'p': prime, 'cells': len(cc),
                    'predicted_positive_cells': len(inside),
                    'predicted_positive_observed_positive': counts['positive'],
                    'predicted_positive_observed_zero': counts['zero'],
                    'predicted_positive_unresolved_aggregate': counts['unresolved'],
                    'observed_positive_cells': sum(c['state'] == 'positive' for c in cc),
                    'max_exact_delay': max((c['aggregate_delay'] for c in cc
                                            if c['aggregate_delay'] is not None), default=None)})
            law['mem']['gamma_current_C'] = law['mem']['gamma_original'] * C_current / C_OLD
            law['mem']['b'] = math.exp(law['mem']['ln_b'])
            if name in prior:
                for kind, keys in [('mem', ('ln_b', 'gamma_original')), ('gen', ('lnA', 'a', 'beta'))]:
                    for key in keys:
                        assert np.isclose(law[kind][key], prior[name][kind][key], rtol=1e-12, atol=1e-12)
                assert np.isclose(law['fold'], prior[name]['fold'], rtol=1e-12)
            onsets = get_onsets(variants[metric])
            law['central_onset_agreement'] = agreement(law, onsets)
            score_rows.append({'cohort': cohort, 'metric': metric, **law['central_onset_agreement']})
            delta = law['central_onset_agreement']['calibration_offset_dex']
            for p in sorted(set(onsets) | {int(math.floor(law['fold'])), 251}):
                lo, hi = branches(p, law)
                onset = onsets.get(p, {'value': None, 'status': 'not_measured'})
                calibrated = lo * 10 ** delta if lo is not None else None
                prediction_rows.append({'cohort': cohort, 'metric': metric, 'p': p,
                    'lower_crossing_P': lo, 'upper_crossing_P': hi, 'calibrated_lower_P': calibrated,
                    'observed_onset_dim_le_256': onset['value'], 'onset_status': onset['status'],
                    'calibrated_residual_dex': math.log10(onset['value'] / calibrated) if onset['value'] and calibrated else None})
            for p in np.linspace(97, law['fold'], 600):
                lo, hi = branches(p, law)
                assert lo is not None and hi is not None and lo <= hi * (1 + 1e-10)
                if p < law['fold'] - 1e-6:
                    assert np.isclose(lo, crossing(p, law['mem'], law['gen'], C_OLD), rtol=1e-9)
                for P in (lo, hi):
                    residual = law['gen']['lnA'] + law['gen']['a'] * math.log(p) - law['gen']['beta'] * math.log(P) - law['mem']['ln_b'] - law['mem']['gamma_original'] / C_OLD * K(p) / P
                    max_root_residual = max(max_root_residual, abs(residual))
                curve_rows.append({'cohort': cohort, 'metric': metric, 'p': p, 'lower_P': lo, 'upper_P': hi,
                                   'within_fit_prime_range': p <= 139})
            pd.DataFrame([{'kind': kind, **r} for kind, rr in pp.items() for r in rr]).to_csv(out / f'fit_cells_{name}.csv', index=False)
    assert max_root_residual < 1e-8
    # Paired prime-cluster bootstrap: identical cluster draws for every fit.
    rng = np.random.default_rng(20261106)
    bootstrap = []
    for rep in range(args.bootstrap):
        sample = rng.choice(CENTRAL, len(CENTRAL), replace=True)
        row = {'replicate': rep}
        for name, pp in points.items():
            try:
                row[name] = fold(fit('mem', pp['mem'], sample), fit('gen', pp['gen'], sample), C_OLD)
            except (ValueError, AssertionError, np.linalg.LinAlgError):
                row[name] = None
        bootstrap.append(row)
    boot = pd.DataFrame(bootstrap)
    differences = []
    for name, law in laws.items():
        bb = boot[name].dropna()
        law['fold_prime_bootstrap_95'] = np.quantile(bb, [.025, .975]).tolist()
        law['bootstrap_finite'] = len(bb)
    for cohort in cohorts:
        for i, a in enumerate(args.metrics):
            for b in args.metrics[i + 1:]:
                aa, bb = f'{cohort}_{a}', f'{cohort}_{b}'
                diff = (boot[aa] - boot[bb]).dropna()
                differences.append({'cohort': cohort, 'contrast': f'{a} minus {b}',
                    'fold_difference': laws[aa]['fold'] - laws[bb]['fold'],
                    'paired_prime_bootstrap_95': np.quantile(diff, [.025, .975]).tolist()})
    boot.to_csv(out / 'fold_bootstrap.csv', index=False)
    pd.DataFrame(prediction_rows).to_csv(out / 'predictions.csv', index=False)
    pd.DataFrame(curve_rows).to_csv(out / 'crossing_curves.csv', index=False)
    pd.DataFrame(score_rows).to_csv(out / 'central_onset_agreement.csv', index=False)
    pd.DataFrame(high_prime_rows).to_csv(out / 'high_prime_diagnostics.csv', index=False)
    for metric, cc in variants.items():
        pd.DataFrame(cc).to_csv(out / f'cells_{metric}.csv', index=False)
    summary = {'laws': laws, 'paired_fold_differences': differences, 'C_current': C_current,
        'C_original': C_OLD, 'fit_window_original_f': [.08, 1.],
        'fit_primes': CENTRAL, 'fitting': 'OLS on log of the named per-cell time statistic; one equal-weight point per cell, separately for each clock.',
        'thresholds': {'timescale_fits': 'original 99% events', 'cone_delays': 'training99, validation98'},
        'calibration': 'Median log10 observed/predicted onset over ten central primes, separately for each metric and refit. Primary cones show RAW lower crossing.',
        'censoring': 'Unobserved events are never replaced by cap. Complete cohort conditions on all ten events; observed cohort uses available events. Neither is censoring-corrected.',
        'cell_counts': {m: dict(Counter(c['state'] for c in cc)) for m, cc in variants.items()},
        'interpretation': 'A crossing of separately aggregated clocks need not equal the zero boundary of an aggregate of paired clipped delays. Results are exploratory, not new confirmatory tests.'}
    save_json(out / 'summary.json', summary)
    write_report(out, summary, prediction_rows, high_prime_rows)
    import matplotlib
    matplotlib.use('Agg')
    for cohort in cohorts:
        draw_cones(out, laws, variants, args.metrics, cohort)
        draw_cones(out, laws, variants, args.metrics, cohort, zoom=True)
    draw_boundaries(out, laws, args.metrics, cohorts)
    assert all(sha(ROOT / p) == digest for p, digest in protected.items())
    validation = {'both_clocks_independently_refitted': True, 'numpy_lstsq_coefficients_verified': True, 'prior_coefficients_reproduced': True,
        'max_abs_log_timescale_difference_at_roots': max_root_residual,
        'curve_domains_verified': {'fit_prime_max': max(CENTRAL), 'first_held_out_prime': 149, 'extrapolation_linestyle': 'dotted'},
        'exact_ten_seed_cells': len(next(iter(variants.values()))), 'unresolved_cells_preserved': 102,
        'all_training_for_law_fits_A100': True, 'protected_artifacts_unchanged': protected,
        'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [INPUT, RUNS, PRIOR]},
        'source_sha256': sha(__file__),
        'figures': {p.name: sha(p) for p in sorted([*out.glob('*.pdf'), *out.glob('*.png')])}}
    save_json(out / 'validation.json', validation)
    shutil.copy2(__file__, out / Path(__file__).name)
    print(json.dumps(clean({'laws': laws, 'paired_fold_differences': differences,
                           'root_residual': max_root_residual, 'out': str(out)}), indent=2))


if __name__ == '__main__':
    main()
