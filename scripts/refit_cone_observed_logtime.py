#!/usr/bin/env python3
"""Fit both clocks to all uncensored individual seeds; plot the selected delay aggregate.

Ordinary least squares in log time, with equal weight per observed run.
No seed aggregation, censoring likelihood, or uncertainty model is fitted.
"""
from grokking_capacity.analysis.timescale_laws import normalise_timescale_laws
from collections import Counter
import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from check_law_complete_ten_seeds import fit, fold, crossing
from cone_data import aggregate
from plot_cone_matched_aggregations import (
    ROOT, INPUT, RUNS, CENTRAL, C_OLD, branches, sha, save_json,
    get_onsets, agreement, verify_curve_domains,
)
from grokking_capacity.analysis._figure_style import save_paper_figure

BASELINE = ROOT / 'results/cone_observed_logtime_20261106'
OUT = ROOT / 'results/cone_observed_logtime_min_20261106'


def cone_figure(law, cells, metric, *, output_name=None):
    label = {'min': 'minimum', 'median': 'median', 'mean': 'mean'}.get(metric, metric)
    fig, ax = plt.subplots(figsize=(9.5, 6.0), layout='constrained')
    cmap, norm = plt.get_cmap('viridis'), LogNorm(.1, 1500)
    ax.axvline(139, color='#6e7c8c', lw=.9, zorder=1)
    for state in ('zero', 'positive'):
        cc = [c for c in cells if c['state'] == state]
        kw = dict(s=17, facecolors='none', edgecolors='#969fa9', linewidths=.6, alpha=.7)
        if state == 'positive':
            kw = dict(s=22, c=[c['aggregate_delay'] for c in cc], cmap=cmap, norm=norm,
                      edgecolors='#263c45', linewidths=.3)
        ax.scatter([c['p'] for c in cc], [c['param_count'] for c in cc], zorder=2, **kw)
    missing = [c for c in cells if c['unresolved_seeds']]
    ax.scatter([c['p'] for c in missing], [c['param_count'] for c in missing],
               color='#b76814', marker='x', s=30, linewidths=.9, zorder=3)
    ps = np.unique(np.r_[np.linspace(97, law['fold'], 800), 139])
    curve = np.array([branches(p, law)[0] for p in ps])
    inside, outside = ps <= 139, ps >= 139
    ax.plot(ps[inside], curve[inside], color='#c7353b', lw=1.8, zorder=4, gid='fitted_prime_curve')
    ax.plot(ps[outside], curve[outside], color='#c7353b', lw=2, ls=':', zorder=4, gid='extrapolated_prime_curve')
    ax.axvline(law['fold'], color='#c7353b', ls='--', lw=1, alpha=.75)
    ax.set(yscale='log', xlim=(95, 201), ylim=(1.1e4, 2.1e8), xlabel='Prime $p$', ylabel='Parameters $P$')
    ax.set_xticks([97, 113, 127, 139, 149, 163, 179, 197])
    ax.grid(axis='y', alpha=.25, lw=.6); ax.spines[['top', 'right']].set_visible(False)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=.045, pad=.025)
    cb.set_label(f'{label.capitalize()} delay across ten seeds (epochs)')
    handles = [Line2D([], [], ls='', marker='o', color=cmap(.5), label=f'Positive {label} delay'),
               Line2D([], [], ls='', marker='o', markerfacecolor='none', markeredgecolor='#969fa9', label=f'Zero {label} delay'),
               Line2D([], [], ls='', marker='x', color='#b76814', label='At least one unresolved seed'),
               Line2D([], [], color='#c7353b', label='Fitted lower crossing'),
               Line2D([], [], color='#c7353b', ls=':', lw=2, label='Extrapolation beyond 139'),
               Line2D([], [], color='#c7353b', ls='--', label=f'Terminal prime {law["fold"]:.2f}')]
    fig.legend(handles=handles, fontsize=8.5)
    verify_curve_domains(fig)
    save_paper_figure(fig, OUT / (output_name or f'cone_{metric}_observed_logtime.pdf'),
                      notes=[], max_columns=3)
    plt.close(fig)


def clock_figure(law, included, current_C, *, cell_means=False):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), layout='constrained')
    cmap, norm = plt.get_cmap('viridis'), Normalize(min(CENTRAL), max(CENTRAL))
    m, g = law['mem'], law['gen']
    mem = included[included.kind == 'mem']; gen = included[included.kind == 'gen']
    ff = mem.f_original.to_numpy() * C_OLD / current_C
    axes[0].scatter(ff, mem.event_epoch, c=mem.p, cmap=cmap, norm=norm, s=10, alpha=.4, linewidths=0)
    grid = np.linspace(ff.min(), ff.max(), 300)
    axes[0].plot(grid, np.exp(m['ln_B'] + m['gamma_current_C'] * grid), color='#c7353b', lw=1.8)
    axes[0].set(yscale='log', xlabel='Capacity fraction $f$ (current $C$)', ylabel='Memorisation time (epochs)')
    yy = gen.event_epoch.to_numpy() / gen.p.to_numpy() ** g['a']
    axes[1].scatter(gen.P, yy, c=gen.p, cmap=cmap, norm=norm, s=10, alpha=.3, linewidths=0)
    pp = np.geomspace(gen.P.min(), gen.P.max(), 300)
    axes[1].plot(pp, np.exp(g['lnA']) * pp ** -g['b'], color='#c7353b', lw=1.8)
    axes[1].set(xscale='log', yscale='log', xlabel='Parameters $P$', ylabel=r'Generalisation time / $p^a$ (epochs)')
    axes[1].set_xticks([2e4, 5e4, 1e5, 2e5, 4e5], ['20k', '50k', '100k', '200k', '400k'])
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    for ax in axes:
        ax.grid(alpha=.2); ax.spines[['top', 'right']].set_visible(False)
        assert not ax.get_title()
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=list(axes), fraction=.025, pad=.02)
    cb.set_label('Prime $p$')
    notes = [
        ('Every point is one arithmetic cell mean. Both fits use equal cell weights in log time; censored seeds are excluded.'
         if cell_means else 'Every point is one observed seed. Both fits minimise squared residuals in log time; censored runs are excluded.'),
        f'Memorisation: ln T = {m["ln_B"]:.6f} + {m["gamma_current_C"]:.6f} f; N = {m["n_observations"]} {"cells" if cell_means else "runs"}; log-space R² = {m["r2"]:.4f}.',
        f'Generalisation: ln T = {g["lnA"]:.6f} {g["a"]:+.6f} ln p − {g["b"]:.6f} ln P; N = {g["n_observations"]} {"cells" if cell_means else "runs"}; log-space R² = {g["r2"]:.4f}.',
        'The right panel removes the fitted prime dependence to display the parameter scaling. No minima or cell medians enter either regression.',
    ]
    save_paper_figure(fig, OUT / 'clock_fits.pdf', notes=[fig.text(0, 0, s, fontsize=8.5) for s in notes])
    plt.close(fig)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delay-aggregation', default='min',
                        help='Observed delay reducer: min (default), median, mean, max, or quantile:<q>. Clock fits always use individual observed runs.')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    metric = args.delay_aggregation
    OUT = (args.out or ROOT / f'results/cone_observed_logtime_{metric}_20261106').resolve()
    assert OUT != BASELINE.resolve(), 'Preserve the earlier median comparison in its original directory'
    baseline_fit = normalise_timescale_laws(json.loads((BASELINE / 'fit.json').read_text()))
    baseline_hashes = {str(p.relative_to(ROOT)): sha(p) for p in BASELINE.iterdir() if p.is_file()}
    OUT.mkdir(parents=True, exist_ok=True)
    protected_paths = json.loads((ROOT / 'results/seed_aggregation_sensitivity_20261106/protected_before.json').read_text())
    protected = {p: sha(ROOT / p) for p in protected_paths}
    data = pd.read_csv(INPUT)
    assert set(data.p) == set(CENTRAL) and set(data.gpu_type) == {'A100'}
    assert data.f_original.between(.08, 1).all() and data.dim.max() <= 256
    for _, group in data.groupby(['kind', 'p', 'P']):
        assert len(group) == 10 and set(group.seed) == set(range(42, 52))
    included = data[data.observed].copy(); excluded = data[~data.observed].copy()
    assert included.event_epoch.notna().all() and (included.event_epoch > 0).all()
    assert set(included.uuid).isdisjoint(set(excluded.uuid))
    included.to_csv(OUT / 'included_runs.csv', index=False)
    excluded.to_csv(OUT / 'excluded_censored_runs.csv', index=False)
    counts, clocks = {}, {}
    for kind in ('mem', 'gen'):
        dd = included[included.kind == kind]
        points = [{'p': r.p, 'P': r.P, 'f': r.f_original, 't': r.event_epoch} for r in dd.itertuples()]
        clocks[kind] = fit(kind, points)
        X = np.array([[1, r['f']] if kind == 'mem' else [1, math.log(r['p']), -math.log(r['P'])] for r in points])
        coefficients = np.linalg.lstsq(X, np.log(dd.event_epoch.to_numpy()), rcond=None)[0]
        keys = ['ln_B', 'gamma_original'] if kind == 'mem' else ['lnA', 'a', 'b']
        assert np.allclose(coefficients, [clocks[kind][k] for k in keys], rtol=1e-11, atol=1e-11)
        counts[kind] = {'included_runs': len(dd), 'excluded_censored_runs': int(((data.kind == kind) & ~data.observed).sum()),
                        'contributing_cells': dd.groupby(['p', 'P']).ngroups}
    historical = normalise_timescale_laws(json.loads((ROOT / 'results/theory/forecast_onset.json').read_text()))
    assert np.isclose(clocks['mem']['gamma_original'], historical['mem_law']['gamma'])
    current_C = normalise_timescale_laws(json.loads((ROOT / 'results/capacity_h100_matched_20261103/forecast_coefficients.json').read_text()))['C_convention']
    clocks['mem']['gamma_current_C'] = clocks['mem']['gamma_original'] * current_C / C_OLD
    law = {**clocks, 'fold': fold(clocks['mem'], clocks['gen'], C_OLD)}
    cells = aggregate(json.loads(RUNS.read_text()), metric)
    assert len(cells) == 1736 and all(c['n_seeds'] == 10 for c in cells)
    assert sum(bool(c['unresolved_seeds']) for c in cells) == 102
    if metric == 'min':
        canonical = json.loads((ROOT / 'results/cone_ten_seeds_h100_20261106/figure/cells.json').read_text())
        assert len(canonical) == len(cells)
        for old, new in zip(canonical, cells):
            for key in ('p', 'dim', 'n_seeds', 'seeds', 'state', 'minimum_delay', 'unresolved_seeds'):
                assert old[key] == new[key], (key, old['p'], old['dim'])
            assert new['aggregate_delay'] == new['minimum_delay']
    assert law == baseline_fit['law'], 'Changing delay aggregation must not change either clock fit'
    assert counts == baseline_fit['counts']
    pd.DataFrame(cells).to_csv(OUT / f'{metric}_delay_cells.csv', index=False)
    onsets = get_onsets(cells)
    predictions, diagnostics, curves = [], [], []
    for p in sorted({c['p'] for c in cells} | {251}):
        lo, hi = branches(p, law)
        pp = crossing(p, law['mem'], law['gen'], C_OLD)
        assert (pp is None and lo is None) or np.isclose(pp, lo, rtol=1e-9)
        predictions.append({'p': p, 'lower_crossing_P': lo, 'upper_crossing_P': hi,
                            f'{metric}_onset_dim_le_256': onsets.get(p, {}).get('value'),
                            'onset_status': onsets.get(p, {}).get('status', 'not_measured')})
        cc = [c for c in cells if c['p'] == p]
        inside = [c for c in cc if lo is not None and lo < c['param_count'] < hi]
        diagnostics.append({'p': p, 'measured_cells': len(cc), 'predicted_inside': len(inside),
                            'observed_inside': dict(Counter(c['state'] for c in inside)),
                            'observed_all': dict(Counter(c['state'] for c in cc)),
                            'maximum_identified_aggregate_delay': max((c['aggregate_delay'] for c in cc if c['aggregate_delay'] is not None), default=None)})
    p149 = next(r for r in predictions if r['p'] == 149)
    max_error = 0.
    for p in np.linspace(97, law['fold'], 600):
        lo, hi = branches(p, law)
        for P in (lo, hi):
            k = p * (p - 1) / 2 * math.log2(p + 2)
            gap = law['gen']['lnA'] + law['gen']['a'] * math.log(p) - law['gen']['b'] * math.log(P) - law['mem']['ln_B'] - law['mem']['gamma_original'] * k / (C_OLD * P)
            max_error = max(max_error, abs(gap))
        curves.append({'p': p, 'lower_P': lo, 'upper_P': hi})
    assert max_error < 1e-8
    pd.DataFrame(predictions).to_csv(OUT / 'predictions.csv', index=False)
    pd.DataFrame(curves).to_csv(OUT / 'crossing_curves.csv', index=False)
    result = {'method': 'OLS of individual observed seed log times for BOTH clocks; equal weight per retained seed; no aggregation before regression.',
              'censoring': f'All censored runs excluded from fits as requested; {metric}-delay cone retains all ten seeds and unresolved markers.',
              'delay_aggregation': metric,
              'fit_primes': CENTRAL, 'fit_window_original_f': [.08, 1], 'C_current': current_C,
              'counts': counts, 'law': law, 'delay_counts': dict(Counter(c['state'] for c in cells)),
              'high_prime_diagnostics': diagnostics,
              'central_onset_agreement': agreement(law, onsets),
              'interpretation': 'Exponentiated log regression is a fitted central time; it is a median under symmetric log residuals. It is not a median over uncertain fits.',
              'publication': 'Separate fit and previews; existing paper, coefficient artifacts, and cone pointer unchanged.'}
    save_json(OUT / 'fit.json', result)
    cone_figure(law, cells, metric); clock_figure(law, included, current_C)
    assert all(sha(ROOT / p) == digest for p, digest in protected.items())
    assert all(sha(ROOT / p) == digest for p, digest in baseline_hashes.items())
    assert sha(OUT / 'crossing_curves.csv') == sha(BASELINE / 'crossing_curves.csv')
    save_json(OUT / 'validation.json', {'observed_only_fits_verified': True, 'law_and_crossing_curve_unchanged': True,
        'delay_aggregation': metric, 'minimum_cells_match_canonical': metric == 'min',
        'earlier_median_comparison_unchanged': baseline_hashes, 'numpy_lstsq_coefficients_verified': True,
        'max_abs_log_clock_gap_at_roots': max_error, 'all_1736_ten_seed_cone_cells_retained': True,
        'all_102_incomplete_cells_marked': True, 'extrapolation_dotted_after_139_verified': True,
        'publication_hashes_unchanged': protected, 'source_sha256': sha(__file__),
        'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [INPUT, RUNS]}})
    shutil.copy2(__file__, OUT / Path(__file__).name)
    p197_diagnostic = next(r for r in diagnostics if r['p'] == 197)
    positives_beyond_fold = [{'p': r['p'], 'positive_cells': r['observed_all'].get('positive', 0)}
                            for r in diagnostics if r['p'] > law['fold'] and r['observed_all'].get('positive', 0)]
    (OUT / 'report.txt').write_text(
        'Both laws fit all observed individual seed log times with ordinary least squares.\n'
        'There is no seed aggregation before regression and no censoring likelihood or uncertainty fit.\n'
        f'Included: {counts["mem"]["included_runs"]} memorisation and {counts["gen"]["included_runs"]} generalisation runs.\n'
        f'Excluded censored: {counts["mem"]["excluded_censored_runs"]} memorisation and {counts["gen"]["excluded_censored_runs"]} generalisation runs.\n'
        f'Fit primes: {CENTRAL}; original capacity-fraction window [0.08,1], widths <=256.\n'
        f'Terminal prime: {law["fold"]:.8f}. Raw crossing at p149: {p149["lower_crossing_P"]}.\n'
        f'Fits use A100 data. All 1,736 ten-seed cone cells are retained, with {metric} delay and unresolved marks.\n'
        'Point curves are ordinary fitted curves, not medians of uncertainty draws. No onset calibration is applied.\n'
        f'At p197 the fitted clocks have no crossing; observed {metric}-delay states: {p197_diagnostic["observed_all"]}.\n'
        f'Measured positive-{metric} cells beyond the fitted terminal prime: {positives_beyond_fold}.\n'
        'This finite grid does not establish an exact cutoff at every width or seed. See fit.json for per-prime diagnostics.\n'
        'Fits are conditional on observed crossings; no inference about unobserved completion times is made.\n'
        f'Both clock fits and the raw crossing curve are unchanged from the earlier median comparison.\n'
        f'Paper and existing publication artifacts unchanged. Reproduce: python scripts/refit_cone_observed_logtime.py --delay-aggregation {metric}\n')
    print(json.dumps({'counts': counts, 'law': law, 'delay_aggregation': metric, 'delay_counts': result['delay_counts'], 'p149': p149,
                      'p197': p197_diagnostic, 'positive_cells_beyond_fold': positives_beyond_fold, 'out': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()
