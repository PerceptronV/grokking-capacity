#!/usr/bin/env python3
"""Fit both clocks to log arithmetic cell means, excluding censored seed times.

One equally weighted regression point per contributing (clock, prime, width)
cell. Keep ten-seed minimum delay for the cone. Export separate PDF/PNG previews
and numerical comparisons; historical and publication artifacts are preserved.
"""
from grokking_capacity.analysis.timescale_laws import normalise_timescale_laws
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from grokking_capacity.analysis.law_sensitivity.plot_cone_matched_aggregations import ROOT, INPUT, RUNS, CENTRAL, C_OLD, fit_clock_pair, branches, get_onsets, agreement, sha, save_json
from grokking_capacity.experiments.cone_workflow.cone_data import aggregate
from grokking_capacity.analysis.law_sensitivity.check_law_complete_ten_seeds import crossing
import grokking_capacity.analysis.law_sensitivity.refit_cone_observed_logtime as plots

DEFAULT_OUT = ROOT / 'results/cone_cellmeans_min_20261106'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    prior_dirs = [ROOT / name for name in (
        'results/cone_observed_logtime_20261106',
        'results/cone_observed_logtime_min_20261106',
        'results/cone_matched_aggregations_20261106',
    )]
    assert not any(out == d or out.is_relative_to(d) for d in prior_dirs)
    protected_paths = json.loads((ROOT / 'results/seed_aggregation_sensitivity_20261106/protected_before.json').read_text())
    protected = {p: sha(ROOT / p) for p in protected_paths}
    prior_hashes = {str(p.relative_to(ROOT)): sha(p) for d in prior_dirs for p in d.iterdir() if p.is_file()}
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT)
    assert set(data.p) == set(CENTRAL) and set(data.gpu_type) == {'A100'}
    assert data.f_original.between(.08, 1).all() and data.dim.max() <= 256
    for _, group in data.groupby(['kind', 'p', 'P']):
        assert len(group) == 10 and set(group.seed) == set(range(42, 52))
        assert group.f_original.nunique() == group.dim.nunique() == 1
    included, excluded = data[data.observed].copy(), data[~data.observed].copy()
    assert included.event_epoch.notna().all() and (included.event_epoch > 0).all()
    assert set(included.uuid).isdisjoint(excluded.uuid)
    included.to_csv(out / 'included_runs.csv', index=False)
    excluded.to_csv(out / 'excluded_censored_runs.csv', index=False)

    fitted, points = fit_clock_pair(data, 'mean', 'observed')
    law = {key: fitted[key] for key in ('mem', 'gen', 'fold')}
    current_C = normalise_timescale_laws(json.loads((ROOT / 'results/capacity_h100_matched_20261103/forecast_coefficients.json').read_text()))['C_convention']
    law['mem']['gamma_current_C'] = law['mem']['gamma_original'] * current_C / C_OLD
    counts, cell_rows = {}, []
    for kind in ('mem', 'gen'):
        dd = included[included.kind == kind]
        # Independent pandas grouping verifies arithmetic mean before log,
        # equal cell weights, and the number of retained seed times per mean.
        grouped = dd.groupby(['p', 'P'], sort=True)
        means = grouped.event_epoch.mean()
        rows = points[kind]
        assert list(means.index) == [(r['p'], r['P']) for r in rows]
        np.testing.assert_allclose(means, [r['t'] for r in rows], rtol=1e-13)
        assert list(grouped.size()) == [r['n_observed'] for r in rows]
        X = np.array([[1, r['f']] if kind == 'mem' else [1, math.log(r['p']), -math.log(r['P'])] for r in rows])
        coef = np.linalg.lstsq(X, np.log(means.to_numpy()), rcond=None)[0]
        keys = ['ln_B', 'gamma_original'] if kind == 'mem' else ['lnA', 'a', 'b']
        np.testing.assert_allclose(coef, [law[kind][k] for k in keys], rtol=1e-11, atol=1e-11)
        counts[kind] = {
            'included_runs': len(dd),
            'excluded_censored_runs': int(((data.kind == kind) & ~data.observed).sum()),
            'contributing_cells': len(rows),
            'complete_cells': sum(r['n_observed'] == 10 for r in rows),
            'partial_cells': sum(r['n_observed'] < 10 for r in rows),
            'all_censored_cells_omitted': data[data.kind == kind].groupby(['p', 'P']).ngroups - len(rows),
        }
        for r in rows:
            group = dd[(dd.p == r['p']) & (dd.P == r['P'])]
            cell_rows.append({
                'kind': kind, 'p': r['p'], 'dim': int(group.dim.iloc[0]), 'P': r['P'],
                'f_original': r['f'], 'event_epoch': r['t'],
                'n_observed': r['n_observed'], 'n_censored': 10 - r['n_observed'],
                'observed_seeds': json.dumps(sorted(group.seed.astype(int).tolist())),
                'regression_weight': 1,
            })
    clock_cells = pd.DataFrame(cell_rows)
    clock_cells.to_csv(out / 'clock_cells.csv', index=False)
    historical = normalise_timescale_laws(json.loads((ROOT / 'results/theory/forecast_onset.json').read_text()))
    for k in ('lnA', 'a', 'b'):
        assert np.isclose(law['gen'][k], historical['gen_law'][k], rtol=1e-12, atol=1e-12)
    individual = normalise_timescale_laws(json.loads((prior_dirs[1] / 'fit.json').read_text()))['law']
    original = {
        'mem': {'ln_B': historical['mem_law']['ln_B'], 'gamma_original': historical['mem_law']['gamma']},
        'gen': {k: historical['gen_law'][k] for k in ('lnA', 'a', 'b')},
    }
    from grokking_capacity.analysis.law_sensitivity.check_law_complete_ten_seeds import fold
    original['fold'] = fold(original['mem'], original['gen'], C_OLD)
    variants = {'original_asymmetric': original, 'individual_seed_logtimes': individual, 'both_cell_means': law}

    cells = aggregate(json.loads(RUNS.read_text()), 'min')
    canonical = json.loads((ROOT / 'results/cone_ten_seeds_h100_20261106/figure/cells.json').read_text())
    assert len(cells) == len(canonical) == 1736
    assert all(c['n_seeds'] == 10 for c in cells)
    assert sum(bool(c['unresolved_seeds']) for c in cells) == 102
    for old, new in zip(canonical, cells):
        for key in ('p', 'dim', 'n_seeds', 'seeds', 'state', 'minimum_delay', 'unresolved_seeds'):
            assert old[key] == new[key], (key, old['p'], old['dim'])
        assert new['aggregate_delay'] == new['minimum_delay']
    pd.DataFrame(cells).to_csv(out / 'min_delay_cells.csv', index=False)
    onsets = get_onsets(cells)
    predictions, curves, diagnostics = [], [], []
    max_error = 0.
    for p in sorted({c['p'] for c in cells} | {251}):
        lo, hi = branches(p, law)
        check = crossing(p, law['mem'], law['gen'], C_OLD)
        assert (lo is None and check is None) or np.isclose(lo, check, rtol=1e-9)
        predictions.append({'p': p, 'lower_crossing_P': lo, 'upper_crossing_P': hi,
                            'min_onset_dim_le_256': onsets.get(p, {}).get('value'),
                            'onset_status': onsets.get(p, {}).get('status', 'not_measured')})
        cc = [c for c in cells if c['p'] == p]
        diagnostics.append({'p': p, 'predicted_crossing_exists': lo is not None,
                            'observed_states': dict(Counter(c['state'] for c in cc)),
                            'unresolved_cells': sum(bool(c['unresolved_seeds']) for c in cc)})
    for p in np.linspace(97, law['fold'], 600):
        lo, hi = branches(p, law)
        for P in (lo, hi):
            k = p * (p - 1) / 2 * math.log2(p + 2)
            gap = law['gen']['lnA'] + law['gen']['a'] * math.log(p) - law['gen']['b'] * math.log(P) - law['mem']['ln_B'] - law['mem']['gamma_original'] * k / (C_OLD * P)
            max_error = max(max_error, abs(gap))
        curves.append({'p': p, 'lower_P': lo, 'upper_P': hi})
    assert max_error < 1e-8
    pd.DataFrame(predictions).to_csv(out / 'predictions.csv', index=False)
    pd.DataFrame(curves).to_csv(out / 'crossing_curves.csv', index=False)

    # Keep missing roots explicit. Never treat them as zero error or drop them
    # silently when comparing finite-prediction dex errors.
    direct = pd.read_csv(ROOT / 'results/theory/crossing_fixed.csv', na_values=['none', 'none (mem slower everywhere)'])
    errors, comparisons = [], {}
    for name, variant in variants.items():
        vv = []
        for r in direct.itertuples():
            lo, hi = branches(r.p, variant)
            measured = None if pd.isna(r.crossing_log10P) else float(r.crossing_log10P)
            error = None if lo is None or measured is None else math.log10(lo) - measured
            row = {'variant': name, 'p': r.p, 'predicted_log10P': None if lo is None else math.log10(lo),
                   'measured_log10P': measured, 'error_dex': error,
                   'status': 'no_measured_root' if measured is None else 'no_predicted_root' if lo is None else 'both_roots'}
            vv.append(row); errors.append(row)
        comparisons[name] = {
            'terminal_prime': variant['fold'],
            'finite_prediction_primes': [r['p'] for r in vv if r['status'] == 'both_roots'],
            'max_abs_error_dex_finite_predictions': max(abs(r['error_dex']) for r in vv if r['error_dex'] is not None),
            'measured_root_but_no_predicted_root': [r['p'] for r in vv if r['status'] == 'no_predicted_root'],
        }
    pd.DataFrame(errors).to_csv(out / 'direct_crossing_comparison.csv', index=False)
    positive_beyond = [r for r in diagnostics if r['p'] > law['fold'] and r['observed_states'].get('positive', 0)]
    result = {
        'method': 'Arithmetic mean of observed seed times within each cell, THEN OLS of log cell means; equal weight per cell for BOTH clocks.',
        'censoring': 'Exclude censored runs before averaging. Omit all-censored cells; retain partial cells with at least one observed event.',
        'clock_aggregation': 'mean', 'delay_aggregation': 'min',
        'fit_primes': CENTRAL, 'fit_window_original_f': [.08, 1], 'C_current': current_C,
        'counts': counts, 'law': law, 'delay_counts': dict(Counter(c['state'] for c in cells)),
        'central_onset_agreement': agreement(law, onsets), 'direct_crossing_comparisons': comparisons,
        'direct_crossing_source_caveat': 'Legacy measured-crossing table uses its original estimator, selection and rounded roots; these are a numerical comparison, not newly recomputed measured means.',
        'high_prime_diagnostics': diagnostics, 'positive_min_cells_beyond_fold': positive_beyond,
        'interpretation': 'Regression targets observed-only arithmetic cell means. Censoring exclusion conditions the estimate on observed completion; no full seed-population mean or exact cutoff is established.',
        'publication': 'Updated analysis and previews only; historical forecasts, paper and publication artifacts unchanged.',
    }
    save_json(out / 'fit.json', result)
    plots.OUT = out
    plots.cone_figure(law, cells, 'min', output_name='cone_min_cellmeans.pdf')
    plots.clock_figure(law, clock_cells, current_C, cell_means=True)
    for p, digest in {**protected, **prior_hashes}.items():
        assert sha(ROOT / p) == digest, p
    save_json(out / 'validation.json', {
        'arithmetic_means_before_log_verified': True, 'equal_weight_per_cell_verified': True,
        'independent_numpy_fit_verified': True, 'all_fit_runs_A100': True,
        'original_generalisation_coefficients_reproduced': True,
        'minimum_cells_match_canonical': True, 'all_1736_ten_seed_cells_retained': True,
        'all_102_unresolved_cells_marked': True, 'max_abs_log_clock_gap_at_roots': max_error,
        'publication_hashes_unchanged': protected, 'previous_analysis_hashes_unchanged': prior_hashes,
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__).resolve(), Path(plots.__file__).resolve()]},
        'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [INPUT, RUNS]},
    })
    for path in [Path(__file__), Path(plots.__file__)]:
        shutil.copy2(path, out / path.name)
    (out / 'report.txt').write_text(
        result['method'] + '\n' + result['censoring'] + '\n'
        f'Counts: {counts}\nFit primes: {CENTRAL}; original f window [0.08,1]; width <=256.\n'
        f'Current capacity constant: {current_C}; clock fits use A100 runs only.\n'
        f'Terminal prime: {law["fold"]:.8f}.\n'
        f'Coefficients: {law}\n'
        f'Comparison against legacy measured crossings: {comparisons}\n'
        + result['direct_crossing_source_caveat'] + '\n'
        f'Positive-min cells beyond predicted terminal prime: {positive_beyond}\n'
        'All 1736 ten-seed cone cells and 102 unresolved markers retained.\n'
        'No exact empirical cutoff claimed. Missing predicted roots are reported explicitly.\n'
        'Measured-clock main-test rank statistics are unaffected by changing the global clock estimator.\n'
        + result['interpretation'] + '\n' + result['publication'] + '\n'
        'Reproduce: python src/grokking_capacity/analysis/law_sensitivity/refit_cone_cellmeans.py\n')
    print(json.dumps({'out': str(out), 'counts': counts, 'law': law, 'comparisons': comparisons,
                      'positive_min_cells_beyond_fold': positive_beyond}, indent=2))


if __name__ == '__main__':
    main()
