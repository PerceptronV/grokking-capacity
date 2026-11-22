#!/usr/bin/env python3
"""Build the uniform ten-seed cone, preserving censored observations explicitly."""
from pathlib import Path
from collections import Counter
import argparse, json, sqlite3
from grokking_capacity.experiments.cone_workflow.cone_data import aggregate, score, write_csv, law_coefficients, law_onset
from grokking_capacity.experiments.cone_workflow.run_cone_ten_seeds import ROOT, OUT, sha, write_json
from grokking_capacity.analysis._figure_style import save_paper_figure


def render(cells, out, pending=0, checked=0, total_new=1495, *, metric="min", filename=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    import numpy as np
    out = Path(out); out.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(7.1, 5.3), layout='constrained')
    assert all(c.get('aggregation_metric', 'min') == metric for c in cells), 'aggregate cells with the requested metric first'
    color_field = 'mean_delay' if metric == 'min' else 'aggregate_delay'
    name = filename or ('cone_seedmin_fixed' if metric == 'min' else 'cone_' + metric.replace(':', '_') + '_fixed')
    norm = LogNorm(1 if metric == 'min' else .1, 1500); cmap = plt.get_cmap('viridis')
    for state in ['zero', 'positive', 'unresolved', 'pending']:
        rows = [c for c in cells if c['state'] == state]
        xx = [c['p'] for c in rows]; yy = [c['param_count'] for c in rows]
        if state == 'positive':
            ax.scatter(xx, yy, c=[c[color_field] for c in rows], cmap=cmap, norm=norm,
                       s=17, edgecolors='#202d38', linewidths=.3, zorder=3)
        elif state == 'zero':
            ax.scatter(xx, yy, s=12, facecolors='none', edgecolors='#9299a2', linewidths=.5, alpha=.65, zorder=1)
        elif state == 'pending':
            ax.scatter(xx, yy, s=20, marker='^', facecolors='none', edgecolors='#377ba8', linewidths=.65, zorder=2)
        # All unresolved seeds, including those in known-zero cells, are marked below.
    censored = [c for c in cells if c['state'] != 'pending' and c['unresolved_seeds'] > 0]
    ax.scatter([c['p'] for c in censored], [c['param_count'] for c in censored],
               marker='x', s=24, linewidths=.8, color='#b35d12', zorder=5)
    coeffs = law_coefficients(); fold = coeffs['capacity_update']['fold_new']
    pp = np.linspace(97, fold - 1e-7, 500)
    ax.plot(pp, [law_onset(p, coeffs) for p in pp], color='#c7353b', lw=1.7, zorder=4)
    ax.axvline(fold, color='#c7353b', ls='--', lw=1.1, zorder=2)
    ax.set(yscale='log', xlabel='Prime $p$', ylabel='Parameters $P$', xlim=(95, 201))
    ax.set_xticks([97, 113, 127, 139, 149, 163, 179, 199]); ax.tick_params(labelsize=9)
    ax.grid(axis='y', which='major', color='#d4d8de', alpha=.55, lw=.6); ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    handles = [Line2D([], [], marker='o', ls='', markerfacecolor=cmap(.55), markeredgecolor='#202d38', markersize=5, label='Positive delay in all ten seeds' if metric == 'min' else f'Positive {metric} delay'),
               Line2D([], [], marker='o', ls='', markerfacecolor='none', markeredgecolor='#9299a2', markersize=5, label='At least one observed zero delay' if metric == 'min' else f'Zero {metric} delay'),
               Line2D([], [], marker='x', ls='', color='#b35d12', markersize=5, label='At least one unresolved seed'),
               Line2D([], [], color='#c7353b', lw=1.7, label='Law-derived onset crossing'),
               Line2D([], [], color='#c7353b', lw=1.1, ls='--', label=rf'$p^*={fold:.1f}$')]
    if pending:
        handles.append(Line2D([], [], marker='^', ls='', markerfacecolor='none', markeredgecolor='#377ba8', markersize=5, label='Awaiting ten verified seeds'))
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0, 1.02), ncol=2, fontsize=8,
              frameon=False, borderaxespad=0, columnspacing=1.2, handlelength=2.3)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=.035, pad=.025)
    cb.set_label('Mean delay in complete positive cells (epochs)' if metric == 'min' else f'{metric.capitalize()} delay (epochs)', fontsize=9); cb.ax.tick_params(labelsize=9)
    save_paper_figure(fig, out / (name + '.pdf'), max_columns=2)
    plt.close(fig)
    write_json(out / (name + '_plot.json'), {'aggregation_metric': metric, 'cells': len(cells), 'pending_cells': pending,
               'unresolved_seeds_explicitly_marked': True, 'cells_with_unresolved_seeds': len(censored),
               'fold': fold, 'minimum_completed_seed_count': min(c['n_seeds'] for c in cells if c['state'] != 'pending')})


def load_new(m, partial=False):
    expected = {(j['p'], j['dim'], j['seed']): j for j in m['items']}
    checks = {r['uuid']: r for p in (OUT / 'checkpoint_checks').glob('*.json') for r in [json.loads(p.read_text())]}
    con = sqlite3.connect((OUT / 'runs.db').as_uri() + '?mode=ro', uri=True, timeout=120)
    con.row_factory = sqlite3.Row
    records = [dict(r) for r in con.execute("select * from runs where status='completed'")]; con.close()
    fresh = []
    for r in records:
        if r['uuid'] not in checks:
            assert partial, ('Unverified row', r['uuid'])
            continue
        spec = expected[r['p'], r['dim'], r['seed']]
        assert r['host'] == m['host'] and r['gpu_type'] == 'H100'
        assert r['max_epochs'] == spec['max_epochs'] and r['n_samples'] == r['p'] * (r['p'] - 1) // 2
        assert 0 < r['epochs_trained'] <= r['max_epochs']
        if r['epochs_trained'] < r['max_epochs']:
            assert r['final_train_acc'] >= 99 and r['final_val_acc'] >= 99, ('Unexpected early stopping', r['uuid'])
        for k, v in m['protocol'].items():
            assert r[k] == v
        s = score({**r, 'database': str(OUT / 'runs.db')})
        for k in ['trace_sha256', 'train99', 'val98', 'state', 'delay']:
            assert s[k] == checks[r['uuid']][k]
        assert s['state'] == 'unresolved' or s['val98'] is not None
        s['validation98_observed'] = s['val98'] is not None
        fresh.append(s)
    if not partial:
        assert len(fresh) == len(expected) == m['jobs']
        assert {(r['p'], r['dim'], r['seed']) for r in fresh} == set(expected)
    return fresh


def analyse(partial=False):
    m = json.loads((OUT / 'manifest.json').read_text()); base = OUT / 'base_cohort'
    for name, digest in m['base_cohort_hashes'].items():
        assert sha(base / name) == digest
    for rel, digest in m['source_hashes'].items():
        assert sha(OUT / 'source' / rel) == digest
    if not partial:
        done = json.loads((OUT / 'completion.json').read_text())
        checks = json.loads((OUT / 'checkpoint_verification.json').read_text())
        assert done['completed'] == checks['runs_verified'] == m['jobs'] and not done['failed']
    old = json.loads((base / 'runs.json').read_text()); old_cells = json.loads((base / 'cells.json').read_text())
    targets = {(c['p'], c['dim']): c for c in m['targets']}
    replace = {k for k, c in targets.items() if c['gpu_type'] == 'A100'}
    fresh = load_new(m, partial=partial)
    retained = [r for r in old if (r['p'], r['dim']) not in replace]
    rows = retained + fresh
    cells = aggregate(rows)
    idx = {(c['p'], c['dim']): c for c in cells}
    pending = []
    for c in old_cells:
        key = c['p'], c['dim']
        if key not in idx or idx[key]['n_seeds'] != 10:
            assert partial
            pending.append(key)
            idx[key] = {**c, 'state': 'pending'}
    cells = [idx[k] for k in sorted(idx)]
    assert len(cells) == m['final_cells']
    if partial:
        render(cells, OUT / 'live', pending=len(pending), checked=len(fresh))
        write_json(OUT / 'live/status.json', {'new_runs_verified': len(fresh), 'new_runs_planned': m['jobs'],
                   'pending_cells': len(pending), 'completed_ten_seed_cells': len(cells) - len(pending)})
        return {'verified': len(fresh), 'pending_cells': len(pending)}
    assert len(rows) == m['final_unique_runs'] and len({r['uuid'] for r in rows}) == len(rows)
    assert all(c['n_seeds'] == 10 and c['seeds'] == list(range(42, 52)) for c in cells)
    oldidx = {(c['p'], c['dim']): c for c in old_cells}
    assert all(c == oldidx[c['p'], c['dim']] for c in cells if (c['p'], c['dim']) not in targets)
    # No unobserved validation event is silently converted to zero or positive.
    assert all((r['state'] == 'unresolved') == (r['val98'] is None) for r in rows)
    unresolved = [r for r in rows if r['state'] == 'unresolved']
    old_runs = {(r['p'], r['dim'], r['seed']): r for r in old}
    paired = []
    for r in fresh:
        key = r['p'], r['dim'], r['seed']
        if key in old_runs:
            prev = old_runs[key]
            paired.append({'p': r['p'], 'dim': r['dim'], 'seed': r['seed'], 'old_uuid': prev['uuid'],
                           'new_uuid': r['uuid'], 'old_state': prev['state'], 'new_state': r['state'],
                           'old_delay': prev['delay'], 'new_delay': r['delay'], 'old_train99': prev['train99'],
                           'new_train99': r['train99'], 'old_val98': prev['val98'], 'new_val98': r['val98']})
    assert len(paired) == 150
    metadata = [json.loads((OUT / 'jobs' / f'p{r["p"]}_d{r["dim"]}_s{r["seed"]}.json').read_text()) for r in fresh]
    runtime_keys = ['torch', 'cuda', 'dtype', 'tf32_matmul', 'tf32_cudnn', 'gpu']
    assert len({tuple(v[k] for k in runtime_keys) for v in metadata}) == 1
    assert all(v['dtype'] == 'torch.float32' and v['tf32_matmul'] is False for v in metadata)
    assert {v['physical_gpu'] for v in metadata} == {0, 1}
    scheduling_counts = Counter()
    for v in metadata:
        launch_path = OUT / 'launch_provenance' / f'p{v["p"]}_d{v["dim"]}_s{v["seed"]}.json'
        if launch_path.exists():
            launch = json.loads(launch_path.read_text())
            assert launch['gpu'] == v['physical_gpu']
            assert launch['gpu_uuid'].removeprefix('GPU-') == v['gpu_uuid'].removeprefix('GPU-')
            assert launch['manifest_sha256'] == sha(OUT / 'manifest.json')
            scheduling_counts['MPS'] += 1
        else:
            scheduling_counts['native_CUDA'] += 1
    if scheduling_counts['MPS']:
        for name in ['scheduling_benchmark', 'scheduling_benchmark_wide']:
            assert json.loads((OUT / name / 'result.json').read_text())['all_traces_and_checkpoints_identical']
    runtime_contract = {k: metadata[0][k] for k in runtime_keys}
    changes = []
    for c in cells:
        prev = oldidx[c['p'], c['dim']]
        if c['state'] != prev['state'] or c['minimum_delay'] != prev['minimum_delay']:
            changes.append({'p': c['p'], 'dim': c['dim'], 'old_state': prev['state'], 'new_state': c['state'],
                            'old_minimum_delay': prev['minimum_delay'], 'new_minimum_delay': c['minimum_delay'],
                            'old_seeds': prev['n_seeds'], 'new_seeds': c['n_seeds'],
                            'positive_seeds': c['positive_seeds'], 'zero_seeds': c['zero_seeds'],
                            'unresolved_seeds': c['unresolved_seeds']})
    prime_summary = []
    for p in sorted({c['p'] for c in cells}):
        cc = [c for c in cells if c['p'] == p]; rr = [r for r in rows if r['p'] == p]
        prime_summary.append({'p': p, 'cells': len(cc), 'runs': len(rr),
                              'positive_cells': sum(c['state'] == 'positive' for c in cc),
                              'zero_cells': sum(c['state'] == 'zero' for c in cc),
                              'unresolved_cells': sum(c['state'] == 'unresolved' for c in cc),
                              'positive_runs': sum(r['state'] == 'positive' for r in rr),
                              'zero_runs': sum(r['state'] == 'zero' for r in rr),
                              'unresolved_runs': sum(r['state'] == 'unresolved' for r in rr)})
    # Compare the common d<=256 window used by the central onset figures.
    # This reports the change without overwriting any historical test cohort.
    import math
    def positive_tail_onset(cc):
        cc = sorted(cc, key=lambda c: c['param_count'])
        for i, c in enumerate(cc):
            if all(x['state'] == 'positive' for x in cc[i:]):
                return c['param_count']
        return None
    onset_comparison = []
    for prime in sorted({c['p'] for c in cells}):
        before = [c for c in old_cells if c['p'] == prime and c['dim'] <= 256]
        after = [c for c in cells if c['p'] == prime and c['dim'] <= 256]
        old_onset = positive_tail_onset(before); new_onset = positive_tail_onset(after)
        onset_comparison.append({'p': prime, 'comparison_max_width': 256,
            'old_positive_tail_onset_P': old_onset, 'new_positive_tail_onset_P': new_onset,
            'old_log10_P': None if old_onset is None else math.log10(old_onset),
            'new_log10_P': None if new_onset is None else math.log10(new_onset),
            'changed': old_onset != new_onset,
            'old_cells_with_unresolved_seeds': sum(c['unresolved_seeds'] > 0 for c in before),
            'new_cells_with_unresolved_seeds': sum(c['unresolved_seeds'] > 0 for c in after)})
    write_csv(OUT / 'onset_comparison_max_width256.csv', onset_comparison)
    # The first positive cell is a different summary from a sustained positive
    # tail. Report both; unchanged tails do not prove a historical onset cell
    # remains positive after completing the seed set.
    first_positive_comparison = []
    for prime in sorted({c['p'] for c in cells}):
        before = [c for c in old_cells if c['p'] == prime and c['state'] == 'positive']
        after = [c for c in cells if c['p'] == prime and c['state'] == 'positive']
        bp = min((c['param_count'] for c in before), default=None)
        ap = min((c['param_count'] for c in after), default=None)
        first_positive_comparison.append({'p': prime, 'old_first_positive_P': bp,
            'new_first_positive_P': ap,
            'old_log10_P': None if bp is None else math.log10(bp),
            'new_log10_P': None if ap is None else math.log10(ap), 'changed': bp != ap})
    write_csv(OUT / 'first_positive_comparison.csv', first_positive_comparison)
    historical = json.loads((ROOT / 'results/theory/forecast_onset.json').read_text())['held_out_p149']
    historical_cell = next(c for c in cells if c['p'] == 149 and c['param_count'] == historical['P_onset_empirical'])
    main_text_discrepancies = []
    if historical_cell['state'] != 'positive':
        first = next(x for x in first_positive_comparison if x['p'] == 149)
        tail = next(x for x in onset_comparison if x['p'] == 149)
        main_text_discrepancies.append({
            'claim': 'Historical held-out p149 onset at log10(P)=5.51',
            'historical_P': historical['P_onset_empirical'],
            'historical_log10_P': math.log10(historical['P_onset_empirical']),
            'width': historical_cell['dim'], 'current_state': historical_cell['state'],
            'positive_seeds': historical_cell['positive_seeds'], 'zero_seeds': historical_cell['zero_seeds'],
            'current_first_positive_log10_P': first['new_log10_P'],
            'current_positive_tail_log10_P_max_width256': tail['new_log10_P'],
            'historical_tests_refitted': False,
            'interpretation': 'The historical onset cell is not all-ten-positive in the current cone. '
                'The expanded-grid first-positive and sustained-tail summaries are distinct; neither silently replaces the historical test.'})
    five = aggregate([r for r in rows if r['seed'] in range(42, 47)])
    five_idx = {(c['p'], c['dim']): c for c in five}
    sensitivity = [{'p': c['p'], 'dim': c['dim'], 'ten_seed_state': c['state'],
                    'five_seed_state': five_idx[c['p'], c['dim']]['state']} for c in cells
                   if c['state'] != five_idx[c['p'], c['dim']]['state']]
    s = json.loads((base / 'summary.json').read_text())
    s.update({'selected_rows': s['selected_rows'] + len(fresh), 'unique_runs': len(rows), 'cells': len(cells), 'primes': [r['p'] for r in prime_summary],
              'seed_count_distribution': {'10': len(cells)},
              'event_counts': dict(Counter(r['event_status'] for r in rows)),
              'run_state_counts': dict(Counter(r['state'] for r in rows)),
              'cell_state_counts': dict(Counter(c['state'] for c in cells)),
              'budget_distribution': dict(Counter(r['max_epochs'] for r in rows)),
              'retained_run_counts_by_host': dict(Counter(r['host'] for r in rows)),
              'cells_with_unresolved_seeds': sum(c['unresolved_seeds'] > 0 for c in cells),
              'zero_cells_with_unresolved_seeds': sum(c['state'] == 'zero' and c['unresolved_seeds'] > 0 for c in cells),
              'ten_seed_completion': {'new_runs': len(fresh), 'target_cells': len(targets),
                  'A100_cells_replaced': len(replace), 'archived_observations_replaced': 150,
                  'manifest_sha256': sha(OUT / 'manifest.json')},
              'five_seed_sensitivity': {'common_seeds': list(range(42, 47)), 'cells': len(five),
                  'incomplete_five_seed_cells': 0, 'classification_changes': sensitivity},
              'final_five_seed_classification_changes': sensitivity,
              'five_seed_sensitivity_scope': 'Numerical diagnostics only; the only displayed figure uses ten seeds everywhere.'})
    s['source_reports'].append({'database': str(OUT / 'runs.db'), 'host': m['host'], 'gpu_type': 'H100',
                               'selected_rows': len(fresh), 'replaces_archive_rows': 150, 'protocol': m['protocol']})
    fig = OUT / 'figure'; fig.mkdir(exist_ok=True)
    for name, data in [('runs', rows), ('cells', cells), ('summary', s)]:
        write_json(fig / (name + '.json'), data)
    write_csv(fig / 'cells.csv', cells); write_csv(fig / 'prime_summary.csv', prime_summary)
    write_csv(fig / 'unresolved_runs.csv', [{k: r[k] for k in ['uuid', 'p', 'dim', 'seed', 'max_epochs', 'train99', 'val98', 'followup_epochs', 'event_status', 'trace_path']} for r in unresolved])
    write_json(OUT / 'scored_runs.json', fresh); write_json(OUT / 'changed_cells.json', changes)
    write_csv(OUT / 'changed_cells.csv', changes); write_csv(OUT / 'paired_A100_H100_runs.csv', paired)
    write_json(fig / 'common_five_summary.json', {'classification_changes': sensitivity})
    render(cells, fig)
    for name in ['cone_seedmin_fixed.png', 'cone_seedmin_fixed.pdf', 'cone_seedmin_fixed_plot.json']:
        import shutil
        (OUT / 'live').mkdir(exist_ok=True)
        shutil.copy2(fig / name, OUT / 'live' / name)
    p197 = [r for r in rows if r['p'] == 197 and r['dim'] == 316]
    result = {'new_runs': len(fresh), 'total_runs': len(rows), 'total_cells': len(cells),
              'uniform_seed_ids': list(range(42, 52)), 'complete_original_cells_unchanged': 1467,
              'single_hardware_cohort_per_cell': True, 'physical_gpus': [0, 1],
              'new_run_scheduling_counts': dict(scheduling_counts), 'runtime_contract': runtime_contract,
              'new_run_gpu_uuid_counts': dict(Counter(v['gpu_uuid'] for v in metadata)),
              'run_states': s['run_state_counts'], 'cell_states': s['cell_state_counts'],
              'unresolved_outcomes_flagged': len(unresolved), 'prime_summary': prime_summary,
              'changed_cells': changes, 'onset_comparison_max_width256': onset_comparison,
              'first_positive_comparison': first_positive_comparison, 'main_text_discrepancies': main_text_discrepancies,
              'paired_replacement_seed_state_changes': sum(r['old_state'] != r['new_state'] for r in paired),
              'p197_d316': idx[197, 316], 'p197_d316_delays': [r['delay'] for r in sorted(p197, key=lambda r: r['seed'])],
              'no_fitted_laws_or_main_tests_changed': True}
    write_json(OUT / 'analysis.json', result)
    write_json(OUT / 'live/status.json', {'status': 'complete_and_verified', 'new_runs_verified': len(fresh),
               'new_runs_planned': m['jobs'], 'pending_cells': 0, 'completed_ten_seed_cells': len(cells)})
    (OUT / 'claim_flags.txt').write_text(
        'TEN-SEED CONE: CLAIMS TO CHECK\n\n'
        'All displayed cells contain exactly ten distinct seeds (42--51). Missing validation98 is unresolved, never a zero or a positive. Crosses also overlay open circles when their minimum is known zero but other seeds are unresolved.\n\n'
        f'p197,d316 delays by seed: {result["p197_d316_delays"]}; cell state {idx[197,316]["state"]}. Historical statements about an all-five-positive point must not describe the new ten-seed figure.\n\n'
        f'{len(changes)} cells changed state or minimum; see changed_cells.csv. The paired A100/H100 comparison retains all150 replaced observations. The original main-test and measured-crossing cohorts are separate and were not overwritten.\n\n'
        'The common d<=256 onset comparison is in onset_comparison_max_width256.csv. Any changed value is a review flag for main-text comparisons with the cone; historical main tests retain their original cohorts and are not silently recomputed.\n\n'
        'First-positive-cell changes are listed separately in first_positive_comparison.csv. Unchanged positive-tail boundaries do not establish that every historical onset cell is reproduced.\n\n'
        + ''.join('MAIN-TEXT DISCREPANCY: ' + json.dumps(flag) + '\n\n' for flag in main_text_discrepancies)
        + f'{len(unresolved)} unresolved seed outcomes remain. No claim of never generalising is justified from a finite budget. Individual positive-delay observations also rule out a universal absence of delayed generalisation, even where every cell minimum is zero.\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    result = analyse(partial=args.preview)
    print(json.dumps({k: v for k, v in result.items() if k not in ['changed_cells', 'prime_summary']}, indent=2))
