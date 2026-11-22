#!/usr/bin/env python3
"""Retrospective dense-crossing reanalysis using observed arithmetic cell means.

Historical registered tables are read-only. All current events are extracted
from raw traces. Empty cells, missing roots, unresolved delays and bootstrap
root-existence fractions remain explicit in the exported results.
"""
from grokking_capacity.analysis.timescale_laws import normalise_timescale_laws
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
from .paths import ROOT, RESULTS
import sqlite3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter
import numpy as np
import pandas as pd

from grokking_capacity.analysis.aggregate import intersection_summary, summarize_delays, onset_summary
from grokking_capacity.analysis._figure_style import save_paper_figure
from grokking_capacity.utils.artifacts import save_json, sha
from .laws import branches

OUT = RESULTS / 'crossings'


def first(values, threshold):
    hit = np.flatnonzero(np.asarray(values) >= threshold)
    return int(hit[0] + 1) if len(hit) else None


def observed_mean(values):
    known = [v for v in values if v is not None]
    return float(np.mean(known)) if known else None


def curve_cross(cells, mem_key='mem_mean', gen_key='gen_mean'):
    return intersection_summary(
        {r['P']: r[mem_key] for r in cells if r[mem_key] is not None},
        {r['P']: r[gen_key] for r in cells if r[gen_key] is not None},
        direction='decreasing')


def dense_onset(cells):
    """First identified onset on the full sampled grid, with no tail trimming."""
    result = onset_summary({r['P']: r['min_delay'] for r in cells})
    return {**result, 'domain_cells': len(cells), 'total_cells': len(cells)}


def extract(which):
    db = Path.home() / f'crossing_{which}_runs.db'
    con = sqlite3.connect(db.as_uri()+'?mode=ro', uri=True); con.row_factory = sqlite3.Row
    statuses = [dict(r) for r in con.execute('select experiment_type,status,count(*) n from runs group by experiment_type,status')]
    rows = [dict(r) for r in con.execute('select * from runs where status="completed" and experiment_type in ("speed","groks")')]
    con.close()
    def read(r):
        path = Path(r['npz_path'])
        with np.load(path, allow_pickle=False) as z:
            if r['experiment_type'] == 'speed':
                mem = first(z['train_acc_trace'], 99)
                val99 = train99 = val98 = delay = None
                followup = len(z['train_acc_trace'])
                assert (mem is not None) == bool(r['saturated'])
                if mem is not None: assert np.isclose(mem, r['saturation_epoch'])
            else:
                mem = None
                train99, val98, val99 = first(z['train_acc'],99), first(z['val_acc'],98), first(z['val_acc'],99)
                delay = None if val98 is None else max(0, val98-train99) if train99 is not None else 0
                followup = len(z['val_acc'])
                assert val99 == r['grokking_epoch']
        return {**{k:r[k] for k in ('uuid','experiment_type','p','dim','param_count','seed','host','gpu_type','mup_base')},
                'P':r['param_count'], 'mem99':mem, 'val99':val99, 'train99':train99, 'val98':val98,
                'delay':delay, 'followup_epochs':followup, 'trace_path':str(path), 'trace_sha256':sha(path)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        traces = list(pool.map(read, rows))
    assert {r['gpu_type'] for r in traces} == {'H100'}
    assert {r['mup_base'] for r in traces} == ({0} if which=='fixed' else {236})
    return traces, statuses


def run(which, out, bootstraps):
    traces, statuses = extract(which)
    save_json(out/f'runs_{which}.json', traces)
    by = defaultdict(lambda: defaultdict(list))
    for r in traces: by[r['p'],r['dim']][r['experiment_type']].append(r)
    cells = []
    for (p,d), arms in sorted(by.items()):
        mm, gg = arms['speed'], arms['groks']
        assert len(mm) == len(gg) == 5
        assert {r['seed'] for r in mm} == {r['seed'] for r in gg} == set(range(42,47))
        assert len({r['P'] for r in mm+gg}) == 1
        mm, gg = sorted(mm,key=lambda r:r['seed']), sorted(gg,key=lambda r:r['seed'])
        mt, gt = [r['mem99'] for r in mm], [r['val99'] for r in gg]
        ds = summarize_delays([r['delay'] for r in gg], 'min')
        cells.append({'p':p,'dim':d,'P':mm[0]['P'], 'mem_times':mt, 'gen_times':gt,
                      'mem_mean':observed_mean(mt), 'gen_mean':observed_mean(gt),
                      'n_mem_observed':sum(t is not None for t in mt), 'n_gen_observed':sum(t is not None for t in gt),
                      'min_delay':ds['value'], 'unresolved_delay_seeds':sum(r['delay'] is None for r in gg),
                      'n_seeds':5})
    law = normalise_timescale_laws(json.loads((ROOT/'results/cone_cellmeans_min_20261106/fit.json').read_text()))['law']
    rows=[]; boot_rows=[]; rng=np.random.default_rng(20261107)
    for p in sorted({r['p'] for r in cells}):
        cc = sorted([r for r in cells if r['p']==p],key=lambda r:r['P'])
        cross, onset = curve_cross(cc), dense_onset(cc)
        samples=[]
        for draw in range(bootstraps):
            boot=[]
            for r in cc:
                ix=rng.integers(0,5,5)
                boot.append({**r, 'mem_mean':observed_mean([r['mem_times'][i] for i in ix]),
                             'gen_mean':observed_mean([r['gen_times'][i] for i in ix])})
            b=curve_cross(boot)
            value=None if b['value'] is None else math.log10(b['value'])
            boot_rows.append({'p':p,'draw':draw,'log10_crossing':value,'status':b['status']})
            if value is not None:samples.append(value)
        interval = np.quantile(samples,[.025,.975]) if len(samples)>=30 and cross['value'] is not None else [None,None]
        lawP=branches(p,law)[0] if which=='fixed' else None
        identified=[r for r in cc if r['min_delay'] is not None and r['mem_mean'] is not None and r['gen_mean'] is not None]
        agreements=[(r['mem_mean']<r['gen_mean']) == (r['min_delay']>0) for r in identified]
        lo=None if cross['value'] is None else math.log10(cross['value'])
        onset_value=onset['value'] if onset['status']=='observed' else None
        rows.append({'p':p,'n_widths':len(cc),'crossing_log10P':lo,'crossing_status':cross['status'],
                     'ci_lo':interval[0], 'ci_hi':interval[1], 'bootstrap_root_fraction':len(samples)/bootstraps,
                     'onset_log10P':None if onset_value is None else math.log10(onset_value), 'onset_status':onset['status'],
                     'onset_domain_cells':onset['domain_cells'],
                     'onset_rule':onset['rule'],'onset_first_confirmed':onset['first_confirmed'],
                     'onset_earliest_possible':onset['earliest'],'onset_earlier_unresolved':onset['earlier_unresolved'],
                     'law_crossing_log10P':None if lawP is None else math.log10(lawP),
                     'law_status':'not_fitted_for_protocol' if which!='fixed' else 'no_predicted_root' if lawP is None else 'predicted_root',
                     'law_minus_measured_dex':None if lawP is None or lo is None else math.log10(lawP)-lo,
                     'mech_agreement':float(np.mean(agreements)) if agreements else None,
                     'agreement_matches':sum(agreements),'scored_cells':len(agreements),
                     'n_mem_censored':sum(5-r['n_mem_observed'] for r in cc),
                     'n_gen_censored':sum(5-r['n_gen_observed'] for r in cc),
                     'unresolved_delay_seeds':sum(r['unresolved_delay_seeds'] for r in cc)})
    pd.DataFrame(rows).to_csv(out/f'crossing_{which}.csv',index=False)
    pd.DataFrame(cells).to_csv(out/f'cells_{which}.csv',index=False)
    pd.DataFrame(boot_rows).to_csv(out/f'bootstrap_{which}.csv',index=False)
    plot_crossings(which,out,rows,cells,traces)
    finite_errors=[abs(r['law_minus_measured_dex']) for r in rows if r['law_minus_measured_dex'] is not None]
    offsets=[r['onset_log10P']-r['crossing_log10P'] for r in rows if r['onset_log10P'] is not None and r['crossing_log10P'] is not None]
    summary={'which':which,'method':'Observed arithmetic means; first decreasing root of piecewise-linear log times against log P.',
             'scope':'Retrospective reanalysis, not a new preregistered test.',
             'onset_domain':'First positive minimum on the full sampled grid. Any unresolved smaller minimum leaves onset unresolved; left-censored onsets are excluded from point comparisons.',
             'bootstrap':f'{bootstraps} paired seed-label draws within each cell before dropping unobserved events; percentile interval conditional on an observed point root and bootstrap root existence.',
             'registry_statuses':statuses,'completed_runs':len(traces),'cells':len(cells),'seeds_per_cell':5,
             'censored_mem':sum(5-r['n_mem_observed'] for r in cells),'censored_gen':sum(5-r['n_gen_observed'] for r in cells),
             'scored_cells':sum(r['scored_cells'] for r in rows),'agreement_matches':sum(r['agreement_matches'] for r in rows),
             'agreement':sum(r['agreement_matches'] for r in rows)/sum(r['scored_cells'] for r in rows),
             'max_abs_law_error_dex':max(finite_errors) if finite_errors else None,
             'onset_crossing_offsets':{'n':len(offsets),'mean':float(np.mean(offsets)) if offsets else None,'sd':float(np.std(offsets,ddof=1)) if len(offsets)>1 else None},
             'primes':rows,'source_sha256':sha(__file__),'law_sha256':sha(ROOT/'results/cone_cellmeans_min_20261106/fit.json')}
    save_json(out/f'summary_{which}.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('primes','registry_statuses')},indent=2),flush=True)


def plot_crossings(which,out,rows,cells,traces):
    fig, axes=plt.subplots(math.ceil(len(rows)/4),4,figsize=(12,2.5*math.ceil(len(rows)/4)),squeeze=False,sharey=True,layout='constrained')
    for ax,row in zip(axes.flat,rows):
        cc=[r for r in cells if r['p']==row['p']]
        for name,color,marker in [('mem','C3','s'),('gen','C0','o')]:
            # NaNs break the line where no mean is observed.
            yy=[r[f'{name}_mean'] if r[f'{name}_mean'] is not None else np.nan for r in cc]
            ax.plot([r['P'] for r in cc],yy,marker+'-',color=color,ms=3,lw=1)
            partial=[r for r in cc if 0<r[f'n_{name}_observed']<5]
            ax.scatter([r['P'] for r in partial],[r[f'{name}_mean'] for r in partial],marker='x',s=35,color='#b76814',zorder=4)
            # All-censored cell: show follow-up lower bound as an arrow, not a mean.
            missing=[r for r in cc if r[f'n_{name}_observed']==0]
            for r in missing:
                follow=[v['followup_epochs'] for v in traces if v['p']==r['p'] and v['dim']==r['dim'] and v['experiment_type']==('speed' if name=='mem' else 'groks')]
                ax.scatter([r['P']],[min(follow)],marker='^',s=26,facecolors='none',edgecolors=color)
        for key,color,style in [('crossing_log10P','k','--'),('onset_log10P','C2',':'),('law_crossing_log10P','C1','-.')]:
            if row[key] is not None:ax.axvline(10**row[key],color=color,lw=.8,ls=style)
        ax.set(xscale='log',yscale='log')
        ax.text(.5,.95,f'$p={row["p"]}$',transform=ax.transAxes,ha='center',va='top',fontsize=9)
        ax.tick_params(labelsize=7); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_locator(LogLocator(base=10,numticks=4));ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10))
    for ax in axes.flat[len(rows):]:ax.axis('off')
    for ax in axes[-1]:ax.set_xlabel('Parameters $P$',fontsize=8)
    for ax in axes[:,0]:ax.set_ylabel('Epochs',fontsize=8)
    fig.legend(handles=[Line2D([],[],color='C3',marker='s',label='$T_{mem}$ arithmetic mean'),
                        Line2D([],[],color='C0',marker='o',label='$T_{gen}$ arithmetic mean'),
                        Line2D([],[],color='k',ls='--',label='Measured lower crossing'),
                        Line2D([],[],color='C2',ls=':',label='Identified minimum-delay onset'),
                        *([Line2D([],[],color='C1',ls='-.',label='Fitted-law lower crossing')] if which=='fixed' else []),
                        Line2D([],[],color='#b76814',ls='',marker='x',label='Mean excludes censored seeds')],fontsize=8)
    save_paper_figure(fig,out/f'crossing_{which}.pdf',notes=[fig.text(0,0,s,fontsize=8) for s in [
        'Five seeds per cell in this dense-crossing experiment; the separate cone uses ten. Censored times are excluded.',
        'Triangles denote lower bounds for all-censored cells. Curves interpolate log arithmetic means in log parameters.',
        'Missing crossings remain missing. Bootstrap intervals condition on a crossing; root-existence fractions are reported in the table.',
    ]]);plt.close(fig)


def run_analysis(out=OUT, bootstraps=1000, plots_only=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if plots_only:
        for which in ('fixed','mup'):
            rows=json.loads((out/f'summary_{which}.json').read_text())['primes']
            cells=pd.read_csv(out/f'cells_{which}.csv').replace({np.nan:None}).to_dict('records')
            traces=json.loads((out/f'runs_{which}.json').read_text())
            plot_crossings(which,out,rows,cells,traces)
        return
    historic={str(p):sha(p) for p in [ROOT/'results/theory/crossing_fixed.csv',ROOT/'results/theory/crossing_mup.csv',ROOT/'results/theory/forecast_onset.json']}
    for which in ('fixed','mup'):run(which,out,bootstraps)
    assert all(sha(p)==digest for p,digest in historic.items())
    save_json(out/'validation.json',{'historical_results_unchanged':historic,'bootstrap_draws':bootstraps,'raw_events_verified_against_registry':True})
