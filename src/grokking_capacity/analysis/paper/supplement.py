#!/usr/bin/env python3
"""Regenerate primary example plots and estimator/threshold sensitivities."""
from dataclasses import replace
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .paths import RESULTS
import math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from grokking_capacity.analysis.config_view import load_npz
from grokking_capacity.analysis.aggregate import intersection_summary, summarize_delays, onset_summary, sustained_onset_summary
from grokking_capacity.analysis.plots import render_intersection, _passes_filters
from .statistics import prepare, first
from grokking_capacity.utils.artifacts import save_json, sha

OUT=RESULTS/'supplement'


def render_primary_figures(view, out=OUT):
    group=view.groups[0]
    primes=[97,113,139]
    figure=view.intersection_figures[0]
    colours=[r[figure.colour_field] for r in group.groks_runs
             if r['p'] in primes and _passes_filters(r,figure)]
    colour_limits=(float(min(colours)),float(max(colours)))
    for p in primes:
        sub=replace(group,speed_runs=[r for r in group.speed_runs if r['p']==p],groks_runs=[r for r in group.groks_runs if r['p']==p])
        render_intersection(sub,out/'primary_figures',figure,
                            show_unresolved=False,legend_inside=True,
                            colour_limits=colour_limits,show_colourbar=p==primes[-1])


def run(out=OUT, statistics=None, figures_only=False):
    OUT=Path(out)
    statistics=Path(statistics) if statistics is not None else OUT.parent/'statistics'
    OUT.mkdir(parents=True,exist_ok=True)
    views,pool,manifest,_,_=prepare();view=views['central'];group=view.groups[0]
    assert len(view.groups)==1
    if figures_only:
        render_primary_figures(view,OUT)
        print('Regenerated primary PDF/PNG panels with in-panel legends and flagged delay points omitted.',flush=True)
        return
    primary={r['uuid']:r for r in group.speed_runs+group.groks_runs}
    def read(row):
        with load_npz(row) as z:
            if row['experiment_type']=='speed':
                values={f'mem{t}':first(z['train_acc_trace'],t) for t in [95,98,99]}
            else:
                values={f'{arm}{t}':first(z[f'{arm}_acc'],t) for arm in ['train','val'] for t in [95,98,99]}
        return {'uuid':row['uuid'],'p':row['p'],'dim':row['dim'],'P':row['param_count'],'kind':row['experiment_type'],
                'trace_path':row['npz_path'],'trace_sha256':sha(row['npz_path']),**values}
    with ThreadPoolExecutor(max_workers=8) as workers:events=list(workers.map(read,primary.values()))
    pd.DataFrame(events).to_csv(OUT/'primary_event_manifest.csv',index=False)
    variants=[('canonical',99,99,98,'min'),('matched95',95,95,95,'min'),('matched98',98,98,98,'min'),('matched99',99,99,99,'min'),
              ('median_delay',99,99,98,'median'),('mean_delay',99,99,98,'mean'),('sustained_tail',99,99,98,'min')]
    comparisons=[];point_rows=[]
    for name,train,gen,val,metric in variants:
        for p in sorted({r['p'] for r in events}):
            cells=defaultdict(lambda:defaultdict(list))
            for r in events:
                if r['p']!=p:continue
                if r['kind']=='speed':
                    if r[f'mem{train}'] is not None:cells[r['P']]['mem'].append(r[f'mem{train}'])
                else:
                    if r[f'val{gen}'] is not None:cells[r['P']]['gen'].append(r[f'val{gen}'])
                    t,v=r[f'train{train}'],r[f'val{val}']
                    delay=None if v is None else max(0,v-t) if t is not None else 0
                    cells[r['P']]['delays'].append(delay)
            m={P:float(np.mean(c['mem'])) for P,c in cells.items() if c['mem']}
            g={P:float(np.mean(c['gen'])) for P,c in cells.items() if c['gen']}
            d={P:summarize_delays(c['delays'],metric)['value'] for P,c in cells.items() if c['delays']}
            cr=intersection_summary(m,g,direction='decreasing');on=sustained_onset_summary(d) if name=='sustained_tail' else onset_summary(d)
            observed=on['value'] if on['status']=='observed' else None
            point_rows.append({'variant':name,'p':p,'crossing':cr['value'],'crossing_status':cr['status'],
                               'onset':observed,'onset_status':on['status'],
                               'log_ratio':math.log10(observed/cr['value']) if observed and cr['value'] else None})
        vv=[r for r in point_rows if r['variant']==name and r['log_ratio'] is not None]
        comparisons.append({'variant':name,'train_threshold':train,'gen_threshold':gen,'delay_val_threshold':val,'delay_aggregation':metric,
                            'valid':len(vv),'eligible':10,'median_log_ratio':float(np.median([r['log_ratio'] for r in vv])) if vv else None,
                            'rho':float(spearmanr([r['crossing'] for r in vv],[r['onset'] for r in vv]).statistic) if len(vv)>2 else None})
    save_json(OUT/'sensitivities.json',comparisons);pd.DataFrame(point_rows).to_csv(OUT/'sensitivity_points.csv',index=False)
    baseline=pd.read_csv(statistics/'central/predictiveness.csv')
    canonical=pd.DataFrame([r for r in point_rows if r['variant']=='canonical']).sort_values('p')
    np.testing.assert_allclose(canonical.crossing,baseline.sort_values('p').predicted_onset_x,rtol=1e-12)
    np.testing.assert_allclose(canonical.onset,baseline.sort_values('p').empirical_onset_x,rtol=1e-12)
    render_primary_figures(view,OUT)
    save_json(OUT/'validation.json',{'fresh_raw_canonical_matches_main_statistics':True,'raw_runs':len(events),'source_sha256':sha(__file__)})
    print(comparisons,flush=True)
