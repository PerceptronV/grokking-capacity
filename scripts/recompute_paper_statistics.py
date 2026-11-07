#!/usr/bin/env python3
"""Freeze explicit publication cohorts and regenerate measured-curve statistics.

The revised secondary analyses use the declared prime set and seed pool at all
settings, including a separately identified central reference. This is a
retrospective reconciliation; archived point tables are not silently reused.
"""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import yaml

from grokking_capacity.analysis.config_view import ConfigView, ArchGroup, load_npz, _detect_swept_axes
from grokking_capacity.analysis.aggregate import intersection_summary, onset_summary, summarize_delays
from grokking_capacity.analysis.matching import compute_n_equiv
from grokking_capacity.analysis.plots import _curve_for_slice, _slice_values, _passes_filters
from grokking_capacity.analysis.stats import compute_hypothesis_tests, _capacity_baseline, plot_predicted_vs_empirical, plot_error_vs_axis
from plot_cone_matched_aggregations import ROOT, CENTRAL, sha, save_json

OUT=ROOT/'results/paper_revision_20261107/statistics'
CONFIGS=['central','init_scale_sweep','weight_decay_sweep','lr_sweep','alpha_sweep','depth_scaling','task_add','task_mul']
AXES={'init_scale_sweep':'init_scale','weight_decay_sweep':'weight_decay','lr_sweep':'lr','alpha_sweep':'train_fraction','depth_scaling':'depth','task_add':'operation','task_mul':'operation'}


def first(a,t):
    hit=np.flatnonzero(np.asarray(a)>=t)
    return int(hit[0]+1) if len(hit) else None


def prepare():
    central=ConfigView.from_yaml(ROOT/'configs/central.yaml',db_path=str(ROOT/'runs.db'))
    base=next(g for g in central.groups if g.speed_runs and g.groks_runs)
    current_C=json.loads((ROOT/'results/capacity_h100_matched_20261103/summary.json').read_text())['C']
    views={};manifest=[];pool={};selections=[];coverage_exclusions=[]
    for name in CONFIGS:
        spec=yaml.safe_load((ROOT/'configs'/f'{name}.yaml').read_text())
        view=central if name=='central' else ConfigView.from_yaml(ROOT/'configs'/f'{name}.yaml',db_path=str(ROOT/'runs.db'))
        primes=set(CENTRAL if name=='central' else spec['defaults']['primes'])
        seeds=set(range(42,52)) if name=='central' else set(spec['defaults']['seeds'])
        # A whole reference group comes from one host. Do not fill individual
        # missing seeds with hardware replicas selected by their outcomes.
        groups=[g for g in view.groups if g.speed_runs or g.groks_runs]
        if name!='central':
            groups=[g for g in groups if g.key!=base.key]
            groups.append(base)
        groups=deepcopy(groups)
        for gi,g in enumerate(groups):
            g.capacity_constant=current_C;g.capacity_constant_source='matched H100 capacity normalization; held fixed across these comparisons'
            for attr,kind in [('speed_runs','mem'),('groks_runs','gen')]:
                rows=[r for r in getattr(g,attr) if r['p'] in primes and r['seed'] in seeds and r['dim']<=256]
                # Matched memorisation dataset size is a protocol invariant.
                for r in rows:
                    assert r.get('mup_base')==0
                    if kind=='mem':assert r['n_samples']==compute_n_equiv(r['p'],g.key.operation,g.key.train_fraction)[0]
                bins=defaultdict(list)
                for r in rows:bins[r['p'],r['dim']].append(r)
                retained=[]
                for cell,rr in bins.items():
                    assert len({r['seed'] for r in rr})==len(rr),(name,kind,cell,'duplicate seed')
                    if {r['seed'] for r in rr} != seeds:
                        coverage_exclusions.append({'config':name,'group':gi,'kind':kind,'p':cell[0],'dim':cell[1],'available_seeds':sorted(r['seed'] for r in rr),'required_seeds':sorted(seeds),'reason':'incomplete planned seed pool; no outcome-based selection'})
                        continue
                    assert len({r['host'] for r in rr})==1
                    retained.extend(rr)
                rows=retained
                setattr(g,attr,rows)
                for r in rows:
                    pool[r['uuid']]=r
                    manifest.append({'config':name,'group':gi,'kind':kind,**{k:r[k] for k in ('uuid','p','dim','seed','host','gpu_type','mup_base','param_count')}})
            selections.append({'config':name,'group':gi,'key':asdict(g.key),'reference':g.key==base.key,
                'mem_runs':len(g.speed_runs),'gen_runs':len(g.groks_runs),'seeds':sorted(seeds),'primes':sorted(primes),
                'hosts':sorted({r['host'] for r in g.speed_runs+g.groks_runs}),
                'gpu_types':sorted({r['gpu_type'] for r in g.speed_runs+g.groks_runs})})
        view=replace(view,groups=groups,swept_axes=_detect_swept_axes(groups),intersection_figures=[view.intersection_figures[0]])
        views[name]=view
    return views,pool,manifest,selections,coverage_exclusions


def raw_events(row):
    with load_npz(row) as z:
        if row['experiment_type']=='speed':
            mem=first(z['train_acc_trace'],99)
            assert (mem is not None)==(row.get('saturated') is not False)
            if mem is not None:assert np.isclose(mem,row['saturation_epoch'])
            return {'uuid':row['uuid'],'mem99':mem,'train99':None,'val98':None,'val99':None,'delay':None,'followup':len(z['train_acc_trace'])}
        t,v,g=first(z['train_acc'],99),first(z['val_acc'],98),first(z['val_acc'],99)
        assert g==row['grokking_epoch']
        delay=None if v is None else max(0,v-t) if t is not None else 0
        return {'uuid':row['uuid'],'mem99':None,'train99':t,'val98':v,'val99':g,'delay':delay,'followup':len(z['val_acc'])}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    views,pool,manifest,selections,coverage_exclusions=prepare()
    save_json(OUT/'coverage_exclusions.json',coverage_exclusions)
    pd.DataFrame(manifest).to_csv(OUT/'cohort_runs.csv',index=False)
    save_json(OUT/'cohorts.json',selections)
    with ThreadPoolExecutor(max_workers=8) as workers:
        events={r['uuid']:r for r in workers.map(raw_events,pool.values())}
    save_json(OUT/'raw_events.json',list(events.values()))
    overview=[];within=[];all_cross=[]
    for name,view in views.items():
        figure=replace(view.intersection_figures[0],timescale_aggregation='mean',delay_aggregation='min',delay_censoring='unresolved')
        rows=[];cell_rows=[]
        for gi,group in enumerate(view.groups):
            for sv in _slice_values(group,figure):
                speed=_curve_for_slice(group.speed_runs,figure,sv,'saturation_epoch')
                gen=_curve_for_slice(group.groks_runs,figure,sv,'grokking_epoch')
                crossing=intersection_summary(speed,gen,direction="decreasing")
                bins=defaultdict(list)
                for r in group.groks_runs:
                    if r[figure.slice_field]==sv and _passes_filters(r,figure):bins[r[figure.x_field]].append(events[r['uuid']]['delay'])
                delay_curve={P:summarize_delays(dd,'min')['value'] for P,dd in bins.items()}
                onset=onset_summary(delay_curve)
                empirical=onset['value'] if onset['status']=='observed' else None
                prediction=crossing['value']
                k,capacity=_capacity_baseline(group,figure,sv)
                row={'config':name,'group':gi,'slice_field':figure.slice_field,'slice_value':sv,figure.slice_field:sv,
                     'x_field':figure.x_field,'predicted_onset_x':prediction,'empirical_onset_x':empirical,
                     'crossing_status':crossing['status'],'crossing_count':len(crossing['roots']),
                     'empirical_onset_status':onset['status'],'empirical_onset_bound':onset['value'],
                     'log_ratio':math.log10(empirical/prediction) if empirical and prediction else None,
                     'capacity_constant':group.capacity_constant,'k_mem_bits':k,'capacity_baseline_x':capacity,
                     'n_delay_cells':len(bins),'unresolved_delay_seeds':sum(d is None for vv in bins.values() for d in vv),
                     'n_seeds':len({r['seed'] for r in group.groks_runs}),
                     'delay_aggregation':'min','timescale_aggregation':'mean','delay_censoring':'unresolved'}
                for axis in view.swept_axes:row[axis]=getattr(group.key,axis)
                rows.append(row)
                for P,dd in bins.items():cell_rows.append({'group':gi,'p':sv,'P':P,**summarize_delays(dd,'min')})
                all_cross.append({'config':name,'group':gi,'p':sv,**crossing})
        df=pd.DataFrame(rows);dest=OUT/name;dest.mkdir(exist_ok=True)
        df.to_csv(dest/'predictiveness.csv',index=False);pd.DataFrame(cell_rows).to_csv(dest/'delay_cells.csv',index=False)
        tests=compute_hypothesis_tests(view,figure,df);save_json(dest/'hypothesis_tests.json',tests)
        plot_predicted_vs_empirical(df,dest/'predicted_vs_empirical.pdf')
        for axis in view.swept_axes:plot_error_vs_axis(df,axis,dest/f'error_vs_{axis}.pdf')
        valid=df.dropna(subset=['log_ratio'])
        overview.append({'config':name,'total':len(df),'valid':len(valid),
                         'rho':tests.get('predictiveness',{}).get('spearman',{}).get('rho'),
                         'ccc':tests.get('calibration',{}).get('ccc',{}).get('value'),
                         'median_log_ratio':float(valid.log_ratio.median()) if len(valid) else None,
                         'crossing_statuses':dict(Counter(df.crossing_status)),
                         'onset_statuses':dict(Counter(df.empirical_onset_status))})
        axis=AXES.get(name)
        if axis:
            for setting,rr in df.groupby(axis):
                vv=rr.dropna(subset=['log_ratio'])
                within.append({'config':name,'axis':axis,'setting':setting,'total':len(rr),'valid':len(vv),
                               'median':float(vv.log_ratio.median()) if len(vv) else None,
                               'sd':float(vv.log_ratio.std(ddof=1)) if len(vv)>1 else None,
                               'rho':float(spearmanr(vv.predicted_onset_x,vv.empirical_onset_x).statistic) if len(vv)>2 else None,
                               'no_crossing':int(rr.predicted_onset_x.isna().sum()),
                               'onset_not_identified':int(rr.empirical_onset_x.isna().sum())})
        print(name,overview[-1],flush=True)
    pd.DataFrame(overview).to_csv(OUT/'overview.csv',index=False);save_json(OUT/'overview.json',overview)
    pd.DataFrame(within).to_csv(OUT/'within_setting.csv',index=False);save_json(OUT/'crossings.json',all_cross)
    save_json(OUT/'validation.json',{'selected_unique_runs':len(pool),'selected_run_memberships':len(manifest),'raw_events_checked':len(events),
        'every_cell_exact_declared_seed_pool':True,'one_hardware_cohort_per_cell':True,'mup_base_zero_throughout':True,
        'source_sha256':sha(__file__),'configs':{str(ROOT/'configs'/f'{n}.yaml'):sha(ROOT/'configs'/f'{n}.yaml') for n in CONFIGS},
        'cohort_note':'Primary: ten central primes and ten seeds. Secondary: declared prime and seed sets applied equally at every setting; baseline explicitly inherited from central A100 cohort. Baseline width grids remain denser; seed counts and hardware do not change within a cell.',
        'retrospective':True})


if __name__=='__main__':main()
