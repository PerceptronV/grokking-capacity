#!/usr/bin/env python3
"""Matched-cohort diagnostics, calibration sensitivity and review report inputs."""
from pathlib import Path
from collections import defaultdict
import json,math,sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analyse_seed_aggregation import ROOT,OUT,METRICS,save,csvsave,clean
from grokking_capacity.analysis.stats import _ccc,_joint_calibration_ftest,compute_hypothesis_tests
from grokking_capacity.analysis.config_view import ConfigView
from dataclasses import replace
from check_law_complete_ten_seeds import crossing


def metrics(df):
    lp=np.log10(df.predicted_onset_x.to_numpy());le=np.log10(df.empirical_onset_x.to_numpy());r=le-lp
    return {'n':len(df),'rho':float(spearmanr(lp,le).statistic) if len(df)>2 and np.std(lp)>0 and np.std(le)>0 else None,
            'ccc':_ccc(lp,le),'MAE_dex':float(np.mean(np.abs(r))),'median_offset_dex':float(np.median(r)),
            'sd_offset_dex':float(np.std(r,ddof=1)) if len(r)>1 else None,'joint_F_p':_joint_calibration_ftest(lp,le)['p']}


def matched():
    overview=pd.read_csv(OUT/'statistics/overview.csv');result=[];intervals=[]
    for config in overview.config.unique():
        frames={m:pd.read_csv(OUT/f'statistics/{config}/intersection/{m}/predictiveness.csv') for m in METRICS}
        for df in frames.values():df['cell_key']=[f'{g}:{sv}' for g,sv in zip(df.group,df.slice_value)]
        keys=set.intersection(*[set(df.loc[(df.onset_status=='observed')&df.predicted_onset_x.notna(),'cell_key']) for df in frames.values()])
        if len(keys)<3:continue
        selected={m:df[df.cell_key.isin(keys)].sort_values('cell_key').reset_index(drop=True) for m,df in frames.items()}
        for m,df in selected.items():result.append({'config':config,'metric':m,**metrics(df)})
        # Paired prime-cluster resampling protects against treating settings at
        # the same prime as independent. The statistic is descriptive MAE.
        base=selected['min'];primes=base.slice_value.unique();rng=np.random.default_rng(20261106);draws=defaultdict(list)
        residuals={m:np.log10(df.empirical_onset_x.to_numpy()/df.predicted_onset_x.to_numpy()) for m,df in selected.items()}
        byp={p:np.flatnonzero(base.slice_value.to_numpy()==p) for p in primes}
        for _ in range(2000):
            idx=np.concatenate([byp[p] for p in rng.choice(primes,len(primes),replace=True)])
            for m in ['median','mean']:draws[m].append(float(np.mean(np.abs(residuals[m][idx]))-np.mean(np.abs(residuals['min'][idx]))))
        for m,dd in draws.items():intervals.append({'config':config,'metric_vs_min':m,'common_observed_cells':len(base),
            'MAE_change_dex':float(np.mean(np.abs(residuals[m]))-np.mean(np.abs(residuals['min']))),
            'paired_prime_bootstrap_95':np.quantile(dd,[.025,.975]).tolist()})
    csvsave('statistics/common_observed_cohort.csv',result);save('statistics/paired_MAE_changes.json',intervals)


def calibration():
    laws=json.loads((OUT/'laws/comparison.json').read_text());base=laws['baseline_asymmetric_estimators']
    fits={'original':base['current'],'complete_original':base['complete_both']}
    for name,v in laws['unified_cell_reducers'].items():
        if v['metric'] in ['mean','median']:fits[name]=v
    rows=[];nulls=[]
    for cohort in ['historical_A100','canonical_ten_seed']:
        for metric in METRICS:
            folder=metric if cohort=='historical_A100' else metric+'_canonical'
            df=pd.read_csv(OUT/f'statistics/central/intersection/{folder}/predictiveness.csv')
            train=df[df.slice_value<149];on149=float(df[df.slice_value==149].empirical_onset_x.iloc[0])
            for name,f in fits.items():
                mem,gen=f['mem'],f['gen']
                residues=[math.log10(row.empirical_onset_x/crossing(row.slice_value,mem,gen,2.16)) for row in train.itertuples()]
                delta=float(np.median(residues))
                p149=crossing(149,mem,gen,2.16);p197=crossing(197,mem,gen,2.16)
                pred=math.log10(p149)+delta if p149 else None
                rows.append({'cohort':cohort,'delay_metric':metric,'law_fit':name,'calibration_primes':len(train),
                    'delta_dex':delta,'p149_predicted_log10P':pred,'p149_measured_log10P':math.log10(on149),
                    'p149_residual_dex':math.log10(on149)-pred if pred is not None else None,
                    'p197_predicted_log10P':math.log10(p197)+delta if p197 else None})
            lp=np.log10(train.slice_value);lP=np.log10(train.empirical_onset_x);X=np.column_stack([np.ones(len(lp)),lp]);cf=np.linalg.lstsq(X,lP,rcond=None)[0]
            pred=float(cf@[1,math.log10(197)]);rng=np.random.default_rng(20261106);draws=[]
            for _ in range(2000):
                ii=rng.integers(0,len(train),len(train));b=np.linalg.lstsq(X[ii],lP.to_numpy()[ii],rcond=None)[0]
                draws.append(float(b@[1,math.log10(197)]))
            nulls.append({'cohort':cohort,'delay_metric':metric,'slope':float(cf[1]),'p197_log10P':pred,
                'prime_bootstrap_fit_interval':np.quantile(draws,[.025,.975]).tolist(),
                'note':'Exploratory re-fit of rival under changed estimand; fit interval excludes onset seed-noise term and is not the registered prediction.'})
    csvsave('laws/recalibrated_forecasts.csv',rows);save('laws/refitted_powerlaw_null.json',nulls)


def audit_inputs():
    cache=json.loads((OUT/'statistics/events.json').read_text())
    save('statistics/trace_availability.json',{'unique_traces_scored_or_reused':len(cache),'missing_traces':[k for k,v in cache.items() if v.get('missing_trace')]})
    # Paper claims are frozen; explicitly flag inability to reproduce older
    # hyperparameter table denominators with the current registry.
    expected={'central':(10,.976),'init_scale_sweep':(15,.839),'weight_decay_sweep':(42,.900),
              'lr_sweep':(15,.972),'alpha_sweep':(31,.977),'depth_scaling':(9,.600),'task_add':(10,.801),'task_mul':(10,.648)}
    df=pd.read_csv(OUT/'statistics/overview.csv');out=[]
    for name,(n,rho) in expected.items():
        r=df[(df.config==name)&(df.figure=='intersection')&(df.variant=='legacy_min')].iloc[0]
        if name=='central':
            central=pd.read_csv(OUT/'statistics/central/intersection/min/predictiveness.csv');central=central[central.slice_value<149]
            rr=metrics(central);r=r.copy();r['n_valid']=rr['n'];r['rho']=rr['rho']
        out.append({'config':name,'paper_n':n,'paper_rho':rho,'legacy_recomputed_n':int(r.n_valid),'legacy_recomputed_rho':float(r.rho),
            'matches_paper_to_reported_precision':int(r.n_valid)==n and abs(float(r.rho)-rho)<.0006,
            'note':'Mismatch precedes switching aggregation; current-registry sensitivity is not a verified replacement for this paper table.'})
    csvsave('statistics/paper_baseline_reproduction.csv',out)


def central_main():
    view=ConfigView.from_yaml(ROOT/'configs/central.yaml',db_path=str(ROOT/'runs.db'))
    figure=view.intersection_figures[0];summary=[]
    for variant in ['min','median','mean','median_all']:
        df=pd.read_csv(OUT/f'statistics/central/intersection/{variant}/predictiveness.csv')
        df=df[df.slice_value<149].copy();assert len(df)==10
        df['dropout']=.2
        test=compute_hypothesis_tests(view,figure,df)
        save(f'statistics/central_main_ten/{variant}/hypothesis_tests.json',test)
        csvsave(f'statistics/central_main_ten/{variant}/predictiveness.csv',df.to_dict('records'))
        summary.append({'variant':variant,**metrics(df),'permutation_p':test['predictiveness']['spearman']['p_permutation'],
            'slope':test['calibration']['joint_ftest']['slope'],'intercept':test['calibration']['joint_ftest']['intercept'],
            'kendall':test['predictiveness']['kendall']['tau']})
    csvsave('statistics/central_main_ten/overview.csv',summary)


def main():
    central_main();matched();calibration();audit_inputs()
    print('Matched-cohort, recalibrated forecast and paper-baseline audit complete.',flush=True)
if __name__=='__main__':main()
