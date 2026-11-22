#!/usr/bin/env python3
"""Isolated aggregation sensitivity; no training, publication, or coefficient writes."""
from __future__ import annotations
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import replace
import argparse, csv, hashlib, itertools, json, math, sys
import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'results/seed_aggregation_sensitivity_20261106'
METRICS=('min','median','mean')

from grokking_capacity.experiments.cone_workflow.cone_data import aggregate as cone_aggregate, law_coefficients, law_onset
from grokking_capacity.analysis.aggregate import (summarize_delays, reduce_values, delay_curve,
    onset_summary, aggregate_over_seeds, find_intersection)
from grokking_capacity.analysis.config_view import ConfigView, load_npz
from grokking_capacity.analysis.plots import _passes_filters, _slice_values, _curve_for_slice
from grokking_capacity.analysis.stats import compute_hypothesis_tests, _capacity_baseline
from grokking_capacity.analysis.law_sensitivity.check_law_complete_ten_seeds import fit, fold, crossing


def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.bool_,)):return bool(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x

def save(name,x):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')

def csvsave(name,rows):
    rows=list(rows)
    if not rows:return
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)

def frozen():return json.loads((ROOT/'results/cone_ten_seeds_h100_20261106/figure/runs.json').read_text())


def cone():
    rows=frozen(); variants={m:cone_aggregate(rows,m) for m in METRICS}
    baseline=json.loads((ROOT/'results/cone_ten_seeds_h100_20261106/figure/cells.json').read_text())
    for old,new in zip(baseline,variants['min']):
        for k,v in old.items():assert new[k]==v,(k,old['p'],old['dim'])
    by=defaultdict(list)
    for r in rows:by[r['p'],r['dim']].append(r)
    all_cells=[];primes=[];onsets=[];thresholds=[]
    for metric,cells in variants.items():
        csvsave(f'cone/cells_{metric}.csv',cells)
        for p in sorted({c['p'] for c in cells}):
            cc=[c for c in cells if c['p']==p]
            primes.append({'metric':metric,'p':p,'cells':len(cc),**dict(Counter(c['state'] for c in cc)),
                'incomplete_cells':sum(c['unresolved_seeds']>0 for c in cc),
                'max_exact_delay':max((c['aggregate_delay'] for c in cc if c['aggregate_delay'] is not None),default=None)})
            for domain in ['all_widths','dim_le_256']:
                subset=[c for c in cc if domain=='all_widths' or c['dim']<=256]
                curve={c['param_count']:c['aggregate_delay'] for c in subset}
                onset=onset_summary(curve)
                pos=[c for c in subset if c['state']=='positive']
                onsets.append({'metric':metric,'p':p,'domain':domain,**onset,
                    'log10P':None if onset['value'] is None else math.log10(onset['value']),
                    'first_positive_dim':pos[0]['dim'] if pos else None,
                    'first_positive_P':pos[0]['param_count'] if pos else None})
            for threshold in [0,1,5,10]:
                thresholds.append({'metric':metric,'p':p,'threshold_strictly_greater_than':threshold,
                    'positive':sum(c['aggregate_delay'] is not None and c['aggregate_delay']>threshold for c in cc),
                    'nonpositive':sum(c['aggregate_delay'] is not None and c['aggregate_delay']<=threshold for c in cc),
                    'unresolved':sum(c['aggregate_delay'] is None for c in cc)})
        all_cells.extend({'metric':metric,**c} for c in cells)
    # Same fully observed cells in all comparisons: no missingness selection advantage.
    complete={m:[c for c in cc if c['unresolved_seeds']==0] for m,cc in variants.items()}
    counts={m:{'all_cells':dict(Counter(c['state'] for c in cc)),
               'common_complete_cells':dict(Counter(c['state'] for c in complete[m])),
               'aggregate_unresolved':sum(c['aggregate_delay'] is None for c in cc)} for m,cc in variants.items()}
    # Registered p197 grid is kept distinct from the expanded cone grid.
    import yaml
    dims=set(yaml.safe_load((ROOT/'configs/p197_forecast.yaml').read_text())['experiments']['groks']['dims'])
    registered=[]
    for metric,cc in variants.items():
        sub=[c for c in cc if c['p']==197 and c['dim'] in dims];assert len(sub)==18
        registered.append({'metric':metric,'counts':dict(Counter(c['state'] for c in sub)),
            'onset':onset_summary({c['param_count']:c['aggregate_delay'] for c in sub}),
            'positive_cells':[{'dim':c['dim'],'delay':c['aggregate_delay']} for c in sub if c['state']=='positive']})
    # Sensitivity to using five versus ten seeds: enumerate all 252 five-seed subsets.
    seed_sensitivity=[]
    for p,d in sorted(by):
        rr=sorted(by[p,d],key=lambda r:r['seed'])
        if any(r['delay'] is None for r in rr):continue
        x=np.array([r['delay'] for r in rr]); subs=np.array(list(itertools.combinations(range(10),5)))
        for metric,fn in [('min',np.min),('median',np.median),('mean',np.mean)]:
            values=fn(x[subs],axis=1);full=float(fn(x))
            seed_sensitivity.append({'p':p,'dim':d,'metric':metric,'ten_seed_delay':full,
                'five_seed_positive_fraction':float(np.mean(values>0)),
                'five_vs_ten_state_disagreement_fraction':float(np.mean((values>0)!=(full>0))),
                'five_seed_delay_min':float(values.min()),'five_seed_delay_max':float(values.max())})
    ss={m:{'cells':sum(r['metric']==m for r in seed_sensitivity),
            'mean_five_vs_ten_state_disagreement':float(np.mean([r['five_vs_ten_state_disagreement_fraction'] for r in seed_sensitivity if r['metric']==m]))} for m in METRICS}
    csvsave('cone/prime_counts.csv',primes);csvsave('cone/onsets.csv',onsets)
    csvsave('cone/threshold_sensitivity.csv',thresholds);csvsave('cone/five_seed_subsets.csv',seed_sensitivity)
    save('cone/summary.json',{'counts':counts,'registered_p197':registered,'seed_count_sensitivity':ss,
        'baseline_min_reproduced':True,'censoring':'Unresolved delays remain unknown in [0,infinity]; identified min/median may coexist with unresolved seeds, marked in all plots.'})
    hardware()
    plot_cones(variants)
    print('CONE',json.dumps(clean({'counts':counts,'registered_p197':registered,'seed_sensitivity':ss})),flush=True)


def hardware():
    pairs=pd.read_csv(ROOT/'results/a100_h100_grokking_agreement_20261106/paired_runs.csv')
    output=[];cells=[]
    for metric in METRICS:
        agree=[];diffs=[]
        for (p,d),g in pairs.groupby(['p','dim']):
            a=reduce_values(g.A100_delay,metric);h=reduce_values(g.H100_delay,metric)
            agree.append((a>0)==(h>0));diffs.append(h-a)
            cells.append({'metric':metric,'p':p,'dim':d,'A100':a,'H100':h,'classification_agrees':agree[-1]})
        output.append({'metric':metric,'cells':len(agree),'classification_agreement':sum(agree),
            'mean_absolute_aggregate_difference':float(np.mean(np.abs(diffs))),
            'median_difference':float(np.median(diffs))})
    csvsave('hardware/cells.csv',cells);save('hardware/summary.json',output)


def plot_cones(variants):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from grokking_capacity.analysis._figure_style import save_paper_figure
    coefficients=law_coefficients()
    new=json.loads((ROOT/'results/law_complete_ten_seed_sensitivity_20261106/comparison.json').read_text())['comparisons']['complete_both']
    for fitname in ['published_fit','complete_ten_fit']:
        fig,axs=plt.subplots(1,3,figsize=(15.0,5.3),sharex=True,sharey=True,layout='constrained')
        norm=LogNorm(.1,1500);cmap=plt.get_cmap('viridis')
        pstar=coefficients['capacity_update']['fold_new'] if fitname=='published_fit' else new['divergence_prime']
        pp=np.linspace(97,pstar-1e-7,450)
        curve=[law_onset(p,coefficients) for p in pp] if fitname=='published_fit' else [crossing(p,new['mem'],new['gen'],2.16) for p in pp]
        for ax,metric in zip(axs,METRICS):
            cells=variants[metric]
            for state in ['zero','positive']:
                cc=[c for c in cells if c['state']==state]
                xx=[c['p'] for c in cc]; yy=[c['param_count'] for c in cc]
                if state=='positive':ax.scatter(xx,yy,c=[c['aggregate_delay'] for c in cc],norm=norm,cmap=cmap,s=15,edgecolors='#20333a',linewidths=.25,zorder=3)
                else:ax.scatter(xx,yy,s=11,facecolors='none',edgecolors='#9ba1aa',linewidths=.5,alpha=.7,zorder=1)
            incomplete=[c for c in cells if c['unresolved_seeds']]
            ax.scatter([c['p'] for c in incomplete],[c['param_count'] for c in incomplete],marker='x',c='#b86b14',s=22,lw=.7,zorder=4)
            ax.plot(pp,curve,color='#c7353b',lw=1.4);ax.axvline(pstar,color='#c7353b',ls='--',lw=.9)
            ax.set(xlabel='Prime $p$',yscale='log',xlim=(95,207 if fitname=='complete_ten_fit' else 201))
            ax.set_xticks([97,113,139,163,179,199]);ax.tick_params(labelsize=8)
            ax.grid(axis='y',alpha=.25);ax.spines[['top','right']].set_visible(False)
        axs[0].set_ylabel('Parameters $P$')
        cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axs,fraction=.017,pad=.015)
        cb.set_label('Aggregate delay (epochs)')
        handles=[Line2D([],[],marker='o',ls='',color=cmap(.5),label='Positive aggregate'),
                 Line2D([],[],marker='o',ls='',markerfacecolor='none',markeredgecolor='#9ba1aa',label='Zero aggregate'),
                 Line2D([],[],marker='x',ls='',color='#b86b14',label='At least one unresolved seed'),
                 Line2D([],[],color='#c7353b',label='Unchanged law crossing')]
        axs[0].legend(handles=handles)
        notes=[f'Left: minimum. Centre: median. Right: mean. Same 1,736 cells and ten seeds per cell; train 99%, validation 98%.',
               f'Colour uses the named aggregate. Unknown delays are never replaced by the epoch cap; orange marks remain even when an aggregate is identified.',
               f'Curve: {"published" if fitname=="published_fit" else "complete-ten-seed sensitivity"} timescale fit, fold p* = {pstar:.2f}. Changing the delay summary alone does not refit the curve.']
        save_paper_figure(fig,OUT/f'cone_{fitname}_aggregation_comparison.pdf',notes=[fig.text(0,0,n,fontsize=9) for n in notes],max_columns=4);plt.close(fig)
    # A direct reliability view keeps probability and delay magnitude distinct.
    fig,ax=plt.subplots(figsize=(7.5,5.3),layout='constrained')
    cells=variants['min'];cc=[c for c in cells if not c['unresolved_seeds']]
    sc=ax.scatter([c['p'] for c in cc],[c['param_count'] for c in cc],
        c=[c['positive_seeds']/10 for c in cc],vmin=0,vmax=1,cmap='viridis',s=18,edgecolors='#47535b',linewidths=.25)
    incomplete=[c for c in cells if c['unresolved_seeds']]
    ax.scatter([c['p'] for c in incomplete],[c['param_count'] for c in incomplete],marker='x',c='#b86b14',s=25,lw=.8,label='Unresolved seed outcomes')
    ax.set(yscale='log',xlabel='Prime $p$',ylabel='Parameters $P$',xlim=(95,201))
    ax.set_xticks([97,113,127,139,149,163,179,199]);ax.grid(axis='y',alpha=.25);ax.spines[['top','right']].set_visible(False)
    cb=fig.colorbar(sc,ax=ax,pad=.02);cb.set_label('Fraction of ten seeds with positive delay');ax.legend()
    save_paper_figure(fig,OUT/'positive_seed_fraction.pdf',notes=[fig.text(0,0,
        'Complete cells show the observed positive-delay fraction. Orange crosses retain incomplete cells;\n'
        'their fraction bounds are in the cell tables. Zero delay means validation reached 98% without lagging training 99%.',fontsize=9)],max_columns=1)
    plt.close(fig)
    # Show the scale of high-prime changes, including zeros and unresolved cells.
    fig,axs=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    colors={'min':'#333333','median':'#147c80','mean':'#d56a20'}
    for ax,p in zip(axs,[149,197,199]):
        for metric in METRICS:
            cc=[c for c in variants[metric] if c['p']==p]
            ax.plot([c['param_count'] for c in cc],[np.nan if c['aggregate_delay'] is None else c['aggregate_delay'] for c in cc],'.-',lw=1,ms=4,color=colors[metric],label=metric)
        ax.set(xscale='log',xlabel=f'Parameters $P$ (p = {p})')
        if p == 149:
            ax.set_yscale('symlog',linthresh=1);ax.set_yticks([0,1,5,10,20]);ax.set_yticklabels(['0','1','5','10','20']);ax.set_ylim(-.1,22)
        else:
            ax.set_ylim(-.05,1.8);ax.set_yticks([0,.5,1,1.5])
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Aggregate delay (epochs)');axs[0].legend()
    save_paper_figure(fig,OUT/'delay_profiles.pdf',notes=[fig.text(0,0,'Same ten-seed cells. Gaps indicate an unresolved aggregate; zero is an observed aggregate, not failed generalisation.',fontsize=9)],max_columns=3);plt.close(fig)


EVENTS={}
def event(row):
    u=row['uuid']
    if u not in EVENTS:
        try:
            with load_npz(row) as z:
                tr=np.flatnonzero(z['train_acc']>=99); va=np.flatnonzero(z['val_acc']>=98)
                EVENTS[u]={'train99':int(tr[0]+1) if len(tr) else None,'val98':int(va[0]+1) if len(va) else None,'followup_epochs':len(z['val_acc'])}
        except (FileNotFoundError,KeyError):
            EVENTS[u]={'train99':None,'val98':None,'followup_epochs':None,'missing_trace':True}
    return EVENTS[u]


def get_delay(row,legacy=False):
    e=event(row);t,v=e['train99'],e['val98']
    if legacy:
        if t is None:return None,False
        return float(max(0,(v if v is not None else e['followup_epochs']+1)-t)),True
    if v is None:return None,True
    return float(max(0,v-t)) if t is not None else 0.,True


def onset_rows(view,figure,metric,*,legacy=False,canonical=False,time_metric='mean'):
    output=[];summary=[]
    for gi,group in enumerate(view.iter_groups()):
        g_rows=group.groks_runs
        if canonical and group.speed_runs and group.groks_runs:
            g_rows=[r for r in frozen() if r['p']<=149]
        f=replace(figure,delay_aggregation=metric,timescale_aggregation=time_metric)
        for sv in _slice_values(group,f):
            bins=defaultdict(list)
            selected=[r for r in g_rows if r.get(f.slice_field)==sv and _passes_filters(r,f) and r.get(f.x_field) is not None]
            for r in selected:
                delay,include=get_delay(r,legacy)
                if include:bins[float(r[f.x_field])].append(delay)
            curve={x:summarize_delays(values,metric)['value'] for x,values in sorted(bins.items())}
            on=onset_summary(curve,threshold=f.delay_positive_threshold)
            speed=_curve_for_slice(group.speed_runs,f,sv,'saturation_epoch')
            gen=_curve_for_slice(group.groks_runs,f,sv,'grokking_epoch')
            point=find_intersection(speed,gen);prediction=point[0] if point else None
            k,capacity=_capacity_baseline(group,f,sv)
            row={'config':view.config_name,'figure':f.name,'group':gi,'slice_field':f.slice_field,'slice_value':sv,
                 'x_field':f.x_field,'predicted_onset_x':prediction,'empirical_onset_x':on['value'],
                 'onset_status':on['status'],'log_ratio':math.log10(on['value']/prediction) if on['value'] and prediction else None,
                 'k_mem_bits':k,'capacity_baseline_x':capacity,'n_delay_cells':len(bins),
                 'n_unresolved_aggregates':sum(v is None for v in curve.values()),
                 'n_unresolved_seeds':sum(v is None for vv in bins.values() for v in vv)}
            for ax in view.swept_axes:row[ax]=getattr(group.key,ax,None)
            output.append(row)
            for x,values in bins.items():summary.append({'group':gi,'slice_value':sv,'x':x,**summarize_delays(values,metric)})
    return pd.DataFrame(output),summary


def statistical():
    global EVENTS
    EVENTS={r['uuid']:{k:r[k] for k in ['train99','val98','followup_epochs']} for r in frozen()}
    cache=OUT/'statistics/events.json'
    if cache.exists():EVENTS.update(json.loads(cache.read_text()))
    overview_path=OUT/'statistics/overview.csv'
    overview=pd.read_csv(overview_path).to_dict('records') if overview_path.exists() else []
    published_path=OUT/'statistics/archived_comparison.json'
    published_comparisons=json.loads(published_path.read_text()) if published_path.exists() else []
    configs=['central','init_scale_sweep','weight_decay_sweep','lr_sweep','alpha_sweep','depth_scaling','task_add','task_mul']
    for config in configs:
        path=ROOT/'configs'/f'{config}.yaml'
        if not path.exists():continue
        completed=OUT/f'statistics/{config}/intersection/median_all/hypothesis_tests.json'
        if config not in ['lr_sweep']:
            completed=OUT/f'statistics/{config}/intersection_by_prime/median_all/hypothesis_tests.json'
        if completed.exists():
            print('STATS reuse completed',config,flush=True)
            continue
        print('STATS loading',config,flush=True)
        # ConfigView uses expanded runs only to enumerate architecture keys.
        # Width matching would instantiate thousands of models for no query benefit.
        import copy
        import grokking_capacity.dispatch.config as cv
        original_expand=cv.expand_runs
        def architecture_expansion(spec):
            spec=copy.deepcopy(spec)
            for entry in spec.get('experiments',{}).values():
                if entry.get('match_by')=='param_count':
                    entry.pop('match_by');entry.pop('param_count_targets',None);entry['dims']=[20]
            return original_expand(spec)
        try:
            cv.expand_runs=architecture_expansion
            view=ConfigView.from_yaml(path,db_path=str(ROOT/'runs.db'))
        finally:
            cv.expand_runs=original_expand
        for f in view.intersection_figures:
            # Keep both axes; main paper comparisons use the first.
            for variant,metric,legacy,canonical,tmetric in [
                ('legacy_min','min',True,False,'mean'),
                ('min','min',False,False,'mean'),('median','median',False,False,'mean'),('mean','mean',False,False,'mean'),
                ('median_all','median',False,False,'median'),
                *([(m+'_canonical',m,False,True,'mean') for m in METRICS] if config=='central' and f.slice_field=='p' else [])]:
                folder=f'statistics/{config}/{f.name}/{variant}'
                df,cells=onset_rows(view,f,metric,legacy=legacy,canonical=canonical,time_metric=tmetric)
                (OUT/folder).mkdir(parents=True,exist_ok=True);df.to_csv(OUT/folder/'predictiveness.csv',index=False)
                csvsave(folder+'/cell_aggregates.csv',cells)
                results=compute_hypothesis_tests(view,f,df)
                save(folder+'/hypothesis_tests.json',results)
                valid=df.dropna(subset=['log_ratio']) if 'log_ratio' in df else df
                row={'config':config,'figure':f.name,'variant':variant,'delay_metric':metric,'time_metric':tmetric,
                    'cells':len(df),'n_valid':results['n_valid'],
                    'onsets_unresolved':int(sum(df.onset_status=='unresolved')) if 'onset_status' in df else 0,
                    'rho':results.get('predictiveness',{}).get('spearman',{}).get('rho'),
                    'rho_p':results.get('predictiveness',{}).get('spearman',{}).get('p_permutation'),
                    'ccc':results.get('calibration',{}).get('ccc',{}).get('value'),
                    'joint_F_p':results.get('calibration',{}).get('joint_ftest',{}).get('p'),
                    'median_offset_dex':float(valid.log_ratio.median()) if len(valid) else None,
                    'MAE_dex':float(valid.log_ratio.abs().mean()) if len(valid) else None,
                    'sd_offset_dex':float(valid.log_ratio.std()) if len(valid)>1 else None}
                overview.append(row);print('STATS',json.dumps(clean(row)),flush=True)
                if variant=='legacy_min':
                    archived=ROOT/'figures'/config/f.name/'predictiveness.csv'
                    if archived.exists():
                        old=pd.read_csv(archived)
                        published_comparisons.append({'config':config,'figure':f.name,'archived_path':str(archived.relative_to(ROOT)),
                            'archived_rows':len(old),'current_registry_rows':len(df),
                            'archived_median_offset':float(old.log_ratio.median()) if 'log_ratio' in old else None,
                            'current_registry_median_offset':row['median_offset_dex']})
            save('statistics/events.json',EVENTS)
            csvsave('statistics/overview.csv',overview)
    save('statistics/archived_comparison.json',published_comparisons)


def laws():
    data=pd.read_csv(ROOT/'results/law_complete_ten_seed_sensitivity_20261106/selected_runs.csv')
    original=json.loads((ROOT/'results/law_complete_ten_seed_sensitivity_20261106/comparison.json').read_text())
    fits={};points={};C=2.16
    for cohort in ['observed','complete']:
        for metric in ['mean','median','min','geometric_mean']:
            name=f'{cohort}_{metric}';points[name]={}
            for kind in ['mem','gen']:
                chosen=[]
                for (p,P),g in data[data.kind==kind].groupby(['p','P']):
                    if cohort=='complete' and not g.included_complete_ten_fit.all():continue
                    obs=g[g.observed]
                    if not len(obs):continue
                    chosen.append({'p':int(p),'P':int(P),'f':float(g.f_original.iloc[0]),'t':reduce_values(obs.event_epoch,metric)})
                points[name][kind]=chosen
            mem=fit('mem',points[name]['mem']);gen=fit('gen',points[name]['gen'])
            fits[name]={'cohort':cohort,'metric':metric,'mem':mem,'gen':gen,'fold':fold(mem,gen,C),
                'p149':crossing(149,mem,gen,C),'p197':crossing(197,mem,gen,C)}
    # Baseline refits preserve the original asymmetric estimators exactly.
    baselines={k:original['comparisons'][k] for k in ['current','complete_both']}
    # Paired prime-cluster bootstrap, fixed eligibility, same draws for all metrics.
    primes=sorted(data.p.unique());rng=np.random.default_rng(20261106);draws=[]
    for rep in range(2000):
        sample=rng.choice(primes,size=len(primes),replace=True)
        row={'replicate':rep}
        for name,pp in points.items():
            try:row[name]=fold(fit('mem',pp['mem'],sample),fit('gen',pp['gen'],sample),C)
            except (ValueError,AssertionError,np.linalg.LinAlgError):row[name]=None
        draws.append(row)
    for name,r in fits.items():
        values=[d[name] for d in draws if d[name] is not None]
        r['fold_prime_bootstrap_95']=np.quantile(values,[.025,.975]).tolist() if values else None
        r['bootstrap_finite']=len(values)
    csvsave('laws/prime_bootstrap.csv',draws)
    save('laws/comparison.json',{'baseline_asymmetric_estimators':baselines,'unified_cell_reducers':fits,
        'interpretation':'Changing delay aggregation alone leaves timescale fits and fold unchanged. These additional fits change the timescale estimand and give each observed cell equal weight. Complete-cell selection remains conditional on all ten threshold crossings before cap.'})
    print('LAWS',json.dumps(clean(fits)),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['cone','statistics','laws','all']);args=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for name,fn in [('cone',cone),('statistics',statistical),('laws',laws)]:
        if args.stage in [name,'all']:fn()

if __name__=='__main__':main()
