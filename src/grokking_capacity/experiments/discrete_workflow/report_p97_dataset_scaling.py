#!/usr/bin/env python3
"""Paired mean curves, censoring bounds, and bootstrap crossing diagnostics."""
from grokking_capacity.experiments.registered import verify_sources as verify_registered_sources

import csv
import json
import math
from pathlib import Path
import shutil
import sys
import numpy as np
import yaml
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'src'))
from grokking_capacity.experiments.dataset_scaling import RUN, read, sha, atomic_json, source_hashes
OUT = ROOT/'reports/p97_dataset_scaling'


def csv_write(path,rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def summarize(values,limit):
    observed=[v for v in values if v is not None]
    capped=[limit if v is None else v for v in values]
    return dict(n=len(values),observed=len(observed),censored=len(values)-len(observed),
        mean=float(np.mean(capped)) if len(observed)==len(values) else None,
        capped_mean=float(np.mean(capped)),mean_lower_bound=float(np.mean(capped)),
        median=float(np.median(capped)) if len(observed)>len(values)//2 else None,
        median_lower_bound=float(np.median(capped)),min_observed=min(observed) if observed else None)


def crossing_brackets(sizes,differences,complete):
    result=[]
    for i in range(1,len(sizes)):
        a,b=differences[i-1:i+1]
        if complete[i-1] and complete[i] and a*b<0:
            fraction=-a/(b-a)
            result.append(dict(lower_n=int(sizes[i-1]),upper_n=int(sizes[i]),
                log_interpolated_n=float(np.exp(np.log(sizes[i-1])+fraction*np.log(sizes[i]/sizes[i-1]))),
                direction='C faster to B faster' if a>0 else 'B faster to C faster'))
    return result


def analyze(rows,sizes,seeds,limit):
    matrix={role:np.array([[next(r['t_gen_updates'] if r['t_gen_updates'] is not None else limit
        for r in rows if r['role']==role and r['seed']==seed and r['n_train']==n)
        for n in sizes] for seed in seeds],float) for role in ['C','B']}
    rng=np.random.default_rng(9002)
    draws=rng.integers(0,len(seeds),size=(10000,len(seeds)))
    diff=matrix['B']-matrix['C']
    estimate=diff.mean(axis=0)
    bootstrap=diff[draws].mean(axis=1)
    interval=np.percentile(bootstrap,[2.5,97.5],axis=0).T
    radius=float(np.percentile(np.max(abs(bootstrap-estimate),axis=1),95))
    boots={r:matrix[r][draws].mean(axis=1) for r in ['C','B']}
    ratio=matrix['B'].mean(axis=0)/matrix['C'].mean(axis=0)
    ratio_ci=np.percentile(boots['B']/boots['C'],[2.5,97.5],axis=0).T
    complete=[];table=[]
    for i,n in enumerate(sizes):
        row=dict(n_train=n,dataset_bits=float(n*math.log2(99)))
        for role in ['C','B']:
            rr=[r for r in rows if r['n_train']==n and r['role']==role]
            summary=summarize([r['t_gen_updates'] for r in rr],limit)
            row.update({role+'_'+k:v for k,v in summary.items()})
            row[role+'_final_learning']=sum(r['final_learning'] for r in rr)
            row[role+'_joint_stop']=sum(r['joint_stop'] for r in rr)
            row[role+'_positive_paper_delay']=sum(r['paper_delay_updates'] is not None and r['paper_delay_updates']>0 for r in rr)
            row[role+'_undefined_paper_delay']=sum(r['paper_delay_updates'] is None for r in rr)
            delays=[r['paper_delay_updates'] for r in rr]
            row[role+'_min_paper_delay']=min(delays) if all(v is not None for v in delays) else None
            row[role+'_capped_mean_ci_low'],row[role+'_capped_mean_ci_high']=np.percentile(boots[role][:,i],[2.5,97.5]).tolist()
        complete.append(row['C_censored']==row['B_censored']==0)
        row.update(mean_difference=float(estimate[i]) if complete[-1] else None,
            capped_mean_difference=float(estimate[i]),difference_ci_low=float(interval[i,0]),difference_ci_high=float(interval[i,1]),
            simultaneous_low=float(estimate[i]-radius),simultaneous_high=float(estimate[i]+radius),
            mean_ratio=float(ratio[i]) if complete[-1] else None,capped_mean_ratio=float(ratio[i]),
            ratio_ci_low=float(ratio_ci[i,0]),ratio_ci_high=float(ratio_ci[i,1]))
        table.append(row)
    observed=crossing_brackets(sizes,estimate,complete)
    capped=crossing_brackets(sizes,estimate,[True]*len(sizes))
    positive=[sizes[i] for i in range(len(sizes)) if complete[i] and estimate[i]-radius>0]
    negative=[sizes[i] for i in range(len(sizes)) if complete[i] and estimate[i]+radius<0]
    return dict(sizes=table,observed_mean_crossing_brackets=observed,capped_mean_crossing_brackets=capped,
        resolved_crossing=bool(positive and negative),simultaneously_C_faster_sizes=positive,
        simultaneously_B_faster_sizes=negative,simultaneous_band_radius=radius,
        confidence='Paired ten-seed bootstrap, 10000 draws, seed9002. Pointwise95% intervals; centered maximum absolute deviation across ten capped contrasts gives simultaneous95% bands. Conditional on fixed architectures and calibrated constants.',
        interpretation='Capped curves are not uncensored mean generalisation times where failures occur. Crossing interpolation only describes adjacent fully observed points, not a fitted physical threshold.')


def plot(summary,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table=summary['sizes'];sizes=np.array([r['n_train'] for r in table])
    colors={'C':'#2467ae','B':'#cb6131'}
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for role in ['C','B']:
        y=np.array([r[role+'_capped_mean'] for r in table])
        complete=np.array([r[role+'_censored']==0 for r in table])
        axes[0,0].plot(sizes,y,color=colors[role],alpha=.6,label=role+' capped mean')
        axes[0,0].scatter(sizes[complete],y[complete],color=colors[role],marker='o')
        axes[0,0].scatter(sizes[~complete],y[~complete],color=colors[role],marker='^',s=55)
        low=[r[role+'_capped_mean_ci_low'] for r in table];high=[r[role+'_capped_mean_ci_high'] for r in table]
        axes[0,0].fill_between(sizes,low,high,color=colors[role],alpha=.12)
        axes[1,0].plot(sizes,[r[role+'_observed'] for r in table],'-o',color=colors[role],label=role+' first99')
        axes[1,0].plot(sizes,[r[role+'_final_learning'] for r in table],'--',color=colors[role],label=role+' final learning')
        for seed in summary['seeds']:
            rr=sorted([r for r in rows if r['role']==role and r['seed']==seed],key=lambda r:r['n_train'])
            ys=[r['t_gen_updates'] if r['t_gen_updates'] is not None else summary['limit'] for r in rr]
            axes[1,1].plot(sizes,ys,color=colors[role],alpha=.2,linewidth=.8)
        axes[1,1].plot(sizes,y,color=colors[role],linewidth=2,label=role+' capped mean')
    ax=axes[0,1]
    delta=np.array([r['capped_mean_difference'] for r in table])
    ax.axhline(0,color='black',linewidth=1)
    ax.fill_between(sizes,[r['simultaneous_low'] for r in table],[r['simultaneous_high'] for r in table],color='gray',alpha=.15,label='Simultaneous 95%')
    ax.fill_between(sizes,[r['difference_ci_low'] for r in table],[r['difference_ci_high'] for r in table],color='gray',alpha=.3,label='Pointwise 95%')
    ax.plot(sizes,delta,'o-',color='#62449b',label='Mean B − C (capped)')
    for row in table:
        if row['mean_difference'] is None:
            ax.scatter(row['n_train'],row['capped_mean_difference'],marker='s',facecolors='white',edgecolors='#62449b',zorder=4)
    titles=['Mean time; triangles = lower bounds, shading = capped-time 95% CI',
        'Positive means C is faster; squares mark censoring',
        'Successes out of ten seeds', 'Every seed; failures plotted at 50,000-update ceiling']
    for ax,title in zip(axes.flat,titles):
        ax.set_xscale('log');ax.set_xlabel('Training examples (p=97)');ax.set_title(title,fontsize=10)
        ax.set_xticks(sizes,labels=[str(n) for n in sizes],rotation=45,fontsize=8)
        ax.grid(alpha=.15);ax.legend(fontsize=8)
    axes[0,0].set_yscale('log');axes[1,1].set_yscale('log')
    axes[0,0].set_ylabel('Optimizer updates to first held-out 99%')
    axes[1,1].set_ylabel('Optimizer updates')
    axes[0,1].set_ylabel('B minus C, optimizer updates')
    axes[1,0].set_ylim(-.3,10.5)
    fig.suptitle('Dataset-matched predicted capacity: continuous C vs categorical B; dropout 0.2')
    for ext in ['png','pdf']:
        fig.savefig(OUT/('dataset_scaling.'+ext),dpi=180)
    plt.close(fig)


def main():
    assert read(RUN/'campaign_complete.json')['all_passed']
    verify_registered_sources(read(RUN/'manifest.json')['source_hashes'])
    assert not (RUN/'campaign_error.json').exists()
    reg=yaml.safe_load((RUN/'registration.yaml').read_text())
    manifest=read(RUN/'manifest.json')
    assert manifest['registry_sha256']==sha(RUN/'job_registry.json')
    assert manifest['selection_sha256']==sha(RUN/'selection.json')
    assert manifest['registration_sha256']==sha(RUN/'registration.yaml')
    selection=read(RUN/'selection.json')
    assert sha(selection['source'])==selection['source_sha256']
    for path,digest in selection['calibration_evidence'].items():
        assert sha(path)==digest
    rows=[];hashes={};audits={}
    registry=read(RUN/'job_registry.json')
    for name,c in registry.items():
        path=RUN/'jobs'/name;s=read(path/'summary.json');m=read(path/'manifest.json')
        audit=read(RUN/'audits'/(name+'.json'))
        assert audit['all_passed'] and c==s['config']==m['config']
        for f,h in audit['input_hashes'].items():
            assert sha(path/f)==h
            hashes[str(path/f)]=h
        audits[name]=audit
        # Independent first crossing from raw trace, without calling worker metrics.
        trace=[json.loads(line) for line in (path/'trace.jsonl').read_text().splitlines()]
        first=next((r for r in trace if r['val_acc']>=.99),None)
        assert (first['step'] if first else None)==s['metrics']['val_acc_99_updates']
        rows.append(dict(name=name,role=c['role'],dim=c['dim'],seed=c['seed'],n_train=c['n_train'],
            n_test=m['n_test'],parameters=s['param_count'],estimated_capacity_bits=selection['pairs'][c['pair_index']][c['role']+'_estimated_bits'],
            capacity_dataset_mismatch=selection['pairs'][c['pair_index']][c['role']+'_dataset_mismatch'],
            calibration_extrapolation=selection['pairs'][c['pair_index']][c['role']+'_calibration_extrapolation'],
            t_gen_updates=first['step'] if first else None,t_gen_examples=first['examples'] if first else None,
            t_gen_epochs=first['epoch_fraction'] if first else None,censored=first is None,
            sustained_updates=s['metrics']['val_99_sustained_updates'],final_learning=s['learning_gate_passed'],
            joint_stop=s['joint_stop_passed'],final_train_acc=s['final']['train_eval_acc'],final_test_acc=s['final']['val_acc'],
            paper_delay_updates=s['metrics']['paper_delay_updates'],eval_delay_updates=s['metrics']['eval_delay_updates'],
            final_updates=s['final']['step'],wall_seconds=s['final']['wall_seconds'],test_sha256=m['test_sha256'],
            train_sha256=m['train_sha256'],initial_sha256=m['initial_weights_sha256']))
    assert len(rows)==200
    for seed in reg['confirmation']['seeds']:
        rr=[r for r in rows if r['seed']==seed]
        assert len({r['test_sha256'] for r in rr})==1
        for n in reg['dataset']['sizes']:
            assert len({r['train_sha256'] for r in rr if r['n_train']==n})==1
    summary=analyze(rows,reg['dataset']['sizes'],reg['confirmation']['seeds'],reg['common']['max_updates'])
    summary.update(seeds=reg['confirmation']['seeds'],limit=reg['common']['max_updates'],pairs=selection['pairs'],
        all_audits_passed=True,run_count=200,hashes_verified=len(hashes))
    for row,pair in zip(summary['sizes'],selection['pairs']):
        assert row['n_train']==pair['n_train']
        for role in ['C','B']:
            row[role+'_width']=pair[role]['dim']
            for key in ['parameters','estimated_bits','dataset_mismatch','calibration_extrapolation']:
                row[role+'_'+key]=pair[role+'_'+key]
        row['predicted_relative_mismatch']=pair['predicted_relative_mismatch']
    OUT.mkdir(parents=True,exist_ok=True)
    atomic_json(OUT/'summary.json',summary);atomic_json(OUT/'input_hashes.json',hashes)
    atomic_json(OUT/'checkpoint_audits.json',audits)
    csv_write(OUT/'runs.csv',rows);csv_write(OUT/'size_summary.csv',summary['sizes'])
    for name in ['manifest.json','selection.json','registration.yaml','campaign_complete.json','job_registry.json']:
        shutil.copyfile(RUN/name,OUT/name)
    plot(summary,rows)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['sizes','pairs']},indent=2))


if __name__=='__main__':
    main()
