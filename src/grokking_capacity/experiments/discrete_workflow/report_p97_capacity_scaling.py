#!/usr/bin/env python3
"""Audit and report the capacity-constant matched C/B size sweep."""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np
import yaml

from grokking_capacity.experiments.registered import verify_sources as verify_registered_sources
from grokking_capacity.analysis.paired import summarize
from grokking_capacity.utils.artifacts import write_csv as csv_write
from grokking_capacity.experiments.capacity_scaling import RUN,read,sha,source_hashes,fit_constant,match_pairs,parameter_count
from grokking_capacity.experiments.categorical_groks import independent_metrics
from grokking_capacity.experiments.multilevel_dropout import atomic_json,late_drift


ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'reports/p97_capacity_scaling'
LIMIT=50000


def time_summary(values,limit=LIMIT):
    result = summarize(values, limit)
    result['mean_capped'] = result.pop('capped_mean')
    result['max_observed'] = max((v for v in values if v is not None), default=None)
    return result


def delay_summary(values):
    observed=[v for v in values if v is not None]
    return dict(min_observed=min(observed) if observed else None,positive=sum(v>0 for v in observed),zero=observed.count(0),
        undefined=len(values)-len(observed),mean_all=statistics.mean(observed) if len(observed)==len(values) else None)


def slopes(rows,indices,seeds,capped=False):
    if len(indices)<3:return dict(available=False,reason='Fewer than three common size points',pair_indices=indices)
    result={};vectors={}
    for role in ['C','B']:
        values=[]
        for seed in seeds:
            group=sorted([r for r in rows if r['role']==role and r['seed']==seed and r['pair_index'] in indices],key=lambda r:r['pair_index'])
            assert len(group)==len(indices)
            x=np.log([r['estimated_capacity_bits'] for r in group]);times=[r['t_gen_updates'] for r in group]
            if not capped:assert all(v is not None for v in times)
            y=np.log([LIMIT if v is None else v for v in times]);values.append(float(np.polyfit(x,y,1)[0]))
        vectors[role]=np.array(values)
    rng=np.random.default_rng(9001);draw=rng.integers(0,len(seeds),size=(10000,len(seeds)))
    for role in ['C','B']:
        boot=vectors[role][draw].mean(axis=1)
        result[role]=dict(mean_slope=float(vectors[role].mean()),seed_slopes=vectors[role].tolist(),
            bootstrap_95_interval=np.percentile(boot,[2.5,97.5]).tolist())
    diff=vectors['B']-vectors['C'];boot=diff[draw].mean(axis=1)
    return dict(available=True,pair_indices=indices,roles=result,B_minus_C=dict(mean_slope_difference=float(diff.mean()),
        seed_differences=diff.tolist(),bootstrap_95_interval=np.percentile(boot,[2.5,97.5]).tolist()),
        interpretation='Per-seed log(t) on log(estimated capacity), with intercept. Negative slope means faster generalisation as capacity increases.',
        scope='All-size capped-time sensitivity' if capped else 'Conditional on common sizes with all twenty observed times',
        seed_bootstrap_resamples=10000,bootstrap_seed=9001)


def mean_time_slopes(rows,indices,seeds,capped=False):
    """Fit the arithmetic-mean t_gen curve; resample paired seeds before averaging."""
    if len(indices)<3:return dict(available=False,reason='Fewer than three common size points',pair_indices=indices)
    rng=np.random.default_rng(9001);draw=rng.integers(0,len(seeds),size=(10000,len(seeds)))
    roles={};boots={}
    for role in ['C','B']:
        matrix=[];x=None
        for seed in seeds:
            rr=sorted([r for r in rows if r['role']==role and r['seed']==seed and r['pair_index'] in indices],key=lambda r:r['pair_index'])
            assert len(rr)==len(indices)
            values=[r['t_gen_updates'] for r in rr]
            if not capped:assert all(v is not None for v in values)
            matrix.append([LIMIT if v is None else v for v in values])
            x=np.log([r['estimated_capacity_bits'] for r in rr])
        matrix=np.array(matrix,float);center=x-x.mean();denom=center@center
        estimate=float(np.log(matrix.mean(axis=0))@center/denom)
        boot=np.log(matrix[draw].mean(axis=1))@center/denom;boots[role]=boot
        roles[role]=dict(slope=estimate,bootstrap_95_interval=np.percentile(boot,[2.5,97.5]).tolist())
    difference=boots['B']-boots['C']
    return dict(available=True,pair_indices=indices,roles=roles,
        B_minus_C=dict(slope_difference=roles['B']['slope']-roles['C']['slope'],
            bootstrap_95_interval=np.percentile(difference,[2.5,97.5]).tolist()),
        aggregation='Log of arithmetic mean across all ten seeds at each size; paired seed resampling before computing each mean.',
        scope='All-size capped-time sensitivity' if capped else 'Common sizes with all twenty observed times',
        seed_bootstrap_resamples=10000,bootstrap_seed=9001)


def paired_outcomes(rows,pairs,seeds):
    result=[]
    for pair in pairs:
        for seed in seeds:
            rr={role:next(r for r in rows if r['pair_index']==pair['index'] and r['seed']==seed and r['role']==role) for role in ['C','B']}
            c,b=rr['C']['t_gen_updates'],rr['B']['t_gen_updates']
            winner=('B' if b<c else 'C' if c<b else 'tie') if c is not None and b is not None else ('B' if b is not None else 'C' if c is not None else 'both_censored')
            result.append(dict(pair_index=pair['index'],seed=seed,C_t_gen=c,B_t_gen=b,winner=winner,
                B_over_C=b/c if b is not None and c is not None else None,
                B_over_C_lower_bound=LIMIT/c if b is None and c is not None else None,
                B_over_C_upper_bound=b/LIMIT if b is not None and c is None else None))
    return result


def main():
    assert (RUN/'campaign_complete.json').exists(),'Only publish final report after the campaign and audits finish.'
    manifest=read(RUN/'manifest.json');verify_registered_sources(manifest['source_hashes'])
    assert not (RUN/'campaign_error.json').exists()
    recovery={p.stem:read(p) for p in sorted((RUN/'audit_recovery').glob('*.json'))}
    assert recovery['audits_completed']['all_passed']
    assert recovery['manifest']['training_jobs_restarted']==0
    assert recovery['manifest']['recovery_source_sha256']==sha(Path(__file__).with_name('frozen')/'resume_p97_capacity_scaling_audit_20261122.py')
    selection=read(RUN/'selection.json');reg=yaml.safe_load((RUN/'registration.yaml').read_text())
    assert manifest['registration_sha256']==selection['registration_sha256']==sha(RUN/'registration.yaml')
    for path,h in selection['evidence_sha256'].items():assert sha(path)==h
    fits=read(RUN/'capacity_constants.json')
    for role in ['C','B']:assert fits[role]==fit_constant(fits[role]['points'])
    search=reg['matching']['B_width_candidates'];pairs=match_pairs({r:f['bits_per_parameter'] for r,f in fits.items()},reg['matching']['C_widths'],range(search['start'],search['stop']+1,search['step']))
    assert pairs==selection['pairs'] and len(pairs)==10
    registry=read(RUN/'job_registry.json');jobs={};stages={}
    for p in (RUN/'stages').glob('*/controller.json'):
        s=read(p);assert s['phase']=='complete' and not s['errors'] and not s['running'] and not s['pending']
        assert all(j['exit_code']==0 for j in s['completed'])
        assert {j['name'] for j in s['completed']}=={j['name'] for j in s['jobs']}
        assert s['jobs']==read(p.parent/'jobs.json')
        for job in s['jobs']:assert registry[job['name']]==job['config'];jobs[job['name']]=job['config']
        stages[p.parent.name]=dict(jobs=len(s['jobs']),started_at=s['started_at'],finished_at=s['finished_at'],sha256=sha(p))
    assert jobs==registry
    order=sorted(stages,key=lambda k:stages[k]['started_at'])
    assert all(stages[a]['finished_at']<stages[b]['started_at'] for a,b in zip(order,order[1:]))
    assert order[-1]=='confirmation' and stages[order[-2]]['finished_at']<selection['frozen_at']<stages['confirmation']['started_at']
    calibration_roles={path:key.split('_')[0] for key,paths in selection['calibration_paths'].items() for path in paths}
    calibration_winners={p['path'] for fit in fits.values() for p in fit['points']}
    inputs={};audits=[];summaries={};manifests={};traces={};arithmetic=[];capacity=[]
    for p in sorted((RUN/'audits').glob('*.json')):
        a=read(p);assert a['all_passed'] and a['initial_verified'] and a['data_verified'];path=Path(a['path'])
        for name,h in a['input_hashes'].items():assert sha(path/name)==h;inputs[str(path/name)]=h
        s=read(path/'summary.json');m=read(path/'manifest.json');c=s['config'];assert c==m['config']
        for key,value in reg['common'].items():
            if c['task']=='capacity' or key not in ['patience','min_delta']:assert c[key]==value
        verify_registered_sources(m['source_hashes'])
        summaries[str(path)]=s;manifests[str(path)]=m;audits.append(a)
        if path.parent==RUN/'jobs':assert c==registry[path.name] and m['source_hashes']==manifest['source_hashes']
        row=dict(name=path.name,path=str(path),role=c['role'],phase=c['phase'],seed=c['seed'],model=c['model'],dim=c['dim'],mlp_dim=c['mlp_dim'],parameters=s['param_count'],epochs=s['epochs'])
        if c['task']=='arithmetic':
            t=[json.loads(line) for line in (path/'trace.jsonl').read_text().splitlines()];assert t[-1]==s['final'] and len(t)==s['epochs']
            metrics=independent_metrics(t);assert metrics==s.get('metrics',metrics)
            row.update(learning_success=s['learning_gate_passed'],
                joint99_at_final=bool(len(t)>=3 and t[-1]['train_online_acc']>=.99 and all(x['val_acc']>=.99 for x in t[-3:])),final_val=t[-1]['val_acc'],final_train_eval=t[-1]['train_eval_acc'],final_train_online=t[-1]['train_online_acc'],**metrics)
            for out,key in [('t_gen_updates','val_acc_99_epoch'),('sustained99_updates','val_99_sustained_epoch'),('paper_delay_updates','paper_delay_epochs'),('eval_delay_updates','eval_delay_epochs')]:
                row[out]=10*metrics[key] if metrics[key] is not None else None
            if c['phase']=='confirmation':
                assert m['started_at']>selection['frozen_at'] and c['selection_sha256']==sha(RUN/'selection.json')
                pair=pairs[c['pair_index']];role=c['role'];assert all(c[k]==v for k,v in pair[role].items())
                assert s['param_count']==pair[role+'_parameters']==parameter_count(role,c['dim'])
                row.update(pair_index=c['pair_index'],estimated_capacity_bits=pair[role+'_estimated_bits'],censored=row['t_gen_updates'] is None)
                if row['censored']:assert t[-1]['step']==LIMIT
            arithmetic.append(row);traces[str(path)]=t
        else:
            arr=np.load(path/'trace.npz',allow_pickle=False);assert len(arr['bits_trace'])==s['epochs']
            row.update(original_role=c['role'],role=calibration_roles[str(path)],n_samples=c['n_samples'],capacity_bits=s['final_bits'],final_eval_acc=s['final_eval']['acc'],late_drift=late_drift(arr['bits_trace']),
                at_epoch_limit=s['epochs']>=c['epochs'],entropy_fraction=s['final_bits']/(c['n_samples']*np.log2(99)),selected_maximum=str(path) in calibration_winners)
            capacity.append(row)
    done=read(RUN/'campaign_complete.json')
    assert len(list((RUN/'jobs').glob('*/summary.json')))==len(registry)==done['new_jobs']
    assert len(audits)==done['new_jobs']+done['inherited_jobs']
    confirm=[r for r in arithmetic if r['phase']=='confirmation'];assert len(confirm)==done['confirmation_runs']==200
    seeds=reg['confirmation']['seeds'];groups=[]
    for seed in seeds:
        rr=[r for r in confirm if r['seed']==seed];assert len(rr)==20
        assert len({manifests[r['path']]['dataset_sha256'] for r in rr})==1
    for pair in pairs:
        for role in ['C','B']:
            rows=[r for r in confirm if r['pair_index']==pair['index'] and r['role']==role]
            assert sorted(r['seed'] for r in rows)==seeds
            groups.append(dict(pair_index=pair['index'],role=role,dim=pair[role]['dim'],parameters=pair[role+'_parameters'],
                estimated_capacity_bits=pair[role+'_estimated_bits'],estimated_capacity_to_dataset=pair[role+'_estimated_bits']/reg['dataset']['target_bits'],
                extrapolated=not fits[role]['parameter_range'][0]<=pair[role+'_parameters']<=fits[role]['parameter_range'][1],
                t_gen=time_summary([r['t_gen_updates'] for r in rows]),sustained99=time_summary([r['sustained99_updates'] for r in rows]),
                learning_successes=sum(r['learning_success'] for r in rows),
                joint_stopping_successes=sum(r['joint99_at_final'] for r in rows),paper_delay=delay_summary([r['paper_delay_updates'] for r in rows]),
                eval_delay=delay_summary([r['eval_delay_updates'] for r in rows])))
    complete=[p['index'] for p in pairs if all(g['t_gen']['observed']==10 for g in groups if g['pair_index']==p['index'])]
    paired=paired_outcomes(confirm,pairs,seeds);comparisons=[];rng=np.random.default_rng(9001)
    for pair in pairs:
        rows=[r for r in paired if r['pair_index']==pair['index']];ratio=[r['B_over_C'] for r in rows]
        gmean=ci=None
        if all(v is not None for v in ratio):
            logs=np.log(ratio);gmean=float(np.exp(logs.mean()));draw=rng.integers(0,10,size=(10000,10))
            ci=np.exp(np.percentile(logs[draw].mean(axis=1),[2.5,97.5])).tolist()
        comparisons.append(dict(pair_index=pair['index'],B_wins=sum(r['winner']=='B' for r in rows),C_wins=sum(r['winner']=='C' for r in rows),
            ties=sum(r['winner']=='tie' for r in rows),both_censored=sum(r['winner']=='both_censored' for r in rows),
            geometric_mean_B_over_C=gmean,bootstrap_ratio_95_interval=ci))
    constant_sensitivity={str(seed):[dict(pair_index=p['index'],
        predicted_relative_mismatch=fits['B']['per_seed_constants'][str(seed)]*p['B_parameters']/
            (fits['C']['per_seed_constants'][str(seed)]*p['C_parameters'])-1) for p in pairs] for seed in [61,62]}
    summary=dict(complete=True,new_jobs=done['new_jobs'],audited_checkpoints=len(audits),calibration_cells=len(capacity),
        audit_recovery=recovery,constants=fits,constant_seed_sensitivity=constant_sensitivity,pairs=pairs,groups=groups,paired_comparisons=comparisons,
        complete_range_scaling=slopes(confirm,complete,seeds),capped_time_scaling=slopes(confirm,list(range(10)),seeds,capped=True),
        mean_t_gen_scaling=mean_time_slopes(confirm,complete,seeds),
        mean_capped_t_gen_scaling=mean_time_slopes(confirm,list(range(10)),seeds,capped=True),
        notes=['Capacities are estimated as family-specific constant times actual parameter count; only calibration sizes have direct empirical measurements.',
            'One constant rescales the horizontal axis; within each family the log-log slope versus estimated capacity equals the slope versus parameter count.',
            'All models use FF4d; C is fully continuous and B uses normalized deterministic eight-symbol categorical interfaces.',
            'Every role and size shares the same seed-specific dataset, but different tensor shapes consume initialization RNG differently.',
            'Missing first held-out99 crossings are right-censored at50000 updates. No success-only average is presented as an all-seed mean.',
            'Complete-range slopes condition on sizes with all twenty events; capped-time slopes are a separate descriptive sensitivity analysis.',
            'Both registered per-seed log slopes (geometric-mean trend) and additional arithmetic-mean t_gen curve slopes are reported; these can differ with size-dependent tails.',
            'Fit residuals and parameter-range extrapolation limit the accuracy of capacity matching.',
            'Bootstrap intervals resample the ten arithmetic seeds and condition on the fitted constants and frozen architectures; calibration uncertainty is reported separately.',
            'The capacity constants are empirical measurements under this training protocol, not intrinsic information-theoretic bounds.',
            'An observed first held-out99 crossing is not the same as sustained learning or passing the final learning check; both outcomes are retained.'])
    OUT.mkdir(parents=True,exist_ok=True)
    csv_write(OUT/'confirmation_runs.csv',confirm);csv_write(OUT/'arithmetic_runs.csv',arithmetic);csv_write(OUT/'capacity_cells.csv',capacity)
    csv_write(OUT/'paired_outcomes.csv',paired)
    flat=[]
    for g in groups:
        flat.append({**{k:v for k,v in g.items() if k not in ['t_gen','sustained99','paper_delay','eval_delay']},
            **{'t_gen_'+k:v for k,v in g['t_gen'].items()},**{'paper_delay_'+k:v for k,v in g['paper_delay'].items()}})
    csv_write(OUT/'size_summary.csv',flat)
    atomic_json(OUT/'audit_recovery.json',recovery)
    atomic_json(OUT/'summary.json',summary);atomic_json(OUT/'checkpoint_audits.json',audits);atomic_json(OUT/'input_hashes.json',inputs)
    atomic_json(OUT/'validation.json',dict(all_passed=True,checkpoints=len(audits),confirmation_runs=200,paired_datasets=10,
        calibration_cells=len(capacity),arithmetic_traces_recomputed=len(arithmetic),input_artifact_hashes=len(inputs),
        stages=stages,stage_order=order,selection_frozen_before_confirmation=True,analysis_sha256=sha(Path(__file__))))
    for name in ['registration.yaml','selection.json','capacity_constants.json','calibration_diagnostics.json','manifest.json','campaign_complete.json','inherited.json','analysis_plan_extension.json']:
        (OUT/name).write_bytes((RUN/name).read_bytes())
    plots(summary,confirm)
    print(json.dumps(dict(constants={r:f['bits_per_parameter'] for r,f in fits.items()},groups=groups,
        complete_range_scaling=summary['complete_range_scaling'],capped_time_scaling=summary['capped_time_scaling'],
        mean_t_gen_scaling=summary['mean_t_gen_scaling'],mean_capped_t_gen_scaling=summary['mean_capped_t_gen_scaling']),indent=2))


def plots(summary,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'C':'tab:blue','B':'tab:orange'};groups=summary['groups']
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(OUT/(name+'.'+ext),dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,key,title in zip(axes,['estimated_capacity_bits','parameters'],['Estimated empirical capacity (bits)','Actual parameter count']):
        for role in ['C','B']:
            gg=[g for g in groups if g['role']==role];x=[g[key] for g in gg];y=[g['t_gen']['mean_lower_bound'] for g in gg]
            ax.plot(x,y,color=colors[role],label=role+(' continuous' if role=='C' else ' categorical'),lw=1)
            for g,xx,yy in zip(gg,x,y):ax.scatter(xx,yy,color=colors[role],marker='o' if g['t_gen']['censored']==0 else '^',s=45)
            ax.plot(x,[g['t_gen']['median_lower_bound'] for g in gg],color=colors[role],ls=':',alpha=.6)
            for g,xx in zip(gg,x):
                if g['t_gen']['median'] is None:
                    ax.scatter(xx,g['t_gen']['median_lower_bound'],edgecolors=colors[role],facecolors='none',marker='^',s=55)
        ax.set(xscale='log',yscale='log',xlabel=title);ax.legend()
    axes[0].set_ylabel('Optimizer updates to first held-out 99%')
    fig.suptitle('p=97 · dropout 0.2 · ten paired seeds per size\nSolid: mean; dotted: median; triangles: lower bounds (open for median) · censoring at 50,000')
    save(fig,'scaling')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,role in zip(axes,['C','B']):
        fit=summary['constants'][role];gg=[g for g in groups if g['role']==role];lo,hi=fit['parameter_range'];k=fit['bits_per_parameter']
        full_lo=min(lo,min(g['parameters'] for g in gg));full_hi=max(hi,max(g['parameters'] for g in gg))
        ax.plot([full_lo,full_hi],[k*full_lo,k*full_hi],ls='--',color=colors[role],label='Constant extrapolation')
        ax.plot([lo,hi],[k*lo,k*hi],color=colors[role],label='Calibrated range')
        for seed,marker in [(61,'o'),(62,'s')]:
            pts=[p for p in fit['points'] if p['seed']==seed]
            ax.scatter([p['parameters'] for p in pts],[p['capacity_bits'] for p in pts],color=colors[role],marker=marker,label=f'Seed {seed}')
        ax.set(xlabel='Parameters',ylabel='Empirical capacity (bits)',title=f"{role}: κ={k:.4f} bits/parameter\nMaximum relative fit residual {100*fit['max_absolute_relative_residual']:.1f}%")
        ax.legend(fontsize=8)
    fig.suptitle('Capacity constants from interior random-label maxima; FF width = 4 × residual width')
    save(fig,'calibration')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for role in ['C','B']:
        gg=[g for g in groups if g['role']==role]
        axes[0].plot([g['estimated_capacity_bits'] for g in gg],[g['t_gen']['observed'] for g in gg],'-o',color=colors[role],label=role)
    axes[0].set(xscale='log',xlabel='Estimated capacity (bits)',ylabel='Seeds reaching held-out 99%',ylim=(-.4,10.4),yticks=range(0,11,2));axes[0].legend()
    comparisons=summary['paired_comparisons']
    x=[p['C_estimated_bits'] for p in summary['pairs']]
    for role in ['C','B']:axes[1].plot(x,[p[role+'_wins'] for p in comparisons],'-o',color=colors[role],label=role+' faster')
    axes[1].plot(x,[p['both_censored'] for p in comparisons],':s',color='grey',label='Both censored')
    axes[1].set(xscale='log',xlabel='Target estimated capacity (C bits)',ylabel='Paired seeds',ylim=(-.4,10.4),yticks=range(0,11,2));axes[1].legend()
    fig.suptitle('Held-out 99% crossings and paired speed comparisons · all ten seeds retained')
    save(fig,'success_and_pairs')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,role in zip(axes,['C','B']):
        for seed in range(401,411):
            rr=sorted([r for r in rows if r['role']==role and r['seed']==seed],key=lambda r:r['pair_index']);color=plt.get_cmap('tab10')(seed-401)
            xx=[r['estimated_capacity_bits'] for r in rr];yy=[r['t_gen_updates'] or LIMIT for r in rr]
            ax.plot(xx,yy,color=color,lw=.8,alpha=.6,label=seed)
            for r,x,y in zip(rr,xx,yy):ax.scatter(x,y,color=color,marker='o' if r['t_gen_updates'] is not None else '^',s=18)
        ax.set(title=role,xscale='log',yscale='log',xlabel='Estimated capacity (bits)')
    axes[0].set_ylabel('Optimizer updates to first held-out 99%');axes[-1].legend(ncol=2,fontsize=7,title='Seed')
    fig.suptitle('Individual trajectories across size · triangles are censored at 50,000 updates')
    save(fig,'individual_times')


if __name__=='__main__':main()
