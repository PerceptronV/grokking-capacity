#!/usr/bin/env python3
"""Read-only scientific audit/report for the registered p97 dropout experiment."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from grokking_capacity.experiments.matched_dropout import (
    data, evaluate, metrics, tensor_digest, boundary_audit, source_hashes, sha, atomic_json)
from grokking_capacity.models.discrete import make_model
from grokking_capacity.models.transformer import count_parameters
RUN=ROOT/'results/discrete/p97-dropout-match-v1'
REPORT=ROOT/'reports/p97_dropout_match'
TARGET=4656*np.log2(99)


def read(path):return json.loads(path.read_text())


def rows(path):return [json.loads(s) for s in path.read_text().splitlines()]


def csv_write(path,records):
    if not records:return
    keys=list(dict.fromkeys(k for r in records for k in r))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(records)


def recipe(c):return f"{c['model']}_d{c['dim']}_f{c['mlp_dim']}"


def capacity_diagnostics(path,s):
    trace=np.load(path/'trace.npz');bits=trace['bits_trace']*s['config']['n_samples']
    drift=None if len(bits)<200 else float(abs(bits[-100:].mean()/bits[-200:-100].mean()-1))
    entropy=s['config']['n_samples']*np.log2(s['config']['p']+2)
    return dict(late_drift=drift,epoch_limit=s['epochs']>=s['config']['epochs'],
                entropy_fraction=s['final_bits']/entropy,
                late_drift_flag=drift is None or drift>.01,
                entropy_limited=bool(s['final_bits']>=.95*entropy))



def effective_capacity(path,s):
    extension=RUN/'diagnostics'/path.name
    if not (extension/'summary.json').exists():return s,capacity_diagnostics(path,s),None
    e=read(extension/'summary.json');m=read(extension/'manifest.json')
    assert e['config']==s['config'] and e['primary_job']==path.name and e['replay_exact']
    assert m['primary_summary_sha256']==sha(path/'summary.json')
    assert m['primary_checkpoint_sha256']==sha(path/'model.pt')
    assert m['source_hashes']==source_hashes()
    assert m['extension_source_sha256']==sha(extension/'extension_source.py')
    assert e['converged'] and e['epochs']<=s['config']['epochs']
    assert sha(extension/'model.pt')==e['checkpoint_sha256']
    original=np.load(path/'trace.npz');continued=np.load(extension/'trace.npz')
    for key in ('bits_trace','train_loss_trace','train_acc_trace'):
        np.testing.assert_array_equal(original[key],continued[key][:s['epochs']])
    updated={**s,'final_bits':e['final_bits'],'final_eval':e['final_eval'],'epochs':e['epochs'],
             'peak_bits':float(continued['bits_trace'].max()*s['config']['n_samples'])}
    return updated,capacity_diagnostics(extension,updated),extension


def audit_extension(path,s):
    effective,d,extension=effective_capacity(path,s)
    if extension is None:return None
    c=s['config'];datasets=data(c);saved=torch.load(extension/'model.pt',map_location='cpu',weights_only=True)
    assert saved['config']==c
    model=make_model(c);model.load_state_dict(saved['model_state_dict'],strict=True)
    evaluation=evaluate(model,*datasets,'cpu',512)
    bits=len(datasets[1])*(np.log2(c['p']+2)-evaluation['loss']/np.log(2))
    assert abs(bits-effective['final_bits'])<.05
    assert evaluation['acc']==effective['final_eval']['acc']
    assert d['late_drift']<=.01 and not d['epoch_limit']
    return dict(name=extension.name,kind='capacity_continuation',epochs=effective['epochs'],
                bits_error=abs(bits-effective['final_bits']),final_bits=effective['final_bits'],
                boundary=boundary_audit(model,datasets[0],c,'cpu'),primary_trace_prefix_identical=True)


def audit_job(path,s):
    c=s['config'];manifest=read(path/'manifest.json')
    assert manifest['config']==c
    assert manifest['source_hashes']==source_hashes()
    assert sha(path/'model.pt')==s['checkpoint_sha256']
    torch.manual_seed(c['seed']);datasets=data(c);fresh=make_model(c)
    assert tensor_digest(datasets)==manifest['dataset_sha256']
    assert tensor_digest(fresh.state_dict().values())==manifest['initial_weights_sha256']
    initial=torch.load(path/'model_init.pt',map_location='cpu',weights_only=True)
    assert initial['config']==c
    assert tensor_digest(initial['model_state_dict'].values())==manifest['initial_weights_sha256']
    saved=torch.load(path/'model.pt',map_location='cpu',weights_only=True)
    assert saved['config']==c
    fresh.load_state_dict(saved['model_state_dict'],strict=True)
    assert count_parameters(fresh)==s['param_count']==manifest['param_count']
    assert c['dropout']==.2
    dropout_modules=[m for m in fresh.modules() if isinstance(m,torch.nn.Dropout)]
    assert dropout_modules and all(m.p==.2 for m in dropout_modules)
    train=evaluate(fresh,*datasets[:2],'cpu',512)
    result=dict(name=path.name,config=c,params=s['param_count'],train_eval=train,
                boundary=boundary_audit(fresh,datasets[0],c,'cpu'))
    if c['task']=='capacity':
        bits=len(datasets[1])*(np.log2(c['p']+2)-train['loss']/np.log(2))
        result.update(recomputed_bits=float(bits),bits_error=abs(bits-s['final_bits']))
        assert result['bits_error']<.05
        assert abs(train['acc']-s['final_eval']['acc'])<1e-9
        assert read(Path(c['gate_path'])/'summary.json')['learning_gate_passed']
        if c['n_samples']>2048:
            assert read(Path(c['random_gate_path'])/'summary.json')['random_learning_passed']
    else:
        val=evaluate(fresh,*datasets[2:],'cpu',512);result['val_eval']=val
        trace=rows(path/'trace.jsonl')
        assert len(trace)==s['epochs'] and trace[-1]==s['final'] and metrics(trace)==s['metrics']
        assert abs(train['acc']-trace[-1]['train_eval_acc'])<1e-9
        assert abs(val['acc']-trace[-1]['val_acc'])<1e-9
        assert all(r['epoch']==i+1 and r['step']==10*(i+1) for i,r in enumerate(trace))
    return result


def distributions(values):
    observed=[v for v in values if v is not None]
    return dict(total=len(values),observed=len(observed),missing=len(values)-len(observed),
                minimum=None if not observed else min(observed),
                median_observed=None if not observed else float(np.median(observed)),
                positive=sum(v>0 for v in observed),zero=sum(v==0 for v in observed))


def build(require_complete=False):
    frozen=read(RUN/'manifest.json');assert frozen['source_hashes']==source_hashes()
    assert frozen['registration_sha256']==sha(RUN/'registration.yaml')
    all_jobs=[(p,read(p/'summary.json')) for p in sorted((RUN/'jobs').iterdir()) if (p/'summary.json').exists()]
    capacity=[];arithmetic=[];files={str(RUN/'manifest.json'):sha(RUN/'manifest.json')}
    for p,s in all_jobs:
        for f in ('summary.json','manifest.json','model.pt','model_init.pt','trace.jsonl','trace.npz'):
            if (p/f).exists():files[str(p/f)]=sha(p/f)
        c=s['config'];base=dict(name=p.name,role=c['role'],recipe=recipe(c),seed=c['seed'],params=s['param_count'],
                              dim=c['dim'],mlp_dim=c['mlp_dim'],phase=c['phase'],epochs=s['epochs'])
        if s['task']=='capacity':
            effective,diagnostics,extension=effective_capacity(p,s)
            capacity.append({**base,'epochs':effective['epochs'],'primary_epochs':s['epochs'],
                'n_samples':c['n_samples'],'final_bits':effective['final_bits'],'primary_final_bits':s['final_bits'],
                'peak_bits':effective['peak_bits'],'ratio_to_dataset':effective['final_bits']/TARGET,
                'train_eval_acc':effective['final_eval']['acc'],'continuation_path':str(extension) if extension else None,
                **diagnostics})
            if extension:
                for artifact in extension.iterdir():
                    if artifact.is_file():files[str(artifact)]=sha(artifact)
        else:
            arithmetic.append({**base,**s['metrics'],'final_train_online':s['final']['train_online_acc'],
                'final_train_eval':s['final']['train_eval_acc'],'final_val':s['final']['val_acc'],
                'joint_stop_reached':s['epochs']<c['epochs'],'learning_gate_passed':s['learning_gate_passed'],
                'val95_updates':None if s['metrics']['val_95_sustained_epoch'] is None else 10*s['metrics']['val_95_sustained_epoch'],
                'val99_updates':None if s['metrics']['val_99_sustained_epoch'] is None else 10*s['metrics']['val_99_sustained_epoch']})
    maxima=[]
    for rec,seed in sorted({(r['recipe'],r['seed']) for r in capacity}):
        cells=[r for r in capacity if (r['recipe'],r['seed'])==(rec,seed)]
        winner=max(cells,key=lambda r:r['final_bits'])
        maxima.append({**winner,'largest_dataset_boundary':winner['n_samples']==max(r['n_samples'] for r in cells),
                       'n_grid':','.join(str(r['n_samples']) for r in sorted(cells,key=lambda r:r['n_samples']))})
    confirm=[r for r in arithmetic if r['phase']=='confirmation']
    selection=read(RUN/'selection.json') if (RUN/'selection.json').exists() else None
    aggregates={};pairing=[];paired_speeds={}
    if selection:
        assert selection['registration_sha256']==sha(RUN/'registration.yaml')
        for path,expected in selection['evidence_sha256'].items():assert sha(path)==expected
        assert selection['param_counts']['B']==selection['param_counts']['C']
        for seed in ('61','62'):
            a=selection['evidence']['A'][seed]['final_bits'];b=selection['evidence']['B'][seed]['final_bits']
            assert abs(a/TARGET-1)<=.1 and abs(b/TARGET-1)<=.1 and abs(a-b)/min(a,b)<=.1
            assert selection['evidence']['C'][seed]['final_bits']>1.1*TARGET
        files[str(RUN/'selection.json')]=sha(RUN/'selection.json')
        for role in 'ABC':
            rr=[r for r in confirm if r['role']==role]
            assert len({r['seed'] for r in rr})==len(rr)
            if require_complete:assert sorted(r['seed'] for r in rr)==list(range(101,111))
            aggregates[role]={key:distributions([r[key] for r in rr]) for key in
                             ['paper_delay_epochs','eval_delay_epochs','val95_updates','val99_updates']}
            for speed in ('val95_updates','val99_updates'):
                times=sorted(r[speed] for r in rr if r[speed] is not None)
                aggregates[role][speed]['median_all_ten']=float(np.mean(times[4:6])) if len(rr)==10 and len(times)>=6 else None
                aggregates[role][speed]['right_censored_at_updates']=50000
            aggregates[role]['ever_held99']=sum(r['val_acc_99_epoch'] is not None for r in rr)
            aggregates[role]['final_held99']=sum(r['final_val']>=.99 for r in rr)
            aggregates[role]['final_joint99']=sum(r['final_val']>=.99 and r['final_train_online']>=.99 for r in rr)
            aggregates[role]['confirmation_runs']=len(rr)
            for p,s in all_jobs:
                c=s['config']
                if c['phase']=='confirmation' and c['role']==role:
                    assert {k:c[k] for k in selection['models'][role]}==selection['models'][role]
                    assert read(p/'manifest.json')['started_at']>=selection['frozen_at']
                    assert c['selection_sha256']==sha(RUN/'selection.json')
        for seed in range(101,111):
            pp={s['config']['role']:(p,s) for p,s in all_jobs if s['config']['phase']=='confirmation' and s['config']['seed']==seed}
            if len(pp)!=3:continue
            mm={k:read(v[0]/'manifest.json') for k,v in pp.items()}
            assert len({m['dataset_sha256'] for m in mm.values()})==1
            assert mm['B']['initial_weights_sha256']==mm['C']['initial_weights_sha256']
            assert mm['B']['param_count']==mm['C']['param_count']
            pairing.append(dict(seed=seed,dataset_sha256=mm['B']['dataset_sha256'],BC_initial_sha256=mm['B']['initial_weights_sha256']))
        for threshold in ('val95_updates','val99_updates'):
            pairs={}
            for left,right in [('A','B'),('A','C'),('B','C')]:
                counts=dict(left_faster=0,right_faster=0,tied=0,left_only_reached=0,right_only_reached=0,neither=0,missing_runs=0)
                for seed in range(101,111):
                    pair={r['role']:r[threshold] for r in confirm if r['seed']==seed}
                    if left not in pair or right not in pair:counts['missing_runs']+=1;continue
                    a,b=pair[left],pair[right]
                    key='neither' if a is None and b is None else 'right_only_reached' if a is None else 'left_only_reached' if b is None else 'tied' if a==b else 'left_faster' if a<b else 'right_faster'
                    counts[key]+=1
                pairs[left+'_vs_'+right]=counts
            paired_speeds[threshold]=pairs
    if require_complete:assert selection is not None and len(confirm)==30 and len(pairing)==10
    summary=dict(target_bits=float(TARGET),completed_jobs=len(all_jobs),capacity_cells=len(capacity),
                 arithmetic_runs=len(arithmetic),selection=selection,capacity_maxima=maxima,
                 aggregates=aggregates,pairing=pairing,paired_speeds=paired_speeds,
                 notes=['Missing threshold crossings are censored, never imputed as zero.',
                        'Observed-only medians are conditional on reaching the threshold.',
                        'Paper delay uses dropout-active online train99 and held-out eval98.',
                        'Each epoch is ten optimizer updates; shared GPU wall time is not a model speed metric.'])
    REPORT.mkdir(parents=True,exist_ok=True)
    csv_write(REPORT/'capacity_cells.csv',capacity);csv_write(REPORT/'capacity_maxima.csv',maxima)
    csv_write(REPORT/'arithmetic_runs.csv',arithmetic);csv_write(REPORT/'confirmation_runs.csv',confirm)
    atomic_json(REPORT/'summary.json',summary);atomic_json(REPORT/'input_hashes.json',files)
    return all_jobs,summary


def plots(all_jobs,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharex=True,sharey=True)
    for ax,role in zip(axes,'ABC'):
        for p,s in all_jobs:
            if s['config']['phase']!='confirmation' or s['config']['role']!=role:continue
            t=rows(p/'trace.jsonl');x=[r['step'] for r in t]
            ax.plot(x,[r['val_acc'] for r in t],alpha=.65,lw=.75,label=str(s['config']['seed']))
        ax.set(title=role,xlabel='Optimizer updates',ylim=(0,1.02),xlim=(0,50000));ax.axhline(.99,c='k',ls=':',lw=.7)
    axes[0].set_ylabel('Held-out accuracy');axes[-1].legend(title='Seed',fontsize=6,ncol=2)
    fig.suptitle('p=97 modular division · dropout 0.2 · capacity-matched A/B, continuous C')
    fig.tight_layout();fig.savefig(REPORT/'confirmation_accuracy.png',dpi=180);fig.savefig(REPORT/'confirmation_accuracy.pdf');plt.close(fig)
    if summary['selection']:
        fig,ax=plt.subplots(figsize=(7,4))
        for role,color in zip('ABC',('tab:blue','tab:orange','tab:green')):
            model=summary['selection']['models'][role]
            for seed in (61,62):
                points=sorted((s['config']['n_samples'],effective_capacity(p,s)[0]['final_bits']/1000) for p,s in all_jobs
                    if s['task']=='capacity' and s['config']['seed']==seed
                    and all(s['config'].get(k)==v for k,v in model.items()))
                ax.plot(*zip(*points),color=color,marker='o',ls='-' if seed==61 else '--',
                        label=f'{role}, seed {seed}')
        ax.axhspan(.9*TARGET/1000,1.1*TARGET/1000,color='grey',alpha=.15,label='Dataset ±10%')
        ax.axhline(TARGET/1000,color='k',lw=.7)
        ax.set(xscale='log',xlabel='Random-label examples',ylabel='Final learned information (kbit)',
               title='Capacity under the same dropout 0.2 training protocol')
        ax.legend(fontsize=8,ncol=2);fig.tight_layout()
        fig.savefig(REPORT/'capacity_curves.png',dpi=180);fig.savefig(REPORT/'capacity_curves.pdf');plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit',action='store_true')
    parser.add_argument('--plots',action='store_true');parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2)
    jobs,summary=build(args.require_complete)
    if args.audit:
        audits=[]
        for i,(p,s) in enumerate(jobs):
            audits.append(audit_job(p,s))
            if s['task']=='capacity':
                extension_audit=audit_extension(p,s)
                if extension_audit:audits.append(extension_audit)
            if i%10==0:print(f'audited {i+1}/{len(jobs)}',flush=True)
        atomic_json(REPORT/'checkpoint_audits.json',audits)
        atomic_json(REPORT/'validation.json',dict(checkpoints_audited=len(audits),all_passed=True,
            max_capacity_bits_error=max((a.get('bits_error',0.) for a in audits),default=0.),
            initial_weights_recreated=True,datasets_recreated=True,binary_train_eval_boundaries_verified=True,
            BC_pairing_verified=len(summary['pairing']),analysis_sha256=sha(Path(__file__))))
    if args.plots:plots(jobs,summary)
    print(json.dumps({k:summary[k] for k in ('completed_jobs','capacity_cells','arithmetic_runs','aggregates','paired_speeds')},indent=2))


if __name__=='__main__':main()
