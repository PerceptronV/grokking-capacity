#!/usr/bin/env python3
"""Audit and compare the two fixed-p97 matched discrete families."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from grokking_capacity.experiments.multilevel_dropout import (
    data,evaluate,metrics,tensor_digest,boundary_audit,source_hashes,sha,atomic_json,make_model)
from grokking_capacity.models.transformer import count_parameters
from manage_p97_multilevel import RUN,TARGET,read,jobs,reg,diagnostics,verify_inherited_a
REPORT=ROOT/'reports/p97_multilevel_dropout'


def rows(path):return [json.loads(s) for s in path.read_text().splitlines()]


def csv_write(path,records):
    if not records:return
    keys=list(dict.fromkeys(k for r in records for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)


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
    result=dict(name=path.name,config=c,params=s['param_count'],train_eval=train,verification_device='cpu',
                boundary=boundary_audit(fresh,datasets[0],c,'cpu'))
    if c['task']=='capacity':
        bits=len(datasets[1])*(np.log2(c['p']+2)-train['loss']/np.log(2))
        if abs(bits-s['final_bits'])>=.05 or train['acc']!=s['final_eval']['acc']:
            result.update(cpu_bits_error=float(bits-s['final_bits']),cpu_train_eval=train,
                          verification_device='cuda:1',cpu_gpu_discrepancy=True)
            fresh=fresh.to('cuda:1');train=evaluate(fresh,*datasets[:2],'cuda:1',512)
            bits=len(datasets[1])*(np.log2(c['p']+2)-train['loss']/np.log(2));result['train_eval']=train
        result.update(recomputed_bits=float(bits),bits_error=abs(bits-s['final_bits']))
        assert result['bits_error']<.05,(path.name,result['bits_error'])
        assert abs(train['acc']-s['final_eval']['acc'])<1e-9
        assert read(Path(c['gate_path'])/'summary.json')['learning_gate_passed']
        if c['n_samples']>2048:
            assert read(Path(c['random_gate_path'])/'summary.json')['random_learning_passed']
    else:
        val=evaluate(fresh,*datasets[2:],'cpu',512);result['val_eval']=val
        trace=rows(path/'trace.jsonl')
        assert len(trace)==s['epochs'] and trace[-1]==s['final'] and metrics(trace)==s['metrics']
        if train['acc']!=trace[-1]['train_eval_acc'] or val['acc']!=trace[-1]['val_acc']:
            result.update(cpu_train_eval=train,cpu_val_eval=val,verification_device='cuda:1',cpu_gpu_discrepancy=True)
            fresh=fresh.to('cuda:1');train=evaluate(fresh,*datasets[:2],'cuda:1',512)
            val=evaluate(fresh,*datasets[2:],'cuda:1',512);result.update(train_eval=train,val_eval=val)
        assert abs(train['acc']-trace[-1]['train_eval_acc'])<1e-9
        assert abs(val['acc']-trace[-1]['val_acc'])<1e-9
        assert all(r['epoch']==i+1 and r['step']==10*(i+1) for i,r in enumerate(trace))
    return result


def independent_metrics(trace):
    result={}
    def first(mask):
        indices=np.flatnonzero(mask)
        return None if len(indices)==0 else int(trace[int(indices[0])]['epoch'])
    for key in ('train_online_acc','train_eval_acc','val_acc'):
        array=np.asarray([r[key] for r in trace])
        for percent in (95,98,99):result[f'{key}_{percent}_epoch']=first(array>=percent/100)
    val=np.asarray([r['val_acc'] for r in trace])
    for percent in (95,99):
        windows=np.lib.stride_tricks.sliding_window_view(val>=percent/100,3) if len(val)>=3 else np.empty((0,3),bool)
        result[f'val_{percent}_sustained_epoch']=first(windows.all(axis=1))
    for label,key in [('paper','train_online_acc'),('eval','train_eval_acc')]:
        a,b=result[f'{key}_99_epoch'],result['val_acc_98_epoch']
        result[f'{label}_delay_epochs']=None if a is None or b is None else max(0,b-a)
    return result


def bound(trace,percent):
    # If a final streak is not yet confirmed, its onset may still become the
    # first qualifying three-epoch window. Keep that possibility in the bound.
    tail=0
    for row in reversed(trace):
        if row['val_acc']<percent/100:break
        tail+=1
    assert tail<3
    return (trace[-1]['epoch']-tail+1)*10-10


def aggregate(records):
    out={'runs':len(records)}
    for key in ('paper_delay_updates','eval_delay_updates'):
        values=[r[key] for r in records if r[key] is not None]
        out[key]=dict(min_observed=min(values) if values else None,
                      positive=sum(v>0 for v in values),zero=sum(v==0 for v in values),
                      undefined=len(records)-len(values),median_observed=float(np.median(values)) if values else None)
    for threshold in (95,99):
        key=f'val{threshold}_updates';values=sorted(r[key] for r in records if r[key] is not None)
        bounds=[r[f'val{threshold}_censor_updates'] for r in records if r[key] is None]
        out[key]=dict(reached=len(values),censored=len(bounds),median_all_ten=float(np.mean(values[4:6])) if len(records)==10 and len(values)>=6 else None,
                      median_all_ten_lower_bound=float(np.median(values+bounds)) if len(records)==10 else None,
                      median_observed_only=float(np.median(values)) if values else None)
    out['final_held99']=sum(r['final_val']>=.99 for r in records)
    return out


def build(require_complete=False,audit=False,plots=False):
    REPORT.mkdir(parents=True,exist_ok=True);all_jobs=jobs();registration=reg()
    assert read(RUN/'manifest.json')['source_hashes']==source_hashes()
    inherited=verify_inherited_a();arithmetic=[];capacity=[];audits=[];files={};pairing=[]
    prior=read(REPORT/'checkpoint_audits.json') if (REPORT/'checkpoint_audits.json').exists() else []
    cache={v['name']:v for v in prior}
    for path,s in all_jobs:
        c=s['config'];m=read(path/'manifest.json')
        assert m['config']==c and m['source_hashes']==source_hashes()
        assert sha(path/'model.pt')==s['checkpoint_sha256']
        for name in ('summary.json','manifest.json','model.pt','model_init.pt'):
            files[str(path/name)]=sha(path/name)
        if audit:
            # A result may only reuse an audit when input bytes AND audit source match.
            key=sha(path/'summary.json')+sha(path/'model.pt')+sha(path/'model_init.pt')+sha(Path(__file__))
            cached=cache.get(path.name)
            if cached and cached.get('cache_key')==key:a=cached
            else:
                a=audit_job(path,s);a['cache_key']=key
                print('audited '+path.name,flush=True)
            audits.append(a);atomic_json(REPORT/'checkpoint_audits.json',audits)
        if s['task']=='capacity':
            d=diagnostics(path,s)
            capacity.append(dict(name=path.name,family=c['family'],role=c['role'],model=c['model'],dim=c['dim'],
                mlp_dim=c['mlp_dim'],params=s['param_count'],seed=c['seed'],n_samples=c['n_samples'],
                epochs=s['epochs'],final_bits=s['final_bits'],primary_bits=s['primary_bits'],
                train_acc=s['final_eval']['acc'],**d))
            files[str(path/'trace.npz')]=sha(path/'trace.npz')
        else:
            trace=rows(path/'trace.jsonl');mm=independent_metrics(trace)
            assert mm==s['metrics'] and trace[-1]==s['final'] and len(trace)==s['epochs']
            assert all(r['epoch']==i+1 and r['step']==10*(i+1) for i,r in enumerate(trace))
            r=dict(name=path.name,family=c['family'],role=c['role'],phase=c['phase'],seed=c['seed'],
                model=c['model'],dim=c['dim'],mlp_dim=c['mlp_dim'],params=s['param_count'],epochs=s['epochs'],
                final_val=trace[-1]['val_acc'],final_train_eval=trace[-1]['train_eval_acc'],
                final_train_online=trace[-1]['train_online_acc'],gate_passed=s['learning_gate_passed'],**mm)
            for key in ('paper_delay','eval_delay'):
                value=mm[key+'_epochs'];r[key+'_updates']=None if value is None else value*10
            for threshold in (95,99):
                value=mm[f'val_{threshold}_sustained_epoch']
                r[f'val{threshold}_updates']=None if value is None else value*10
                r[f'val{threshold}_censor_updates']=bound(trace,threshold) if value is None else None
            arithmetic.append(r);files[str(path/'trace.jsonl')]=sha(path/'trace.jsonl')
    confirm=[r for r in arithmetic if r['phase']=='confirmation'];families={}
    for family in registration['families']:
        relevant=[r for r in confirm if r['family'] in ('shared',family)]
        models={role:aggregate([r for r in relevant if r['role']==role]) for role in 'ABC'}
        selection=read(RUN/f'{family}_AB_selection.json') if (RUN/f'{family}_AB_selection.json').exists() else None
        control=read(RUN/f'{family}_C_construction.json') if (RUN/f'{family}_C_construction.json').exists() else None
        binding=read(RUN/f'{family}_confirmation_binding.json') if (RUN/f'{family}_confirmation_binding.json').exists() else None
        if selection:
            assert selection['registration_sha256']==sha(RUN/'registration.yaml')
            for p,h in selection['evidence_sha256'].items():assert sha(p)==h
        if control:
            assert control['constructed_at']>=selection['frozen_at']
            assert control['AB_selection_sha256']==sha(RUN/f'{family}_AB_selection.json')
        if binding:
            for p,h in binding['C_evidence_sha256'].items():assert sha(p)==h
            assert binding['C_construction_sha256']==sha(RUN/f'{family}_C_construction.json')
        pairs={}
        for threshold in (95,99):
            counts={}
            for left,right in [('A','B'),('A','C'),('B','C')]:
                wins=dict(left_faster=0,right_faster=0,tie=0,left_only=0,right_only=0,neither=0,missing=0)
                for seed in registration['confirmation']['seeds']:
                    rr={r['role']:r[f'val{threshold}_updates'] for r in relevant if r['seed']==seed}
                    if left not in rr or right not in rr:key='missing'
                    else:
                        a,b=rr[left],rr[right]
                        key='neither' if a is None and b is None else 'right_only' if a is None else 'left_only' if b is None else 'tie' if a==b else 'left_faster' if a<b else 'right_faster'
                    wins[key]+=1
                counts[left+'_vs_'+right]=wins
            pairs[str(threshold)]=counts
        for seed in registration['confirmation']['seeds']:
            rr={s['config']['role']:(p,s) for p,s in all_jobs if s['config']['phase']=='confirmation'
                and s['config']['family'] in ('shared',family) and s['config']['seed']==seed}
            if len(rr)!=3:continue
            manifests={role:read(p/'manifest.json') for role,(p,s) in rr.items()}
            assert len({v['dataset_sha256'] for v in manifests.values()})==1
            assert abs(manifests['B']['param_count']/manifests['C']['param_count']-1)<=.001
            for role in ('B','C'):
                config=rr[role][1]['config']
                expected=selection['models']['B'] if role=='B' else control['model']
                assert all(config[k]==v for k,v in expected.items())
                assert config['binding_sha256']==sha(RUN/f'{family}_confirmation_binding.json')
                assert manifests[role]['started_at']>=binding['frozen_at']
            equal=manifests['B']['initial_weights_sha256']==manifests['C']['initial_weights_sha256']
            if family=='q8':assert equal
            pairing.append(dict(family=family,seed=seed,dataset_sha256=manifests['A']['dataset_sha256'],
                                BC_full_initial_weights_equal=equal))
        if require_complete:
            assert selection and control and binding
            for role in 'ABC':assert sorted(r['seed'] for r in relevant if r['role']==role)==registration['confirmation']['seeds']
        families[family]=dict(AB_selection=selection,C_construction=control,confirmation_binding=binding,
                              aggregates=models,paired_speeds=pairs)
    if require_complete:assert len(confirm)==50 and len(pairing)==20
    summary=dict(target_bits=float(TARGET),completed_jobs=len(all_jobs),families=families,
        confirmation_runs=len(confirm),arithmetic_runs=len(arithmetic),capacity_cells=len(capacity),pairing=pairing,
        inherited_A=inherited,notes=[
          'The same ten new A trajectories are shared across families, not twenty independent runs.',
          'Missing threshold crossings stay undefined; speed medians include all ten seeds with censoring.',
          'Categorical B uses deterministic argmax with normalized-softmax STE; no sampled training.',
          'All arithmetic speeds are optimizer updates, ten per epoch; wall time is descriptive on shared GPUs.',
          'Capacity matching and finite-budget delay estimates are not intrinsic capacity bounds or a theorem.'])
    csv_write(REPORT/'capacity_cells.csv',capacity);csv_write(REPORT/'arithmetic_runs.csv',arithmetic)
    csv_write(REPORT/'confirmation_runs.csv',confirm)
    atomic_json(REPORT/'summary.json',summary);atomic_json(REPORT/'input_hashes.json',files)
    if audit:
        atomic_json(REPORT/'validation.json',dict(all_passed=True,checkpoints_audited=len(audits),
            metrics_independently_recomputed=len(arithmetic),pairings_verified=len(pairing),
            analysis_sha256=sha(Path(__file__)),complete=require_complete,
            max_bits_error=max((a.get('bits_error',0) for a in audits),default=0)))
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for family in families:
            fig,axes=plt.subplots(1,3,figsize=(13,4),sharex=True,sharey=True)
            for ax,role in zip(axes,'ABC'):
                for p,s in all_jobs:
                    c=s['config']
                    if c['phase']!='confirmation' or c['family'] not in ('shared',family) or c['role']!=role:continue
                    t=rows(p/'trace.jsonl');ax.plot([r['step'] for r in t],[r['val_acc'] for r in t],lw=.8,alpha=.65,label=str(c['seed']))
                ax.set(title=role,xlabel='Optimizer updates',ylim=(0,1.02),xlim=(0,50000));ax.axhline(.99,c='k',ls=':',lw=.7)
            axes[0].set_ylabel('Held-out accuracy');axes[-1].legend(fontsize=6,ncol=2)
            fig.suptitle(f'p=97 · dropout 0.2 · {family} · capacity-matched A/B')
            fig.tight_layout();fig.savefig(REPORT/f'{family}_accuracy.png',dpi=160);fig.savefig(REPORT/f'{family}_accuracy.pdf');plt.close(fig)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',action='store_true');p.add_argument('--plots',action='store_true');p.add_argument('--require-complete',action='store_true')
    a=p.parse_args();torch.set_num_threads(2);s=build(a.require_complete,a.audit,a.plots)
    print(json.dumps({k:s[k] for k in ('completed_jobs','confirmation_runs','arithmetic_runs','capacity_cells')},indent=2))


if __name__=='__main__':main()
