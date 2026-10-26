#!/usr/bin/env python3
"""Independent parameter, data, trace, selection, stopping and reporting checks."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/discrete/p97-dataset-scaling-v1'
OUT=ROOT/'reports/p97_dataset_scaling'
sys.path.insert(0,str(ROOT/'src'))
from grokking_capacity.data import grokking_data_torch


def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def tensor_digest(values):
    h=hashlib.sha256()
    for value in values:
        h.update(str(tuple(value.shape)).encode());h.update(value.numpy().tobytes())
    return h.hexdigest()


def main():
    assert read(RUN/'campaign_complete.json')['all_passed']
    manifest=read(RUN/'manifest.json');selection=read(RUN/'selection.json')
    reg=yaml.safe_load((RUN/'registration.yaml').read_text());registry=read(RUN/'job_registry.json')
    for p,h in manifest['source_hashes'].items():assert digest(ROOT/p)==h
    for path,h in selection['calibration_evidence'].items():assert digest(Path(path))==h
    assert manifest['registry_sha256']==digest(RUN/'job_registry.json')
    assert manifest['selection_sha256']==digest(RUN/'selection.json')
    assert manifest['registration_sha256']==digest(RUN/'registration.yaml')
    sizes=reg['dataset']['sizes'];seeds=reg['confirmation']['seeds']
    assert sizes==np.rint(np.exp(np.linspace(np.log(928),np.log(7449),10))).astype(int).tolist()
    assert len(registry)==200
    constants={}
    for role in ['C','B']:
        f=selection['constants'][role];points=f['points']
        constants[role]=sum(r['parameters']*r['capacity_bits'] for r in points)/sum(r['parameters']**2 for r in points)
        assert math.isclose(constants[role],f['bits_per_parameter'],rel_tol=1e-14)
    for n,pair in zip(sizes,selection['pairs']):
        assert pair['n_train']==n and pair['dataset_bits']==n*math.log2(99)
        for role,factor in [('C',32),('B',64)]:
            def estimate(w):return constants[role]*(factor*w*w+203*w)
            chosen=min(range(2,162,2),key=lambda w:(abs(estimate(w)/(n*math.log2(99))-1),w))
            assert pair[role]['dim']==chosen and pair[role]['mlp_dim']==4*chosen
            assert pair[role+'_parameters']==factor*chosen*chosen+203*chosen
            assert math.isclose(pair[role+'_estimated_bits'],estimate(chosen),rel_tol=1e-14)
            assert abs(estimate(chosen)/(n*math.log2(99))-1)<=.1
        assert abs(pair['B_estimated_bits']/pair['C_estimated_bits']-1)<=.1
    # Construct data using the original half-split generator, then concatenate;
    # this independently recovers the full permutation used by the new worker.
    data={}
    for seed in seeds:
        np.random.seed(seed)
        x,t,v,u=grokking_data_torch(97,train_fraction=.5)
        X,T=torch.cat((x,v)),torch.cat((t,u))
        assert len(T)==9312
        test=X[7449:],T[7449:]
        for n in sizes:
            data[seed,n]=(X[:n],T[:n],*test)
            assert not set(map(tuple,X[:n].tolist())).intersection(map(tuple,test[0].tolist()))
    reportrows={r['name']:r for r in csv.DictReader((OUT/'runs.csv').open())}
    computed=[];crossings=0;joint_success=0;learning=0
    for name,c in registry.items():
        p=RUN/'jobs'/name;s=read(p/'summary.json');m=read(p/'manifest.json')
        assert c==s['config']==m['config']
        assert m['started_at']>selection['frozen_at']
        pair=selection['pairs'][c['pair_index']]
        assert pair['n_train']==c['n_train']
        assert all(c[k]==v for k,v in pair[c['role']].items())
        assert tensor_digest(data[c['seed'],c['n_train']])==m['dataset_sha256']
        trace=[json.loads(line) for line in (p/'trace.jsonl').read_text().splitlines()]
        assert trace[-1]==s['final']
        earliest_stop=None
        steps_per_epoch=(c['n_train']+c['batch_size']-1)//c['batch_size']
        for i,row in enumerate(trace):
            step=(i+1)*10
            assert row['step']==step<=50000
            epochs,partial=divmod(step,steps_per_epoch)
            examples=epochs*c['n_train']+partial*c['batch_size']
            assert row['examples']==examples
            assert row['completed_epochs']==epochs
            assert row['epoch_fraction']==epochs+partial*c['batch_size']/c['n_train']
            if i>=2 and all(trace[j]['val_acc']>=.99 for j in range(i-2,i+1)) and row['train_online_acc']>=.99:
                if earliest_stop is None:earliest_stop=step
        assert trace[-1]['step']==(earliest_stop if earliest_stop is not None else 50000)
        assert s['joint_stop_passed']==(earliest_stop is not None)
        expected={}
        for key in ['train_online_acc','train_eval_acc','val_acc']:
            for percent in [95,98,99]:
                found=[r['step'] for r in trace if r[key]>=percent/100]
                expected[f'{key}_{percent}_updates']=min(found) if found else None
        sustained=[trace[i-2]['step'] for i in range(2,len(trace)) if min(r['val_acc'] for r in trace[i-2:i+1])>=.99]
        expected['val_99_sustained_updates']=min(sustained) if sustained else None
        for label,key in [('paper','train_online_acc'),('eval','train_eval_acc')]:
            tr,va=expected[f'{key}_99_updates'],expected['val_acc_98_updates']
            expected[f'{label}_delay_updates']=None if tr is None or va is None else max(0,va-tr)
        assert expected==s['metrics']
        final_ok=bool(sustained and trace[-1]['val_acc']>=.99 and trace[-1]['train_eval_acc']>=.99)
        assert final_ok==s['learning_gate_passed']
        r=reportrows[name];t=expected['val_acc_99_updates']
        assert r['t_gen_updates']==('' if t is None else str(t))
        assert r['final_learning']==str(final_ok) and r['joint_stop']==str(earliest_stop is not None)
        computed.append(dict(role=c['role'],seed=c['seed'],n_train=c['n_train'],t=t))
        crossings+=t is not None;joint_success+=earliest_stop is not None;learning+=final_ok
    summary=read(OUT/'summary.json')
    allcomplete=[];difference=[]
    for row,n in zip(summary['sizes'],sizes):
        means={};complete=True
        for role in ['C','B']:
            ts=[r['t'] for r in computed if r['role']==role and r['n_train']==n]
            assert len(ts)==10
            observed=[t for t in ts if t is not None]
            means[role]=sum(t if t is not None else 50000 for t in ts)/10
            assert row[role+'_capped_mean']==means[role]
            assert row[role+'_observed']==len(observed)
            assert row[role+'_mean']==(means[role] if len(observed)==10 else None)
            complete=complete and len(observed)==10
        difference.append(means['B']-means['C']);allcomplete.append(complete)
    bracket_indices=[i for i in range(1,10) if allcomplete[i-1] and allcomplete[i] and difference[i-1]*difference[i]<0]
    assert [(r['lower_n'],r['upper_n']) for r in summary['observed_mean_crossing_brackets']]==[(sizes[i-1],sizes[i]) for i in bracket_indices]
    for path,h in read(OUT/'input_hashes.json').items():assert digest(Path(path))==h
    result=dict(all_passed=True,runs=200,dataset_sizes=10,seeds=10,first99=crossings,
        joint_success=joint_success,final_learning=learning,observed_crossing_brackets=len(bracket_indices),
        verifier_sha256=digest(Path(__file__)),source_commit=manifest['source_commit'])
    (OUT/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
