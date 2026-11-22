#!/usr/bin/env python3
"""Independent NumPy threshold/count verification; no imports from the runner."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
RUN=ROOT/'results/discrete/p97-dropout-match-v1'
REPORT=ROOT/'reports/p97_dropout_match'


def first(values,threshold):
    where=np.flatnonzero(np.asarray(values)>=threshold)
    return int(where[0])+1 if len(where) else None


def sustained(values,threshold):
    flags=(np.asarray(values)>=threshold).astype(np.int32)
    where=np.flatnonzero(np.convolve(flags,np.ones(3,dtype=np.int32),mode='valid')==3)
    return int(where[0])+1 if len(where) else None


def integer(value):return None if value in ('',None) else int(value)


def main():
    summary=json.loads((REPORT/'summary.json').read_text())
    with (REPORT/'confirmation_runs.csv').open() as f:table={r['name']:r for r in csv.DictReader(f)}
    assert len(table)==30
    checked=[]
    for seed in range(101,111):
        for role in 'ABC':
            name=f'{role}_confirmation_s{seed}'
            path=RUN/'jobs'/name
            trace=[json.loads(s) for s in (path/'trace.jsonl').read_text().splitlines()]
            result=json.loads((path/'summary.json').read_text())
            c=result['config'];assert c['seed']==seed and c['role']==role and c['dropout']==.2 and c['p']==97
            assert c['epochs']==5000 and len(trace)<=5000
            arrays={k:np.array([r[k] for r in trace]) for k in ('train_online_acc','train_eval_acc','val_acc')}
            expected={}
            for key in arrays:
                for percent in (95,98,99):expected[f'{key}_{percent}_epoch']=first(arrays[key],percent/100)
            for percent in (95,99):
                t=sustained(arrays['val_acc'],percent/100)
                expected[f'val_{percent}_sustained_epoch']=t
                assert integer(table[name][f'val{percent}_updates'])==(None if t is None else t*10)
                if t is None:
                    below=np.flatnonzero(arrays['val_acc']<percent/100)
                    bound=10*(int(below[-1])+1)
                    assert integer(table[name][f'val{percent}_censor_updates'])==bound
                else:assert integer(table[name][f'val{percent}_censor_updates']) is None
            for label,key in [('paper','train_online_acc'),('eval','train_eval_acc')]:
                a=expected[f'{key}_99_epoch'];b=expected['val_acc_98_epoch']
                expected[f'{label}_delay_epochs']=None if a is None or b is None else max(0,b-a)
            assert result['metrics']==expected,(name,'summary metrics differ')
            for key,value in expected.items():assert integer(table[name][key])==value,(name,key)
            assert [r['step'] for r in trace]==list(range(10,10*len(trace)+1,10))
            if len(trace)<5000:
                assert trace[-1]['train_online_acc']>=.99 and min(r['val_acc'] for r in trace[-3:])>=.99
            checked.append(dict(name=name,role=role,seed=seed,epochs=len(trace),**expected))
    for role in 'ABC':
        group=[r for r in checked if r['role']==role];agg=summary['aggregates'][role]
        for metric in ('paper_delay_epochs','eval_delay_epochs'):
            values=[r[metric] for r in group if r[metric] is not None]
            assert agg[metric]['minimum']==(min(values) if values else None)
            assert agg[metric]['observed']==len(values) and agg[metric]['missing']==10-len(values)
            assert agg[metric]['positive']==sum(v>0 for v in values)
            assert agg[metric]['zero']==sum(v==0 for v in values)
        for threshold in (95,99):
            values=sorted(10*r[f'val_{threshold}_sustained_epoch'] for r in group if r[f'val_{threshold}_sustained_epoch'] is not None)
            median=None if len(values)<6 else (values[4]+values[5])/2
            assert agg[f'val{threshold}_updates']['median_all_ten']==median
            assert agg[f'val{threshold}_updates']['observed']==len(values)
            bounds=[integer(table[r['name']][f'val{threshold}_censor_updates']) for r in group if r[f'val_{threshold}_sustained_epoch'] is None]
            truncated=sorted(values+bounds)
            assert agg[f'val{threshold}_updates']['median_all_ten_lower_bound']==(truncated[4]+truncated[5])/2
            assert agg[f'val{threshold}_updates']['median_is_identified']==(len(values)>=6)
    for threshold,pairs in summary['paired_speeds'].items():
        for pair,counts in pairs.items():
            assert sum(counts.values())==10 and counts['missing_runs']==0
    artifact=dict(all_passed=True,runs_checked=len(checked),method='Independent NumPy first-passage and convolution checks',
                  results=checked,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (REPORT/'independent_metrics_validation.json').write_text(json.dumps(artifact,indent=2)+'\n')
    print('Independent threshold/count verification passed for all 30 runs.')


if __name__=='__main__':main()
