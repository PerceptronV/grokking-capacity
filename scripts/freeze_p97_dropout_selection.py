#!/usr/bin/env python3
"""Freeze validated architecture selection before the ten confirmation seeds."""
import argparse
import json
from pathlib import Path
import sys
import time
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_p97_dropout_match import RUN, TARGET, build, read, recipe, capacity_diagnostics, sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--A',required=True);parser.add_argument('--B',required=True)
    args=parser.parse_args();assert not (RUN/'selection.json').exists()
    reg=yaml.safe_load((RUN/'registration.yaml').read_text());jobs,summary=build()
    assert not any(s['config']['phase']=='confirmation' for p,s in jobs)
    named={p.name:(p,s) for p,s in jobs}
    chosen={role:named[name] for role,name in [('A',args.A),('B',args.B)]}
    A=chosen['A'][1]['config'];B=chosen['B'][1]['config']
    assert A['model']=='baseline' and B['model']=='quantized'
    keys=('model','dim','mlp_dim','levels','quantizer_clip','quantizer_surrogate')
    models={role:{k:v for k,v in chosen[role][1]['config'].items() if k in keys} for role in 'AB'}
    models['C']={k:v for k,v in models['B'].items() if k in ('dim','mlp_dim')};models['C']['model']='baseline'
    assert chosen['B'][1]['param_count']/chosen['A'][1]['param_count']>=reg['capacity']['minimum_B_to_A_parameter_ratio']
    evidence={};counts={};hashes={}
    for role,model in models.items():
        evidence[role]={};counts[role]=None
        for seed in reg['capacity']['seeds']:
            cells=[(p,s) for p,s in jobs if s['task']=='capacity' and s['config']['seed']==seed
                   and all(s['config'].get(k)==v for k,v in model.items())]
            assert len({s['config']['n_samples'] for p,s in cells})==len(cells),'duplicate capacity cells'
            assert set(reg['capacity']['n_samples']).issubset({s['config']['n_samples'] for p,s in cells})
            path,winner=max(cells,key=lambda ps:ps[1]['final_bits'])
            d=capacity_diagnostics(path,winner)
            assert not any(d[k] for k in ('epoch_limit','late_drift_flag','entropy_limited')),d
            assert winner['config']['n_samples']<max(s['config']['n_samples'] for p,s in cells)
            if role in 'AB':assert abs(winner['final_bits']/TARGET-1)<=reg['capacity']['target_tolerance']
            else:assert winner['final_bits']>TARGET*(1+reg['capacity']['target_tolerance'])
            evidence[role][str(seed)]={k:winner[k] for k in ('final_bits','param_count')}
            evidence[role][str(seed)].update(winner=path.name,grid=[p.name for p,s in cells],diagnostics=d)
            counts[role]=winner['param_count']
            for p,s in cells:hashes[str(p/'summary.json')]=sha(p/'summary.json')
    for seed in reg['capacity']['seeds']:
        a=evidence['A'][str(seed)]['final_bits'];b=evidence['B'][str(seed)]['final_bits']
        assert abs(a-b)/min(a,b)<=reg['capacity']['pair_tolerance']
    assert counts['B']==counts['C']
    selected=dict(frozen_at=time.time(),models=models,param_counts=counts,evidence=evidence,
                  target_bits=float(TARGET),registration_sha256=sha(RUN/'registration.yaml'),
                  evidence_sha256=hashes,rule='Capacity and learning gates only; all confirmation seeds unseen.')
    with (RUN/'selection.json').open('x') as f:json.dump(selected,f,indent=2);f.write('\n')
    print(json.dumps(selected,indent=2))


if __name__=='__main__':main()
