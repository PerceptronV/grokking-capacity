#!/usr/bin/env python3
"""Independently verify the user's dataset -> matched A/B -> continuous C order."""
import argparse
import hashlib
import json
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[4]
RUN=ROOT/'results/discrete/p97-multilevel-dropout-v1'
OUT=ROOT/'reports/p97_multilevel_dropout'


def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def recipe(config):
    keys=('model','dim','mlp_dim','levels','quantizer_clip','quantizer_surrogate',
          'alphabet','categorical_temperature','categorical_normalization','categorical_training_sampling')
    return {key:config[key] for key in keys if key in config}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
    reg=yaml.safe_load((RUN/'registration.yaml').read_text())
    target=reg['fixed_dataset']['label_information_bits'];assert reg['common']['p']==97
    selections={};controls={}
    for family in ('q8','categorical'):
        s=read(RUN/f'{family}_AB_selection.json');c=read(RUN/f'{family}_C_construction.json')
        assert s['target_bits']==target and s['registration_sha256']==digest(RUN/'registration.yaml')
        assert c['AB_selection_sha256']==digest(RUN/f'{family}_AB_selection.json')
        assert c['constructed_at']>=s['frozen_at'] and c['model']['model']=='baseline'
        b=s['models']['B'];assert c['model']['dim']==b['dim']
        delta=0 if family=='q8' else round(2*b['alphabet']*b['dim']/3)
        assert c['model']['mlp_dim']==b['mlp_dim']+delta
        assert abs(c['param_count']/s['param_counts']['B']-1)<=.001
        for seed in ('61','62'):
            a,bb=s['evidence']['A'][seed]['final_bits'],s['evidence']['B'][seed]['final_bits']
            assert abs(a/target-1)<=.1 and abs(bb/target-1)<=.1 and abs(a-bb)/min(a,bb)<=.1
            for name in s['evidence']['B'][seed]['grid']:
                result=read(RUN/'jobs'/name/'summary.json')
                assert result['finished_at']<=s['frozen_at']
        for path,h in s['evidence_sha256'].items():assert digest(Path(path))==h
        selections[family]=s;controls[family]=c
    manifests=0;capacity_gates=0;c_after_freeze=0;confirmation=[];manifest_names=set()
    stage_jobs={};stage_states={}
    for path in sorted((RUN/'stages').glob('*/controller.json')):
        state=read(path);assert not state['errors'],path
        assert all(job['exit_code']==0 for job in state['completed']),path
        assert state['jobs']==read(path.parent/'jobs.json'),path
        for job in state['jobs']:
            assert job['name'] not in stage_jobs,job['name']
            stage_jobs[job['name']]=job['config']
        if args.require_complete:
            assert state['phase']=='complete' and not state['running'] and not state['pending'],path
            assert {j['name'] for j in state['completed']}=={j['name'] for j in state['jobs']},path
        stage_states[path.parent.name]=dict(phase=state['phase'],completed=len(state['completed']),
            running=len(state['running']),pending=len(state['pending']),controller_sha256=digest(path))
    for path in sorted((RUN/'jobs').glob('*/manifest.json')):
        m=read(path);c=m['config'];manifests+=1;manifest_names.add(path.parent.name)
        assert c==stage_jobs[path.parent.name]
        for key,v in reg['common'].items():assert c[key]==v,(path.parent.name,key)
        if c['task']=='capacity':
            gate=read(Path(c['gate_path'])/'summary.json')
            assert gate['learning_gate_passed'] and gate['finished_at']<=m['started_at']
            assert recipe(gate['config'])==recipe(c)
            assert all(gate['config'][key]==c[key] for key in reg['common'])
            if c['n_samples']>2048:
                rnd=read(Path(c['random_gate_path'])/'summary.json')
                assert rnd['config']['n_samples']==2048 and rnd['random_learning_passed']
                assert rnd['finished_at']<=m['started_at']
                assert recipe(rnd['config'])==recipe(c) and rnd['config']['seed']==c['seed']
                assert all(rnd['config'][key]==c[key] for key in reg['common'])
            capacity_gates+=1
        if c['role']=='C':
            family=c['family'];control=controls[family]
            assert m['started_at']>=control['constructed_at']
            assert all(c[k]==v for k,v in control['model'].items())
            assert m['param_count']==control['param_count'];c_after_freeze+=1
            if c['phase']!='confirmation':assert c['AB_selection_sha256']==control['AB_selection_sha256']
        if c['phase']=='confirmation':
            assert m['started_at']>=max(s['frozen_at'] for s in selections.values())
            assert c['seed'] in reg['confirmation']['seeds'];confirmation.append(c)
    if args.require_complete:
        assert manifest_names==set(stage_jobs)
        assert len(confirmation)==50
        assert sum(c['role']=='A' for c in confirmation)==10
    OUT.mkdir(parents=True,exist_ok=True)
    result=dict(all_passed=True,manifests_checked=manifests,capacity_gates_verified=capacity_gates,
        C_jobs_after_AB_freeze=c_after_freeze,confirmation_started=len(confirmation),complete=args.require_complete,
        fixed_target_bits=target,selection_before_C=True,all_confirmation_after_both_selections=True,
        stage_states=stage_states,verifier_sha256=digest(Path(__file__)))
    temp=OUT/'ordering_validation.tmp';temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(OUT/'ordering_validation.json')
    print(json.dumps({**{k:v for k,v in result.items() if k!='stage_states'},'stages_checked':len(stage_states)},indent=2))


if __name__=='__main__':main()
