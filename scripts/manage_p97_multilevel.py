#!/usr/bin/env python3
"""Gate-aware planning and immutable A/B selection; C is constructed afterward."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from grokking_capacity.experiments.multilevel_dropout import (
    atomic_json,sha,source_hashes,make_model,continuous_from_b,gate_check,late_drift)
from grokking_capacity.models.transformer import count_parameters
RUN=ROOT/'results/discrete/p97-multilevel-dropout-v1'
TARGET=4656*np.log2(99)
MODEL_KEYS=('model','dim','mlp_dim','levels','quantizer_clip','quantizer_surrogate',
            'alphabet','categorical_temperature','categorical_normalization','categorical_training_sampling')


def read(path):return json.loads(Path(path).read_text())
def reg():return yaml.safe_load((RUN/'registration.yaml').read_text())
def model_spec(c):return {k:c[k] for k in MODEL_KEYS if k in c}
def jobs():return [(p.parent,read(p)) for p in sorted((RUN/'jobs').glob('*/summary.json'))]
def assert_sources():
    assert read(RUN/'manifest.json')['source_hashes']==source_hashes()
    assert sha(RUN/'registration.yaml')==read(RUN/'manifest.json')['registration_sha256']


def verify_inherited_a():
    a=read(RUN/'inherited_A.json')
    for path,h in a['files_sha256'].items():assert sha(path)==h,path
    for path,h in a['prior_source_hashes'].items():assert sha(ROOT/path)==h,path
    for seed,e in a['evidence'].items():
        assert abs(e['final_bits']/TARGET-1)<=.1
        assert not any(e['diagnostics'][k] for k in ('epoch_limit','late_drift_flag','entropy_limited','largest_dataset_boundary'))
    return a


def submit(stage,specs,devices='cuda:1,cuda:1,cuda:1,cuda:1'):
    assert_sources();assert specs
    request=RUN/'requests'/(stage+'.json');assert not request.exists()
    assert not (RUN/'stages'/stage).exists()
    for j in specs:assert not (RUN/'jobs'/j['name']).exists()
    with request.open('x') as f:json.dump(specs,f,indent=2);f.write('\n')
    log=request.with_suffix('.launch.log').open('x')
    cmd=[sys.executable,'-u',str(ROOT/'scripts/run_p97_multilevel_stage.py'),
         '--stage',stage,'--jobs',str(request),'--devices',devices]
    p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close();record=dict(pid=p.pid,command=cmd,started_at=time.time(),jobs=[j['name'] for j in specs],
                            planner_sha256=sha(Path(__file__)))
    atomic_json(request.with_suffix('.launch.json'),record);return record


def capacity_jobs(gate_name,ns,seeds):
    gate=RUN/'jobs'/gate_name;s=read(gate/'summary.json');c=s['config']
    gate_check(gate,c)
    stem=gate_name.split('_learning_s')[0];answer=[]
    for seed in seeds:
        for n in ns:
            name=f'{stem}_capacity_n{n}_s{seed}'
            conf={**c,'task':'capacity','phase':'capacity','seed':seed,'n_samples':n,'gate_path':str(gate)}
            conf.pop('stop_rule',None)
            if n>2048:
                rand=RUN/'jobs'/f'{stem}_capacity_n2048_s{seed}'
                assert read(rand/'summary.json')['random_learning_passed']
                conf['random_gate_path']=str(rand)
            answer.append(dict(name=name,config=conf))
    return answer


def diagnostics(path,s):
    trace=np.load(path/'trace.npz');bits=trace['bits_trace']
    assert len(bits)==s['epochs']
    drift=late_drift(bits)
    if 'primary_epochs' in s:
        primary=np.load(path/'primary_trace.npz')
        for key in ('train_loss_trace','train_acc_trace','bits_trace'):
            np.testing.assert_array_equal(trace[key][:s['primary_epochs']],primary[key])
    return dict(late_drift=drift,epoch_limit=s['epochs']>=s['config']['epochs'],
                late_drift_flag=drift is None or drift>.01,
                entropy_fraction=s['final_bits']/(s['config']['n_samples']*np.log2(99)),
                entropy_limited=s['final_bits']>=.95*s['config']['n_samples']*np.log2(99))


def capacity_evidence(model,family,strict=True):
    evidence={};files={}
    for seed in reg()['capacity']['seeds']:
        cells=[(p,s) for p,s in jobs() if s['task']=='capacity' and s['config']['family']==family
               and s['config']['seed']==seed and model_spec(s['config'])==model]
        assert sorted(s['config']['n_samples'] for p,s in cells)==reg()['capacity']['n_samples'],[(p.name,s['config']['n_samples']) for p,s in cells]
        p,s=max(cells,key=lambda ps:ps[1]['final_bits']);d=diagnostics(p,s)
        d['largest_dataset_boundary']=s['config']['n_samples']==max(v['config']['n_samples'] for _,v in cells)
        if strict:
            assert not any(d[k] for k in ('epoch_limit','late_drift_flag','entropy_limited','largest_dataset_boundary')),d
            assert abs(s['final_bits']/TARGET-1)<=.1,s['final_bits']
        evidence[str(seed)]=dict(final_bits=s['final_bits'],param_count=s['param_count'],winner=p.name,
            grid=[p.name for p,s in cells],diagnostics=d,
            lower_bound_only=any(d[k] for k in ('epoch_limit','late_drift_flag','entropy_limited','largest_dataset_boundary')))
        for cell,value in cells:
            assert sha(cell/'model.pt')==value['checkpoint_sha256']
            assert read(cell/'manifest.json')['source_hashes']==source_hashes()
            gate_check(value['config']['gate_path'],value['config'])
            for name in ('summary.json','manifest.json','model.pt','model_init.pt','trace.npz','primary_trace.npz'):
                files[str(cell/name)]=sha(cell/name)
            gate=Path(value['config']['gate_path'])
            for name in ('summary.json','manifest.json','model.pt','trace.jsonl'):files[str(gate/name)]=sha(gate/name)
    return evidence,files


def freeze_ab(family,gate_name):
    assert_sources();a=verify_inherited_a();out=RUN/f'{family}_AB_selection.json'
    assert not out.exists()
    assert not any(s['config']['phase']=='confirmation' for p,s in jobs())
    gate=RUN/'jobs'/gate_name;pilot=read(gate/'summary.json');b=model_spec(pilot['config'])
    assert pilot['config']['family']==family and pilot['config']['role']=='B'
    gate_check(gate,pilot['config']);evidence,files=capacity_evidence(b,family)
    for seed,e in evidence.items():
        av=a['evidence'][seed]['final_bits'];bv=e['final_bits']
        assert abs(av-bv)/min(av,bv)<=.1,(seed,av,bv)
        assert e['param_count']>a['param_count']
    value=dict(frozen_at=time.time(),family=family,target_bits=float(TARGET),
        models={'A':a['models'],'B':b},param_counts={'A':a['param_count'],'B':pilot['param_count']},
        evidence={'A':a['evidence'],'B':evidence},evidence_sha256={**a['files_sha256'],**files},
        registration_sha256=sha(RUN/'registration.yaml'),arithmetic_gate=str(gate),
        rule='Fixed dataset first, A/B selected from gates and capacity only; no C constructed yet.')
    with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    return value


def construct_c(family):
    selection=read(RUN/f'{family}_AB_selection.json');assert_sources()
    for path,h in selection['evidence_sha256'].items():assert sha(path)==h
    b=selection['models']['B'];c=continuous_from_b(b)
    bm=make_model({**reg()['common'],**b});cm=make_model({**reg()['common'],**c})
    bp,cp=count_parameters(bm),count_parameters(cm)
    assert bp==selection['param_counts']['B'] and abs(cp/bp-1)<=.001
    record=dict(family=family,constructed_at=time.time(),AB_selection_sha256=sha(RUN/f'{family}_AB_selection.json'),
                model=c,param_count=cp,B_param_count=bp,parameter_relative_difference=cp/bp-1,
                rule=reg()['C']['scalar' if family=='q8' else 'categorical'])
    with (RUN/f'{family}_C_construction.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    stem=f"{family}_C_d{c['dim']}_f{c['mlp_dim']}"
    conf={**reg()['common'],**c,'family':family,'role':'C','phase':'development','task':'arithmetic',
          'seed':41,'stop_rule':'learning_gate','AB_selection_sha256':record['AB_selection_sha256']}
    return record,[dict(name=stem+'_learning_s41',config=conf)]


def confirmation_jobs():
    assert_sources();a=verify_inherited_a();models={};bindings={}
    for family in reg()['families']:
        selection=read(RUN/f'{family}_AB_selection.json');c=read(RUN/f'{family}_C_construction.json')
        assert c['AB_selection_sha256']==sha(RUN/f'{family}_AB_selection.json')
        for path,h in selection['evidence_sha256'].items():assert sha(path)==h
        # C is measured after AB selection; missing excess is reported, not used to resize B.
        evidence,files=capacity_evidence(c['model'],family,strict=False)
        binding=dict(AB_selection_sha256=sha(RUN/f'{family}_AB_selection.json'),
                     C_construction_sha256=sha(RUN/f'{family}_C_construction.json'),C_evidence=evidence,
                     C_exceeds_target={seed:e['final_bits']>1.1*TARGET for seed,e in evidence.items()},
                     C_evidence_sha256=files,frozen_at=time.time())
        with (RUN/f'{family}_confirmation_binding.json').open('x') as f:json.dump(binding,f,indent=2);f.write('\n')
        models[family]={'B':selection['models']['B'],'C':c['model']};bindings[family]=sha(RUN/f'{family}_confirmation_binding.json')
    result=[]
    for seed in reg()['confirmation']['seeds']:
        result.append(dict(name=f'A_shared_confirmation_s{seed}',config={**reg()['common'],**a['models'],
            'family':'shared','role':'A','phase':'confirmation','task':'arithmetic','seed':seed,
            'stop_rule':'joint99','bindings':bindings}))
        for family in models:
            for role,model in models[family].items():
                result.append(dict(name=f'{family}_{role}_confirmation_s{seed}',config={**reg()['common'],**model,
                    'family':family,'role':role,'phase':'confirmation','task':'arithmetic','seed':seed,
                    'stop_rule':'joint99','binding_sha256':bindings[family]}))
    return result


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    cap=sub.add_parser('capacity');cap.add_argument('--gate',required=True);cap.add_argument('--ns',nargs='+',type=int,required=True)
    cap.add_argument('--seeds',nargs='+',type=int,default=[61]);cap.add_argument('--stage',required=True)
    freeze=sub.add_parser('freeze');freeze.add_argument('--family',required=True);freeze.add_argument('--gate',required=True)
    c=sub.add_parser('construct-c');c.add_argument('--family',required=True)
    sub.add_parser('confirmation');sub.add_parser('status')
    args=p.parse_args()
    if args.action=='capacity':value=submit(args.stage,capacity_jobs(args.gate,args.ns,args.seeds))
    elif args.action=='freeze':value=freeze_ab(args.family,args.gate)
    elif args.action=='construct-c':
        value,j=construct_c(args.family);value['launch']=submit(args.family+'_C_learning',j)
    elif args.action=='confirmation':value=submit('confirmation',confirmation_jobs())
    else:
        value=[]
        for path,s in jobs():
            c=s['config'];value.append(dict(job=path.name,role=c['role'],task=s['task'],epochs=s['epochs'],
                bits=s.get('final_bits'),final=s.get('final_eval',s.get('final')),passed=s.get('learning_gate_passed',s.get('random_learning_passed'))))
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
