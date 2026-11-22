#!/usr/bin/env python3
"""Advance registered learning gates/calibration without bypassing dataset-first selection.

Architecture additions remain explicit immutable requests. This controller never
invents a new recipe or selects from confirmation outcomes.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import torch
from grokking_capacity.experiments.discrete_workflow.manage_p97_multilevel import RUN, ROOT, TARGET, read, reg, jobs, submit, capacity_jobs, model_spec, capacity_evidence, freeze_ab, construct_c, confirmation_jobs, sha, atomic_json, assert_sources
from grokking_capacity.experiments.discrete_workflow.report_p97_multilevel import audit_job


def known_jobs():
    specs={};used={'cuda:0':0,'cuda:1':0}
    for path in (RUN/'requests').glob('*.json'):
        content=read(path)
        if isinstance(content,list):
            for spec in content:
                assert spec['name'] not in specs or specs[spec['name']]==spec
                specs[spec['name']]=spec
    for path in (RUN/'stages').glob('*/controller.json'):
        state=read(path)
        if state['errors']:raise RuntimeError(f"Failed stage {path.parent.name}: {state['errors']}")
        for entry in state['running']:
            try:os.kill(entry['pid'],0)
            except ProcessLookupError:continue
            used[entry['device']]=used.get(entry['device'],0)+1
    return specs,used


def verify_job(path,s):
    out=RUN/'audits';out.mkdir(exist_ok=True)
    artifact=out/(path.name+'.json');key=sha(path/'summary.json')+sha(path/'model.pt')
    if artifact.exists():
        assert read(artifact)['binding']==key;return
    result=audit_job(path,s);result['binding']=key
    atomic_json(artifact,result)


def tick():
    assert_sources();known,used=known_jobs();completed={p.name:(p,s) for p,s in jobs()}
    capacity=[(p,s) for p,s in completed.values() if s['task']=='capacity']
    pilots=[(p,s) for p,s in completed.values() if s['task']=='arithmetic' and s['config']['phase']=='development']
    registered=reg();planned=[];rejected=[];waiting=[]
    for p,s in pilots:
        c=s['config'];family=c['family'];stem=p.name.split('_learning_s')[0]
        if c['role']=='B' and (RUN/f'{family}_AB_selection.json').exists():continue
        if not s['learning_gate_passed']:
            if c['seed']==41:
                name=stem+'_learning_s42'
                if name not in known:
                    planned.append(dict(name=name,config={**c,'seed':42}))
            else:waiting.append(dict(candidate=stem,reason='Arithmetic gates failed at both development seeds.'))
            continue
        # The first successful development seed defines the gate. Duplicate
        # pilots never cause duplicate capacity cells or conflicting recipes.
        gate=p.name
        matching=[(q,v) for q,v in capacity if v['config']['family']==family and v['config']['role']==c['role']
                  and model_spec(v['config'])==model_spec(c)]
        over=[(q,v) for q,v in matching if v['final_bits']>1.1*TARGET]
        if c['role']=='B' and over:
            rejected.append(dict(candidate=stem,reason='Observed capacity already exceeds target+10%.',
                                 witnesses=[dict(job=q.name,bits=v['final_bits']) for q,v in over]))
            continue
        n2048=stem+'_capacity_n2048_s61'
        if n2048 not in completed:
            if n2048 not in known:
                verify_job(p,s);planned+=capacity_jobs(gate,[2048],[61])
            continue
        if not completed[n2048][1]['random_learning_passed']:
            waiting.append(dict(candidate=stem,reason='N2048 random-label gate failed; no larger datasets launched.'))
            continue
        screen=stem+'_capacity_n8192_s61'
        if screen not in completed:
            if screen not in known:planned+=capacity_jobs(gate,[8192],[61])
            continue
        # B exceeding the target was rejected above. Otherwise complete both
        # full grids, including the second seed's own random-learning gate.
        for seed in registered['capacity']['seeds']:
            rnd=stem+f'_capacity_n2048_s{seed}'
            if rnd not in completed:
                if rnd not in known:planned+=capacity_jobs(gate,[2048],[seed])
                continue
            if not completed[rnd][1]['random_learning_passed']:
                waiting.append(dict(candidate=stem,reason=f'N2048 seed{seed} failed.'));continue
            for n in registered['capacity']['n_samples']:
                name=stem+f'_capacity_n{n}_s{seed}'
                if name not in known:planned+=capacity_jobs(gate,[n],[seed])
        names=[stem+f'_capacity_n{n}_s{seed}' for seed in registered['capacity']['seeds'] for n in registered['capacity']['n_samples']]
        if c['role']=='B' and all(name in completed for name in names):
            try:
                evidence,_=capacity_evidence(model_spec(c),family)
                a=read(RUN/'inherited_A.json')
                assert all(abs(e['final_bits']-a['evidence'][seed]['final_bits'])/min(e['final_bits'],a['evidence'][seed]['final_bits'])<=.1 for seed,e in evidence.items())
            except AssertionError as exc:
                waiting.append(dict(candidate=stem,reason='Complete calibration does not pass registered match/convergence criteria.',detail=str(exc)))
            else:
                for name in names:verify_job(*completed[name])
                frozen=freeze_ab(family,gate)
                print('FROZEN '+family+' '+json.dumps(frozen['models']),flush=True)
    for family in registered['families']:
        if (RUN/f'{family}_AB_selection.json').exists() and not (RUN/f'{family}_C_construction.json').exists():
            control,cpilots=construct_c(family);planned+=cpilots
            print('CONSTRUCTED C '+family+' '+json.dumps(control['model']),flush=True)
    # Do not leave a constructed but not-yet-submitted C pilot stranded when
    # all slots were occupied on the construction tick.
    for family in registered['families']:
        path=RUN/f'{family}_C_construction.json'
        if path.exists():
            control=read(path);m=control['model'];stem=f"{family}_C_d{m['dim']}_f{m['mlp_dim']}"
            name=stem+'_learning_s41'
            if name not in known and not any(j['name']==name for j in planned):
                planned.append(dict(name=name,config={**registered['common'],**m,'family':family,'role':'C',
                    'phase':'development','task':'arithmetic','seed':41,'stop_rule':'learning_gate',
                    'AB_selection_sha256':control['AB_selection_sha256']}))
    ready=all((RUN/f'{family}_C_construction.json').exists() for family in registered['families'])
    if ready and not (RUN/'requests/confirmation.json').exists():
        for family in registered['families']:
            try:capacity_evidence(read(RUN/f'{family}_C_construction.json')['model'],family,strict=False)
            except AssertionError:ready=False
        # All preceding development/calibration jobs finish before confirmation;
        # no pilot is silently discarded on selecting a successful alternative.
        if ready and all(name in completed for name in known):
            for p,s in capacity:verify_job(p,s)
            confirm=confirmation_jobs()
            launch=submit('confirmation',confirm,devices='cuda:1,cuda:1,cuda:1,cuda:1,cuda:0,cuda:0')
            print('CONFIRMATION '+json.dumps(launch),flush=True)
    unique={j['name']:j for j in planned if j['name'] not in known}
    # Prefer shorter learning/random gates; capacity work follows when free.
    queue=sorted(unique.values(),key=lambda j:(j['config']['task']=='capacity',j['config'].get('n_samples',0),j['name']))
    launched=[]
    for device,limit in [('cuda:1',4),('cuda:0',2)]:
        free=max(0,limit-used.get(device,0))
        if free and queue:
            batch=queue[:free];queue=queue[free:]
            stage='auto_'+str(time.time_ns())
            launched.append(submit(stage,batch,devices=','.join([device]*len(batch))))
    confirm_done=sum(s['config']['phase']=='confirmation' for p,s in completed.values())
    status=dict(updated_at=time.time(),pid=os.getpid(),completed=len(completed),known=len(known),
        occupied=used,launched=launched,rejected=rejected,needs_development_review=waiting,
        pending_eligible=[j['name'] for j in queue],confirmation_completed=confirm_done,
        AB_frozen={family:(RUN/f'{family}_AB_selection.json').exists() for family in registered['families']})
    atomic_json(RUN/'advancement.json',status)
    return confirm_done==50,status


def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    torch.set_num_threads(2)
    manifest=RUN/'advancer_launch.json'
    if a.watch:
        assert not manifest.exists()
        atomic_json(manifest,dict(pid=os.getpid(),started_at=time.time(),source_sha256=sha(Path(__file__))))
    try:
        while True:
            done,status=tick()
            if done:
                from grokking_capacity.experiments.discrete_workflow.report_p97_multilevel import build
                build(require_complete=True,audit=True,plots=True)
                atomic_json(RUN/'campaign_complete.json',dict(finished_at=time.time(),confirmation_runs=50))
                break
            if not a.watch:
                print(json.dumps(status,indent=2));break
            time.sleep(10)
    except Exception as exc:
        atomic_json(RUN/'advancer_error.json',dict(time=time.time(),error=repr(exc)))
        raise


if __name__=='__main__':main()
