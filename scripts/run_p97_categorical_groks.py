#!/usr/bin/env python3
"""Durable C-first search; B starts only after two ten-seed C groups grok."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from grokking_capacity.experiments.categorical_groks import (
    RUN,read,sha,atomic_json,source_hashes,models,qualify,audit_job)
from grokking_capacity.experiments.multilevel_dropout import make_model
from grokking_capacity.models.transformer import count_parameters


def run_stage(name,specs,devices):
    stage=RUN/'stages'/name;stage.mkdir(parents=True,exist_ok=False)
    atomic_json(stage/'jobs.json',specs)
    for spec in specs:
        assert not (RUN/'jobs'/spec['name']).exists()
        atomic_json(stage/(spec['name']+'.json'),spec['config'])
    state=dict(phase='running',pid=os.getpid(),started_at=time.time(),jobs=specs,completed=[],running=[],pending=[],errors=[])
    queue=list(specs);active={}
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    while queue or active:
        for slot,device in enumerate(devices):
            if slot in active or not queue or state['errors']:continue
            assert read(RUN/'manifest.json')['source_hashes']==source_hashes()
            spec=queue.pop(0);handle=(stage/(spec['name']+'.log')).open('x')
            cmd=[sys.executable,'-u','-m','grokking_capacity.experiments.categorical_groks',
                '--config',str(stage/(spec['name']+'.json')),'--output',str(RUN/'jobs'/spec['name']),'--device',device]
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            entry=dict(name=spec['name'],device=device,pid=proc.pid,started_at=time.time(),command=cmd)
            active[slot]=(proc,handle,entry)
        state.update(running=[v[2] for v in active.values()],pending=[s['name'] for s in queue])
        atomic_json(stage/'controller.json',state);time.sleep(5)
        for slot,(proc,handle,entry) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            handle.close();state['completed'].append({**entry,'exit_code':code,'finished_at':time.time()})
            if code:state['errors'].append(entry['name'])
            del active[slot]
        if state['errors']:
            state['not_launched_after_failure']=[s['name'] for s in queue];queue.clear()
    state.update(running=[],pending=[],phase='failed' if state['errors'] else 'complete',finished_at=time.time())
    atomic_json(stage/'controller.json',state);assert not state['errors'],state['errors']
    return [(RUN/'jobs'/s['name'],read(RUN/'jobs'/s['name']/'summary.json')) for s in specs]


def specs(common,model,role,phase,seeds,**extra):
    return [dict(name=f"{role}_d{model['dim']}_f{model['mlp_dim']}_{phase}_s{seed}",
        config={**common,**model,'role':role,'phase':phase,'seed':seed,
            'task':'arithmetic','stop_rule':'joint99',**extra}) for seed in seeds]


def audit(items):
    for path,result in items:
        output=RUN/'audits'/(path.name+'.json')
        a=audit_job(path);a['input_hashes']={name:sha(path/name) for name in ['manifest.json','summary.json','model.pt','model_init.pt','trace.jsonl']}
        atomic_json(output,a);print('AUDITED '+path.name,flush=True)


def main():
    torch.set_num_threads(2)
    assert not RUN.exists(),'Use a new root or explicitly recover an interrupted campaign; never overwrite.'
    RUN.mkdir(parents=True);(RUN/'jobs').mkdir();(RUN/'audits').mkdir()
    config_path=ROOT/'configs/discrete_p97_categorical_groks.yaml';reg=yaml.safe_load(config_path.read_text())
    shutil.copyfile(config_path,RUN/'registration.yaml');hashes=source_hashes()
    for name in hashes:
        dest=RUN/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    atomic_json(RUN/'manifest.json',dict(source_hashes=hashes,registration_sha256=sha(RUN/'registration.yaml'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),started_at=time.time(),pid=os.getpid()))
    try:
        common=reg['common'];devices=reg['resources']['devices'];attempts=[];selected=None
        for width in reg['search']['widths']:
            c,b=models(width,reg['categorical']['alphabet']);cp=count_parameters(make_model({**common,**c}));bp=count_parameters(make_model({**common,**b}))
            assert cp==bp
            search=run_stage(f'C_d{width}_search',specs(common,c,'C','search',reg['search']['seeds']),devices)
            audit(search);assessment=qualify([s for p,s in search],reg['search']['seeds'])
            attempt=dict(width=width,C=c,parameters=cp,search=assessment)
            attempts.append(attempt);atomic_json(RUN/'search_results.json',attempts)
            print('C SEARCH '+json.dumps(attempt),flush=True)
            if not assessment['passed']:continue
            validation=run_stage(f'C_d{width}_validation',specs(common,c,'C','validation',reg['search']['validation_seeds']),devices)
            audit(validation);confirmed=qualify([s for p,s in validation],reg['search']['validation_seeds'])
            attempt['validation']=confirmed;atomic_json(RUN/'search_results.json',attempts)
            print('C VALIDATION '+json.dumps(confirmed),flush=True)
            if not confirmed['passed']:continue
            evidence={str(p/name):sha(p/name) for p,s in search+validation for name in ['summary.json','manifest.json','model.pt','model_init.pt','trace.jsonl']}
            selected=dict(C=c,B=b,C_parameters=cp,B_parameters=bp,search=assessment,validation=confirmed,
                C_validation_jobs=[p.name for p,s in validation],evidence_sha256=evidence,
                registration_sha256=sha(RUN/'registration.yaml'),frozen_at=time.time())
            atomic_json(RUN/'selection.json',selected);break
        if selected is None:
            atomic_json(RUN/'search_exhausted.json',dict(finished_at=time.time(),attempts=len(attempts),reason='No continuous candidate passed both ten-seed positive-delay groups.'))
            return
        binding=sha(RUN/'selection.json')
        bjobs=run_stage('B_validation',specs(common,b,'B','validation',reg['B']['seeds'],selection_sha256=binding),devices)
        audit(bjobs);atomic_json(RUN/'B_results.json',qualify([s for p,s in bjobs],reg['B']['seeds']))
        memo=[]
        for role,model,arithmetic in [('C',c,validation),('B',b,bjobs)]:
            passed=[p for p,s in arithmetic if s['learning_gate_passed']]
            if not passed:
                atomic_json(RUN/f'{role}_memorisation_not_launched.json',dict(reason='No same-recipe arithmetic checkpoint passed its learning gate.'))
                continue
            memo+=specs(common,model,role,'memorisation',reg['memorisation']['seeds'],
                task='memorisation',n_samples=reg['memorisation']['n_samples'],gate_path=str(passed[0]),selection_sha256=binding)
        memjobs=run_stage('matched_memorisation',memo,devices);audit(memjobs)
        atomic_json(RUN/'campaign_complete.json',dict(finished_at=time.time(),selected_width=width,
            arithmetic_jobs=sum(len(read(p)['jobs']) for p in (RUN/'stages').glob('*/controller.json') if p.parent.name!='matched_memorisation'),
            memorisation_jobs=len(memjobs),new_jobs=len(list((RUN/'jobs').glob('*/summary.json')))))
    except Exception as exc:
        atomic_json(RUN/'campaign_error.json',dict(time=time.time(),error=repr(exc)));raise


if __name__=='__main__':main()
