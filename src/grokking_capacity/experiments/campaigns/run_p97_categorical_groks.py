#!/usr/bin/env python3
"""Durable C-first search; B starts only after both registered C groups grok."""
import json
from grokking_capacity.experiments.campaigns import runner
from grokking_capacity.experiments.campaigns.config import options, initialize, freeze_selection, check_registration
from pathlib import Path
import time
import torch
from ..registered import ROOT
from grokking_capacity.experiments.categorical_groks import (
    RUN,read,sha,atomic_json,source_hashes,models,qualify,audit_job)
from grokking_capacity.experiments.model_protocols import make_model
from grokking_capacity.models.transformer import count_parameters


RESUME=False
AUDIT_DEVICE="cpu"

def run_stage(name, specs, devices):
    paths = runner.run_stage(RUN, name, specs, devices,
        'grokking_capacity.experiments.categorical_groks', resume=RESUME,
        source_check=lambda: check_registration(RUN, source_hashes()), audit=None)
    return [(p, read(p/'summary.json')) for p in paths]


def specs(common,model,role,phase,seeds,**extra):
    return [dict(name=f"{role}_d{model['dim']}_f{model['mlp_dim']}_{phase}_s{seed}",
        config={**common,**model,'role':role,'phase':phase,'seed':seed,
            'task':'arithmetic','stop_rule':'joint99',**extra}) for seed in seeds]


def audit(items):
    for path,result in items:
        output=RUN/'audits'/(path.name+'.json')
        a=audit_job(path,device=AUDIT_DEVICE);a['input_hashes']={name:sha(path/name) for name in ['manifest.json','summary.json','model.pt','model_init.pt','trace.jsonl']}
        atomic_json(output,a);print('AUDITED '+path.name,flush=True)


def main():
    global RUN, RESUME, AUDIT_DEVICE
    args=options('categorical_groks',ROOT/'results/discrete/p97-categorical-groks-v2')
    RUN=args.root;RESUME=args.resume;AUDIT_DEVICE=args.audit_device
    torch.set_num_threads(2)
    reg=initialize(RUN,args.config,source_hashes(),resume=RESUME)
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
            freeze_selection(RUN/'selection.json',selected);break
        if selected is None:
            atomic_json(RUN/'search_exhausted.json',dict(finished_at=time.time(),attempts=len(attempts),reason='No continuous candidate passed both registered positive-delay groups.'))
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
        (RUN/'campaign_error.json').unlink(missing_ok=True)
        atomic_json(RUN/'campaign_complete.json',dict(finished_at=time.time(),selected_width=width,
            arithmetic_jobs=sum(len(read(p)['jobs']) for p in (RUN/'stages').glob('*/controller.json') if p.parent.name!='matched_memorisation'),
            memorisation_jobs=len(memjobs),new_jobs=len(list((RUN/'jobs').glob('*/summary.json')))))
    except Exception as exc:
        atomic_json(RUN/'campaign_error.json',dict(time=time.time(),error=repr(exc)));raise


if __name__=='__main__':main()
