#!/usr/bin/env python3
"""Calibrate two constants, freeze ten pairs, and complete 200 paired runs."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import math
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from grokking_capacity.experiments.capacity_scaling import (
    RUN,read,sha,atomic_json,source_hashes,model_spec,parameter_count,fit_constant,match_pairs,audit_job,checked_artifact,RECIPE_KEYS)
from grokking_capacity.experiments.multilevel_dropout import late_drift,make_model
from grokking_capacity.models.transformer import count_parameters


def run_stage(name,specs,devices):
    stage=RUN/'stages'/name;stage.mkdir(parents=True,exist_ok=False)
    registry=read(RUN/'job_registry.json')
    for spec in specs:
        assert spec['name'] not in registry;registry[spec['name']]=spec['config']
        assert not (RUN/'jobs'/spec['name']).exists()
        atomic_json(stage/(spec['name']+'.json'),spec['config'])
    atomic_json(RUN/'job_registry.json',registry);atomic_json(stage/'jobs.json',specs)
    state=dict(phase='running',pid=os.getpid(),started_at=time.time(),jobs=specs,completed=[],running=[],pending=[],errors=[])
    queue=list(specs);active={};env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    while queue or active:
        for slot,device in enumerate(devices):
            if slot in active or not queue or state['errors']:continue
            assert read(RUN/'manifest.json')['source_hashes']==source_hashes()
            spec=queue.pop(0);handle=(stage/(spec['name']+'.log')).open('x')
            cmd=[sys.executable,'-u','-m','grokking_capacity.experiments.capacity_scaling',
                '--config',str(stage/(spec['name']+'.json')),'--output',str(RUN/'jobs'/spec['name']),'--device',device]
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            entry=dict(name=spec['name'],device=device,pid=proc.pid,started_at=time.time(),command=cmd)
            active[slot]=(proc,handle,entry)
        state.update(running=[v[2] for v in active.values()],pending=[s['name'] for s in queue]);atomic_json(stage/'controller.json',state)
        time.sleep(5)
        for slot,(proc,handle,entry) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            handle.close();state['completed'].append({**entry,'exit_code':code,'finished_at':time.time()})
            if code:state['errors'].append(entry['name'])
            del active[slot]
        if state['errors']:state['not_launched_after_failure']=[s['name'] for s in queue];queue.clear()
    state.update(running=[],pending=[],phase='failed' if state['errors'] else 'complete',finished_at=time.time())
    atomic_json(stage/'controller.json',state);assert not state['errors'],state['errors']
    paths=[RUN/'jobs'/s['name'] for s in specs]
    for p in paths:audit(p)
    return paths


def audit(path):
    a=audit_job(path);atomic_json(RUN/'audits'/(path.name+'.json'),a);print('AUDITED '+path.name,flush=True)
    return a


def spec(common,role,width,seed,task,phase,**extra):
    config={**common,**model_spec(role,width),'role':role,'seed':seed,'task':task,'phase':phase,**extra}
    name=f'{role}_d{width}_{phase}'+(f"_n{config['n_samples']}" if task=='capacity' else '')+f'_s{seed}'
    return dict(name=name,config=config)


def diagnostic(path):
    s=read(path/'summary.json');c=s['config'];arr=np.load(path/'trace.npz',allow_pickle=False)
    drift=late_drift(arr['bits_trace']);entropy=c['n_samples']*math.log2(99)
    return dict(path=str(path),seed=c['seed'],dim=c['dim'],parameters=s['param_count'],n_samples=c['n_samples'],
        capacity_bits=s['final_bits'],epochs=s['epochs'],late_drift=drift,
        entropy_fraction=s['final_bits']/entropy,at_epoch_limit=s['epochs']>=c['epochs'])


def maxima(paths):
    rows=[diagnostic(p) for p in paths];winner=max(rows,key=lambda r:r['capacity_bits'])
    grid=sorted(r['n_samples'] for r in rows)
    return {**winner,'grid':grid,'interior':grid[0]<winner['n_samples']<grid[-1],
        'acceptable':bool(grid[0]<winner['n_samples']<grid[-1] and winner['entropy_fraction']<.95
            and not winner['at_epoch_limit'] and winner['late_drift'] is not None and winner['late_drift']<=.01)}


def main():
    torch.set_num_threads(2);assert not RUN.exists(),'Never overwrite a campaign root.'
    RUN.mkdir(parents=True);(RUN/'jobs').mkdir();(RUN/'audits').mkdir()
    registration=ROOT/'configs/discrete_p97_capacity_scaling.yaml';reg=yaml.safe_load(registration.read_text())
    shutil.copyfile(registration,RUN/'registration.yaml');hashes=source_hashes()
    for name in hashes:
        p=RUN/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
    atomic_json(RUN/'manifest.json',dict(source_hashes=hashes,registration_sha256=sha(RUN/'registration.yaml'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),started_at=time.time(),pid=os.getpid()))
    atomic_json(RUN/'job_registry.json',{})
    try:
        common=reg['common'];devices=reg['resources']['devices'];calibration={};inherited={}
        old=ROOT/'results/discrete/p97-dropout-match-v1/jobs';multi=ROOT/'results/discrete/p97-multilevel-dropout-v1/jobs'
        for seed in reg['calibration']['seeds']:
            calibration['C',32,seed]=[old/(f'A32f128_random_n2048_s{seed}' if n==2048 else f'A32f128_capacity_n{n}_s{seed}')
                for n in reg['calibration']['existing_C32_grid']]
        calibration['B',48,61]=[multi/f'categorical_d48_f192_capacity_n{n}_s61' for n in [2048,8192]]
        gates={('B',48):multi/'categorical_d48_f192_learning_s41',
            ('B',60):ROOT/'results/discrete/p97-categorical-groks-v1/jobs/B_d60_f240_validation_s321'}
        for key,paths in calibration.items():
            for path in paths:
                s,m=checked_artifact(path);expected={**common,**model_spec(key[0],key[1])}
                assert all(s['config'].get(k)==expected.get(k) for k in RECIPE_KEYS)
                assert s['config']['seed']==key[2]
                a=audit(path);inherited[str(path)]=a['input_hashes']
        for path in gates.values():a=audit(path);inherited[str(path)]=a['input_hashes']
        atomic_json(RUN/'inherited.json',inherited)
        for seed in [41,42]:
            gate=run_stage(f'C48_learning_s{seed}',[spec(common,'C',48,seed,'arithmetic','learning',stop_rule='learning_gate')],devices)[0]
            if read(gate/'summary.json')['learning_gate_passed']:gates['C',48]=gate;break
        assert ('C',48) in gates,'C48 failed both arithmetic gates; no capacity claim.'
        random_specs=[]
        for role,width in [('C',48),('B',48),('B',60)]:
            for seed in reg['calibration']['seeds']:
                key=role,width,seed;paths=calibration.setdefault(key,[])
                if not any(read(p/'summary.json')['config']['n_samples']==2048 for p in paths):
                    random_specs.append(spec(common,role,width,seed,'capacity','capacity',n_samples=2048,gate_path=str(gates[role,width])))
        random=run_stage('random_learning',random_specs,devices)
        for path in random:
            s=read(path/'summary.json');c=s['config'];assert s['random_learning_passed'],'N2048 random learning failed'
            calibration[c['role'],c['dim'],c['seed']].append(path)
        pending=[]
        for (role,width,seed),paths in calibration.items():
            if role=='C' and width==32:continue
            existing={read(p/'summary.json')['config']['n_samples'] for p in paths}
            rg=next(p for p in paths if read(p/'summary.json')['config']['n_samples']==2048)
            for n in reg['calibration']['n_samples']:
                if n not in existing:pending.append(spec(common,role,width,seed,'capacity','capacity',
                    n_samples=n,gate_path=str(gates[role,width]),random_gate_path=str(rg)))
        new=run_stage('capacity_grid',pending,devices)
        for path in new:
            c=read(path/'summary.json')['config'];calibration[c['role'],c['dim'],c['seed']].append(path)
        for extension in range(3):
            assessments={f'{role}_d{width}_s{seed}':maxima(paths) for (role,width,seed),paths in calibration.items()}
            atomic_json(RUN/'calibration_diagnostics.json',assessments);pending=[]
            for (role,width,seed),paths in calibration.items():
                a=assessments[f'{role}_d{width}_s{seed}']
                if a['acceptable']:continue
                if a['n_samples']==max(a['grid']) and max(a['grid'])<131072:
                    rg=next(p for p in paths if read(p/'summary.json')['config']['n_samples']==2048)
                    gp=gates.get((role,width),Path(read(rg/'summary.json')['config']['gate_path']))
                    pending.append(spec(common,role,width,seed,'capacity','capacity',n_samples=2*max(a['grid']),gate_path=str(gp),random_gate_path=str(rg)))
            if not pending:break
            new=run_stage(f'capacity_extension_{extension}',pending,devices)
            for path in new:
                c=read(path/'summary.json')['config'];calibration[c['role'],c['dim'],c['seed']].append(path)
        assessments={f'{role}_d{width}_s{seed}':maxima(paths) for (role,width,seed),paths in calibration.items()}
        atomic_json(RUN/'calibration_diagnostics.json',assessments)
        assert all(a['acceptable'] for a in assessments.values()),'Calibration maxima failed registered quality checks'
        fits={role:fit_constant([v for key,v in assessments.items() if key.startswith(role+'_')]) for role in ['C','B']}
        atomic_json(RUN/'capacity_constants.json',fits)
        search=reg['matching']['B_width_candidates'];pairs=match_pairs({r:f['bits_per_parameter'] for r,f in fits.items()},
            reg['matching']['C_widths'],range(search['start'],search['stop']+1,search['step']))
        for pair in pairs:
            for role in ['C','B']:assert count_parameters(make_model({**common,**pair[role]}))==pair[role+'_parameters']
        evidence={str(p/name):sha(p/name) for paths in calibration.values() for p in paths for name in ['manifest.json','summary.json','model.pt','model_init.pt','trace.npz']}
        selection=dict(frozen_at=time.time(),pairs=pairs,constants=fits,registration_sha256=sha(RUN/'registration.yaml'),evidence_sha256=evidence,
            calibration_paths={f'{r}_d{w}_s{s}':[str(p) for p in paths] for (r,w,s),paths in calibration.items()})
        atomic_json(RUN/'selection.json',selection);print('FROZEN '+str([(p['C']['dim'],p['B']['dim']) for p in pairs]),flush=True)
        binding=sha(RUN/'selection.json');pending=[]
        for seed in reg['confirmation']['seeds']:
            for pair in reversed(pairs):
                for role in ['C','B']:pending.append(spec(common,role,pair[role]['dim'],seed,'arithmetic','confirmation',
                    stop_rule='joint99',pair_index=pair['index'],selection_sha256=binding))
        confirmations=run_stage('confirmation',pending,devices)
        assert len(confirmations)==200
        atomic_json(RUN/'campaign_complete.json',dict(finished_at=time.time(),confirmation_runs=200,calibration_cells=sum(map(len,calibration.values())),
            new_jobs=len(list((RUN/'jobs').glob('*/summary.json'))),inherited_jobs=len(inherited),selected_pairs=10))
    except Exception as exc:
        atomic_json(RUN/'campaign_error.json',dict(time=time.time(),error=repr(exc)));raise


if __name__=='__main__':main()
