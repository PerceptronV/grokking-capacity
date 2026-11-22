#!/usr/bin/env python3
"""Calibrate capacity constants, freeze matched pairs, and run confirmation seeds."""
from grokking_capacity.experiments.campaigns import runner
from grokking_capacity.experiments.campaigns.config import options, initialize, freeze_selection, check_registration
from pathlib import Path
import time
import math
import numpy as np
import torch
from ..registered import ROOT
from grokking_capacity.experiments.capacity_scaling import (
    RUN,read,sha,atomic_json,source_hashes,model_spec,parameter_count,fit_constant,match_pairs,audit_job,checked_artifact,RECIPE_KEYS)
from grokking_capacity.experiments.model_protocols import late_drift,make_model
from grokking_capacity.models.transformer import count_parameters


RESUME=False
AUDIT_DEVICE="cpu"

def run_stage(name, specs, devices):
    paths = runner.run_stage(RUN, name, specs, devices,
        'grokking_capacity.experiments.capacity_scaling', resume=RESUME,
        source_check=lambda: check_registration(RUN, source_hashes()), audit=audit)
    return paths


def audit(path):
    a=audit_job(path,device=AUDIT_DEVICE);atomic_json(RUN/'audits'/(path.name+'.json'),a);print('AUDITED '+path.name,flush=True)
    return a


def spec(common,role,width,seed,task,phase,**extra):
    config={**common,**model_spec(role,width),'role':role,'seed':seed,'task':task,'phase':phase,**extra}
    name=f'{role}_d{width}_{phase}'+(f"_n{config['n_samples']}" if task=='capacity' else '')+f'_s{seed}'
    return dict(name=name,config=config)


def diagnostic(path):
    s=read(path/'summary.json');c=s['config'];arr=np.load(path/'trace.npz',allow_pickle=False)
    drift=late_drift(arr['bits_trace']);entropy=c['n_samples']*math.log2(c['p']+2)
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
    global RUN, RESUME, AUDIT_DEVICE
    args=options('capacity_scaling',ROOT/'results/discrete/p97-capacity-scaling-v2')
    RUN=args.root;RESUME=args.resume;AUDIT_DEVICE=args.audit_device
    torch.set_num_threads(2)
    reg=initialize(RUN,args.config,source_hashes(),resume=RESUME)
    try:
        common={**reg['common'],'random_learning_size':reg['calibration']['random_learning_size']};devices=reg['resources']['devices'];calibration={};inherited={}
        for cell in reg['calibration']['inherited_cells']:
            calibration[cell['role'],cell['width'],cell['seed']]=[ROOT/p for p in cell['paths']]
        gates={(g['role'],g['width']):ROOT/g['path'] for g in reg['calibration']['inherited_gates']}
        for key,paths in calibration.items():
            for path in paths:
                s,m=checked_artifact(path);expected={**common,**model_spec(key[0],key[1])}
                assert all(s['config'].get(k)==expected.get(k) for k in RECIPE_KEYS)
                assert s['config']['seed']==key[2]
                a=audit(path);inherited[str(path)]=a['input_hashes']
        for path in gates.values():a=audit(path);inherited[str(path)]=a['input_hashes']
        atomic_json(RUN/'inherited.json',inherited)
        for seed in reg['calibration']['learning_seeds']:
            gate=run_stage(f"{reg['calibration']['learning_role']}{reg['calibration']['learning_width']}_learning_s{seed}",[spec(common,reg['calibration']['learning_role'],reg['calibration']['learning_width'],seed,'arithmetic','learning',stop_rule='learning_gate')],devices)[0]
            if read(gate/'summary.json')['learning_gate_passed']:gates[reg['calibration']['learning_role'],reg['calibration']['learning_width']]=gate;break
        assert (reg['calibration']['learning_role'],reg['calibration']['learning_width']) in gates,'Calibration model failed all configured arithmetic gates.'
        random_specs=[]
        for model in reg['calibration']['new_models']:
            role,width=model['role'],model['width']
            for seed in reg['calibration']['seeds']:
                key=role,width,seed;paths=calibration.setdefault(key,[])
                if not any(read(p/'summary.json')['config']['n_samples']==reg['calibration']['random_learning_size'] for p in paths):
                    random_specs.append(spec(common,role,width,seed,'capacity','capacity',n_samples=reg['calibration']['random_learning_size'],gate_path=str(gates[role,width])))
        random=run_stage('random_learning',random_specs,devices)
        for path in random:
            s=read(path/'summary.json');c=s['config'];assert s['random_learning_passed'],'Registered random-learning gate failed'
            calibration[c['role'],c['dim'],c['seed']].append(path)
        pending=[]
        for (role,width,seed),paths in calibration.items():
            if {'role':role,'width':width} not in reg['calibration']['new_models']:continue
            existing={read(p/'summary.json')['config']['n_samples'] for p in paths}
            rg=next(p for p in paths if read(p/'summary.json')['config']['n_samples']==reg['calibration']['random_learning_size'])
            for n in reg['calibration']['n_samples']:
                if n not in existing:pending.append(spec(common,role,width,seed,'capacity','capacity',
                    n_samples=n,gate_path=str(gates[role,width]),random_gate_path=str(rg)))
        new=run_stage('capacity_grid',pending,devices)
        for path in new:
            c=read(path/'summary.json')['config'];calibration[c['role'],c['dim'],c['seed']].append(path)
        for extension in range(reg['calibration']['extension_rounds']):
            assessments={f'{role}_d{width}_s{seed}':maxima(paths) for (role,width,seed),paths in calibration.items()}
            atomic_json(RUN/'calibration_diagnostics.json',assessments);pending=[]
            for (role,width,seed),paths in calibration.items():
                a=assessments[f'{role}_d{width}_s{seed}']
                if a['acceptable']:continue
                if a['n_samples']==max(a['grid']) and max(a['grid'])<reg['calibration']['maximum_dataset_size']:
                    rg=next(p for p in paths if read(p/'summary.json')['config']['n_samples']==reg['calibration']['random_learning_size'])
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
        freeze_selection(RUN/'selection.json',selection);print('FROZEN '+str([(p['C']['dim'],p['B']['dim']) for p in pairs]),flush=True)
        binding=sha(RUN/'selection.json');pending=[]
        for seed in reg['confirmation']['seeds']:
            for pair in reversed(pairs):
                for role in ['C','B']:pending.append(spec(common,role,pair[role]['dim'],seed,'arithmetic','confirmation',
                    stop_rule='joint99',pair_index=pair['index'],selection_sha256=binding))
        confirmations=run_stage('confirmation',pending,devices)
        assert len(confirmations)==2*len(pairs)*len(reg['confirmation']['seeds'])
        (RUN/'campaign_error.json').unlink(missing_ok=True)
        atomic_json(RUN/'campaign_complete.json',dict(finished_at=time.time(),confirmation_runs=len(confirmations),calibration_cells=sum(map(len,calibration.values())),
            new_jobs=len(list((RUN/'jobs').glob('*/summary.json'))),inherited_jobs=len(inherited),selected_pairs=len(pairs)))
    except Exception as exc:
        atomic_json(RUN/'campaign_error.json',dict(time=time.time(),error=repr(exc)));raise


if __name__=='__main__':main()
