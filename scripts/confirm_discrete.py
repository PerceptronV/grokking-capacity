#!/usr/bin/env python3
"""Freeze and run paired modular confirmation only after the capacity match passes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import yaml

from grokking_capacity.experiments.discrete import atomic_json, check_gate, run, source_hashes
from scripts.assess_discrete_match import assess, load_run
from scripts.discrete_status import statuses


def process_live(pid, start_ticks):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split()
        return fields[21] == start_ticks and fields[2] != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False


def confirmation_runs(a, b):
    """Use new seeds, remove calibration-only fields, and rotate execution order."""
    shared = ('p', 'operation', 'depth', 'heads', 'dropout', 'lr', 'weight_decay',
              'beta1', 'beta2', 'batch_size')
    common = {key: a[key] for key in shared}
    if common['p'] != 149 or common['operation'] != '/':
        raise ValueError('The user fixed p149 modular division')
    common.update(task='modular', train_fraction=.5, max_steps=60000, eval_every=100,
                  checkpoint_every=2000, sustained_evals=3, stop_on_success=True)
    arms = dict(A=dict(model='baseline', dim=a['dim']),
                B={key: b[key] for key in ('model', 'dim', 'latent_dim', 'levels')},
                C={**{key: b[key] for key in ('dim', 'latent_dim', 'levels')}, 'model':'continuous'})
    return [(f'{arm}_s{seed}', {**common, **arms[arm], 'seed':seed})
            for seed, order in [(43,'ABC'),(44,'BCA'),(45,'CAB')] for arm in order]


def confirm(a_paths, b_paths, output, device, *, wait=False, resume=False):
    inputs = dict(a_runs=[str(p.resolve()) for p in a_paths],
                  b_runs=[str(p.resolve()) for p in b_paths], device=str(device),
                  source_hashes=source_hashes(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    state_path = output / 'orchestrator.json'
    if output.exists():
        if not resume:
            raise FileExistsError('Inspect the existing confirmation before --resume')
        old = json.loads(state_path.read_text())
        if old['inputs'] != inputs:
            raise ValueError('Confirmation inputs or implementation changed')
        if process_live(old['pid'], old['process_start_ticks']):
            raise RuntimeError('Original confirmation worker remains live')
    else:
        output.mkdir(parents=True)
    state = dict(inputs=inputs, pid=os.getpid(),
                 process_start_ticks=Path('/proc/self/stat').read_text().split()[21],
                 phase='waiting_for_calibration')
    atomic_json(state_path, state)
    try:
        while True:
            pending = [p for p in a_paths+b_paths if not (p/'summary.json').exists()]
            if not pending:
                break
            if not wait:
                raise ValueError(f'Incomplete calibration: {pending}')
            for p in pending:
                manifest = json.loads((p/'manifest.json').read_text())
                if not process_live(manifest['pid'], manifest['process_start_ticks']):
                    raise RuntimeError(f'Calibration stopped without completion: {p}')
            print(f'Waiting for {len(pending)} verified live calibration workers', flush=True)
            time.sleep(10)
        a, b = [load_run(p) for p in a_paths], [load_run(p) for p in b_paths]
        assessment = assess(a, b, p=149)
        atomic_json(output/'capacity_assessment.json', assessment)
        if not assessment['accepted']:
            raise ValueError(f'Capacity match rejected: {assessment["issues"]}')
        if a[0]['source_hashes'] != source_hashes():
            raise ValueError('Calibration and confirmation training sources differ')
        b_manifest = json.loads((b_paths[0]/'manifest.json').read_text())
        gate = Path(b_manifest['gate']['path'])
        if check_gate(gate, b[0]['config']) != b_manifest['gate']:
            raise ValueError('Arithmetic gate evidence changed')
        audit = json.loads((gate/'audit.json').read_text())
        if not audit['arithmetic_gate_passed']:
            raise ValueError('Discrete arithmetic audit has not passed')
        runs = confirmation_runs(a[0]['config'], b[0]['config'])
        frozen = dict(assessment=assessment,
                      runs=[dict(name=name, **cfg) for name,cfg in runs],
                      artifact_hashes={str(p.resolve()):{
                          name:hashlib.sha256((p/name).read_bytes()).hexdigest()
                          for name in ['manifest.json','summary.json','trace.jsonl']}
                          for p in a_paths+b_paths})
        freeze_path = output/'selection.json'
        if freeze_path.exists() and json.loads(freeze_path.read_text()) != frozen:
            raise ValueError('Frozen model selection or evidence changed')
        atomic_json(freeze_path, frozen)
        (output/'configuration.yaml').write_text(yaml.safe_dump({'runs':frozen['runs']}, sort_keys=False))
        state['phase'] = 'waiting_for_device'
        atomic_json(state_path, state)
        while True:
            busy = [r for r in statuses(output.parent) if r['status']=='running'
                    and r['device']==str(device) and r['pid']!=os.getpid()]
            if not busy:
                break
            if not wait:
                raise RuntimeError(f'Another experiment is using the confirmation GPU: {busy}')
            print(f'Waiting for other experiment workers on {device}', flush=True)
            time.sleep(10)
        state['phase']='confirming'
        atomic_json(state_path,state)
        for name, cfg in runs:
            state['current_run']=name
            atomic_json(state_path,state)
            run(cfg, output/name, device, resume=resume)
        state['phase']='complete'
        atomic_json(state_path,state)
    except Exception as error:
        state.update(phase='failed',error=str(error))
        atomic_json(state_path,state)
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a-runs',type=Path,nargs='+',required=True)
    parser.add_argument('--b-runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--wait',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    confirm(args.a_runs,args.b_runs,args.output,torch.device(args.device),wait=args.wait,resume=args.resume)
