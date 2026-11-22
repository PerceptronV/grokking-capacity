#!/usr/bin/env python3
"""Amend C to an ordinary wide transformer, preserving the frozen A/B experiment.

The user explicitly accepted C's 1.7% smaller parameter count on 2026-10-24.
Completed A/B runs are inherited verbatim; unfinished A/B runs must finish in
or be explicitly resumed from the original queue before this launcher proceeds.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch
import yaml

from grokking_capacity.experiments.discrete import atomic_json, check_gate, run, source_hashes
from grokking_capacity.models.discrete import make_model
from grokking_capacity.models.transformer import count_parameters
from .capacity import assess, load_run
from .confirmation import confirmation_runs, process_live


def unrestricted_runs(a,b):
    runs=confirmation_runs(a,b)
    return [(name, {key:value for key,value in {**cfg,'model':'baseline'}.items()
                   if key not in ('latent_dim','levels')} if name.startswith('C_') else cfg)
            for name,cfg in runs]


def shared_initial_tensors(b_config,c_config):
    torch.manual_seed(b_config['seed'])
    b=make_model(b_config)
    torch.manual_seed(c_config['seed'])
    c=make_model(c_config)
    return all(key in b.state_dict() and torch.equal(value,b.state_dict()[key])
               for key,value in c.state_dict().items())


def confirm(original,output,device,*,resume=False):
    old_state=json.loads((original/'orchestrator.json').read_text())
    if process_live(old_state['pid'],old_state['process_start_ticks']):
        raise RuntimeError('Original queue still live; preserve A/B before starting amended queue')
    old_selection=json.loads((original/'selection.json').read_text())
    a=[load_run(Path(p)) for p in old_state['inputs']['a_runs']]
    b=[load_run(Path(p)) for p in old_state['inputs']['b_runs']]
    assessment=assess(a,b,p=149)
    if not assessment['accepted'] or assessment!=old_selection['assessment']:
        raise ValueError('Original frozen capacity match changed')
    if source_hashes()!=old_state['inputs']['source_hashes']:
        raise ValueError('Scientific training sources changed')
    for directory,hashes in old_selection['artifact_hashes'].items():
        for name,value in hashes.items():
            if hashlib.sha256((Path(directory)/name).read_bytes()).hexdigest()!=value:
                raise ValueError('Frozen calibration evidence changed')
    manifest=json.loads((Path(b[0]['path'])/'manifest.json').read_text())
    if check_gate(Path(manifest['gate']['path']),b[0]['config'])!=manifest['gate']:
        raise ValueError('Original arithmetic gate changed')
    runs=unrestricted_runs(a[0]['config'],b[0]['config'])
    for name,cfg in runs:
        old=original/name
        if not name.startswith('C_') and old.exists() and not (old/'summary.json').exists():
            raise RuntimeError(f'Unfinished original A/B run must not be discarded: {old}')
    count_c=count_parameters(make_model(next(cfg for name,cfg in runs if name=='C_s43')))
    for seed in (43,44,45):
        paired={name[0]:cfg for name,cfg in runs if cfg['seed']==seed}
        if not shared_initial_tensors(paired['B'],paired['C']):
            raise ValueError('B and unrestricted C shared base tensors differ at initialisation')
    inputs=dict(a_runs=old_state['inputs']['a_runs'],b_runs=old_state['inputs']['b_runs'],
                original_confirmation=str(original.resolve()),device=str(device),source_hashes=source_hashes(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    frozen=dict(assessment=assessment,runs=[dict(name=name,**cfg) for name,cfg in runs],
                artifact_hashes=old_selection['artifact_hashes'],control='unrestricted',
                amendment='User requested ordinary width96 C without any channel; accepted 1.7% parameter difference.',
                original_selection_sha256=hashlib.sha256((original/'selection.json').read_bytes()).hexdigest(),
                c_params=count_c,b_params=b[0]['params'],c_relative_parameter_difference=count_c/b[0]['params']-1,
                comparison_note='A/B selection and seed set remain frozen. Corrected C is an explicitly amended control; old compressed C is auxiliary.')
    if output.exists():
        old=json.loads((output/'orchestrator.json').read_text())
        if not resume or old['inputs']!=inputs:
            raise ValueError('Existing amended queue requires unchanged inputs and explicit resume')
        if process_live(old['pid'],old['process_start_ticks']):
            raise RuntimeError('Amended queue already live')
        if json.loads((output/'selection.json').read_text())!=frozen:
            raise ValueError('Amended frozen selection changed')
    else:
        output.mkdir(parents=True)
        atomic_json(output/'selection.json',frozen)
        atomic_json(output/'capacity_assessment.json',assessment)
        (output/'configuration.yaml').write_text(yaml.safe_dump({'runs':frozen['runs']},sort_keys=False))
        for name,cfg in runs:
            path=original/name
            if not name.startswith('C_') and (path/'summary.json').exists():
                summary=json.loads((path/'summary.json').read_text())
                if summary['config']!=cfg:
                    raise ValueError(f'Original A/B configuration differs: {name}')
                shutil.copytree(path,output/name)
                atomic_json(output/name/'inherited_run.json',dict(path=str(path.resolve()),
                    artifact_hashes={file:hashlib.sha256((path/file).read_bytes()).hexdigest()
                                     for file in ('manifest.json','summary.json','trace.jsonl','checkpoint.pt')}))
    state=dict(inputs=inputs,pid=os.getpid(),process_start_ticks=Path('/proc/self/stat').read_text().split()[21],
               phase='confirming')
    atomic_json(output/'orchestrator.json',state)
    try:
        for name,cfg in runs:
            state['current_run']=name
            atomic_json(output/'orchestrator.json',state)
            run(cfg,output/name,device,resume=resume)
        state['phase']='complete'
        atomic_json(output/'orchestrator.json',state)
    except Exception as error:
        state.update(phase='failed',error=str(error))
        atomic_json(output/'orchestrator.json',state)
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    torch.set_num_threads(2)
    confirm(args.original,args.output,torch.device(args.device),resume=args.resume)


if __name__ == "__main__":
    main()
