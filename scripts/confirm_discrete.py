#!/usr/bin/env python3
"""Freeze and run paired modular confirmation only after the capacity match passes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
import yaml

from grokking_capacity.experiments.discrete import atomic_json, check_gate, run, source_hashes
from scripts.assess_discrete_match import assess, load_run
from scripts.discrete_source_compatibility import baseline_random_sources_compatible
from grokking_capacity.models.discrete import make_model
from grokking_capacity.models.transformer import count_parameters
from scripts.discrete_status import statuses


def process_live(pid, start_ticks):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split()
        return fields[21] == start_ticks and fields[2] != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False


def fullwidth_control(b):
    """Ordinary continuous C, with the nearest valid width by total parameters."""
    if b['model'] not in {'quantized', 'categorical'}:
        raise ValueError('Expected a full-width discrete model')
    extra = {'levels', 'alphabet', 'categorical_temperature', 'categorical_normalization', 'categorical_evaluation', 'quantizer_clip', 'quantizer_surrogate', 'latent_dim'}
    control = {key: value for key, value in b.items() if key not in extra}
    control['model'] = 'baseline'
    if b['model'] == 'categorical':
        target = count_parameters(make_model(b))
        depth, heads = b.get('depth', 2), b.get('heads', 1)
        project_out = int(heads != 1)
        quadratic = depth * (16 + project_out)
        linear = 2 * (b['p'] + 2) + depth * (2 + project_out) + 1
        if b.get('mlp_dim') is not None:
            quadratic = depth * (4 + project_out)
            linear += 3 * depth * b['mlp_dim']
        root = (math.sqrt(linear ** 2 + 4 * quadratic * target) - linear) / (2 * quadratic)
        multiple = 2 * heads  # even head width for RoPE
        lower = max(multiple, int(root // multiple) * multiple)
        widths = [lower, lower + multiple]
        control['dim'] = min(widths, key=lambda width: abs(quadratic * width ** 2 + linear * width - target))
        actual = count_parameters(make_model(control))
        if abs(actual / target - 1) > .017:
            raise ValueError('Nearest ordinary control exceeds the documented 1.7% parameter tolerance')
    return control


def confirmation_runs(a, b, *, seeds=(43, 44, 45)):
    """Use registered seeds, remove calibration-only fields, and rotate execution order."""
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError('Register three distinct nonnegative integer confirmation seeds')
    shared = ('p', 'operation', 'depth', 'heads', 'dropout', 'lr', 'weight_decay',
              'beta1', 'beta2', 'batch_size')
    common = {key: a[key] for key in shared}
    p = common['p']
    if (type(p) is not int or p < 2 or any(p % d == 0 for d in range(2, math.isqrt(p) + 1))
            or common['operation'] != '/'):
        raise ValueError('Confirmation requires prime-modulus division')
    if any(a.get(key) != b.get(key) for key in shared):
        raise ValueError('A/B task and optimizer must match before confirmation')
    common.update(task='modular', train_fraction=.5,
                  max_steps=max(60000, a.get('max_steps', 60000), b.get('max_steps', 60000)), eval_every=100,
                  checkpoint_every=2000, sustained_evals=3, stop_on_success=True)
    if b['model'] in {'quantized', 'categorical'}:
        keys = ('model', 'dim', 'mlp_dim', 'levels', 'alphabet', 'categorical_temperature', 'categorical_normalization', 'categorical_evaluation', 'quantizer_clip', 'quantizer_surrogate')
        discrete = {key: b[key] for key in keys if key in b}
        control = fullwidth_control(b)
        ordinary_keys = ('model', 'dim', 'mlp_dim')
        arms = dict(A={key: a[key] for key in ordinary_keys if key in a}, B=discrete,
                    C={key: control[key] for key in ordinary_keys if key in control})
    else:
        arms = dict(A=dict(model='baseline', dim=a['dim']),
                    B={key: b[key] for key in ('model', 'dim', 'latent_dim', 'levels')},
                    C={**{key: b[key] for key in ('dim', 'latent_dim', 'levels')}, 'model':'continuous'})
    return [(f'{arm}_s{seed}', {**common, **arms[arm], 'seed':seed})
            for seed, order in zip(seeds, ('ABC', 'BCA', 'CAB')) for arm in order]


def confirm(a_paths, b_paths, output, device, *, wait=False, resume=False, min_parameter_ratio=4., confirmation_seeds=(43, 44, 45)):
    inputs = dict(a_runs=[str(p.resolve()) for p in a_paths],
                  b_runs=[str(p.resolve()) for p in b_paths], device=str(device), min_parameter_ratio=min_parameter_ratio,
                  confirmation_seeds=list(confirmation_seeds),
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
        assessment = assess(a, b, p=a[0]['config']['p'], min_parameter_ratio=min_parameter_ratio)
        atomic_json(output/'capacity_assessment.json', assessment)
        if not assessment['accepted']:
            raise ValueError(f'Capacity match rejected: {assessment["issues"]}')
        if (not baseline_random_sources_compatible(a[0]['source_hashes'], source_hashes())
                or b[0]['source_hashes'] != source_hashes()):
            raise ValueError('Calibration and confirmation training sources differ')
        b_manifest = json.loads((b_paths[0]/'manifest.json').read_text())
        gate = Path(b_manifest['gate']['path'])
        if check_gate(gate, b[0]['config']) != b_manifest['gate']:
            raise ValueError('Arithmetic gate evidence changed')
        gate_config = json.loads((gate/'summary.json').read_text())['config']
        for key in ('lr', 'weight_decay', 'beta1', 'beta2', 'batch_size', 'dropout'):
            if any(row['config'].get(key) != gate_config.get(key) for row in b):
                raise ValueError(f'Calibration optimizer differs from the arithmetic gate: {key}')
        audit = json.loads((gate/'audit.json').read_text())
        if not audit['arithmetic_gate_passed']:
            raise ValueError('Discrete arithmetic audit has not passed')
        runs = confirmation_runs(a[0]['config'], b[0]['config'], seeds=confirmation_seeds)
        frozen = dict(assessment=assessment,
                      runs=[dict(name=name, **cfg) for name,cfg in runs],
                      artifact_hashes={str(p.resolve()):{
                          name:hashlib.sha256((p/name).read_bytes()).hexdigest()
                          for name in ['manifest.json','summary.json','trace.jsonl']}
                          for p in a_paths+b_paths})
        if b[0]['config']['model'] in {'quantized', 'categorical'}:
            control = fullwidth_control(b[0]['config'])
            c_params = count_parameters(make_model(control))
            frozen.update(control='fullwidth_unrestricted', b_params=b[0]['params'],
                          c_params=c_params, c_relative_parameter_difference=c_params/b[0]['params']-1,
                          comparison_note='C is an ordinary continuous transformer. Quantized B/C have identical tensors; categorical C uses the nearest ordinary width by total parameters.')
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
    parser.add_argument('--min-parameter-ratio',type=float,default=4.)
    parser.add_argument('--confirmation-seeds',type=int,nargs=3,default=(43,44,45))
    parser.add_argument('--wait',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    confirm(args.a_runs,args.b_runs,args.output,torch.device(args.device),wait=args.wait,resume=args.resume,min_parameter_ratio=args.min_parameter_ratio,confirmation_seeds=args.confirmation_seeds)
