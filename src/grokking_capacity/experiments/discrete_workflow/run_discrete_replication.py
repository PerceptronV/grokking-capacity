#!/usr/bin/env python3
"""Complete one registered A/B/C candidate after its arithmetic gate passes.

Architectures and capacity grids come from YAML; failed calibration stops the
pipeline before confirmation. Separate processes isolate paired RNG streams.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from grokking_capacity.experiments.discrete import atomic_json, check_gate
from grokking_capacity.experiments.discrete_workflow.confirmation import process_live


def configured_paths(config, output):
    return [output / item['name'] for item in yaml.safe_load(config.read_text())['runs']]


def replicate(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / 'pipeline.json'
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if process_live(previous['pid'], previous['process_start_ticks']):
            raise RuntimeError('Original replication worker is live')
        if not args.resume:
            raise ValueError('Inspect the existing replication before --resume')
    state = dict(pid=os.getpid(), process_start_ticks=Path('/proc/self/stat').read_text().split()[21],
                 phase='checking_arithmetic_gate', commands=[], arguments={k: str(v) for k,v in vars(args).items()})
    atomic_json(state_path, state)
    env = {**os.environ, 'PYTHONPATH':'src:.', 'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2'}

    def stage(name, jobs):
        state.update(phase=name, workers=[])
        atomic_json(state_path, state)
        workers = []
        for label, command in jobs:
            log_path = root / f'{name}_{label}.log'
            with log_path.open('a' if args.resume else 'x') as log:
                proc = subprocess.Popen([sys.executable, '-u', *map(str, command)], env=env,
                                        stdout=log, stderr=subprocess.STDOUT)
            worker = dict(label=label, pid=proc.pid, command=list(map(str, command)), log=str(log_path))
            state['workers'].append(worker)
            state['commands'].append(worker)
            atomic_json(state_path, state)
            workers.append(proc)
        # A failed sibling does not terminate another valid, auditable run.
        while any(proc.poll() is None for proc in workers):
            time.sleep(10)
        codes = [proc.returncode for proc in workers]
        if any(codes):
            raise RuntimeError(f'{name} failed with exit codes {codes}; inspect its logs')

    def training(config, output, device):
        return ['-m', 'grokking_capacity.experiments.discrete', '--config', config,
                '--output', output, '--device', device, '--gate', args.gate,
                *(['--resume'] if args.resume else [])]

    try:
        gate = check_gate(args.gate)
        audit = json.loads((args.gate / 'audit.json').read_text())
        if not audit['arithmetic_gate_passed'] or audit.get('gate') != gate:
            raise ValueError('Arithmetic checkpoint must have an independent passing audit')
        random_root = root / 'random-learning'
        stage('random_learning', [('B', training(args.random_learning, random_root, args.b_device))])
        random_paths = configured_paths(args.random_learning, random_root)
        stage('random_audit', [('B', ['-m', 'grokking_capacity.experiments.discrete_workflow.audit_discrete_capacity', *random_paths])])
        for path in random_paths:
            summary = json.loads((path / 'summary.json').read_text())
            if summary['metrics']['train_99'] is None or summary['final']['train_acc'] < .99:
                raise ValueError('Small random-label learning validation did not pass')
        a_root, b_root, c_root = (root / label for label in ('capacity-A','capacity-B','capacity-C'))
        a_paths, b_paths, c_paths = (configured_paths(config, output) for config, output in
                                   [(args.a_capacity,a_root),(args.b_capacity,b_root),(args.c_capacity,c_root)])
        stage('capacity', [('A', training(args.a_capacity,a_root,args.a_device)),
                           ('B', training(args.b_capacity,b_root,args.b_device))])
        stage('capacity_audit', [('AB', ['-m', 'grokking_capacity.experiments.discrete_workflow.audit_discrete_capacity', *a_paths, *b_paths])])
        p = json.loads((b_paths[0] / 'manifest.json').read_text())['config']['p']
        stage('capacity_match', [('AB', ['-m', 'grokking_capacity.experiments.discrete_workflow.capacity','--a-runs',*a_paths,
               '--b-runs',*b_paths,'--p',p,'--min-parameter-ratio',3,'--output',root/'capacity_assessment.json'])])
        confirmation = root / 'confirmation'
        stage('confirmation_and_control', [
            ('ABC', ['-m', 'grokking_capacity.experiments.discrete_workflow.confirmation','--a-runs',*a_paths,'--b-runs',*b_paths,
                     '--output',confirmation,'--device',args.b_device,'--min-parameter-ratio',3,
                     '--confirmation-seeds',73,74,75,*(['--resume'] if args.resume else [])]),
            ('C', training(args.c_capacity,c_root,args.a_device))])
        selection = json.loads((confirmation / 'selection.json').read_text())
        stage('confirmation_audit', [('ABC', ['-m', 'grokking_capacity.experiments.discrete_workflow.audit_discrete',
                *[confirmation / row['name'] for row in selection['runs']]])])
        stage('control_audit', [('C', ['-m', 'grokking_capacity.experiments.discrete_workflow.audit_discrete_capacity',*c_paths])])
        stage('report', [('ABC', ['-m', 'grokking_capacity.experiments.discrete_workflow.report',confirmation,'--output',args.report,'--c-runs',*c_paths])])
        state.update(phase='complete', report=str(args.report), completed_at=time.time())
        atomic_json(state_path, state)
    except Exception as error:
        state.update(phase='failed', error=str(error))
        atomic_json(state_path, state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('gate','random-learning','a-capacity','b-capacity','c-capacity','output','report'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--a-device', default='cuda:0')
    parser.add_argument('--b-device', default='cuda:1')
    parser.add_argument('--resume', action='store_true')
    replicate(parser.parse_args())
