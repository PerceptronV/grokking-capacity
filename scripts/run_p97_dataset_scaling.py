#!/usr/bin/env python3
"""Freeze and execute 200 nested dataset-size confirmations; audit all checkpoints."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from grokking_capacity.experiments.dataset_scaling import (
    RUN, read, sha, atomic_json, source_hashes, logarithmic_sizes, select_pairs, audit_job)


def initialize():
    regpath = ROOT/'configs/discrete_p97_dataset_scaling.yaml'
    reg = yaml.safe_load(regpath.read_text())
    d = reg['dataset']
    sizes = logarithmic_sizes(d['minimum_training_examples'], d['maximum_training_examples'], d['size_count'])
    assert sizes == d['sizes'] and len(reg['confirmation']['seeds']) == 10
    previous = ROOT/reg['models']['source']
    old = read(previous)
    for path, digest in old['evidence_sha256'].items():
        assert sha(path) == digest
    pairs = select_pairs(old['constants'], sizes, tolerance=reg['models']['tolerance'])
    RUN.mkdir(parents=True, exist_ok=False)
    for directory in ['jobs', 'audits', 'configs', 'logs']:
        (RUN/directory).mkdir()
    shutil.copyfile(regpath, RUN/'registration.yaml')
    selection = dict(pairs=pairs, constants=old['constants'], frozen_at=time.time(),
        source=str(previous.resolve()), source_sha256=sha(previous),
        calibration_evidence=old['evidence_sha256'], sizes=sizes, seeds=reg['confirmation']['seeds'])
    atomic_json(RUN/'selection.json', selection)
    binding = sha(RUN/'selection.json')
    registry = {}
    # Complete each dataset size across ten seeds, largest first; never adaptive.
    for pair in reversed(pairs):
        n = pair['n_train']
        for seed in reg['confirmation']['seeds']:
            for role in ['C', 'B']:
                c = dict(**reg['common'], **pair[role], role=role, seed=seed, n_train=n,
                    selection_sha256=binding, pair_index=pair['index'])
                name = f'{role}_d{c["dim"]}_n{n}_s{seed}'
                registry[name] = c
                atomic_json(RUN/'configs'/(name+'.json'), c)
    assert len(registry) == reg['confirmation']['runs'] == 200
    atomic_json(RUN/'job_registry.json', registry)
    hashes = source_hashes()
    for name in hashes:
        dest = RUN/'source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
    atomic_json(RUN/'manifest.json', dict(source_hashes=hashes,
        registration_sha256=sha(RUN/'registration.yaml'), selection_sha256=binding,
        registry_sha256=sha(RUN/'job_registry.json'), started_at=time.time(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    return reg, registry


def main():
    torch.set_num_threads(2)
    reg, registry = initialize()
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    queue, active = list(registry), {}
    state = dict(pid=os.getpid(), phase='running', started_at=time.time(), completed=[], errors=[])
    try:
        while queue or active:
            while queue and len(active) < reg['resources']['workers'] and not state['errors']:
                assert source_hashes() == read(RUN/'manifest.json')['source_hashes']
                name = queue.pop(0)
                handle = (RUN/'logs'/(name+'.log')).open('x')
                cmd = [sys.executable, '-u', '-m', 'grokking_capacity.experiments.dataset_scaling',
                    '--config', str(RUN/'configs'/(name+'.json')), '--output', str(RUN/'jobs'/name),
                    '--device', reg['resources']['device']]
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                active[name] = (process, handle, dict(name=name, pid=process.pid, started_at=time.time()))
            state.update(running=[v[2] for v in active.values()], pending=queue)
            atomic_json(RUN/'controller.json', state)
            time.sleep(5)
            for name, (process, handle, entry) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                state['completed'].append(dict(**entry, exit_code=code, finished_at=time.time()))
                if code:
                    state['errors'].append(name)
                del active[name]
            if state['errors']:
                state['not_launched_after_failure'] = queue
                queue = []
        assert not state['errors'], state['errors']
        state.update(phase='auditing', running=[], pending=[])
        atomic_json(RUN/'controller.json', state)
        for name in registry:
            result = audit_job(RUN/'jobs'/name, reg['resources']['device'])
            atomic_json(RUN/'audits'/(name+'.json'), result)
            print('AUDITED '+name, flush=True)
        assert len(list((RUN/'audits').glob('*.json'))) == 200
        atomic_json(RUN/'campaign_complete.json', dict(finished_at=time.time(), runs=200, audits=200, all_passed=True))
        state.update(phase='complete', finished_at=time.time())
        atomic_json(RUN/'controller.json', state)
    except Exception as exc:
        atomic_json(RUN/'campaign_error.json', dict(time=time.time(), error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
