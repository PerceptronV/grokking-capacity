#!/usr/bin/env python3
"""Freeze nested dataset-size confirmations and audit their checkpoints."""
from grokking_capacity.experiments.campaigns import runner
from grokking_capacity.experiments.campaigns.config import options, initialize as register, freeze_selection, check_registration
from pathlib import Path
import time
import torch
from ..registered import ROOT
from grokking_capacity.experiments.dataset_scaling import (
    RUN, read, sha, atomic_json, source_hashes, logarithmic_sizes, select_pairs, audit_job)


def initialize(args):
    regpath = args.config
    reg = register(RUN, regpath, source_hashes(), resume=args.resume)
    d = reg['dataset']
    sizes = logarithmic_sizes(d['minimum_training_examples'], d['maximum_training_examples'], d['size_count'])
    assert sizes == d['sizes']
    previous = ROOT/reg['models']['source']
    old = read(previous)
    for path, digest in old['evidence_sha256'].items():
        assert sha(path) == digest
    widths = reg['models']['width_candidates']
    pairs = select_pairs(old['constants'], sizes,
        candidates=range(widths['start'], widths['stop']+1, widths['step']),
        tolerance=reg['models']['tolerance'])
    selection = dict(pairs=pairs, constants=old['constants'], frozen_at=time.time(),
        source=str(previous.resolve()), source_sha256=sha(previous),
        calibration_evidence=old['evidence_sha256'], sizes=sizes, seeds=reg['confirmation']['seeds'])
    freeze_selection(RUN/'selection.json', selection)
    binding = sha(RUN/'selection.json')
    registry = {}
    # Complete each dataset size across the registered seeds, largest first; never adaptive.
    for pair in reversed(pairs):
        n = pair['n_train']
        for seed in reg['confirmation']['seeds']:
            for role in ['C', 'B']:
                c = dict(**reg['common'], **pair[role], role=role, seed=seed, n_train=n,
                    selection_sha256=binding, pair_index=pair['index'])
                name = f'{role}_d{c["dim"]}_n{n}_s{seed}'
                registry[name] = c
                # Job configuration files are frozen by the shared executor.
    assert len(registry) == 2*len(pairs)*len(reg['confirmation']['seeds'])
    existing = read(RUN/'job_registry.json')
    if existing and existing != registry:
        raise ValueError('Registered jobs changed')
    atomic_json(RUN/'job_registry.json', registry)
    manifest = read(RUN/'manifest.json')
    manifest.update(selection_sha256=sha(RUN/'selection.json'), registry_sha256=sha(RUN/'job_registry.json'))
    atomic_json(RUN/'manifest.json', manifest)
    return reg, registry


def main():
    global RUN
    args=options('dataset_scaling',ROOT/'results/discrete/p97-dataset-scaling-v2')
    RUN=args.root
    torch.set_num_threads(2)
    reg,registry=initialize(args)
    def audit(path):
        result=audit_job(path,args.audit_device)
        atomic_json(RUN/'audits'/(path.name+'.json'),result)
    paths=runner.run_stage(RUN,'confirmation',[{'name':n,'config':c} for n,c in registry.items()],
        [reg['resources']['device']]*reg['resources']['workers'],
        'grokking_capacity.experiments.dataset_scaling',resume=args.resume,
        source_check=lambda: check_registration(RUN, source_hashes()),audit=audit)
    (RUN/'campaign_error.json').unlink(missing_ok=True)
    atomic_json(RUN/'campaign_complete.json',dict(finished_at=time.time(),runs=len(paths),audits=len(paths),all_passed=True))


if __name__=='__main__':main()
