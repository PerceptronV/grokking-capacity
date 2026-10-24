#!/usr/bin/env python3
"""Audit completed checkpoints while training finishes, then export final evidence."""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from grokking_capacity.experiments.discrete import atomic_json
from scripts.audit_discrete import audit
from scripts.confirm_discrete import process_live
from scripts.report_discrete import report


def finish(root,output,c_runs):
    state_path=root/'report_worker.json'
    if state_path.exists():
        previous=json.loads(state_path.read_text())
        if process_live(previous['pid'],previous['process_start_ticks']):
            raise RuntimeError('Report worker already live')
    state=dict(pid=os.getpid(),process_start_ticks=Path('/proc/self/stat').read_text().split()[21],
               phase='waiting',output=str(output),c_runs=[str(p) for p in c_runs])
    atomic_json(state_path,state)
    try:
        while True:
            queue=json.loads((root/'orchestrator.json').read_text())
            if queue['phase'] not in ('confirming','complete'):
                raise RuntimeError(f'Unexpected training state: {queue}')
            for entry in json.loads((root/'selection.json').read_text())['runs']:
                path=root/entry['name']
                if (path/'summary.json').exists() and not (path/'audit.json').exists():
                    state.update(phase='auditing',current_run=entry['name'])
                    atomic_json(state_path,state)
                    audit(path)
            pending=[p for p in c_runs if not (p/'summary.json').exists()]
            for path in pending:
                manifest=json.loads((path/'manifest.json').read_text())
                if not process_live(manifest['pid'],manifest['process_start_ticks']):
                    raise RuntimeError(f'Capacity control stopped before completion: {path}')
            if queue['phase']=='complete' and not pending:
                break
            if queue['phase']!='complete' and not process_live(queue['pid'],queue['process_start_ticks']):
                raise RuntimeError('Training worker stopped before queue completion')
            state['phase']='waiting'
            atomic_json(state_path,state)
            time.sleep(10)
        state['phase']='reporting'
        atomic_json(state_path,state)
        report(root,output,c_runs)
        state['phase']='complete'
        atomic_json(state_path,state)
    except Exception as error:
        state.update(phase='failed',error=str(error))
        atomic_json(state_path,state)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--c-runs',type=Path,nargs='+',required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    finish(args.root,args.output,args.c_runs)
