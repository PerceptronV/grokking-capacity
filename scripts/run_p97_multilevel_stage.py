#!/usr/bin/env python3
"""Launch one immutable stage of the registered dropout matching experiment."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from grokking_capacity.experiments.multilevel_dropout import atomic_json,source_hashes,sha
OUT=ROOT/'results/discrete/p97-multilevel-dropout-v1'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True)
    parser.add_argument('--jobs',required=True)
    parser.add_argument('--devices',default='cuda:1,cuda:1,cuda:1,cuda:1')
    args=parser.parse_args()
    specs=json.loads(Path(args.jobs).read_text())
    assert len({j['name'] for j in specs})==len(specs)
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'jobs').mkdir(exist_ok=True)
    if not (OUT/'manifest.json').exists():
        hashes=source_hashes()
        for path in hashes:
            dest=OUT/'source'/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/path,dest)
        shutil.copyfile(ROOT/'configs/discrete_p97_multilevel_dropout.yaml',OUT/'registration.yaml')
        atomic_json(OUT/'manifest.json',dict(source_hashes=hashes,registration_sha256=sha(OUT/'registration.yaml'),
                    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    manifest=json.loads((OUT/'manifest.json').read_text());assert manifest['source_hashes']==source_hashes()
    stage=OUT/'stages'/args.stage;stage.mkdir(parents=True,exist_ok=False)
    atomic_json(stage/'jobs.json',specs)
    for job in specs:
        assert not (OUT/'jobs'/job['name']).exists(),job['name']
        atomic_json(stage/(job['name']+'.json'),job['config'])
    state=dict(phase='running',pid=os.getpid(),started_at=time.time(),jobs=specs,completed=[],running=[],errors=[])
    queued=list(specs);active={};slots=args.devices.split(',')
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TQDM_MININTERVAL='10')
    while queued or active:
        for slot,device in enumerate(slots):
            if slot in active or not queued or state['errors']:continue
            assert source_hashes()==manifest['source_hashes']
            job=queued.pop(0)
            cmd=[sys.executable,'-u','-m','grokking_capacity.experiments.multilevel_dropout',
                 '--config',str(stage/(job['name']+'.json')),'--output',str(OUT/'jobs'/job['name']),'--device',device]
            handle=(stage/(job['name']+'.log')).open('x')
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            entry=dict(name=job['name'],device=device,pid=proc.pid,command=cmd,started_at=time.time())
            active[slot]=(proc,handle,entry)
        state['running']=[v[2] for v in active.values()];state['pending']=[j['name'] for j in queued]
        atomic_json(stage/'controller.json',state);time.sleep(5)
        for slot,(proc,handle,entry) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            handle.close();state['completed'].append({**entry,'exit_code':code,'finished_at':time.time()})
            if code:state['errors'].append(entry['name'])
            del active[slot]
        if state['errors']:
            state['not_launched_after_failure']=[j['name'] for j in queued];queued.clear()
    state['running']=[];state['pending']=[];state['phase']='failed' if state['errors'] else 'complete';state['finished_at']=time.time()
    atomic_json(stage/'controller.json',state)


if __name__=='__main__':main()
