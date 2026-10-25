#!/usr/bin/env python3
"""Run the registered A/dropout follow-up and capacity confirmation in isolation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/discrete_p97_dropout_followup.yaml'
OUT = ROOT/'results/discrete/p97-dropout-followup'
SOURCES = ['src/grokking_capacity/experiments/groks.py',
           'src/grokking_capacity/experiments/capacity.py',
           'src/grokking_capacity/models/transformer.py',
           'src/grokking_capacity/data/modular.py',
           'src/grokking_capacity/utils/optim.py',
           'src/grokking_capacity/utils/cli_args.py',
           'scripts/run_p97_dropout_followup.py']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def jobs(config):
    result=[]
    common=config['common']
    for seed in config['arithmetic']['seeds']:
        cfg={**common, **{k:config['arithmetic'][k] for k in ('p','operation','train_fraction','split_type','dim')}, 'seed':seed}
        result.append(dict(name=f'A_s{seed}', kind='groks', config=cfg))
    for key in ('A_direct_capacity','capacity_confirmation'):
        group=config[key]
        for seed in group['seeds']:
            for dim in group['dimensions']:
                for n in group['n_samples']:
                    cfg={**common, 'p':group['p'], 'dim':dim, 'seed':seed, 'n_samples':n,
                         'dataset_type':'random', 'patience':group['patience'], 'min_delta':group['min_delta']}
                    result.append(dict(name=f'{key}_d{dim}_n{n}_s{seed}',kind='capacity',config=cfg))
    return result


def main():
    if (OUT/'controller.json').exists():
        raise FileExistsError('Inspect existing controller before restarting')
    OUT.mkdir(parents=True,exist_ok=True)
    config=yaml.safe_load(CONFIG.read_text())
    queued=jobs(config)
    for name in SOURCES:
        dest=OUT/'source'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,dest)
    shutil.copyfile(CONFIG,OUT/'configuration.yaml')
    state=dict(pid=os.getpid(),process_start_ticks=Path('/proc/self/stat').read_text().split()[21],
               phase='running',started_at=time.time(),source_hashes={p:digest(ROOT/p) for p in SOURCES},
               config_sha256=digest(CONFIG),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               jobs=list(queued), completed=[], running=[], errors=[])
    slots=[('cuda:1',0),('cuda:0',1),('cuda:1',2),('cuda:0',3)]
    active={}
    environment=dict(os.environ,PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                     TQDM_MININTERVAL='10',GC_DATA_DIR=str(OUT/'data'))
    # Separate registries per worker avoid unrelated registry writes and locking.
    try:
        while queued or active:
            for slot,(device,lane) in enumerate(slots):
                if slot in active or not queued:
                    continue
                job=queued.pop(0)
                for source,sha in state['source_hashes'].items():
                    assert digest(ROOT/source)==sha,source
                command=[sys.executable,'-u','-m',f"grokking_capacity.experiments.{job['kind']}"]
                for key,value in job['config'].items():
                    command.extend(['--'+key.replace('_','-'),str(value)])
                command.extend(['--device',device,'--db-path',str(OUT/f'lane_{lane}.db'),'--save-model'])
                if job['kind']=='groks':
                    command.extend(['--ignore-memorisation','--norm-log-every','0'])
                log=(OUT/f"{job['name']}.log").open('x')
                proc=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
                entry=dict(name=job['name'],pid=proc.pid,device=device,lane=lane,command=command,started_at=time.time())
                active[slot]=(proc,log,entry)
            state['running']=[value[2] for value in active.values()]
            state['pending_names']=[job['name'] for job in queued]
            write(OUT/'controller.json',state)
            time.sleep(5)
            for slot,(proc,log,entry) in list(active.items()):
                code=proc.poll()
                if code is None:
                    continue
                log.close()
                state['completed'].append({**entry,'exit_code':code,'finished_at':time.time()})
                if code:
                    state['errors'].append(entry['name'])
                del active[slot]
            if state['errors']:
                # Finish already-running jobs, but do not launch more after an error.
                queued.clear()
        state['running']=[]
        state['phase']='failed' if state['errors'] else 'complete'
        state['finished_at']=time.time()
        write(OUT/'controller.json',state)
    except Exception as error:
        state['phase']='failed';state['error']=str(error)
        write(OUT/'controller.json',state)
        raise


if __name__=='__main__':
    main()
