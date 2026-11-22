#!/usr/bin/env python3
"""Launch the already frozen ten-seed A/B/C comparison on both GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'src'))
from grokking_capacity.experiments.matched_dropout import sha
RUN=ROOT/'results/discrete/p97-dropout-match-v1'


def main():
    selected=json.loads((RUN/'selection.json').read_text())
    registration=yaml.safe_load((RUN/'registration.yaml').read_text())
    assert selected['registration_sha256']==sha(RUN/'registration.yaml')
    for file,expected in selected['evidence_sha256'].items():assert sha(file)==expected
    requests=[]
    for part,seeds in enumerate((range(101,106),range(106,111)),1):
        jobs=[]
        for seed in seeds:
            for role in 'ABC':
                config={**registration['common'],**selected['models'][role],
                        'role':role,'phase':'confirmation','task':'arithmetic','seed':seed,
                        'stop_rule':'joint99','selection_sha256':sha(RUN/'selection.json')}
                jobs.append(dict(name=f'{role}_confirmation_s{seed}',config=config))
        name=f'confirmation_part{part}'
        request=RUN/'requests'/(name+'.json')
        assert not request.exists() and not (RUN/'stages'/name).exists()
        requests.append((name,request,jobs))
    for name,request,jobs in requests:
        with request.open('x') as f:json.dump(jobs,f,indent=2);f.write('\n')
        log=request.with_suffix('.launch.log').open('x')
        child=subprocess.Popen([sys.executable,'-u','-m','grokking_capacity.experiments.campaigns.run_p97_dropout_match',
            '--root',str(RUN),'--config',str(RUN/'registration.yaml'),'--stage',name,'--jobs',str(request)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(f'{name}: PID {child.pid}, {len(jobs)} jobs',flush=True)


if __name__=='__main__':main()
