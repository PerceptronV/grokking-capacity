#!/usr/bin/env python3
"""Fill free GPU slots with already-authorized C calibration cells only."""
import json
import os
from pathlib import Path
import signal
import time
from grokking_capacity.experiments.discrete_workflow.manage_p97_multilevel import RUN, read, reg, capacity_jobs, submit, sha
from grokking_capacity.experiments.discrete_workflow.advance_p97_multilevel import known_jobs


def main():
    if (RUN/'requests/confirmation.json').exists():
        print('Confirmation already submitted; no calibration jobs added.');return
    pid=read(RUN/'advancer_launch.json')['pid']
    assert 'advance_p97_multilevel.py' in Path(f'/proc/{pid}/cmdline').read_text()
    for _ in range(120):
        if Path(f'/proc/{pid}/wchan').read_text().strip()=='hrtimer_nanosleep':break
        time.sleep(.25)
    else:raise RuntimeError('Queue controller not idle; no mutation performed.')
    os.kill(pid,signal.SIGSTOP)
    try:
        known,used=known_jobs();plans=[]
        # Existing multi-job stages reserve any slots they can fill from their
        # pending list, even between worker exit and the next state publication.
        for path in (RUN/'stages').glob('*/controller.json'):
            state=read(path)
            if not state.get('pending'):continue
            launch=read(RUN/'requests'/(path.parent.name+'.launch.json'))
            cmd=launch['command'];devices=cmd[cmd.index('--devices')+1].split(',')
            if len(set(devices))==1:
                device=devices[0];alive=0
                for entry in state['running']:
                    try:os.kill(entry['pid'],0);alive+=1
                    except ProcessLookupError:pass
                used[device]+=min(len(state['pending']),max(0,len(devices)-alive))
        for family in reg()['families']:
            path=RUN/f'{family}_C_construction.json'
            if not path.exists():continue
            control=read(path)
            assert control['AB_selection_sha256']==sha(RUN/f'{family}_AB_selection.json')
            m=control['model'];stem=f"{family}_C_d{m['dim']}_f{m['mlp_dim']}"
            gate=RUN/'jobs'/(stem+'_learning_s41')
            if not (gate/'summary.json').exists() or not read(gate/'summary.json')['learning_gate_passed']:continue
            for seed in reg()['capacity']['seeds']:
                random=RUN/'jobs'/f'{stem}_capacity_n2048_s{seed}'/'summary.json'
                if random.exists() and read(random)['random_learning_passed']:
                    ns=[16384,8192,4096]
                elif not random.exists():ns=[2048]
                else:continue
                plans.extend(j for j in capacity_jobs(gate.name,ns,[seed]) if j['name'] not in known)
        plans.sort(key=lambda j:({2048:0,16384:1,8192:2,4096:3}[j['config']['n_samples']],j['name']))
        launched=[]
        for device in ('cuda:0','cuda:1'):
            free=max(0,4-used.get(device,0));batch=plans[:free];plans=plans[free:]
            if batch:
                launched.append(submit('control_grid_'+str(time.time_ns()),batch,
                    devices=','.join([device]*len(batch))))
        print(json.dumps(dict(occupied_before=used,launched=launched,pending_eligible=[j['name'] for j in plans]),indent=2))
    finally:os.kill(pid,signal.SIGCONT)


if __name__=='__main__':main()
