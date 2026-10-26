#!/usr/bin/env python3
"""Read-only live status for the dataset-matched capacity sweep."""
import json
from pathlib import Path
import time
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/discrete/p97-dataset-scaling-v1'
def read(path):return json.loads(path.read_text())

def main():
    state=read(RUN/'controller.json');registry=read(RUN/'job_registry.json')
    complete=[]
    for p in (RUN/'jobs').glob('*/summary.json'):
        s=read(p);c=s['config'];complete.append(dict(name=p.parent.name,**c,
            first99=s['metrics']['val_acc_99_updates'],learned=s['learning_gate_passed']))
    print(f"{state['phase']}: {len(complete)}/{len(registry)} finished; {len(state['running'])} active; {len(state['pending'])} queued; errors={state['errors']}")
    for n in sorted({r['n_train'] for r in registry.values()},reverse=True):
        counts=[]
        for role in ['C','B']:
            rr=[r for r in complete if r['n_train']==n and r['role']==role]
            times=[r['first99'] for r in rr if r['first99'] is not None]
            mean=sum(times)/10 if len(times)==10 else None
            counts.append(f"{role}: {len(rr)}/10 done, {len(times)} first99, {sum(r['learned'] for r in rr)} final, mean={mean}")
        print(n,'; '.join(counts))
    for entry in state['running']:
        p=RUN/'jobs'/entry['name']/'trace.jsonl'
        lines=p.read_text().splitlines() if p.exists() else []
        if lines:
            try:r=json.loads(lines[-1])
            except json.JSONDecodeError:r=json.loads(lines[-2])
            print(entry['name'],r['step'],'updates',f"train={100*r['train_eval_acc']:.2f}% test={100*r['val_acc']:.2f}%",f"elapsed={r['wall_seconds']:.0f}s")
        else:print(entry['name'],'starting')

if __name__=='__main__':main()
