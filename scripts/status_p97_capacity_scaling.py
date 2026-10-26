#!/usr/bin/env python3
"""Read-only compact status for the durable capacity/scaling campaign."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/discrete/p97-capacity-scaling-v1'

def read(path):return json.loads(path.read_text())

def main():
    states=[(p,read(p)) for p in (RUN/'stages').glob('*/controller.json')]
    if not states:print(json.dumps(dict(phase='initializing')));return
    path,state=max(states,key=lambda pair:pair[1]['started_at']);recent=[]
    for job in state['completed'][-8:]:
        p=RUN/'jobs'/job['name']/'summary.json'
        if not p.exists():recent.append(dict(name=job['name'],exit_code=job['exit_code']));continue
        s=read(p);c=s['config'];row=dict(name=job['name'],role=c['role'],dim=c['dim'],seed=c['seed'],epochs=s['epochs'])
        if s['task']=='capacity':row.update(n=c['n_samples'],bits=round(s['final_bits']),drift=s.get('late_drift'))
        else:row.update(t_gen=10*s['metrics']['val_acc_99_epoch'] if s['metrics']['val_acc_99_epoch'] is not None else None,learned=s['learning_gate_passed'])
        recent.append(row)
    result=dict(stage=path.parent.name,phase=state['phase'],completed=len(state['completed']),running=len(state['running']),
        pending=len(state['pending']),errors=state['errors'],recent=recent,audits=len(list((RUN/'audits').glob('*.json'))))
    live=[]
    for job in state['running']:
        trace=RUN/'jobs'/job['name']/'trace.jsonl'
        if not trace.exists():continue
        lines=trace.read_text().splitlines();rows=[]
        for line in lines:
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: pass
        if not rows:continue
        last=rows[-1]
        live.append(dict(name=job['name'],step=last['step'],train=round(last['train_online_acc'],4),
            val=round(last['val_acc'],4),first99=next((r['step'] for r in rows if r['val_acc']>=.99),None)))
    result['live']=live
    for name in ['campaign_error','campaign_complete']:
        p=RUN/(name+'.json')
        if p.exists():result[name]=read(p)
    p=RUN/'selection.json'
    if p.exists():
        s=read(p);result['selection']=dict(frozen_at=s['frozen_at'],constants={r:dict(kappa=f['bits_per_parameter'],max_residual=f['max_absolute_relative_residual']) for r,f in s['constants'].items()},
            pairs=[dict(index=p['index'],C=p['C']['dim'],B=p['B']['dim'],C_params=p['C_parameters'],B_params=p['B_parameters'],capacity=p['C_estimated_bits'],mismatch=p['predicted_relative_mismatch']) for p in s['pairs']])
    print(json.dumps(result))

if __name__=='__main__':main()
