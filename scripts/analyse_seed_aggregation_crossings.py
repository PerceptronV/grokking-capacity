#!/usr/bin/env python3
"""Registered crossing cohorts: aggregation sensitivity with cap use explicit."""
from pathlib import Path
from collections import defaultdict
import json, math, sqlite3, sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analyse_seed_aggregation import ROOT,OUT,METRICS,save,csvsave,clean
from cone_data import first_events
from grokking_capacity.analysis.aggregate import reduce_values,summarize_delays,onset_summary


def crossing(lP,lm,lg):
    gap=lm-lg
    if np.all(gap>0):return None,'mem_slower_everywhere'
    if np.all(gap<0):return None,'mem_faster_everywhere'
    for i in range(len(gap)-1):
        if gap[i]>0 and gap[i+1]<=0:
            return float(lP[i]+(lP[i+1]-lP[i])*gap[i]/(gap[i]-gap[i+1])),'crossing'
    return None,'no_downward_crossing'


def onset_domain(values,coords):
    # The registered rule removes a terminal run of at least two exact zeros.
    # Unresolved aggregates are never converted to zero, and cannot trigger trimming.
    W=len(values)
    for i in range(len(values)-1,0,-1):
        if values[i]==0 and values[i-1]==0:W=i-1
        else:break
    if W==0:return {'value':None,'status':'zero_everywhere','trimmed_widths':len(values)}
    return {**onset_summary(dict(zip(coords[:W],values[:W]))),'trimmed_widths':len(values)-W}


def main():
    allcells=[];allprimes=[];summary=[];raw=[]
    for setting in ['fixed','mup']:
        db=Path.home()/f'crossing_{setting}_runs.db'
        con=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
        rr=[dict(r) for r in con.execute("select * from runs where status='completed'")];con.close()
        mem=defaultdict(list);gen=defaultdict(list);P={}
        for r in rr:
            k=r['p'],r['dim'];P[k]=r['param_count']
            if r['experiment_type']=='speed':
                mem[k].append((r['saturation_epoch'] if r['saturated'] else r['max_epochs'],not bool(r['saturated'])))
            elif r['experiment_type']=='groks':
                with np.load(r['npz_path'],allow_pickle=False) as z:e=first_events(z['train_acc'],z['val_acc'])
                gen[k].append((r['grokking_epoch'] or r['max_epochs'],r['grokking_epoch'] is None,e['delay']))
                raw.append({'setting':setting,'p':k[0],'dim':k[1],'seed':r['seed'],'uuid':r['uuid'],**e})
        for tmetric in ['geometric_mean','median']:
            for metric in METRICS:
                agreements=[];complete_agreements=[];primeoffsets=[]
                for p in sorted({p for p,d in gen}&{p for p,d in mem}):
                    ds=sorted(d for pp,d in gen if pp==p and (pp,d) in mem)
                    if len(ds)<4:continue
                    lp=np.log10([P[p,d] for d in ds])
                    tm=np.array([reduce_values([t for t,_ in mem[p,d]],tmetric) for d in ds])
                    tg=np.array([reduce_values([t for t,_,_ in gen[p,d]],tmetric) for d in ds])
                    cross,cstatus=crossing(lp,np.log(tm),np.log(tg))
                    vals=[];local=[];complete=[]
                    for i,d in enumerate(ds):
                        value=summarize_delays([dd for _,_,dd in gen[p,d]],metric)
                        vals.append(value['value'])
                        all_times_observed=not any(c for _,c in mem[p,d]) and not any(c for _,c,_ in gen[p,d])
                        agree=None if value['value'] is None else bool((tm[i]<tg[i])==(value['value']>0))
                        if agree is not None:
                            agreements.append(agree);local.append(agree)
                            if all_times_observed:complete_agreements.append(agree);complete.append(agree)
                        allcells.append({'setting':setting,'time_metric':tmetric,'delay_metric':metric,'p':p,'dim':d,'P':P[p,d],
                            'mem_time_cap_substituted':float(tm[i]),'gen_time_cap_substituted':float(tg[i]),
                            'all_times_observed':all_times_observed,'n_mem':len(mem[p,d]),'n_gen':len(gen[p,d]),
                            'delay':value['value'],'unresolved_delays':value['n_unresolved'],'agreement':agree})
                    on=onset_domain(vals,lp)
                    offset=on['value']-cross if cross is not None and on['value'] is not None and on['status']=='observed' else None
                    if offset is not None:primeoffsets.append(offset)
                    allprimes.append({'setting':setting,'time_metric':tmetric,'delay_metric':metric,'p':p,'crossing_log10P':cross,
                        'crossing_status':cstatus,'onset_log10P':on['value'],'onset_status':on['status'],
                        'trimmed_widths':on['trimmed_widths'],'offset':offset,'score':float(np.mean(local)) if local else None,
                        'score_n':len(local),'complete_score':float(np.mean(complete)) if complete else None,'complete_n':len(complete)})
                summary.append({'setting':setting,'time_metric':tmetric,'delay_metric':metric,'score':float(np.mean(agreements)),
                    'n_score':len(agreements),'complete_times_score':float(np.mean(complete_agreements)) if complete_agreements else None,
                    'n_complete_times_score':len(complete_agreements),'n_onset_crossing_pairs':len(primeoffsets),
                    'mean_onset_crossing_offset_dex':float(np.mean(primeoffsets)) if primeoffsets else None,
                    'sd_onset_crossing_offset_dex':float(np.std(primeoffsets,ddof=1)) if len(primeoffsets)>1 else None})
    csvsave('crossings/cells.csv',allcells);csvsave('crossings/primes.csv',allprimes);csvsave('crossings/runs.csv',raw)
    save('crossings/summary.json',{'comparisons':summary,'warning':'Registered measured timescale curves substitute epoch caps for missing crossings. Scores over all cells isolate delay aggregation under that legacy predictor; complete_times_score excludes any cell with a censored timescale. Delay outcomes themselves remain unresolved. Domain trimming is the registered rule, distinct from the untrimmed fixed-grid onset.'})
    print(json.dumps(clean(summary),indent=2),flush=True)
if __name__=='__main__':main()
