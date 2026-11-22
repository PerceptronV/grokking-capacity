"""Observed first-passage events and seed aggregation for cone experiments."""
from collections import Counter, defaultdict
import numpy as np
from .aggregate import summarize_delays


def first_events(train,val):
    train=np.asarray(train);val=np.asarray(val)
    assert len(train)==len(val)>0 and train.ndim==val.ndim==1
    assert np.isfinite(train).all() and np.isfinite(val).all()
    assert ((train>=0)&(train<=100.00001)).all() and ((val>=0)&(val<=100.00001)).all()
    ti=np.flatnonzero(train>=99);vi=np.flatnonzero(val>=98)
    t=int(ti[0]+1) if len(ti) else None;v=int(vi[0]+1) if len(vi) else None
    if t is not None and v is not None:
        delay=max(0,v-t);status='positive' if delay>0 else 'zero';reason='both_thresholds_observed';lower=delay
    elif v is not None:
        # T_train is later than follow-up; validation already crossed first.
        delay=0;status='zero';reason='validation_before_unobserved_training';lower=0
    elif t is not None:
        delay=None;status='unresolved';reason='validation_right_censored';lower=len(val)-t
    else:
        delay=None;status='unresolved';reason='both_thresholds_unobserved';lower=None
    return {'train99':t,'val98':v,'followup_epochs':len(val),'delay':delay,'delay_lower_bound':lower,'state':status,'event_status':reason}

def aggregate(rows, metric="min"):
    by=defaultdict(list)
    for r in rows:by[r['mup_base'],r['p'],r['dim']].append(r)
    cells=[]
    for (mb,p,d),rr in sorted(by.items()):
        assert len({(r['seed']) for r in rr})==len(rr),('duplicate seed',mb,p,d)
        assert len({(r['host'],r['gpu_type']) for r in rr})==1,('hardware pooled',mb,p,d)
        assert len({r['param_count'] for r in rr})==1
        counts=Counter(r['state'] for r in rr)
        summary=summarize_delays([r['delay'] for r in rr],metric)
        state=summary['state']
        exact=all(r['delay'] is not None for r in rr)
        cells.append({'mup_base':mb,'p':p,'dim':d,'param_count':rr[0]['param_count'],'host':rr[0]['host'],'gpu_type':rr[0]['gpu_type'],'n_seeds':len(rr),'seeds':sorted(r['seed'] for r in rr),'state':state,'positive_seeds':counts['positive'],'zero_seeds':counts['zero'],'unresolved_seeds':counts['unresolved'],'fraction_positive_lower':counts['positive']/len(rr),'fraction_positive_upper':(counts['positive']+counts['unresolved'])/len(rr),'minimum_delay':summarize_delays([r['delay'] for r in rr],'min')['value'],'mean_delay':float(np.mean([r['delay'] for r in rr])) if exact else None,'budgets':sorted({r['max_epochs'] for r in rr}),'min_followup':min(r['followup_epochs'] for r in rr),'max_followup':max(r['followup_epochs'] for r in rr),'aggregation_metric':summary['metric'],'aggregate_delay':summary['value'],'aggregate_lower':summary['lower'],'aggregate_upper':summary['upper'],'aggregate_sign':summary['sign'],'median_delay':summarize_delays([r['delay'] for r in rr],'median')['value']})
    return cells
