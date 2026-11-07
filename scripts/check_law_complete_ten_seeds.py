#!/usr/bin/env python3
"""Calculate complete-ten-seed law sensitivity; never change paper/figure outputs."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json
import math
import shutil
import sqlite3
import numpy as np
from scipy.optimize import brentq
from grokking_capacity.analysis.config_view import ConfigView
from grokking_capacity.analysis.forecast_onset import _pick_group, CENTRAL_PRIMES, _ols
from grokking_capacity.analysis.plots import _passes_filters
from grokking_capacity.analysis.matching import compute_n_equiv

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/law_complete_ten_seed_sensitivity_20261106'
EXPECTED=set(range(42,52))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):(OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def K(p):return p*(p-1)/2*math.log2(p+2)


def extract():
    view=ConfigView.from_yaml(ROOT/'configs/central.yaml',db_path=str(ROOT/'runs.db'))
    group=_pick_group(view);figure=view.intersection_figures[0];C=float(group.capacity_constant)
    con=sqlite3.connect((ROOT/'runs.db').as_uri()+'?mode=ro',uri=True)
    hardware={u:(h,g) for u,h,g in con.execute('select uuid,host,gpu_type from runs')};con.close()
    out={};details=[]
    for kind,rows in [('mem',group.speed_runs),('gen',group.groks_runs)]:
        cells=defaultdict(list)
        for r in rows:
            if r['p'] not in CENTRAL_PRIMES or not _passes_filters(r,figure):continue
            bits=(r.get('dataset_bits') or r['n_samples']*math.log2(r['p']+2)) if kind=='mem' else compute_n_equiv(r['p'],group.key.operation,float(group.key.train_fraction))[1]
            f=bits/(C*r['param_count'])
            if not .08<=f<=1.:continue
            host,gpu=hardware[r['uuid']]
            assert host=='147-224-250-73' and gpu=='A100'
            event=r.get('saturation_epoch' if kind=='mem' else 'grokking_epoch')
            observed=event is not None and r.get('saturated') is not False
            cells[r['p'],r['param_count']].append({'uuid':r['uuid'],'kind':kind,'p':r['p'],
                'dim':r['dim'],'P':r['param_count'],'seed':r['seed'],'host':host,'gpu_type':gpu,
                'f_original':f,'event_epoch':event,'observed':observed})
        current=[];complete=[];summaries=[]
        for (p,P),rr in sorted(cells.items()):
            assert len(rr)==10 and {r['seed'] for r in rr}==EXPECTED
            observed=[r for r in rr if r['observed']];all_ten=len(observed)==10
            summaries.append({'p':p,'dim':rr[0]['dim'],'P':P,'observed_seeds':len(observed),
                              'observed_seed_ids':[r['seed'] for r in observed],'included_complete_ten':all_ten})
            for r in rr:details.append({**r,'included_current_fit':r['observed'],'included_complete_ten_fit':all_ten})
            if kind=='mem':points=[{'p':p,'P':P,'f':r['f_original'],'t':r['event_epoch']} for r in observed]
            else:points=[] if not observed else [{'p':p,'P':P,'f':rr[0]['f_original'],'t':float(np.mean([r['event_epoch'] for r in observed]))}]
            current.extend(points)
            if all_ten:complete.extend(points)
        out[kind]={'current':current,'complete':complete,'cells':summaries}
    return C,out,details


def fit(kind,points,prime_sample=None):
    if prime_sample is not None:points=[r for p in prime_sample for r in points if r['p']==p]
    if kind=='mem':X=np.array([[1,r['f']] for r in points]);keys=['ln_b','gamma_original']
    else:X=np.array([[1,math.log(r['p']),-math.log(r['P'])] for r in points]);keys=['lnA','a','beta']
    assert np.linalg.matrix_rank(X)==len(keys)
    coeff,r2,rss=_ols(X,np.log([r['t'] for r in points]))
    return {**dict(zip(keys,map(float,coeff))),'r2':float(r2),'n_observations':len(points)}


def crossing(p,mem,gen,C):
    beta=gen['beta'];q=mem['gamma_original']/C
    rhs=gen['lnA']-mem['ln_b']+gen['a']*math.log(p)-beta*math.log(q*K(p))
    if beta<=0 or q<=0 or rhs<beta*(1-math.log(beta)):return None
    hi=max(100.,beta*2)
    while hi-beta*math.log(hi)<rhs:hi*=2
    x=brentq(lambda x:x-beta*math.log(x)-rhs,beta,hi)
    return q*K(p)/x


def fold(mem,gen,C):
    beta=gen['beta'];q=mem['gamma_original']/C
    if beta<=0 or q<=0:return None
    margin=lambda p:gen['lnA']-mem['ln_b']+gen['a']*math.log(p)-beta*math.log(q*K(p))-beta*(1-math.log(beta))
    if margin(2)*margin(10000)>=0:return None
    return float(brentq(margin,2,10000))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protected=[ROOT/'results/cone_latest.json',ROOT/'results/capacity_h100_matched_20261103/forecast_coefficients.json',
        ROOT/'results/theory/forecast_onset.json',ROOT/'docs/papers/2026 Grokking Capacity/paper-neurips-revised.pdf',
        ROOT/'docs/papers/2026 Grokking Capacity/paper-neurips-revised.tex',
        *[ROOT/'results/cone_ten_seeds_h100_20261106/figure'/('cone_seedmin_fixed'+ext) for ext in ['.pdf','.png']]]
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    historical=json.loads(protected[2].read_text());calibration=json.loads(protected[1].read_text())
    C,data,details=extract();fits={kind:{mode:fit(kind,data[kind][mode]) for mode in ['current','complete']} for kind in ['mem','gen']}
    assert np.allclose([fits['mem']['current'][k] for k in ['ln_b','gamma_original']],
                       [historical['mem_law'][k] for k in ['ln_b','gamma']],rtol=1e-12,atol=1e-12)
    assert np.allclose([fits['gen']['current'][k] for k in ['lnA','a','beta']],
                       [historical['gen_law'][k] for k in ['lnA','a','beta']],rtol=1e-12,atol=1e-12)
    comparisons={};prediction_rows=[]
    primes=sorted(set(CENTRAL_PRIMES)|{149,151,157,163,167,173,179,181,191,193,197,199,251})
    for name,mm,gg in [('current','current','current'),('complete_mem_only','complete','current'),
                       ('complete_gen_only','current','complete'),('complete_both','complete','complete')]:
        mem=fits['mem'][mm];gen=fits['gen'][gg];pstar=fold(mem,gen,C)
        central_cross={p:crossing(p,mem,gen,C) for p in CENTRAL_PRIMES}
        delta=float(np.median([math.log10(historical['per_prime'][str(p)]['P_onset']/P)
                               for p,P in central_cross.items() if P is not None]))
        comp={'mem':{**mem,'gamma_current_C':mem['gamma_original']*calibration['C_convention']/C,
                     'b':math.exp(mem['ln_b'])},'gen':gen,'divergence_prime':pstar,
              'calibrated_delta_dex_same_historical_onsets':delta,'predictions':{}}
        for p in primes:
            P=crossing(p,mem,gen,C);L=None if P is None else math.log10(P)
            row={'variant':name,'p':p,'raw_crossing_P':P,'raw_crossing_log10P':L,
                 'calibrated_onset_log10P':None if L is None else L+delta}
            prediction_rows.append(row);comp['predictions'][p]=row
        comparisons[name]=comp
    assert abs(comparisons['current']['divergence_prime']-calibration['capacity_update']['fold_new'])<1e-8
    assert abs(comparisons['current']['predictions'][149]['calibrated_onset_log10P']-historical['forecast_log10']['149'])<1e-6
    # Paired prime bootstrap: same resampled primes for the two estimators.
    rng=np.random.default_rng(20261106);draws=[]
    for _ in range(2000):
        sample=rng.choice(CENTRAL_PRIMES,size=len(CENTRAL_PRIMES),replace=True)
        try:
            aa={mode:(fit('mem',data['mem'][mode],sample),fit('gen',data['gen'][mode],sample)) for mode in ['current','complete']}
        except (AssertionError,ValueError,np.linalg.LinAlgError):continue
        fs={mode:fold(*pair,C) for mode,pair in aa.items()}
        if all(v is not None for v in fs.values()):draws.append({'current':fs['current'],'complete':fs['complete'],'difference':fs['complete']-fs['current']})
    bootstrap={'method':'Paired resampling of the ten prime clusters; cell eligibility fixed to observed data.',
               'requested_draws':2000,'paired_finite_draws':len(draws),
               'percentile_95':{k:np.quantile([r[k] for r in draws],[.025,.975]).tolist() for k in ['current','complete','difference']}}
    counts={kind:{'planned_cells':len(data[kind]['cells']),
                  'cells_by_observed_seed_count':dict(Counter(c['observed_seeds'] for c in data[kind]['cells'])),
                  'current_fit_observations':len(data[kind]['current']),'complete_fit_observations':len(data[kind]['complete']),
                  'partial_cells_removed':[c for c in data[kind]['cells'] if 0<c['observed_seeds']<10]}
            for kind in ['mem','gen']}
    result={'scope':'Calculation only; published cone, paper and coefficient artifacts unchanged.',
            'selection':'Each law separately retains cells with all ten distinct seeds 42--51 having its original observed 99% threshold crossing.',
            'hardware':'A100 only, host147-224-250-73; unchanged H100 capacity normalisation.',
            'C_current':calibration['C_convention'],'original_fit_window_f':[.08,1.],
            'current_C_fit_window_f':[.08*C/calibration['C_convention'],C/calibration['C_convention']],
            'aggregation':'Original estimators unchanged: per-run OLS for memorisation, OLS of log arithmetic seed-mean T_gen for generalisation.',
            'baseline_coefficients_reproduced':True,'counts':counts,'comparisons':comparisons,'fold_bootstrap':bootstrap,
            'caveat':'Complete-cell selection still conditions on success before the epoch cap; this is a sensitivity calculation, not a censoring correction.'}
    write('comparison.json',result)
    for name,rows in [('selected_runs.csv',details),('predictions.csv',prediction_rows),('fold_bootstrap.csv',draws)]:
        with (OUT/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for p in protected:assert sha(p)==before[str(p.relative_to(ROOT))]
    write('validation.json',{'protected_artifacts_unchanged':before,'baseline_coefficients_reproduced':True,
          'exact_seed_set':sorted(EXPECTED),'all_fit_training_runs_A100':True,'source_sha256':sha(__file__)})
    shutil.copy2(__file__,OUT/Path(__file__).name)
    small={name:{'mem':x['mem'],'gen':x['gen'],'p_star':x['divergence_prime'],'p149':x['predictions'][149],
                 'p197':x['predictions'][197],'delta':x['calibrated_delta_dex_same_historical_onsets']} for name,x in comparisons.items()}
    print(json.dumps({'comparisons':small,'bootstrap':bootstrap},indent=2),flush=True)

if __name__=='__main__':main()
