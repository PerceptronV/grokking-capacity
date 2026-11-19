#!/usr/bin/env python3
"""Independent read-only numerical checks; outputs only to the current audit."""
from grokking_capacity.analysis.timescale_laws import normalise_timescale_laws
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv, hashlib, json, math, sqlite3, sys
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import spearmanr, kendalltau
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/paper_rigour_new_conventions_20261106'
sys.path.insert(0,str(ROOT/'scripts'))
from check_law_complete_ten_seeds import extract

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
def first(x,threshold):
 i=np.flatnonzero(np.asarray(x)>=threshold);return int(i[0]+1) if len(i) else None

def traces():
 runs=json.loads((ROOT/'results/cone_ten_seeds_h100_20261106/figure/runs.json').read_text())
 def check(r):
  errors=[];p=Path(r['trace_path'])
  try:
   with np.load(p,allow_pickle=False) as z:
    t,v,v99=first(z['train_acc'],99),first(z['val_acc'],98),first(z['val_acc'],99)
    n=len(z['val_acc']);delay=(max(0,v-t) if t else 0) if v else None
    state='unresolved' if delay is None else ('positive' if delay>0 else 'zero')
    for k,x in [('train99',t),('val98',v),('grokking_epoch',v99),('delay',delay),('state',state),('followup_epochs',n)]:
     if r[k]!=x:errors.append([k,r[k],x])
    if not np.isfinite(z['train_acc']).all() or not np.isfinite(z['val_acc']).all():errors.append(['nonfinite'])
   if sha(p)!=r['trace_sha256']:errors.append(['trace_hash'])
   if r['param_count']!=32*r['dim']**2+(2*r['p']+9)*r['dim']:errors.append(['parameter_count'])
  except Exception as e:errors.append(['read_error',str(e)])
  return {'uuid':r['uuid'],'errors':errors}
 with ThreadPoolExecutor(max_workers=8) as pool:checks=list(pool.map(check,runs))
 by=defaultdict(list)
 for r in runs:by[r['p'],r['dim']].append(r)
 cells=[]
 for (p,d),rr in sorted(by.items()):
  delays=[r['delay'] for r in rr];known=[x for x in delays if x is not None]
  val=0 if 0 in known else (min(known) if len(known)==10 else None)
  state='unresolved' if val is None else ('positive' if val>0 else 'zero')
  cells.append({'p':p,'dim':d,'P':rr[0]['param_count'],'min':val,'state':state,'unresolved':delays.count(None),'hardware':sorted({r['gpu_type'] for r in rr}),'seeds':sorted(r['seed'] for r in rr)})
 old=json.loads((ROOT/'results/cone_ten_seeds_h100_20261106/figure/cells.json').read_text());lookup={(r['p'],r['dim']):r for r in old}
 disagreements=[c for c in cells if any(c[k]!=lookup[c['p'],c['dim']][j] for k,j in [('state','state'),('min','minimum_delay'),('unresolved','unresolved_seeds')])]
 save('cone_cells_independent.json',cells)
 result={'runs_checked':len(checks),'trace_errors':[r for r in checks if r['errors']],'unique_uuids':len({r['uuid'] for r in runs}),'cells':len(cells),'cell_disagreements':disagreements,'cell_counts':dict(Counter(c['state'] for c in cells)),'unresolved_seeds':sum(c['unresolved'] for c in cells),'cells_with_unresolved_seeds':sum(c['unresolved']>0 for c in cells),'hardware_cells':dict(Counter(tuple(c['hardware']) for c in cells)),'invalid_seed_cells':[c for c in cells if c['seeds']!=list(range(42,52))],'mixed_hardware_cells':[c for c in cells if len(c['hardware'])!=1]}
 result['hardware_cells']={str(k):v for k,v in result['hardware_cells'].items()}
 save('cone_raw_verification.json',result);print('CONE', {k:v for k,v in result.items() if k!='trace_errors'},flush=True)

def laws():
 C,_,details=extract();data=pd.DataFrame(details).sort_values('uuid').reset_index(drop=True)
 frozen=pd.read_csv(ROOT/'results/law_complete_ten_seed_sensitivity_20261106/selected_runs.csv').sort_values('uuid').reset_index(drop=True)
 pd.testing.assert_frame_equal(data,frozen,check_dtype=False,rtol=1e-12)
 con=sqlite3.connect((ROOT/'runs.db').as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
 registry={r['uuid']:dict(r) for r in con.execute('select * from runs')};con.close()
 def check(r):
  row=registry[r['uuid']];path=Path(row['npz_path']);err=[]
  try:
   with np.load(path,allow_pickle=False) as z:
    event=first(z['train_acc_trace'] if r['kind']=='mem' else z['val_acc'],99)
    if bool(r['observed'])!=(event is not None):err.append('observed_flag')
    if event is not None and event!=r['event_epoch']:err.append('epoch')
   return {'uuid':r['uuid'],'kind':r['kind'],'path':str(path),'sha256':sha(path),'event':event,'errors':err}
  except Exception as e:return {'uuid':r['uuid'],'errors':[str(e)]}
 with ThreadPoolExecutor(max_workers=8) as pool:checked=list(pool.map(check,details))
 pd.DataFrame(checked).to_csv(OUT/'law_trace_manifest.csv',index=False)
 fitted={}
 for kind in ['mem','gen']:
  d=data[(data.kind==kind)&data.observed];X=np.column_stack([np.ones(len(d)),d.f_original]) if kind=='mem' else np.column_stack([np.ones(len(d)),np.log(d.p),-np.log(d.P)])
  y=np.log(d.event_epoch);coef=np.linalg.lstsq(X,y,rcond=None)[0];r2=1-sum((y-X@coef)**2)/sum((y-y.mean())**2)
  fitted[kind]={'coefficients':coef.tolist(),'r2':float(r2),'N':len(d),'censored':int(((data.kind==kind)&~data.observed).sum())}
 lb,gamma=fitted['mem']['coefficients'];la,a,b=fitted['gen']['coefficients'];q=gamma/C
 K=lambda p:p*(p-1)/2*math.log2(p+2)
 R=lambda p:la-lb+a*math.log(p)-b*math.log(q*K(p))
 star=brentq(lambda p:R(p)-b*(1-math.log(b)),2,10000)
 capacity=json.loads((ROOT/'results/capacity_h100_matched_20261103/summary.json').read_text())['C']
 forecasts=[]
 for p in [97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,251]:
  if R(p)<b*(1-math.log(b)):lo=hi=None
  else:
   xp=brentq(lambda x:x-b*math.log(x)-R(p),b,1000)
   xm=brentq(lambda x:x-b*math.log(x)-R(p),1e-14,b)
   lo=q*K(p)/xp;hi=q*K(p)/xm
  forecasts.append({'p':p,'lower':lo,'upper':hi,'log10_lower':math.log10(lo) if lo else None})
 pd.DataFrame(forecasts).to_csv(OUT/'current_law_forecasts.csv',index=False)
 existing=normalise_timescale_laws(json.loads((ROOT/'results/cone_observed_logtime_min_20261106/fit.json').read_text()))['law']
 assert np.allclose(fitted['mem']['coefficients'],[existing['mem']['ln_B'],existing['mem']['gamma_original']])
 assert np.allclose(fitted['gen']['coefficients'],[existing['gen']['lnA'],existing['gen']['a'],existing['gen']['b']])
 # Expansion at a fold: x=b +/- sqrt(2 b dR)+(2/3)dR+O(dR^1.5).
 expansion=[]
 for dr in [1e-3,1e-5,1e-7]:
  xp=brentq(lambda x:x-b*math.log(x)-(b*(1-math.log(b))+dr),b,100)
  expansion.append({'delta_R':dr,'residual_divided_by_delta_R':(xp-b-math.sqrt(2*b*dr))/dr})
 save('law_raw_verification.json',{'fresh_selection_matches_frozen_input':True,'eligible_runs':len(data),'trace_errors':[r for r in checked if r['errors']],'hardware':data.groupby(['kind','gpu_type']).size().to_dict().__str__(),'fits':fitted,'fold':star,'gamma_current':gamma*capacity/C,'fold_fraction':b/(gamma*capacity/C),'fold_parameters':q*K(star)/b,'fold_expansion_check':expansion,'claim_2200_nats_example':{'P':1e5,'f':.275,'C':capacity,'nats':capacity*.275*1e5*math.log(2)}})
 print('LAW',fitted,'fold',star,flush=True)

def mechanisms():
 cap=json.loads((ROOT/'results/capacity_h100_matched_20261103/cells.json').read_text());errors=[]
 for r in cap:
  with np.load(r['npz_path']) as z:
   if not np.isclose(z['total_bits_memorized'].item(),r['bits'],rtol=1e-12):errors.append(r['uuid'])
 frame=pd.DataFrame(cap);maxima=frame.groupby(['dim','seed']).bits.max().groupby('dim').mean();P=frame.groupby('dim').P.first().reindex(maxima.index)
 slope,intercept=np.polyfit(P,maxima,1)
 norm=json.loads((ROOT/'results/normfrozen_h100_paired_20261103/analysis.json').read_text());nr={}
 for arm in ['frozen','unfrozen']:
  rr=norm[arm]['runs'];events=[]
  for r in rr:
   with np.load(r['npz_path']) as z:t=first(z['val_acc'],99)
   if t!=r['grokking_epoch']:errors.append(r['uuid'])
   events.append({**r,'T':t})
  df=pd.DataFrame(events);lam=df[df.dim==128].groupby('weight_decay').T if False else df[df.dim==128].groupby('weight_decay')['T'].mean()
  widths=df[df.weight_decay==1].groupby('total_parameters')['T'].mean()
  nr[arm]={'lambda_slope':float(np.polyfit(np.log(lam.index),np.log(lam.values),1)[0]),'b':float(-np.polyfit(np.log(widths.index),np.log(widths.values),1)[0]),'n':len(df),'hardware':sorted(df.gpu_type.unique())}
 wd=json.loads((ROOT/'results/weight_decay_matched_20261103/summary.json').read_text());con=sqlite3.connect((ROOT/'runs.db').as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
 prot=wd['protocol'];sql='select * from runs where '+' and '.join(k+'=?' for k in prot);rr=[dict(r) for r in con.execute(sql,list(prot.values())) if r['dim'] in wd['dims'] and r['weight_decay'] in wd['lambdas'] and r['seed'] in wd['seeds']];con.close()
 for r in rr:
  with np.load(r['npz_path']) as z:t=first(z['val_acc'],99)
  if t!=r['grokking_epoch']:errors.append(r['uuid'])
 df=pd.DataFrame(rr);slopes=[]
 for d in wd['primary']['dims']:
  times=df[df.dim==d].groupby('weight_decay').grokking_epoch.mean()
  slopes.append(np.polyfit(np.log(times.index),np.log(times),1)[0])
 save('mechanisms_verification.json',{'capacity':{'runs':len(cap),'widths':len(P),'C':float(slope),'intercept':float(intercept),'matches_summary':bool(np.isclose(slope,json.loads((ROOT/'results/capacity_h100_matched_20261103/summary.json').read_text())['C']))},'norm_freezing':nr,'weight_decay':{'runs':len(rr),'hardware':sorted(df.gpu_type.unique()),'slope':float(np.median(slopes)),'included_widths':wd['primary']['dims'],'censored':int(df.grokking_epoch.isna().sum())},'trace_errors':errors})
 print('MECHANISMS',nr,'WD',np.median(slopes),flush=True)

if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True)
 for part in sys.argv[1:] or ['traces','laws','mechanisms']:globals()[part]()
