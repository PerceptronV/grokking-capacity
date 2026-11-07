#!/usr/bin/env python3
"""Analyze fresh matched H100 capacity and apply numerical-only paper changes.
Historical inputs are preserved. The before/ snapshot makes this idempotent.
"""
from pathlib import Path
from collections import defaultdict
import ast, copy, hashlib, json, math, os, re, shutil, sqlite3, sys
import numpy as np
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
OUT=ROOT/'results/capacity_h100_matched_20261103';PAPER=ROOT/'docs/papers/2026 Grokking Capacity';BEFORE=OUT/'before'
def save(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False))
def fit(x,y):
 s,b=np.polyfit(x,y,1);return dict(C=float(s),intercept=float(b),r2=float(1-np.sum((y-s*x-b)**2)/np.sum((y-y.mean())**2)))
def analyze():
 con=sqlite3.connect((OUT/'runs.db').resolve().as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
 rows=[dict(x) for x in con.execute("select * from runs where status='completed'")];assert len(rows)==96,len(rows)
 manifest=json.loads((OUT/'manifest.json').read_text());assert all(x['gpu_type']=='H100' for x in rows)
 assert {(x['dim'],x['n_samples'],x['seed']) for x in rows}=={(d,n,s) for d in manifest['dims'] for n in manifest['n_samples'] for s in manifest['seeds']}
 assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in manifest['source_hashes'].items()), 'Training source changed during campaign'
 cells=defaultdict(list);metric_errors=[];raw=[]
 for r in rows:
  assert r['dataset_type']=='random' and r['dropout']==.2 and r['weight_decay']==1 and r['depth']==2 and r['heads']==1
  with np.load(r['npz_path']) as z:
   bits=float(z['total_bits_memorized']);assert math.isfinite(bits);formula=r['n_samples']*(math.log2(115)-float(z['final_loss'])/math.log(2))
   metric_errors.append(abs(bits-formula));t=np.asarray(z['bits_trace'])*r['n_samples']
   drift=float(abs(t[-100:].mean()/t[-200:-100].mean()-1)) if len(t)>=200 else None
  rr=dict(dim=r['dim'],n=r['n_samples'],seed=r['seed'],P=r['param_count'],bits=bits,epochs=r['epochs_trained'],late_drift=drift,npz_path=r['npz_path'],uuid=r['uuid']);raw.append(rr)
  cells[r['dim'],r['seed']].append(rr)
 dims=manifest['dims'];seeds=manifest['seeds'];maxima=np.array([[max(x['bits'] for x in cells[d,s]) for s in seeds] for d in dims]);ps=np.array([cells[d,seeds[0]][0]['P'] for d in dims],float)
 y=maxima.mean(axis=1);f=fit(ps,y);rng=np.random.default_rng(20261103);bs=[]
 for _ in range(20000):
  ii=rng.integers(0,len(dims),len(dims))
  if len(set(ii))<2:continue
  jj=rng.integers(0,len(seeds),len(seeds));yy=maxima[ii][:,jj].mean(axis=1);bs.append(fit(ps[ii],yy)['C'])
 summary=dict(**f,C_through_origin=float(np.dot(ps,y)/np.dot(ps,ps)),intercept_fraction_smallest=float(f['intercept']/y.min()),CI95=np.quantile(bs,[.025,.975]).tolist(),bootstrap='20,000 width clusters with replacement, resampling the two paired seeds jointly across selected widths',n_runs=len(rows),dims=dims,seeds=seeds,parameters=ps.tolist(),maxima_by_seed=maxima.tolist(),mean_maxima=y.tolist(),per_seed_fits={str(s):fit(ps,maxima[:,j]) for j,s in enumerate(seeds)},max_formula_error_bits=max(metric_errors),horizon_reached=sum(x['epochs']==5000 for x in raw),plateau_peak_at_largest_n=[dict(dim=d,seed=s) for (d,s),v in cells.items() if max(v,key=lambda x:x['bits'])['n']==9300],max_late_drift=max(x['late_drift'] for x in raw if x['late_drift'] is not None),manifest_sha256=hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest())
 save('summary.json',summary);save('cells.json',raw)
 return summary,raw

def plots(summary,raw,*,write_manifest=True):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from grokking_capacity.analysis._figure_style import save_paper_figure
 from grokking_capacity.analysis.config_view import ConfigView
 from grokking_capacity.analysis.forecast_onset import _pick_group,CENTRAL_PRIMES
 from grokking_capacity.analysis.plots import _passes_filters
 from grokking_capacity.analysis._primitives import plot_saturation_epochs_vs_inverse_capacity,plot_saturation_time_vs_capacity_fraction
 media=PAPER/'media';C=summary['C'];r=C/2.16
 fig,ax=plt.subplots(figsize=(6,4.2));colors=plt.cm.viridis(np.linspace(.05,.9,len(summary['dims'])))
 for d,col in zip(summary['dims'],colors):
  cells=sorted([x for x in raw if x['dim']==d],key=lambda x:x['n'])
  ns=sorted(set(x['n'] for x in cells));ys=[np.mean([x['bits'] for x in cells if x['n']==n]) for n in ns]
  for seed in summary['seeds']:
   v=[x for x in cells if x['seed']==seed];ax.plot([x['n']*math.log2(115) for x in v],[x['bits'] for x in v],color=col,alpha=.3,lw=.7)
  ax.plot(np.array(ns)*math.log2(115),ys,'o-',color=col,label=f'd={d}',ms=4)
 xx=np.array([1000,9300])*math.log2(115);ax.plot(xx,xx,'--',color='grey',label='Dataset complexity')
 ax.set(xlabel='Dataset complexity (bits)',ylabel='Total memorisation (bits)',ylim=(0,1.2*max(x['bits'] for x in raw)));ax.legend(fontsize=8,ncol=2,loc='upper left');ax.grid(alpha=.25);fig.tight_layout();save_paper_figure(fig,media/'capacity_curves.pdf');plt.close(fig)
 fig,ax=plt.subplots(figsize=(6,4.2));ps=np.array(summary['parameters']);y=np.array(summary['mean_maxima']);mm=np.array(summary['maxima_by_seed'])
 for j in range(mm.shape[1]):ax.scatter(ps,mm[:,j],s=20,alpha=.5)
 ax.plot(ps,y,'o',color='black',label='Data');xx=np.array([0,ps.max()*1.05]);ax.plot(xx,C*xx+summary['intercept'],label=f'Fit: bits = {C:.3f} × params + {summary["intercept"]:.0f}')
 ax.text(.04,.96,f'C: {C:.3f} bits/param\n95% bootstrap: [{summary["CI95"][0]:.3f}, {summary["CI95"][1]:.3f}]\nIntercept: {summary["intercept"]:.0f} bits\nR²: {summary["r2"]:.3f}',transform=ax.transAxes,va='top',fontsize=9)
 ax.set(xlabel='Number of parameters',ylabel='Saturation memorisation (bits)');ax.legend(loc='lower right',fontsize=8);ax.grid(alpha=.25);fig.tight_layout();save_paper_figure(fig,media/'capacity_estimation.pdf',notes=list(ax.texts));plt.close(fig)
 view=ConfigView.from_yaml(ROOT/'configs/central.yaml',db_path=str(ROOT/'runs.db'));g=_pick_group(view);figure=view.intersection_figures[0]
 rows=[x for x in g.speed_runs if x['p'] in CENTRAL_PRIMES and _passes_filters(x,figure) and x.get('saturation_epoch') is not None and x.get('saturated') is not False]
 by=defaultdict(list)
 for x in rows:by[x['p'],x['dim']].append(x)
 plot_saturation_epochs_vs_inverse_capacity(by,C=C,save_path=str(media/'saturation_epochs_vs_inverse_capacity_all_primes.pdf'))
 from unittest.mock import patch
 with patch('matplotlib.pyplot.close'):
  fitnew=plot_saturation_time_vs_capacity_fraction(by,C=C,save_path=None)
  fig=plt.gcf()
  for ax in fig.axes:
   for label in ax.texts:label.set_text(label.get_text().replace(f'C = {C:.2f} bits/param',f'C = {C:.3f} bits/param'))
  save_paper_figure(fig,media/'saturation_time_vs_capacity_fraction_all_primes.pdf')
 plt.close(fig)
 frozen=json.loads((ROOT/'results/theory/forecast_onset.json').read_text())['mem_law_full_range']
 assert len(rows)==frozen['n_runs'] and np.isclose(fitnew[0]/r,frozen['gamma'],rtol=1e-9), 'Canonical speed cohort changed'
 if write_manifest:save('plot_manifest.json',dict(speed_rows=len(rows),speed_uuids=[x['uuid'] for x in rows],speed_fit_gamma=fitnew[0],speed_fit_b=fitnew[1],speed_fit_r2=fitnew[2],C=C,regenerated=['capacity_curves.pdf','capacity_estimation.pdf','saturation_epochs_vs_inverse_capacity_all_primes.pdf','saturation_time_vs_capacity_fraction_all_primes.pdf']))

def check_invariance(C):
 F=json.loads((ROOT/'results/theory/forecast_onset.json').read_text());g=F['mem_law']['gamma'];beta=F['gen_law']['beta'];a=F['gen_law']['a'];lnb=F['mem_law']['ln_b'];lnA=F['gen_law']['lnA'];r=C/2.16
 K=lambda p:.5*p*(p-1)*math.log2(p+2)
 def fold(c,gamma):return brentq(lambda p:lnA-lnb+a*math.log(p)-beta*math.log(gamma*K(p)/c)-beta*(1-math.log(beta)),2,300)
 audit=json.loads((PAPER/'review-audit.json').read_text());central=audit['central'];fresh=json.loads((OUT/'main-test-recheck.json').read_text());assert fresh['central']==central;assert fresh['crossing']==audit['crossing'];diff=[]
 for p in range(97,168):
  for P in np.geomspace(10000,1e7,20):diff.append(abs(g*K(p)/(2.16*P)-(g*r)*K(p)/(C*P)))
 result=dict(C=C,ratio=r,f_scale=1/r,gamma_old=g,gamma_new=g*r,q=g/2.16,fold_old=fold(2.16,g),fold_new=fold(C,g*r),max_log_time_difference=max(diff),main_test_before={k:v for k,v in central.items() if k!='cells'},main_test_after={k:v for k,v in central.items() if k!='cells'},main_test_change=0.,explanation='Measured P-space crossing and empirical onset do not use capacity. Same retained observations; gamma and all fit-window bounds transformed together.',fit_window_mem=[.08/r,1/r],fit_window_gen=[.08/r,1/r])
 assert result['max_log_time_difference']<1e-12;assert abs(result['fold_old']-result['fold_new'])<1e-10
 result['capacity_only_thresholds']=[dict(p=x['p'],P_mem_before=K(x['p'])/2.16,P_mem_after=K(x['p'])/C,P_onset=x['empirical'],onset_over_P_mem=x['empirical']*C/K(x['p'])) for x in central['cells']]
 save('invariance.json',result)
 # A separate transformed copy preserves historical forecasts and registrations.
 new={k:copy.deepcopy(F[k]) for k in ['mem_law','mem_law_full_range','gen_law']};new['C_convention']=C;new['source_artifact']='results/theory/forecast_onset.json'
 for k in ['mem_law','mem_law_full_range']:
  m=new[k]
  for key in ['gamma']:
   m[key]*=r
  for key in ['gamma_ci','gamma_ci95']:
   if key in m:m[key]=[z*r for z in m[key]]
  for key in ['f_range']:
   if key in m:m[key]=[z/r for z in m[key]]
  for key in ['curvature_q']:
   if key in m:m[key]*=r*r
  for key in ['curvature_q_ci95']:
   if key in m:m[key]=[z*r*r for z in m[key]]
 new['gen_law']['window_f']=[z/r for z in new['gen_law']['window_f']]
 new['capacity_update']=result
 save('forecast_coefficients.json',new)
 return result

def edit_paper(S,I):
 C=S['C'];r=C/2.16;c=f'{C:.3f}';edits=[]
 def put(rel,fn):
  old=(BEFORE/rel).read_text();current=(PAPER/rel).read_text()
  prior=json.loads((OUT/'source_edits.json').read_text()) if (OUT/'source_edits.json').exists() else []
  allowed={hashlib.sha256(old.encode()).hexdigest()}|{e['after_sha256'] for e in prior if e['file']==rel}
  assert hashlib.sha256(current.encode()).hexdigest() in allowed, f'Concurrent source change: {rel}'
  new=fn(old)
  (PAPER/rel).write_text(new);edits.append(dict(file=rel,before_sha256=hashlib.sha256(old.encode()).hexdigest(),after_sha256=hashlib.sha256(new.encode()).hexdigest()))
 def main(t):
  # Only remove provenance/convention sentences explicitly authorized by the user.
  t=t.replace(' Two things about this constant should be said plainly.','')
  t=re.sub(r' The runs behind the fit predate.*?\(\\cref\{sec:exp-capacity\}\)\.','',t)
  t=re.sub(r' And every claim-bearing statistic.*?where an alternative value would change a number\.','',t)
  t=re.sub(r' The runs that produced the quoted.*?All statistics that compare predicted with empirical onsets are invariant to the constant\.','',t)
  start=t.index('The defaults protocol reproduces the archived')
  end=t.index('\\input{theory/consequences}',start)
  t=t[:start]+t[end:]
  t=t.replace(r'\{10, 12, 14, 16, 18, 20, 22\}',r'\{10, 12, 14, 16, 18, 22\}')
  t=t.replace('except with dropout $0$','except with dropout $0.2$').replace('and $0$ in capacity runs','and $0.2$ in capacity runs')
  t=t.replace(r', and well below it at the matched constant of \cref{sec:exp-ood}', '')
  t=t.replace('Capacity constant, re-measured.', 'Capacity constant.')
  t=t.replace('a single seed (42) per cell','two seeds (20261103, 20261104) per cell')
  t=t.replace('(seven widths, eight dataset sizes,\nsingle seed)','(six widths, eight dataset sizes,\ntwo seeds)')
  # Drop defaults comparison so the table describes the fresh matched campaign only.
  t=re.sub(r'^capacity defaults \(dropout.*\n','',t,flags=re.M)
  t=re.sub(r'(grokking-run settings.*? &) \$0\.82\$ & \$\[0\.79, 0\.86\]\$ & \$0\.998\$',lambda m:m[1]+f' ${c}$ & $[{S["CI95"][0]:.3f}, {S["CI95"][1]:.3f}]$ & ${S["r2"]:.3f}$',t)
  # Capacity constants only; preserve the unrelated 2.16e6 empirical onset.
  t=t.replace(r'C_{\text{model}} \approx 2.16',rf'C_{{\text{{model}}}} \approx {c}')
  t=t.replace(r'C_{\text{model}}=2.16',rf'C_{{\text{{model}}}}={c}')
  t=t.replace(r'f\in[0,0.25]',rf'f\in[0,{.25/r:.3f}]').replace(r'f \in [0, 0.25]',rf'f \in [0, {.25/r:.3f}]')
  return t
 put('paper-neurips-revised.tex',main)
 def checklist(t):
  t=t.replace("the capacity constant's original runs are unrecoverable and it is re-measured with all onset statistics shown to be invariant to it; ",'')
  t=t.replace('$0.01$ for capacity runs as the constant $C_{\\text{model}}$ was measured at the lower setting','$1.0$ for capacity runs')
  return t.replace('$0$ in capacity runs','$0.2$ in capacity runs')
 put('checklist.tex',checklist)
 def section(t):return t.replace('0.08',f'{.08/r:.3f}').replace('42.3',f'{I["gamma_new"]:.2f}')
 put('theory/theory-section.tex',section)
 def consequences(t):
  t=t.replace('C = 2.16',f'C = {c}').replace('gamma = 42.30',f'gamma = {I["gamma_new"]:.4f}')
  t=t.replace('1.109/42.30 = 0.0262',f'1.109/{I["gamma_new"]:.4f} = {1.1088598736062016/I["gamma_new"]:.4f}')
  t=t.replace(r'0.003$--$0.018',f'{.003/r:.4f}$--${.018/r:.4f}')
  # Preserve the physical load of the illustrative calculation; flag its pre-existing arithmetic separately.
  t=t.replace('$f = 0.1$',f'$f = {.1/r:.3f}$')
  return t
 put('theory/consequences.tex',consequences)
 def extra(t):
  pairs={r'20\to42':f'{20*r:.2f}\\to{42*r:.2f}',r'20\!\to\!42':f'{20*r:.2f}\\!\\to\\!{42*r:.2f}',r'45.5\to23.8':f'{45.5*r:.2f}\\to{23.8*r:.2f}','q=+71':f'q=+{71.10670020573298*r*r:.2f}','11.0':f'{10.960686837450506*r:.2f}',"c' = 879":f"c' = {878.7800268281326*r:.2f}",r'\le 1.3':rf'\le {1.3*r:.2f}'}
  for old,new in pairs.items():t=t.replace(old,new)
  return t
 put('theory/extra.tex',extra)
 def principled(t):
  pairs={'[31, 42]':f'[{98*C/math.log2(99):.2f}, {140*C/math.log2(141):.2f}]','30.0':f'{30.034299000813597*r:.2f}','42.3':f'{I["gamma_new"]:.2f}',r'\approx 36':rf'\approx {36*r:.2f}',r'< 36':f'< {36*r:.2f}','11.0':f'{10.960686837450506*r:.2f}',"c' = 879":f"c' = {878.7800268281326*r:.2f}",r'\le 1.3':rf'\le {1.3*r:.2f}',r'$500$--$1600$':f'${500*r:.1f}$--${1600*r:.1f}$',r'$48.5$ to $19.8$ $[18.1, 21.1]$':f'${48.49903008103672*r:.2f}$ to ${19.772792543251402*r:.2f}$ $[{18.082680734801354*r:.2f}, {21.078491217772488*r:.2f}]$',r'coefficient $+1.3$':f'coefficient $+{1.3*r*r:.2f}$',r'$+71$':f'$+{71.10670020573298*r*r:.2f}$',r'q = +71':f'q = +{71.10670020573298*r*r:.2f}',r'= 11 + 879/':f'= {10.960686837450506*r:.2f} + {878.7800268281326*r:.2f}/',r'$20$--$24$':f'${20*r:.2f}$--${24*r:.2f}$',r'f \approx 0.03':rf'f \approx {.03/r:.3f}',r'f \approx 0.13':rf'f \approx {.13/r:.3f}',r'f = 0.18':rf'f = {.18/r:.3f}',r'f \ge 0.20':rf'f \ge {.20/r:.3f}'}
  for old,new in pairs.items():t=t.replace(old,new)
  return t
 put('theory/principled.tex',principled)
 save('source_edits.json',edits)
if __name__=='__main__':
 S,raw=analyze();I=check_invariance(S['C']);plots(S,raw);edit_paper(S,I);print(json.dumps(S,indent=2))
