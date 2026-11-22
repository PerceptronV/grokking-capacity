#!/usr/bin/env python3
"""Recompute the matched H100 capacity estimate and optional figure assets.

The measurement cohort, estimator, and bootstrap are unchanged. This command
writes analysis outputs; it does not patch manuscript prose or checklist answers.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from grokking_capacity.analysis.timescale_laws import normalise_timescale_laws

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'results/capacity_h100_matched_20261103'
PAPER = ROOT / 'docs/papers/2026 Grokking Capacity'


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
 frozen=normalise_timescale_laws(json.loads((ROOT/'results/theory/forecast_onset.json').read_text()))['mem_law_full_range']
 assert len(rows)==frozen['n_runs'] and np.isclose(fitnew[0]/r,frozen['gamma'],rtol=1e-9), 'Canonical speed cohort changed'
 if write_manifest:save('plot_manifest.json',dict(speed_rows=len(rows),speed_uuids=[x['uuid'] for x in rows],speed_fit_gamma=fitnew[0],speed_fit_B=fitnew[1],speed_fit_r2=fitnew[2],C=C,regenerated=['capacity_curves.pdf','capacity_estimation.pdf','saturation_epochs_vs_inverse_capacity_all_primes.pdf','saturation_time_vs_capacity_fraction_all_primes.pdf']))

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures', action='store_true',
                        help='Also regenerate the capacity and memorisation figure assets.')
    args = parser.parse_args(argv)
    summary, raw = analyze()
    if args.figures:
        plots(summary, raw)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
