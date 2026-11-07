#!/usr/bin/env python3
"""Verify and package the selected cell-mean clock laws and their consequences."""
from pathlib import Path
import json, math, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from grokking_capacity.analysis._figure_style import save_paper_figure
from check_law_complete_ten_seeds import extract, K
from plot_cone_matched_aggregations import ROOT, C_OLD, CENTRAL, branches, save_json, sha

OUT=ROOT/'results/paper_revision_20261107/models'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    current=json.loads((ROOT/'results/cone_cellmeans_min_20261106/fit.json').read_text())
    law=current['law'];C=current['C_current']
    _,_,fresh=extract()
    frozen=pd.read_csv(ROOT/'results/law_complete_ten_seed_sensitivity_20261106/selected_runs.csv').sort_values('uuid').reset_index(drop=True)
    pd.testing.assert_frame_equal(pd.DataFrame(fresh).sort_values('uuid').reset_index(drop=True),frozen,check_dtype=False,rtol=1e-12)
    prior=json.loads((ROOT/'results/cone_matched_aggregations_20261106/summary.json').read_text())['laws']['observed_mean']
    for kind,keys in [('mem',['ln_b','gamma_original','r2','n_observations']),('gen',['lnA','a','beta','r2','n_observations'])]:
        for key in keys:assert law[kind][key]==prior[kind][key]
    assert law['fold']==prior['fold']
    boot=pd.read_csv(ROOT/'results/cone_matched_aggregations_20261106/fold_bootstrap.csv')['observed_mean']
    interval=np.quantile(boot.dropna(),[.025,.975])
    np.testing.assert_allclose(interval,prior['fold_prime_bootstrap_95'])
    pd.DataFrame({'fold':boot}).to_csv(OUT/'fold_prime_bootstrap.csv',index=False)
    # Full-range descriptive law also uses one log arithmetic mean per cell.
    stats=ROOT/'results/paper_revision_20261107/statistics'
    manifest=pd.read_csv(stats/'cohort_runs.csv')
    events=pd.read_json(stats/'raw_events.json')
    data=manifest[(manifest.config=='central')&(manifest.kind=='mem')].merge(events,on='uuid',validate='one_to_one')
    retained=data.dropna(subset=['mem99'])
    means=retained.groupby(['p','param_count'],as_index=False).agg(t=('mem99','mean'),n_observed=('mem99','size'))
    means['f']=[K(p)/(C*P) for p,P in zip(means.p,means.param_count)]
    X=np.column_stack([np.ones(len(means)),means.f]);y=np.log(means.t)
    lb,gm=np.linalg.lstsq(X,y,rcond=None)[0];r2=1-sum((y-X@[lb,gm])**2)/sum((y-y.mean())**2)
    full={'ln_b':float(lb),'b':float(np.exp(lb)),'gamma_current_C':float(gm),'r2_log':float(r2),'cells':len(means),
          'observed_runs':len(retained),'censored_runs':len(data)-len(retained),'observed_f_range':[float(means.f.min()),float(means.f.max())]}
    means.to_csv(OUT/'full_range_mem_cells.csv',index=False)
    for mode in ['inverse_capacity','capacity_fraction']:
        fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
        for i,(p,dd) in enumerate(means.groupby('p')):
            dd=dd.sort_values('param_count',ascending=False);color=plt.get_cmap('viridis')(i/9)
            xx=1/(C*dd.param_count) if mode=='inverse_capacity' else dd.f
            ax.plot(xx,dd.t,'o-',color=color,ms=3,lw=.8,label=f'$p={p}$')
        if mode=='capacity_fraction':
            ff=np.linspace(means.f.min(),means.f.max(),300)
            ax.plot(ff,np.exp(lb+gm*ff),color='#c7353b',ls='--',lw=1.6,label='Pooled cell-mean fit')
        ax.set(yscale='log',xlabel='1 / model capacity (1/bits)' if mode=='inverse_capacity' else 'Capacity fraction $f$',ylabel='Mean epochs to memorise')
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8)
        notes=['Arithmetic means of observed training-99% events; one point per prime/parameter cell; censored seed times excluded.',
               f'{len(means)} cells from {len(retained):,} observed A100 runs; {len(data)-len(retained):,} censored runs excluded. $C={C:.6f}$ bits/parameter.']
        if mode=='capacity_fraction':notes.append(f'Full-range fit: $T={np.exp(lb):.3f}\\exp({gm:.3f} f)$; log-space $R^2={r2:.4f}$. Crossing-window fit is reported separately.')
        save_paper_figure(fig,OUT/f'mem_{mode}.pdf',notes=[fig.text(0,0,n,fontsize=8) for n in notes]);plt.close(fig)
    consequences=[]
    for p in sorted(set(CENTRAL)|{105,149,151,157,163,167,173,179,181,191,193,197,199}):
        lo,hi=branches(p,law);kap=1+p/(p-1)+p/((p+2)*math.log(p+2))
        x=None if lo is None else law['mem']['gamma_original']*K(p)/(C_OLD*lo)
        consequences.append({'p':p,'lower_P':lo,'upper_P':hi,'x_plus':x,
             'f_lower':None if x is None else x/law['mem']['gamma_current_C'],
             'f_upper':None if hi is None else K(p)/(C*hi),
             'kappa':kap,'local_exponent':None if x is None else kap+(law['gen']['beta']*kap-law['gen']['a'])/(x-law['gen']['beta'])})
    fp=law['fold'];beta=law['gen']['beta'];gamma=law['mem']['gamma_current_C']
    row167=next(r for r in consequences if r['p']==167)
    R=lambda p:law['gen']['lnA']-law['mem']['ln_b']+law['gen']['a']*math.log(p)-beta*math.log(gamma*K(p)/C)
    dr=R(167)-beta*(1-math.log(beta))
    mid=[r for r in consequences if r['p'] in CENTRAL]
    observed=pd.read_csv(stats/'central/predictiveness.csv')
    result={**current,'law':law,'fold_prime_bootstrap_95':interval.tolist(),'bootstrap_finite':int(boot.notna().sum()),
       'bootstrap_method':'Paired resampling of the ten prime clusters; observed cell means and eligibility fixed. This interval omits unobserved completion-time and seed-population uncertainty.',
       'full_range_mem':full,'fold_fraction':beta/gamma,'fold_parameters':gamma*K(fp)/(C*beta),
       'local_consequences':consequences,
       'predicted_logP_logp_slope':float(np.polyfit(np.log([r['p'] for r in mid]),np.log([r['lower_P'] for r in mid]),1)[0]),
       'observed_logP_logp_slope':float(np.polyfit(np.log(observed.p),np.log(observed.empirical_onset_x),1)[0]),
       'fold_expansion_at_167':{'exact_x_minus_beta':row167['x_plus']-beta,'leading_sqrt':math.sqrt(2*beta*dr),'linear_correction':2*dr/3},
       'nats_example_P1e5_f0275':1e5*C*.275*math.log(2),
       'verification':{'fresh_registry_selection_matches_frozen':True,'bootstrap_matches_exact_current_coefficients':True,
                       'input_sha256':sha(ROOT/'results/cone_cellmeans_min_20261106/fit.json'),'source_sha256':sha(__file__)}}
    save_json(OUT/'models.json',result)
    for name in ['cone_min_cellmeans.pdf','cone_min_cellmeans.png','clock_fits.pdf','clock_fits.png']:
        shutil.copy2(ROOT/'results/cone_cellmeans_min_20261106'/name,OUT/name)
    print(json.dumps({k:result[k] for k in ['fold_prime_bootstrap_95','full_range_mem','fold_fraction','fold_parameters','predicted_logP_logp_slope','observed_logP_logp_slope','fold_expansion_at_167']},indent=2))


if __name__=='__main__':main()
