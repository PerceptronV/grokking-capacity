#!/usr/bin/env python3
"""Preview the complete-ten-observed-seed law on unchanged cone observations."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from cone_data import law_onset
from grokking_capacity.analysis._figure_style import save_paper_figure

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/law_complete_ten_seed_sensitivity_20261106'
OUT=BASE/'cone_preview'
DATA=ROOT/'results/cone_ten_seeds_h100_20261106/figure/cells.json'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    protected=json.loads((BASE/'validation.json').read_text())['protected_artifacts_unchanged']
    for rel,digest in protected.items():assert sha(ROOT/rel)==digest
    data_hash=sha(DATA);fit_hash=sha(BASE/'comparison.json')
    cells=json.loads(DATA.read_text());sensitivity=json.loads((BASE/'comparison.json').read_text())
    fit=sensitivity['comparisons']['complete_both'];fold=fit['divergence_prime']
    coeff={'C_convention':sensitivity['C_current'],
           'mem_law':{'gamma':fit['mem']['gamma_current_C'],'ln_b':fit['mem']['ln_b']},
           'gen_law':fit['gen']}
    assert len(cells)==1736
    assert all(c['n_seeds']==10 and c['seeds']==list(range(42,52)) for c in cells)
    assert all(c['gpu_type']=='H100' for c in cells if c['p']>=149)
    assert np.isclose(law_onset(197,coeff),fit['predictions']['197']['raw_crossing_P'],rtol=1e-12)
    ps=np.linspace(97,fold-1e-7,600);curve=np.array([law_onset(p,coeff) for p in ps])
    norm=LogNorm(1,1500);cmap=plt.get_cmap('viridis')
    logp=np.log10([c['param_count'] for c in cells]);margin=.05*(logp.max()-logp.min())
    ylim=(10**(logp.min()-margin),10**(logp.max()+margin))
    handles=[Line2D([],[],marker='o',ls='',markerfacecolor=cmap(.55),markeredgecolor='#202d38',markersize=5,label='Positive delay in all ten seeds'),
             Line2D([],[],marker='o',ls='',markerfacecolor='none',markeredgecolor='#9299a2',markersize=5,label='At least one observed zero delay'),
             Line2D([],[],marker='x',ls='',color='#b35d12',markersize=5,label='At least one unresolved seed'),
             Line2D([],[],color='#c7353b',lw=1.7,label='Law crossing: complete-ten-seed A100 fit'),
             Line2D([],[],color='#c7353b',lw=1.1,ls='--',label=rf'$p^*={fold:.1f}$')]
    metadata={}
    for split in [False,True]:
        fig,axes=plt.subplots(1,2 if split else 1,figsize=(12.8,5.4) if split else (7.1,5.3),
                              squeeze=False,sharey=True,layout='constrained')
        panels=[('A100',[c for c in cells if c['gpu_type']=='A100']),
                ('H100',[c for c in cells if c['gpu_type']=='H100'])] if split else [('Both',cells)]
        counts=[]
        for ax,(hardware,subset) in zip(axes.flat,panels):
            positive=[c for c in subset if c['state']=='positive'];zero=[c for c in subset if c['state']=='zero']
            censored=[c for c in subset if c['unresolved_seeds']>0]
            ax.scatter([c['p'] for c in zero],[c['param_count'] for c in zero],s=12,facecolors='none',
                       edgecolors='#9299a2',linewidths=.5,alpha=.65,zorder=1)
            ax.scatter([c['p'] for c in positive],[c['param_count'] for c in positive],
                       c=[c['mean_delay'] for c in positive],cmap=cmap,norm=norm,s=17,
                       edgecolors='#202d38',linewidths=.3,zorder=3)
            ax.scatter([c['p'] for c in censored],[c['param_count'] for c in censored],
                       marker='x',s=24,linewidths=.8,color='#b35d12',zorder=5)
            ax.plot(ps,curve,color='#c7353b',lw=1.7,zorder=4)
            ax.axvline(fold,color='#c7353b',ls='--',lw=1.1,zorder=2)
            ax.set(yscale='log',xlim=(95,207),ylim=ylim,
                   xlabel=(hardware+' observations — prime $p$') if split else 'Prime $p$')
            ax.set_xticks([97,113,127,139,149,163,179,199]);ax.tick_params(labelsize=9)
            ax.grid(axis='y',which='major',color='#d4d8de',alpha=.55,lw=.6);ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)
            counts.append({'hardware':hardware,'cells':len(subset),'states':dict(Counter(c['state'] for c in subset)),
                           'cells_with_unresolved_markers':len(censored)})
        axes.flat[0].set_ylabel('Parameters $P$')
        cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=list(axes.flat),fraction=.035,pad=.025)
        cb.set_label('Mean delay in complete positive cells (epochs)',fontsize=9);cb.ax.tick_params(labelsize=9)
        fig.legend(handles=handles,fontsize=8,frameon=False)
        note=fig.text(.5,0,
          'Curve: complete-ten-seed A100 fit at primes 97–139; 67 memorisation and 269 generalisation cells.\n'
          'Points: 1,407 A100 cells and 329 H100 cells; ten seeds per cell, one hardware cohort per cell.\n'
          'All observations at primes ≥149 are H100. Measurements end at prime 199.',
          fontsize=7.5 if not split else 8,ha='center')
        name='cone_complete_ten_fit'+('_by_hardware' if split else '')
        save_paper_figure(fig,OUT/(name+'.pdf'),notes=[note],max_columns=2,dpi=220)
        plt.close(fig)
        assert sum(c['cells'] for c in counts)==1736
        assert sum(c['cells_with_unresolved_markers'] for c in counts)==102
        metadata[name]={'panels':counts,'pdf_sha256':sha(OUT/(name+'.pdf')),'png_sha256':sha(OUT/(name+'.png'))}
    with (OUT/'curve_points.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['prime','raw_law_crossing_parameters']);writer.writerows(zip(ps,curve))
    for rel,digest in protected.items():assert sha(ROOT/rel)==digest
    assert sha(DATA)==data_hash and sha(BASE/'comparison.json')==fit_hash
    report={'purpose':'Separate exploratory preview; published cone, paper and pointer unchanged.',
            'law_fit':'Complete-ten-observed-seed A100 timescale fits; original 99% thresholds and fitting windows.',
            'curve':'Raw timescale crossing, without the empirical onset-offset calibration.',
            'cone_classification':'Unchanged: online training99 and validation98, with all unresolved seeds explicitly marked.',
            'fold':fold,'new_curve_p197':law_onset(197,coeff),'data_sha256':data_hash,'fit_sha256':fit_hash,
            'all_points_retained':True,'protected_publication_artifacts_unchanged':True,
            'hardware_by_prime':[{ 'p':p,'cells':dict(Counter(c['gpu_type'] for c in cells if c['p']==p))}
                                 for p in sorted({c['p'] for c in cells})],
            'figures':metadata,'source_sha256':sha(__file__)}
    (OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    shutil.copy2(__file__,OUT/Path(__file__).name)
    print(json.dumps({'fold':fold,'cells':len(cells),'artifacts':list(metadata),'publication_unchanged':True},indent=2))

if __name__=='__main__':main()
