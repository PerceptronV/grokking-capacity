#!/usr/bin/env python3
"""Render the cone from explicit hardware cohorts, unique seeds and censored clocks.

Defaults to both learning-rate panels. --fixed-only selects the paper's opening
figure. --final uses the latest published audited cohort (including later extensions).
--publish copies the PDF into the paper; otherwise outputs stay in the
analysis directory. --reuse reads already audited rows/cells without rescoring.
"""
from pathlib import Path
import argparse,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from grokking_capacity.experiments.cone_workflow.cone_data import ROOT, OUT, build, law_coefficients, law_onset, write_json
from grokking_capacity.analysis._figure_style import save_paper_figure

def plot(cells,summary,out,fixed_only=True,filename=None,title=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # Regenerating the published ten-seed cone must preserve unresolved-seed
    # overlays in known-zero cells, its ten-seed legend and canonical layout.
    if fixed_only and summary.get('ten_seed_completion') and filename is None:
        if any(c.get('aggregation_metric', 'min') != 'min' for c in cells):
            raise ValueError('Use plot_cone_aggregate.py for alternative metrics; this entrypoint reproduces the published minimum cone.')
        assert all(c['mup_base']==0 and c['n_seeds']==10 and c['seeds']==list(range(42,52))
                   and c['state']!='pending' for c in cells)
        from grokking_capacity.experiments.cone_workflow.analyse_cone_ten_seeds import render as render_ten_seed_cone
        render_ten_seed_cone(cells,out)
        return out/'cone_seedmin_fixed.pdf'
    panels=[0] if fixed_only else [0,236]
    fig,axs=plt.subplots(1,len(panels),figsize=(6.5,4.8) if fixed_only else (12,5),squeeze=False,sharey=True,layout='constrained')
    coeffs=law_coefficients();fold=coeffs['capacity_update']['fold_new']
    norm=LogNorm(1,1500);cmap=plt.get_cmap('viridis')
    for ax,mb in zip(axs.flat,panels):
        cs=[c for c in cells if c['mup_base']==mb]
        for state in ['zero','positive','unresolved','pending']:
            rr=[c for c in cs if c['state']==state]
            xx=[c['p'] for c in rr];yy=[c['param_count'] for c in rr]
            if state=='positive':ax.scatter(xx,yy,c=[c['mean_delay'] for c in rr],cmap=cmap,norm=norm,s=17,edgecolors='#202d38',linewidths=.3,zorder=3)
            elif state=='zero':ax.scatter(xx,yy,s=12,facecolors='none',edgecolors='#9299a2',linewidths=.5,alpha=.65,zorder=1)
            elif state=='unresolved':ax.scatter(xx,yy,s=26,marker='x',color='#b35d12',linewidths=1,zorder=5)
            else:ax.scatter(xx,yy,s=24,marker='^',facecolors='none',edgecolors='#377ba8',linewidths=.8,zorder=5)
        if mb==0:
            pp=np.linspace(97,fold-1e-7,500);pred=[law_onset(p,coeffs) for p in pp]
            ax.plot(pp,pred,color='#c7353b',lw=1.7,zorder=4)
            ax.axvline(fold,color='#c7353b',ls='--',lw=1.1,zorder=2)
        ax.set(yscale='log',xlabel='Prime $p$',ylabel='Parameters $P$' if mb==0 else '',xlim=(min(c['p'] for c in cs)-2,max(c['p'] for c in cs)+2))
        ticks=[97,113,127,139,149,163,179,max(c['p'] for c in cs)] if mb==0 else sorted({c['p'] for c in cs})[::2]
        ax.set_xticks(ticks);ax.tick_params(labelsize=9)
        ax.grid(axis='y',which='major',color='#d4d8de',alpha=.55,lw=.6);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    handles=[Line2D([],[],marker='o',ls='',markerfacecolor=cmap(.55),markeredgecolor='#202d38',markersize=5,label='Positive delay in every seed'),Line2D([],[],marker='o',ls='',markerfacecolor='none',markeredgecolor='#9299a2',markersize=5,label='At least one zero-delay seed'),Line2D([],[],marker='x',ls='',color='#b35d12',markersize=5,label='Unresolved'),Line2D([],[],color='#c7353b',lw=1.7,label='Law-derived onset crossing'),Line2D([],[],color='#c7353b',lw=1.1,ls='--',label=rf'$p^*={fold:.1f}$')]
    if any(c['state']=='pending' for c in cells):handles.append(Line2D([],[],marker='^',ls='',markerfacecolor='none',markeredgecolor='#377ba8',markersize=5,label='Awaiting planned seeds'))
    axs.flat[0].legend(handles=handles,loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,fontsize=8,frameon=False,borderaxespad=0,columnspacing=1.4,handlelength=2.3)
    sm=plt.cm.ScalarMappable(norm=norm,cmap=cmap);cb=fig.colorbar(sm,ax=list(axs.flat),fraction=.035,pad=.025);cb.set_label('Mean delay (epochs)',fontsize=10);cb.ax.tick_params(labelsize=9)
    name=filename or ('cone_seedmin_fixed' if fixed_only else 'cone_seedmin')
    save_paper_figure(fig,out/(name+'.pdf'),max_columns=2);plt.close(fig)
    write_json(out/(name+'_plot.json'),{'cells':len(cells),'fixed_only':fixed_only,'capacity':coeffs['C_convention'],'fold':fold,'law_invariant_under_capacity_rescaling':True,'classification_rules':summary['state_rules']})
    return out/(name+'.pdf')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--fixed-only',action='store_true');ap.add_argument('--out',type=Path,default=OUT);ap.add_argument('--reuse',action='store_true');ap.add_argument('--publish',action='store_true');ap.add_argument('--final',action='store_true');args=ap.parse_args()
    latest=ROOT/'results/cone_latest.json'
    latest_out=ROOT/json.loads(latest.read_text())['analysis_directory'] if latest.exists() else OUT/'final'
    if args.final:
        assert args.fixed_only,'Final cohort is for the fixed-rate cone'
        if args.out==OUT:args.out=latest_out
        args.reuse=True
    if args.publish and latest.exists() and args.out.resolve()!=latest_out.resolve():
        raise RuntimeError('A newer audited cone is published. Use --fixed-only --final --publish to preserve it.')
    if args.publish and args.out==OUT and (OUT/'final/summary.json').exists():
        raise RuntimeError('The completed replication cohort exists. Use --fixed-only --final --publish to preserve it.')
    if args.reuse:summary=json.loads((args.out/'summary.json').read_text());cells=json.loads((args.out/'cells.json').read_text())
    else:summary,cells=build(args.out,panels=(0,) if args.fixed_only else (0,236))
    path=plot(cells,summary,args.out,args.fixed_only)
    if args.publish:
        for parent in [ROOT/'figures/theory',ROOT/'docs/papers/2026 Grokking Capacity/media/theory']:
            parent.mkdir(parents=True,exist_ok=True)
            for suffix in ['.pdf','.png']:shutil.copy2(path.with_suffix(suffix),parent/path.with_suffix(suffix).name)
    print(json.dumps({k:summary[k] for k in ['selected_rows','unique_runs','duplicate_seed_observations_removed','cells','cell_state_counts','event_counts','seed_count_distribution','budget_distribution','five_seed_sensitivity']},indent=2),flush=True)
if __name__=='__main__':main()
