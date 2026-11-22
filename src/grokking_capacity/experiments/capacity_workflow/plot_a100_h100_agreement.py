#!/usr/bin/env python3
"""Display all paired seed classifications from the preselected hardware check."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch,Rectangle
from grokking_capacity.analysis._figure_style import save_paper_figure

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'results/a100_h100_grokking_agreement_20261106'

def main():
    report=json.loads((OUT/'analysis.json').read_text())
    with (OUT/'paired_runs.csv').open() as f:rows=list(csv.DictReader(f))
    cells=sorted({(int(r['p']),int(r['dim'])) for r in rows});seeds=list(range(42,52))
    lookup={(int(r['p']),int(r['dim']),int(r['seed'])):r for r in rows}
    assert len(rows)==200 and len(cells)==20
    colors=['#e9edef','#167d83','#efc36b'];code={'zero':0,'positive':1,'unresolved':2};symbol={'zero':'0','positive':'+','unresolved':'?'}
    fig,axes=plt.subplots(1,2,figsize=(9.5,7.2),sharey=True,layout='constrained')
    for ax,hardware in zip(axes,['A100','H100']):
        grid=np.array([[code[lookup[p,d,s][hardware+'_state']] for s in seeds] for p,d in cells])
        ax.pcolormesh(np.arange(11),np.arange(21),grid,cmap=ListedColormap(colors),norm=BoundaryNorm([-.5,.5,1.5,2.5],3),edgecolors='#d7dfe2',linewidth=.5)
        for i,(p,d) in enumerate(cells):
            for k,seed in enumerate(seeds):
                r=lookup[p,d,seed];state=r[hardware+'_state']
                ax.text(k+.5,i+.5,symbol[state],ha='center',va='center',fontsize=9,color='white' if state=='positive' else '#303b43')
                if hardware=='H100' and r['state_agrees']=='False':ax.add_patch(Rectangle((k,i),1,1,fill=False,edgecolor='#c7353b',lw=2))
        ax.set_xticks(np.arange(10)+.5,seeds,fontsize=9);ax.set_yticks(np.arange(20)+.5,[f'p={p}, d={d}' for p,d in cells],fontsize=8)
        ax.set_xlim(0,10);ax.set_ylim(20,0);ax.set_xlabel(hardware+' — seed',fontsize=10);ax.tick_params(length=0,pad=5)
        for spine in ax.spines.values():spine.set_visible(False)
    axes[0].set_ylabel('Matched prime and model width',fontsize=10)
    fig.legend(handles=[Patch(facecolor=colors[1],label='+  Observed positive delay'),Patch(facecolor=colors[0],label='0  Observed zero delay'),
                        Patch(facecolor=colors[2],label='?  Unresolved validation crossing'),Patch(facecolor='none',edgecolor='#c7353b',label='Seed classification changed')],fontsize=9,frameon=False)
    note=fig.text(.5,0,f'Seed agreement: {report["seed_state_agreement"]}/200. Cell classification agreement: {report["cell_state_agreement"]}/20.\n'
                  'Delay uses first online training ≥99% and evaluation validation ≥98%; unobserved validation stays unresolved.\n'
                  'Same seed IDs, model and optimiser settings; archived A100 versus current H100 runtime.',ha='center',fontsize=8)
    save_paper_figure(fig,OUT/'paired_classifications.pdf',notes=[note],max_columns=2,dpi=220);plt.close(fig)
    print('Saved paired_classifications.pdf and paired_classifications.png')

if __name__=='__main__':main()
