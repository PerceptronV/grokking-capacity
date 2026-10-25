#!/usr/bin/env python3
"""Standalone complete-result speed and capacity figures for both families."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/discrete/p97-multilevel-dropout-v1'
OUT=ROOT/'reports/p97_multilevel_dropout'


def read(path):return json.loads(path.read_text())
def save(fig,name):
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUT/f'{name}.{ext}',dpi=180)
    plt.close(fig)


def main():
    summary=read(OUT/'summary.json');assert summary['confirmation_runs']==50
    records=list(csv.DictReader((OUT/'confirmation_runs.csv').open()))
    colors=plt.get_cmap('tab10')
    for family,details in summary['families'].items():
        selected=[r for r in records if r['family'] in ('shared',family)]
        assert len(selected)==30
        fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
        for ax,threshold in zip(axes,(95,99)):
            for i,seed in enumerate(range(201,211)):
                group={r['role']:r for r in selected if int(r['seed'])==seed}
                values=[float(group[role][f'val{threshold}_updates']) if group[role][f'val{threshold}_updates'] else None for role in 'ABC']
                x=np.arange(3)+(i-4.5)*.018;y=[50000 if v is None else v for v in values]
                ax.plot(x,y,color=colors(i),lw=.7,alpha=.5,label=str(seed))
                for xx,yy,v in zip(x,y,values):ax.scatter(xx,yy,color=colors(i),marker='^' if v is None else 'o',s=24)
            ax.set(xticks=[0,1,2],xticklabels=['A continuous','B '+family,'C continuous'],yscale='log',
                   title=f'Sustained held-out {threshold}%',ylim=(300,65000))
            ax.axhline(50000,color='black',ls=':',lw=.7)
        axes[0].set_ylabel('Optimizer updates (log scale)');axes[-1].legend(title='Seed',fontsize=6,ncol=2)
        fig.suptitle(f'p=97 · dropout 0.2 · {family}\nTriangles: no crossing by 50,000 updates; ten shared A seeds')
        save(fig,f'{family}_speed')
        fig,ax=plt.subplots(figsize=(7,4))
        roots={'A':Path(summary['inherited_A']['source_selection']).parent/'jobs','B':RUN/'jobs','C':RUN/'jobs'}
        evidence={'A':details['AB_selection']['evidence']['A'],'B':details['AB_selection']['evidence']['B'],
                  'C':details['confirmation_binding']['C_evidence']}
        for role,color in zip('ABC',('tab:blue','tab:orange','tab:green')):
            for seed,ev in evidence[role].items():
                points=[]
                for name in ev['grid']:
                    cell=read(roots[role]/name/'summary.json');points.append((cell['config']['n_samples'],cell['final_bits']/1000))
                points.sort();ax.plot(*zip(*points),color=color,marker='o',ls='-' if seed=='61' else '--',label=f'{role}, seed {seed}')
        target=summary['target_bits']/1000
        ax.axhspan(.9*target,1.1*target,color='grey',alpha=.15,label='Fixed dataset ±10%')
        ax.axhline(target,color='black',lw=.7)
        ax.set(xscale='log',xlabel='Random-label examples',ylabel='Final learned information (kbit)',
               title=f'p=97 · dropout 0.2 · {family}: measured capacity')
        ax.legend(ncol=2,fontsize=8);save(fig,f'{family}_capacity')
    print('Saved speed and capacity PNG/PDF figures for both families.')


if __name__=='__main__':main()
