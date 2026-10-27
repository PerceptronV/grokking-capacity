#!/usr/bin/env python3
"""Render the completed sweep without overlapping logarithmic minor labels.

The frozen statistical report and original figures remain unchanged. These
presentation figures use the same saved numbers and include their own hashes.
"""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/p97_dataset_scaling'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    summary=json.loads((OUT/'summary.json').read_text())
    assert summary['all_audits_passed']
    rows=summary['sizes'];x=np.array([r['n_train'] for r in rows])
    fig,(left,right)=plt.subplots(1,2,figsize=(13,5.1),constrained_layout=True)
    for role,color in [('C','#2467ae'),('B','#cb6131')]:
        y=np.array([r[role+'_capped_mean'] for r in rows])
        complete=np.array([r[role+'_censored']==0 for r in rows])
        left.plot(x,y,color=color,label=('Continuous C' if role=='C' else 'Categorical B'),linewidth=1.8)
        left.scatter(x[complete],y[complete],color=color,marker='o',s=40,zorder=3)
        left.scatter(x[~complete],y[~complete],edgecolors=color,
            facecolors='none' if role=='C' else color,marker='^',s=110 if role=='C' else 40,zorder=3)
        left.fill_between(x,[r[role+'_capped_mean_ci_low'] for r in rows],
            [r[role+'_capped_mean_ci_high'] for r in rows],color=color,alpha=.12)
    left.set_yscale('log')
    left.set_ylabel('Optimizer updates to first held-out 99%')
    left.set_title('Means capped at 50,000 updates\nTriangles mark lower bounds on uncensored means',fontsize=10)
    left.legend(loc='lower left')
    delta=np.array([r['capped_mean_difference'] for r in rows])
    right.axhline(0,color='black',linewidth=.8)
    right.fill_between(x,[r['simultaneous_low'] for r in rows],[r['simultaneous_high'] for r in rows],
        color='gray',alpha=.15,label='Simultaneous 95% band')
    right.fill_between(x,[r['difference_ci_low'] for r in rows],[r['difference_ci_high'] for r in rows],
        color='gray',alpha=.3,label='Pointwise 95% interval')
    right.plot(x,delta,'o-',color='#62449b',label='Capped mean B minus C')
    censored=np.array([r['mean_difference'] is None for r in rows])
    right.scatter(x[censored],delta[censored],marker='s',facecolors='white',edgecolors='#62449b',zorder=4)
    right.set_ylabel('Difference in optimizer updates')
    right.set_title('Positive means C is faster\nSquares mark sizes with censored runs',fontsize=10)
    right.legend(loc='upper left',fontsize=9)
    for ax in [left,right]:
        ax.set_xscale('log')
        ax.set_xticks(x,labels=[f'{n:,}' for n in x],rotation=45,ha='right',fontsize=9)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel('Training examples (p=97)')
        ax.grid(alpha=.15)
    fig.suptitle('B and C resized to match each dataset via capacity constants; ten seeds, dropout 0.2',fontsize=12)
    for ext in ['png','pdf']:
        fig.savefig(OUT/('dataset_scaling_final.'+ext),dpi=180)
    plt.close(fig)
    paths=[OUT/'dataset_scaling_final.png',OUT/'dataset_scaling_final.pdf']
    provenance=dict(presentation_only=True,summary_sha256=sha(OUT/'summary.json'),
        renderer_sha256=sha(Path(__file__)),output_hashes={p.name:sha(p) for p in paths},
        reason='Suppress duplicate logarithmic minor tick labels; preserve frozen statistics and original report artifacts.')
    (OUT/'rendering_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')

if __name__=='__main__':main()
