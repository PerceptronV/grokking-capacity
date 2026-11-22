#!/usr/bin/env python3
"""Show complete observed-ten-seed cells used by the isolated law sensitivity fit."""
from pathlib import Path
from collections import Counter,defaultdict
import csv
import hashlib
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from grokking_capacity.analysis._figure_style import save_paper_figure

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'results/law_complete_ten_seed_sensitivity_20261106'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    analysis=json.loads((OUT/'comparison.json').read_text())
    protected=json.loads((OUT/'validation.json').read_text())['protected_artifacts_unchanged']
    for rel,digest in protected.items():assert sha(ROOT/rel)==digest
    with (OUT/'selected_runs.csv').open() as f:runs=list(csv.DictReader(f))
    by=defaultdict(list)
    for r in runs:by[r['kind'],int(r['p']),int(r['dim'])].append(r)
    cells=[]
    for (kind,p,d),rr in sorted(by.items()):
        assert len(rr)==10 and {int(r['seed']) for r in rr}==set(range(42,52))
        assert {r['gpu_type'] for r in rr}=={'A100'}
        assert len({r['P'] for r in rr})==1
        n=sum(r['observed']=='True' for r in rr)
        retained=n==10
        assert all((r['included_complete_ten_fit']=='True')==retained for r in rr)
        cells.append({'law':kind,'p':p,'width':d,'parameters':int(rr[0]['P']),
                      'runs':10,'observed_crossings':n,'complete_ten':retained,
                      'observed_seed_ids':';'.join(str(r['seed']) for r in rr if r['observed']=='True'),
                      'hardware':'A100'})
    colours=['#ffffff','#e9edef','#efc36b','#167d83']
    cmap=ListedColormap(colours);norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4)
    fig,axs=plt.subplots(2,1,figsize=(13.6,7.3),gridspec_kw={'hspace':.37})
    summaries={}
    for ax,kind,label in zip(axs,['mem','gen'],['Memorisation','Generalisation']):
        rr=[r for r in cells if r['law']==kind]
        primes=sorted({r['p'] for r in rr});widths=sorted({r['width'] for r in rr})
        grid=np.zeros((len(primes),len(widths)),dtype=int)
        lookup={(r['p'],r['width']):r for r in rr}
        rows=[]
        for i,p in enumerate(primes):
            csvrow={'prime':p}
            for k,d in enumerate(widths):
                cell=lookup.get((p,d));csvrow[str(d)]='' if cell is None else cell['observed_crossings']
                if cell is None:continue
                n=cell['observed_crossings'];grid[i,k]=3 if n==10 else 2 if n else 1
                ax.text(k+.5,i+.5,str(n),ha='center',va='center',fontsize=8.4 if kind=='mem' else 7.5,
                        color='white' if n==10 else '#303b43',fontweight='medium' if n else 'normal')
            rows.append(csvrow)
        ax.pcolormesh(np.arange(len(widths)+1),np.arange(len(primes)+1),grid,cmap=cmap,norm=norm,
                      edgecolors='#d7dfe2',linewidth=.45,rasterized=False,zorder=0)
        ax.set_xticks(np.arange(len(widths))+.5,widths,fontsize=8.5 if kind=='mem' else 7.8)
        ax.set_yticks(np.arange(len(primes))+.5,primes,fontsize=9)
        ax.set_xlim(0,len(widths));ax.set_ylim(len(primes),0)
        ax.tick_params(length=0,pad=5)
        ax.set_xlabel('Model width $d$ (step '+('4' if kind=='mem' else '2')+')',fontsize=10)
        ax.set_ylabel(label+' — prime $p$',fontsize=11,labelpad=9)
        for spine in ax.spines.values():spine.set_visible(False)
        with (OUT/f'{kind}_observed_seed_grid.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        complete=[r for r in rr if r['complete_ten']]
        with (OUT/f'{kind}_complete_cells.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(complete[0]));writer.writeheader();writer.writerows(complete)
        summaries[kind]={'candidate_cells':len(rr),'complete_cells':len(complete),
                         'partial_cells':sum(0<r['observed_crossings']<10 for r in rr),
                         'zero_observed_crossing_cells':sum(r['observed_crossings']==0 for r in rr)}
        assert len(complete)*(10 if kind=='mem' else 1)==analysis['counts'][kind]['complete_fit_observations']
    fig.tight_layout()
    handles=[Patch(facecolor=colours[3],edgecolor='#d7dfe2',label='10/10 crossings: retained in complete-cell fit'),
             Patch(facecolor=colours[2],edgecolor='#d7dfe2',label='1–9/10 crossings: excluded from complete-cell fit'),
             Patch(facecolor=colours[1],edgecolor='#d7dfe2',label='0/10 crossings: threshold not observed in any seed'),
             Patch(facecolor=colours[0],edgecolor='#d7dfe2',label='Blank: not a candidate in this law’s fitted grid')]
    fig.legend(handles=handles,fontsize=9,frameon=False)
    note=fig.text(.5,0,'Every candidate cell has ten runs; numbers count observed threshold crossings. A100 only.\n'
                  'Memorisation: training ≥99%; generalisation: validation ≥99%. Original fitting window retained.\n'
                  'Complete cells: 67 of 164 for memorisation; 269 of 327 for generalisation.',fontsize=9,ha='center')
    save_paper_figure(fig,OUT/'complete_ten_seed_grid.pdf',notes=[note],max_columns=2,dpi=220)
    plt.close(fig)
    with (OUT/'cell_completeness.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(cells[0]));writer.writeheader();writer.writerows(cells)
    for rel,digest in protected.items():assert sha(ROOT/rel)==digest
    record={'input_runs_sha256':sha(OUT/'selected_runs.csv'),'cells':len(cells),'counts':summaries,
            'exactly_ten_distinct_runs_per_candidate_cell':True,'all_fit_candidates_A100':True,
            'complete_flags_match_sensitivity_fit':True,'published_cone_and_paper_unchanged':True,
            'figure_pdf_sha256':sha(OUT/'complete_ten_seed_grid.pdf'),
            'figure_png_sha256':sha(OUT/'complete_ten_seed_grid.png'),'plot_source_sha256':sha(__file__)}
    (OUT/'grid_validation.json').write_text(json.dumps(record,indent=2)+'\n')
    shutil.copy2(__file__,OUT/Path(__file__).name)
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
