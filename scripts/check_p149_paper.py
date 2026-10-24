#!/usr/bin/env python3
"""Read-only extraction of the paper's p149, weight-decay-1 baseline.

Verify all registry generalisation crossings against the original saved traces.
Export first-passage epochs, equivalent optimiser updates, and paper-defined
train-99 / validation-98 delays. No new training or registry mutation.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


FILTERS = dict(p=149, operation='/', train_fraction=.5, split_type='random',
               depth=2, heads=1, dropout=.2, init_scale=1., lr=.001,
               weight_decay=1., beta1=.9, beta2=.98, batch_size=512,
               mup_base=0, architecture_family='transformer_gated',
               host='147-224-250-73', status='completed', experiment_type='groks',
               dataset_type='modular', max_epochs=5000)


def first(values, threshold):
    indices=np.flatnonzero(values>=threshold)
    return int(indices[0])+1 if len(indices) else None


def csv_write(path, rows):
    with path.open('w') as file:
        writer=csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract(database, output):
    connection=sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True)
    connection.row_factory=sqlite3.Row
    query='SELECT * FROM runs WHERE '+' AND '.join(f'{key}=?' for key in FILTERS)+' ORDER BY dim,seed'
    records=[]
    for row in connection.execute(query,tuple(FILTERS.values())):
        path=Path(row['npz_path'])
        with np.load(path,allow_pickle=False) as saved:
            assert int(saved['n_train'])==row['n_samples']==11026
            assert int(saved['p'])==149 and int(saved['dim'])==row['dim']
            assert int(saved['param_count'])==row['param_count']
            train99, val98, val99=(first(saved[key],threshold) for key,threshold in
                                  [('train_acc',99),('val_acc',98),('val_acc',99)])
            assert val99==row['grokking_epoch'] and row['grokking_threshold']==.99
            steps_per_epoch=(row['n_samples']+row['batch_size']-1)//row['batch_size']
            records.append(dict(uuid=row['uuid'],dim=row['dim'],params=row['param_count'],
                seed=row['seed'],epochs_observed=len(saved['val_acc']),
                train99_epoch=train99,val98_epoch=val98,val99_epoch=val99,
                val99_updates=val99*steps_per_epoch if val99 is not None else None,
                delay_epochs=max(0,val98-train99) if train99 is not None and val98 is not None else None,
                steps_per_epoch=steps_per_epoch,trace_path=str(path),
                trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),git_hash=row['git_hash']))
    connection.close()
    assert len(records)==80
    by_width=defaultdict(list)
    for row in records: by_width[row['dim']].append(row)
    aggregates=[]
    for dim,rows in sorted(by_width.items()):
        assert {r['seed'] for r in rows}==set(range(42,47))
        observed=[r for r in rows if r['val99_epoch'] is not None]
        delays=[r['delay_epochs'] for r in rows if r['delay_epochs'] is not None]
        aggregates.append(dict(dim=dim,params=rows[0]['params'],total=5,reached99=len(observed),
            mean_observed_val99_epoch=mean(r['val99_epoch'] for r in observed) if observed else None,
            mean_observed_val99_updates=mean(r['val99_updates'] for r in observed) if observed else None,
            min_observed_val99_epoch=min((r['val99_epoch'] for r in observed),default=None),
            max_observed_val99_epoch=max((r['val99_epoch'] for r in observed),default=None),
            delay_observations=len(delays),min_observed_delay_epochs=min(delays,default=None),
            mean_observed_delay_epochs=mean(delays) if delays else None))
    output.mkdir(parents=True,exist_ok=True)
    csv_write(output/'runs.csv',records)
    csv_write(output/'baseline.csv',aggregates)
    provenance=dict(filters=FILTERS,paper='https://arxiv.org/pdf/2605.09724v2',
        paper_section='Appendix C.2 weight-decay sweep; lambda=1 baseline, p=149',
        configuration='configs/weight_decay_sweep.yaml',verified_traces=len(records),
        units='1 epoch = 22 optimiser updates; first epoch >=99% validation, no three-evaluation requirement',
        delay='max(0, first validation>=98% epoch - first training>=99% epoch)',
        missing='Missing crossings stay null; means are observed-only, with reached/total shown')
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    complete=[r for r in aggregates if r['reached99']==5]
    partial=[r for r in aggregates if 0<r['reached99']<5]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    x=[r['params'] for r in complete]; y=[r['mean_observed_val99_epoch'] for r in complete]
    axes[0].plot(x,y,'o-',color='#3366a0',label='Mean; all 5 seeds reached 99%')
    axes[0].fill_between(x,[r['min_observed_val99_epoch'] for r in complete],
                        [r['max_observed_val99_epoch'] for r in complete],alpha=.14,color='#3366a0',label='Seed range')
    for row in partial:
        axes[0].scatter(row['params'],row['mean_observed_val99_epoch'],facecolors='none',edgecolors='#3366a0')
        axes[0].annotate(f"{row['reached99']}/5 reached",(row['params'],row['mean_observed_val99_epoch']),
                         xytext=(0,8),textcoords='offset points',fontsize=8,ha='center')
    axes[0].set_yscale('log')
    axes[0].set_ylim(30,6000)
    axes[0].text(.02,.98,'Widths 20 and 24: 0/5 reached 99%',transform=axes[0].transAxes,va='top',fontsize=8)
    axes[0].set_ylabel('Epochs to first 99% validation')
    axes[0].set_title('Larger ordinary models generalise sooner')
    axes[0].legend(fontsize=8,frameon=False,loc='upper right')
    axes[1].plot(x,[r['min_observed_delay_epochs'] for r in complete],'o-',color='#d36a25')
    axes[1].set_ylabel('Minimum train-99 / validation-98 delay (epochs)')
    axes[1].set_title('Positive delay appears at width 96')
    axes[1].axhline(0,color='grey',ls=':',lw=.8)
    for ax in axes:
        ax.set_xscale('log'); ax.set_xlabel('Parameters'); ax.grid(alpha=.2)
    fig.suptitle('Original p=149 baseline: dropout 0.2, weight decay 1, seeds 42–46')
    fig.tight_layout()
    for extension in ('png','pdf'): fig.savefig(output/f'width_trend.{extension}',dpi=170)
    plt.close(fig)
    print(json.dumps(aggregates,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,default=Path('runs.db'))
    parser.add_argument('--output',type=Path,default=Path('reports/paper_p149'))
    args=parser.parse_args()
    extract(args.database,args.output)
