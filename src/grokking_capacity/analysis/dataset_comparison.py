#!/usr/bin/env python3
"""Paired mean curves, censoring bounds, and bootstrap crossing diagnostics."""
import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np
import yaml

from grokking_capacity.experiments.registered import (
    atomic_json,
    read,
    sha,
    verify_sources as verify_registered_sources,
)
from grokking_capacity.utils.artifacts import write_csv as csv_write
from .paired import summarize, crossing_brackets, analyze


def plot(summary, rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table=summary['sizes'];sizes=np.array([r['n_train'] for r in table])
    colors={'C':'#2467ae','B':'#cb6131'}
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for role in ['C','B']:
        y=np.array([r[role+'_capped_mean'] for r in table])
        complete=np.array([r[role+'_censored']==0 for r in table])
        axes[0,0].plot(sizes,y,color=colors[role],alpha=.6,label=role+' capped mean')
        axes[0,0].scatter(sizes[complete],y[complete],color=colors[role],marker='o')
        axes[0,0].scatter(sizes[~complete],y[~complete],color=colors[role],marker='^',s=55)
        low=[r[role+'_capped_mean_ci_low'] for r in table];high=[r[role+'_capped_mean_ci_high'] for r in table]
        axes[0,0].fill_between(sizes,low,high,color=colors[role],alpha=.12)
        axes[1,0].plot(sizes,[r[role+'_observed'] for r in table],'-o',color=colors[role],label=role+' first99')
        axes[1,0].plot(sizes,[r[role+'_final_learning'] for r in table],'--',color=colors[role],label=role+' final learning')
        for seed in summary['seeds']:
            rr=sorted([r for r in rows if r['role']==role and r['seed']==seed],key=lambda r:r['n_train'])
            ys=[r['t_gen_updates'] if r['t_gen_updates'] is not None else summary['limit'] for r in rr]
            axes[1,1].plot(sizes,ys,color=colors[role],alpha=.2,linewidth=.8)
        axes[1,1].plot(sizes,y,color=colors[role],linewidth=2,label=role+' capped mean')
    ax=axes[0,1]
    delta=np.array([r['capped_mean_difference'] for r in table])
    ax.axhline(0,color='black',linewidth=1)
    ax.fill_between(sizes,[r['simultaneous_low'] for r in table],[r['simultaneous_high'] for r in table],color='gray',alpha=.15,label='Simultaneous 95%')
    ax.fill_between(sizes,[r['difference_ci_low'] for r in table],[r['difference_ci_high'] for r in table],color='gray',alpha=.3,label='Pointwise 95%')
    ax.plot(sizes,delta,'o-',color='#62449b',label='Mean B − C (capped)')
    for row in table:
        if row['mean_difference'] is None:
            ax.scatter(row['n_train'],row['capped_mean_difference'],marker='s',facecolors='white',edgecolors='#62449b',zorder=4)
    titles=['Mean time; triangles = lower bounds, shading = capped-time 95% CI',
        'Positive means C is faster; squares mark censoring',
        f'Successes out of {len(summary["seeds"])} seeds', f'Every seed; failures plotted at {summary["limit"]:,}-update ceiling']
    for ax,title in zip(axes.flat,titles):
        ax.set_xscale('log');ax.set_xlabel(f'Training examples (p={summary["prime"]})');ax.set_title(title,fontsize=10)
        ax.set_xticks(sizes,labels=[str(n) for n in sizes],rotation=45,fontsize=8)
        ax.grid(alpha=.15);ax.legend(fontsize=8)
    axes[0,0].set_yscale('log');axes[1,1].set_yscale('log')
    axes[0,0].set_ylabel('Optimizer updates to first held-out 99%')
    axes[1,1].set_ylabel('Optimizer updates')
    axes[0,1].set_ylabel('B minus C, optimizer updates')
    axes[1,0].set_ylim(-.3,len(summary['seeds'])+.5)
    fig.suptitle(f'Dataset-matched predicted capacity: continuous C vs categorical B; dropout {summary["dropout"]:g}')
    for ext in ['png','pdf']:
        fig.savefig(out/('dataset_scaling.'+ext),dpi=180)
    plt.close(fig)


def generate(run, out, *, calibration_selection=None):
    RUN, OUT = Path(run).resolve(), Path(out).resolve()
    assert read(RUN/'campaign_complete.json')['all_passed']
    verify_registered_sources(read(RUN/'manifest.json')['source_hashes'])
    assert not (RUN/'campaign_error.json').exists()
    reg=yaml.safe_load((RUN/'registration.yaml').read_text())
    manifest=read(RUN/'manifest.json')
    assert manifest['registry_sha256']==sha(RUN/'job_registry.json')
    assert manifest['selection_sha256']==sha(RUN/'selection.json')
    assert manifest['registration_sha256']==sha(RUN/'registration.yaml')
    selection=read(RUN/'selection.json')
    calibration_selection = Path(calibration_selection or selection['source'])
    assert sha(calibration_selection)==selection['source_sha256'], 'Calibration evidence differs from the frozen selection'
    for path,digest in selection['calibration_evidence'].items():
        assert sha(path)==digest
    rows=[];hashes={};audits={}
    registry=read(RUN/'job_registry.json')
    for name,c in registry.items():
        path=RUN/'jobs'/name;s=read(path/'summary.json');m=read(path/'manifest.json')
        audit=read(RUN/'audits'/(name+'.json'))
        assert audit['all_passed'] and c==s['config']==m['config']
        for f,h in audit['input_hashes'].items():
            assert sha(path/f)==h
            hashes[str(path/f)]=h
        audits[name]=audit
        # Independent first crossing from raw trace, without calling worker metrics.
        trace=[json.loads(line) for line in (path/'trace.jsonl').read_text().splitlines()]
        first=next((r for r in trace if r['val_acc']>=.99),None)
        assert (first['step'] if first else None)==s['metrics']['val_acc_99_updates']
        rows.append(dict(name=name,role=c['role'],dim=c['dim'],seed=c['seed'],n_train=c['n_train'],
            n_test=m['n_test'],parameters=s['param_count'],estimated_capacity_bits=selection['pairs'][c['pair_index']][c['role']+'_estimated_bits'],
            capacity_dataset_mismatch=selection['pairs'][c['pair_index']][c['role']+'_dataset_mismatch'],
            calibration_extrapolation=selection['pairs'][c['pair_index']][c['role']+'_calibration_extrapolation'],
            t_gen_updates=first['step'] if first else None,t_gen_examples=first['examples'] if first else None,
            t_gen_epochs=first['epoch_fraction'] if first else None,censored=first is None,
            sustained_updates=s['metrics']['val_99_sustained_updates'],final_learning=s['learning_gate_passed'],
            joint_stop=s['joint_stop_passed'],final_train_acc=s['final']['train_eval_acc'],final_test_acc=s['final']['val_acc'],
            paper_delay_updates=s['metrics']['paper_delay_updates'],eval_delay_updates=s['metrics']['eval_delay_updates'],
            final_updates=s['final']['step'],wall_seconds=s['final']['wall_seconds'],test_sha256=m['test_sha256'],
            train_sha256=m['train_sha256'],initial_sha256=m['initial_weights_sha256']))
    assert len(rows)==2*len(reg['dataset']['sizes'])*len(reg['confirmation']['seeds'])
    for seed in reg['confirmation']['seeds']:
        rr=[r for r in rows if r['seed']==seed]
        assert len({r['test_sha256'] for r in rr})==1
        for n in reg['dataset']['sizes']:
            assert len({r['train_sha256'] for r in rr if r['n_train']==n})==1
    summary=analyze(rows,reg['dataset']['sizes'],reg['confirmation']['seeds'],reg['common']['max_updates'],label_count=reg['common']['p']+2)
    summary.update(seeds=reg['confirmation']['seeds'],limit=reg['common']['max_updates'],pairs=selection['pairs'],
        all_audits_passed=True,run_count=len(rows),hashes_verified=len(hashes),prime=reg['common']['p'],dropout=reg['common']['dropout'],run_root=str(RUN.resolve()))
    for row,pair in zip(summary['sizes'],selection['pairs']):
        assert row['n_train']==pair['n_train']
        for role in ['C','B']:
            row[role+'_width']=pair[role]['dim']
            for key in ['parameters','estimated_bits','dataset_mismatch','calibration_extrapolation']:
                row[role+'_'+key]=pair[role+'_'+key]
        row['predicted_relative_mismatch']=pair['predicted_relative_mismatch']
    OUT.mkdir(parents=True,exist_ok=True)
    atomic_json(OUT/'summary.json',summary);atomic_json(OUT/'input_hashes.json',hashes)
    atomic_json(OUT/'checkpoint_audits.json',audits)
    csv_write(OUT/'runs.csv',rows);csv_write(OUT/'size_summary.csv',summary['sizes'])
    for name in ['manifest.json','selection.json','registration.yaml','campaign_complete.json','job_registry.json']:
        shutil.copyfile(RUN/name,OUT/name)
    plot(summary,rows,OUT)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['sizes','pairs']},indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--calibration-selection', type=Path,
                        help='Relocated calibration file; its hash must match the frozen selection')
    args = parser.parse_args(argv)
    generate(args.run, args.out, calibration_selection=args.calibration_selection)


if __name__ == '__main__':
    main()
