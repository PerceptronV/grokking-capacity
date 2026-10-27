#!/usr/bin/env python3
"""Descriptive N=2,952 follow-up; no new training or changed primary analysis."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/p97_dataset_scaling'
OUT=SOURCE/'n2952'
RUN=ROOT/'results/discrete/p97-dataset-scaling-v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def main():
    OUT.mkdir(exist_ok=True)
    primary=next(r for r in read(SOURCE/'summary.json')['sizes'] if r['n_train']==2952)
    known_hashes=read(SOURCE/'input_hashes.json')
    rows=[];inputs={str(SOURCE/'summary.json'):sha(SOURCE/'summary.json')}
    for seed in range(501,511):
        for role,width in [('C',26),('B',30)]:
            path=RUN/'jobs'/f'{role}_d{width}_n2952_s{seed}'
            for filename in ['trace.jsonl','summary.json']:
                p=path/filename;h=sha(p)
                assert h==known_hashes[str(p)],p
                inputs[str(p)]=h
            s=read(path/'summary.json');m=s['metrics']
            trace=[json.loads(line) for line in (path/'trace.jsonl').read_text().splitlines()]
            recomputed={k:next(r['step'] for r in trace if r[k]>=threshold)
                for k,threshold in [('train_online_acc',.99),('train_eval_acc',.99),('val_acc',.98)]}
            t99=next(r['step'] for r in trace if r['val_acc']>=.99)
            active=max(0,recomputed['val_acc']-recomputed['train_online_acc'])
            evaluation=max(0,recomputed['val_acc']-recomputed['train_eval_acc'])
            assert t99==m['val_acc_99_updates'] and active==m['paper_delay_updates'] and evaluation==m['eval_delay_updates']
            assert s['learning_gate_passed'] and s['joint_stop_passed']
            rows.append(dict(seed=seed,role=role,width=width,parameters=s['param_count'],
                train99_online_updates=recomputed['train_online_acc'],train99_eval_updates=recomputed['train_eval_acc'],
                heldout98_updates=recomputed['val_acc'],heldout99_updates=t99,
                sustained99_updates=m['val_99_sustained_updates'],active_dropout_delay_updates=active,
                dropout_free_delay_updates=evaluation,final_updates=s['final']['step']))
    stats={}
    for role in ['C','B']:
        rr=[r for r in rows if r['role']==role]
        stats[role]={}
        for key in ['heldout99_updates','sustained99_updates','train99_online_updates','train99_eval_updates',
                    'active_dropout_delay_updates','dropout_free_delay_updates']:
            values=[r[key] for r in rr]
            stats[role][key]=dict(mean=statistics.mean(values),median=statistics.median(values),
                sample_sd=statistics.stdev(values),minimum=min(values),maximum=max(values),positive=sum(x>0 for x in values))
    differences=[]
    for seed in range(501,511):
        pair={r['role']:r for r in rows if r['seed']==seed}
        differences.append(pair['B']['heldout99_updates']-pair['C']['heldout99_updates'])
    result=dict(n_train=2952,n_test=1863,training_fraction_of_domain=2952/9312,steps_per_epoch=6,
        statistics=stats,paired_differences_B_minus_C=differences,
        C_faster=sum(d>0 for d in differences),B_faster=sum(d<0 for d in differences),ties=differences.count(0),
        registered_mean_difference=primary['mean_difference'],
        registered_pointwise_95_interval=[primary['difference_ci_low'],primary['difference_ci_high']],
        registered_mean_ratio=primary['mean_ratio'],
        registered_ratio_pointwise_95_interval=[primary['ratio_ci_low'],primary['ratio_ci_high']],
        delay_definition='max(0, first held-out98 updates minus first training99 updates); online accuracy covers the preceding ten dropout-active updates, eval accuracy evaluates all training examples without dropout.',
        interpretation='Descriptive follow-up to the completed ten-size sweep. Close observed update counts do not establish statistical equivalence. Capacity predictions at both widths extrapolate below the calibration range.')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/'seeds.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    fig,(a,b)=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
    colors={'C':'#2467ae','B':'#cb6131'}
    for seed in range(501,511):
        pair={r['role']:r for r in rows if r['seed']==seed}
        ys=[pair[role]['heldout99_updates'] for role in ['C','B']]
        a.plot([0,1],ys,color='gray',alpha=.45,linewidth=.9)
        for x,role in enumerate(['C','B']):a.scatter(x,ys[x],color=colors[role],s=30,zorder=3)
    for x,role in enumerate(['C','B']):
        mean=stats[role]['heldout99_updates']['mean']
        a.scatter(x,mean,color=colors[role],marker='D',edgecolors='black',s=100,zorder=5)
        a.annotate(f'Mean {mean:,}',(x,mean),xytext=(-72 if x==0 else 10,6),textcoords='offset points',fontsize=9)
    a.set_xlim(-.55,1.65);a.set_xticks([0,1],['Continuous C\nwidth 26','Categorical B\nwidth 30'])
    a.set_ylabel('Updates to first held-out 99%')
    a.set_title('Ten paired seeds: C wins 7, B wins 2, one tie',fontsize=10)
    a.grid(axis='y',alpha=.15)
    offsets=np.linspace(-.12,.12,10)
    labels=[]
    for x,(role,key) in enumerate([('C','active_dropout_delay_updates'),('B','active_dropout_delay_updates'),
                                   ('C','dropout_free_delay_updates'),('B','dropout_free_delay_updates')]):
        values=[r[key] for r in rows if r['role']==role]
        b.scatter(x+offsets,values,color=colors[role],alpha=.85,s=30)
        b.plot([x-.2,x+.2],[statistics.mean(values)]*2,color=colors[role],linewidth=2.5)
        labels.append(role+' online' if x<2 else role+' eval')
    b.axhline(0,color='gray',linewidth=.7)
    b.set_xticks(range(4),labels);b.set_ylim(-60,1400)
    b.set_ylabel('Delay: training 99% to held-out 98% (updates)')
    b.set_title('Delay depends on how training accuracy is measured\nOnline: dropout active; eval: dropout off; bars: means',fontsize=10)
    b.grid(axis='y',alpha=.15)
    fig.suptitle('2,952 examples: both models learn on all ten seeds; predicted capacity matched',fontsize=12)
    for ext in ['png','pdf']:fig.savefig(OUT/('seed_comparison.'+ext),dpi=180)
    plt.close(fig)
    (OUT/'provenance.json').write_text(json.dumps(dict(analysis_source_sha256=sha(Path(__file__)),
        input_hashes=inputs,output_hashes={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='provenance.json'}),indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
