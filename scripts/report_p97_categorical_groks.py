#!/usr/bin/env python3
"""Complete audited C-first categorical comparison and equal-complexity timing."""
import csv
import json
from pathlib import Path
import statistics
import sys
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from grokking_capacity.experiments.categorical_groks import RUN,read,sha,source_hashes,independent_metrics
from grokking_capacity.experiments.multilevel_dropout import atomic_json
OUT=ROOT/'reports/p97_categorical_groks'


def csv_write(path,records):
    if not records:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in records for k in r)),lineterminator='\n')
        w.writeheader();w.writerows(records)


def timing(rows,key,ceiling=50000):
    values=[r[key] for r in rows if r[key] is not None]
    complete=len(values)==len(rows)
    return dict(observed=len(values),censored=len(rows)-len(values),
        mean_all=statistics.mean(values) if complete else None,
        mean_lower_bound=statistics.mean(values+[ceiling]*(len(rows)-len(values))),
        median_all=statistics.median(values) if complete else None,
        min_observed=min(values) if values else None,max_observed=max(values) if values else None)


def delays(rows,key):
    values=[r[key] for r in rows if r[key] is not None]
    return dict(min_observed=min(values) if values else None,max_observed=max(values) if values else None,
        mean_all=statistics.mean(values) if len(values)==len(rows) else None,
        median_all=statistics.median(values) if len(values)==len(rows) else None,positive=sum(v>0 for v in values),
        zero=values.count(0),undefined=len(rows)-len(values),
        positive_minimum_across_ten=bool(len(rows)==10 and len(values)==10 and min(values)>0))


def main():
    done=read(RUN/'campaign_complete.json');selection=read(RUN/'selection.json');manifest=read(RUN/'manifest.json')
    assert manifest['source_hashes']==source_hashes()
    reg=yaml.safe_load((RUN/'registration.yaml').read_text())
    assert manifest['registration_sha256']==selection['registration_sha256']==sha(RUN/'registration.yaml')
    for p,h in selection['evidence_sha256'].items():assert sha(p)==h
    stage_jobs={};stages={}
    for p in sorted((RUN/'stages').glob('*/controller.json')):
        s=read(p);assert s['phase']=='complete' and not s['running'] and not s['pending'] and not s['errors']
        assert s['jobs']==read(p.parent/'jobs.json')
        assert all(j['exit_code']==0 for j in s['completed'])
        assert {j['name'] for j in s['completed']}=={j['name'] for j in s['jobs']}
        for j in s['jobs']:assert j['name'] not in stage_jobs;stage_jobs[j['name']]=j['config']
        stages[p.parent.name]=dict(jobs=len(s['jobs']),controller_sha256=sha(p),
            started_at=s['started_at'],finished_at=s['finished_at'])
    ordered=sorted(stages,key=lambda name:stages[name]['started_at'])
    assert all(stages[a]['finished_at']<stages[b]['started_at'] for a,b in zip(ordered,ordered[1:]))
    assert ordered[-3:]==[f"C_d{selection['C']['dim']}_validation",'B_validation','matched_memorisation']
    assert stages[ordered[-3]]['finished_at']<selection['frozen_at']<stages['B_validation']['started_at']
    arithmetic=[];memo=[];audits=[];inputs={};jobs={}
    for p in sorted((RUN/'jobs').glob('*/summary.json')):
        path=p.parent;s=read(p);c=s['config'];m=read(path/'manifest.json');a=read(RUN/'audits'/(path.name+'.json'))
        assert a['all_passed'] and a['checkpoint_sha256']==s['checkpoint_sha256']
        assert c==m['config']==stage_jobs[path.name] and m['source_hashes']==source_hashes()
        assert all(c[k]==v for k,v in reg['common'].items())
        for filename,h in a['input_hashes'].items():assert sha(path/filename)==h;inputs[str(path/filename)]=h
        trace=[json.loads(line) for line in (path/'trace.jsonl').read_text().splitlines()]
        assert len(trace)==s['epochs'] and trace[-1]==s['final']
        assert all(r['epoch']==i+1 and r['step']==10*(i+1) for i,r in enumerate(trace))
        row=dict(name=path.name,role=c['role'],phase=c['phase'],seed=c['seed'],model=c['model'],dim=c['dim'],
            mlp_dim=c['mlp_dim'],parameters=s['param_count'],epochs=s['epochs'],updates=s['final']['step'])
        if c['task']=='arithmetic':
            metrics=independent_metrics(trace);assert metrics==s['metrics']
            row.update(learning_success=s['learning_gate_passed'],final_train_online=trace[-1]['train_online_acc'],
                final_train_eval=trace[-1]['train_eval_acc'],final_val=trace[-1]['val_acc'],**metrics)
            for key in ['paper_delay','eval_delay']:
                epoch=metrics[key+'_epochs'];row[key+'_updates']=10*epoch if epoch is not None else None
            for threshold in [95,99]:
                epoch=metrics[f'val_acc_{threshold}_epoch'];row[f'first_val{threshold}_updates']=10*epoch if epoch is not None else None
                epoch=metrics[f'val_{threshold}_sustained_epoch'];row[f'sustained_val{threshold}_updates']=10*epoch if epoch is not None else None
            arithmetic.append(row)
        else:
            hits=[r for r in trace if r['train_online_acc']>=.99];hit=hits[0]['epoch'] if hits else None
            assert hit==s['memorisation_epoch'] and (hit==len(trace) if hit else len(trace)==c['epochs'])
            gate=read(Path(c['gate_path'])/'summary.json');assert gate['learning_gate_passed'] and gate['finished_at']<m['started_at']
            row.update(first_train99_updates=s['memorisation_updates'],censor_updates=s['censor_updates'],
                final_train_online=trace[-1]['train_online_acc'],final_train_eval=s['final_eval']['acc'])
            memo.append(row)
        if c['role']=='B':
            assert m['started_at']>selection['frozen_at'] and c['selection_sha256']==sha(RUN/'selection.json')
            assert all(c[k]==v for k,v in selection['B'].items())
        jobs[path.name]=(s,m,trace);audits.append(a)
    assert len(jobs)==done['new_jobs']==len(stage_jobs)
    confirm=[r for r in arithmetic if r['phase']=='validation' and r['dim']==selection['C']['dim']]
    groups={};pairings=[]
    for role in ['C','B']:
        rows=[r for r in confirm if r['role']==role];random=[r for r in memo if r['role']==role]
        assert sorted(r['seed'] for r in rows)==selection['validation']['seeds']
        for row in rows+random:
            config=jobs[row['name']][0]['config'];assert all(config[k]==v for k,v in selection[role].items())
        groups[role]=dict(parameters=selection[role+'_parameters'],model=selection[role],
            learning_successes=sum(r['learning_success'] for r in rows),
            paper_delay_updates=delays(rows,'paper_delay_updates'),eval_delay_updates=delays(rows,'eval_delay_updates'),
            first_val99_updates=timing(rows,'first_val99_updates'),sustained_val99_updates=timing(rows,'sustained_val99_updates'),
            memorisation_updates=timing(random,'first_train99_updates') if random else None,
            random_final_accuracy={key:dict(min=min(r[key] for r in random),max=max(r[key] for r in random),
                mean=statistics.mean(r[key] for r in random),reached99=sum(r[key]>=.99 for r in random))
                for key in ['final_train_online','final_train_eval']} if random else None)
        groups[role]['paper_criterion_passed']=bool(groups[role]['learning_successes']==10 and groups[role]['paper_delay_updates']['positive_minimum_across_ten'])
        if random:assert sorted(r['seed'] for r in random)==selection['validation']['seeds']
        for row in random:
            ar=next(v for v in rows if v['seed']==row['seed'])
            assert jobs[ar['name']][1]['initial_weights_sha256']==jobs[row['name']][1]['initial_weights_sha256']
    for seed in selection['validation']['seeds']:
        rr={role:next(r for r in confirm if r['role']==role and r['seed']==seed) for role in ['C','B']}
        manifests={role:jobs[r['name']][1] for role,r in rr.items()}
        assert manifests['C']['dataset_sha256']==manifests['B']['dataset_sha256']
        assert manifests['C']['param_count']==manifests['B']['param_count']
        random={r['role']:r for r in memo if r['seed']==seed}
        if len(random)==2:assert len({jobs[r['name']][1]['dataset_sha256'] for r in random.values()})==1
        record=dict(seed=seed,C_delay_updates=rr['C']['paper_delay_updates'],B_delay_updates=rr['B']['paper_delay_updates'],
            C_gen99_updates=rr['C']['first_val99_updates'],B_gen99_updates=rr['B']['first_val99_updates'])
        for role in ['C','B']:record[role+'_mem99_updates']=random[role]['first_train99_updates'] if role in random else None
        pairings.append(record)
    assert groups['C']['paper_criterion_passed']
    selected_search=[r for r in arithmetic if r['role']=='C' and r['phase']=='search' and r['dim']==selection['C']['dim']]
    assert sorted(r['seed'] for r in selected_search)==reg['search']['seeds']
    search_metrics=dict(learning_successes=sum(r['learning_success'] for r in selected_search),
        paper_delay_updates=delays(selected_search,'paper_delay_updates'))
    assert search_metrics['learning_successes']==10 and search_metrics['paper_delay_updates']['positive_minimum_across_ten']
    summary=dict(complete=True,selected_width=selection['C']['dim'],new_jobs=len(jobs),
        arithmetic_runs=len(arithmetic),memorisation_runs=len(memo),groups=groups,paired_outcomes=pairings,
        C_selection=selection,C_search_metrics=search_metrics,search_results=read(RUN/'search_results.json'),notes=[
            'C was selected to have positive delay on all ten search seeds and all ten fresh validation seeds before B training.',
            'Both models use p97, 4656 training examples, dropout0.2 and ten optimizer updates per epoch.',
            'Categorical B has the same parameter count and residual width as ordinary continuous C; feedforward allocation differs.',
            'This is C-first selection, not a new claim of B capacity matching to the dataset.',
            'Exact-size random memorisation uses first online99, without capacity-patience stopping.',
            'Previous categorical-versus-scalar performance is confounded by parameter count, widths, allocation, and estimator; this pair does not isolate those causes.',
            'C selection is conditional; it does not estimate grokking frequency for unselected models.'])
    OUT.mkdir(parents=True,exist_ok=True)
    csv_write(OUT/'arithmetic_runs.csv',arithmetic);csv_write(OUT/'confirmation_runs.csv',confirm)
    csv_write(OUT/'memorisation_runs.csv',memo);csv_write(OUT/'paired_outcomes.csv',pairings)
    atomic_json(OUT/'summary.json',summary);atomic_json(OUT/'checkpoint_audits.json',audits);atomic_json(OUT/'input_hashes.json',inputs)
    atomic_json(OUT/'validation.json',dict(all_passed=True,complete=True,checkpoints=len(audits),
        arithmetic_metrics_recomputed=len(arithmetic),memorisation_crossings_recomputed=len(memo),
        paired_seeds=len(pairings),stages=stages,stage_order=ordered,ordering_verified=True,
        analysis_sha256=sha(Path(__file__))))
    for name in ['registration.yaml','selection.json','manifest.json','search_results.json','B_results.json','campaign_complete.json']:
        (OUT/name).write_bytes((RUN/name).read_bytes())
    plots(confirm,pairings,jobs)
    print(json.dumps(dict(new_jobs=len(jobs),groups=groups),indent=2))


def plots(confirm,pairings,jobs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=plt.get_cmap('tab10')
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(OUT/(name+'.'+ext),dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharex=True,sharey=True)
    for ax,role in zip(axes,['C','B']):
        for row in confirm:
            if row['role']!=role:continue
            trace=jobs[row['name']][2];color=colors(row['seed']-321)
            ax.plot([r['step'] for r in trace],[r['train_online_acc'] for r in trace],ls='--',color=color,lw=.7,alpha=.5)
            ax.plot([r['step'] for r in trace],[r['val_acc'] for r in trace],color=color,lw=.8,label=row['seed'])
        ax.set(title=role+(' continuous' if role=='C' else ' categorical'),xlabel='Optimizer updates',xscale='log',xlim=(10,50000),ylim=(0,1.02))
        ax.axhline(.99,color='black',lw=.7,ls=':')
    axes[0].set_ylabel('Accuracy');axes[-1].legend(ncol=2,fontsize=7,title='Seed')
    fig.suptitle('p=97 · dropout 0.2 · dashed: dropout-active online training; solid: held-out\nCurves end at actual stopping updates')
    save(fig,'train_test')
    fig,ax=plt.subplots(figsize=(9,4))
    keys=['C_gen99_updates','C_mem99_updates','B_gen99_updates','B_mem99_updates']
    for i,r in enumerate(pairings):
        vals=[r[k] for k in keys];x=np.arange(4)+(i-4.5)*.018;y=[50000 if v is None else v for v in vals]
        ax.plot(x,y,color=colors(i),alpha=.45,lw=.8,label=r['seed'])
        for xx,yy,v in zip(x,y,vals):ax.scatter(xx,yy,color=colors(i),marker='^' if v is None else 'o',s=25)
    ax.set(xticks=range(4),xticklabels=['C arithmetic','C random labels','B arithmetic','B random labels'],yscale='log',
        ylabel='Optimizer updates to first 99%',title='Matched 4,656-example tasks · dropout 0.2\nArithmetic: held-out99; random: online training99; triangles: censored at 50,000')
    ax.legend(title='Seed',ncol=2,fontsize=7);save(fig,'learning_times')
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharex=True)
    for ax,key,title in zip(axes,['paper_delay_updates','eval_delay_updates'],['Paper metric: online training99','Secondary: dropout-free training99']):
        for offset,role,color in [(-.16,'C','tab:blue'),(.16,'B','tab:orange')]:
            vals=[next(r[key] for r in confirm if r['seed']==seed and r['role']==role) for seed in range(321,331)]
            for i,v in enumerate(vals):
                if v is None:ax.text(i+offset,0,'?',ha='center',color=color)
                else:ax.scatter(i+offset,v,color=color,label=role if i==0 else None)
        ax.set(xticks=range(10),xticklabels=range(321,331),xlabel='Seed',title=title,ylabel='Delay (optimizer updates)');ax.axhline(0,c='black',lw=.6)
        ax.legend()
    fig.suptitle('Delay = max(0, first held-out98 − first training99); ? means undefined')
    save(fig,'grokking_delays')
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharex=True,sharey=True)
    for ax,role in zip(axes,['C','B']):
        for result,manifest,trace in jobs.values():
            config=result['config']
            if config['task']!='memorisation' or config['role']!=role:continue
            color=colors(config['seed']-321)
            ax.plot([r['step'] for r in trace],[r['train_online_acc'] for r in trace],color=color,lw=.7,alpha=.7,label=config['seed'])
            ax.scatter(trace[-1]['step'],result['final_eval']['acc'],color=color,marker='x',s=35)
        ax.set(title=role,xlabel='Optimizer updates',xscale='log',xlim=(10,60000),ylim=(0,1.02))
        ax.axhline(.99,color='black',lw=.7,ls=':')
    axes[0].set_ylabel('Random-label training accuracy');axes[-1].legend(title='Seed',ncol=2,fontsize=7)
    fig.suptitle('4,656 random examples · dropout 0.2\nLines: dropout-active online accuracy; crosses: final dropout-free accuracy')
    save(fig,'random_learning')


if __name__=='__main__':main()
