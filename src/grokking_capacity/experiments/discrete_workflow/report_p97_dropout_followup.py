#!/usr/bin/env python3
"""Recompute the registered follow-up, preserving censoring and source provenance.

Safe to run during training: only completed registry rows are analyzed. Final
claims require controller.phase == complete and all expected grid cells.
Checkpoint audits use weights_only=True and an independent CPU forward pass.
"""
from grokking_capacity.experiments.registered import verify_sources as verify_registered_sources

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'src'))
RUN = ROOT/'results/discrete/p97-dropout-followup'
OUT = ROOT/'reports/p97_dropout_followup'
D = 4656 * math.log2(99)
DIMS = [10,12,14,16,18,20,22]
SIZES = [1000,1374,1891,2601,3576,4917,6761,9300]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    if not rows:
        return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def read_rows(path, group):
    with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        return [{**dict(row),'database':str(path),'group':group} for row in db.execute('SELECT * FROM runs WHERE status="completed"')]


def crossing(a, threshold):
    hits=np.flatnonzero(a>=threshold)
    return int(hits[0])+1 if len(hits) else None


def arithmetic(row):
    with np.load(row['npz_path']) as z:
        for key in ('dim','p','param_count','depth','heads'):
            assert z[key].item()==row[key],key
        assert z['n_train'].item()==4656
        tr,va=z['train_acc'],z['val_acc']
        assert len(tr)==len(va) and np.isfinite(tr).all() and np.isfinite(va).all()
        t,v=crossing(tr,99),crossing(va,98)
        delay=None if t is None or v is None else max(0,v-t)
        result=dict(seed=row['seed'],uuid=row['uuid'],param_count=row['param_count'],epochs=len(tr),updates=len(tr)*10,
                    first_train99_epoch=t,first_val98_epoch=v,first_val99_epoch=crossing(va,99),
                    delay_epochs=delay,delay_updates=None if delay is None else delay*10,
                    final_online_train_pct=float(tr[-1]),final_val_pct=float(va[-1]),
                    max_online_train_pct=float(tr.max()),max_val_pct=float(va.max()),
                    joint99_reached=bool(np.any((tr>=99)&(va>=99))),
                    hit_epoch_ceiling=len(tr)==5000,missing_train99=t is None,missing_val98=v is None,
                    npz_path=row['npz_path'],npz_sha256=sha(row['npz_path']))
        assert result['first_val99_epoch']==row['grokking_epoch']
        if len(tr)<5000:
            assert result['joint99_reached'] and tr[-1]>=99 and va[-1]>=99
        return result


def capacity(row):
    with np.load(row['npz_path']) as z:
        for key in ('n_samples','dim','depth','heads','param_count','epochs_trained'):
            assert z[key].item()==row[key],(key,row['uuid'])
        n=row['n_samples'];bits=float(z['total_bits_memorized']);trace=z['bits_trace']*n
        assert np.isfinite(trace).all() and len(trace)==row['epochs_trained']
        assert abs(bits-row['total_bits_memorized'])<1e-8
        assert abs(bits-trace[-1])<1e-8
        formula=n*(math.log2(row['p']+2)-float(z['final_loss'])/math.log(2))
        assert abs(formula-bits)<.02,(formula,bits)
        previous=float(np.mean(trace[-200:-100])) if len(trace)>=200 else None
        recent=float(np.mean(trace[-100:]))
        drift=None if previous is None else (recent-previous)/max(abs(recent),1)
        return dict(group=row['group'],seed=row['seed'],p=row['p'],dim=row['dim'],n_samples=n,
                    param_count=row['param_count'],final_bits=bits,peak_bits=float(trace.max()),
                    final_bits_per_parameter=bits/row['param_count'],epochs=row['epochs_trained'],
                    final_online_train_acc=row['final_acc'],final_eval_loss=row['final_loss'],
                    label_entropy_bits=n*math.log2(row['p']+2),entropy_fraction=bits/(n*math.log2(row['p']+2)),
                    late_relative_drift=drift,formula_error_bits=abs(formula-bits),
                    max_epochs_reached=row['epochs_trained']==5000,uuid=row['uuid'],
                    git_hash=row['git_hash'],git_dirty=row['git_dirty'],host=row['host'],
                    npz_path=row['npz_path'],npz_sha256=sha(row['npz_path']))


def linear_fit(points, ykey='final_bits'):
    x=np.array([p['param_count'] for p in points],float)
    y=np.array([p[ykey] for p in points],float)
    slope,intercept=np.polyfit(x,y,1)
    r2=1-np.sum((y-(slope*x+intercept))**2)/np.sum((y-y.mean())**2)
    return dict(slope_bits_per_parameter=float(slope),intercept_bits=float(intercept),r_squared=float(r2),
                through_origin_slope=float(np.dot(x,y)/np.dot(x,x)))


def capacity_fit(rows):
    expected={(d,n) for d in DIMS for n in SIZES}
    observed={(r['dim'],r['n_samples']) for r in rows}
    assert len(observed)==len(rows),'duplicate grid cells'
    if observed != expected:
        return dict(complete=False,completed_cells=len(rows),expected_cells=len(expected))
    points=[max((r for r in rows if r['dim']==dim),key=lambda r:r['final_bits']) for dim in DIMS]
    result=dict(complete=True,**linear_fit(points),points=points)
    rng=np.random.default_rng(47061)
    slopes=[]
    x=np.array([p['param_count'] for p in points],float);y=np.array([p['final_bits'] for p in points],float)
    for _ in range(10000):
        idx=rng.integers(0,len(points),len(points))
        if len(np.unique(idx))>1:
            slopes.append(float(np.polyfit(x[idx],y[idx],1)[0]))
    result['width_bootstrap_slope_95pct_interval']=np.quantile(slopes,[.025,.975]).tolist()
    result['bootstrap_scope']='Width-point resampling: fit sensitivity only, not a population or seed confidence interval.'
    peakpoints=[max((r for r in rows if r['dim']==dim),key=lambda r:r['peak_bits']) for dim in DIMS]
    result['max_over_trace_sensitivity']=linear_fit(peakpoints,'peak_bits')
    result['data_limited_winner_dimensions']=[p['dim'] for p in points if p['entropy_fraction']>=.95]
    result['largest_dataset_winner_dimensions']=[p['dim'] for p in points if p['n_samples']==max(SIZES)]
    result['winner_drift_over_one_percent_dimensions']=[p['dim'] for p in points if p['late_relative_drift'] is not None and abs(p['late_relative_drift'])>.01]
    result['winner_at_epoch_ceiling_dimensions']=[p['dim'] for p in points if p['max_epochs_reached']]
    return result


def audit_checkpoint(row):
    import torch
    import torch.nn.functional as F
    from grokking_capacity.models import TransformerTorch,count_parameters
    from grokking_capacity.data import grokking_data_torch,random_target_data_torch
    torch.set_num_threads(2)
    path=Path(row['artefacts_dir'])/'model.pt'
    ckpt=torch.load(path,map_location='cpu',weights_only=True)
    metadata=ckpt['metadata']
    args={k:metadata[k] for k in ('depth','dim','heads','n_tokens','seq_len','dropout')}
    model=TransformerTorch(**args)
    model.load_state_dict(ckpt['model_state_dict']);model.eval()
    assert count_parameters(model)==row['param_count']
    np.random.seed(row['seed']);torch.manual_seed(row['seed'])
    if row['experiment_type']=='groks':
        datasets=grokking_data_torch(row['p'],op='/',split_type='random',train_fraction=.5,device='cpu')
        pairs=[datasets[:2],datasets[2:]]
    else:
        pairs=[random_target_data_torch(row['n_samples'],row['p'],seq_len=4,device='cpu')]
    metrics=[]
    with torch.no_grad():
        for X,T in pairs:
            correct=0;loss=0.;logprob=0.
            for offset in range(0,len(T),512):
                xb,tb=X[offset:offset+512],T[offset:offset+512]
                logits=model(xb)
                correct+=int((logits.argmax(-1)==tb).sum())
                loss+=float(F.cross_entropy(logits,tb,reduction='sum'))
                lp=F.log_softmax(logits,-1).gather(1,tb[:,None]).squeeze(1)
                logprob+=float((lp/math.log(2)).double().sum())
            h=hashlib.sha256()
            h.update(X.numpy().tobytes());h.update(T.numpy().tobytes())
            metrics.append(dict(n=len(T),accuracy=correct/len(T),loss=loss/len(T),
                                learned_bits=len(T)*math.log2(row['p']+2)+logprob,dataset_sha256=h.hexdigest()))
    result=dict(uuid=row['uuid'],kind=row['experiment_type'],seed=row['seed'],p=row['p'],dim=row['dim'],
                n_samples=row['n_samples'],checkpoint_path=str(path),checkpoint_sha256=sha(path),
                npz_sha256=sha(row['npz_path']),metrics=metrics,param_count=count_parameters(model))
    if row['experiment_type']=='groks':
        with np.load(row['npz_path']) as z:
            result['heldout_accuracy_error']=abs(metrics[1]['accuracy']-float(z['val_acc'][-1])/100)
        assert result['heldout_accuracy_error']<1e-8
        initial=Path(row['artefacts_dir'])/'model_init.pt'
        ini=torch.load(initial,map_location='cpu',weights_only=True)
        state=ini.get('model_state_dict',ini)
        torch.manual_seed(row['seed'])
        reference=TransformerTorch(**args,init_scale=row['init_scale']).state_dict()
        assert all(torch.equal(reference[k],v) for k,v in state.items())
        result['initialization_exact_match']=True;result['initial_checkpoint_sha256']=sha(initial)
        result['online_training_accuracy_note']='Final eval-mode training accuracy is diagnostic; it does not reconstruct training-mode minibatch accuracy during the epoch.'
    else:
        result['capacity_error_bits']=abs(metrics[0]['learned_bits']-row['total_bits_memorized'])
        # Independent CPU kernels and floating-point reduction can differ slightly.
        assert result['capacity_error_bits']<.1,result
    return result


def plots(arows,crows,fits):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,5,figsize=(16,6),sharex=True,sharey=True)
    for ax,seed in zip(axes.flat,range(73,83)):
        row=next((r for r in arows if r['seed']==seed),None)
        if row:
            with np.load(row['npz_path']) as z:
                e=np.arange(1,len(z['train_acc'])+1)
                ax.plot(e,z['train_acc'],lw=.65,alpha=.8,label='Train, dropout on')
                ax.plot(e,z['val_acc'],lw=.65,alpha=.8,label='Held out, dropout off')
            label='missing train 99%' if row['delay_epochs'] is None else f"delay {row['delay_epochs']} epochs"
            ax.set_title(f'Seed {seed}: {label}')
        else:
            ax.set_title(f'Seed {seed}: pending')
        ax.axhline(99,color='.6',lw=.5,ls=':');ax.set_ylim(0,102);ax.set_xlim(0,5000)
        ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8,loc='lower right');fig.supxlabel('Epoch (10 optimizer updates)')
    fig.supylabel('Accuracy (%)');fig.suptitle('Fixed A: p97 division, width 26 / FF104, dropout 0.2')
    fig.tight_layout();fig.savefig(OUT/'A_accuracy.png',dpi=180);fig.savefig(OUT/'A_accuracy.pdf');plt.close(fig)
    labels={'historical_defaults':'Historical defaults: dropout 0 / WD 0.01, seed 42',
            'historical_matched':'Matched: dropout 0.2 / WD 1, seed 42',
            'fresh_matched':'Fresh matched: dropout 0.2 / WD 1, seed 61'}
    fig,ax=plt.subplots(figsize=(9,5.5))
    for group,fit in fits.items():
        if not fit['complete']:continue
        points=fit['points'];x=np.array([p['param_count'] for p in points]);y=np.array([p['final_bits'] for p in points])
        line=ax.plot(x/1000,y/1000,'o',label=labels[group]+f"; C={fit['slope_bits_per_parameter']:.3f}")[0]
        xx=np.array([5000,21500]);ax.plot(xx/1000,(fit['slope_bits_per_parameter']*xx+fit['intercept_bits'])/1000,color=line.get_color())
    xx=np.array([5000,21500]);ax.plot(xx/1000,2.16*xx/1000,'--',color='.5',label='Paper reference: 2.16 P (not a refit)')
    ax.set(xlabel='Parameters (thousands)',ylabel='Maximum final learned bits (thousands)',title='Local p113 capacity calibration: maximize over eight dataset sizes')
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'capacity_fits.png',dpi=180);fig.savefig(OUT/'capacity_fits.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),sharey=True)
    for ax,(group,label) in zip(axes,labels.items()):
        for dim in DIMS:
            rows=sorted((r for r in crows if r['group']==group and r['dim']==dim),key=lambda r:r['n_samples'])
            if rows:ax.plot([r['n_samples'] for r in rows],[r['final_bits']/1000 for r in rows],'-o',ms=3,label=f'd={dim}')
        ax.set_title(label.replace(': ','\n').replace(', seed','\nseed'),fontsize=10)
        ax.set_xlabel('Random training examples');ax.grid(alpha=.2)
    axes[0].set_ylabel('Final learned bits (thousands)');axes[-1].legend(fontsize=8,ncol=2)
    fig.tight_layout();fig.savefig(OUT/'capacity_grid.png',dpi=180);fig.savefig(OUT/'capacity_grid.pdf');plt.close(fig)
    direct=[r for r in crows if r['group']=='A_direct']
    if direct:
        fig,ax=plt.subplots(figsize=(7,4.5))
        for seed in (61,62):
            rows=sorted((r for r in direct if r['seed']==seed),key=lambda r:r['n_samples'])
            ax.plot([r['n_samples'] for r in rows],[r['final_bits']/1000 for r in rows],'-o',label=f'Seed {seed}')
        ax.axhline(D/1000,ls='--',color='.4',label='Modular dataset: 30,866 bits')
        ax.set(xlabel='Random training examples',ylabel='Final learned bits (thousands)',title='Direct A capacity with dropout 0.2 / weight decay 1')
        ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'A_capacity.png',dpi=180);fig.savefig(OUT/'A_capacity.pdf');plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit',action='store_true');parser.add_argument('--plots',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    historical=[]
    for group,path in [('historical_defaults','/home/ubuntu/capacity_defaults_runs.db'),('historical_matched','/home/ubuntu/capacity_matched_runs.db')]:
        rr=read_rows(path,group)
        assert len(rr)==56
        for r in rr:
            assert r['host']=='68-209-74-145' and r['p']==113 and r['seed']==42 and r['dataset_type']=='random'
            assert r['dropout']==(0 if group=='historical_defaults' else .2)
            assert r['weight_decay']==(.01 if group=='historical_defaults' else 1.)
        historical.extend(rr)
    fresh=[]
    for path in sorted(RUN.glob('lane_*.db')):
        for r in read_rows(path,'fresh'):
            r['group']='A_arithmetic' if r['experiment_type']=='groks' else ('A_direct' if r['p']==97 else 'fresh_matched')
            fresh.append(r)
    for r in historical+fresh:
        expected_common={'depth':2,'heads':1,'init_scale':1.,'lr':.001,'beta1':.9,'beta2':.98,
                         'batch_size':512,'max_epochs':5000,'mup_base':0}
        for key,value in expected_common.items():
            assert r[key]==value,(r['uuid'],key,r[key],value)
    expected_fresh={('groks',97,26,4656,seed) for seed in range(73,83)}
    expected_fresh.update(('capacity',97,26,n,seed) for n in (2048,4096,8192,16384) for seed in (61,62))
    expected_fresh.update(('capacity',113,dim,n,61) for dim in DIMS for n in SIZES)
    identities=[(r['experiment_type'],r['p'],r['dim'],r['n_samples'],r['seed']) for r in fresh]
    assert len(identities)==len(set(identities)) and set(identities)<=expected_fresh
    for r in fresh:
        assert r['dropout']==.2 and r['weight_decay']==1. and r['host']=='68-209-74-145'
    allrows=historical+fresh
    write_csv(OUT/'registry_rows.csv',allrows)
    arows=sorted([arithmetic(r) for r in fresh if r['group']=='A_arithmetic'],key=lambda r:r['seed'])
    crows=[capacity(r) for r in allrows if r['experiment_type']=='capacity']
    write_csv(OUT/'A_runs.csv',arows);write_csv(OUT/'capacity_runs.csv',crows)
    fits={g:capacity_fit([r for r in crows if r['group']==g]) for g in ('historical_defaults','historical_matched','fresh_matched')}
    delays=[r['delay_epochs'] for r in arows if r['delay_epochs'] is not None]
    summary=dict(completed_runs=len(arows),expected_runs=10,observed_delays=len(delays),
                 missing_delays=len(arows)-len(delays),zero_delays=delays.count(0),positive_delays=sum(d>0 for d in delays),
                 minimum_observed_delay_epochs=min(delays) if delays else None,
                 ten_seed_minimum_fully_observed=len(delays)==10,
                 train99_count=sum(r['first_train99_epoch'] is not None for r in arows),
                 heldout98_count=sum(r['first_val98_epoch'] is not None for r in arows),
                 heldout99_count=sum(r['first_val99_epoch'] is not None for r in arows),
                 joint99_count=sum(r['joint99_reached'] for r in arows),dataset_bits=D,param_count=26910)
    direct={}
    for seed in (61,62):
        rr=[r for r in crows if r['group']=='A_direct' and r['seed']==seed]
        if rr:
            best=max(rr,key=lambda r:r['final_bits'])
            direct[str(seed)]={'complete':{r['n_samples'] for r in rr}=={2048,4096,8192,16384},
                               'winner':best,'capacity_to_dataset_ratio':best['final_bits']/D}
    controller=json.loads((RUN/'controller.json').read_text())
    verify_registered_sources(controller['source_hashes'])
    for path,digest in controller['source_hashes'].items():
        assert sha(RUN/'source'/path)==digest,path
    assert sha(RUN/'configuration.yaml')==controller['config_sha256']
    audits_path=OUT/'checkpoint_audits.json'
    audits=json.loads(audits_path.read_text()) if audits_path.exists() else {}
    if args.audit:
        for row in fresh:
            cached=audits.get(row['uuid'])
            if cached:
                assert cached['checkpoint_sha256']==sha(Path(row['artefacts_dir'])/'model.pt')
                assert cached['npz_sha256']==sha(row['npz_path'])
                continue
            audits[row['uuid']]=audit_checkpoint(row)
            save_json(audits_path,audits)
            print('audited',row['group'],row['dim'],row['seed'],row['n_samples'],flush=True)
    for row in arows:
        if row['uuid'] in audits:
            row['final_eval_train_pct']=100*audits[row['uuid']]['metrics'][0]['accuracy']
            row['final_eval_heldout_pct']=100*audits[row['uuid']]['metrics'][1]['accuracy']
    write_csv(OUT/'A_runs.csv',arows)
    report=dict(controller_phase=controller['phase'],controller_jobs_completed=len(controller['completed']),
                expected_jobs=74,analysis_script_sha256=sha(Path(__file__)),source_commit=controller['git_commit'],source_hashes=controller['source_hashes'],
                configuration_sha256=controller['config_sha256'],arithmetic=summary,capacity_fits=fits,A_direct_capacity=direct,
                audited_fresh_checkpoints=len(audits),historical_dirty_source_rows=sum(r['git_dirty'] for r in historical),
                historical_commits=sorted({r['git_hash'] for r in historical}),
                limits=['Capacity is finite-training and protocol-dependent, not an architecture-only bound.',
                        'Historical default vs matched changes both dropout and weight decay.',
                        'Compared with prior A/B/C: original runner also changes batch RNG, optimizer implementation, evaluation and 5000 vs 12000 epoch horizon; not a dropout-only causal ablation.',
                        'Historical dirty source flags cannot be resolved by inspecting recorded commits.',
                        'Online dropout training accuracy is not full-training-set eval-mode accuracy.',
                        'Minimum is over observed delays; missing threshold crossings remain censored.'])
    if controller['phase']=='complete':
        assert not controller['running'] and not controller['pending_names'] and not controller['errors']
        assert len(controller['completed'])==74 and all(r['exit_code']==0 for r in controller['completed'])
        assert {r['name'] for r in controller['completed']}=={r['name'] for r in controller['jobs']}
        assert len(fresh)==74 and len(arows)==10 and all(f['complete'] for f in fits.values())
        assert all(r['complete'] for r in direct.values()) and len(direct)==2
        if args.audit:assert set(audits)=={r['uuid'] for r in fresh} and len(audits)==74
    save_json(OUT/'summary.json',report)
    save_json(OUT/'controller_snapshot.json',controller)
    (OUT/'configuration.yaml').write_bytes((RUN/'configuration.yaml').read_bytes())
    if args.plots:plots(arows,crows,fits)
    artifacts={r['npz_path']:sha(r['npz_path']) for r in allrows}
    for r in fresh:
        for path in Path(r['artefacts_dir']).glob('*.pt'):artifacts[str(path)]=sha(path)
    artifacts.update({str(RUN/'source'/p):h for p,h in controller['source_hashes'].items()})
    save_json(OUT/'artifact_hashes.json',artifacts)
    if controller['phase']=='complete' and args.audit:
        import torch
        assert len(artifacts)==277
        assert len({r['uuid'] for r in allrows})==186
        for row in fresh:
            md=torch.load(Path(row['artefacts_dir'])/'model.pt',map_location='cpu',weights_only=True)['metadata']
            for key in ('p','dim','depth','heads','dropout','param_count'):
                assert md[key]==row[key],(row['uuid'],key)
            assert md['n_tokens']==row['p']+2 and md['seq_len']==4
        save_json(OUT/'validation.json',dict(
            new_jobs=74,completed_registry_rows=186,historical_traces_checked=112,
            final_checkpoints_audited=len(audits),initial_checkpoints_audited=10,
            hashed_input_artifacts=len(artifacts),checkpoint_metadata_matches_registry=True,
            source_snapshot_matches_current=True,all_job_exit_codes_zero=True,
            max_capacity_checkpoint_error_bits=max(a.get('capacity_error_bits',0) for a in audits.values()),
            max_heldout_accuracy_error=max(a.get('heldout_accuracy_error',0) for a in audits.values()),
            analysis_script_sha256=sha(Path(__file__))))
    print(json.dumps({'phase':controller['phase'],'arithmetic':summary,'fits':{k:{a:b for a,b in v.items() if a!='points'} for k,v in fits.items()},'A_direct':direct,'audited':len(audits)},indent=2))


if __name__=='__main__':
    main()
