#!/usr/bin/env python3
"""Matched A100 decay-law analysis, with common seeds and explicit censoring."""
from pathlib import Path
import argparse,csv,hashlib,itertools,json,sqlite3
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
DEFAULT_OUT=ROOT/'results/weight_decay_matched_20261103'
LAMBDAS=[.01,.03,.1,.3,1.]
DIMS=[20,24,28,32,40,48,56,64,80,96,112,128,160,192,224,256]
SEEDS=[42,43,44,45,46]
PROTOCOL={'status':'completed','experiment_type':'groks','host':'147-224-250-73','gpu_type':'A100','p':113,'operation':'/','train_fraction':.5,'split_type':'random','dataset_type':'modular','n_samples':6328,'depth':2,'heads':1,'dropout':.2,'init_scale':1.,'lr':.001,'beta1':.9,'beta2':.98,'batch_size':512,'max_epochs':5000,'mup_base':0,'architecture_family':'transformer_gated'}
def slopes(times):
    x=np.log(LAMBDAS);xc=x-x.mean()
    return np.log(times)@xc/(xc@xc)
def analyse(db=ROOT/'runs.db',out=DEFAULT_OUT):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    c=sqlite3.connect(Path(db).resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
    q='select * from runs where '+' and '.join(f'{k}=?' for k in PROTOCOL)
    rr=[dict(r) for r in c.execute(q,list(PROTOCOL.values())) if r['dim'] in DIMS and r['weight_decay'] in LAMBDAS and r['seed'] in SEEDS];c.close()
    idx={(r['dim'],r['weight_decay'],r['seed']):r for r in rr}
    assert len(rr)==len(idx)==len(DIMS)*len(LAMBDAS)*len(SEEDS)==400,'Missing or duplicate matched cells'
    restricted=np.empty((len(DIMS),len(LAMBDAS),len(SEEDS)));observed=np.empty_like(restricted);events=np.zeros_like(restricted,dtype=bool);audits=[]
    for i,d in enumerate(DIMS):
        for j,w in enumerate(LAMBDAS):
            for k,s in enumerate(SEEDS):
                r=idx[d,w,s];p=Path(r['npz_path'])
                with np.load(p,allow_pickle=False) as z:
                    va=z['val_acc'];assert np.all(np.isfinite(va));hit=np.flatnonzero(va>=99)
                    first=int(hit[0]+1) if len(hit) else None
                    assert first==r['grokking_epoch'],(r['uuid'],first,r['grokking_epoch'])
                    if first is None:assert len(va)==5000,'Unexpected early censoring'
                events[i,j,k]=first is not None;restricted[i,j,k]=first if first is not None else 5000;observed[i,j,k]=first if first is not None else np.nan
                audits.append({**{key:r[key] for key in ['uuid','host','gpu_type','git_hash','git_dirty','dim','weight_decay','seed','param_count','npz_path']},'first_val99':first,'event_observed':first is not None,'followup_epochs':len(va),'restricted_time_5000':restricted[i,j,k],'trace_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    complete=events.all(axis=(1,2));primary=slopes(restricted[complete].mean(axis=2));perdim=slopes(restricted.mean(axis=2))
    # Exact nonparametric paired-seed bootstrap, conditional on this fixed width grid.
    # Each seed index is resampled jointly across all decay values and widths.
    boots=[]
    for inds in itertools.product(range(len(SEEDS)),repeat=len(SEEDS)):
        means=restricted[complete][:,:,list(inds)].mean(axis=2);boots.append(float(np.median(slopes(means))))
    cells=[]
    for i,d in enumerate(DIMS):
        cells.append({'dim':d,'included_primary':bool(complete[i]),'censored_runs':int((~events[i]).sum()),'median_seed_time_slope':float(perdim[i]),'mean_times_restricted_5000':restricted[i].mean(axis=1).tolist(),'mean_times_observed_only':np.nanmean(observed[i],axis=1).tolist(),'events_by_lambda':events[i].sum(axis=1).tolist()})
    result={'protocol':PROTOCOL,'lambdas':LAMBDAS,'seeds':SEEDS,'dims':DIMS,'selected_runs':len(rr),'censored_runs':int((~events).sum()),'primary':{'estimator':'Median across fully observed widths of OLS slope of log(arithmetic seed-mean first-val99 epochs) versus log(weight_decay). Same five seeds at every decay/width. Entire width groups with any censored outcome are excluded from the primary estimate and retained in the sensitivity analysis.','dims':[d for d,ok in zip(DIMS,complete) if ok],'runs':int(complete.sum())*25,'slope':float(np.median(primary)),'seed_bootstrap_CI95':np.quantile(boots,[.025,.975]).tolist(),'bootstrap_draws':len(boots),'bootstrap_scope':'Exact paired resampling of five seeds; fixed observed width inclusion mask and fixed width grid; descriptive, not a population-of-widths interval.'},'sensitivity':{'all_widths_restricted_mean_to_5000_slope':float(np.median(perdim)),'all_widths_observed_only_slope':float(np.median(slopes(np.nanmean(observed,axis=2)))),'excluded_dims':[d for d,ok in zip(DIMS,complete) if not ok],'restriction_note':'Restricted means use min(T,5000), a different estimand; censored outcomes are lower bounds on unrestricted times, not events at epoch 5000.'},'per_width':cells,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'runs.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(audits[0]));w.writeheader();w.writerows(audits)
    plot(result,out)
    return result

def plot(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    primary=result['primary'];x=np.array(LAMBDAS)
    fig,ax=plt.subplots(figsize=(5.6,4))
    for d,color in [(56,'#2a78d6'),(112,'#eb6834'),(224,'#1baf7a')]:
        cell=next(c for c in result['per_width'] if c['dim']==d);assert cell['included_primary']
        ax.plot(x,cell['mean_times_restricted_5000'],'o-',color=color,label=f'd={d} (5 matched seeds)')
    # Reference anchored to the measured d112 mean at lambda=.1; slope is the primary summary.
    anchor=next(c for c in result['per_width'] if c['dim']==112)['mean_times_restricted_5000'][2];xx=np.geomspace(.01,1,100)
    ax.plot(xx,anchor*(xx/.1)**primary['slope'],'k--',lw=1.5,label=rf"Median slope: ${primary['slope']:.2f}$ (12 complete widths)")
    ax.plot(xx,anchor*(xx/.1)**-1,':',color='#c23b3b',label=r'Reference: $\lambda^{-1}$')
    ax.set(xscale='log',yscale='log',xlabel=r'Weight decay $\lambda$',ylabel=r'$T_{\mathrm{gen}}$ (epochs)',title='Matched A100 protocol, p=113');ax.grid(alpha=.25);ax.legend(fontsize=7.5);fig.tight_layout();fig.savefig(Path(out)/'tgen_lambda.pdf');fig.savefig(Path(out)/'tgen_lambda.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for c in result['per_width']:
        ax.scatter(c['dim'],c['median_seed_time_slope'],marker='o' if c['included_primary'] else '^',facecolors='#2a78d6' if c['included_primary'] else 'none',edgecolors='#2a78d6' if c['included_primary'] else '#c23b3b')
    ax.axhline(primary['slope'],color='black',label=f"Primary median: {primary['slope']:.3f}");ax.axhline(-.5,color='grey',ls='--',label='Half-power reference');ax.set(xlabel='Width',ylabel='Slope versus weight decay',title='Filled: complete widths; open: restricted-time sensitivity');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(Path(out)/'width_sensitivity.png',dpi=160);plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--db',type=Path,default=ROOT/'runs.db');ap.add_argument('--out',type=Path,default=DEFAULT_OUT);args=ap.parse_args()
    r=analyse(args.db,args.out);print(json.dumps({k:r[k] for k in ['selected_runs','censored_runs','primary','sensitivity']},indent=2))
if __name__=='__main__':main()
