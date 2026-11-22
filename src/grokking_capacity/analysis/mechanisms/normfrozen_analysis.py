#!/usr/bin/env python3
"""Analyse explicitly paired frozen/unfrozen runs from one H100 campaign.

Requires separate arm registries, identical starting data/model/RNG hashes,
and complete outcomes. Never imports a control arm from the merged registry.
"""
from pathlib import Path
import argparse,hashlib,itertools,json,sqlite3
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
DEFAULT_CAMPAIGN=ROOT/'results/normfrozen_h100_paired_20261103'
SETTINGS=[(128,.1),(128,.3),(128,1.),(64,1.),(256,1.)]
SEEDS=[42,43,44]
def model_hashes(state):
    return {k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in state.items()}
def fit_metrics(times,parameters):
    lam_slope=float(np.polyfit(np.log([.1,.3,1.]),np.log(times[:3].mean(axis=1)),1)[0])
    indices=[3,2,4]
    b=-float(np.polyfit(np.log([parameters[i] for i in indices]),np.log(times[indices].mean(axis=1)),1)[0])
    return lam_slope,b

def analyse(campaign=DEFAULT_CAMPAIGN,verify_checkpoints=True):
    campaign=Path(campaign);manifest=json.loads((campaign/'manifest.json').read_text())
    completion=json.loads((campaign/'completion.json').read_text());assert len(completion)==30 and all(r['returncode']==0 for r in completion)
    assert manifest['settings']==[list(x) for x in SETTINGS] and manifest['seeds']==SEEDS
    # Source hashes at training time are copied into the campaign; verify the snapshot.
    for rel,h in manifest['source_hashes'].items():assert hashlib.sha256((campaign/'source'/rel).read_bytes()).hexdigest()==h,rel
    rows={};times={};metas={};checks=[];all_uuids=[];params=[]
    for arm in ['frozen','unfrozen']:
        c=sqlite3.connect((campaign/(arm+'.db')).resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
        rs=[dict(r) for r in c.execute('select * from runs')];c.close();assert len(rs)==15 and all(r['status']=='completed' for r in rs)
        idx={(r['dim'],r['weight_decay'],r['seed']):r for r in rs};assert len(idx)==15
        times[arm]=np.empty((5,3));rows[arm]=[]
        for i,(d,w) in enumerate(SETTINGS):
            for k,s in enumerate(SEEDS):
                r=idx[d,w,s];all_uuids.append(r['uuid']);meta=json.loads((campaign/'jobs'/f'd{d}_wd{w:g}_s{s}_{arm}.json').read_text());metas[arm,d,w,s]=meta
                assert r['host']==manifest['host'] and r['gpu_type']=='H100' and meta['host']==manifest['host']
                assert r['p']==113 and r['operation']=='/' and r['train_fraction']==.5 and r['depth']==2 and r['heads']==1 and r['dropout']==.2 and r['init_scale']==1 and r['batch_size']==512 and r['mup_base']==0
                assert r['lr']==.001 and r['beta1']==.9 and r['beta2']==.98 and r['max_epochs']==5000
                assert r['param_count']==meta['trainable_parameters']
                assert meta['arm']==arm and meta['freeze_norm_scales']==(arm=='frozen') and meta['frozen_parameters_unchanged']
                with np.load(r['npz_path'],allow_pickle=False) as z:
                    tr=z['train_acc'];va=z['val_acc'];assert np.all(np.isfinite(tr)) and np.all(np.isfinite(va))
                    hits=np.flatnonzero(va>=99);first=int(hits[0]+1) if len(hits) else None
                    assert first==r['grokking_epoch'];assert len(va)==r['epochs_trained']
                    if first is None:raise RuntimeError('Censored outcome: preserve row and revise analysis; do not omit it from the paired slope')
                    assert tr[-1]>=99 and va[-1]>=99
                times[arm][i,k]=first
                row={key:r[key] for key in ['uuid','dim','weight_decay','seed','host','gpu_type','git_hash','git_dirty','grokking_epoch','epochs_trained','npz_path','param_count','wallclock_seconds']};row.update(arm=arm,total_parameters=meta['total_parameters'],trainable_parameters=meta['trainable_parameters']);rows[arm].append(row)
                if arm=='frozen' and k==0:params.append(meta['total_parameters'])
                if verify_checkpoints:checks.append(checkpoint_check(r,meta))
    assert len(set(all_uuids))==30
    pairs=[]
    for d,w in SETTINGS:
        for s in SEEDS:
            a=metas['frozen',d,w,s];b=metas['unfrozen',d,w,s]
            for field in ['dataset_hashes','initial_state_hashes','cpu_rng_hash','cuda_rng_hash','total_parameters','host','gpu_uuid','torch','cuda','dtype','tf32_matmul','tf32_cudnn']:assert a[field]==b[field],(d,w,s,field)
            pairs.append({'dim':d,'weight_decay':w,'seed':s,'identical_data_initialisation_rng':True,'gpu_uuid':a['gpu_uuid']})
    result={'campaign':str(campaign),'manifest_sha256':hashlib.sha256((campaign/'manifest.json').read_bytes()).hexdigest(),'settings':SETTINGS,'seeds':SEEDS,'pairs_verified':len(pairs),'host':manifest['host'],'gpu_uuid':pairs[0]['gpu_uuid'],'censored_runs':0,'paired_checks':pairs,'checkpoint_checks':checks,'width_parameter_convention':'Common total architecture parameter count in both arms; trainable counts also retained per run.','bootstrap_scope':'Exact paired resampling of 3 seeds jointly over both arms and all settings (27 draws); conditional on this grid; descriptive uncertainty only.'}
    boot={arm:[] for arm in times};delta=[]
    for inds in itertools.product(range(3),repeat=3):
        values={arm:fit_metrics(t[:,list(inds)],params) for arm,t in times.items()}
        for arm in times:boot[arm].append(values[arm])
        delta.append(np.asarray(values['frozen'])-np.asarray(values['unfrozen']))
    for arm,t in times.items():
        lam,b=fit_metrics(t,params)
        result[arm]={'lambda_slope_d128':lam,'b_lambda1':b,'seed_bootstrap_CI95':np.quantile(np.array(boot[arm]),[.025,.975],axis=0).T.tolist(),'cells':{f'd{d}_wd{w:g}':{'n':3,'mean_T':float(t[i].mean()),'T':t[i].tolist(),'total_parameters':params[i]} for i,(d,w) in enumerate(SETTINGS)},'runs':rows[arm]}
    result['paired_effect']={'lambda_slope_difference':result['frozen']['lambda_slope_d128']-result['unfrozen']['lambda_slope_d128'],'b_difference':result['frozen']['b_lambda1']-result['unfrozen']['b_lambda1'],'seed_bootstrap_CI95':np.quantile(np.array(delta),[.025,.975],axis=0).T.tolist()}
    result['unfrozen_baseline']=result['unfrozen']  # preserve the published result schema
    (campaign/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');plot(result,campaign);return result

_DATA_CACHE={}
def checkpoint_check(r,meta):
    import torch
    from grokking_capacity.data import grokking_data_torch
    from grokking_capacity.models import TransformerTorch
    torch.set_num_threads(1)
    if r['seed'] not in _DATA_CACHE:
        np.random.seed(r['seed']);_DATA_CACHE[r['seed']]=grokking_data_torch(113,op='/',split_type='random',train_fraction=.5,device='cpu')
    data=_DATA_CACHE[r['seed']]
    assert [hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in data]==meta['dataset_hashes']
    folder=Path(r['npz_path']).parent
    initial=torch.load(folder/'model_init.pt',map_location='cpu',weights_only=True)['model_state_dict']
    state=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)['model_state_dict']
    assert model_hashes(initial)==meta['initial_state_hashes']
    gains=[k for k in state if 'norm.weight' in k]
    max_gain_change=max(float((state[k]-initial[k]).abs().max()) for k in gains)
    assert (max_gain_change==0)==(meta['arm']=='frozen')
    assert any(not torch.equal(state[k],initial[k]) for k in state if k not in gains)
    model=TransformerTorch(depth=2,dim=r['dim'],heads=1,n_tokens=115,seq_len=4,dropout=.2,init_scale=1.)
    model.load_state_dict(state);model.eval();X,Y=data[2:];correct=0
    with torch.no_grad():
        for start in range(0,len(Y),512):correct+=(model(X[start:start+512]).argmax(1)==Y[start:start+512]).sum().item()
    acc=100*correct/len(Y);assert abs(acc-r['final_val_acc'])<=100/len(Y)+1e-8,(r['uuid'],acc,r['final_val_acc'])
    return {'uuid':r['uuid'],'arm':meta['arm'],'cpu_validation_accuracy':acc,'recorded_validation_accuracy':r['final_val_acc'],'maximum_norm_gain_change':max_gain_change,'checkpoint_sha256':hashlib.sha256((folder/'model.pt').read_bytes()).hexdigest()}

def plot(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for arm,col in [('unfrozen','#2a78d6'),('frozen','#eb6834')]:
        res=result[arm];cells=res['cells'];ls=[.1,.3,1.]
        for s in range(3):axs[0].plot(ls,[cells[f'd128_wd{w:g}']['T'][s] for w in ls],color=col,alpha=.3,lw=.8)
        axs[0].plot(ls,[cells[f'd128_wd{w:g}']['mean_T'] for w in ls],'o-',color=col,label=f"{arm}: slope {res['lambda_slope_d128']:.2f}")
        ds=[64,128,256];ps=[cells[f'd{d}_wd1']['total_parameters'] for d in ds]
        axs[1].plot(ps,[cells[f'd{d}_wd1']['mean_T'] for d in ds],'o-',color=col,label=f"{arm}: b {res['b_lambda1']:.2f}")
    axs[0].set(xscale='log',yscale='log',xlabel='Weight decay',ylabel='First validation-99% epoch',title='Width 128: paired seeds on one H100')
    axs[1].set(xscale='log',yscale='log',xlabel='Total architecture parameters',ylabel='First validation-99% epoch',title='Weight decay 1: same three seeds')
    from matplotlib.ticker import NullFormatter
    axs[1].set_xticks(ps,labels=[f'{p/1e6:.2f}M' for p in ps])
    axs[1].xaxis.set_minor_formatter(NullFormatter())
    for ax in axs:ax.grid(alpha=.2,which='both');ax.legend(fontsize=9)
    fig.savefig(out/'normfrozen_paired.pdf');fig.savefig(out/'normfrozen_paired.png',dpi=170);plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--campaign',type=Path,default=DEFAULT_CAMPAIGN);ap.add_argument('--out',type=Path,default=ROOT/'results/theory/normfrozen_analysis.json');args=ap.parse_args()
    result=analyse(args.campaign);args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'frozen':result['frozen']['lambda_slope_d128'],'unfrozen':result['unfrozen']['lambda_slope_d128'],'paired_effect':result['paired_effect'],'pairs_verified':result['pairs_verified'],'checkpoint_checks':len(result['checkpoint_checks'])},indent=2))
if __name__=='__main__':main()
