"""Capacity-constant matched size sweep; prior scientific sources stay frozen."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import torch
from . import multilevel_dropout as base
from .categorical_groks import independent_metrics
from .capacity import CapacityTrainer
from ..models.transformer import count_parameters
from ..utils.optim import build_adamw

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'results/discrete/p97-capacity-scaling-v1'
SOURCES=base.SOURCES+['src/grokking_capacity/experiments/categorical_groks.py',
    'src/grokking_capacity/experiments/capacity_scaling.py',
    'scripts/run_p97_capacity_scaling.py','configs/discrete_p97_capacity_scaling.yaml']
sha=base.sha
atomic_json=base.atomic_json
RECIPE_KEYS=['model','dim','mlp_dim','p','depth','heads','dropout','lr','weight_decay',
    'beta1','beta2','batch_size','alphabet','categorical_temperature',
    'categorical_normalization','categorical_training_sampling']


def read(path):return json.loads(Path(path).read_text())
def source_hashes():return {p:sha(ROOT/p) for p in SOURCES}
def verify_sources(manifest):
    for p,h in manifest['source_hashes'].items():assert sha(ROOT/p)==h,('changed source',p)


def model_spec(role,width):
    if width<2 or width%2:raise ValueError('Use positive even widths for rotary attention')
    if role not in ['B','C']:raise ValueError(role)
    result=dict(model='baseline' if role=='C' else 'categorical',dim=width,mlp_dim=4*width)
    if role=='B':result.update(alphabet=8,categorical_temperature=1.,
        categorical_normalization='slot_layernorm',categorical_training_sampling=False)
    return result


def parameter_count(role,width,p=97):
    # Depth2, heads1, FF4d: 32d^2 continuous, plus32d^2 categorical; embeddings/norms=(2(p+2)+5)d.
    return (32 if role=='C' else 64)*width**2+(2*(p+2)+5)*width


def fit_constant(points):
    if len({r['parameters'] for r in points})<2:raise ValueError('Need at least two architecture sizes')
    x=np.array([r['parameters'] for r in points],float);y=np.array([r['capacity_bits'] for r in points],float)
    if np.any(y<=0):raise ValueError('Capacity must be positive')
    k=float(x@y/(x@x));a,b=np.polyfit(x,y,1);pred=k*x
    return dict(bits_per_parameter=k,relative_residuals=(y/pred-1).tolist(),
        max_absolute_relative_residual=float(np.max(np.abs(y/pred-1))),
        affine_diagnostic=dict(slope=float(a),intercept=float(b)),
        parameter_range=[int(x.min()),int(x.max())],capacity_range=[float(y.min()),float(y.max())],
        per_seed_constants={str(seed):float(sum(r['parameters']*r['capacity_bits'] for r in points if r['seed']==seed)/
            sum(r['parameters']**2 for r in points if r['seed']==seed)) for seed in sorted({r['seed'] for r in points})},points=points)


def match_pairs(constants,widths,candidates,tolerance=.05):
    pairs=[]
    for index,cw in enumerate(widths):
        cp=parameter_count('C',cw);target=constants['C']*cp
        bw=min(candidates,key=lambda w:(abs(constants['B']*parameter_count('B',w)/target-1),w))
        bp=parameter_count('B',bw);error=constants['B']*bp/target-1
        if abs(error)>tolerance:raise ValueError('Cannot match predicted capacity within tolerance')
        pairs.append(dict(index=index,C=model_spec('C',cw),B=model_spec('B',bw),C_parameters=cp,B_parameters=bp,
            C_estimated_bits=target,B_estimated_bits=constants['B']*bp,predicted_relative_mismatch=error))
    if any(a['B']['dim']>=b['B']['dim'] for a,b in zip(pairs,pairs[1:])):raise ValueError('Require distinct increasing B sizes')
    return pairs


def checked_artifact(path):
    path=Path(path);s=read(path/'summary.json');m=read(path/'manifest.json')
    verify_sources(m);assert m['config']==s['config'] and sha(path/'model.pt')==s['checkpoint_sha256']
    return s,m


def check_gate(config):
    path=Path(config['gate_path']);s,m=checked_artifact(path)
    assert s['task']=='arithmetic' and s['learning_gate_passed']
    assert all(s['config'].get(k)==config.get(k) for k in RECIPE_KEYS)
    assert s['final']['train_eval_acc']>=.99 and s['final']['val_acc']>=.99
    result=dict(path=str(path),summary_sha256=sha(path/'summary.json'))
    if config['n_samples']>2048:
        rp=Path(config['random_gate_path']);r,rm=checked_artifact(rp)
        assert r['task']=='capacity' and r['config']['n_samples']==2048 and r['config']['seed']==config['seed']
        assert r['random_learning_passed'] and r['final_eval']['acc']>=.99
        assert all(r['config'].get(k)==config.get(k) for k in RECIPE_KEYS)
        result.update(random_path=str(rp),random_summary_sha256=sha(rp/'summary.json'))
    return result


def train_capacity(model,opt,datasets,config,device,output):
    trainer=CapacityTrainer(model,opt,config['p']+2,config['batch_size'],device)
    cap=trainer.train(datasets,max_epochs=config['epochs'],patience=config['patience'],min_delta=config['min_delta'],verbose=True)
    primary=dict(epochs=cap['epochs_trained'],bits=cap['total_bits_memorized'])
    np.savez(output/'primary_trace.npz',**cap)
    torch.save(dict(model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},config=config),output/'model_primary.pt')
    extensions=[];drift=base.late_drift(trainer.bits_per_example_trace)
    while (drift is None or drift>.01) and len(trainer.bits_per_example_trace)<config['epochs']:
        cap=trainer.train(datasets,max_epochs=min(200,config['epochs']-len(trainer.bits_per_example_trace)),
            patience=200,min_delta=config['min_delta'],verbose=True)
        drift=base.late_drift(trainer.bits_per_example_trace)
        extensions.append(dict(total_epochs=len(trainer.bits_per_example_trace),final_bits=cap['total_bits_memorized'],late_drift=drift))
        atomic_json(output/'continuation_progress.json',extensions)
    cap['epochs_trained']=len(trainer.bits_per_example_trace);np.savez(output/'trace.npz',**cap)
    evaluation=base.evaluate(model,*datasets,device,config['batch_size'])
    return dict(epochs=cap['epochs_trained'],final_eval=evaluation,final_bits=cap['total_bits_memorized'],
        peak_bits=float(np.max(cap['bits_trace'])*len(datasets[1])),random_learning_passed=evaluation['acc']>=.99,
        primary_epochs=primary['epochs'],primary_bits=primary['bits'],late_drift=drift,extensions=extensions)


def run(config,output,device):
    assert read(RUN/'manifest.json')['source_hashes']==source_hashes()
    registered=read(RUN/'job_registry.json');assert registered[output.name]==config
    assert config['p']==97 and config['dropout']==.2
    assert all(config[k]==v for k,v in model_spec(config['role'],config['dim']).items())
    if config['phase']=='confirmation':
        assert config['selection_sha256']==sha(RUN/'selection.json')
        pair=read(RUN/'selection.json')['pairs'][config['pair_index']]
        assert all(config[k]==v for k,v in pair[config['role']].items())
    gate=check_gate(config) if config['task']=='capacity' else None
    output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(config['seed']);datasets=base.data(config);model=base.make_model(config)
    params=count_parameters(model);assert params==parameter_count(config['role'],config['dim'])
    torch.save(dict(model_state_dict=model.state_dict(),config=config),output/'model_init.pt')
    manifest=dict(config=config,param_count=params,source_hashes=source_hashes(),gate=gate,
        initial_weights_sha256=base.tensor_digest(model.state_dict().values()),dataset_sha256=base.tensor_digest(datasets),
        started_at=time.time(),device=device,pid=os.getpid(),host=platform.node(),torch=str(torch.__version__),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    atomic_json(output/'manifest.json',manifest);model=model.to(device)
    opt=build_adamw(model,lr=config['lr'],betas=(config['beta1'],config['beta2']),weight_decay=config['weight_decay'],mup_base=0,dim=config['dim'])
    if config['task']=='arithmetic':
        trace=base.train_arithmetic(model,opt,datasets,config,device,output/'trace.jsonl');metrics=base.metrics(trace)
        result=dict(epochs=len(trace),final=trace[-1],metrics=metrics,
            learning_gate_passed=bool(metrics['val_99_sustained_epoch'] is not None and trace[-1]['val_acc']>=.99 and trace[-1]['train_eval_acc']>=.99))
    else:result=train_capacity(model,opt,datasets,config,device,output)
    result.update(task=config['task'],config=config,param_count=params,boundary_audit=base.boundary_audit(model,datasets[0],config,device))
    torch.save(dict(model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},config=config),output/'model.pt')
    result.update(checkpoint_sha256=sha(output/'model.pt'),finished_at=time.time());atomic_json(output/'summary.json',result)
    print(json.dumps(result),flush=True)


def audit_job(path,device='cpu'):
    path=Path(path);s,m=checked_artifact(path);c=s['config']
    torch.manual_seed(c['seed']);datasets=base.data(c);model=base.make_model(c)
    assert base.tensor_digest(datasets)==m['dataset_sha256']
    assert base.tensor_digest(model.state_dict().values())==m['initial_weights_sha256']
    initial=torch.load(path/'model_init.pt',map_location='cpu',weights_only=True)
    assert initial['config']==c and base.tensor_digest(initial['model_state_dict'].values())==m['initial_weights_sha256']
    saved=torch.load(path/'model.pt',map_location='cpu',weights_only=True);assert saved['config']==c
    model.load_state_dict(saved['model_state_dict'],strict=True);assert count_parameters(model)==s['param_count']==m['param_count']
    drops=[x for x in model.modules() if isinstance(x,torch.nn.Dropout)];assert drops and all(x.p==.2 for x in drops)
    result=dict(name=path.name,path=str(path.resolve()),initial_verified=True,data_verified=True,checkpoint_sha256=s['checkpoint_sha256'])
    model=model.to(device);train=base.evaluate(model,*datasets[:2],device,512)
    val=base.evaluate(model,*datasets[2:],device,512) if c['task']=='arithmetic' else None
    target=s['final']['train_eval_acc'] if val else s['final_eval']['acc']
    if train['acc']!=target or (val and val['acc']!=s['final']['val_acc']):
        result.update(cpu_train=train,cpu_val=val,cpu_gpu_discrepancy=True)
        device='cuda:1';model=model.to(device);train=base.evaluate(model,*datasets[:2],device,512)
        if val:val=base.evaluate(model,*datasets[2:],device,512)
    assert train['acc']==target
    if val:
        assert val['acc']==s['final']['val_acc']
        trace=[json.loads(line) for line in (path/'trace.jsonl').read_text().splitlines()]
        assert len(trace)==s['epochs'] and trace[-1]==s['final']
        assert all(r['epoch']==i+1 and r['step']==10*(i+1) for i,r in enumerate(trace))
        assert independent_metrics(trace)==s.get('metrics',base.metrics(trace))
    else:
        arr=np.load(path/'trace.npz',allow_pickle=False)
        assert len(arr['bits_trace'])==s['epochs']
        calculated=len(datasets[1])*(math.log2(c['p']+2)-train['loss']/math.log(2))
        assert abs(calculated-s['final_bits'])<max(.1,2e-6*abs(s['final_bits']))
        result.update(capacity_recomputed_bits=calculated,capacity_error_bits=abs(calculated-s['final_bits']))
    result.update(train_eval=train,val_eval=val,verification_device=device,boundary=base.boundary_audit(model,datasets[0],c,device),all_passed=True)
    result['input_hashes']={p.name:sha(p) for p in sorted(path.iterdir()) if p.is_file() and p.name in
        ['manifest.json','summary.json','model.pt','model_init.pt','trace.jsonl','trace.npz','primary_trace.npz','model_primary.pt']}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--device',required=True)
    a=p.parse_args();torch.set_num_threads(2);run(read(a.config),Path(a.output),a.device)


if __name__=='__main__':main()
