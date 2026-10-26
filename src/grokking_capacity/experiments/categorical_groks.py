"""C-first categorical experiment; completed earlier experiment sources stay frozen."""
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
from .capacity import CapacityTrainer
from ..models.transformer import count_parameters
from ..utils.optim import build_adamw

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'results/discrete/p97-categorical-groks-v1'
SOURCES=base.SOURCES+[
    'src/grokking_capacity/experiments/categorical_groks.py',
    'scripts/run_p97_categorical_groks.py','configs/discrete_p97_categorical_groks.yaml']
sha=base.sha
atomic_json=base.atomic_json


def source_hashes():return {name:sha(ROOT/name) for name in SOURCES}
def read(path):return json.loads(Path(path).read_text())


def models(width,alphabet=8):
    if width%3:raise ValueError('Width must allow exact parameter reallocation')
    b=dict(model='categorical',dim=width,mlp_dim=4*width,alphabet=alphabet,
        categorical_temperature=1.,categorical_normalization='slot_layernorm',categorical_training_sampling=False)
    c=dict(model='baseline',dim=width,mlp_dim=4*width+2*alphabet*width//3)
    return c,b


def datasets(config):
    c=dict(config)
    if c['task']=='memorisation':c['task']='capacity'
    return base.data(c)


def qualify(results,seeds):
    assert sorted(r['config']['seed'] for r in results)==sorted(seeds)
    values=[r['metrics']['paper_delay_epochs'] for r in results]
    learned=[bool(r['learning_gate_passed']) for r in results]
    return dict(passed=bool(all(learned) and all(v is not None and v>0 for v in values)),
        seeds=list(seeds),delays_epochs=values,min_observed_delay_epochs=min((v for v in values if v is not None),default=None),
        positive=sum(v is not None and v>0 for v in values),zero=values.count(0),undefined=values.count(None),
        learning_successes=sum(learned))


def train_memorisation(model,optimizer,data,config,device,trace_path=None):
    trainer=CapacityTrainer(model,optimizer,config['p']+2,config['batch_size'],device)
    n=len(data[1]);trace=[];started=time.time();handle=trace_path.open('x') if trace_path else None
    try:
        for epoch in range(1,config['epochs']+1):
            result=trainer.train(data,max_epochs=1,patience=1,min_delta=0.,verbose=False)
            row=dict(epoch=epoch,step=epoch*math.ceil(n/config['batch_size']),
                train_online_acc=float(trainer.train_acc_trace[-1]),
                train_online_loss=float(trainer.train_loss_trace[-1]),
                train_eval_loss=float(result['final_loss']),bits=float(result['total_bits_memorized']),
                wall_seconds=time.time()-started)
            trace.append(row)
            if handle:handle.write(json.dumps(row)+'\n');handle.flush()
            if epoch==1 or epoch%100==0:print(json.dumps(row),flush=True)
            if row['train_online_acc']>=.99:break
    finally:
        if handle:handle.close()
    return trace


def check_gate(config):
    path=Path(config['gate_path']);s=read(path/'summary.json');m=read(path/'manifest.json')
    assert s['task']=='arithmetic' and s['learning_gate_passed']
    assert m['source_hashes']==source_hashes() and sha(path/'model.pt')==s['checkpoint_sha256']
    keys=['model','dim','mlp_dim','p','depth','heads','dropout','lr','weight_decay','beta1','beta2','batch_size',
          'alphabet','categorical_temperature','categorical_normalization','categorical_training_sampling']
    assert all(config.get(k)==s['config'].get(k) for k in keys)
    return dict(path=str(path.resolve()),summary_sha256=sha(path/'summary.json'))


def run(config,output,device):
    assert read(RUN/'manifest.json')['source_hashes']==source_hashes()
    assert config['p']==97 and config['dropout']==.2
    gate=check_gate(config) if config['task']=='memorisation' else None
    output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(config['seed']);data=datasets(config);model=base.make_model(config)
    assert len(data[1])==4656
    if config['role']=='B':
        s=read(RUN/'selection.json');assert config['selection_sha256']==sha(RUN/'selection.json')
        assert all(config[k]==v for k,v in s['B'].items())
    initial=base.tensor_digest(model.state_dict().values());params=count_parameters(model)
    torch.save(dict(model_state_dict=model.state_dict(),config=config),output/'model_init.pt')
    manifest=dict(config=config,param_count=params,source_hashes=source_hashes(),gate=gate,
        initial_weights_sha256=initial,dataset_sha256=base.tensor_digest(data),started_at=time.time(),
        device=device,pid=os.getpid(),host=platform.node(),torch=str(torch.__version__),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    atomic_json(output/'manifest.json',manifest)
    model=model.to(device);opt=build_adamw(model,lr=config['lr'],betas=(config['beta1'],config['beta2']),
        weight_decay=config['weight_decay'],mup_base=0,dim=config['dim'])
    if config['task']=='arithmetic':
        trace=base.train_arithmetic(model,opt,data,config,device,output/'trace.jsonl')
        m=base.metrics(trace)
        result=dict(metrics=m,learning_gate_passed=bool(m['val_99_sustained_epoch'] is not None
            and trace[-1]['val_acc']>=.99 and trace[-1]['train_eval_acc']>=.99))
    else:
        trace=train_memorisation(model,opt,data,config,device,output/'trace.jsonl')
        hit=base.first(trace,'train_online_acc',.99)
        result=dict(memorisation_epoch=hit,memorisation_updates=10*hit if hit else None,
            reached=hit is not None,censor_updates=trace[-1]['step'] if hit is None else None,
            final_eval=base.evaluate(model,*data,device,config['batch_size']))
    result.update(task=config['task'],config=config,param_count=params,epochs=len(trace),final=trace[-1],
        boundary_audit=base.boundary_audit(model,data[0],config,device))
    torch.save(dict(model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},config=config),output/'model.pt')
    result.update(checkpoint_sha256=sha(output/'model.pt'),finished_at=time.time())
    atomic_json(output/'summary.json',result);print(json.dumps(result),flush=True)


def independent_metrics(trace):
    def first(key,threshold):
        positions=np.flatnonzero(np.array([r[key] for r in trace])>=threshold)
        return int(positions[0])+1 if len(positions) else None
    result={}
    for key in ['train_online_acc','train_eval_acc','val_acc']:
        for threshold in [95,98,99]:result[f'{key}_{threshold}_epoch']=first(key,threshold/100)
    for threshold in [95,99]:
        arr=np.array([r['val_acc']>=threshold/100 for r in trace])
        windows=np.lib.stride_tricks.sliding_window_view(arr,3) if len(arr)>=3 else np.empty((0,3),bool)
        hit=np.flatnonzero(windows.all(axis=1));result[f'val_{threshold}_sustained_epoch']=int(hit[0])+1 if len(hit) else None
    for label,key in [('paper','train_online_acc'),('eval','train_eval_acc')]:
        a,b=result[f'{key}_99_epoch'],result['val_acc_98_epoch']
        result[f'{label}_delay_epochs']=None if a is None or b is None else max(0,b-a)
    return result


def audit_job(path,device='cpu'):
    s=read(path/'summary.json');m=read(path/'manifest.json');c=s['config']
    assert m['config']==c and m['source_hashes']==source_hashes()
    assert sha(path/'model.pt')==s['checkpoint_sha256']
    torch.manual_seed(c['seed']);data=datasets(c);model=base.make_model(c)
    assert base.tensor_digest(data)==m['dataset_sha256']
    assert base.tensor_digest(model.state_dict().values())==m['initial_weights_sha256']
    initial=torch.load(path/'model_init.pt',map_location='cpu',weights_only=True)
    assert initial['config']==c and base.tensor_digest(initial['model_state_dict'].values())==m['initial_weights_sha256']
    saved=torch.load(path/'model.pt',map_location='cpu',weights_only=True);assert saved['config']==c
    model.load_state_dict(saved['model_state_dict'],strict=True);assert count_parameters(model)==s['param_count']==m['param_count']
    drop=[x for x in model.modules() if isinstance(x,torch.nn.Dropout)];assert drop and all(x.p==.2 for x in drop)
    trace=[json.loads(line) for line in (path/'trace.jsonl').read_text().splitlines()]
    assert len(trace)==s['epochs'] and trace[-1]==s['final']
    assert all(r['epoch']==i+1 and r['step']==10*(i+1) for i,r in enumerate(trace))
    result=dict(name=path.name,checkpoint_sha256=s['checkpoint_sha256'],initial_verified=True,data_verified=True)
    model=model.to(device);train=base.evaluate(model,*data[:2],device,512)
    target=s['final']['train_eval_acc'] if c['task']=='arithmetic' else s['final_eval']['acc']
    val=base.evaluate(model,*data[2:],device,512) if c['task']=='arithmetic' else None
    if train['acc']!=target or (val and val['acc']!=s['final']['val_acc']):
        result.update(cpu_train=train,cpu_val=val,cpu_gpu_discrepancy=True)
        device='cuda:0';model=model.to(device);train=base.evaluate(model,*data[:2],device,512)
        if val:val=base.evaluate(model,*data[2:],device,512)
    assert train['acc']==target
    if val:assert val['acc']==s['final']['val_acc']
    if c['task']=='arithmetic':assert independent_metrics(trace)==s['metrics']
    else:
        hits=np.flatnonzero(np.array([r['train_online_acc'] for r in trace])>=.99)
        hit=int(hits[0])+1 if len(hits) else None
        assert hit==s['memorisation_epoch']
        assert (hit==len(trace)) if hit else (len(trace)==c['epochs'])
        check_gate(c)
        assert read(Path(c['gate_path'])/'summary.json')['finished_at']<=m['started_at']
    result.update(train_eval=train,val_eval=val,verification_device=device,
        boundary=base.boundary_audit(model,data[0],c,device),all_passed=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--device',required=True)
    a=p.parse_args();torch.set_num_threads(2);run(read(a.config),Path(a.output),a.device)


if __name__=='__main__':main()
