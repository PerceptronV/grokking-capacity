"""Eight-level and categorical capacity-matched dropout A/B/C workers.

Existing experiment sources are unchanged. Capacity updates use CapacityTrainer;
arithmetic additionally logs end-of-epoch eval-mode training accuracy without
consuming RNG, while preserving online training accuracy as the primary metric.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..data import grokking_data_torch, random_target_data_torch
from ..models.discrete import make_model as original_make_model
from ..models.fullwidth_discrete import FullWidthCategoricalTransformer
from ..models.transformer import count_parameters
from ..utils.optim import build_adamw
from .capacity import CapacityTrainer

ROOT = Path(__file__).resolve().parents[3]
SOURCES = ['src/grokking_capacity/experiments/matched_dropout.py',
           'src/grokking_capacity/experiments/capacity.py',
           'src/grokking_capacity/experiments/groks.py',
           'src/grokking_capacity/models/discrete.py',
           'src/grokking_capacity/models/fullwidth_discrete.py',
           'src/grokking_capacity/models/transformer.py',
           'src/grokking_capacity/data/modular.py',
           'src/grokking_capacity/utils/optim.py',
           'scripts/run_p97_dropout_match.py',
           'src/grokking_capacity/experiments/multilevel_dropout.py',
           'scripts/run_p97_multilevel_stage.py',
           'configs/discrete_p97_multilevel_dropout.yaml']


class ModeCategoricalTransformer(FullWidthCategoricalTransformer):
    """Explicit train sampling policy; evaluation always uses hard argmax."""
    def __init__(self, *, training_sampling=False, **kwargs):
        super().__init__(**kwargs)
        self.training_sampling = training_sampling

    def forward(self, tokens, *, sample=None):
        if sample is None:
            sample = self.training and self.training_sampling
        return super().forward(tokens, sample=sample)


def make_model(config):
    if config['model'] != 'categorical':
        return original_make_model(config)
    assert 'latent_dim' not in config
    return ModeCategoricalTransformer(
        depth=config['depth'], dim=config['dim'], heads=config['heads'],
        n_tokens=config['p']+2, seq_len=4, dropout=config['dropout'],
        mlp_dim=config['mlp_dim'], alphabet=config['alphabet'],
        categorical_temperature=config['categorical_temperature'],
        categorical_normalization=config['categorical_normalization'],
        training_sampling=config['categorical_training_sampling'])


def continuous_from_b(config):
    """Ordinary continuous C: same width/depth, parameter count within 0.1%."""
    result = {k:config[k] for k in ('dim','mlp_dim')}
    result['model'] = 'baseline'
    if config['model'] == 'categorical':
        # Per block B adds two d x (d*A) matrices. Reallocate those parameters
        # to the three FF matrices in an ordinary C; no narrow adapter remains.
        result['mlp_dim'] += round(2*config['alphabet']*config['dim']/3)
    return result


def late_drift(bits):
    if len(bits)<200 or float(np.mean(bits[-200:-100]))<=0:
        return None
    return float(abs(np.mean(bits[-100:])/np.mean(bits[-200:-100])-1))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def source_hashes():
    return {name:sha(ROOT/name) for name in SOURCES}


def tensor_digest(values):
    h=hashlib.sha256()
    for value in values:
        h.update(str(tuple(value.shape)).encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def data(config):
    np.random.seed(config['seed'])
    if config['task']=='arithmetic':
        return grokking_data_torch(config['p'],op='/',split_type='random',train_fraction=.5,device='cpu')
    if config['task']=='capacity':
        return random_target_data_torch(config['n_samples'],config['p'],seq_len=4,device='cpu')
    raise ValueError(config['task'])


@torch.no_grad()
def evaluate(model, X, T, device, batch_size=512):
    model.eval();loss=0.;correct=0
    for start in range(0,len(T),batch_size):
        xb,tb=X[start:start+batch_size].to(device),T[start:start+batch_size].to(device)
        logits=model(xb)
        loss+=F.cross_entropy(logits,tb,reduction='sum').item()
        correct+=(logits.argmax(-1)==tb).sum().item()
    return dict(loss=loss/len(T),acc=correct/len(T))


def first(trace,key,threshold,count=1):
    for i in range(count-1,len(trace)):
        if all(r[key]>=threshold for r in trace[i-count+1:i+1]):
            return trace[i-count+1]['epoch']
    return None


def metrics(trace):
    result={}
    for key in ('train_online_acc','train_eval_acc','val_acc'):
        for percent in (95,98,99):
            result[f'{key}_{percent}_epoch']=first(trace,key,percent/100)
    for percent in (95,99):
        result[f'val_{percent}_sustained_epoch']=first(trace,'val_acc',percent/100,3)
    val=result['val_acc_98_epoch']
    for label,key in [('paper','train_online_acc'),('eval','train_eval_acc')]:
        train=result[f'{key}_99_epoch']
        result[f'{label}_delay_epochs']=None if train is None or val is None else max(0,val-train)
    return result


def train_arithmetic(model, optimizer, datasets, config, device, trace_path=None):
    X,T,Xv,Tv=datasets;n=len(T);bs=config['batch_size'];trace=[];start_time=time.time()
    handle=trace_path.open('x') if trace_path else None
    try:
        for epoch in range(1,config['epochs']+1):
            model.train();perm=torch.randperm(n);Xs,Ts=X[perm],T[perm]
            total_loss=0.;correct=0
            for start in range(0,n,bs):
                xb,tb=Xs[start:start+bs].to(device),Ts[start:start+bs].to(device)
                optimizer.zero_grad();logits=model(xb);loss=F.cross_entropy(logits,tb)
                loss.backward();optimizer.step()
                total_loss+=loss.item()*len(tb);correct+=(logits.argmax(-1)==tb).sum().item()
            val=evaluate(model,Xv,Tv,device,bs)
            train=evaluate(model,X,T,device,bs)
            row=dict(epoch=epoch,step=epoch*math.ceil(n/bs),examples=epoch*n,
                     train_online_loss=total_loss/n,train_online_acc=correct/n,
                     train_eval_loss=train['loss'],train_eval_acc=train['acc'],
                     val_loss=val['loss'],val_acc=val['acc'],wall_seconds=time.time()-start_time)
            trace.append(row)
            if handle:handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush()
            if epoch==1 or epoch%100==0:
                print(json.dumps(row),flush=True)
            held_success=len(trace)>=3 and all(r['val_acc']>=.99 for r in trace[-3:])
            rule=config.get('stop_rule','joint99')
            if rule=='learning_gate' and held_success and train['acc']>=.99:break
            if rule=='joint99' and held_success and row['train_online_acc']>=.99:break
            if rule not in ('learning_gate','joint99','none'):raise ValueError(rule)
    finally:
        if handle:handle.close()
    return trace


def gate_check(path, config):
    if not path:raise ValueError('Capacity training requires a successful same-recipe arithmetic gate')
    path=Path(path);result=json.loads((path/'summary.json').read_text());manifest=json.loads((path/'manifest.json').read_text())
    assert result['task']=='arithmetic' and result['learning_gate_passed']
    assert manifest['source_hashes']==source_hashes()
    keys=('model','dim','mlp_dim','levels','quantizer_clip','quantizer_surrogate','p','dropout',
          'depth','heads','lr','weight_decay','beta1','beta2','batch_size',
          'alphabet','categorical_temperature','categorical_normalization','categorical_training_sampling')
    assert all(result['config'].get(k)==config.get(k) for k in keys),'gate recipe mismatch'
    with (path/'trace.jsonl').open() as f:trace=[json.loads(line) for line in f]
    assert metrics(trace)['val_99_sustained_epoch'] is not None
    assert trace[-1]['val_acc']>=.99 and trace[-1]['train_eval_acc']>=.99
    assert sha(path/'model.pt')==result['checkpoint_sha256']
    return dict(path=str(path.resolve()),summary_sha256=sha(path/'summary.json'))


def boundary_audit(model, tokens, config, device):
    if config['model']=='baseline':
        assert not hasattr(model,'quantizers') and not hasattr(model,'channels')
        assert not hasattr(model,'categorical_states')
        return dict(kind='unrestricted_continuous')
    captured=[]
    def hook(module,args,output):
        if config['model']=='quantized':
            assert output.shape[-1]==config['dim']
            grid=torch.arange(config['levels'],device=output.device,dtype=output.dtype)
            grid=grid*(2*config['quantizer_clip']/(config['levels']-1))-config['quantizer_clip']
            assert (output[...,None]==grid).any(-1).all()
        else:
            assert output.shape[-2:]==(config['dim'],config['alphabet'])
            assert ((output==0)|(output==1)).all() and (output.sum(-1)==1).all()
        captured.append(tuple(output.shape))
    modules=model.quantizers if config['model']=='quantized' else model.categorical_states
    hooks=[m.register_forward_hook(hook) for m in modules]
    was_training=model.training
    try:
        devices=[torch.device(device).index or 0] if str(device).startswith('cuda') else []
        with torch.random.fork_rng(devices=devices),torch.no_grad():
            for training in (False,True):
                model.train(training);model(tokens[:32].to(device))
        assert len(captured)==2*config['depth']
    finally:
        for handle in hooks:handle.remove()
        model.train(was_training)
    return dict(kind='full_width_'+config['model'],train_and_eval_verified=True,shapes=captured)


def run(config, output, device):
    expected=json.loads((output.parent.parent/'manifest.json').read_text())['source_hashes']
    assert expected==source_hashes(),'frozen scientific sources changed'
    assert config['model'] in ('baseline','quantized','categorical')
    assert config['dropout']==.2 and config['p']==97 and config['weight_decay']==1.
    gate=gate_check(config.get('gate_path'),config) if config['task']=='capacity' else None
    if config['task']=='capacity' and config['n_samples']>2048:
        random_path=Path(config['random_gate_path'])
        random_result=json.loads((random_path/'summary.json').read_text())
        random_manifest=json.loads((random_path/'manifest.json').read_text())
        assert random_result['task']=='capacity' and random_result['config']['n_samples']==2048
        assert random_result['random_learning_passed'] and random_result['final_eval']['acc']>=.99
        assert random_manifest['source_hashes']==expected
        excluded={'seed','n_samples','phase','role','gate_path','random_gate_path'}
        assert {k:v for k,v in random_result['config'].items() if k not in excluded}=={
            k:v for k,v in config.items() if k not in excluded},'random gate recipe mismatch'
        assert sha(random_path/'model.pt')==random_result['checkpoint_sha256']
        gate['random_path']=str(random_path.resolve())
        gate['random_summary_sha256']=sha(random_path/'summary.json')
    output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(config['seed']);datasets=data(config)
    model=make_model(config);params=count_parameters(model)
    initial=tensor_digest(model.state_dict().values())
    torch.save({'model_state_dict':model.state_dict(),'config':config},output/'model_init.pt')
    manifest=dict(config=config,param_count=params,source_hashes=expected,gate=gate,
                  dataset_sha256=tensor_digest(datasets),initial_weights_sha256=initial,
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  host=platform.node(),torch=str(torch.__version__),python=platform.python_version(),
                  device=str(device),pid=__import__('os').getpid(),started_at=time.time())
    atomic_json(output/'manifest.json',manifest)
    model=model.to(device)
    optimizer=build_adamw(model,lr=config['lr'],betas=(config['beta1'],config['beta2']),
                          weight_decay=config['weight_decay'],mup_base=0,dim=config['dim'])
    if config['task']=='arithmetic':
        trace=train_arithmetic(model,optimizer,datasets,config,device,output/'trace.jsonl')
        result=dict(task='arithmetic',config=config,param_count=params,epochs=len(trace),
                    final=trace[-1],metrics=metrics(trace),
                    learning_gate_passed=(first(trace,'val_acc',.99,3) is not None
                                          and trace[-1]['val_acc']>=.99 and trace[-1]['train_eval_acc']>=.99))
    else:
        trainer=CapacityTrainer(model,optimizer,config['p']+2,config['batch_size'],device)
        cap=trainer.train(datasets,max_epochs=config['epochs'],patience=config['patience'],
                          min_delta=config['min_delta'],verbose=True)
        primary_epochs=cap['epochs_trained']
        primary_bits=cap['total_bits_memorized']
        np.savez(output/'primary_trace.npz',**cap)
        torch.save({'model_state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'config':config},output/'model_primary.pt')
        extensions=[]
        drift=late_drift(trainer.bits_per_example_trace)
        # Continue the exact live optimizer/RNG if the patience stop is drifting.
        # No primary result is overwritten; every segment remains in the trace.
        while (drift is None or drift>.01) and len(trainer.bits_per_example_trace)<config['epochs']:
            remaining=config['epochs']-len(trainer.bits_per_example_trace)
            cap=trainer.train(datasets,max_epochs=min(200,remaining),patience=200,
                              min_delta=config['min_delta'],verbose=True)
            drift=late_drift(trainer.bits_per_example_trace)
            extensions.append(dict(total_epochs=len(trainer.bits_per_example_trace),
                                   final_bits=cap['total_bits_memorized'],late_drift=drift))
            atomic_json(output/'continuation_progress.json',extensions)
        cap['epochs_trained']=len(trainer.bits_per_example_trace)
        np.savez(output/'trace.npz',**cap)
        evaluation=evaluate(model,*datasets,device,config['batch_size'])
        result=dict(task='capacity',config=config,param_count=params,epochs=cap['epochs_trained'],
                    final_eval=evaluation,final_bits=cap['total_bits_memorized'],
                    peak_bits=float(np.max(cap['bits_trace'])*len(datasets[1])),
                    random_learning_passed=evaluation['acc']>=.99,
                    primary_epochs=primary_epochs,primary_bits=primary_bits,
                    late_drift=drift,extensions=extensions)
    result['boundary_audit']=boundary_audit(model,datasets[0],config,device)
    torch.save({'model_state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'config':config,'param_count':params},output/'model.pt')
    result['checkpoint_sha256']=sha(output/'model.pt');result['finished_at']=time.time()
    atomic_json(output/'summary.json',result)
    print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--device',required=True)
    args=parser.parse_args();run(json.loads(Path(args.config).read_text()),Path(args.output),args.device)


if __name__=='__main__':main()
