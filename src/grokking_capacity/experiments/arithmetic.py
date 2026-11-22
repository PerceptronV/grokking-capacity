"""Shared epoch-based arithmetic protocol; optimizer and RNG order are preserved."""
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from ..data import grokking_data_torch, random_target_data_torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


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
