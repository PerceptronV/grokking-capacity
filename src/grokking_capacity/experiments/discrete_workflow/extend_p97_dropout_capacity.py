#!/usr/bin/env python3
"""Resolve a flagged capacity plateau by exact replay and continued optimization.

Preserves the primary result. Replays its original stopping rule, verifies every
weight and trace, then continues the same optimizer/RNG within 5,000 total epochs.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'src'))
from grokking_capacity.experiments.matched_dropout import (
    data, sha, source_hashes, atomic_json, tensor_digest, gate_check, evaluate, boundary_audit)
from grokking_capacity.experiments.capacity import CapacityTrainer
from grokking_capacity.models.discrete import make_model
from grokking_capacity.utils.optim import build_adamw
RUN=ROOT/'results/discrete/p97-dropout-match-v1'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();source=RUN/'jobs'/args.job
    primary=json.loads((source/'summary.json').read_text());c=primary['config'];manifest=json.loads((source/'manifest.json').read_text())
    assert c['task']=='capacity' and c['epochs']==5000
    assert manifest['source_hashes']==source_hashes()
    gate_check(c['gate_path'],c)
    assert json.loads((Path(c['random_gate_path'])/'summary.json').read_text())['random_learning_passed']
    out=RUN/'diagnostics'/args.job;out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'manifest.json',dict(config=c,primary_summary_sha256=sha(source/'summary.json'),
        primary_checkpoint_sha256=sha(source/'model.pt'),source_hashes=source_hashes(),
        extension_source_sha256=sha(Path(__file__)),started_at=time.time(),device=args.device,
        rule='Exact primary replay, then original optimizer/RNG continuation in <=200 epoch segments; minimum 200 additional epochs, stop at <=1% late drift or 5000 total epochs.'))
    (out/'extension_source.py').write_bytes(Path(__file__).read_bytes())
    torch.manual_seed(c['seed']);datasets=data(c);model=make_model(c)
    assert tensor_digest(datasets)==manifest['dataset_sha256']
    assert tensor_digest(model.state_dict().values())==manifest['initial_weights_sha256']
    model=model.to(args.device)
    optim=build_adamw(model,lr=c['lr'],betas=(c['beta1'],c['beta2']),weight_decay=c['weight_decay'],mup_base=0,dim=c['dim'])
    trainer=CapacityTrainer(model,optim,c['p']+2,c['batch_size'],args.device)
    initial=trainer.train(datasets,max_epochs=c['epochs'],patience=c['patience'],min_delta=c['min_delta'],verbose=True)
    saved=torch.load(source/'model.pt',map_location='cpu',weights_only=True)
    max_error=max(float((value.detach().cpu()-saved['model_state_dict'][key]).abs().max()) for key,value in model.state_dict().items())
    assert initial['epochs_trained']==primary['epochs']
    assert max_error==0.,f'Primary replay differs: maximum parameter error {max_error}'
    reference=np.load(source/'trace.npz')
    for key in ('train_loss_trace','train_acc_trace','bits_trace'):np.testing.assert_array_equal(initial[key],reference[key])
    atomic_json(out/'replay_verified.json',dict(epochs=primary['epochs'],max_parameter_error=max_error,
        traces_identical=True,final_bits=initial['total_bits_memorized']))
    segments=[]
    while len(trainer.bits_per_example_trace)<c['epochs']:
        cap=trainer.train(datasets,max_epochs=min(200,c['epochs']-len(trainer.bits_per_example_trace)),
                          patience=c['patience'],min_delta=c['min_delta'],verbose=True)
        total=len(trainer.bits_per_example_trace);bits=np.array(trainer.bits_per_example_trace)*c['n_samples']
        drift=float(abs(bits[-100:].mean()/bits[-200:-100].mean()-1))
        segments.append(dict(total_epochs=total,final_bits=float(cap['total_bits_memorized']),late_drift=drift))
        atomic_json(out/'progress.json',dict(segments=segments))
        print(json.dumps(segments[-1]),flush=True)
        if total-primary['epochs']>=200 and drift<=.01:break
    evaluation=evaluate(model,*datasets,args.device,c['batch_size'])
    np.savez(out/'trace.npz',bits_trace=np.array(trainer.bits_per_example_trace),
             train_loss_trace=np.array(trainer.train_loss_trace),train_acc_trace=np.array(trainer.train_acc_trace))
    boundary=boundary_audit(model,datasets[0],c,args.device)
    torch.save(dict(config=c,model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}),out/'model.pt')
    result=dict(config=c,primary_job=args.job,primary_final_bits=primary['final_bits'],
        final_bits=float(cap['total_bits_memorized']),final_eval=evaluation,epochs=total,late_drift=drift,
        converged=drift<=.01 and total<c['epochs'],replay_exact=True,segments=segments,boundary=boundary,
        checkpoint_sha256=sha(out/'model.pt'),finished_at=time.time())
    atomic_json(out/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
