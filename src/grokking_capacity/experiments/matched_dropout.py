"""Common, original-optimizer protocol for capacity-matched dropout A/B/C.

Training rules match the archived protocol. Capacity updates use CapacityTrainer;
arithmetic additionally logs end-of-epoch eval-mode training accuracy without
consuming RNG, while preserving online training accuracy as the primary metric.
"""
from .registered import protocol_sources, verify_sources
import argparse
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from ..models.discrete import make_model
from ..models.transformer import count_parameters
from ..utils.optim import build_adamw
from .capacity import CapacityTrainer
from .arithmetic import atomic_json, data, evaluate, first, metrics, sha, tensor_digest, train_arithmetic

ROOT = Path(__file__).resolve().parents[3]
SOURCES = protocol_sources("matched_dropout")


def source_hashes():
    return {name:sha(ROOT/name) for name in SOURCES}


def gate_check(path, config):
    if not path:raise ValueError('Capacity training requires a successful same-recipe arithmetic gate')
    path=Path(path);result=json.loads((path/'summary.json').read_text());manifest=json.loads((path/'manifest.json').read_text())
    assert result['task']=='arithmetic' and result['learning_gate_passed']
    verify_sources(manifest['source_hashes'])
    keys=('model','dim','mlp_dim','levels','quantizer_clip','quantizer_surrogate','p','dropout',
          'depth','heads','lr','weight_decay','beta1','beta2','batch_size')
    assert all(result['config'].get(k)==config.get(k) for k in keys),'gate recipe mismatch'
    with (path/'trace.jsonl').open() as f:trace=[json.loads(line) for line in f]
    assert metrics(trace)['val_99_sustained_epoch'] is not None
    assert trace[-1]['val_acc']>=.99 and trace[-1]['train_eval_acc']>=.99
    assert sha(path/'model.pt')==result['checkpoint_sha256']
    return dict(path=str(path.resolve()),summary_sha256=sha(path/'summary.json'))


def boundary_audit(model, tokens, config, device):
    if config['model']=='baseline':
        assert not hasattr(model,'quantizers') and not hasattr(model,'channels')
        return dict(kind='unrestricted_continuous')
    assert config['model']=='quantized' and config['levels']==2
    captured=[]
    def hook(module,args,output):
        assert output.shape[-1]==config['dim']
        assert torch.all((output==-1)|(output==1))
        captured.append(tuple(output.shape))
    hooks=[q.register_forward_hook(hook) for q in model.quantizers]
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
    return dict(kind='full_width_binary',train_and_eval_verified=True,shapes=captured)


def run(config, output, device):
    expected=json.loads((output.parent.parent/'manifest.json').read_text())['source_hashes']
    assert expected==source_hashes(),'frozen scientific sources changed'
    assert config['model'] in ('baseline','quantized')
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
        np.savez(output/'trace.npz',**cap)
        evaluation=evaluate(model,*datasets,device,config['batch_size'])
        result=dict(task='capacity',config=config,param_count=params,epochs=cap['epochs_trained'],
                    final_eval=evaluation,final_bits=cap['total_bits_memorized'],
                    peak_bits=float(np.max(cap['bits_trace'])*len(datasets[1])),
                    random_learning_passed=evaluation['acc']>=.99)
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
