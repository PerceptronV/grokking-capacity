"""Nested dataset-size sweep, with fixed update budgets and evaluation cadence.

Prior campaign sources are imported without modification. No scientific source
may change after registration; new campaigns never overwrite existing runs.
"""
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
import torch.nn.functional as F

from . import multilevel_dropout as base
from . import capacity_scaling as previous
from ..data import grokking_data_torch
from ..models.transformer import count_parameters
from ..utils.optim import build_adamw

ROOT = previous.ROOT
RUN = ROOT / 'results/discrete/p97-dataset-scaling-v1'
SOURCES = previous.SOURCES + [
    'src/grokking_capacity/experiments/dataset_scaling.py',
    'scripts/run_p97_dataset_scaling.py',
    'configs/discrete_p97_dataset_scaling.yaml',
    'scripts/report_p97_dataset_scaling.py',
]
read, sha, atomic_json = previous.read, base.sha, base.atomic_json


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCES}


def logarithmic_sizes(low, high, count):
    if not 0 < low < high or count < 2:
        raise ValueError('Require positive increasing endpoints and at least two sizes')
    sizes = np.rint(np.geomspace(low, high, count)).astype(int).tolist()
    if len(set(sizes)) != count:
        raise ValueError('Rounded logarithmic sizes must be distinct')
    return sizes


def select_pairs(constants, sizes, candidates=range(2,162,2), tolerance=.1):
    """Fix dataset sizes first, then round inverse family-capacity predictions."""
    pairs=[]
    candidates=list(candidates)
    if not candidates or any(w<2 or w%2 for w in candidates):
        raise ValueError('Positive even model widths required')
    for i,n in enumerate(sizes):
        target=n*math.log2(99)
        pair=dict(index=i,n_train=n,dataset_bits=target)
        for role in ['C','B']:
            k=constants[role]['bits_per_parameter']
            if k<=0:
                raise ValueError('Positive empirical capacity constant required')
            width=min(candidates,key=lambda w:(abs(k*previous.parameter_count(role,w)/target-1),w))
            parameters=previous.parameter_count(role,width)
            estimate=k*parameters
            mismatch=estimate/target-1
            if abs(mismatch)>tolerance:
                raise ValueError(f'{role} N={n}: nearest width misses target by {mismatch:.2%}')
            pair[role]=previous.model_spec(role,width)
            pair[role+'_parameters']=parameters
            pair[role+'_estimated_bits']=estimate
            pair[role+'_dataset_mismatch']=mismatch
            lo,hi=constants[role]['parameter_range']
            pair[role+'_calibration_extrapolation']=not lo<=parameters<=hi
        pair['predicted_relative_mismatch']=pair['B_estimated_bits']/pair['C_estimated_bits']-1
        if abs(pair['predicted_relative_mismatch'])>tolerance:
            raise ValueError(f'N={n}: models miss each other by more than {tolerance:.0%}')
        pairs.append(pair)
    return pairs


def data(config):
    """One seed-specific permutation: nested train prefixes, fixed test suffix."""
    p, n, pool = config['p'], config['n_train'], config['train_pool_size']
    if not 0 < n <= pool < p * (p - 1):
        raise ValueError('Training subset must fit the pool and leave held-out data')
    state = np.random.get_state()
    try:
        np.random.seed(config['seed'])
        X, T, _, _ = grokking_data_torch(p, op='/', train_fraction=1., device='cpu')
    finally:
        np.random.set_state(state)
    return X[:n], T[:n], X[pool:], T[pool:]


def joint_stop(trace):
    return len(trace) >= 3 and all(r['val_acc'] >= .99 for r in trace[-3:]) and trace[-1]['train_online_acc'] >= .99


def trace_metrics(trace):
    result = {}
    for key in ['train_online_acc', 'train_eval_acc', 'val_acc']:
        for percent in [95, 98, 99]:
            result[f'{key}_{percent}_updates'] = next((r['step'] for r in trace if r[key] >= percent / 100), None)
    result['val_99_sustained_updates'] = next((trace[i-2]['step'] for i in range(2, len(trace))
        if all(r['val_acc'] >= .99 for r in trace[i-2:i+1])), None)
    for label, key in [('paper', 'train_online_acc'), ('eval', 'train_eval_acc')]:
        train, val = result[f'{key}_99_updates'], result['val_acc_98_updates']
        result[f'{label}_delay_updates'] = None if train is None or val is None else max(0, val-train)
    return result


def train(model, optimizer, datasets, config, device, trace_path):
    X, T, Xv, Tv = datasets
    n, bs, interval, limit = len(T), config['batch_size'], config['eval_every'], config['max_updates']
    if min(bs, interval, limit) <= 0 or limit % interval:
        raise ValueError('Positive batch/cadence/budget required; budget must divide by cadence')
    trace, step, examples, epoch = [], 0, 0, 0
    window_n = window_correct = 0
    window_loss = 0.
    started = time.time()
    with trace_path.open('x') as handle:
        while step < limit:
            epoch += 1
            perm = torch.randperm(n)
            Xs, Ts = X[perm], T[perm]
            model.train()
            for offset in range(0, n, bs):
                xb, tb = Xs[offset:offset+bs].to(device), Ts[offset:offset+bs].to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = F.cross_entropy(logits, tb)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                optimizer.step()
                step += 1
                examples += len(tb)
                window_n += len(tb)
                window_loss += loss.item() * len(tb)
                window_correct += (logits.argmax(-1) == tb).sum().item()
                if step % interval == 0:
                    val = base.evaluate(model, Xv, Tv, device, bs)
                    tr = base.evaluate(model, X, T, device, bs)
                    row = dict(step=step, epoch=epoch, examples=examples,
                        completed_epochs=epoch-1+int(offset+len(tb)==n),
                        epoch_fraction=epoch-1+(offset+len(tb))/n,
                        train_online_loss=window_loss/window_n, train_online_acc=window_correct/window_n,
                        online_window_examples=window_n, train_eval_loss=tr['loss'], train_eval_acc=tr['acc'],
                        val_loss=val['loss'], val_acc=val['acc'], wall_seconds=time.time()-started)
                    trace.append(row)
                    handle.write(json.dumps(row, allow_nan=False)+'\n')
                    handle.flush()
                    if step == interval or step % 1000 == 0:
                        print(json.dumps(row), flush=True)
                    window_n = window_correct = 0
                    window_loss = 0.
                    if joint_stop(trace):
                        return trace
                    model.train()
                if step == limit:
                    break
    return trace


def run(config, output, device):
    manifest = read(RUN/'manifest.json')
    assert manifest['source_hashes'] == source_hashes(), 'Frozen sources changed'
    assert read(RUN/'job_registry.json')[output.name] == config
    selection = read(RUN/'selection.json')
    assert config['selection_sha256'] == sha(RUN/'selection.json')
    pair = selection['pairs'][config['pair_index']]
    assert pair['n_train'] == config['n_train']
    assert all(config[k] == v for k, v in pair[config['role']].items())
    assert config['p'] == 97 and config['dropout'] == .2
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config['seed'])
    datasets = data(config)
    model = base.make_model(config)
    params = count_parameters(model)
    assert params == previous.parameter_count(config['role'], config['dim'])
    initial = base.tensor_digest(model.state_dict().values())
    torch.save(dict(model_state_dict=model.state_dict(), config=config), output/'model_init.pt')
    job_manifest = dict(config=config, source_hashes=source_hashes(), param_count=params,
        initial_weights_sha256=initial, dataset_sha256=base.tensor_digest(datasets),
        train_sha256=base.tensor_digest(datasets[:2]), test_sha256=base.tensor_digest(datasets[2:]),
        n_test=len(datasets[3]), started_at=time.time(), device=device, pid=os.getpid(),
        host=platform.node(), torch=str(torch.__version__), python=platform.python_version(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip())
    atomic_json(output/'manifest.json', job_manifest)
    model = model.to(device)
    optimizer = build_adamw(model, lr=config['lr'], betas=(config['beta1'],config['beta2']),
        weight_decay=config['weight_decay'], mup_base=0, dim=config['dim'])
    trace = train(model, optimizer, datasets, config, device, output/'trace.jsonl')
    metrics = trace_metrics(trace)
    result = dict(config=config, param_count=params, final=trace[-1], metrics=metrics,
        joint_stop_passed=joint_stop(trace),
        learning_gate_passed=metrics['val_99_sustained_updates'] is not None and
            trace[-1]['val_acc'] >= .99 and trace[-1]['train_eval_acc'] >= .99,
        boundary_audit=base.boundary_audit(model, datasets[0], config, device))
    torch.save(dict(model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}, config=config), output/'model.pt')
    result.update(checkpoint_sha256=sha(output/'model.pt'), finished_at=time.time())
    atomic_json(output/'summary.json', result)
    print(json.dumps(result), flush=True)


def audit_job(path, device):
    s, m = read(path/'summary.json'), read(path/'manifest.json')
    c = s['config']
    assert m['source_hashes'] == source_hashes() == read(RUN/'manifest.json')['source_hashes']
    assert m['config'] == c == read(RUN/'job_registry.json')[path.name]
    assert c['selection_sha256'] == sha(RUN/'selection.json')
    assert sha(path/'model.pt') == s['checkpoint_sha256']
    torch.manual_seed(c['seed'])
    datasets = data(c)
    model = base.make_model(c)
    assert base.tensor_digest(model.state_dict().values()) == m['initial_weights_sha256']
    assert base.tensor_digest(datasets) == m['dataset_sha256']
    assert len(datasets[1]) == c['n_train'] and len(datasets[3]) == m['n_test'] == 1863
    assert base.tensor_digest(datasets[:2]) == m['train_sha256']
    assert base.tensor_digest(datasets[2:]) == m['test_sha256']
    initial = torch.load(path/'model_init.pt', map_location='cpu', weights_only=True)
    assert initial['config'] == c and base.tensor_digest(initial['model_state_dict'].values()) == m['initial_weights_sha256']
    saved = torch.load(path/'model.pt', map_location='cpu', weights_only=True)
    assert saved['config'] == c
    model.load_state_dict(saved['model_state_dict'], strict=True)
    assert count_parameters(model) == s['param_count'] == m['param_count'] == previous.parameter_count(c['role'], c['dim'])
    drops = [x for x in model.modules() if isinstance(x, torch.nn.Dropout)]
    assert drops and all(x.p == .2 for x in drops)
    model = model.to(device)
    tr = base.evaluate(model, *datasets[:2], device, c['batch_size'])
    val = base.evaluate(model, *datasets[2:], device, c['batch_size'])
    assert tr['acc'] == s['final']['train_eval_acc'] and val['acc'] == s['final']['val_acc']
    assert abs(tr['loss']-s['final']['train_eval_loss']) < 1e-6
    assert abs(val['loss']-s['final']['val_loss']) < 1e-6
    trace = [json.loads(line) for line in (path/'trace.jsonl').read_text().splitlines()]
    assert trace[-1] == s['final'] and trace_metrics(trace) == s['metrics']
    per_epoch = math.ceil(c['n_train']/c['batch_size'])
    for i, row in enumerate(trace):
        step = (i+1)*c['eval_every']
        assert row['step'] == step <= c['max_updates']
        full, tail = divmod(step, per_epoch)
        assert row['examples'] == full*c['n_train']+min(tail*c['batch_size'], c['n_train'])
        assert row['epoch'] == math.ceil(step/per_epoch)
        assert row['completed_epochs'] == full
        assert row['epoch_fraction'] == full+min(tail*c['batch_size'],c['n_train'])/c['n_train']
        assert row['online_window_examples'] == row['examples']-(trace[i-1]['examples'] if i else 0)
        if i < len(trace)-1:
            assert not joint_stop(trace[:i+1]), 'Continued beyond first stopping point'
    assert joint_stop(trace) == s['joint_stop_passed']
    assert joint_stop(trace) or trace[-1]['step'] == c['max_updates']
    assert s['learning_gate_passed'] == (s['metrics']['val_99_sustained_updates'] is not None and tr['acc']>=.99 and val['acc']>=.99)
    return dict(name=path.name, all_passed=True, verification_device=device, train_eval=tr, test_eval=val,
        boundary=base.boundary_audit(model, datasets[0], c, device),
        input_hashes={p.name:sha(p) for p in path.iterdir() if p.is_file()}, audited_at=time.time())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(read(args.config), Path(args.output), args.device)


if __name__ == '__main__':
    main()
