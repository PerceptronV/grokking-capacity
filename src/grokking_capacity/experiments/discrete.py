"""Auditable arithmetic-first runner for the discrete-latent experiment.

PYTHONPATH=src python -m grokking_capacity.experiments.discrete \
    --config configs/discrete_learning.yaml --output results/discrete/learning-v1

This is intentionally separate from the legacy registry, whose identifying
schema cannot distinguish the channel architectures. No random task is allowed
without a verified successful discrete arithmetic run supplied as --gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from ..data.modular import grokking_data_torch
from ..models.discrete import make_model
from ..models.transformer import count_parameters

ROOT = Path(__file__).resolve().parents[3]
SOURCES = [Path(__file__).resolve(),
           ROOT / 'src/grokking_capacity/models/discrete.py',
           ROOT / 'src/grokking_capacity/models/fullwidth_discrete.py',
           ROOT / 'src/grokking_capacity/models/transformer.py',
           ROOT / 'src/grokking_capacity/data/modular.py']
DISCRETE_MODELS = {'discrete', 'quantized', 'categorical'}
BOUND_SOURCES = {p: p.read_bytes() for p in SOURCES}
DEFAULTS = dict(task='modular', p=97, operation='/', train_fraction=0.5,
                depth=2, heads=1, dropout=0.0, seed=42, lr=0.001,
                weight_decay=1.0, beta1=0.9, beta2=0.98, batch_size=512,
                max_steps=20000, eval_every=100, checkpoint_every=1000,
                sustained_evals=3, stop_on_success=True)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def source_hashes():
    return {str(p.relative_to(ROOT)): digest(BOUND_SOURCES[p]) for p in SOURCES}


def dataset(config):
    # Data randomness is independent of model initialisation and batch order.
    seed = config.get('data_seed', config['seed'])
    np.random.seed(seed)
    if config['task'] == 'modular':
        return grokking_data_torch(config['p'], config['operation'], 'random',
                                   config['train_fraction'])
    if config['task'] != 'random':
        raise ValueError('Unknown task')
    rng = np.random.default_rng(seed)
    v = config['p'] + 2
    n = config['n_samples']
    # Unique random four-token strings avoid incompatible duplicate labels.
    ids = rng.choice(v ** 4, size=n, replace=False)
    x = np.stack([(ids // v ** i) % v for i in range(4)], axis=1)
    y = rng.integers(v, size=n)
    return torch.tensor(x), torch.tensor(y), None, None


def tensor_digest(tensors):
    h = hashlib.sha256()
    for t in tensors:
        if t is not None:
            h.update(str(tuple(t.shape)).encode())
            h.update(t.numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def evaluate(model, x, y, batch_size=1024, *, categorical_mode=False):
    model.eval()
    loss, correct = torch.zeros((), device=x.device), torch.zeros((), device=x.device)
    for start in range(0, len(x), batch_size):
        xb, yb = x[start:start + batch_size], y[start:start + batch_size]
        logits = model(xb, sample=False) if categorical_mode else model(xb)
        loss += F.cross_entropy(logits, yb, reduction='sum')
        correct += (logits.argmax(-1) == yb).sum()
    return float(loss / len(x)), float(correct / len(x))


def sustained_crossing(trace, key, threshold, count=3):
    for end in range(count - 1, len(trace)):
        window = trace[end - count + 1:end + 1]
        if all(r.get(key, -1) >= threshold for r in window):
            first = window[0]
            return {k: first[k] for k in ['step', 'examples', 'epochs', 'wall_seconds']}
    return None


def primary_accuracy_keys(config):
    policy = config.get('categorical_evaluation', 'sampled')
    if policy not in {'sampled', 'hard_mode'}:
        raise ValueError('Unknown categorical evaluation policy')
    if config['model'] == 'categorical' and policy == 'hard_mode':
        return 'train_mode_acc', 'val_mode_acc'
    return 'train_acc', 'val_acc'


def summarise(config, trace, param_count):
    train_key, val_key = primary_accuracy_keys(config)
    def metrics_for(train_key, val_key):
        metrics = {}
        for label, key in [('train', train_key), ('generalise', val_key)]:
            for percent in [95, 99]:
                metrics[f'{label}_{percent}'] = sustained_crossing(
                    trace, key, percent / 100, config['sustained_evals'])
        train, gen = metrics['train_99'], metrics['generalise_99']
        metrics['delay_steps_99'] = gen['step'] - train['step'] if gen and train else None
        metrics['max_train_val_gap'] = max(
            (r[train_key] - r[val_key] for r in trace if val_key in r), default=None)
        return metrics
    metrics = metrics_for(train_key, val_key)
    result = dict(config=config, param_count=param_count, final=trace[-1],
                  best_train_bits=max(r['train_bits'] for r in trace),
                  arithmetic_gate_passed=(config['task'] == 'modular'
                      and config['model'] in DISCRETE_MODELS
                      and metrics['generalise_99'] is not None
                      and trace[-1].get(val_key, -1) >= .99),
                  metrics=metrics)
    result['primary_evaluation'] = (config.get('categorical_evaluation', 'sampled')
                                    if config['model'] == 'categorical' else 'deterministic')
    if config['model'] == 'categorical':
        result['evaluation_metrics'] = {
            'sampled': metrics_for('train_acc', 'val_acc'),
            'hard_mode': metrics_for('train_mode_acc', 'val_mode_acc')}
    return result


def check_gate(path, candidate=None):
    """Verify successful discrete arithmetic evidence before any random labels."""
    if path is None:
        raise ValueError('Random-label runs require --gate with successful discrete arithmetic evidence')
    path = Path(path)
    summary = json.loads((path / 'summary.json').read_text())
    manifest = json.loads((path / 'manifest.json').read_text())
    trace = [json.loads(x) for x in (path / 'trace.jsonl').read_text().splitlines()]
    recomputed = summarise(summary['config'], trace, summary['param_count'])
    if not recomputed['arithmetic_gate_passed'] or not summary['arithmetic_gate_passed']:
        raise ValueError('Discrete arithmetic gate has not passed')
    if candidate is not None and candidate['model'] in DISCRETE_MODELS:
        keys = ['model', 'p', 'operation', 'train_fraction', 'depth', 'heads', 'dim', 'mlp_dim',
                'latent_dim', 'levels', 'alphabet', 'categorical_temperature', 'categorical_normalization', 'categorical_evaluation',
                'quantizer_clip', 'quantizer_surrogate', 'dropout',
                'lr', 'weight_decay', 'beta1', 'beta2', 'batch_size']
        if any(candidate.get(k) != summary['config'].get(k) for k in keys):
            raise ValueError('This discrete architecture has not passed the arithmetic gate')
    if manifest['source_hashes'] != source_hashes():
        raise ValueError('Arithmetic gate source differs from current scientific code')
    return dict(path=str(path.resolve()), summary_sha256=digest((path / 'summary.json').read_bytes()))


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run(config, output, device, *, resume=False, gate=None):
    config = {**DEFAULTS, **config}
    primary_accuracy_keys(config)
    if config['max_steps'] < 1 or config['eval_every'] < 1 or config['checkpoint_every'] < 1:
        raise ValueError('Positive budgets and intervals required')
    gate_evidence = check_gate(gate, config) if config['task'] == 'random' else None
    # Fail invalid architecture settings before creating output directories.
    torch.manual_seed(config['seed'])
    model = make_model(config)
    params = count_parameters(model)
    data_cpu = dataset(config)
    data_hash = tensor_digest(data_cpu)
    scientific = dict(config=config, source_hashes=source_hashes(), dataset_sha256=data_hash,
                      gate=gate_evidence)
    existed = output.exists()
    if existed:
        manifest = json.loads((output / 'manifest.json').read_text())
        if any(manifest[k] != v for k, v in scientific.items()):
            raise ValueError(f'Existing run differs in config/source/data: {output}')
        if (output / 'summary.json').exists():
            print(f'SKIP completed {output}', flush=True)
            return
        if not resume:
            raise ValueError(f'Incomplete run exists: inspect live process before --resume: {output}')
        old = manifest['pid']
        # PID reuse must not be mistaken for a live original worker.
        proc = Path(f'/proc/{old}/stat')
        if proc.exists() and proc.read_text().split()[21] == manifest['process_start_ticks']:
            raise RuntimeError(f'Original worker {old} remains live')
    else:
        output.mkdir(parents=True)
        try:
            gpu_snapshot = subprocess.check_output(['nvidia-smi'], text=True)
        except (OSError, subprocess.CalledProcessError):
            gpu_snapshot = 'unavailable'
        manifest = dict(**scientific, param_count=params,
                        initial_weights_sha256=tensor_digest([v.detach() for v in model.state_dict().values()]),
                        pid=os.getpid(),
                        process_start_ticks=Path('/proc/self/stat').read_text().split()[21],
                        host=platform.node(), python=platform.python_version(),
                        torch=torch.__version__, device=str(device),
                        gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                        gpu_snapshot=gpu_snapshot, start_time=time.time(),
                        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
        atomic_json(output / 'manifest.json', manifest)
        for p in SOURCES:
            dest = output / 'source' / p.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(BOUND_SOURCES[p])
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config['lr'],
        betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'],
        fused=device.type == 'cuda')
    x, y, xv, yv = [a.to(device) if a is not None else None for a in data_cpu]
    generator = torch.Generator().manual_seed(config['seed'] + 1729)
    torch.manual_seed(config['seed'] + 2718)
    step, examples, elapsed = 0, 0, 0.0
    order, cursor, trace = torch.randperm(len(x), generator=generator), 0, []
    if resume and existed and (output / 'checkpoint.pt').exists():
        saved = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=False)
        if saved['scientific'] != scientific:
            raise ValueError('Checkpoint provenance mismatch')
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        step, examples, elapsed, trace = (saved[k] for k in ['step', 'examples', 'elapsed', 'trace'])
        order, cursor = saved['order'], saved['cursor']
        generator.set_state(saved['batch_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(saved['cuda_rng'], device)
        # Remove observations newer than the last complete checkpoint.
        (output / 'trace.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in trace))
        manifest.update(pid=os.getpid(), process_start_ticks=Path('/proc/self/stat').read_text().split()[21])
        atomic_json(output / 'manifest.json', manifest)
    elif resume and existed and not (output / 'checkpoint.pt').exists():
        raise ValueError('No checkpoint to resume; retain this attempt and use a fresh output')
    sync(device)
    started = time.perf_counter()

    def checkpoint():
        sync(device)
        value = dict(scientific=scientific, model=model.state_dict(), optimizer=opt.state_dict(),
                     step=step, examples=examples, elapsed=elapsed + time.perf_counter() - started,
                     trace=trace, order=order, cursor=cursor, batch_rng=generator.get_state(),
                     torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
        tmp = output / 'checkpoint.tmp'
        torch.save(value, tmp)
        tmp.replace(output / 'checkpoint.pt')

    def observe():
        # Evaluation draws never change subsequent training randomness. Re-seed
        # per observation so sustained categorical success spans fresh samples.
        devices = [device.index] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config['seed'] + 1000003 + step)
            observe_with_rng()

    def observe_with_rng():
        train_loss, train_acc = evaluate(model, x, y)
        record = dict(step=step, examples=examples, epochs=examples / len(x),
                      train_loss=train_loss, train_acc=train_acc,
                      train_bits=len(x) * (math.log2(config['p'] + 2) - train_loss / math.log(2)))
        if xv is not None:
            val_loss, val_acc = evaluate(model, xv, yv)
            record.update(val_loss=val_loss, val_acc=val_acc)
        if config['model'] == 'categorical':
            mode_loss, mode_acc = evaluate(model, x, y, categorical_mode=True)
            record.update(train_mode_loss=mode_loss, train_mode_acc=mode_acc,
                          train_mode_bits=len(x) * (math.log2(config['p'] + 2)
                                                   - mode_loss / math.log(2)))
            if xv is not None:
                mode_loss, mode_acc = evaluate(model, xv, yv, categorical_mode=True)
                record.update(val_mode_loss=mode_loss, val_mode_acc=mode_acc)
        sync(device)
        record['wall_seconds'] = elapsed + time.perf_counter() - started
        if not all(math.isfinite(v) for v in record.values()):
            raise FloatingPointError(f'Non-finite training metrics: {record}')
        trace.append(record)
        with (output / 'trace.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(dict(run=output.name, **record)), flush=True)

    if step == 0:
        observe()
        checkpoint()
    while step < config['max_steps']:
        if cursor == len(order):
            order, cursor = torch.randperm(len(x), generator=generator), 0
        idx = order[cursor:cursor + config['batch_size']].to(device)
        cursor += len(idx)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x[idx]), y[idx])
        loss.backward()
        opt.step()
        step += 1
        examples += len(idx)
        terminal = False
        if step % config['eval_every'] == 0 or step == config['max_steps']:
            observe()
            if config['stop_on_success'] and config['task'] == 'modular':
                terminal = all(sustained_crossing(trace[-config['sustained_evals']:], key, .99, config['sustained_evals'])
                               for key in primary_accuracy_keys(config))
        if step % config['checkpoint_every'] == 0 or terminal:
            checkpoint()
        if terminal:
            break
    checkpoint()
    summary = summarise(config, trace, params)
    summary['termination'] = 'sustained_success' if step < config['max_steps'] else 'budget'
    summary['peak_cuda_memory_bytes'] = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
    atomic_json(output / 'summary.json', summary)
    print(f'DONE {output.name}: {summary["metrics"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--gate', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    spec = yaml.safe_load(args.config.read_text())
    for item in spec['runs']:
        if args.only and item['name'] not in args.only:
            continue
        config = {**spec.get('defaults', {}), **item}
        name = config.pop('name')
        run(config, args.output / name, torch.device(args.device), resume=args.resume, gate=args.gate)


if __name__ == '__main__':
    main()
