#!/usr/bin/env python3
"""Reload random-label checkpoints, including early above-target rejections."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from grokking_capacity.experiments.discrete import (
    check_gate, dataset, evaluate, source_hashes, tensor_digest,
)
from grokking_capacity.models.discrete import make_model
from grokking_capacity.models.transformer import count_parameters


def audit_capacity(path):
    manifest = json.loads((path / 'manifest.json').read_text())
    cfg = manifest['config']
    assert cfg['task'] == 'random'
    if not ((path / 'summary.json').exists() or (path / 'screen_rejection.json').exists()):
        raise ValueError('Audit a completed or explicitly rejected screen, not a live worker')
    rejection = json.loads((path / 'screen_rejection.json').read_text()) if (path / 'screen_rejection.json').exists() else {}
    checkpoint = path / rejection.get('checkpoint_file', 'checkpoint.pt')
    if rejection.get('checkpoint_sha256'):
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == rejection['checkpoint_sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['scientific']['config'] == cfg
    assert saved['scientific']['source_hashes'] == manifest['source_hashes'] == source_hashes()
    assert check_gate(Path(manifest['gate']['path']), cfg) == manifest['gate']
    assert (Path(manifest['gate']['path']) / 'audit.json').stat().st_mtime < manifest['start_time']
    x, y, _, _ = data = dataset(cfg)
    assert tensor_digest(data) == manifest['dataset_sha256']
    torch.manual_seed(cfg['seed'])
    model = make_model(cfg)
    assert count_parameters(model) == manifest['param_count']
    assert tensor_digest([v.detach() for v in model.state_dict().values()]) == manifest['initial_weights_sha256']
    model.load_state_dict(saved['model'])
    row = saved['trace'][-1]
    assert row['step'] == saved['step']
    categorical = cfg['model'] == 'categorical'
    cpu_loss, cpu_acc = evaluate(model, x, y, categorical_mode=categorical)
    prefix = 'train_mode' if categorical else 'train'
    # Binary quantization produced one CPU/GPU prediction difference at a
    # zero boundary (CPU 0, GPU 3.58e-7) in the d80/FF480 checkpoint. Permit
    # Categorical argmax also showed one tie-level difference (8.9e-8 vs
    # -3.0e-8 competing-logit margin). Permit at most one example for hard
    # discrete cross-device accuracy, still require
    # tight loss agreement and reproduce original-device metrics below.
    cpu_accuracy_tolerance = 1e-6 + (1 / len(x) if cfg['model'] in {'quantized', 'categorical'} else 0)
    assert abs(cpu_acc - row[prefix + '_acc']) < cpu_accuracy_tolerance
    assert abs(cpu_loss - row[prefix + '_loss']) < 1e-4
    device = torch.device(manifest['device'])
    model.to(device)
    with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
        torch.manual_seed(cfg['seed'] + 1000003 + saved['step'])
        sampled_loss, sampled_acc = evaluate(model, x.to(device), y.to(device))
        original_mode = evaluate(model, x.to(device), y.to(device), categorical_mode=True) if categorical else (sampled_loss, sampled_acc)
    assert abs(sampled_acc - row['train_acc']) < 1e-6
    assert abs(sampled_loss - row['train_loss']) < 1e-5
    assert abs(original_mode[1] - row[prefix + '_acc']) < 1e-6
    assert abs(original_mode[0] - row[prefix + '_loss']) < 1e-5
    information = lambda loss: len(x) * (math.log2(cfg['p'] + 2) - loss / math.log(2))
    target = int(cfg['p'] * (cfg['p'] - 1) * cfg['train_fraction']) * math.log2(cfg['p'] + 2)
    result = dict(run=str(path.resolve()), checkpoint_step=saved['step'],
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  param_count=count_parameters(model), dataset_sha256=manifest['dataset_sha256'],
                  scientific_source_hashes=source_hashes(), cpu_loss=cpu_loss, cpu_accuracy=cpu_acc,
                  cpu_mode_evaluation=categorical, cpu_loss_difference=cpu_loss-row[prefix + '_loss'],
                  cpu_accuracy_difference=cpu_acc-row[prefix + '_acc'],
                  cpu_accuracy_tolerance=cpu_accuracy_tolerance, cpu_loss_tolerance=1e-4,
                  original_device_mode_loss=original_mode[0], sampled_loss=sampled_loss,
                  sampled_accuracy=sampled_acc,
                  checkpoint_capacity_bits=max(information(cpu_loss), information(sampled_loss), information(original_mode[0])),
                  cpu_original_mode_capacity_difference_bits=information(cpu_loss)-information(original_mode[0]),
                  fixed_dataset_target_bits=target, gate=manifest['gate'])
    if (path / 'screen_rejection.json').exists():
        # A trace peak is insufficient for this assertion: reloaded weights
        # themselves must still demonstrate the above-target memory.
        assert min(information(cpu_loss), max(information(original_mode[0]), information(sampled_loss))) > 1.1 * target
        result['above_target_rejection_independently_verified'] = True
    (path / 'audit_capacity.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    args = parser.parse_args()
    torch.set_num_threads(2)
    for path in args.runs:
        audit_capacity(path)
