#!/usr/bin/env python3
"""Independently reload and verify a completed modular arithmetic checkpoint."""
import argparse
import json
from pathlib import Path

import torch

from grokking_capacity.experiments.discrete import (
    check_gate, dataset, evaluate, source_hashes, summarise, tensor_digest,
)
from grokking_capacity.models.discrete import make_model
from grokking_capacity.models.transformer import count_parameters


def audit(path):
    manifest = json.loads((path / 'manifest.json').read_text())
    summary = json.loads((path / 'summary.json').read_text())
    trace = [json.loads(line) for line in (path / 'trace.jsonl').read_text().splitlines()]
    saved = torch.load(path / 'checkpoint.pt', map_location='cpu', weights_only=False)
    config = manifest['config']
    assert config['task'] == 'modular'
    assert saved['scientific']['config'] == config == summary['config']
    assert saved['scientific']['source_hashes'] == manifest['source_hashes'] == source_hashes()
    assert saved['step'] == trace[-1]['step'] == summary['final']['step']
    assert saved['trace'] == trace
    assert summarise(config, trace, manifest['param_count'])['metrics'] == summary['metrics']
    tensors = dataset(config)
    assert tensor_digest(tensors) == manifest['dataset_sha256']
    torch.manual_seed(config['seed'])
    model = make_model(config)
    assert tensor_digest([v.detach() for v in model.state_dict().values()]) == manifest['initial_weights_sha256']
    assert count_parameters(model) == manifest['param_count']
    model.load_state_dict(saved['model'])
    x, y, xv, yv = tensors
    train, heldout = evaluate(model, x, y), evaluate(model, xv, yv)
    for metric, observed in [('train_acc', train[1]), ('val_acc', heldout[1])]:
        assert abs(summary['final'][metric] - observed) < 1e-6
    result = dict(run=str(path.resolve()), checkpoint_step=saved['step'],
                  train_acc=train[1], val_acc=heldout[1], param_count=count_parameters(model),
                  dataset_sha256=manifest['dataset_sha256'],
                  initial_weights_sha256=manifest['initial_weights_sha256'],
                  arithmetic_gate_passed=summary['arithmetic_gate_passed'])
    if config['model'] == 'discrete':
        codes = []
        with torch.no_grad():
            z = model.embedding(xv)
            for layer, channel in zip(model.layers, model.channels):
                latent = channel.latent(layer(z))
                ids = ((latent + 1) * ((channel.levels - 1) / 2)).round().long()
                assert torch.equal(latent, ids.to(latent) * (2 / (channel.levels - 1)) - 1)
                _, counts = torch.unique(ids[:, -1], dim=0, return_counts=True)
                probs = counts.float() / counts.sum()
                codes.append(dict(bits_per_token=channel.bits_per_token,
                                  unique_final_position_codes=len(counts),
                                  empirical_code_entropy=float(-(probs * probs.log2()).sum())))
                z = channel.decode(latent)
        result['hard_code_audit'] = codes
        if summary['arithmetic_gate_passed']:
            result['gate'] = check_gate(path)
    (path / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    args = parser.parse_args()
    torch.set_num_threads(2)
    for path in args.runs:
        audit(path)
