#!/usr/bin/env python3
"""Independently reload and verify a completed modular arithmetic checkpoint."""
import argparse
import json
from pathlib import Path

import torch

from grokking_capacity.experiments.discrete import (
    check_gate, dataset, evaluate, primary_accuracy_keys, source_hashes, summarise, tensor_digest,
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
    categorical = config['model'] == 'categorical'
    hard_mode_primary = categorical and config.get('categorical_evaluation', 'sampled') == 'hard_mode'
    train = evaluate(model, x, y, categorical_mode=categorical)
    heldout = evaluate(model, xv, yv, categorical_mode=categorical)
    for metric, observed in [('train_acc', train[1]), ('val_acc', heldout[1])]:
        recorded_key = metric.replace('_acc', '_mode_acc') if categorical else metric
        assert abs(summary['final'][recorded_key] - observed) < 1e-6
    result = dict(run=str(path.resolve()), checkpoint_step=saved['step'],
                  train_acc=train[1], val_acc=heldout[1], param_count=count_parameters(model),
                  dataset_sha256=manifest['dataset_sha256'],
                  initial_weights_sha256=manifest['initial_weights_sha256'],
                  arithmetic_gate_passed=summary['arithmetic_gate_passed'],
                  primary_evaluation=summary.get('primary_evaluation', 'sampled' if categorical else 'deterministic'))
    if categorical:
        # Reproduce the original sampled measurements on their original device.
        # CPU argmax accuracy above remains an independent-device check.
        device = torch.device(manifest['device'])
        model.to(device)
        devices = [device.index] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config['seed'] + 1000003 + saved['step'])
            sampled_train = evaluate(model, x.to(device), y.to(device))
            sampled_val = evaluate(model, xv.to(device), yv.to(device))
            for label, observed in [('train', sampled_train), ('val', sampled_val)]:
                assert abs(summary['final'][label + '_acc'] - observed[1]) < 1e-6
                assert abs(summary['final'][label + '_loss'] - observed[0]) < 1e-5
            # Fresh independent draws measure evaluation noise without changing
            # the recorded primary metric or substituting mode for sampling.
            repeats = []
            for seed in [91241, 91242, 91243]:
                torch.manual_seed(seed)
                repeats.append(evaluate(model, xv.to(device), yv.to(device))[1])
        result['cpu_mode_train_acc'], result['cpu_mode_val_acc'] = train[1], heldout[1]
        result['sampled_train_acc'], result['sampled_val_acc'] = sampled_train[1], sampled_val[1]
        if not hard_mode_primary:
            result['train_acc'], result['val_acc'] = sampled_train[1], sampled_val[1]
        result['sampled_heldout_repeats'] = repeats
        result['sampled_repeat_mean'] = sum(repeats) / len(repeats)
        if summary['arithmetic_gate_passed']:
            assert (heldout[1] if hard_mode_primary else result['sampled_repeat_mean']) >= 0.99
        model.cpu()
    if config['model'] in {'categorical', 'quantized'}:
        states = model.categorical_states if categorical else model.quantizers
        codes = [[] for _ in states]
        def capture(index):
            def hook(module, _args, state):
                if categorical:
                    assert state.shape[-2:] == (config['dim'], config['alphabet'])
                    assert torch.all((state == 0) | (state == 1))
                    assert torch.all(state.sum(-1) == 1)
                    ids = state.argmax(-1)
                else:
                    assert state.shape[-1] == config['dim']
                    ids = ((state + module.clip) * ((module.levels - 1) /
                                                    (2 * module.clip))).round().long()
                    assert torch.equal(state, ids.to(state) * (2 * module.clip /
                                                               (module.levels - 1)) - module.clip)
                codes[index].append(ids[:, -1].detach())
            return hook
        handles = [module.register_forward_hook(capture(i)) for i, module in enumerate(states)]
        try:
            evaluate(model, xv, yv, categorical_mode=categorical)
        finally:
            for handle in handles:
                handle.remove()
        result['hard_code_audit'] = []
        for module, chunks in zip(states, codes):
            ids = torch.cat(chunks)
            _, counts = torch.unique(ids, dim=0, return_counts=True)
            probs = counts.float() / counts.sum()
            result['hard_code_audit'].append(dict(
                width=config['dim'], bits_per_token=module.bits_per_token,
                unique_final_position_codes=len(counts),
                empirical_code_entropy=float(-(probs * probs.log2()).sum()),
                mean_used_symbols_per_coordinate=sum(len(torch.unique(column))
                                                     for column in ids.T) / ids.shape[1]))
        if summary['arithmetic_gate_passed']:
            result['gate'] = check_gate(path)
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
