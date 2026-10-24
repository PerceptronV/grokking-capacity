#!/usr/bin/env python3
"""Assess a fixed-dataset, finite-budget A/B capacity match from saved runs.

Passing supports an empirical match under the registered optimiser and budget;
it does not prove a representational bound or imply that A cannot grok.
"""
import argparse
import json
import math
from pathlib import Path


SHARED = ('p', 'operation', 'depth', 'heads', 'dropout', 'lr', 'weight_decay',
          'beta1', 'beta2', 'batch_size', 'max_steps')
ARCH = ('model', 'dim', 'latent_dim', 'levels', 'alphabet', 'quantizer_clip')


def load_run(path):
    manifest = json.loads((path / 'manifest.json').read_text())
    summary = json.loads((path / 'summary.json').read_text())
    trace = [json.loads(line) for line in (path / 'trace.jsonl').read_text().splitlines()]
    cfg = manifest['config']
    if summary['config'] != cfg or summary['final'] != trace[-1]:
        raise ValueError(f'Inconsistent completed artifacts: {path}')
    if cfg['task'] != 'random' or trace[-1]['step'] != cfg['max_steps']:
        raise ValueError(f'Not a completed fixed-budget calibration: {path}')
    best = max(r['train_bits'] for r in trace)
    if abs(best - summary['best_train_bits']) > 1e-6:
        raise ValueError(f'Inconsistent peak memorisation: {path}')
    recent = [r['train_bits'] for r in trace if r['step'] >= .8 * cfg['max_steps']]
    previous = [r['train_bits'] for r in trace if .6 * cfg['max_steps'] <= r['step'] < .8 * cfg['max_steps']]
    if not recent or not previous:
        raise ValueError(f'Insufficient late observations: {path}')
    mean = sum(recent) / len(recent)
    drift = (mean - sum(previous) / len(previous)) / max(abs(mean), 1.)
    return dict(path=str(path.resolve()), config=cfg, params=manifest['param_count'],
                source_hashes=manifest['source_hashes'], dataset_sha256=manifest['dataset_sha256'],
                best_bits=best, tail_relative_drift=drift)


def assess(a, b, *, p=149, fraction=.5, tolerance=.1, min_steps=60000, max_drift=.05):
    if p < 2 or any(p % d == 0 for d in range(2, math.isqrt(p) + 1)):
        raise ValueError('This assessment is for prime-modulus division')
    if not 0 < fraction < 1 or not 0 < tolerance < 1:
        raise ValueError('Invalid split or tolerance')
    target = int(p * (p - 1) * fraction) * math.log2(p + 2)
    issues, cells = [], {}
    if not a or not b:
        return dict(accepted=False, target_bits=target, issues=['Both arms are required'])
    reference = a[0]
    for arm, rows, models in [('A', a, {'baseline'}),
                              ('B', b, {'discrete', 'quantized', 'categorical'})]:
        arm_ref = rows[0]
        cells[arm] = {}
        for row in rows:
            cfg = row['config']
            cell = cfg['seed'], cfg['n_samples']
            if cell in cells[arm]:
                issues.append(f'{arm}: duplicate seed/size {cell}')
            cells[arm][cell] = row
            if cfg['model'] not in models or any(cfg.get(k) != arm_ref['config'].get(k) for k in ARCH):
                issues.append(f'{arm}: inconsistent architecture')
            if cfg['p'] != p or cfg['operation'] != '/' or any(cfg.get(k) != reference['config'].get(k) for k in SHARED):
                issues.append(f'{arm}: task/optimiser/horizon mismatch')
            if row['source_hashes'] != reference['source_hashes']:
                issues.append(f'{arm}: scientific source mismatch')
            if cfg['max_steps'] < min_steps:
                issues.append(f'{arm} {cell}: insufficient horizon')
            if row['best_bits'] >= .95 * cfg['n_samples'] * math.log2(p + 2):
                issues.append(f'{arm} {cell}: dataset-limited calibration')
            if abs(row['tail_relative_drift']) > max_drift:
                issues.append(f'{arm} {cell}: unresolved late drift')
    seeds = sorted({s for s, _ in cells['A']} | {s for s, _ in cells['B']})
    sizes = sorted({n for _, n in cells['A']} | {n for _, n in cells['B']})
    if len(seeds) < 2 or len(sizes) < 2:
        issues.append('Need at least two paired calibration seeds and two dataset sizes')
    estimates = []
    for seed in seeds:
        paired = True
        for n in sizes:
            cell = seed, n
            if cell not in cells['A'] or cell not in cells['B']:
                issues.append(f'Missing paired calibration cell {cell}')
                paired = False
            elif cells['A'][cell]['dataset_sha256'] != cells['B'][cell]['dataset_sha256']:
                issues.append(f'Unpaired calibration data at {cell}')
        if not paired:
            continue
        ca = max(cells['A'][seed, n]['best_bits'] for n in sizes)
        cb = max(cells['B'][seed, n]['best_bits'] for n in sizes)
        if abs(ca / target - 1) > tolerance or abs(cb / target - 1) > tolerance:
            issues.append(f'Seed {seed}: A or B does not match the fixed dataset')
        if ca <= 0 or abs(cb / ca - 1) > tolerance:
            issues.append(f'Seed {seed}: A and B do not match each other')
        estimates.append(dict(seed=seed, a_bits=ca, b_bits=cb,
                              a_to_dataset=ca / target, b_to_dataset=cb / target))
    ratio = b[0]['params'] / a[0]['params']
    if ratio < 4:
        issues.append('B/A parameter ratio is below four')
    return dict(accepted=not issues, target_bits=target, p=p, train_fraction=fraction,
                tolerance=tolerance, min_steps=min_steps, max_tail_relative_drift=max_drift,
                parameter_ratio=ratio, estimates=estimates, issues=sorted(set(issues)),
                interpretation='Finite-budget empirical estimate; does not guarantee absent grokking',
                a_runs=[r.get('path') for r in a], b_runs=[r.get('path') for r in b])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a-runs', type=Path, nargs='+', required=True)
    parser.add_argument('--b-runs', type=Path, nargs='+', required=True)
    parser.add_argument('--p', type=int, default=149)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = assess([load_run(p) for p in args.a_runs], [load_run(p) for p in args.b_runs], p=args.p)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['accepted'] else 1)
