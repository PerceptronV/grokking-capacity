#!/usr/bin/env python3
"""Describe random-label calibration without mistaking interpolation for capacity.

A finite-horizon training plateau is an optimisation observation, not a proof of
representational capacity. The table reports this limitation explicitly.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def analyse(root):
    rows = []
    for path in sorted(root.glob('*/manifest.json')):
        manifest = json.loads(path.read_text())
        config = manifest['config']
        if config['task'] != 'random':
            continue
        trace = []
        for line in (path.parent / 'trace.jsonl').read_text().splitlines():
            try:
                trace.append(json.loads(line))
            except json.JSONDecodeError:
                break
        if not trace:
            continue
        n = config['n_samples']
        ceiling = n * math.log2(config['p'] + 2)
        # Equal-length late windows: absolute drift can hide optimisation noise,
        # hence include std/min/max and never certify a capacity plateau here.
        recent = [r for r in trace if r['step'] >= .8 * config['max_steps']]
        previous = [r for r in trace if .6 * config['max_steps'] <= r['step'] < .8 * config['max_steps']]
        tail = np.array([r['train_bits'] for r in recent])
        mean = float(tail.mean()) if len(tail) else None
        drift = mean - float(np.mean([r['train_bits'] for r in previous])) if mean is not None and previous else None
        completed = (path.parent / 'summary.json').exists()
        best = max(trace, key=lambda r: r['train_bits'])
        row = dict(name=path.parent.name, model=config['model'], dim=config['dim'],
                   p=config['p'], depth=config['depth'], heads=config['heads'],
                   lr=config['lr'], weight_decay=config['weight_decay'], dropout=config['dropout'],
                   max_steps=config['max_steps'], batch_size=config['batch_size'],
                   latent_dim=config.get('latent_dim'), levels=config.get('levels'),
                   seed=config['seed'], n_samples=n, params=manifest['param_count'],
                   completed=completed, steps=trace[-1]['step'],
                   train_acc=trace[-1]['train_acc'], label_bits=ceiling,
                   final_bits=trace[-1]['train_bits'], best_bits=best['train_bits'],
                   best_bits_step=best['step'],
                   best_train_acc=max(r['train_acc'] for r in trace),
                   best_fraction_of_dataset=best['train_bits'] / ceiling,
                   tail_mean_bits=mean,
                   tail_std_bits=float(tail.std()) if len(tail) else None,
                   late_window_drift_bits=drift,
                   tail_bits_per_parameter=mean / manifest['param_count'] if mean is not None else None,
                   tail_fraction_of_dataset=mean / ceiling if mean is not None else None,
                   interpretation=('stopped: scope changed' if (path.parent / 'scope_stop.json').exists()
                                   else 'incomplete: inspect process status') if not completed else (
                       'dataset-limited: increase n' if best['train_bits'] / ceiling >= .95
                       else 'finite-budget: need more n, seeds and horizon checks'))
        rows.append(row)
    output = root / 'analysis'
    output.mkdir(exist_ok=True)
    (output / 'capacity_screen.json').write_text(json.dumps(rows, indent=2) + '\n')
    if rows:
        with (output / 'capacity_screen.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    for row in rows:
        print(json.dumps(row))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    analyse(args.root)
