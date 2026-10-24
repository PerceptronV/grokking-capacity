#!/usr/bin/env python3
"""Summarise discrete experiment traces, retaining incomplete/censored runs."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from grokking_capacity.experiments.discrete import summarise


def analyse(root):
    rows, traces = [], []
    for path in sorted(root.glob('*/manifest.json')):
        manifest = json.loads(path.read_text())
        records = []
        for line in (path.parent / 'trace.jsonl').read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                break  # A live writer may have an incomplete final line.
        if not records:
            continue
        config = manifest['config']
        summary = summarise(config, records, manifest['param_count'])
        last, metrics = records[-1], summary['metrics']
        row = dict(name=path.parent.name, model=config['model'], seed=config['seed'],
                   params=manifest['param_count'], steps=last['step'],
                   completed=(path.parent / 'summary.json').exists(),
                   train_acc=last['train_acc'], val_acc=last.get('val_acc'),
                   best_train_bits=summary['best_train_bits'],
                   arithmetic_gate=summary['arithmetic_gate_passed'],
                   delay_steps_99=metrics['delay_steps_99'],
                   max_train_val_gap=metrics['max_train_val_gap'])
        for field in ['train_95', 'train_99', 'generalise_95', 'generalise_99']:
            for unit in ['step', 'epochs', 'wall_seconds']:
                row[f'{field}_{unit}'] = metrics[field][unit] if metrics[field] else None
        rows.append(row)
        traces.append((row['name'], records))
    report = root / 'analysis'
    report.mkdir(exist_ok=True)
    (report / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    if not rows:
        return
    with (report / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(len(traces), 2, figsize=(10, 2.8 * len(traces)), squeeze=False)
    for (name, records), ax in zip(traces, axes):
        for key, label in [('train_acc', 'Train'), ('val_acc', 'Held out')]:
            if key not in records[-1]:
                continue
            for column, unit in enumerate(['step', 'wall_seconds']):
                ax[column].plot([r[unit] for r in records], [r[key] for r in records], label=label)
                ax[column].set_xlabel('Optimizer updates' if unit == 'step' else 'Elapsed seconds (shared GPU)')
                ax[column].set_ylim(0, 1.02)
                ax[column].axhline(.99, ls=':', c='grey', lw=.6)
                ax[column].set_ylabel('Accuracy')
                ax[column].set_title(name)
                ax[column].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(report / 'learning_curves.pdf')
    fig.savefig(report / 'learning_curves.png', dpi=130)
    plt.close(fig)
    for row in rows:
        print(json.dumps(row))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    analyse(args.root)
