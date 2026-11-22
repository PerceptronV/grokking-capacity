#!/usr/bin/env python3
"""Overlay saved train/held-out curves for explicitly selected models."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(paths, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = plt.get_cmap('tab10').colors
    for index, path in enumerate(paths):
        records = [json.loads(line) for line in (path / 'trace.jsonl').read_text().splitlines()]
        spec = json.loads((path / 'manifest.json').read_text())
        label = f"{path.name} ({spec['param_count']:,} params)"
        for ax, unit in zip(axes, ['step', 'wall_seconds']):
            x = [r[unit] for r in records]
            for key, style, suffix in [('train_acc', '--', 'train'), ('val_acc', '-', 'held out')]:
                ax.plot(x, [r[key] for r in records], style, color=colors[index % len(colors)],
                        lw=1.5, label=f'{label}, {suffix}')
            ax.set_ylim(0, 1.02)
            ax.axhline(.99, ls=':', c='grey', lw=.7)
            ax.set_ylabel('Accuracy')
            ax.set_xlabel('Optimizer updates' if unit == 'step' else 'Elapsed seconds (shared GPU)')
    axes[0].legend(fontsize=7, frameon=False, loc='lower right')
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(output.with_suffix('.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.runs, args.output)
