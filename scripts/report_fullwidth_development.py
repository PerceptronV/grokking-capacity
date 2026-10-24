#!/usr/bin/env python3
"""Export all full-width development evidence without implying a matched result."""
import argparse
import csv
import json
from pathlib import Path

from grokking_capacity.experiments.discrete import sustained_crossing
from scripts.discrete_status import statuses


def read(path):
    return json.loads(path.read_text())


def export(root, output, *, prefix='fullwidth-'):
    output.mkdir(parents=True, exist_ok=True)
    state = {row['path']: row['status'] for row in statuses(root)}
    groups = {'arithmetic': [], 'capacity': []}
    for path in sorted(root.glob(f'{prefix}*/**/manifest.json')):
        folder = path.parent
        if (folder.parent/'selection.json').exists():
            continue  # Frozen confirmation has its own report, not development data.
        manifest = read(path)
        cfg = manifest['config']
        if cfg['model'] not in {'baseline', 'quantized', 'categorical'}:
            continue
        trace = []
        if (folder/'trace.jsonl').exists():
            for line in (folder/'trace.jsonl').read_text().splitlines():
                try:
                    trace.append(json.loads(line))
                except json.JSONDecodeError:
                    break
        if not trace:
            continue
        final = trace[-1]
        summary = read(folder/'summary.json') if (folder/'summary.json').exists() else {}
        row = dict(path=str(folder.resolve()), family=cfg['model'], purpose=cfg.get('purpose', 'calibration' if cfg['task']=='random' else 'arithmetic_development'), status=state[str(folder.relative_to(root))],
                   termination=summary.get('termination'), p=cfg['p'], depth=cfg['depth'], dim=cfg['dim'],
                   mlp_dim=cfg.get('mlp_dim') or 4*cfg['dim'], params=manifest['param_count'],
                   levels=cfg.get('levels'), alphabet=cfg.get('alphabet'),
                   categorical_temperature=cfg.get('categorical_temperature', 1.) if cfg['model']=='categorical' else None,
                   categorical_normalization=cfg.get('categorical_normalization', 'none') if cfg['model']=='categorical' else None,
                   quantizer_surrogate=cfg.get('quantizer_surrogate', 'identity') if cfg['model']=='quantized' else None,
                   quantizer_clip=cfg.get('quantizer_clip', 1.) if cfg['model']=='quantized' else None,
                   lr=cfg['lr'], weight_decay=cfg['weight_decay'], seed=cfg['seed'],
                   primary_evaluation=cfg.get('categorical_evaluation', 'sampled') if cfg['model']=='categorical' else 'deterministic',
                   budget_updates=cfg['max_steps'], last_update=final['step'],
                   train_accuracy=final['train_acc'], wall_seconds=final['wall_seconds'],
                   scientific_source_hashes=json.dumps(manifest['source_hashes'], sort_keys=True))
        if cfg['task'] == 'modular':
            row.update(heldout_accuracy=final['val_acc'],
                       best_heldout_accuracy=max(x['val_acc'] for x in trace),
                       hard_mode_heldout_accuracy=final.get('val_mode_acc'),
                       best_hard_mode_heldout_accuracy=max((x['val_mode_acc'] for x in trace if 'val_mode_acc' in x), default=None),
                       arithmetic_gate_passed=summary.get('arithmetic_gate_passed', False),
                       audit_present=(folder/'audit.json').exists(),
                       extension_parent=read(folder/'extension.json')['parent'] if (folder/'extension.json').exists() else None)
            for name,key in [('train','train_acc'),('heldout','val_acc'),('hard_mode_heldout','val_mode_acc')]:
                crossing=sustained_crossing(trace,key,.99,cfg['sustained_evals'])
                row[name+'_99_update']=crossing['step'] if crossing else None
            groups['arithmetic'].append(row)
        else:
            audit=read(folder/'audit_capacity.json') if (folder/'audit_capacity.json').exists() else {}
            row.update(n_samples=cfg['n_samples'],
                       best_sampled_bits=max(x['train_bits'] for x in trace),
                       best_hard_mode_bits=max((x['train_mode_bits'] for x in trace if 'train_mode_bits' in x),default=None),
                       audited_checkpoint_bits=audit.get('checkpoint_capacity_bits'),
                       above_target_rejection_verified=audit.get('above_target_rejection_independently_verified',False),
                       arithmetic_gate_path=manifest['gate']['path'] if manifest.get('gate') else None)
            groups['capacity'].append(row)
    for name, rows in groups.items():
        if rows:
            with (output/(name+'.csv')).open('w') as file:
                writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    comparisons = []
    for selection in sorted(root.glob(f'{prefix}*/**/selection.json')):
        frozen = read(selection)
        if frozen.get('control') != 'fullwidth_unrestricted':
            continue
        worker = selection.parent/'orchestrator.json'
        comparisons.append(dict(path=str(selection.parent.resolve()),
                                capacity_match_accepted=frozen['assessment']['accepted'],
                                phase=read(worker)['phase'] if worker.exists() else 'unknown'))
    note=dict(scope='Full-width development only; obsolete learned narrow channels and frozen confirmation runs excluded.',
              comparison_status=('Frozen full-width selections exist; see comparison phases for completion status.'
                                 if comparisons else 'No accepted full-width capacity-matched A/B/C selection yet.'),
              comparisons=comparisons,
              counts={name:len(rows) for name,rows in groups.items()},
              accuracy_note='Categorical raw traces retain sampled and hard-mode metrics separately; primary_evaluation records the registered policy. Null crossings are unobserved, not budget times.',
              capacity_note='Above-target screens establish lower bounds, not saturated capacity estimates. Arithmetic label fit is not a capacity measurement.')
    (output/'status.json').write_text(json.dumps(note,indent=2)+'\n')
    print(json.dumps(note,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/discrete'))
    parser.add_argument('--output',type=Path,default=Path('reports/fullwidth_development'))
    parser.add_argument('--prefix',default='fullwidth-')
    args=parser.parse_args()
    export(args.root,args.output,prefix=args.prefix)
