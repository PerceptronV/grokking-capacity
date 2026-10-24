#!/usr/bin/env python3
"""Validate and export the completed, frozen p149 comparison and compact evidence.

Run from the repository root with PYTHONPATH=src python -m scripts.report_discrete.
Checkpoint audits must already exist; original gate audits are never overwritten.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from statistics import median

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from grokking_capacity.experiments.discrete import check_gate, source_hashes, summarise
from scripts.analyse_discrete import memorisation_intervals
from scripts.assess_discrete_match import assess, load_run
from scripts.confirm_discrete import confirmation_runs


def read(path):
    return json.loads(path.read_text())


def trace(path):
    return [json.loads(line) for line in (path / 'trace.jsonl').read_text().splitlines()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def write_csv(path, rows):
    with path.open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def comparison(rows):
    """Keep censored crossings missing, including paired ratios and medians."""
    aggregates, pairs = [], []
    for arm in 'ABC':
        selected = [row for row in rows if row['arm'] == arm]
        for threshold in (95, 99):
            field = f'generalise_{threshold}_step'
            observed = [row for row in selected if row[field] is not None]
            result = dict(arm=arm, threshold=threshold, total=len(selected),
                          reached=len(observed), censored=len(selected)-len(observed))
            for unit in ('step', 'examples', 'epochs', 'wall_seconds'):
                values = [row[f'generalise_{threshold}_{unit}'] for row in observed]
                result[f'median_observed_{unit}'] = median(values) if values else None
            aggregates.append(result)
    for seed in sorted({row['seed'] for row in rows}):
        arms = {row['arm']:row for row in rows if row['seed'] == seed}
        for threshold in (95, 99):
            result = dict(seed=seed, threshold=threshold)
            for unit in ('step', 'wall_seconds'):
                for arm, row in arms.items():
                    result[f'{arm}_{unit}'] = row[f'generalise_{threshold}_{unit}']
                a, b, c = (result[f'{arm}_{unit}'] for arm in 'ABC')
                result[f'B_over_A_{unit}'] = b/a if a and b is not None else None
                result[f'B_over_C_{unit}'] = b/c if c and b is not None else None
            pairs.append(result)
    return aggregates, pairs


def validate(root, c_paths):
    state, frozen = read(root/'orchestrator.json'), read(root/'selection.json')
    if state['phase'] != 'complete':
        raise ValueError('All nine frozen runs must finish before final reporting')
    if state['inputs']['source_hashes'] != source_hashes():
        raise ValueError('Scientific training sources changed')
    if state['inputs']['script_sha256'] != digest(Path('scripts/confirm_discrete.py')):
        raise ValueError('Confirmation implementation changed')
    calibration = {}
    for arm in 'AB':
        calibration[arm] = [load_run(Path(p)) for p in state['inputs'][f'{arm.lower()}_runs']]
    assessment = assess(calibration['A'], calibration['B'], p=149)
    if not assessment['accepted'] or assessment != frozen['assessment'] or assessment != read(root/'capacity_assessment.json'):
        raise ValueError('Frozen capacity match changed or failed')
    for folder, hashes in frozen['artifact_hashes'].items():
        for name, expected in hashes.items():
            if digest(Path(folder)/name) != expected:
                raise ValueError(f'Calibration evidence changed: {folder}/{name}')
    expected = confirmation_runs(calibration['A'][0]['config'], calibration['B'][0]['config'])
    if frozen['runs'] != [dict(name=name, **cfg) for name, cfg in expected]:
        raise ValueError('Frozen confirmation configurations changed')
    calibration['C'] = [load_run(p) for p in c_paths]
    for control in calibration['C']:
        candidate = next(row for row in calibration['B'] if all(
            row['config'][key] == control['config'][key] for key in ('seed', 'n_samples')))
        expected_config = {**candidate['config'], 'model':'continuous'}
        if control['config'] != expected_config or control['source_hashes'] != source_hashes():
            raise ValueError('C calibration is not the exact continuous control')
        bm, cm = (read(Path(row['path'])/'manifest.json') for row in (candidate,control))
        if any(bm[key] != cm[key] for key in ('param_count','dataset_sha256','initial_weights_sha256')):
            raise ValueError('C calibration pairing failed')
    gates = {}
    for row in calibration['B']:
        manifest = read(Path(row['path'])/'manifest.json')
        gate = Path(manifest['gate']['path'])
        if check_gate(gate, row['config']) != manifest['gate']:
            raise ValueError('Arithmetic gate changed')
        audit = read(gate/'audit.json')
        if not audit['arithmetic_gate_passed'] or (gate/'audit.json').stat().st_mtime >= manifest['start_time']:
            raise ValueError('Arithmetic checkpoint audit did not precede random-label training')
        gates[str(gate)] = dict(summary_sha256=digest(gate/'summary.json'),
                               audit_sha256=digest(gate/'audit.json'),
                               audited_at=(gate/'audit.json').stat().st_mtime,
                               first_random_start=min(r['start_time'] for r in [
                                   read(Path(item['path'])/'manifest.json') for item in calibration['B']]))
    rows, records, manifests = [], {}, {}
    for name, config in expected:
        path = root/name
        manifest, summary, audit = (read(path/f'{kind}.json') for kind in ('manifest','summary','audit'))
        data = trace(path)
        calculated = summarise(config, data, manifest['param_count'])
        if manifest['config'] != config or summary['config'] != config or manifest['source_hashes'] != source_hashes():
            raise ValueError(f'Configuration/source changed: {name}')
        if any(summary[key] != calculated[key] for key in calculated):
            raise ValueError(f'Saved summary does not reproduce: {name}')
        if audit['checkpoint_step'] != data[-1]['step'] or any(
            abs(audit[key] - data[-1][key]) >= 1e-6 for key in ('train_acc','val_acc')):
            raise ValueError(f'Independent checkpoint audit disagrees: {name}')
        if any(audit[key] != manifest[key] for key in ('param_count','dataset_sha256','initial_weights_sha256')):
            raise ValueError(f'Independent provenance audit disagrees: {name}')
        if config['model'] == 'discrete' and len(audit.get('hard_code_audit',[])) != config['depth']:
            raise ValueError(f'Hard-code audit missing: {name}')
        if summary['termination'] == 'budget' and data[-1]['step'] != config['max_steps']:
            raise ValueError(f'Budget run ended early: {name}')
        if summary['termination'] == 'sustained_success' and any(
            calculated['metrics'][key] is None for key in ('train_99','generalise_99')):
            raise ValueError(f'Success incorrectly recorded: {name}')
        intervals = memorisation_intervals(data)
        row = dict(name=name, arm=name[0], seed=config['seed'], params=manifest['param_count'],
                   final_step=data[-1]['step'], final_train_acc=data[-1]['train_acc'],
                   final_val_acc=data[-1]['val_acc'], termination=summary['termination'],
                   delay_steps_99=calculated['metrics']['delay_steps_99'],
                   max_train_val_gap=calculated['metrics']['max_train_val_gap'],
                   longest_observed_memorisation_steps=max((r['observed_steps'] for r in intervals), default=0),
                   memorisation_observations=sum(r['observations'] for r in intervals),
                   memorisation_end_censored=any(r['end_censored'] for r in intervals))
        for field in ('train_95','train_99','generalise_95','generalise_99'):
            crossing = calculated['metrics'][field]
            for unit in ('step','examples','epochs','wall_seconds'):
                row[f'{field}_{unit}'] = crossing[unit] if crossing else None
        rows.append(row)
        records[name], manifests[name] = data, manifest
    for seed in (43,44,45):
        a,b,c = (manifests[f'{arm}_s{seed}'] for arm in 'ABC')
        if len({m['dataset_sha256'] for m in (a,b,c)}) != 1:
            raise ValueError(f'Modular datasets are unpaired: {seed}')
        if b['param_count'] != c['param_count'] or b['initial_weights_sha256'] != c['initial_weights_sha256']:
            raise ValueError(f'B/C tensor pairing failed: {seed}')
    return rows, records, assessment, calibration, gates


def plots(output, rows, records, calibration, target):
    colors = dict(A='#3366a0', B='#d36a25', C='#25856d')
    fig, axes = plt.subplots(3,3, figsize=(12,8), sharex=True, sharey=True)
    for row in rows:
        name,arm = row['name'],row['arm']
        ax = axes[row['seed']-43, 'ABC'.index(arm)]
        data = records[name]
        for key,style,label in [('train_acc','--','Train'),('val_acc','-','Held out')]:
            ax.plot([r['step'] for r in data],[r[key] for r in data],style,
                    color=colors[arm],lw=1.1,alpha=.85 if style=='-' else .6,label=label)
        ax.scatter([data[-1]['step']],[data[-1]['val_acc']],s=22,color=colors[arm],marker='x')
        ax.set_xscale('symlog', linthresh=1000)
        ax.set_xlim(0,60000)
        ax.set_xticks([0,1000,10000,60000],['0','1k','10k','60k'])
        ax.set_ylim(0,1.025)
        ax.axhline(.99,ls=':',c='grey',lw=.7)
        ax.set_title(f"{name}: {'99% criterion met' if row['generalise_99_step'] is not None else '99% not reached by budget'}",fontsize=10)
        ax.grid(alpha=.15)
    for ax in axes[-1]: ax.set_xlabel('Optimizer updates (linear to 1k, then log)')
    for ax in axes[:,0]: ax.set_ylabel('Accuracy')
    axes[0,0].legend(loc='lower right',frameon=False)
    fig.suptitle('Frozen p149 comparison: all three fresh seeds\nCurves stop at completion; × marks the final evaluation',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94))
    for suffix in ('png','pdf'): fig.savefig(output/f'learning_curves.{suffix}',dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(10,6),sharex=True,sharey=True)
    for arm, runs in calibration.items():
        for row in runs:
            config = row['config']
            ax = axes[config['seed']-61, [16384,32768].index(config['n_samples'])]
            data = trace(Path(row['path']))
            ax.plot([r['step'] for r in data],[r['train_bits']/1000 for r in data],
                    label=arm,color=colors[arm],lw=1.2)
    for i, seed in enumerate((61,62)):
        for j, n in enumerate((16384,32768)):
            ax=axes[i,j]
            ax.axhspan(.9*target/1000,1.1*target/1000,color='grey',alpha=.12)
            ax.axhline(target/1000,color='grey',ls=':',lw=.8)
            ax.set_title(f'Seed {seed}, n={n:,}')
            ax.set_xlim(0,60000)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.15)
            ax.legend(frameon=False,loc='lower right')
    for ax in axes[-1]: ax.set_xlabel('Optimizer updates')
    for ax in axes[:,0]: ax.set_ylabel('Learned label information (kbits)')
    fig.suptitle('Finite-budget capacity measurements\nGrey band: fixed modular dataset target ±10%; C is descriptive, seed 61 only',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.92))
    for suffix in ('png','pdf'): fig.savefig(output/f'capacity_curves.{suffix}',dpi=170)
    plt.close(fig)


def report(root, output, c_paths):
    rows, records, assessment, calibration, gates = validate(root, c_paths)
    output.mkdir(parents=True,exist_ok=True)
    aggregates, pairs = comparison(rows)
    write_json(output/'confirmation.json',dict(runs=rows, aggregates=aggregates, paired=pairs,
               median_note='Medians use observed crossings only; consult reached/total, never treat censoring as a crossing.'))
    write_csv(output/'confirmation.csv',rows)
    write_csv(output/'paired.csv',pairs)
    capacities=[]
    for arm,runs in calibration.items():
        for row in runs:
            capacities.append(dict(arm=arm,seed=row['config']['seed'],n_samples=row['config']['n_samples'],
                params=row['params'],best_bits=row['best_bits'],tail_relative_drift=row['tail_relative_drift']))
    write_csv(output/'capacity.csv',capacities)
    write_json(output/'validation.json',dict(all_nine_complete=True,capacity_match=assessment,
               checkpoint_audits=9,paired_datasets=True,paired_B_C_initial_tensors=True,
               scientific_source_hashes=source_hashes(),arithmetic_before_random=gates))
    evidence=output/'evidence'
    for folder in [root/name for name in records]+[Path(row['path']) for runs in calibration.values() for row in runs]+[Path(g) for g in gates]:
        destination=evidence/folder.parent.name/folder.name
        destination.mkdir(parents=True,exist_ok=True)
        for name in ('manifest.json','summary.json','trace.jsonl','audit.json'):
            if (folder/name).exists(): shutil.copyfile(folder/name,destination/name)
    for name in ('selection.json','configuration.yaml','capacity_assessment.json','orchestrator.json'):
        shutil.copyfile(root/name,output/name)
    history=[]
    for manifest_path in sorted(root.parent.glob('*/*/manifest.json')):
        path=manifest_path.parent
        if path.parent==root: continue
        manifest=read(manifest_path)
        item=dict(path=str(path.relative_to(root.parent)),config=manifest['config'],
                  gate=manifest.get('gate'),params=manifest['param_count'])
        if (path/'summary.json').exists(): item['summary']=read(path/'summary.json')
        if (path/'scope_stop.json').exists(): item['scope_stop']=read(path/'scope_stop.json')
        history.append(item)
    write_json(output/'development_history.json',history)
    plots(output,rows,records,calibration,assessment['target_bits'])
    write_json(output/'artifact_hashes.json',{str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*'))
               if p.is_file() and p.name!='artifact_hashes.json'})
    print(json.dumps(dict(output=str(output),runs=len(rows),aggregates=aggregates),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--c-runs',type=Path,nargs='+',required=True)
    args=parser.parse_args()
    report(args.root,args.output,args.c_runs)
