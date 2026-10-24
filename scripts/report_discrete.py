#!/usr/bin/env python3
"""Validate and export the completed, frozen p149 comparison and compact evidence.

Run from the repository root with PYTHONPATH=src python -m scripts.report_discrete.
Checkpoint audits must already exist; original gate audits are never overwritten.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
from statistics import median

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from grokking_capacity.experiments.discrete import check_gate, primary_accuracy_keys, source_hashes, summarise
from scripts.analyse_discrete import memorisation_intervals
from scripts.assess_discrete_match import assess, load_run
from scripts.confirm_discrete import confirmation_runs, fullwidth_control
from scripts.confirm_discrete_unrestricted import unrestricted_runs, shared_initial_tensors


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
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
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


def first_passage_delay(records):
    """Supplementary paper-style delay on the run's primary evaluation policy.

    Crossings are first observations, not sustained crossings. Missing events
    stay missing, and validation preceding training gives zero delay.
    """
    train = next((r for r in records if r['train_acc'] >= .99), None)
    heldout = next((r for r in records if r['val_acc'] >= .98), None)
    return dict(first_train_99_step=train['step'] if train else None,
                first_generalise_98_step=heldout['step'] if heldout else None,
                first_passage_delay_99_98_steps=(max(0, heldout['step'] - train['step'])
                                                if train and heldout else None),
                first_passage_delay_99_98_epochs=(max(0, heldout['epochs'] - train['epochs'])
                                                 if train and heldout else None))


def control_config_matches(actual, expected):
    """Ignore only the descriptive control label; require scientific equality."""
    if actual.get('purpose') not in (None, 'capacity_control'):
        return False
    scientific = lambda config: {key: value for key, value in config.items() if key != 'purpose'}
    return scientific(actual) == scientific(expected)


def validate(root, c_paths):
    state, frozen = read(root/'orchestrator.json'), read(root/'selection.json')
    if state['phase'] != 'complete':
        raise ValueError('All nine frozen runs must finish before final reporting')
    if state['inputs']['source_hashes'] != source_hashes():
        raise ValueError('Scientific training sources changed')
    fullwidth = frozen.get('control') == 'fullwidth_unrestricted'
    unrestricted = fullwidth or frozen.get('control') == 'unrestricted'
    implementation = Path('scripts/confirm_discrete_unrestricted.py' if unrestricted and not fullwidth else 'scripts/confirm_discrete.py')
    if state['inputs']['script_sha256'] != digest(implementation):
        raise ValueError('Confirmation implementation changed')
    calibration = {}
    for arm in 'AB':
        calibration[arm] = [load_run(Path(p)) for p in state['inputs'][f'{arm.lower()}_runs']]
    assessment = assess(calibration['A'], calibration['B'], p=149,
                        min_parameter_ratio=frozen['assessment'].get('min_parameter_ratio', 4.))
    if not assessment['accepted'] or assessment != frozen['assessment'] or assessment != read(root/'capacity_assessment.json'):
        raise ValueError('Frozen capacity match changed or failed')
    for folder, hashes in frozen['artifact_hashes'].items():
        for name, expected in hashes.items():
            if digest(Path(folder)/name) != expected:
                raise ValueError(f'Calibration evidence changed: {folder}/{name}')
    builder = unrestricted_runs if unrestricted and not fullwidth else confirmation_runs
    expected = builder(calibration['A'][0]['config'], calibration['B'][0]['config'],
                       **({'seeds': state['inputs'].get('confirmation_seeds', (43,44,45))}
                          if builder is confirmation_runs else {}))
    if frozen['runs'] != [dict(name=name, **cfg) for name, cfg in expected]:
        raise ValueError('Frozen confirmation configurations changed')
    calibration['C'] = [load_run(p) for p in c_paths]
    for control in calibration['C']:
        candidate = next(row for row in calibration['B'] if all(
            row['config'][key] == control['config'][key] for key in ('seed', 'n_samples')))
        expected_config = {**candidate['config'], 'model':'continuous'}
        if fullwidth:
            expected_config = fullwidth_control(candidate['config'])
        elif unrestricted:
            expected_config = {key:value for key,value in {**candidate['config'],'model':'baseline'}.items()
                               if key not in ('latent_dim','levels')}
        if not control_config_matches(control['config'], expected_config) or control['source_hashes'] != source_hashes():
            raise ValueError('C calibration is not the exact continuous control')
        bm, cm = (read(Path(row['path'])/'manifest.json') for row in (candidate,control))
        if bm['dataset_sha256'] != cm['dataset_sha256']:
            raise ValueError('C calibration data pairing failed')
        if unrestricted:
            if cm['param_count'] != frozen['c_params']:
                raise ValueError('Unrestricted C calibration parameter count differs')
            if candidate['config']['model'] != 'categorical' and not shared_initial_tensors(candidate['config'],control['config']):
                raise ValueError('Unrestricted C calibration base tensor pairing failed')
        elif any(bm[key] != cm[key] for key in ('param_count','initial_weights_sha256')):
            raise ValueError('C calibration tensor pairing failed')
    if fullwidth:
        for runs in calibration.values():
            for row in runs:
                folder = Path(row['path'])
                audit = read(folder/'audit_capacity.json')
                if (audit['checkpoint_sha256'] != digest(folder/'checkpoint.pt') or
                    audit['checkpoint_step'] != row['config']['max_steps'] or
                    audit['param_count'] != row['params'] or
                    audit['dataset_sha256'] != row['dataset_sha256'] or
                    audit['scientific_source_hashes'] != row['source_hashes']):
                    raise ValueError(f'Independent capacity checkpoint audit disagrees: {folder}')
    gates = {}
    for row in calibration['B']:
        manifest = read(Path(row['path'])/'manifest.json')
        gate = Path(manifest['gate']['path'])
        if check_gate(gate, row['config']) != manifest['gate']:
            raise ValueError('Arithmetic gate changed')
        gate_config = read(gate/'summary.json')['config']
        if any(row['config'].get(key) != gate_config.get(key) for key in
               ('lr', 'weight_decay', 'beta1', 'beta2', 'batch_size', 'dropout')):
            raise ValueError('Calibration optimizer differs from its arithmetic gate')
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
        if (path/'inherited_run.json').exists():
            inherited = read(path/'inherited_run.json')
            for filename, expected_hash in inherited['artifact_hashes'].items():
                if digest(path/filename) != expected_hash or digest(Path(inherited['path'])/filename) != expected_hash:
                    raise ValueError(f'Inherited A/B evidence changed: {name}/{filename}')
        calculated = summarise(config, data, manifest['param_count'])
        if manifest['config'] != config or summary['config'] != config or manifest['source_hashes'] != source_hashes():
            raise ValueError(f'Configuration/source changed: {name}')
        if any(summary[key] != calculated[key] for key in calculated):
            raise ValueError(f'Saved summary does not reproduce: {name}')
        train_key, val_key = primary_accuracy_keys(config)
        if audit['checkpoint_step'] != data[-1]['step'] or any(
            abs(audit[key] - data[-1][recorded]) >= 1e-6
            for key, recorded in [('train_acc', train_key), ('val_acc', val_key)]):
            raise ValueError(f'Independent checkpoint audit disagrees: {name}')
        if any(audit[key] != manifest[key] for key in ('param_count','dataset_sha256','initial_weights_sha256')):
            raise ValueError(f'Independent provenance audit disagrees: {name}')
        if config['model'] in {'discrete', 'quantized', 'categorical'} and len(audit.get('hard_code_audit',[])) != config['depth']:
            raise ValueError(f'Hard-code audit missing: {name}')
        if summary['termination'] == 'budget' and data[-1]['step'] != config['max_steps']:
            raise ValueError(f'Budget run ended early: {name}')
        if summary['termination'] == 'sustained_success' and any(
            calculated['metrics'][key] is None for key in ('train_99','generalise_99')):
            raise ValueError(f'Success incorrectly recorded: {name}')
        primary_data = [dict(r, train_acc=r[train_key], val_acc=r[val_key]) for r in data]
        intervals = memorisation_intervals(primary_data)
        row = dict(name=name, arm=name[0], seed=config['seed'], params=manifest['param_count'],
                   budget_steps=config['max_steps'], final_step=data[-1]['step'], final_train_acc=data[-1][train_key],
                   final_val_acc=data[-1][val_key], termination=summary['termination'],
                   primary_evaluation=calculated['primary_evaluation'],
                   final_sampled_val_acc=data[-1]['val_acc'] if config['model']=='categorical' else None,
                   final_hard_mode_val_acc=data[-1].get('val_mode_acc'),
                   delay_steps_99=calculated['metrics']['delay_steps_99'],
                   max_train_val_gap=calculated['metrics']['max_train_val_gap'],
                   longest_observed_memorisation_steps=max((r['observed_steps'] for r in intervals), default=0),
                   memorisation_observations=sum(r['observations'] for r in intervals),
                   memorisation_end_censored=any(r['end_censored'] for r in intervals))
        row.update(first_passage_delay(primary_data))
        for field in ('train_95','train_99','generalise_95','generalise_99'):
            crossing = calculated['metrics'][field]
            for unit in ('step','examples','epochs','wall_seconds'):
                row[f'{field}_{unit}'] = crossing[unit] if crossing else None
        rows.append(row)
        records[name], manifests[name] = data, manifest
    for seed in sorted({config['seed'] for _, config in expected}):
        a,b,c = (manifests[f'{arm}_s{seed}'] for arm in 'ABC')
        if len({m['dataset_sha256'] for m in (a,b,c)}) != 1:
            raise ValueError(f'Modular datasets are unpaired: {seed}')
        if unrestricted:
            if b['param_count'] != frozen['b_params'] or c['param_count'] != frozen['c_params']:
                raise ValueError(f'Amended B/C parameter counts changed: {seed}')
            if b['config']['model'] != 'categorical' and not shared_initial_tensors(b['config'],c['config']):
                raise ValueError(f'B/C shared base initialisation failed: {seed}')
        elif b['param_count'] != c['param_count'] or b['initial_weights_sha256'] != c['initial_weights_sha256']:
            raise ValueError(f'B/C tensor pairing failed: {seed}')
    return rows, records, assessment, calibration, gates


def plots(output, rows, records, calibration, target):
    colors = dict(A='#3366a0', B='#d36a25', C='#25856d')
    budget = max(row.get('budget_steps', 60000) for row in rows)
    seed_rows = {seed: index for index, seed in enumerate(sorted({row['seed'] for row in rows}))}
    fig, axes = plt.subplots(3,3, figsize=(12,8), sharex=True, sharey=True)
    for row in rows:
        name,arm = row['name'],row['arm']
        ax = axes[seed_rows[row['seed']], 'ABC'.index(arm)]
        data = records[name]
        train_key, val_key = ('train_mode_acc', 'val_mode_acc') if row.get('primary_evaluation') == 'hard_mode' else ('train_acc', 'val_acc')
        for key,style,label in [(train_key,'--','Train'),(val_key,'-','Held out')]:
            ax.plot([r['step'] for r in data],[r[key] for r in data],style,
                    color=colors[arm],lw=1.1,alpha=.85 if style=='-' else .6,label=label)
        if row.get('primary_evaluation') == 'hard_mode':
            ax.plot([r['step'] for r in data],[r['val_acc'] for r in data], ':',
                    color=colors[arm], alpha=.45, lw=.8, label='Sampled held out')
        ax.scatter([data[-1]['step']],[data[-1][val_key]],s=22,color=colors[arm],marker='x')
        ax.set_xscale('symlog', linthresh=1000)
        ax.set_xlim(0,budget)
        ax.set_xticks([0,1000,10000,budget],['0','1k','10k',f'{budget//1000}k'])
        ax.set_ylim(0,1.025)
        ax.axhline(.99,ls=':',c='grey',lw=.7)
        ax.set_title(f"{name}: {'99% criterion met' if row['generalise_99_step'] is not None else '99% not reached by budget'}",fontsize=10)
        ax.grid(alpha=.15)
    for ax in axes[-1]: ax.set_xlabel('Optimizer updates (linear to 1k, then log)')
    for ax in axes[:,0]: ax.set_ylabel('Accuracy')
    axes[0,0].legend(loc='lower right',frameon=False)
    policy_note = ('B categorical: hard-mode primary, sampled dotted; × marks final evaluation'
                   if any(row.get('primary_evaluation') == 'hard_mode' for row in rows)
                   else 'Deterministic evaluation; × marks final evaluation')
    fig.suptitle('Frozen p149 comparison: all three fresh seeds\n' + policy_note, fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94))
    for suffix in ('png','pdf'): fig.savefig(output/f'learning_curves.{suffix}',dpi=170)
    plt.close(fig)
    capacity_budget = max(row['config']['max_steps'] for runs in calibration.values() for row in runs)
    fig, axes = plt.subplots(2,2,figsize=(10,6),sharex=True,sharey=True)
    for arm, runs in calibration.items():
        for row in runs:
            config = row['config']
            ax = axes[config['seed']-61, [16384,32768].index(config['n_samples'])]
            data = trace(Path(row['path']))
            data_limited = row['best_bits'] >= .95 * config['n_samples'] * math.log2(config['p'] + 2)
            label = f'{arm} (data limited)' if data_limited else arm
            ax.plot([r['step'] for r in data],[max(r['train_bits'], r.get('train_mode_bits', r['train_bits']))/1000 for r in data],
                    label=label,color=colors[arm],lw=1.2)
    for i, seed in enumerate((61,62)):
        for j, n in enumerate((16384,32768)):
            ax=axes[i,j]
            ax.axhspan(.9*target/1000,1.1*target/1000,color='grey',alpha=.12)
            ax.axhline(target/1000,color='grey',ls=':',lw=.8)
            ax.set_title(f'Seed {seed}, n={n:,}')
            ax.set_xlim(0,capacity_budget)
            ax.grid(alpha=.15)
            ax.legend(frameon=False,loc='lower right')
    for ax in axes[-1]: ax.set_xlabel('Optimizer updates')
    fig.supylabel('Learned label information (kbits)', fontsize=12)
    capacity_note = 'Grey: fixed target ±10%; C is a descriptive control'
    if any(row['config']['model'] == 'categorical' for row in calibration['B']):
        capacity_note += '; categorical memory uses max(sampled, mode)'
    fig.suptitle('Finite-budget capacity measurements\n' + capacity_note, fontsize=12)
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
                params=row['params'],best_bits=row['best_bits'],
                best_sampled_bits=row.get('best_sampled_bits'),best_mode_bits=row.get('best_mode_bits'),
                tail_relative_drift=row['tail_relative_drift'],
                dataset_limited=row['best_bits'] >= .95*row['config']['n_samples']*math.log2(row['config']['p']+2)))
    write_csv(output/'capacity.csv',capacities)
    write_json(output/'delay_definitions.json', {
        'registered_sustained_99': 'delay_steps_99 is held-out 99% minus training 99%, each sustained for three evaluations; signed difference; missing crossings stay null.',
        'registered_large_gap': 'Observed intervals with training >=99% and held-out <90%; not a universal binary definition of grokking.',
        'supplementary_first_passage_99_98': 'max(0, first held-out >=98% time - first training >=99% time); missing crossings stay null; uses each run primary evaluation policy.',
        'supplementary_registration': 'Added after scalar confirmation and before categorical confirmation, for comparison with the paper; does not change selection, stopping, or primary outcomes.',
        'observation_resolution_updates': 100,
        'resolution_note': 'Crossings are measured on recorded evaluations. Sub-100-update delays are unresolved.',
        'paper': 'https://arxiv.org/html/2605.09724v2#S4.SS3',
        'protocol_difference': 'These comparisons use dropout 0; the paper grokking baseline used dropout 0.2 and epoch-level evaluation.'})
    control = read(root/'selection.json').get('control', 'compressed_continuous')
    unrestricted = control in {'unrestricted', 'fullwidth_unrestricted'}
    categorical = calibration['B'][0]['config']['model'] == 'categorical'
    write_json(output/'validation.json',dict(all_nine_complete=True,capacity_match=assessment,
               checkpoint_audits=9,paired_datasets=True,control=control,
               paired_B_C_initial_tensors=(not unrestricted or (control == 'fullwidth_unrestricted' and not categorical)),
               paired_B_C_shared_base_initial_tensors=not categorical,
               scientific_source_hashes=source_hashes(),arithmetic_before_random=gates))
    evidence=output/'evidence'
    for folder in [root/name for name in records]+[Path(row['path']) for runs in calibration.values() for row in runs]+[Path(g) for g in gates]:
        destination=evidence/folder.parent.name/folder.name
        destination.mkdir(parents=True,exist_ok=True)
        for name in ('manifest.json','summary.json','trace.jsonl','audit.json','audit_capacity.json','inherited_run.json'):
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
        if (path/'screen_rejection.json').exists(): item['screen_rejection']=read(path/'screen_rejection.json')
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
