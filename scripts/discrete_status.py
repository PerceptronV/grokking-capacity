#!/usr/bin/env python3
"""Read experiment status from terminal artifacts and live PID/start identities."""
import argparse
import json
from pathlib import Path


def statuses(root):
    rows = []
    for path in sorted(root.rglob('manifest.json')):
        # Frozen source checkouts and exported evidence also contain manifests;
        # they are copies, not additional experiment workers.
        if {'source-worktrees', 'source', 'evidence'} & set(path.relative_to(root).parts[:-1]):
            continue
        manifest = json.loads(path.read_text())
        directory = path.parent
        process = Path(f'/proc/{manifest["pid"]}/stat')
        try:
            fields = process.read_text().split()
            live = fields[21] == manifest['process_start_ticks'] and fields[2] != 'Z'
        except (FileNotFoundError, ProcessLookupError):
            live = False
        records = []
        trace = directory / 'trace.jsonl'
        if trace.exists():
            for line in trace.read_text().splitlines():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    break
        last = records[-1] if records else {}
        summary = directory / 'summary.json'
        state = ('complete' if summary.exists() else 'running' if live else
                 'rejected/capacity-above-target' if (directory / 'screen_rejection.json').exists() else
                 'stopped/scope-change' if (directory / 'scope_stop.json').exists() else
                 'stopped/incomplete')
        policy = manifest['config'].get('categorical_evaluation', 'sampled') if manifest['config']['model']=='categorical' else 'deterministic'
        primary_train, primary_val = ('train_mode_acc', 'val_mode_acc') if policy=='hard_mode' else ('train_acc','val_acc')
        rows.append(dict(path=str(directory.relative_to(root)), status=state,
                         pid=manifest['pid'], device=manifest['device'],
                         step=last.get('step', 0), max_steps=manifest['config']['max_steps'],
                         primary_evaluation=policy,
                         best_bits=max((max(r['train_bits'], r.get('train_mode_bits', r['train_bits'])) for r in records), default=None),
                         best_sampled_bits=max((r['train_bits'] for r in records), default=None),
                         primary_train_acc=last.get(primary_train), primary_val_acc=last.get(primary_val),
                         train_acc=last.get('train_acc'), val_acc=last.get('val_acc')))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/discrete'))
    parser.add_argument('--active-only', action='store_true')
    parser.add_argument('--json-output', type=Path)
    args = parser.parse_args()
    rows = statuses(args.root)
    if args.json_output:
        args.json_output.write_text(json.dumps(rows, indent=2) + '\n')
    for row in rows:
        if args.active_only and row['status'] != 'running':
            continue
        print(json.dumps(row))
