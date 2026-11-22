"""Read campaign progress using registered jobs and recorded optimizer steps."""
import argparse
import json
from pathlib import Path
from ..registered import read


def read_trace(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return rows


def status(root):
    root = Path(root)
    paths = list((root / 'stages').glob('*/controller.json'))
    if (root / 'controller.json').exists():
        paths.append(root / 'controller.json')
    states = [(p, read(p)) for p in paths]
    registry_path = root / 'job_registry.json'
    registry = read(registry_path) if registry_path.exists() else {
        j['name']: j['config'] for _, s in states for j in s.get('jobs', [])}
    completed = []
    for name, config in registry.items():
        path = root / 'jobs' / name
        if not (path / 'summary.json').exists():
            continue
        summary = read(path / 'summary.json')
        trace = read_trace(path / 'trace.jsonl')
        completed.append(dict(name=name, config=config, first99=next(
            (r['step'] for r in trace if r.get('val_acc', 0) >= .99), None),
            learned=summary.get('learning_gate_passed'), epochs=summary.get('epochs')))
    groups = []
    keys = sorted({(c.get('n_train', c.get('dim', 0)), c.get('role', '')) for c in registry.values()})
    for size, role in keys:
        def matches(c):
            return (c.get('n_train', c.get('dim', 0)), c.get('role', '')) == (size, role)
        expected = sum(matches(c) for c in registry.values())
        done = [r for r in completed if matches(r['config'])]
        times = [r['first99'] for r in done if r['first99'] is not None]
        groups.append(dict(size=size, role=role, expected=expected, completed=len(done),
            first99_count=len(times), mean_first99=sum(times)/expected if len(times)==expected else None))
    latest = max(states, key=lambda pair: pair[1].get('started_at', 0)) if states else None
    running = latest[1].get('running', []) if latest else []
    live = []
    for job in running:
        rows = read_trace(root / 'jobs' / job['name'] / 'trace.jsonl')
        live.append(dict(name=job['name'], last=rows[-1] if rows else None))
    return dict(phase=latest[1]['phase'] if latest else 'initializing',
        stage=latest[0].parent.name if latest else None, expected=len(registry),
        completed=len(completed), groups=groups, live=live,
        errors=latest[1].get('errors', []) if latest else [],
        pending=latest[1].get('pending', []) if latest else [])


def main(argv=None, *, default_root=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=default_root, required=default_root is None)
    args = parser.parse_args(argv)
    print(json.dumps(status(args.root), indent=2))


if __name__ == '__main__':
    main()
