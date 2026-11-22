"""Registered grids for the existing capacity, speed, and grokking CLIs."""
import argparse
import itertools
import json
import os
from pathlib import Path
from grokking_capacity.utils.workflow_paths import package_sources
import subprocess
import sys

from .config import initialize, check_registration
from .runner import run_stage
from ..registered import ROOT, read, sha, atomic_json

MODULES = {name: 'grokking_capacity.experiments.' + name for name in ('capacity', 'speed', 'groks')}


def jobs(config):
    result = []
    for group, spec in config['groups'].items():
        kind = spec['kind']
        if kind not in MODULES:
            raise ValueError(f'Unknown experiment kind: {kind}')
        grid = spec['grid']
        for values in itertools.product(*grid.values()):
            options = {**config['common'], **spec.get('options', {}), **dict(zip(grid, values))}
            name = group + ''.join(f'_{key}{value}' for key, value in zip(grid, values))
            result.append(dict(name=name, config=dict(kind=kind, options=options)))
    return result


def command(config, output, device):
    cmd = [sys.executable, '-u', '-m', MODULES[config['kind']]]
    for key, value in config['options'].items():
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    return cmd + ['--device', device, '--db-path', str(output / 'runs.db')]


def worker(config, output, device):
    output = Path(output)
    root = output.resolve().parent.parent
    check_registration(root, source_hashes())
    if read(root/'job_registry.json')[output.name] != config:
        raise ValueError('Job does not match the registered grid')
    output.mkdir(parents=True, exist_ok=False)
    cmd = command(config, output, device)
    env = {**os.environ, 'GC_DATA_DIR': str(output.resolve() / 'data')}
    atomic_json(output/'manifest.json', dict(config=config, command=cmd, source_hashes=source_hashes()))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    atomic_json(output/'summary.json', dict(config=config, command=cmd, exit_code=0))


def source_hashes():
    # These general CLI workers depend on the registry and data layers too.
    return {str(p.relative_to(ROOT)): sha(p) for p in package_sources(ROOT)
            if 'frozen' not in p.parts}


def main(argv=None, *, default_config=None, default_root=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=default_config, required=default_config is None)
    parser.add_argument('--root', type=Path, default=default_root, required=default_root is None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    import yaml
    config = yaml.safe_load(args.config.read_text())
    specs = jobs(config)
    if args.dry_run:
        print(json.dumps(specs, indent=2))
        return
    root = args.root.resolve()
    initialize(root, args.config, source_hashes(), resume=args.resume)
    paths = run_stage(root, 'grid', specs, config['resources']['devices'], __name__, resume=args.resume,
        source_check=lambda: check_registration(root, source_hashes()),
        environment=config['resources'].get('environment', {}))
    atomic_json(root/'campaign_complete.json', dict(runs=len(paths), all_passed=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execute one registered CLI job')
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    worker(read(args.config), args.output, args.device)
