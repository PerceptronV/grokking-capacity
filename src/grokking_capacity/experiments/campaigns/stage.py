"""Run a registered stage from an explicit job list."""
import argparse
import importlib
from pathlib import Path
from .config import initialize, check_registration
from .runner import run_stage
from ..registered import ROOT, read


def main(argv=None, *, protocol):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--jobs', required=True, type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--devices', help='Comma-separated device slots; repeated devices allow concurrent jobs')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    stem = 'dropout_match' if protocol == 'matched_dropout' else protocol
    root = (args.root or ROOT / 'results/discrete' / ('p97-' + stem.replace('_', '-') + '-v2')).resolve()
    config = args.config or ROOT / 'configs/campaigns' / (stem + '.yaml')
    worker = importlib.import_module('grokking_capacity.experiments.' + protocol)
    reg = initialize(root, config, worker.source_hashes(), resume=root.exists())
    devices = args.devices.split(',') if args.devices else reg['resources']['devices']
    return run_stage(root, args.stage, read(args.jobs), devices, worker.__name__, resume=args.resume,
                     source_check=lambda: check_registration(root, worker.source_hashes()))
