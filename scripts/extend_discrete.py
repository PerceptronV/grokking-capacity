#!/usr/bin/env python3
"""Fork a completed fixed-rate run into an auditable longer-budget continuation.

Only max_steps changes. The original run is immutable. Model, optimiser, random
states, batch position, and trace are copied exactly; the parent checkpoint hash
and full original scientific declaration are retained. This does not revise the
parent's original budget or pretend that the extension was pre-registered then.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch

from grokking_capacity.experiments.discrete import atomic_json, run, source_hashes


def extend(parent, output, max_steps, device):
    if output.exists():
        raise FileExistsError(f'Extension output already exists: {output}')
    manifest = json.loads((parent / 'manifest.json').read_text())
    summary = json.loads((parent / 'summary.json').read_text())
    raw = (parent / 'checkpoint.pt').read_bytes()
    saved = torch.load(parent / 'checkpoint.pt', map_location='cpu', weights_only=False)
    old = saved['scientific']
    assert old['config'] == manifest['config'] == summary['config']
    assert old['source_hashes'] == manifest['source_hashes'] == source_hashes()
    assert saved['step'] == summary['final']['step']
    assert saved['trace'] == [json.loads(x) for x in (parent / 'trace.jsonl').read_text().splitlines()]
    if old['config']['task'] != 'random' or old['config']['stop_on_success']:
        raise ValueError('Only completed fixed-budget random-label calibration runs can be extended')
    if saved['step'] != old['config']['max_steps'] or max_steps <= saved['step']:
        raise ValueError('Parent must have completed its budget; new budget must be greater')
    config = {**old['config'], 'max_steps': max_steps}
    scientific = {**copy.deepcopy(old), 'config': config}
    provenance = dict(parent=str(parent.resolve()),
                      parent_checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
                      parent_manifest_sha256=hashlib.sha256((parent / 'manifest.json').read_bytes()).hexdigest(),
                      parent_summary_sha256=hashlib.sha256((parent / 'summary.json').read_bytes()).hexdigest(),
                      parent_scientific=old, parent_step=saved['step'],
                      changes={'max_steps': [old['config']['max_steps'], max_steps]},
                      extension_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.mkdir(parents=True)
    atomic_json(output / 'extension.json', provenance)
    saved['parent_provenance'] = provenance
    saved['scientific'] = scientific
    torch.save(saved, output / 'checkpoint.pt')
    shutil.copy2(parent / 'trace.jsonl', output / 'trace.jsonl')
    shutil.copytree(parent / 'source', output / 'source')
    # The prepared continuation has not started a worker yet. run(...resume=True)
    # claims it with this process's actual PID/start identity after restoring RNG.
    manifest = {**manifest, **scientific, 'pid': 0,
                'process_start_ticks': 'prepared_extension', 'extension': provenance,
                'device': str(device),
                'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                'start_time': time.time()}
    atomic_json(output / 'manifest.json', manifest)
    gate = Path(old['gate']['path'])
    run(config, output, device, resume=True, gate=gate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-steps', type=int, required=True)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    torch.set_num_threads(2)
    extend(args.parent, args.output, args.max_steps, torch.device(args.device))
