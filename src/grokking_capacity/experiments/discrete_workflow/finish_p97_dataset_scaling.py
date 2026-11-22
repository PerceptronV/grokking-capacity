#!/usr/bin/env python3
"""Generate and independently verify the report after the frozen campaign ends."""

from grokking_capacity.utils.workflow_paths import script_source
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[4]
RUN=ROOT/'results/discrete/p97-dataset-scaling-v1'
sys.path.insert(0,str(ROOT/'src'))
from grokking_capacity.experiments.dataset_scaling import atomic_json,sha


def main():
    while not (RUN/'campaign_complete.json').exists():
        if (RUN/'campaign_error.json').exists():
            raise RuntimeError((RUN/'campaign_error.json').read_text())
        time.sleep(30)
    for script in ['report_p97_dataset_scaling.py','verify_p97_dataset_scaling.py']:
        subprocess.run([sys.executable,'-u',str(script_source(script, ROOT))],cwd=ROOT,check=True)
    out=ROOT/'reports/p97_dataset_scaling'
    provenance=dict(completed_at=time.time(),report_source_sha256=sha(script_source('report_p97_dataset_scaling.py', ROOT)),
        verifier_source_sha256=sha(ROOT/'src/grokking_capacity/experiments/discrete_workflow/verify_p97_dataset_scaling.py'),
        report_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='completion_provenance.json'})
    atomic_json(out/'completion_provenance.json',provenance)
    atomic_json(RUN/'pipeline_complete.json',dict(completed_at=time.time(),report_directory=str(out),all_passed=True))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        atomic_json(RUN/'pipeline_error.json',dict(time=time.time(),error=repr(exc)))
        raise
