#!/usr/bin/env python3
"""Wait for all p199 training and CPU checks, then publish the complete extension."""
from pathlib import Path
import json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/cone_p199_h100_20261105'
while True:
    finished=OUT/'completion.json';verified=OUT/'checkpoint_verification.json'
    if finished.exists():assert all(r['returncode']==0 for r in json.loads(finished.read_text()))
    if finished.exists() and verified.exists():
        assert json.loads(verified.read_text())['runs_verified']==190
        break
    time.sleep(10)
for name in ['analyse_cone_p199.py','publish_cone_p199.py']:
    subprocess.run([sys.executable,str(ROOT/'scripts'/name)],cwd=ROOT,check=True)
print('Complete p199 cone extension published.',flush=True)
