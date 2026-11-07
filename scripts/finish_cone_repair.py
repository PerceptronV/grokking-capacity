#!/usr/bin/env python3
"""Wait for the full audited replication, publish its cone, and rebuild the paper."""
from pathlib import Path
import json,os,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];CAMPAIGN=ROOT/'results/cone_h100_boundary_20261104';OUT=ROOT/'results/cone_repair_20261104';PAPER=ROOT/'docs/papers/2026 Grokking Capacity'
def main():
    amendment=CAMPAIGN/'seed_completion_manifest.json'
    expected=225+(json.loads(amendment.read_text())['additional_jobs'] if amendment.exists() else 0)
    def ready():
        required=['completion.json','checkpoint_verification.json','code_equivalence.json']+(['seed_completion.json'] if amendment.exists() else [])
        return all((CAMPAIGN/name).exists() for name in required) and json.loads((CAMPAIGN/'checkpoint_verification.json').read_text())['runs_verified']==expected
    while not ready():
        if (CAMPAIGN/'completion.json').exists():assert all(r['returncode']==0 for r in json.loads((CAMPAIGN/'completion.json').read_text()))
        time.sleep(10)
    assert json.loads((CAMPAIGN/'code_equivalence.json').read_text())['model_and_plain_adamw_equivalent_in_current_runtime']
    subprocess.run([sys.executable,'scripts/analyse_cone_h100_boundary.py'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'scripts/update_cone_paper.py','--analysis',str(OUT/'final'),'--final'],cwd=ROOT,check=True)
    build=OUT/'final/build';build.mkdir(exist_ok=True)
    cmd=['pdflatex','-interaction=nonstopmode','-halt-on-error',f'-output-directory={build}','paper-neurips-revised.tex']
    for i in range(1,4):
        with (build/f'pass{i}.log').open('w') as f:subprocess.run(cmd,cwd=PAPER,stdout=f,stderr=subprocess.STDOUT,check=True)
        if i==1:
            with (build/'bibtex.log').open('w') as f:subprocess.run(['bibtex','paper-neurips-revised'],cwd=build,env={**os.environ,'BIBINPUTS':f'{PAPER}:','BSTINPUTS':f'{PAPER}:'},stdout=f,stderr=subprocess.STDOUT,check=True)
    shutil.copy2(build/'paper-neurips-revised.pdf',PAPER/'paper-neurips-revised.pdf')
    print('Full cone replication published and paper rebuilt.',flush=True)
if __name__=='__main__':main()
