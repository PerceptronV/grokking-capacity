#!/usr/bin/env python3
"""Build the revised paper from arithmetic-cell-mean analyses.

Run: python scripts/build_revised_paper.py
Use --reuse-analyses to rebuild from the existing validated analysis outputs.
Publishing updates generated tables/figures while preserving current author prose.
After changing analyses, reconcile numbers in running prose before publication.
Requires local run registries/traces, Python scientific dependencies and TeX Live.
"""
from pathlib import Path
import argparse,json,os,subprocess,sys,shutil
ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'docs/papers/2026 Grokking Capacity'
REPORT=ROOT/'reports/paper_corrections_20261107'
BUILD=REPORT/'build'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reuse-analyses',action='store_true');args=parser.parse_args()
    BUILD.mkdir(parents=True,exist_ok=True)
    stages=[]
    def run(name,cmd,cwd=ROOT,env=None):
        with (BUILD/f'{name}.log').open('w') as f:
            p=subprocess.run(cmd,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,env=env)
        stages.append({'step':name,'exit_code':p.returncode,'command':cmd})
        (BUILD/'result.json').write_text(json.dumps({'commands':stages},indent=2)+'\n')
        print(name,p.returncode,flush=True)
        if p.returncode:raise SystemExit(f'{name} failed; see {BUILD/name}.log')
    if not args.reuse_analyses:
        for name in ['recompute_paper_crossings','recompute_paper_statistics','prepare_paper_revision_models','prepare_paper_revision_supplement']:
            run(name,[sys.executable,str(ROOT/'scripts'/f'{name}.py')])
    run('publish',[sys.executable,str(ROOT/'scripts/publish_paper_revision.py')])
    latex=['pdflatex','-interaction=nonstopmode','-halt-on-error','-file-line-error',f'-output-directory={BUILD}','paper-neurips-revised.tex']
    run('pass1',latex,PAPER)
    env=os.environ.copy();env['BIBINPUTS']=str(PAPER)+os.pathsep+env.get('BIBINPUTS','');env['BSTINPUTS']=str(PAPER)+os.pathsep+env.get('BSTINPUTS','')
    run('bibtex',['bibtex','paper-neurips-revised'],BUILD,env)
    run('pass2',latex,PAPER);run('pass3',latex,PAPER)
    log=(BUILD/'paper-neurips-revised.log').read_text()
    errors=[line for line in log.splitlines() if ('undefined' in line.lower() or 'multiply defined' in line.lower()) and 'Warning' in line]
    if errors:raise SystemExit('Unresolved references: '+str(errors))
    for dest in [PAPER/'paper-neurips-revised.pdf',PAPER/'exports/paper-neurips-revised.pdf']:
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BUILD/'paper-neurips-revised.pdf',dest)
    run('publication_validation',[sys.executable,str(ROOT/'scripts/validate_revised_paper.py')])
    print('Built and numerically validated revised paper; visual review is recorded separately.',flush=True)


if __name__=='__main__':main()
