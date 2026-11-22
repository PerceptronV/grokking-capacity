#!/usr/bin/env python3
"""Preselected p199 extension: the18 registered p197 widths plus316, ten seeds."""
from pathlib import Path
from grokking_capacity.utils.workflow_paths import package_sources
import argparse,concurrent.futures,json,os,shutil,socket,subprocess,sys,time
import grokking_capacity.experiments.cone_workflow.run_cone_h100_boundary as engine
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'results/cone_p199_h100_20261105'
WIDTHS=[52,64,76,92,108,124,140,156,172,188,204,236,268,300,316,364,428,492,556]
def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['jobs','logs','source']:(OUT/name).mkdir(exist_ok=True)
    paths=[Path(__file__).resolve(),ROOT/'src/grokking_capacity/experiments/cone_workflow/run_cone_h100_boundary.py',*package_sources(ROOT),ROOT/'wallow.toml']
    hashes={str(p.relative_to(ROOT)):engine.sha(p) for p in paths}
    if (OUT/'manifest.json').exists():
        m=json.loads((OUT/'manifest.json').read_text());assert m['source_hashes']==hashes;return
    m={'created_at':time.time(),'host':socket.gethostname(),'prime':199,'widths':WIDTHS,'seeds':list(range(42,52)),'jobs':190,'cells':19,'physical_gpu':0,'workers':2,'source_hashes':hashes,'protocol':{k:v for k,v in engine.PROTOCOL.items() if k not in ['status','host','gpu_type']},'selection':'The entire original18-width p197 grid, plus d316 identified in the preceding discussion, fixed before any p199 outcomes. Ten seeds42-51 at every width. This is a new extension, not an addition to the historical public registration.','stopping':'5000 epochs maximum; stop when online train99 and eval val99 are both reached. Analyse first online train99 and eval val98 with censoring.','analysis_plan':'Include all190 planned runs. Use the unchanged cone scorer/aggregator. Preserve all existing p197 points and all previous replication cohorts. Do not refit timescale laws or redefine positive delay. Report per-seed and cell outcomes plus common-five-seed sensitivity.','base_cohort':'results/cone_repair_20261104/final','base_cohort_hashes':{n:engine.sha(ROOT/'results/cone_repair_20261104/final'/n) for n in ['runs.json','cells.json','summary.json']}}
    (OUT/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    for p in paths:
        dst=OUT/'source'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
def job(item):
    d,s=item;name=f'p199_d{d}_s{s}';cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--dim',str(d),'--seed',str(s)]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','GC_DATA_DIR':str(OUT/'data'),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1','PYTHONPATH':str(ROOT/'src')}
    if (OUT/'jobs'/f'{name}.json').exists():return {'name':name,'returncode':0,'already_completed':True}
    t=time.time()
    with (OUT/'logs'/f'{name}.log').open('a') as f:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    result={'name':name,'returncode':r.returncode,'seconds':time.time()-t};print(json.dumps(result),flush=True);return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true');ap.add_argument('--dim',type=int);ap.add_argument('--seed',type=int);args=ap.parse_args()
    if args.worker:
        engine.OUT=OUT;args.p=199;engine.worker(args)
    else:
        prepare()
        # Alternate widths within each seed, fixed before outcomes.
        items=[(d,s) for s in range(42,52) for d in WIDTHS]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:results=list(ex.map(job,items))
        (OUT/'completion.json').write_text(json.dumps(results,indent=2)+'\n')
        if any(r['returncode'] for r in results):sys.exit(1)
