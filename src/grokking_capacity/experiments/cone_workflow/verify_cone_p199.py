#!/usr/bin/env python3
"""Verify every p199 run against its raw trace, initialization, data and CPU evaluation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import argparse,concurrent.futures,functools,json,multiprocessing,sqlite3,time
from grokking_capacity.experiments.cone_workflow.cone_data import sha, write_json
from grokking_capacity.experiments.cone_workflow.verify_cone_h100_boundary import check
ROOT=Path(__file__).resolve().parents[4];OUT=ROOT/'results/cone_p199_h100_20261105'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');args=ap.parse_args()
    m=json.loads((OUT/'manifest.json').read_text());dest=OUT/'checkpoint_checks';dest.mkdir(exist_ok=True)
    for rel,h in m['source_hashes'].items():assert sha(ROOT/rel)==sha(OUT/'source'/rel)==h
    expected={(199,d,s) for d in m['widths'] for s in m['seeds']}
    with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
        while True:
            rows=[]
            if (OUT/'runs.db').exists():
                c=sqlite3.connect((OUT/'runs.db').as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
                rows=[dict(r) for r in c.execute("select * from runs where status='completed'")];c.close()
            pending=[]
            for r in rows:
                assert (r['p'],r['dim'],r['seed']) in expected
                meta=OUT/'jobs'/f'p199_d{r["dim"]}_s{r["seed"]}.json'
                if not meta.exists() or (dest/(r['uuid']+'.json')).exists():continue
                meta=json.loads(meta.read_text());assert r['host']==meta['host']==m['host'] and r['gpu_type']=='H100'
                for k,v in m['protocol'].items():assert r[k]==v,(r['uuid'],k,r[k],v)
                pending.append((r,meta))
            for value in pool.map(functools.partial(check,cpu_threads=1,linear_backend='numpy'),pending):
                write_json(dest/(value['uuid']+'.json'),value)
                print(json.dumps({'checked':value['uuid'],'dim':value['dim'],'seed':value['seed']}),flush=True)
            values=[json.loads(p.read_text()) for p in sorted(dest.glob('*.json'))]
            if len(values)==m['jobs']:
                assert {(r['p'],r['dim'],r['seed']) for r in values}==expected
                write_json(OUT/'checkpoint_verification.json',{'runs_verified':m['jobs'],'cpu_only':True,'checks':values});break
            if not args.watch:print(f'{len(values)}/{m["jobs"]} verified',flush=True);break
            if (OUT/'completion.json').exists():assert all(r['returncode']==0 for r in json.loads((OUT/'completion.json').read_text()))
            time.sleep(10)
if __name__=='__main__':main()
