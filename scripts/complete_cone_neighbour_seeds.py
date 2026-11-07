#!/usr/bin/env python3
"""Fill ten structural seed gaps in two preselected neighbour cells on GPU0."""
from pathlib import Path
import concurrent.futures,hashlib,json,shutil,time
import run_cone_h100_boundary as run
ROOT=run.ROOT;OUT=run.OUT
items=[(139,d,s,'neighbour') for d in [220,230] for s in range(47,52)]
if __name__=='__main__':
    archive=json.loads((ROOT/'results/cone_repair_20261104/runs.json').read_text());lookup={(r['p'],r['dim'],r['seed']):r for r in archive}
    refs=[]
    for p,d,s,role in items:
        r=lookup[p,d,s];assert r['host']=='147-224-250-73';dst=OUT/'reference_traces'/(r['uuid']+'.npz')
        shutil.copy2(r['trace_path'],dst);refs.append({**r,'role':role,'reference_trace':str(dst),'trace_sha256':run.sha(dst)})
    amendment={'created_at':time.time(),'reason':'Final cohort check found the two preselected neighbours p139,d220 and d230 have ten archived seeds each. Complete seeds47-51 at both widths so the final figure does not reduce their seed counts. Original225-run manifest, references, completion and analysis are preserved. No cells are added or chosen by outcome.','additional_jobs':10,'items':items,'physical_gpu':0,'workers':2,'original_manifest_sha256':run.sha(OUT/'manifest.json'),'controller_sha256':run.sha(__file__),'reference_runs':refs}
    original=OUT/'original_225';original.mkdir(exist_ok=True)
    for name in ['analysis.json','paired_cells.csv','paired_runs.csv','paired_cell_delays.pdf','paired_cell_delays.png','scored_runs.json','scored_reference_runs.json','cells.json','checkpoint_verification.json']:
        if (OUT/name).exists() and not (original/name).exists():shutil.copy2(OUT/name,original/name)
    path=OUT/'seed_completion_manifest.json'
    if path.exists():assert json.loads(path.read_text())['items']==[list(x) for x in items]
    else:path.write_text(json.dumps(amendment,indent=2)+'\n')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:results=list(ex.map(run.job,items))
    (OUT/'seed_completion.json').write_text(json.dumps(results,indent=2)+'\n')
    if any(r['returncode'] for r in results):raise SystemExit(1)
