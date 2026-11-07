#!/usr/bin/env python3
"""Fresh six-width matched-protocol capacity campaign; resumable, GPU 0 only."""
from pathlib import Path
import concurrent.futures, hashlib, json, os, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/capacity_h100_matched_20261103'
DIMS=[10,12,14,16,18,22]
NS=[1000,1374,1891,2601,3576,4917,6761,9300]
SEEDS=[20261103,20261104]
def job(item):
 d,n,s=item
 cmd=[sys.executable,'-m','grokking_capacity.experiments.capacity','--p','113','--dim',str(d),'--n-samples',str(n),'--seed',str(s),'--depth','2','--heads','1','--dropout','0.2','--weight-decay','1.0','--lr','0.001','--beta1','0.9','--beta2','0.98','--batch-size','512','--epochs','5000','--patience','100','--min-delta','0.0001','--device','cuda:0','--save-model','--db-path',str(OUT/'runs.db')]
 env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONPATH':str(ROOT/'src'),'PYTHONUNBUFFERED':'1'}
 t=time.time()
 with (OUT/'logs'/f'd{d}_n{n}_s{s}.log').open('a') as f:
  r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
 result=dict(dim=d,n_samples=n,seed=s,returncode=r.returncode,seconds=time.time()-t)
 print(json.dumps(result),flush=True)
 return result
if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
 import torch
 manifest=dict(started_at=time.time(),dims=DIMS,n_samples=NS,seeds=SEEDS,p=113,depth=2,heads=1,dropout=.2,weight_decay=1.,lr=.001,betas=[.9,.98],batch_size=512,epochs=5000,patience=100,min_delta=1e-4,init_scale=1.,mup_base=0,dtype='float32',torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/grokking_capacity').rglob('*.py')},estimator='Affine OLS of mean of per-seed maxima over the fixed dataset-size grid versus parameter count; report individual-seed slopes and width/seed bootstrap.',jobs=96)
 if not (OUT/'manifest.json').exists(): (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
 # Different widths in flight; smaller datasets first.
 items=[(d,n,s) for s in SEEDS for n in NS for d in DIMS]
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex: results=list(ex.map(job,items))
 (OUT/'completion.json').write_text(json.dumps(results,indent=2))
 if any(x['returncode'] for x in results):sys.exit(1)
