#!/usr/bin/env python3
"""Thirty paired norm-gain runs on GPU 0, with immutable source/data provenance."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,os,socket,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/normfrozen_h100_paired_20261103'
SETTINGS=[(128,.1),(128,.3),(128,1.),(64,1.),(256,1.)]
SEEDS=[42,43,44]
def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def worker(args):
    import torch
    sys.path.insert(0,str(ROOT/'src'))
    import grokking_capacity.experiments.groks as g
    torch.set_num_threads(1)
    name=f'd{args.dim}_wd{args.wd:g}_s{args.seed}_{args.arm}'
    meta={'arm':args.arm,'freeze_norm_scales':args.arm=='frozen','dim':args.dim,'weight_decay':args.wd,'seed':args.seed,'host':socket.gethostname(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'gpu_uuid':str(torch.cuda.get_device_properties(0).uuid),'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,'dtype':str(torch.get_default_dtype()),'started_at':time.time()}
    original_data=g.grokking_data_torch
    def capture_data(*a,**kw):
        result=original_data(*a,**kw);meta['dataset_hashes']=[digest(x) for x in result];return result
    g.grokking_data_torch=capture_data
    original_train=g.GrokkingTrainer.train
    def capture_train(self,*a,**kw):
        meta['initial_state_hashes']={k:digest(v) for k,v in self.model.state_dict().items()}
        meta['frozen_parameters']=[k for k,v in self.model.named_parameters() if not v.requires_grad]
        meta['total_parameters']=sum(v.numel() for v in self.model.parameters())
        meta['trainable_parameters']=sum(v.numel() for v in self.model.parameters() if v.requires_grad)
        meta['cpu_rng_hash']=digest(torch.get_rng_state());meta['cuda_rng_hash']=digest(torch.cuda.get_rng_state())
        assert bool(meta['frozen_parameters'])==(args.arm=='frozen')
        if args.arm=='frozen':assert all('norm.weight' in k for k in meta['frozen_parameters'])
        result=original_train(self,*a,**kw)
        meta['frozen_parameters_unchanged']=all(digest(self.model.state_dict()[k])==meta['initial_state_hashes'][k] for k in meta['frozen_parameters'])
        assert meta['frozen_parameters_unchanged']
        return result
    g.GrokkingTrainer.train=capture_train
    cli=['groks','--p','113','--dim',str(args.dim),'--seed',str(args.seed),'--operation','/','--train-fraction','.5','--split-type','random','--depth','2','--heads','1','--dropout','.2','--init-scale','1','--lr','.001','--weight-decay',str(args.wd),'--beta1','.9','--beta2','.98','--batch-size','512','--epochs','5000','--mup-base','0','--early-stopping-train','.99','--early-stopping-val','.99','--patience','-1','--min-delta','.0001','--ignore-memorisation','--norm-log-every','0','--post-grok-epochs','0','--save-model','--device','cuda:0','--db-path',str(OUT/(args.arm+'.db'))]
    if args.arm=='frozen':cli.append('--freeze-norm-scales')
    meta['cli']=cli;sys.argv=cli;g.main()
    if 'dataset_hashes' not in meta:
        # A completed, already-audited run was skipped on resume.
        assert (OUT/'jobs'/f'{name}.json').exists();return
    meta['finished_at']=time.time();(OUT/'jobs'/f'{name}.json').write_text(json.dumps(meta,indent=2)+'\n')
def job(item):
    d,w,s,arm=item;name=f'd{d}_wd{w:g}_s{s}_{arm}'
    cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--dim',str(d),'--wd',str(w),'--seed',str(s),'--arm',arm]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','GC_DATA_DIR':str(OUT/'data'),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1','PYTHONPATH':str(ROOT/'src')}
    t=time.time()
    with (OUT/'logs'/f'{name}.log').open('a') as f:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    out=dict(name=name,returncode=r.returncode,seconds=time.time()-t);print(json.dumps(out),flush=True);return out
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true');ap.add_argument('--dim',type=int);ap.add_argument('--wd',type=float);ap.add_argument('--seed',type=int);ap.add_argument('--arm',choices=['frozen','unfrozen']);args=ap.parse_args()
    if args.worker:worker(args)
    else:
        OUT.mkdir(parents=True,exist_ok=True)
        for name in ['logs','jobs','source']:(OUT/name).mkdir(exist_ok=True)
        paths=[Path(__file__).resolve(),*sorted((ROOT/'src/grokking_capacity').rglob('*.py')),ROOT/'wallow.toml']
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        manifest={'settings':SETTINGS,'seeds':SEEDS,'arms':['frozen','unfrozen'],'p':113,'protocol':'modular division, train fraction 0.5, depth 2, heads 1, dropout .2, init scale 1, AdamW lr .001 betas .9/.98, batch 512, 5000 epochs, first validation99; stop once training99 and validation99, no patience, no muP','host':socket.gethostname(),'physical_gpu':0,'source_hashes':hashes,'jobs':30,'started_at':time.time(),'estimator':'Per arm, OLS log(mean first-val99 epoch over identical seeds) on log(weight_decay) at d128; width slope versus common total architecture parameter count at wd1. Paired seed bootstrap uses the same resample across both arms and all settings. Censored runs retained and flagged; no slope silently drops censored outcomes.'}
        if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())['source_hashes']==hashes,'Source changed during campaign'
        else:
            (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            for p in paths:
                dst=OUT/'source'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
        items=[(d,w,s,arm) for d,w in SETTINGS for s in SEEDS for arm in ['unfrozen','frozen']]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:result=list(ex.map(job,items))
        (OUT/'completion.json').write_text(json.dumps(result,indent=2)+'\n')
        if any(r['returncode'] for r in result):sys.exit(1)
