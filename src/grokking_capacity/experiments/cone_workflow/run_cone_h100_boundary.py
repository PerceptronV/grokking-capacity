#!/usr/bin/env python3
"""Preselected 225-run cone boundary replication, two workers on one H100."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,os,socket,sqlite3,subprocess,sys,time,shutil
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'results/cone_h100_boundary_20261104'
GROUPS=[(97,56,[54,56,58],10),(107,64,[62,64,66],10),(107,80,[78,80,82],10),(113,64,[62,64,66],10),(127,80,[78,80,82],10),(139,96,[94,96,98],10),(139,112,[110,112,114],10),(139,224,[220,224,230],5)]
PROTOCOL={'experiment_type':'groks','status':'completed','host':'147-224-250-73','gpu_type':'A100','operation':'/','train_fraction':.5,'split_type':'random','dataset_type':'modular','depth':2,'heads':1,'dropout':.2,'init_scale':1.,'lr':.001,'weight_decay':1.,'beta1':.9,'beta2':.98,'batch_size':512,'max_epochs':5000,'mup_base':0,'architecture_family':'transformer_gated'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def items():return [(p,d,s,'target' if d==target else 'neighbour') for p,target,widths,n in GROUPS for d in widths for s in range(42,42+n)]
def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['logs','jobs','source','reference_traces']:(OUT/name).mkdir(exist_ok=True)
    paths=[Path(__file__).resolve(),*sorted((ROOT/'src/grokking_capacity').rglob('*.py')),ROOT/'wallow.toml']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    if (OUT/'manifest.json').exists():
        m=json.loads((OUT/'manifest.json').read_text());assert m['source_hashes']==hashes and m['groups']==[list(x[:2])+[x[2],x[3]] for x in GROUPS];return
    c=sqlite3.connect((ROOT/'runs.db').as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
    rows=[dict(r) for r in c.execute('select * from runs where '+' and '.join(k+'=?' for k in PROTOCOL),list(PROTOCOL.values()))];c.close()
    lookup={}
    for r in rows:
        key=(r['p'],r['dim'],r['seed']);assert key not in lookup;lookup[key]=r
    refs=[]
    for p,d,s,role in items():
        r=lookup[p,d,s];path=Path(r['npz_path']);path=path if path.exists() else ROOT/'data/groks'/r['uuid']/'trace.npz'
        dst=OUT/'reference_traces'/(r['uuid']+'.npz');shutil.copy2(path,dst)
        refs.append({**r,'role':role,'reference_trace':str(dst),'trace_sha256':sha(dst)})
    assert len(refs)==225
    (OUT/'reference_runs.json').write_text(json.dumps(refs,indent=2)+'\n')
    m={'groups':GROUPS,'jobs':225,'cells':24,'target_jobs':75,'neighbour_jobs':150,'seeds':'42-51 except p139 target224 and neighbours220/230 use42-46','physical_gpu':0,'workers':2,'host':socket.gethostname(),'created_at':time.time(),'source_hashes':hashes,'reference_runs_sha256':sha(OUT/'reference_runs.json'),'protocol':{k:v for k,v in PROTOCOL.items() if k not in ['host','gpu_type','status']},'stopping':'Stop after online train99 and eval val99 are both reached; 5000-epoch cap. Analyse first online train99 and first eval val98, with censoring retained.','analysis_plan':'Compare all preselected cells and all identical seed IDs against archived A100 controls. Report every positive/zero/unresolved classification and delay difference. Keep replication separate from archive cohort; do not pool hardware or select outcomes. Initialisation/RNG/data hashes and source snapshot recorded before training.'}
    (OUT/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    for p in paths:
        dst=OUT/'source'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
def worker(args):
    import torch
    sys.path.insert(0,str(ROOT/'src'))
    import grokking_capacity.experiments.groks as g
    torch.set_num_threads(1)
    assert torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name(0)
    name=f'p{args.p}_d{args.dim}_s{args.seed}'
    meta={'p':args.p,'dim':args.dim,'seed':args.seed,'host':socket.gethostname(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'gpu_uuid':str(torch.cuda.get_device_properties(0).uuid),'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,'dtype':str(torch.get_default_dtype()),'started_at':time.time()}
    original_data=g.grokking_data_torch
    def capture_data(*a,**kw):
        result=original_data(*a,**kw);meta['dataset_hashes']=[digest(x) for x in result];return result
    g.grokking_data_torch=capture_data
    original_train=g.GrokkingTrainer.train
    def capture_train(self,*a,**kw):
        meta['initial_state_hashes']={k:digest(v) for k,v in self.model.state_dict().items()}
        meta['total_parameters']=sum(v.numel() for v in self.model.parameters())
        assert all(v.requires_grad for v in self.model.parameters())
        meta['cpu_rng_hash']=digest(torch.get_rng_state());meta['cuda_rng_hash']=digest(torch.cuda.get_rng_state())
        return original_train(self,*a,**kw)
    g.GrokkingTrainer.train=capture_train
    cli=['groks','--p',str(args.p),'--dim',str(args.dim),'--seed',str(args.seed),'--operation','/','--train-fraction','.5','--split-type','random','--depth','2','--heads','1','--dropout','.2','--init-scale','1','--lr','.001','--weight-decay','1','--beta1','.9','--beta2','.98','--batch-size','512','--epochs','5000','--mup-base','0','--early-stopping-train','.99','--early-stopping-val','.99','--patience','-1','--min-delta','.0001','--ignore-memorisation','--norm-log-every','0','--post-grok-epochs','0','--save-model','--device','cuda:0','--db-path',str(OUT/'runs.db')]
    meta['cli']=cli;sys.argv=cli;g.main()
    if 'dataset_hashes' not in meta:assert (OUT/'jobs'/f'{name}.json').exists();return
    meta['finished_at']=time.time();(OUT/'jobs'/f'{name}.json').write_text(json.dumps(meta,indent=2)+'\n')
def job(item):
    p,d,s,role=item;name=f'p{p}_d{d}_s{s}'
    cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--p',str(p),'--dim',str(d),'--seed',str(s)]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','GC_DATA_DIR':str(OUT/'data'),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1','PYTHONPATH':str(ROOT/'src')}
    t=time.time()
    with (OUT/'logs'/f'{name}.log').open('a') as f:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    result={'name':name,'role':role,'returncode':r.returncode,'seconds':time.time()-t};print(json.dumps(result),flush=True);return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true');ap.add_argument('--p',type=int);ap.add_argument('--dim',type=int);ap.add_argument('--seed',type=int);args=ap.parse_args()
    if args.worker:worker(args)
    else:
        prepare()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:results=list(ex.map(job,items()))
        (OUT/'completion.json').write_text(json.dumps(results,indent=2)+'\n')
        if any(r['returncode'] for r in results):sys.exit(1)
