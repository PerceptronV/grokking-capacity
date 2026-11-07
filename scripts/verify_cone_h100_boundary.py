#!/usr/bin/env python3
"""Independently validate completed boundary runs on CPU while training continues."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,multiprocessing,os,sqlite3,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cone_data import first_events,sha,write_json
CAMPAIGN=ROOT/'results/cone_h100_boundary_20261104'
_CACHE={}
def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def check(job, cpu_threads=1, linear_backend="torch"):
    import torch
    from grokking_capacity.data import grokking_data_torch
    from grokking_capacity.models import TransformerTorch
    torch.set_num_threads(cpu_threads)
    r,meta=job;p=r['p'];seed=r['seed']
    key=p,seed
    if key not in _CACHE:
        np.random.seed(seed);_CACHE[key]=grokking_data_torch(p,op='/',split_type='random',train_fraction=.5,device='cpu')
    data=_CACHE[key];assert [digest(x) for x in data]==meta['dataset_hashes']
    folder=Path(r['npz_path']).parent
    initial=torch.load(folder/'model_init.pt',map_location='cpu',weights_only=True)['model_state_dict']
    final=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)['model_state_dict']
    assert {k:digest(v) for k,v in initial.items()}==meta['initial_state_hashes']
    assert any(not torch.equal(initial[k],v) for k,v in final.items())
    with np.load(folder/'trace.npz',allow_pickle=False) as z:
        e=first_events(z['train_acc'],z['val_acc']);hits=np.flatnonzero(z['val_acc']>=99)
        assert (int(hits[0]+1) if len(hits) else None)==r['grokking_epoch']
        assert len(z['val_acc'])==r['epochs_trained']
    model=TransformerTorch(depth=2,dim=r['dim'],heads=1,n_tokens=p+2,seq_len=4,dropout=.2,init_scale=1.);model.load_state_dict(final);model.eval()
    assert sum(t.numel() for t in model.parameters())==r['param_count']==meta['total_parameters']
    assert linear_backend in ['torch','numpy']
    if linear_backend=='numpy':
        # Independent CPU evaluation only. Training and stored traces are untouched.
        # NumPy's optimized float32 BLAS avoids this system Torch build's reference BLAS.
        import types
        def numpy_linear(layer,x):
            xx=x.detach().numpy()
            yy=xx.reshape(-1,xx.shape[-1]) @ layer.weight.detach().numpy().T
            if layer.bias is not None:yy+=layer.bias.detach().numpy()
            return torch.from_numpy(yy.reshape(*xx.shape[:-1],layer.out_features))
        for layer in model.modules():
            if isinstance(layer,torch.nn.Linear):layer.forward=types.MethodType(numpy_linear,layer)
    X,Y=data[2:];correct=0
    with torch.no_grad():
        for start in range(0,len(Y),512):correct+=(model(X[start:start+512]).argmax(1)==Y[start:start+512]).sum().item()
    acc=100*correct/len(Y);assert abs(acc-r['final_val_acc'])<=100/len(Y)+1e-8,(r['uuid'],acc,r['final_val_acc'])
    return {'cpu_linear_backend':linear_backend,'cpu_threads':cpu_threads,'uuid':r['uuid'],'p':p,'dim':r['dim'],'seed':seed,'checkpoint_sha256':sha(folder/'model.pt'),'trace_sha256':sha(folder/'trace.npz'),'dataset_and_initial_checkpoint_verified':True,'cpu_validation_accuracy':acc,'recorded_validation_accuracy':r['final_val_acc'],**e}
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--watch',action='store_true');args=ap.parse_args()
    out=CAMPAIGN/'checkpoint_checks';out.mkdir(exist_ok=True)
    m=json.loads((CAMPAIGN/'manifest.json').read_text())
    for rel,h in m['source_hashes'].items():assert sha(ROOT/rel)==sha(CAMPAIGN/'source'/rel)==h
    amendment=CAMPAIGN/'seed_completion_manifest.json'
    expected=225+(json.loads(amendment.read_text())['additional_jobs'] if amendment.exists() else 0)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        while True:
            con=sqlite3.connect((CAMPAIGN/'runs.db').as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row;rows=[dict(r) for r in con.execute("select * from runs where status='completed'")];con.close()
            pending=[]
            for r in rows:
                name=f'p{r["p"]}_d{r["dim"]}_s{r["seed"]}';meta_path=CAMPAIGN/'jobs'/(name+'.json')
                if not meta_path.exists() or (out/(r['uuid']+'.json')).exists():continue
                meta=json.loads(meta_path.read_text())
                assert r['host']==meta['host']==m['host'] and r['gpu_type']=='H100'
                for k,v in m['protocol'].items():assert r[k]==v,(r['uuid'],k,r[k],v)
                pending.append((r,meta))
            for value in pool.map(check,pending):
                write_json(out/(value['uuid']+'.json'),value);print(json.dumps({'checked':value['uuid'],'p':value['p'],'dim':value['dim'],'seed':value['seed']}),flush=True)
            count=len(list(out.glob('*.json')))
            if count==expected:
                write_json(CAMPAIGN/'checkpoint_verification.json',{'runs_verified':expected,'cpu_only':True,'checks':[json.loads(p.read_text()) for p in sorted(out.glob('*.json'))]});break
            if not args.watch:print(f'{count}/{expected} checked; training still incomplete',flush=True);break
            if (CAMPAIGN/'completion.json').exists():
                assert all(x['returncode']==0 for x in json.loads((CAMPAIGN/'completion.json').read_text()))
            if (CAMPAIGN/'seed_completion.json').exists():assert all(x['returncode']==0 for x in json.loads((CAMPAIGN/'seed_completion.json').read_text()))
            time.sleep(10)
if __name__=='__main__':main()
