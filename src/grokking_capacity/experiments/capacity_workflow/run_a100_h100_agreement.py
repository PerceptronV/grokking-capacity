#!/usr/bin/env python3
"""Preselected paired A100-to-H100 grokking-outcome replication."""
from pathlib import Path
from grokking_capacity.utils.workflow_paths import frozen_source, package_sources
from collections import Counter,defaultdict
import argparse,concurrent.futures,csv,hashlib,json,multiprocessing,os,shutil,socket,sqlite3,subprocess,sys,threading,time

ROOT=Path(os.environ.get('GC_AGREEMENT_ROOT',Path(__file__).resolve().parents[4]))
OUT=ROOT/'results/a100_h100_grokking_agreement_20261106'
PRIMES=[97,107,113,127,139]
PIPE=Path('/tmp/a100_h100_grokking_agreement_20261106')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    p=Path(p);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def read_rows():
    con=sqlite3.connect((OUT/'runs.db').as_uri()+'?mode=ro',uri=True,timeout=120);con.row_factory=sqlite3.Row
    rows=[dict(r) for r in con.execute("select * from runs where status='completed'")];con.close();return rows


def prepare():
    for folder in ['logs','jobs','source','reference_traces','checkpoint_checks','mps_logs','launch_provenance']:(OUT/folder).mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():
        print((OUT/'manifest.json').read_text());return
    from grokking_capacity.experiments.cone_workflow.run_cone_h100_boundary import PROTOCOL
    base=ROOT/'results/cone_ten_seeds_h100_20261106/figure'
    cells=json.loads((base/'cells.json').read_text());runs=json.loads((base/'runs.json').read_text())
    selected=[]
    for p in PRIMES:
        cc=sorted([c for c in cells if c['p']==p and c['gpu_type']=='A100'],key=lambda c:c['dim'])
        first=next(c for c in cc if c['state']=='positive')
        zero=max((c for c in cc if c['dim']<first['dim'] and c['zero_seeds']==10),key=lambda c:c['dim'])
        mixed=max((c for c in cc if c['dim']<first['dim'] and 0<c['positive_seeds']<10 and not c['unresolved_seeds']),key=lambda c:c['dim'])
        wide=next(c for c in cc if c['dim']==128)
        for role,c in [('resolved_zero',zero),('mixed_boundary',mixed),('positive_boundary',first),('positive_width128',wide)]:
            assert c['n_seeds']==10 and not c['unresolved_seeds']
            selected.append({**c,'role':role})
    assert len(selected)==len({(c['p'],c['dim']) for c in selected})==20
    lookup={(r['p'],r['dim'],r['seed']):r for r in runs if r['gpu_type']=='A100'}
    refs=[];items=[]
    for c in selected:
        for seed in range(42,52):
            r=lookup[c['p'],c['dim'],seed]
            assert all(r[k]==v for k,v in PROTOCOL.items())
            assert r['grokking_threshold']==.99 and r['val98'] is not None
            path=Path(r['trace_path']);assert sha(path)==r['trace_sha256']
            dst=OUT/'reference_traces'/(r['uuid']+'.npz');shutil.copy2(path,dst)
            refs.append({**r,'role':c['role'],'reference_trace':str(dst)})
            items.append({'p':c['p'],'dim':c['dim'],'seed':seed,'role':c['role'],'reference_uuid':r['uuid'],
                          'param_count':r['param_count'],'max_epochs':r['max_epochs'],'reference_epochs':r['epochs_trained']})
    # Start longer reference runs first; selection itself depends only on A100 outcomes.
    items.sort(key=lambda j:(-j['reference_epochs'],j['p'],j['dim'],j['seed']))
    write(OUT/'reference_runs.json',refs)
    protected=json.loads((ROOT/'results/law_complete_ten_seed_sensitivity_20261106/validation.json').read_text())['protected_artifacts_unchanged']
    for rel,digest in protected.items():assert sha(ROOT/rel)==digest
    files=[Path(__file__).resolve(),ROOT/'src/grokking_capacity/experiments/cone_workflow/run_cone_h100_boundary.py',ROOT/'src/grokking_capacity/experiments/cone_workflow/verify_cone_h100_boundary.py',ROOT/'src/grokking_capacity/experiments/cone_workflow/cone_data.py',
           *package_sources(ROOT),ROOT/'wallow.toml']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
    for p in files:
        dest=OUT/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    benchmark=ROOT/'results/cone_ten_seeds_h100_20261106/scheduling_benchmark/result.json'
    assert json.loads(benchmark.read_text())['all_traces_and_checkpoints_identical']
    manifest={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'host':socket.gethostname(),
        'cells':selected,'jobs':200,'items':items,'seeds':list(range(42,52)),'physical_gpus':[0,1],'workers_per_gpu':12,
        'protocol':{k:v for k,v in PROTOCOL.items() if k not in ['host','gpu_type','status']},
        'source_hashes':hashes,'reference_runs_sha256':sha(OUT/'reference_runs.json'),
        'scheduling_benchmark_sha256':sha(benchmark),'protected_publication_artifacts':protected,
        'selection':'From current A100 cells only, for each of p97/107/113/127/139: widest all-ten-zero cell below the first all-ten-positive cell; widest fully observed mixed cell below it; that first all-ten-positive cell; and width128. Fixed before fresh H100 outcomes. All seeds42–51 included.',
        'stopping':'Match the 5000-epoch cap; stop after online train99 and evaluation val99 are both observed. No patience stopping.',
        'primary_outcomes':'Per-seed positive/zero/unresolved under first online train99 and evaluation val98; per-cell all-ten-positive/known-zero-minimum/unresolved classification. Report exact agreement counts and every disagreement, with no post-hoc exclusions.',
        'secondary_outcomes':'Positive-seed counts, delays, and first training/validation crossing epochs. Unobserved validation remains unresolved, never a zero.',
        'interpretation':'Test transfer from archived A100 experiments to the current H100 stack. Historical initial checkpoints are unavailable, so this does not isolate GPU architecture from runtime differences. Sampling covers observed-outcome regimes at p97–139, not the extrapolation range p149–199.',
        'publication_policy':'Separate campaign only; no replacement, pooling, refitting or paper/cone update.'}
    write(OUT/'manifest.json',manifest)
    os.environ['WALLOW_JOURNAL_MODE']='DELETE';os.environ['WALLOW_BUSY_TIMEOUT_MS']='120000'
    from grokking_capacity.registry.store import get_store
    get_store(str(OUT/'runs.db'))
    print(json.dumps({'jobs':200,'cells':[{k:c[k] for k in ['p','dim','role','state','positive_seeds']} for c in selected]},indent=2),flush=True)


def worker(args):
    import grokking_capacity.experiments.cone_workflow.run_cone_h100_boundary as engine
    engine.ROOT=OUT/'source';engine.OUT=OUT
    sys.path.insert(0,str(OUT/'source/src'))
    engine.worker(args)
    path=OUT/'jobs'/f'p{args.p}_d{args.dim}_s{args.seed}.json';meta=json.loads(path.read_text())
    meta.update({'physical_gpu':int(os.environ['CUDA_VISIBLE_DEVICES']),'max_epochs':5000,
                 'manifest_sha256':sha(OUT/'manifest.json'),'training_source_snapshot':str(OUT/'source')})
    assert meta['dtype']=='torch.float32' and meta['tf32_matmul'] is False
    write(path,meta)


def check_one(job):
    from grokking_capacity.experiments.cone_workflow.verify_cone_h100_boundary import check
    return check(job,cpu_threads=1,linear_backend='numpy')


def analyse():
    import numpy as np
    from grokking_capacity.experiments.cone_workflow.cone_data import first_events, aggregate
    manifest=json.loads((OUT/'manifest.json').read_text());refs=json.loads((OUT/'reference_runs.json').read_text())
    rows=read_rows();assert len(rows)==200
    expected={(j['p'],j['dim'],j['seed']):j for j in manifest['items']}
    assert {(r['p'],r['dim'],r['seed']) for r in rows}==set(expected)
    assert sha(OUT/'reference_runs.json')==manifest['reference_runs_sha256']
    for rel,digest in manifest['source_hashes'].items():assert sha(OUT/'source'/rel)==digest
    jobs=[]
    for r in rows:
        meta=json.loads((OUT/'jobs'/f'p{r["p"]}_d{r["dim"]}_s{r["seed"]}.json').read_text())
        assert r['gpu_type']=='H100' and r['host']==meta['host']==manifest['host']
        assert all(r[k]==v for k,v in manifest['protocol'].items())
        assert meta['manifest_sha256']==sha(OUT/'manifest.json')
        assert r['param_count']==expected[r['p'],r['dim'],r['seed']]['param_count']
        if not (OUT/'checkpoint_checks'/(r['uuid']+'.json')).exists():jobs.append((r,meta))
    with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
        for value in pool.map(check_one,jobs):
            write(OUT/'checkpoint_checks'/(value['uuid']+'.json'),value)
            print(json.dumps({'checked':value['uuid'],'p':value['p'],'dim':value['dim'],'seed':value['seed']}),flush=True)
    checks={p.stem:json.loads(p.read_text()) for p in (OUT/'checkpoint_checks').glob('*.json')};assert len(checks)==200
    fresh=[]
    for r in rows:
        check=checks[r['uuid']]
        assert check['trace_sha256']==sha(r['npz_path'])
        with np.load(r['npz_path']) as z:event=first_events(z['train_acc'],z['val_acc'])
        assert all(event[k]==check[k] for k in ['train99','val98','state','delay'])
        assert event['state']=='unresolved' or event['val98'] is not None
        fresh.append({**r,**event})
    for r in refs:
        assert sha(r['reference_trace'])==r['trace_sha256']
        with np.load(r['reference_trace']) as z:event=first_events(z['train_acc'],z['val_acc'])
        assert all(r[k]==event[k] for k in ['train99','val98','state','delay'])
    reference={(r['p'],r['dim'],r['seed']):r for r in refs};pairs=[]
    for r in sorted(fresh,key=lambda x:(x['p'],x['dim'],x['seed'])):
        old=reference[r['p'],r['dim'],r['seed']]
        pairs.append({'p':r['p'],'dim':r['dim'],'seed':r['seed'],'role':old['role'],'A100_uuid':old['uuid'],'H100_uuid':r['uuid'],
          'A100_state':old['state'],'H100_state':r['state'],'state_agrees':old['state']==r['state'],
          'A100_delay':old['delay'],'H100_delay':r['delay'],
          'delay_difference':None if old['delay'] is None or r['delay'] is None else r['delay']-old['delay'],
          'A100_train99':old['train99'],'H100_train99':r['train99'],'A100_val98':old['val98'],'H100_val98':r['val98']})
    a={(c['p'],c['dim']):c for c in aggregate(refs)};h={(c['p'],c['dim']):c for c in aggregate(fresh)}
    comparisons=[]
    for key in sorted(a):
        x,y=a[key],h[key];role=reference[*key,42]['role']
        comparisons.append({'p':key[0],'dim':key[1],'role':role,'parameters':x['param_count'],
          'A100_state':x['state'],'H100_state':y['state'],'state_agrees':x['state']==y['state'],
          'A100_positive':x['positive_seeds'],'H100_positive':y['positive_seeds'],
          'A100_zero':x['zero_seeds'],'H100_zero':y['zero_seeds'],'H100_unresolved':y['unresolved_seeds'],
          'A100_minimum_delay':x['minimum_delay'],'H100_minimum_delay':y['minimum_delay'],
          'A100_mean_delay':x['mean_delay'],'H100_mean_delay':y['mean_delay'],
          'seed_states_agree':sum(p['state_agrees'] for p in pairs if (p['p'],p['dim'])==key)})
    for name,data in [('paired_runs.csv',pairs),('paired_cells.csv',comparisons)]:
        with (OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    report={'completed_runs':200,'cells':20,'all_new_checkpoints_independently_CPU_verified':True,
      'seed_state_agreement':sum(p['state_agrees'] for p in pairs),'cell_state_agreement':sum(c['state_agrees'] for c in comparisons),
      'A100_states':dict(Counter(p['A100_state'] for p in pairs)),'H100_states':dict(Counter(p['H100_state'] for p in pairs)),
      'seed_disagreements':[p for p in pairs if not p['state_agrees']],
      'cell_disagreements':[c for c in comparisons if not c['state_agrees']],
      'cells':comparisons,'median_delay_difference':float(np.median([p['delay_difference'] for p in pairs if p['delay_difference'] is not None])),
      'all_seed_delays_exactly_equal':all(p['delay_difference']==0 for p in pairs),
      'interpretation':manifest['interpretation'],'manifest_sha256':sha(OUT/'manifest.json')}
    for rel,digest in manifest['protected_publication_artifacts'].items():assert sha(ROOT/rel)==digest
    report['published_cone_and_paper_unchanged']=True;write(OUT/'analysis.json',report)
    text=(f'PAIRED A100/H100 GROKKING OUTCOME CHECK\n\n200 new H100 runs; 20 preselected cells; seeds42–51 at every cell. All checkpoints independently checked on CPU.\n\n'
          f'Seed classifications agree: {report["seed_state_agreement"]}/200\nCell classifications agree: {report["cell_state_agreement"]}/20\n'
          f'A100 states: {report["A100_states"]}\nH100 states: {report["H100_states"]}\nMedian delay difference: {report["median_delay_difference"]} epochs\n\n'
          +manifest['interpretation']+'\n\nEvery disagreement is recorded in analysis.json, paired_runs.csv and paired_cells.csv. Published cone and paper unchanged.\n')
    (OUT/'report.txt').write_text(text);print(text,flush=True)


def run():
    m=json.loads((OUT/'manifest.json').read_text())
    for rel,digest in m['source_hashes'].items():assert sha(OUT/'source'/rel)==digest
    PIPE.mkdir(exist_ok=True)
    mps_env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1','CUDA_MPS_PIPE_DIRECTORY':str(PIPE),'CUDA_MPS_LOG_DIRECTORY':str(OUT/'mps_logs')}
    assert not (PIPE/'nvidia-cuda-mps-control.pid').exists(),'Private MPS server already running; inspect before starting another controller'
    subprocess.run(['nvidia-cuda-mps-control','-d'],env=mps_env,check=True)
    started=time.time();script=frozen_source(OUT, 'run_a100_h100_agreement.py')
    write(OUT/'controller_identity.json',{'pid':os.getpid(),'started_at':started,'script':str(script)})
    limits={gpu:threading.Semaphore(m['workers_per_gpu']) for gpu in [0,1]}
    def job(index,spec):
        gpu=index%2;name=f'p{spec["p"]}_d{spec["dim"]}_s{spec["seed"]}'
        with limits[gpu]:
            env={**mps_env,'CUDA_VISIBLE_DEVICES':str(gpu),'GC_AGREEMENT_ROOT':str(ROOT),'GC_DATA_DIR':str(OUT/'data'),
                 'GC_WALLOW_TOML':str(OUT/'source/wallow.toml'),'WALLOW_JOURNAL_MODE':'DELETE','WALLOW_BUSY_TIMEOUT_MS':'120000',
                 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1','PYTHONPATH':str(OUT/'source/src')}
            t=time.time();write(OUT/'launch_provenance'/(name+'.json'),{'physical_gpu':gpu,'started_at':t,'manifest_sha256':sha(OUT/'manifest.json'),'MPS_pipe':str(PIPE)})
            cmd=[sys.executable,str(script),'--worker','--p',str(spec['p']),'--dim',str(spec['dim']),'--seed',str(spec['seed'])]
            with (OUT/'logs'/(name+'.log')).open('a') as f:proc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            return {'name':name,'gpu':gpu,'returncode':proc.returncode,'seconds':time.time()-t}
    results=[]
    write(OUT/'status.json',{'phase':'training','planned':200,'completed':0,'failed':[],'started_at':started})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2*m['workers_per_gpu']) as pool:
        futures=[pool.submit(job,i,j) for i,j in enumerate(m['items'])]
        for future in concurrent.futures.as_completed(futures):
            result=future.result();results.append(result);print(json.dumps(result),flush=True)
            write(OUT/'status.json',{'phase':'training','planned':200,'completed':sum(r['returncode']==0 for r in results),
                 'failed':[r for r in results if r['returncode']],'started_at':started,'updated_at':time.time()})
    subprocess.run(['nvidia-cuda-mps-control'],input='quit\n',text=True,env=mps_env,check=True)
    write(OUT/'completion.json',{'results':results,'started_at':started,'finished_at':time.time(),'completed':sum(r['returncode']==0 for r in results)})
    assert len(results)==200 and all(r['returncode']==0 for r in results)
    write(OUT/'status.json',{'phase':'CPU_verification','planned':200,'completed':200,'updated_at':time.time()})
    analyse()
    write(OUT/'status.json',{'phase':'complete','planned':200,'completed':200,'CPU_verified':200,'started_at':started,'finished_at':time.time()})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--run',action='store_true');ap.add_argument('--analyse',action='store_true');ap.add_argument('--worker',action='store_true');ap.add_argument('--p',type=int);ap.add_argument('--dim',type=int);ap.add_argument('--seed',type=int);args=ap.parse_args()
    if args.prepare:prepare()
    elif args.worker:worker(args)
    elif args.analyse:analyse()
    elif args.run:run()
    else:ap.error('Choose --prepare, --run, --worker or --analyse')
