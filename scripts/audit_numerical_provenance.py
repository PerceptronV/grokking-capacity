#!/usr/bin/env python3
"""Read-only census of local experiment data and the published speed cohort."""
from pathlib import Path
import collections, concurrent.futures, csv, datetime, hashlib, json, os, socket, sqlite3, subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/numerical_provenance_20261103'
OUT.mkdir(parents=True, exist_ok=True)

def write_json(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, default=str)+'\n')

def write_csv(name, rows):
    if not rows: return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def group_counts(rows, keys):
    c=collections.Counter(tuple(r.get(k) for k in keys) for r in rows)
    return [dict(zip(keys,k),count=n) for k,n in sorted(c.items(),key=lambda x:str(x[0]))]

def walk_files(base):
    for d,ds,fs in os.walk(base):
        ds[:]=[x for x in ds if x not in {'.git','.venv','venv','__pycache__','node_modules','source-worktrees','source','publication_source'}]
        for f in fs: yield Path(d)/f

files=list(walk_files(ROOT))
dbs=sorted({p.resolve() for p in files if p.suffix=='.db'} | {p.resolve() for p in Path('/home/ubuntu').glob('*runs.db')})
records=[]; db_summary=[]
for p in dbs:
    con=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
    tables=[r[0] for r in con.execute("select name from sqlite_master where type='table'")]
    if 'runs' not in tables:
        db_summary.append({'database':str(p),'error':'no runs table','tables':tables});continue
    rows=[dict(r) for r in con.execute('select * from runs')];con.close()
    for r in rows:r['database']=str(p)
    records.extend(rows)
    db_summary.append({'database':str(p),'rows':len(rows),'by_hardware_status':group_counts(rows,['host','gpu_type','gpu_count','status','experiment_type']), 'by_commit':group_counts(rows,['git_hash','git_dirty']), 'date_range':[min((r['created_at'] for r in rows),default=None),max((r['created_at'] for r in rows),default=None)]})
write_json('registries.json',db_summary)
columns=['database','uuid','experiment_type','status','host','gpu_type','gpu_count','git_hash','git_dirty','created_at','completed_at','p','n_samples','dim','seed','depth','heads','dropout','init_scale','lr','weight_decay','beta1','beta2','batch_size','max_epochs','mup_base','saturated','saturation_epoch','grokking_epoch','npz_path']
write_csv('registry_rows.csv',[{k:r.get(k) for k in columns} for r in records])
# Records copied into several registries are not independent experiments.
unique={}
for r in records:
    k=tuple(r.get(x) for x in ('uuid','host','started_at','completed_at','git_hash','status'))
    unique.setdefault(k,r)
unique=list(unique.values())
completed=[r for r in unique if r['status']=='completed']
root_rows=[r for r in records if r['database']==str(ROOT/'runs.db')]
print('registries',len(dbs),'rows',len(records),'unique',len(unique),'completed',len(completed),flush=True)
print('completed hardware',group_counts(completed,['host','gpu_type','gpu_count']),flush=True)

# Census manifests as files; report copies and campaign-level manifests are not counted as runs.
manifest_rows=[]
for p in files:
    if p.name!='manifest.json':continue
    try:x=json.loads(p.read_text())
    except Exception as e:manifest_rows.append({'path':str(p),'error':str(e)});continue
    env=x.get('environment',{});env=env if isinstance(env,dict) else {}
    manifest_rows.append({'path':str(p.relative_to(ROOT)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'host':x.get('host',env.get('host')), 'gpu':x.get('gpu',env.get('gpu')), 'device':x.get('device',env.get('device')), 'torch':x.get('torch',env.get('torch')), 'git_commit':x.get('git_commit',x.get('git_hash')), 'has_config':isinstance(x.get('config'),dict), 'has_source_hashes':bool(x.get('source_hashes')), 'start_time':x.get('start_time')})
write_csv('manifests.csv',manifest_rows)
manifest_counts=group_counts(manifest_rows,['host','gpu','has_config'])
print('manifests',len(manifest_rows),manifest_counts,flush=True)

# The central speed cohort, including censored cells omitted from the plot.
cfg=json.loads((ROOT/'results/capacity_h100_matched_20261103/plot_manifest.json').read_text())
figure_ids=set(cfg['speed_uuids'])
primes=[97,101,103,107,109,113,127,131,137,139]
dims=list(range(20,129,4))+list(range(136,257,8))
protocol={'experiment_type':'speed','operation':'/','train_fraction':.5,'split_type':'random','depth':2,'heads':1,'dropout':.2,'init_scale':1.,'lr':.001,'weight_decay':1.,'beta1':.9,'beta2':.98,'batch_size':512,'max_epochs':5000,'mup_base':0}
def central_match(r):
    return all(r.get(k)==v for k,v in protocol.items()) and r.get('p') in primes and r.get('dim') in dims and r.get('seed') in range(42,52) and r.get('n_samples')==int(np.ceil(r['p']*(r['p']-1)*.5))
cohort=[r for r in root_rows if central_match(r) and r['host']=='147-224-250-73']
figure=[r for r in cohort if r['uuid'] in figure_ids]
assert len(figure)==len(figure_ids)==3324
assert {r['uuid'] for r in cohort if r['saturated']}==figure_ids
thresholds=[50,90,95,98,98.5,99]
def inspect(r):
    base={k:r.get(k) for k in ['uuid','p','dim','seed','host','gpu_type','saturated','saturation_epoch','param_count','dataset_bits','n_samples']}
    try:
        p=Path(r['npz_path']);p=p if p.exists() else ROOT/'data'/r['experiment_type']/r['uuid']/'trace.npz'
        with np.load(p,allow_pickle=False) as z:
            a=z['train_acc_trace'];loss=z['train_loss_trace'];steps=z['steps_trace']
            base.update(trace_epochs=len(a),nonfinite=int(np.sum(~np.isfinite(a))+np.sum(~np.isfinite(loss))),scalar_epoch_match=bool(float(z['saturation_epoch'])==r['saturation_epoch']),step_trace_match=bool(np.array_equal(steps,np.arange(1,len(a)+1)*int(np.ceil(r['n_samples']/512)))))
            for t in thresholds:
                hit=np.flatnonzero(a>=t);base['epoch_'+str(t)]=int(hit[0]+1) if len(hit) else None
            base['threshold_match']=bool((base['epoch_99']==r['saturation_epoch']) if r['saturated'] else base['epoch_99'] is None)
            base['fraction_epochs_98_to_99']=float(np.mean((a>=98)&(a<99)))
            base['last_acc']=float(a[-1]);base['trace_sha256']=hashlib.sha256(a.tobytes()+loss.tobytes()).hexdigest()
    except Exception as e:base['error']=str(e)
    return base
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    traces=list(ex.map(inspect,cohort))
write_csv('central_speed_traces.csv',traces)
print('traces',len(traces),'errors',sum('error' in t for t in traces),flush=True)
cell_rows=[]
for p in primes:
    for d in dims:
        rr=[r for r in traces if r['p']==p and r['dim']==d];ss=[r for r in rr if r['saturated']]
        x={'p':p,'dim':d,'n':len(rr),'n_saturated':len(ss),'n_censored':len(rr)-len(ss)}
        for t in thresholds:
            v=[r.get('epoch_'+str(t)) for r in rr if r.get('epoch_'+str(t)) is not None]
            x['hits_'+str(t)]=len(v);x['median_'+str(t)]=float(np.median(v)) if v else None
        if ss:x.update(median_saturation=float(np.median([r['saturation_epoch'] for r in ss])),min_saturation=min(r['saturation_epoch'] for r in ss),max_saturation=max(r['saturation_epoch'] for r in ss))
        cell_rows.append(x)
write_csv('central_speed_cells.csv',cell_rows)
# Existing nominally matching H100 cells. Matching registry fields do not prove identical source/runtime.
nominal=[r for r in completed if central_match(r) and r['gpu_type']=='H100']
write_csv('nominal_h100_matches.csv',[{k:r.get(k) for k in columns} for r in nominal])

extensions={'.npz','.npy','.pt','.pth','.csv','.parquet','.db','.json','.jsonl'}
inventory=collections.Counter((str(p.relative_to(ROOT)).split('/')[0],p.suffix) for p in files if p.suffix in extensions)
write_json('file_inventory.json',[{'top_directory':k[0],'extension':k[1],'files':v} for k,v in sorted(inventory.items())])
summary={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'current_host':socket.gethostname(),'hardware_now':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip(),'scope':'All .db registries in repository plus /home/ubuntu/*runs.db; all repository manifest.json files excluding source snapshots; every central speed trace. Metadata audit of other runs, not a numerical revalidation of every trace.', 'database_count':len(dbs),'registry_rows':len(records),'unique_records':len(unique),'unique_completed_records':len(completed),'unique_completed_hardware':group_counts(completed,['host','gpu_type','gpu_count']),'root_registry_hardware':group_counts(root_rows,['host','gpu_type','gpu_count','experiment_type']),'root_registry_status':group_counts(root_rows,['status']),'manifest_counts':manifest_counts,'manifest_files':len(manifest_rows),'central_cohort':len(cohort),'plotted':len(figure),'censored':len(cohort)-len(figure),'figure_hardware':group_counts(figure,['host','gpu_type','gpu_count','git_hash','git_dirty']),'trace_errors':[r for r in traces if 'error' in r],'trace_validation_failures':[r for r in traces if 'error' not in r and (r['nonfinite'] or not r['scalar_epoch_match'] or not r['step_trace_match'] or not r['threshold_match'])],'duplicate_trace_groups':[v for v in (lambda dd:list(dd.values()))(__import__('functools').reduce(lambda d,r:(d[r.get('trace_sha256')].append(r['uuid']) or d),traces,collections.defaultdict(list))) if len(v)>1],'nominal_H100_matches':group_counts(nominal,['host','database'])}
write_json('summary.json',summary)
print('central',len(cohort),'plotted',len(figure),'censored',len(cohort)-len(figure),'validation failures',len(summary['trace_validation_failures']),flush=True)
print('nominal H100 matches',summary['nominal_H100_matches'],flush=True)
