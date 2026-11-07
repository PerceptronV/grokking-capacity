#!/usr/bin/env python3
"""Trace paper cohorts and recompute cone pooling without editing the paper."""
from pathlib import Path
import collections,concurrent.futures,csv,json,sqlite3,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/numerical_provenance_20261103';HOME=Path('/home/ubuntu')
def read(db,table='runs',where="status='completed'"):
    c=sqlite3.connect(Path(db).resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
    out=[]
    for v in c.execute(f'select * from {table} where {where}'):
        r=dict(v);r['uuid']=r.get('uuid',r.get('run_uuid'));r['source_table']=table;r['database']=str(db);out.append(r)
    c.close();return out
def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2,default=str)+'\n')
def csvwrite(name,rows):
    if not rows:return
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def group(rows,keys):
    c=collections.Counter(tuple(r.get(k) for k in keys) for r in rows)
    return [dict(zip(keys,k),count=v) for k,v in sorted(c.items(),key=str)]
root=read(ROOT/'runs.db');sup=read(ROOT/'runs.db','runs_superseded',where='1=1')
original={2:read(ROOT.parent/'runs2.db'),3:read(ROOT.parent/'runs3.db')}
old={r['uuid']:r for r in sup}
for rs in original.values():old.update({r['uuid']:r for r in rs})
with (OUT/'unregistered_artifacts.csv').open() as f:orphans=list(csv.DictReader(f))
recovered=[]
for o in orphans:
    r=old.get(o['uuid']);o['recovered_host']=r.get('host') if r else None;o['recovered_gpu']=r.get('gpu_type') if r else None;o['recovered_status']=r.get('status') if r else None;o['recovered_source']=r.get('database') if r else None;recovered.append(o)
csvwrite('historical_artifact_recovery.csv',recovered)
print('recovered traces',group([r for r in recovered if r['exists']=='True'],['recovered_host','recovered_gpu','recovered_status']),flush=True)
# Match every archived figure meta group to its actual single-machine source.
suites=[]
for suite,which in [('init_scale_sweep',3),('depth_scaling',3),('task_add',3),('lr_sweep',2)]:
    meta=json.loads((ROOT/f'figures/{suite}/meta.json').read_text())
    for g in meta['groups']:
        key=g['arch_key'];rr=[r for r in original[which] if all(r.get(k)==v for k,v in key.items())]
        canonical=[r for r in root if all(r.get(k)==v for k,v in key.items()) and r['host']==('68-209-72-216' if which==3 else '68-209-75-173')]
        counts=collections.Counter(r['experiment_type'] for r in rr);cc=collections.Counter(r['experiment_type'] for r in canonical)
        suites.append({'suite':suite,'arch_key':key,'actual_hardware':group(rr,['host','gpu_type']),'source':str(ROOT.parent/f'runs{which}.db'),'source_counts':dict(counts),'meta_counts':{t:g['n_'+t+'_runs'] for t in ['capacity','speed','groks']},'metadata_matches_source':all(counts[t]==g['n_'+t+'_runs'] for t in ['capacity','speed','groks']),'merged_pinned_counts':dict(cc)})
save('paper_suites_original_sources.json',suites)
print('legacy metadata matches original databases',all(s['metadata_matches_source'] for s in suites),flush=True)
# Fixed cone uses exactly this permissive selector and pools at (p,dim).
base="experiment_type='groks' and status='completed' and operation='/' and train_fraction=0.5 and dropout=0.2 and weight_decay=1.0 and lr=0.001 and batch_size=512 and depth=2 and heads=1"
sources=[(ROOT/'runs.db',base+' and mup_base=0')]+[(HOME/(n+'_runs.db'),base) for n in ['cone_fill','band_gaps','p197','crossing_fixed']]+[(HOME/'lr_wall_check_runs.db',base+' and mup_base=0')]
selected=[]
for db,q in sources:selected.extend(read(db,where=q))
def delay(r):
    p=Path(r['npz_path']);p=p if p.exists() else ROOT/'data/groks'/r['uuid']/'trace.npz'
    try:
        with np.load(p,allow_pickle=False) as z:tr=z['train_acc'];va=z['val_acc']
        ti=np.flatnonzero(tr>=99);vi=np.flatnonzero(va>=98)
        if not len(ti):return dict(r,delay=None,skip='no train99')
        return dict(r,delay=int(max(0,vi[0]-ti[0]) if len(vi) else max(0,len(va)-ti[0]-1)))
    except Exception as e:return dict(r,delay=None,skip=str(e))
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:scored=list(ex.map(delay,selected))
valid=[r for r in scored if r['delay'] is not None];cells=collections.defaultdict(list)
for r in valid:cells[r['p'],r['dim']].append(r)
cr=[]
for (p,d),rr in sorted(cells.items()):
    hw={r['gpu_type'] for r in rr};hosts={r['host'] for r in rr};ini={r['init_scale'] for r in rr};ones=[r for r in rr if r['init_scale']==1];a=[r for r in ones if r['gpu_type']=='A100'];h=[r for r in ones if r['gpu_type']=='H100']
    f=lambda rs: min(r['delay'] for r in rs) if rs else None
    cr.append({'p':p,'dim':d,'n':len(rr),'gpu_types':';'.join(sorted(hw)),'hosts':';'.join(sorted(hosts)),'init_scales':';'.join(map(str,sorted(ini))),'mixed_GPU_types':len(hw)>1,'mixed_hosts':len(hosts)>1,'mixed_init_scales':len(ini)>1,'pooled_min':f(rr),'pooled_mean':float(np.mean([r['delay'] for r in rr])),'init1_min':f(ones),'A100_init1_min':f(a),'H100_init1_min':f(h),'A100_init1_n':len(a),'H100_init1_n':len(h),'class_changes_excluding_nonunit_init':bool(ones and ((f(rr)>0)!=(f(ones)>0))),'class_disagrees_between_hardware_at_init1':bool(a and h and ((f(a)>0)!=(f(h)>0)))})
csvwrite('cone_cell_hardware.csv',cr)
csvwrite('cone_selected_rows.csv',[{k:r.get(k) for k in ['uuid','database','p','dim','seed','gpu_type','host','init_scale','max_epochs','param_count','delay','skip']} for r in scored])
cone={'selected_rows':len(selected),'scored_rows':len(valid),'cells':len(cells),'by_hardware':group(valid,['host','gpu_type','init_scale']),'mixed_GPU_type_cells':sum(r['mixed_GPU_types'] for r in cr),'mixed_host_cells':sum(r['mixed_hosts'] for r in cr),'mixed_initialisation_cells':sum(r['mixed_init_scales'] for r in cr),'class_changes_excluding_nonunit_init':[r for r in cr if r['class_changes_excluding_nonunit_init']],'class_disagreement_between_hardware_init1':[r for r in cr if r['class_disagrees_between_hardware_at_init1']]}
save('cone_mixing_summary.json',cone)
print('cone', {k:v for k,v in cone.items() if k not in ['by_hardware','class_changes_excluding_nonunit_init','class_disagreement_between_hardware_init1']},'init changes',len(cone['class_changes_excluding_nonunit_init']),'hardware disagreement',len(cone['class_disagreement_between_hardware_init1']),flush=True)
# Same-identifying-tuple historical H100 replicas preserved in the merge.
fields=['experiment_type','p','operation','train_fraction','split_type','dataset_type','n_samples','dim','depth','heads','dropout','init_scale','lr','weight_decay','beta1','beta2','batch_size','max_epochs','seed','architecture_family']
idx={tuple(r.get(k) for k in fields):r for r in root if r['gpu_type']=='A100'}
pairs=[]
for rs in original.values():
    for r in rs:
        a=idx.get(tuple(r.get(k) for k in fields))
        if not a or r['experiment_type'] not in ['speed','groks']:continue
        metric='saturation_epoch' if r['experiment_type']=='speed' else 'grokking_epoch'
        ok=(r.get('saturated') and a.get('saturated')) if r['experiment_type']=='speed' else (r.get(metric) is not None and a.get(metric) is not None)
        if not ok:continue
        av,hv=a[metric],r[metric]
        pairs.append({'experiment_type':r['experiment_type'],'H100_host':r['host'],'p':r['p'],'dim':r['dim'],'seed':r['seed'],'A100_uuid':a['uuid'],'H100_uuid':r['uuid'],'A100_epochs':av,'H100_epochs':hv,'log10_ratio':float(np.log10(hv/av)),'exact_equal':av==hv})
csvwrite('historical_paired_hardware.csv',pairs)
ps=[]
for key in sorted({(r['experiment_type'],r['H100_host']) for r in pairs}):
    rr=[r for r in pairs if (r['experiment_type'],r['H100_host'])==key];v=[r['log10_ratio'] for r in rr]
    ps.append({'experiment_type':key[0],'H100_host':key[1],'n':len(rr),'median_log10_ratio':float(np.median(v)),'q25_q75':np.quantile(v,[.25,.75]).tolist(),'exact_equal_fraction':float(np.mean([r['exact_equal'] for r in rr])),'max_abs_log10_ratio':float(np.max(np.abs(v)))})
save('historical_paired_hardware_summary.json',ps);print('paired',ps,flush=True)
# Supplementary intervention and decay-law comparison used by principled.tex.
base="experiment_type='groks' and status='completed' and p=113 and operation='/' and train_fraction=0.5 and depth=2 and heads=1 and dropout=.2 and lr=.001 and init_scale=1 and batch_size=512 and grokking_epoch is not null"
frozen=read(HOME/'normfrozen_runs.db',where=base)
unfrozen=read(ROOT/'runs.db',where=base+' and seed in (42,43,44)')
used=lambda rr:[r for r in rr if (r['dim']==128 and r['weight_decay'] in [.1,.3,1]) or (r['dim'] in [64,128,256] and r['weight_decay']==1)]
mechanism={}
for label,rr in [('frozen',used(frozen)),('unfrozen',used(unfrozen))]:
    mechanism[label]={'n':len(rr),'hardware':group(rr,['host','gpu_type']),'cells':group(rr,['dim','weight_decay'])}
query="experiment_type='groks' and status='completed' and p=113 and grokking_epoch is not null and dropout=.2 and lr=.001 and weight_decay in (.01,.03,.1,.3,1.0)"
rr=read(ROOT/'runs.db',where=query)
matched=lambda r:r['operation']=='/' and r['train_fraction']==.5 and r['depth']==2 and r['heads']==1 and r['init_scale']==1 and r['batch_size']==512 and r.get('mup_base',0)==0
lambdas=[.01,.03,.1,.3,1.];details=[]
for label,rs in [('permissive',rr),('A100_only_same_permissive_selector',[r for r in rr if r['gpu_type']=='A100']),('matched_all_hardware',[r for r in rr if matched(r)]),('matched_A100',[r for r in rr if matched(r) and r['gpu_type']=='A100'])]:
    cells=collections.defaultdict(list)
    for r in rs:cells[r['dim'],r['weight_decay']].append(r['grokking_epoch'])
    dims=sorted(d for d in {r['dim'] for r in rs} if all((d,w) in cells for w in lambdas))
    slopes=[float(np.polyfit(np.log(lambdas),np.log([np.mean(cells[d,w]) for w in lambdas]),1)[0]) for d in dims]
    mechanism['lambda_'+label]={'n':len(rs),'common_widths':dims,'median_slope':float(np.median(slopes)),'hardware':group(rs,['host','gpu_type'])}
    for d,s in zip(dims,slopes):
        details.append({'selector':label,'dim':d,'slope':s,**{f'mean_T_wd{w}':float(np.mean(cells[d,w])) for w in lambdas},**{f'n_wd{w}':len(cells[d,w]) for w in lambdas}})
save('mechanism_hardware_comparisons.json',mechanism);csvwrite('lambda_slope_sensitivity.csv',details)
print('mechanism',mechanism,flush=True)
