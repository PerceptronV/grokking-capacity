#!/usr/bin/env python3
"""Supplement the numerical census with trace provenance gaps and diagnostics."""
from pathlib import Path
import ast, collections, concurrent.futures, csv, hashlib, json, os, sqlite3, subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/numerical_provenance_20261103'
def csvread(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))
def csvwrite(name,rows):
    if not rows:return
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2,default=str)+'\n')
rows=csvread('registry_rows.csv');known={r['uuid'] for r in rows}
# Inventory actual raw files that have no UUID in any discovered registry.
orphans=[]
for typ in ('capacity','speed','groks'):
    for e in os.scandir(ROOT/'data'/typ):
        if e.name not in known and e.is_dir():orphans.append((typ,Path(e.path)))
def examine_orphan(item):
    typ,d=item;p=d/'trace.npz';x={'type':typ,'uuid':d.name,'path':str(p),'exists':p.exists()}
    if x['exists']:
        try:
            with np.load(p,allow_pickle=False) as z:
                x['hardware_keys']=';'.join(k for k in z.files if any(s in k.lower() for s in ('host','gpu','cuda','device')))
                for k in ['p','dim','n_samples','seed','epochs_trained','saturated','saturation_epoch','param_count']:
                    if k in z and z[k].ndim==0:x[k]=z[k].item()
        except Exception as e:x['error']=str(e)
    return x
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:orphan_rows=list(ex.map(examine_orphan,orphans))
csvwrite('unregistered_artifacts.csv',orphan_rows)
print('orphans',collections.Counter((r['type'],r['exists']) for r in orphan_rows),'hardware keys',collections.Counter(r.get('hardware_keys') for r in orphan_rows),flush=True)
# Check that every completed record resolves to an artifact, without loading all traces.
def exists(r):
    p=Path(r['npz_path']) if r['npz_path'] else None
    if p and p.exists():return None
    fallback=ROOT/'data'/r['experiment_type']/r['uuid']/'trace.npz'
    if fallback.exists():return {'uuid':r['uuid'],'database':r['database'],'fallback':str(fallback)}
    return {'uuid':r['uuid'],'database':r['database'],'missing':str(p)}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    missing=[x for x in ex.map(exists,[r for r in rows if r['status']=='completed']) if x]
save('artifact_path_exceptions.json',missing)
print('path exceptions',len(missing),'missing',sum('missing' in r for r in missing),flush=True)

con=sqlite3.connect('file:/home/ubuntu/batch_sweep_runs.db?mode=ro',uri=True);con.row_factory=sqlite3.Row
h=[dict(r) for r in con.execute('select * from runs where batch_size=512')];con.close()
assert len(h)==108 and {r['seed'] for r in h}=={52,53,54}
# These are different seeds and a dirty source tree: a consistency check, not a paired hardware experiment.
protocol={'operation':'/','train_fraction':.5,'split_type':'random','depth':2,'heads':1,'dropout':.2,'init_scale':1.,'lr':.001,'weight_decay':1.,'beta1':.9,'beta2':.98,'batch_size':512,'max_epochs':5000,'mup_base':0}
assert all(all(r[k]==v for k,v in protocol.items()) for r in h)
a=csvread('central_speed_cells.csv');ai={(int(r['p']),int(r['dim'])):r for r in a}
comparison=[];hvalidation=[]
for r in h:
    with np.load(r['npz_path'],allow_pickle=False) as z:
        tr=z['train_acc_trace'];hit=np.flatnonzero(tr>=99);first=int(hit[0]+1) if len(hit) else None
        hvalidation.append({'uuid':r['uuid'],'first99':first,'registry_match':first==r['saturation_epoch'] if r['saturated'] else first is None})
for p,d in sorted({(r['p'],r['dim']) for r in h}):
    rr=[r for r in h if r['p']==p and r['dim']==d];ss=[r['saturation_epoch'] for r in rr if r['saturated']];ar=ai[p,d]
    x={'p':p,'dim':d,'A100_seeds':10,'H100_seeds':3,'A100_saturated':int(ar['n_saturated']),'H100_saturated':len(ss),'A100_median':float(ar['median_saturation']) if ar['median_saturation'] else None,'H100_median':float(np.median(ss)) if ss else None}
    if x['A100_median'] and x['H100_median']:x['H100_over_A100']=x['H100_median']/x['A100_median']
    comparison.append(x)
csvwrite('current_H100_comparison.csv',comparison)
ratios=[r['H100_over_A100'] for r in comparison if 'H100_over_A100' in r]
print('H100 comparison',len(comparison),'bothuncensored',len(ratios),'ratio range',min(ratios),max(ratios),'median',np.median(ratios),flush=True)
# Recover the actual training metric from the clean commit used in the plotted rows.
old=subprocess.check_output(['git','show','f6b0754:src/torch_grokking/experiments/speed.py'],cwd=ROOT,text=True)
new=(ROOT/'src/grokking_capacity/experiments/speed.py').read_text()
def node(source,name):return next(n for n in ast.parse(source).body if getattr(n,'name',None)==name)
source_check={'commit':'f6b0754','old_SpeedTrainer_AST_equals_current':ast.dump(node(old,'SpeedTrainer'),include_attributes=False)==ast.dump(node(new,'SpeedTrainer'),include_attributes=False),'old_source_sha256':hashlib.sha256(old.encode()).hexdigest(),'metric':'First epoch with online training minibatch accuracy >= 99%, model.train() with dropout active; patience=0 by default.','model_diff':subprocess.check_output(['git','diff','f6b0754:src/torch_grokking/models/transformer.py','HEAD:src/grokking_capacity/models/transformer.py'],cwd=ROOT,text=True)}
save('source_comparison.json',source_check)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(12,8.2),layout='constrained')
for ax,p,lo,hi in [(axs[0,0],113,56,104),(axs[0,1],139,76,124)]:
    cells=[r for r in a if int(r['p'])==p and lo<=int(r['dim'])<=hi]
    for t,color in [('95','#178a8d'),('98','#d88824'),('99','#704b98')]:
        good=[r for r in cells if r['median_'+t]]
        ax.plot([int(r['dim']) for r in good],[float(r['median_'+t]) for r in good],'-o',label=t+'% accuracy',color=color,markersize=4)
    ax.set(yscale='log',xlabel='Model width',ylabel='Median first-passage epoch',title=f'p = {p}: the cliff depends on the threshold')
    ax.grid(alpha=.2,which='both');ax.legend(frameon=False)
ax=axs[1,0]
con=sqlite3.connect('file:'+str(ROOT/'runs.db')+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
for uuid,color in [('8eee858c4aea','#704b98'),('127c3045c43e','#178a8d')]:
    r=dict(con.execute('select * from runs where uuid=?',(uuid,)).fetchone())
    with np.load(r['npz_path']) as z:tr=z['train_acc_trace']
    ax.plot(np.arange(1,len(tr)+1),tr,lw=.85,alpha=.85,color=color,label=f"width {r['dim']}: first 99% at epoch {len(tr)}")
ax.axhline(99,color='black',ls='--',lw=1);ax.set(xlim=(60,500),ylim=(90,100),xlabel='Epoch',ylabel='Online training accuracy (%)',title='p = 113, seed 42: a long plateau below 99%');ax.legend(frameon=False,fontsize=9,loc='lower right');ax.grid(alpha=.2)
ax=axs[1,1]
for p,col in [(97,'#178a8d'),(113,'#d88824'),(139,'#704b98')]:
    rr=[r for r in comparison if r['p']==p and 'H100_over_A100' in r]
    ax.scatter([r['A100_median'] for r in rr],[r['H100_median'] for r in rr],color=col,label=f'p = {p}',s=32)
ax.plot([12,2200],[12,2200],'k--',lw=1,label='Equal epochs');ax.set(xscale='log',yscale='log',xlabel='A100 median epochs (10 seeds)',ylabel='Current H100 median epochs (3 other seeds)',title='Existing batch-sweep controls agree at sampled widths');ax.legend(frameon=False);ax.grid(alpha=.2,which='both')
fig.supxlabel('Threshold medians use seeds that reached that threshold; p = 139, width 76 has 2/10 crossings at 99%.', fontsize=9)
fig.savefig(OUT/'speed_diagnostics.png',dpi=180);fig.savefig(OUT/'speed_diagnostics.pdf');plt.close(fig)
save('diagnostics.json',{'unregistered_directories':len(orphan_rows),'unregistered_trace_files':sum(r['exists'] for r in orphan_rows),'unregistered_by_type':dict(collections.Counter(r['type'] for r in orphan_rows if r['exists'])),'unregistered_files_with_hardware_keys':sum(bool(r.get('hardware_keys')) for r in orphan_rows),'unregistered_errors':[r for r in orphan_rows if 'error'in r],'missing_completed_artifacts':[r for r in missing if 'missing' in r],'H100_control_rows':len(h),'H100_control_validation_failures':[r for r in hvalidation if not r['registry_match']],'H100_comparison_cells':len(comparison),'H100_uncensored_comparison_cells':len(ratios),'H100_A100_median_ratio':float(np.median(ratios)),'H100_A100_ratio_range':[min(ratios),max(ratios)],'H100_control_source':{'database':'/home/ubuntu/batch_sweep_runs.db','commit':sorted({r['git_hash'] for r in h}),'dirty':sorted({r['git_dirty'] for r in h}),'host':sorted({r['host'] for r in h}),'seeds':sorted({r['seed'] for r in h})}})

# Auxiliary NPZ files outside data/: retain device-only provenance as unverified hardware.
registry_by_uuid={r['uuid']:r for r in rows}
aux=[]
for base in ('docs','results'):
    for p in (ROOT/base).rglob('*.npz'):
        r={'path':str(p.relative_to(ROOT)),'host':None,'gpu':None,'device':None,'provenance_source':None}
        rr=registry_by_uuid.get(p.parent.name)
        if rr:
            r.update(host=rr['host'],gpu=rr['gpu_type'],provenance_source=rr['database'])
        else:
            for d in [p.parent,*p.parent.parents]:
                if d==ROOT:break
                mf=d/'manifest.json'
                if mf.exists():
                    x=json.loads(mf.read_text())
                    if x.get('host'):
                        r.update(host=x['host'],gpu=x.get('gpu'),device=x.get('device'),provenance_source=str(mf.relative_to(ROOT)));break
            sf=p.with_suffix('.json')
            if sf.exists() and not r['host']:
                x=json.loads(sf.read_text());r['device']=x.get('cfg',{}).get('device');r['provenance_source']=str(sf.relative_to(ROOT))
        aux.append(r)
csvwrite('auxiliary_trace_provenance.csv',aux)
print('auxiliary traces',len(aux),'without recorded host',sum(not r['host'] for r in aux),flush=True)
