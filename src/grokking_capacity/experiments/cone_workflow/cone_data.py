"""Explicit cohorts and first-passage/censoring rules for the cone figure."""
from grokking_capacity.analysis.timescale_laws import normalise_timescale_laws
from pathlib import Path
from collections import defaultdict,Counter
import concurrent.futures,csv,hashlib,json,math,sqlite3
import numpy as np
from grokking_capacity.analysis.cone import first_events, aggregate
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'results/cone_repair_20261104'
ARCHIVE_HOST='147-224-250-73'
H100_HOST='68-209-74-145'
COMMON={'experiment_type':'groks','status':'completed','operation':'/','train_fraction':.5,'split_type':'random','dataset_type':'modular','dropout':.2,'init_scale':1.,'weight_decay':1.,'lr':.001,'beta1':.9,'beta2':.98,'batch_size':512,'depth':2,'heads':1,'architecture_family':'transformer_gated'}
SOURCES={0:[('runs.db',ARCHIVE_HOST,'A100'),*[(str(Path.home()/(name+'_runs.db')),H100_HOST,'H100') for name in ['cone_fill','band_gaps','p197','crossing_fixed','lr_wall_check']]],236:[(str(Path.home()/(name+'_runs.db')),H100_HOST,'H100') for name in ['mup_depth_series','mup_laws','crossing_mup','lr_wall_check','open_top']]}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def write_csv(p,rows):
    rows=list(rows)
    if not rows:return
    with Path(p).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def score(r):
    path=Path(r['npz_path'])
    if not path.exists():path=ROOT/'data/groks'/r['uuid']/'trace.npz'
    with np.load(path,allow_pickle=False) as z:
        e=first_events(z['train_acc'],z['val_acc'])
        va99=np.flatnonzero(z['val_acc']>=99);v99=int(va99[0]+1) if len(va99) else None
        assert v99==r['grokking_epoch'],(r['uuid'],v99,r['grokking_epoch'])
    return {**r,**e,'trace_path':str(path),'trace_sha256':sha(path)}
def build(out=OUT,panels=(0,)):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);selected=[];source_reports=[]
    for mb in panels:
        for db,host,gpu in SOURCES[mb]:
            path=Path(db);path=path if path.is_absolute() else ROOT/path
            con=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
            protocol={**COMMON,'mup_base':mb,'host':host,'gpu_type':gpu}
            rr=[{**dict(r),'database':str(path)} for r in con.execute('select * from runs where '+' and '.join(k+'=?' for k in protocol),list(protocol.values()))];con.close()
            assert all(r['n_samples']==r['p']*(r['p']-1)//2 for r in rr)
            selected.extend(rr);source_reports.append({'database':str(path),'host':host,'gpu_type':gpu,'mup_base':mb,'selected_rows':len(rr),'protocol':protocol})
    write_json(out/'selected_registry_rows.json',selected)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:scored=list(ex.map(score,selected))
    # A run UUID can only be an alias, never an additional seed.
    by_uuid={};aliases=[]
    for r in scored:
        if r['uuid'] in by_uuid:
            assert r['trace_sha256']==by_uuid[r['uuid']]['trace_sha256'];aliases.append(r['uuid'])
        else:by_uuid[r['uuid']]=r
    by_seed=defaultdict(list)
    for r in by_uuid.values():by_seed[r['mup_base'],r['p'],r['dim'],r['seed'],r['host']].append(r)
    retained=[];duplicates=[]
    for key,rr in sorted(by_seed.items()):
        # Outcome-independent rule: largest planned budget, then earliest creation, then UUID.
        rr.sort(key=lambda r:(-r['max_epochs'],str(r.get('created_at') or ''),r['uuid']))
        keep=rr[0];retained.append(keep)
        for other in rr[1:]:
            same=all(other[k]==keep[k] for k in ['train99','val98','state','delay'])
            duplicates.append({'kept_uuid':keep['uuid'],'excluded_uuid':other['uuid'],'p':keep['p'],'dim':keep['dim'],'seed':keep['seed'],'identical_first_passages':same,'kept_budget':keep['max_epochs'],'excluded_budget':other['max_epochs']})
            if not same:raise RuntimeError(f'Conflicting repeated seed {key}; resolve explicitly before publishing')
    cells=aggregate(retained)
    sensitivity=aggregate([r for r in retained if r['seed'] in range(42,47)])
    by_sens={(c['mup_base'],c['p'],c['dim']):c for c in sensitivity}
    changes=[{'mup_base':c['mup_base'],'p':c['p'],'dim':c['dim'],'all_seed_state':c['state'],'five_seed_state':by_sens[c['mup_base'],c['p'],c['dim']]['state']} for c in cells if (c['mup_base'],c['p'],c['dim']) in by_sens and c['state']!=by_sens[c['mup_base'],c['p'],c['dim']]['state']]
    result={'source_reports':source_reports,'selected_rows':len(selected),'unique_runs':len(retained),'duplicate_seed_observations_removed':len(duplicates),'uuid_aliases_removed':len(aliases),'deduplication_rule':'Highest planned budget, then earliest creation time, then UUID. Conflicting first passages abort publication.','thresholds':{'train':99,'val':98,'train_mode':'online with dropout','val_mode':'evaluation'},'state_rules':{'positive':'Both thresholds observed with positive delay for every retained seed.','zero':'At least one retained seed has validation98 no later than train99, including validation98 before an unobserved train99. This refutes positive delay for every seed, not grokking in all other seeds.','unresolved':'No observed zero-delay seed, but at least one seed has an unobserved required threshold. Never counted as successful delayed generalisation.'},'event_counts':dict(Counter(r['event_status'] for r in retained)),'run_state_counts':dict(Counter(r['state'] for r in retained)),'cell_state_counts':dict(Counter(c['state'] for c in cells)),'cells':len(cells),'primes':sorted({c['p'] for c in cells}),'seed_count_distribution':dict(sorted(Counter(c['n_seeds'] for c in cells).items())),'budget_distribution':dict(sorted(Counter(r['max_epochs'] for r in retained).items())),'five_seed_sensitivity':{'common_seeds':[42,43,44,45,46],'cells':len(sensitivity),'incomplete_five_seed_cells':sum(c['n_seeds']!=5 for c in sensitivity),'classification_changes':changes},'selected_registry_sha256':sha(out/'selected_registry_rows.json'),'analysis_source_sha256':sha(__file__)}
    write_json(out/'runs.json',retained);write_json(out/'cells.json',cells);write_json(out/'summary.json',result);write_json(out/'duplicates.json',duplicates);write_json(out/'five_seed_cells.json',sensitivity)
    write_csv(out/'cells.csv',cells);write_csv(out/'runs.csv',({k:r[k] for k in ['uuid','database','host','gpu_type','p','dim','seed','mup_base','max_epochs','git_hash','git_dirty','train99','val98','followup_epochs','state','event_status','delay','delay_lower_bound','trace_path','trace_sha256']} for r in retained))
    return result,cells

def law_coefficients():
    path=ROOT/'results/capacity_h100_matched_20261103/forecast_coefficients.json'
    r=normalise_timescale_laws(json.loads(path.read_text()));historical=normalise_timescale_laws(json.loads((ROOT/'results/theory/forecast_onset.json').read_text()))
    assert np.isclose(r['mem_law']['gamma']/r['C_convention'],historical['mem_law']['gamma']/historical['C_convention'],rtol=1e-12)
    return r

def law_onset(p,r):
    from scipy.optimize import brentq
    g=r['mem_law']['gamma'];C=r['C_convention'];lnB=r['mem_law']['ln_B'];A=r['gen_law']['lnA'];a=r['gen_law']['a'];b=r['gen_law']['b'];K=p*(p-1)/2*math.log2(p+2)
    rhs=A-lnB+a*math.log(p)-b*math.log(g*K/C)
    if rhs<b*(1-math.log(b)):return None
    x=brentq(lambda x:x-b*math.log(x)-rhs,b,1000)
    return g*K/(C*x)
