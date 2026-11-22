#!/usr/bin/env python3
"""Append the complete preselected p199 cohort to the audited cone."""
from pathlib import Path
from collections import Counter
import json,sqlite3
from grokking_capacity.experiments.cone_workflow.cone_data import aggregate, score, sha, write_json, write_csv
from grokking_capacity.experiments.cone_workflow.plot_cone_seedmin import plot
ROOT=Path(__file__).resolve().parents[4];OUT=ROOT/'results/cone_p199_h100_20261105';FIG=OUT/'figure'
def analyse():
    m=json.loads((OUT/'manifest.json').read_text());base=ROOT/m['base_cohort']
    for name,h in m['base_cohort_hashes'].items():assert sha(base/name)==h
    for rel,h in m['source_hashes'].items():assert sha(ROOT/rel)==sha(OUT/'source'/rel)==h
    done=json.loads((OUT/'completion.json').read_text());checks=json.loads((OUT/'checkpoint_verification.json').read_text())
    assert len(done)==m['jobs']==checks['runs_verified']==190 and all(r['returncode']==0 for r in done)
    con=sqlite3.connect((OUT/'runs.db').as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
    records=[dict(r) for r in con.execute('select * from runs')];con.close()
    assert len(records)==190 and {(r['p'],r['dim'],r['seed']) for r in records}=={(199,d,s) for d in m['widths'] for s in m['seeds']}
    fresh=[];runtime=[];data_hashes={};check_idx={r['uuid']:r for r in checks['checks']}
    for r in records:
        assert r['status']=='completed' and r['host']==m['host'] and r['gpu_type']=='H100'
        assert r['n_samples']==199*198//2
        for k,v in m['protocol'].items():assert r[k]==v,(r['uuid'],k,r[k],v)
        s=score({**r,'database':str(OUT/'runs.db')});check=check_idx[r['uuid']]
        for k in ['trace_sha256','train99','val98','state','delay']:assert s[k]==check[k]
        fresh.append(s)
        meta=json.loads((OUT/'jobs'/f'p199_d{r["dim"]}_s{r["seed"]}.json').read_text())
        runtime.append({k:meta[k] for k in ['host','gpu','gpu_uuid','torch','cuda','dtype','tf32_matmul','tf32_cudnn']})
        if r['seed'] in data_hashes:assert data_hashes[r['seed']]==meta['dataset_hashes']
        else:data_hashes[r['seed']]=meta['dataset_hashes']
    assert all(v==runtime[0] for v in runtime)
    old=json.loads((base/'runs.json').read_text());assert all(r['p']!=199 for r in old)
    rows=old+fresh;cells=aggregate(rows);new_cells=aggregate(fresh)
    old_cells=json.loads((base/'cells.json').read_text())
    assert [c for c in cells if c['p']!=199]==old_cells
    five=aggregate([r for r in rows if r['seed'] in range(42,47)]);five_idx={(r['p'],r['dim']):r for r in five}
    changes=[{'p':r['p'],'dim':r['dim'],'all_seed_state':r['state'],'five_seed_state':five_idx[r['p'],r['dim']]['state']} for r in cells if r['state']!=five_idx[r['p'],r['dim']]['state']]
    s=json.loads((base/'summary.json').read_text())
    s.update({'selected_rows':s['selected_rows']+190,'unique_runs':len(rows),'cells':len(cells),'primes':sorted({r['p'] for r in rows}),'event_counts':dict(Counter(r['event_status'] for r in rows)),'run_state_counts':dict(Counter(r['state'] for r in rows)),'cell_state_counts':dict(Counter(r['state'] for r in cells)),'seed_count_distribution':dict(sorted(Counter(r['n_seeds'] for r in cells).items())),'budget_distribution':dict(sorted(Counter(r['max_epochs'] for r in rows).items())),'retained_run_counts_by_host':dict(Counter(r['host'] for r in rows)),'final_five_seed_classification_changes':changes,'p199_extension':{'manifest_sha256':sha(OUT/'manifest.json'),'runs':190,'cells':19,'runtime':runtime[0],'selection':m['selection']}})
    s['archived_five_seed_sensitivity']=s['five_seed_sensitivity']
    s['five_seed_sensitivity']={'common_seeds':[42,43,44,45,46],'cells':len(five),'incomplete_five_seed_cells':sum(c['n_seeds']!=5 for c in five),'classification_changes':changes}
    s['five_seed_sensitivity_scope']='Current extended cohort; the original archive sensitivity is retained separately.'
    s['source_reports'].append({'database':str(OUT/'runs.db'),'host':m['host'],'gpu_type':'H100','mup_base':0,'selected_rows':190,'protocol':m['protocol']})
    assert s['unique_runs']==16015 and s['cells']==1736 and len(s['primes'])==22
    FIG.mkdir(exist_ok=True)
    write_json(OUT/'scored_runs.json',fresh);write_json(OUT/'cells.json',new_cells);write_csv(OUT/'runs.csv',fresh);write_csv(OUT/'cells.csv',new_cells)
    for name,data in [('runs',rows),('cells',cells),('summary',s),('five_seed_cells',five)]:write_json(FIG/(name+'.json'),data)
    write_csv(FIG/'cells.csv',cells);plot(cells,s,FIG,fixed_only=True)
    # Keep seed-count sensitivity as numerical diagnostics; publish only the primary figure.
    write_json(FIG/'common_five_summary.json',{'seeds':[42,43,44,45,46],'cells':len(five),'unique_observations':5*len(five),'cell_state_counts':dict(Counter(c['state'] for c in five)),'classification_changes_from_all_retained_seeds':changes})

    p197_idx={r['dim']:r for r in old_cells if r['p']==197};comparison=[]
    for r in new_cells:
        prev=p197_idx[r['dim']];comparison.append({'dim':r['dim'],'p197_n':prev['n_seeds'],'p199_n':r['n_seeds'],'p197_state':prev['state'],'p199_state':r['state'],'p197_minimum':prev['minimum_delay'],'p199_minimum':r['minimum_delay'],'p197_positive_seeds':prev['positive_seeds'],'p199_positive_seeds':r['positive_seeds'],'p197_five_seed_state':five_idx[197,r['dim']]['state'],'p199_five_seed_state':five_idx[199,r['dim']]['state']})
    write_csv(OUT/'p197_p199_comparison.csv',comparison)
    result={'runs':190,'cells':19,'runtime':runtime[0],'p199_event_counts':dict(Counter(r['event_status'] for r in fresh)),'p199_run_states':dict(Counter(r['state'] for r in fresh)),'p199_cell_states':dict(Counter(r['state'] for r in new_cells)),'positive_cells':[r for r in new_cells if r['state']=='positive'],'five_seed_classification_changes':[r for r in changes if r['p']==199],'positive_delays':[r['delay'] for r in fresh if r['state']=='positive'],'all_prior_rows_and_cells_unchanged':True,'prior_p197_d316_unchanged':True,'total_runs':len(rows),'total_cells':len(cells),'total_primes':len(s['primes']),'maximum_CPU_GPU_validation_difference':max(abs(r['cpu_validation_accuracy']-r['recorded_validation_accuracy']) for r in checks['checks'])}
    result['CPU_verification_backend_counts']=dict(Counter(r.get('cpu_linear_backend','torch') for r in checks['checks']))
    write_json(OUT/'analysis.json',result);return result
if __name__=='__main__':print(json.dumps(analyse(),indent=2))
