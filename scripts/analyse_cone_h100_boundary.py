#!/usr/bin/env python3
"""Compare every preselected H100 replication against its archived A100 pair."""
from pathlib import Path
from collections import Counter,defaultdict
import json,sqlite3
import numpy as np
from cone_data import ROOT,aggregate,first_events,sha,write_json,write_csv
from plot_cone_seedmin import plot
CAMPAIGN=ROOT/'results/cone_h100_boundary_20261104';REPAIR=ROOT/'results/cone_repair_20261104'

def analyse():
    m=json.loads((CAMPAIGN/'manifest.json').read_text());ref=json.loads((CAMPAIGN/'reference_runs.json').read_text());done=json.loads((CAMPAIGN/'completion.json').read_text());checks=json.loads((CAMPAIGN/'checkpoint_verification.json').read_text())
    amendment=json.loads((CAMPAIGN/'seed_completion_manifest.json').read_text()) if (CAMPAIGN/'seed_completion_manifest.json').exists() else None
    expected=225+(amendment['additional_jobs'] if amendment else 0)
    if amendment:
        assert sha(CAMPAIGN/'manifest.json')==amendment['original_manifest_sha256']
        assert sha(ROOT/'scripts/complete_cone_neighbour_seeds.py')==amendment['controller_sha256']
        ref+=amendment['reference_runs'];done+=json.loads((CAMPAIGN/'seed_completion.json').read_text())
    assert len(done)==expected and all(r['returncode']==0 for r in done) and checks['runs_verified']==expected
    assert sha(CAMPAIGN/'reference_runs.json')==m['reference_runs_sha256']
    for rel,h in m['source_hashes'].items():assert sha(ROOT/rel)==sha(CAMPAIGN/'source'/rel)==h
    c=sqlite3.connect((CAMPAIGN/'runs.db').as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
    fresh=[dict(r) for r in c.execute('select * from runs')];c.close();assert len(fresh)==expected and all(r['status']=='completed' for r in fresh)
    idx={(r['p'],r['dim'],r['seed']):r for r in fresh};assert len(idx)==expected
    valid={r['uuid']:r for r in checks['checks']};assert len(valid)==expected
    old_rows=[];new_rows=[];paired=[];data_hashes={};runtime=[]
    for a in ref:
        key=a['p'],a['dim'],a['seed'];b=idx[key]
        for field in m['protocol']:assert a[field]==b[field]==m['protocol'][field]
        assert a['param_count']==b['param_count'] and a['host']=='147-224-250-73' and b['host']==m['host']
        assert sha(a['reference_trace'])==a['trace_sha256']
        with np.load(a['reference_trace'],allow_pickle=False) as z:old=first_events(z['train_acc'],z['val_acc'])
        with np.load(b['npz_path'],allow_pickle=False) as z:new=first_events(z['train_acc'],z['val_acc'])
        check=valid[b['uuid']];assert sha(b['npz_path'])==check['trace_sha256']
        assert all(new[k]==check[k] for k in new)
        meta=json.loads((CAMPAIGN/'jobs'/f'p{key[0]}_d{key[1]}_s{key[2]}.json').read_text())
        assert b['host']==meta['host'];runtime.append({k:meta[k] for k in ['host','gpu','gpu_uuid','torch','cuda','dtype','tf32_matmul','tf32_cudnn']})
        data_key=a['p'],a['seed']
        if data_key in data_hashes:assert data_hashes[data_key]==meta['dataset_hashes']
        else:data_hashes[data_key]=meta['dataset_hashes']
        old_rows.append({**a,**old});new_rows.append({**b,**new,'database':str(CAMPAIGN/'runs.db'),'trace_path':b['npz_path'],'trace_sha256':check['trace_sha256']})
        paired.append({'p':a['p'],'dim':a['dim'],'seed':a['seed'],'role':a['role'],'A100_uuid':a['uuid'],'H100_uuid':b['uuid'],'A100_train99':old['train99'],'H100_train99':new['train99'],'A100_val98':old['val98'],'H100_val98':new['val98'],'A100_delay':old['delay'],'H100_delay':new['delay'],'A100_state':old['state'],'H100_state':new['state'],'state_agreement':old['state']==new['state'],'delay_difference':new['delay']-old['delay'] if new['delay'] is not None and old['delay'] is not None else None})
    assert all(x==runtime[0] for x in runtime),'Runtime/hardware changed during campaign'
    old_cells=aggregate(old_rows);new_cells=aggregate(new_rows);old_idx={(c['p'],c['dim']):c for c in old_cells};cell_pairs=[]
    target_keys={(p,t) for p,t,ds,n in m['groups']}
    for c in new_cells:
        a=old_idx[c['p'],c['dim']];assert a['seeds']==c['seeds']
        cell_pairs.append({'p':c['p'],'dim':c['dim'],'role':'target' if (c['p'],c['dim']) in target_keys else 'neighbour','n_seeds':c['n_seeds'],'A100_state':a['state'],'H100_state':c['state'],'A100_minimum_delay':a['minimum_delay'],'H100_minimum_delay':c['minimum_delay'],'A100_mean_delay':a['mean_delay'],'H100_mean_delay':c['mean_delay'],'A100_positive_seeds':a['positive_seeds'],'H100_positive_seeds':c['positive_seeds'],'state_agreement':a['state']==c['state']})
    diffs=[r['delay_difference'] for r in paired if r['delay_difference'] is not None]
    result={'manifest_sha256':sha(CAMPAIGN/'manifest.json'),'runs':expected,'cells':24,'target_cells':8,'neighbour_cells':16,'checkpoint_checks':expected,'runtime':runtime[0],'source_and_trace_hashes_verified':True,'within_campaign_data_identical_across_widths_for_same_prime_seed':True,'historical_initial_checkpoint_available':False,'interpretation':'Matched architecture, optimiser, data-generation settings and seed IDs across historical A100 and current H100 runtimes. Historical initial checkpoints are unavailable; this is replication sensitivity, not an isolated causal estimate of GPU architecture.','run_state_counts':dict(Counter(r['state'] for r in new_rows)),'paired_run_agreement':sum(r['state_agreement'] for r in paired),'paired_cell_agreement':sum(r['state_agreement'] for r in cell_pairs),'target_cell_agreement':sum(r['state_agreement'] for r in cell_pairs if r['role']=='target'),'median_delay_difference':float(np.median(diffs)) if diffs else None,'delay_difference_quartiles':np.quantile(diffs,[.25,.75]).tolist() if diffs else None,'all_cell_comparisons':cell_pairs,'changed_cells':[r for r in cell_pairs if not r['state_agreement']]}
    result['seed_completion_amendment']=amendment['reason'] if amendment else None
    write_json(CAMPAIGN/'analysis.json',result);write_json(CAMPAIGN/'scored_runs.json',new_rows);write_json(CAMPAIGN/'scored_reference_runs.json',old_rows);write_json(CAMPAIGN/'cells.json',new_cells);write_csv(CAMPAIGN/'paired_runs.csv',paired);write_csv(CAMPAIGN/'paired_cells.csv',cell_pairs)
    comparison_plot(cell_pairs)
    # Use the whole completed preselected cohort, never a result-dependent subset.
    archive=json.loads((REPAIR/'runs.json').read_text());replace_keys={(r['p'],r['dim']) for r in new_rows}
    old_selected=[r for r in archive if (r['p'],r['dim']) in replace_keys]
    assert {(r['p'],r['dim'],r['seed']) for r in old_selected}==set(idx) and len(old_selected)==expected
    final_rows=[r for r in archive if (r['p'],r['dim']) not in replace_keys]+new_rows;final_cells=aggregate(final_rows)
    summary=json.loads((REPAIR/'summary.json').read_text())
    summary.update({'unique_runs':len(final_rows),'cells':len(final_cells),'run_state_counts':dict(Counter(r['state'] for r in final_rows)),'cell_state_counts':dict(Counter(r['state'] for r in final_cells)),'event_counts':dict(Counter(r['event_status'] for r in final_rows)),'budget_distribution':dict(sorted(Counter(r['max_epochs'] for r in final_rows).items())),'replication':{'campaign':str(CAMPAIGN),'replaced_archive_cells':24,'replaced_archive_runs':expected,'new_H100_runs':expected,'rule':'Every preselected cell is replaced in full, irrespective of result. No within-cell pooling.','analysis_sha256':sha(CAMPAIGN/'analysis.json')},'five_seed_sensitivity_scope':'The inherited five-seed sensitivity describes the archived interim cohort; final sensitivity is separately recorded below.'})
    summary['selected_rows']+=expected
    summary['source_reports'].append({'database':str(CAMPAIGN/'runs.db'),'host':m['host'],'gpu_type':'H100','mup_base':0,'selected_rows':expected,'replaces_archive_rows':expected,'protocol':m['protocol']})
    summary['retained_run_counts_by_host']=dict(Counter(r['host'] for r in final_rows))
    five=aggregate([r for r in final_rows if r['seed'] in range(42,47)]);five_idx={(r['p'],r['dim']):r for r in five}
    summary['final_five_seed_classification_changes']=[{'p':r['p'],'dim':r['dim'],'all_seed_state':r['state'],'five_seed_state':five_idx[r['p'],r['dim']]['state']} for r in final_cells if r['state']!=five_idx[r['p'],r['dim']]['state']]
    before_idx={(r['p'],r['dim']):r for r in json.loads((REPAIR/'cells.json').read_text())}
    summary['replication']['figure_classification_changes']=[{'p':r['p'],'dim':r['dim'],'archive_state':before_idx[r['p'],r['dim']]['state'],'final_state':r['state']} for r in final_cells if r['state']!=before_idx[r['p'],r['dim']]['state']]
    out=REPAIR/'final';out.mkdir(exist_ok=True);write_json(out/'runs.json',final_rows);write_json(out/'cells.json',final_cells);write_json(out/'summary.json',summary);write_csv(out/'cells.csv',final_cells);plot(final_cells,summary,out,fixed_only=True)
    return result

def comparison_plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,8),layout='constrained')
    ys=np.arange(len(rows));a=[r['A100_minimum_delay'] for r in rows];h=[r['H100_minimum_delay'] for r in rows]
    for y,x,z in zip(ys,a,h):
        if x is not None and z is not None:ax.plot([x,z],[y,y],color='#bdc3c9',lw=1.3,zorder=1)
    ax.scatter(a,ys-.10,s=27,color='#50738f',label='Archived A100',zorder=2);ax.scatter(h,ys+.10,s=27,color='#e27b32',label='Fresh H100',zorder=2)
    ax.set_yticks(ys,[f"p={r['p']}, d={r['dim']}"+('  *' if r['role']=='target' else '') for r in rows]);ax.invert_yaxis();ax.axvline(0,color='grey',lw=.8);ax.grid(axis='x',alpha=.25);ax.set_xlabel('Minimum observed delay across the matched seeds (epochs)');ax.set_title('Preselected cells: * affected cell, otherwise neighbouring width',fontsize=10);ax.legend();ax.spines[['top','right']].set_visible(False)
    fig.savefig(CAMPAIGN/'paired_cell_delays.pdf');fig.savefig(CAMPAIGN/'paired_cell_delays.png',dpi=180);plt.close(fig)
if __name__=='__main__':
    r=analyse();print(json.dumps({k:v for k,v in r.items() if k not in ['all_cell_comparisons']},indent=2))
