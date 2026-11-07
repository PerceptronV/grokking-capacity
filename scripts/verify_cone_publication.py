#!/usr/bin/env python3
"""Verify the completed cone cohort and its publication against archived inputs."""
from pathlib import Path
from collections import Counter,defaultdict
import datetime,hashlib,json,re,shutil
import numpy as np
import pymupdf
ROOT=Path(__file__).resolve().parents[1];REPAIR=ROOT/'results/cone_repair_20261104';CAMP=ROOT/'results/cone_h100_boundary_20261104';FINAL=REPAIR/'final';PAPER=ROOT/'docs/papers/2026 Grokking Capacity'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    a=json.loads((CAMP/'analysis.json').read_text());s=json.loads((FINAL/'summary.json').read_text());rows=json.loads((FINAL/'runs.json').read_text());cells=json.loads((FINAL/'cells.json').read_text());old=json.loads((REPAIR/'runs.json').read_text());fresh=json.loads((CAMP/'scored_runs.json').read_text());checks=json.loads((CAMP/'checkpoint_verification.json').read_text());manifest=json.loads((CAMP/'manifest.json').read_text())
    assert a['runs']==checks['runs_verified']==len(fresh)==235 and a['cells']==24
    assert s['unique_runs']==len(rows)==15825 and s['cells']==len(cells)==1717
    assert len({(r['p'],r['dim'],r['seed']) for r in rows})==15825
    keys={(r['p'],r['dim']) for r in fresh};removed={r['uuid'] for r in old if (r['p'],r['dim']) in keys};new_ids={r['uuid'] for r in fresh}
    assert len(removed)==len(new_ids)==235
    assert {r['uuid'] for r in rows}==({r['uuid'] for r in old}-removed)|new_ids
    by=defaultdict(list)
    for r in rows:by[r['p'],r['dim']].append(r)
    for c in cells:
        rr=by[c['p'],c['dim']];assert len({(r['host'],r['gpu_type']) for r in rr})==1
        assert len(rr)==c['n_seeds'] and len({r['seed'] for r in rr})==len(rr)
        if c['state']=='positive':assert all(r['train99'] is not None and r['val98'] is not None and r['val98']>r['train99'] for r in rr)
        if c['state']=='zero':assert any(r['val98'] is not None and (r['train99'] is None or r['val98']<=r['train99']) for r in rr)
    metas=[json.loads(p.read_text()) for p in sorted((CAMP/'jobs').glob('*.json'))];assert len(metas)==235
    assert len({m['gpu_uuid'] for m in metas})==1 and manifest['physical_gpu']==0 and manifest['workers']==2
    for rel,h in manifest['source_hashes'].items():assert sha(ROOT/rel)==sha(CAMP/'source'/rel)==h
    pdf=PAPER/'paper-neurips-revised.pdf';doc=pymupdf.open(pdf);text=' '.join(' '.join(p.get_text().split()) for p in doc)
    assert '1,717' in text and '15,825' in text and 'Matched cone replication on one H100' in text
    assert 'empty by 193' not in text and 'no cell was budget-censored' not in text
    log=(FINAL/'build/pass3.log').read_text();assert not re.search(r'(undefined references|undefined citations|Citation .* undefined|Reference .* undefined)',log,re.I)
    old_source=(REPAIR/'before/docs/papers/2026 Grokking Capacity/paper-neurips-revised.tex').read_text();new_source=(PAPER/'paper-neurips-revised.tex').read_text()
    for label in ['tab:hypothesis-tests','tab:crossing']:
        aa=old_source.index('\\label{'+label+'}');bb=new_source.index('\\label{'+label+'}')
        assert old_source[old_source.rfind('\\begin{table',0,aa):old_source.index('\\end{table',aa)]==new_source[new_source.rfind('\\begin{table',0,bb):new_source.index('\\end{table',bb)]
    norm_check=json.loads((ROOT/'results/normfrozen_h100_paired_20261103/wording_update_20261104/validation.json').read_text());assert sha(PAPER/'theory/principled.tex')==norm_check['source_sha256']
    assert sha(FINAL/'cone_seedmin_fixed.pdf')==sha(PAPER/'media/theory/cone_seedmin_fixed.pdf')==sha(ROOT/'figures/theory/cone_seedmin_fixed.pdf')
    assert sha(pdf)==sha(FINAL/'build/paper-neurips-revised.pdf')
    observed_minutes=(max(m['finished_at'] for m in metas)-min(m['started_at'] for m in metas))/60
    result={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'training_runs':235,'completed_and_CPU_checked':235,'physical_gpus_used':1,'physical_gpu_index':0,'gpu_uuid':metas[0]['gpu_uuid'],'training_elapsed_minutes':observed_minutes,'maximum_cpu_gpu_accuracy_difference':max(abs(c['cpu_validation_accuracy']-c['recorded_validation_accuracy']) for c in checks['checks']),'full_preselected_cohort_used':True,'within_cell_hardware_pooling':False,'duplicate_seed_weighting':False,'source_snapshots_verified':True,'main_test_and_crossing_tables_unchanged':True,'approved_norm_freezing_wording_unchanged':True,'paper_pages':len(doc),'paper_sha256':sha(pdf),'figure_sha256':sha(FINAL/'cone_seedmin_fixed.pdf'),'undefined_references_or_citations':False,'overfull_boxes':re.findall(r'Overfull[^\n]+',log),'classification_agreement':{'runs':a['paired_run_agreement'],'cells':a['paired_cell_agreement'],'target_cells':a['target_cell_agreement']},'changed_cells':a['changed_cells'],'final_figure_states':s['cell_state_counts']}
    (FINAL/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    src=FINAL/'analysis_source';src.mkdir(exist_ok=True)
    for name in ['cone_data.py','plot_cone_seedmin.py','run_cone_h100_boundary.py','verify_cone_h100_boundary.py','analyse_cone_h100_boundary.py','update_cone_paper.py','finish_cone_repair.py','verify_cone_publication.py','complete_cone_neighbour_seeds.py']:shutil.copy2(ROOT/'scripts'/name,src/name)
    for page in doc:
        if 'The measured grokking region' in page.get_text():page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(FINAL/'paper_cone_page.png')
        if 'Matched cone replication on one H100' in page.get_text():page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(FINAL/'paper_replication_page.png')
    report=f'''FINAL CONE REPAIR AND H100 REPLICATION

235/235 jobs completed on physical H100 GPU0, two workers. Training elapsed
{observed_minutes:.2f} minutes. All235 checkpoints independently evaluated on CPU;
all data and initial-state hashes verified, and all first crossings recomputed.
Current training source matches its immutable pre-run snapshot. The original225
were supplemented with10 matching neighbour seeds to preserve the archived
seed counts at p139,d220 and d230. The original plan/results and the explicit
seed-completion amendment are retained.

Paired replication
All24 cells and235 matching seed IDs are reported, without selection by outcome.
Cell classifications agree in {a['paired_cell_agreement']}/24 cells;
affected target cells agree in {a['target_cell_agreement']}/8 cells;
individual seed classifications agree in {a['paired_run_agreement']}/235 pairs.
Median H100-minus-A100 delay: {a['median_delay_difference']} epochs;
interquartile range: {a['delay_difference_quartiles']}.
Matched code equivalence was independently checked under the current runtime.
Historical initial checkpoints were not available: this is replication across
historical/current runtimes, not an isolated causal GPU-architecture experiment.

Final plot
The235 fresh measurements replace all24 preselected archival cells in full.
The archive-only interim plot remains at results/cone_repair_20261104/.
The final plot has1717 cells and15825 unique seed observations. Each cell
has one hardware cohort and one observation per seed. State counts:
{s['cell_state_counts']}.
No censored delay is presented as completed grokking. The same-five-seed
sensitivity changes {len(s['final_five_seed_classification_changes'])} classifications.

Publication
All24 paired cell minima appear in the paper's replication table. The main
onset/crossing tests remain unchanged. The opening figure caption, methods,
and directly related interpretation now distinguish zero minimum, unresolved
outcomes, finite seed counts, and the five-seed p197,d316 extension from the
original18-width pre-registered test. The paper has {len(doc)} pages and no
undefined references or citations. The previously approved norm-freezing
wording is unchanged.

Artifacts
Final cone: results/cone_repair_20261104/final/cone_seedmin_fixed.pdf
Paired plot: results/cone_h100_boundary_20261104/paired_cell_delays.pdf
All paired rows: results/cone_h100_boundary_20261104/paired_runs.csv
All paired cells: results/cone_h100_boundary_20261104/paired_cells.csv
Full validation: results/cone_repair_20261104/final/validation.json

Reproduce the final figure from audited results:
python scripts/plot_cone_seedmin.py --fixed-only --final --publish
Recompute the paired analysis and final cohort:
python scripts/analyse_cone_h100_boundary.py
python scripts/update_cone_paper.py --analysis results/cone_repair_20261104/final --final
'''
    (FINAL/'report.txt').write_text(report)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
