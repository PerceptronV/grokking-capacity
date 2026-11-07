#!/usr/bin/env python3
"""Publish the fully verified p199 extension without changing earlier cutoff corrections."""
from pathlib import Path
import argparse,datetime,difflib,json,os,re,shutil,subprocess,sys
from cone_data import sha,write_json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/cone_p199_h100_20261105';FIG=OUT/'figure';PAPER=ROOT/'docs/papers/2026 Grokking Capacity'
FILES=['paper-neurips-revised.tex','paper-neurips-revised.pdf','theory/theory-section.tex','theory/principled.tex','theory/consequences.tex','theory/extra.tex','media/theory/cone_seedmin_fixed.pdf']
def prepare():
    dest=OUT/'publication_before'
    if dest.exists():return
    hashes={}
    for rel in FILES:
        target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PAPER/rel,target);hashes[rel]=sha(PAPER/rel)
    write_json(OUT/'publication_base_manifest.json',hashes)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');args=ap.parse_args()
    if args.prepare:prepare();return
    a=json.loads((OUT/'analysis.json').read_text());s=json.loads((FIG/'summary.json').read_text());m=json.loads((OUT/'manifest.json').read_text());baseline=json.loads((OUT/'publication_base_manifest.json').read_text())
    assert a['runs']==190 and a['cells']==19
    for rel,h in baseline.items():assert sha(PAPER/rel)==h,('Paper changed since snapshot',rel)
    source=(PAPER/'paper-neurips-revised.tex').read_text();new=source
    def change(old,value):
        nonlocal new
        assert new.count(old)==1,(old,new.count(old));new=new.replace(old,value)
    change('Twenty-one primes, $1{,}717$ cells, $15{,}825$ distinct seed observations','Twenty-two primes, $1{,}736$ cells, $16{,}015$ distinct seed observations')
    change(r'''$p=197$ grid has zero minimum delay at all eighteen widths. The upper''',r'''$p=197$ grid has zero minimum delay at all eighteen widths. The $p=199$
    extension measures nineteen widths with ten seeds each. The upper''')
    change(r'''There are $1{,}448$ ten-seed cells and $269$ five-seed cells''',r'''There are $1{,}467$ ten-seed cells and $269$ five-seed cells''')
    change(r'''Restricting the archive to the same five seeds ($42$--$46$)
in every cell changes $63$ of $1{,}717$ classifications''',r'''Before the $p=199$ extension, restricting the archive to the same five seeds
($42$--$46$) in every cell changes $63$ of $1{,}717$ classifications''')
    e=s['event_counts']
    change(r'''The retained cohort includes $340$ runs with neither threshold observed and $1$ with training observed but validation unresolved; $81$ runs establish zero delay by reaching validation before an unobserved training threshold.''',f"The retained cohort includes ${e.get('both_thresholds_unobserved',0)}$ runs with neither threshold observed and ${e.get('validation_right_censored',0)}$ with training observed but validation unresolved; ${e.get('validation_before_unobserved_training',0)}$ runs establish zero delay by reaching validation before an unobserved training threshold.")
    change(r'''The same-five-seed restriction changes $63$ cell classifications in the final plotted cohort.''',f"The same-five-seed restriction changes ${len(s['final_five_seed_classification_changes'])}$ cell classifications in the extended plotted cohort.")
    n=a['p199_cell_states'];events=a['p199_event_counts'];states=a['p199_run_states']
    para=r'''

\paragraph{Extension to $p=199$.} Before collecting this column, we fixed
nineteen widths: the original eighteen-width $p=197$ grid and $d=316$,
with ten seeds ($42$--$51$) at every width. The $190$ runs use the same
fixed-rate protocol on one H100, with a $5{,}000$-epoch cap and the scoring
and censoring rules above. This new extension is separate from the historical
pre-registration and covers $d=52$--$556$ (approximately $10^5$--$10^7$
parameters). All planned runs are included; the earlier columns and fitted
laws are unchanged. Every new checkpoint was independently evaluated on CPU.
'''
    para+=f"Of the $19$ cells, ${n.get('positive',0)}$ have positive delay in all ten seeds, ${n.get('zero',0)}$ have at least one observed zero, and ${n.get('unresolved',0)}$ remain unresolved. Across all $190$ seeds, ${states.get('positive',0)}$ have observed positive delay, ${states.get('zero',0)}$ have zero delay, and ${states.get('unresolved',0)}$ have unresolved delay. These finite-grid outcomes do not establish disappearance at every width or in every seed.\n"
    para+=f"Restricting the new column to the common five seeds ($42$--$46$) changes ${len(a['five_seed_classification_changes'])}$ of its $19$ cell classifications.\n"
    all_new={c['dim']:c for c in json.loads((OUT/'cells.json').read_text())}
    five_new={c['dim']:c for c in json.loads((FIG/'five_seed_cells.json').read_text()) if c['p']==199}
    for changed in a['five_seed_classification_changes']:
        d=changed['dim'];ten=all_new[d];five=five_new[d]
        fmt=lambda value:r'\text{unresolved}' if value is None else str(value)
        para+=f"At $d={d}$, the minimum is ${fmt(five['minimum_delay'])}$ epoch with the common five seeds and ${fmt(ten['minimum_delay'])}$ with all ten; ${ten['positive_seeds']}$ of the ten seeds have positive delay.\n"
    anchor=r'''All preselected comparisons are reported in \cref{tab:cone-replication}.'''
    change(anchor,anchor+para)
    def table(text,label):
        i=text.index('\\label{'+label+'}');return text[text.rfind('\\begin{table',0,i):text.index('\\end{table',i)]
    for label in ['tab:hypothesis-tests','tab:crossing','tab:cone-replication']:assert table(source,label)==table(new,label)
    assert 'so an exact cutoff is not established' in new
    (OUT/'paper_changes.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),new.splitlines(True),fromfile='before/paper-neurips-revised.tex',tofile='after/paper-neurips-revised.tex')))
    (PAPER/'paper-neurips-revised.tex').write_text(new)
    for parent in [PAPER/'media/theory',ROOT/'figures/theory']:shutil.copy2(FIG/'cone_seedmin_fixed.pdf',parent/'cone_seedmin_fixed.pdf')
    build=OUT/'build';build.mkdir(exist_ok=True)
    cmd=['pdflatex','-interaction=nonstopmode','-halt-on-error',f'-output-directory={build}','paper-neurips-revised.tex']
    for i in range(1,4):
        with (build/f'pass{i}.log').open('w') as f:subprocess.run(cmd,cwd=PAPER,stdout=f,stderr=subprocess.STDOUT,check=True)
        if i==1:
            with (build/'bibtex.log').open('w') as f:subprocess.run(['bibtex','paper-neurips-revised'],cwd=build,env={**os.environ,'BIBINPUTS':f'{PAPER}:','BSTINPUTS':f'{PAPER}:'},stdout=f,stderr=subprocess.STDOUT,check=True)
    log=(build/'pass3.log').read_text();assert not re.search(r'(undefined references|undefined citations|Citation .* undefined|Reference .* undefined)',log,re.I)
    import pymupdf
    doc=pymupdf.open(build/'paper-neurips-revised.pdf');text=' '.join(' '.join(p.get_text().split()) for p in doc)
    assert '16,015' in text and '1,736' in text and 'so an exact cutoff is not established' in text
    for rel,h in baseline.items():
        if rel.startswith('theory/'):assert sha(PAPER/rel)==h
    for page in doc:
        if 'The measured grokking region' in page.get_text():page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(OUT/'paper_cone_page.png')
        if 'Extension to p = 199' in ' '.join(page.get_text().split()):page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(OUT/'paper_extension_page.png')
    shutil.copy2(build/'paper-neurips-revised.pdf',PAPER/'paper-neurips-revised.pdf')
    result={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':190,'cells':19,'total_cells':1736,'total_runs':16015,'primes':22,'physical_gpus':1,'gpu_uuid':a['runtime']['gpu_uuid'],'checkpoint_CPU_evaluations':190,'main_test_crossing_and_replication_tables_unchanged':True,'earlier_theory_and_cutoff_corrections_unchanged':True,'all_previous_observations_unchanged':True,'no_fitted_laws_changed':True,'p199_cell_states':n,'p199_run_states':states,'paper_pages':len(doc),'paper_sha256':sha(PAPER/'paper-neurips-revised.pdf'),'paper_source_sha256':sha(PAPER/'paper-neurips-revised.tex'),'figure_sha256':sha(FIG/'cone_seedmin_fixed.pdf'),'undefined_references_or_citations':False,'overfull_boxes':re.findall(r'Overfull[^\n]+',log)}
    write_json(OUT/'validation.json',result)
    write_json(ROOT/'results/cone_latest.json',{'analysis_directory':str(FIG.relative_to(ROOT)),'validation':str((OUT/'validation.json').relative_to(ROOT)),'figure_pdf':str((FIG/'cone_seedmin_fixed.pdf').relative_to(ROOT)),'figure_png':str((FIG/'cone_seedmin_fixed.png').relative_to(ROOT)),'paper':str((PAPER/'paper-neurips-revised.pdf').relative_to(ROOT)),'published_utc':result['verified_utc']})
    (OUT/'report.txt').write_text('P199 CONE EXTENSION\n\nAll190 planned runs at19 preselected widths completed on one H100 and independently checked on CPU. The full190-run cohort is appended, without outcome-dependent selection, to the prior15825 observations. All earlier columns and their underlying data are unchanged.\n\n'+json.dumps(a,indent=2)+'\n\nThe paper contains updated figure counts, seed/censoring counts and the new extension methods. Main numerical tests, fitted laws and the previous cutoff-claim corrections are unchanged.\n\nReproduce analysis: python scripts/analyse_cone_p199.py\nLatest figure: results/cone_p199_h100_20261105/figure/cone_seedmin_fixed.pdf\nLatest publication pointer: results/cone_latest.json\n')
    src=OUT/'analysis_source';src.mkdir(exist_ok=True)
    for name in ['cone_data.py','plot_cone_seedmin.py','verify_cone_p199.py','verify_cone_h100_boundary.py','analyse_cone_p199.py','publish_cone_p199.py']:shutil.copy2(ROOT/'scripts'/name,src/name)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
