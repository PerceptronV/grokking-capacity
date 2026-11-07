#!/usr/bin/env python3
"""Publish the audited uniform-ten-seed cone and narrowly update its paper claims."""
from pathlib import Path
import argparse,datetime,difflib,hashlib,json,os,re,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/cone_ten_seeds_h100_20261106'; FIG=OUT/'figure'
PAPER=ROOT/'docs/papers/2026 Grokking Capacity'
TEX=['paper-neurips-revised.tex','theory/theory-section.tex','theory/principled.tex','theory/consequences.tex','theory/extra.tex']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def tabular(text,label):
    start=text.index('\\label{'+label+'}')
    start=text.index('\\begin{tabular}',start)
    return text[start:text.index('\\end{tabular}',start)+len('\\end{tabular}')]
def prepare():
    baseline=json.loads((OUT/'publication_base_manifest.json').read_text())['files']
    for rel,digest in baseline.items():assert sha(PAPER/rel)==digest,('Concurrent paper change',rel)
    old={rel:(PAPER/rel).read_text() for rel in TEX}; new=dict(old)
    def change(rel,a,b):
        assert new[rel].count(a)==1,(rel,a,new[rel].count(a));new[rel]=new[rel].replace(a,b)
    main=TEX[0]; theory=TEX[1]; principled=TEX[2]; consequences=TEX[3]; extra=TEX[4]
    change(main,'a five-seed extension nevertheless has positive delays','individual seeds nevertheless have positive delays')
    change(main,r'''$16{,}015$ distinct seed observations''',r'''$17{,}360$ distinct seed observations''')
    change(main,'cohort and five or ten distinct seeds.','cohort and ten distinct seeds.')
    change(main,'''crosses: unresolved. Colour: mean delay in filled cells. Unreached
    thresholds are retained as censored''',r'''crosses: at least one unresolved seed, including in open cells. Colour:
    mean delay in filled cells. Unobserved validation-$98\%$ crossings are
    retained as censored''')
    change(main,'''No measured cell at $p=193$ has positive delay at every seed; an extension
    cell at $p=197$ has delays of one to two epochs.''',r'''No measured cell at $p=193$, $197$, or $199$ has positive delay in all ten
    seeds. At $p=197,d=316$, eight seeds have delays of one to two epochs
    and two have zero delay.''')
    change(main,'However, an additional width has positive delay in all five seeds, contradicting exact disappearance under the positive-delay definition.','However, an additional width has positive delay in eight of ten seeds and zero minimum delay; these finite-grid minima do not establish disappearance at every width or in every seed.')
    change(main,'A held-out onset at $p = 149$;','A historical held-out onset at $p = 149$;')
    change(main,r'''There are $1{,}467$ ten-seed cells and $269$ five-seed cells, with''',r'''All $1{,}736$ cells have ten seeds ($42$--$51$), with''')
    change(main,'''has not crossed, we retain a right-censored outcome: it is never counted as
successful delayed generalisation.''','''has not crossed, we retain a right-censored outcome: it is never counted as
successful delayed generalisation or as an observed zero.''')
    change(main,'''zero; otherwise the cell is unresolved. Mean-delay colours use only complete
filled cells. Before the $p=199$ extension, restricting the archive to the same five seeds
($42$--$46$) in every cell changes $63$ of $1{,}717$ classifications; the all-seed region''',r'''zero; otherwise the cell is unresolved. Crosses mark every cell containing
an unresolved seed, including cells whose minimum is fixed at zero by another
seed. Mean-delay colours use only complete filled cells. Restricting the
current cohort to the same five seeds ($42$--$46$) in every cell changes $76$
of $1{,}736$ classifications; the all-seed region''')
    change(main,r'''The retained cohort includes $343$ runs with neither threshold observed and $1$ with training observed but validation unresolved; $83$ runs establish zero delay by reaching validation before an unobserved training threshold.''',r'''The retained cohort includes $369$ runs with neither threshold observed and $1$ with training observed but validation unresolved; $89$ runs establish zero delay by reaching validation before an unobserved training threshold. In total, $13{,}407$ seed outcomes have positive delay, $3{,}583$ have zero delay, and $370$ remain unresolved. There are $1{,}102$ positive-minimum cells, $633$ zero-minimum cells, and $1$ unresolved-minimum cell. The $370$ unresolved seeds occur in $102$ cells: $101$ have a known zero minimum, while all ten seeds at $p=149,d=20$ remain unresolved.''')
    change(main,'''and their nearest measured widths on either side), the figure uses $235$
fresh H100 runs''','''and their nearest measured widths on either side), the initial replication collected $235$
fresh H100 runs''')
    change(main,'''accuracy thresholds have been observed. This targeted replication covers''',r'''online training and evaluation-mode validation accuracies reach $99\%$,
or at the planned epoch cap. This targeted replication covers''')
    change(main,'Training stops after both required\nonline training','Training stops after both\nonline training')
    change(main,'The same-five-seed restriction changes $64$ cell classifications in the extended plotted cohort.','The same-five-seed restriction changes $76$ cell classifications in the current plotted cohort.')
    anchor=r'''\paragraph{Extension to $p=199$.}'''
    completion=r'''\paragraph{Ten-seed completion.} Before collecting the additional runs, we
fixed all $269$ incomplete cells. We added seeds $47$--$51$ to $239$ H100
cells and replaced $30$ five-seed A100 cells with H100 seeds $42$--$51$,
retaining all $150$ replaced observations for paired comparison. These
$1{,}495$ runs used both H100s, with the same model, optimiser, precision and
per-cell epoch budgets; every planned run is included and every new
checkpoint was independently evaluated on CPU. The $1{,}467$ previously
complete cells are unchanged. The final figure contains $1{,}407$ A100 cells
and $329$ H100 cells, with no hardware pooling within a cell.
Completion changes the minimum delay in $53$ cells; $12$ change from positive
to zero minimum. The $p=149,d=96$ cell underlying the historical $5.51$ onset
now has six positive and four zero delays; this discrepancy is reported in
\cref{sec:theory-prereg}. The fitted laws and historical main-test cohorts
are retained separately. The original paired replication table retains its
matched seed sets; the plotted $p=139,d=224$ cell now has ten H100 seeds.

'''
    change(main,anchor,completion+anchor)
    change(main,'''parameters). All planned runs are included; the earlier columns and fitted
laws are unchanged.''','''parameters). All planned runs are included; this extension itself left the
earlier columns and fitted laws unchanged.''')
    change(main,'\\caption{Matched cone replication on one H100. Each row lists the affected','\\caption{Initial matched cone replication on one H100. Each row lists the affected')
    change(main,'''A100 measurements and fresh H100 measurements are compared separately.}''',r'''A100 measurements and fresh H100 measurements are compared separately.
The five-seed entry is the original paired comparison; the current cone
contains ten H100 seeds at that cell.}''')
    change(theory,r'''$\log_{10} P_{\text{onset}} = 5.62$ $[5.56, 5.77]$, measured $5.51$. We then''',r'''$\log_{10} P_{\text{onset}} = 5.62$ $[5.56, 5.77]$, measured $5.51$ in the
historical cohort. The ten-seed completion does not reproduce that onset
cell: at $d=96$, six seeds have positive delay and four have zero delay.
On the expanded cone grid, the first positive-minimum cell is $d=112$
($\log_{10}P=5.64$); the all-positive tail for $d\le256$ begins at $d=128$
($5.75$), since $d=124$ has a zero minimum. These are distinct grid summaries;
the historical $5.51$ datum is not an onset of the updated ten-seed figure.
We then''')
    change(theory,'''grid, has delays $(1,1,1,2,1)$ epochs across five seeds. Its positive minimum
contradicts the literal cutoff under the positive-delay definition. Epoch''','''grid, has delays $(1,1,1,2,1,1,0,1,1,0)$ epochs across ten seeds. Its minimum
is zero, but eight seeds have observed positive delay. These observations
do not establish an absence of delayed generalisation in every seed. Epoch''')
    change(theory,'''the region thins through a grazing band. No cell at $193$ has positive delay
at every retained seed, but the five-seed extension at $197$, $d=316$, has
delays of one to two epochs (minimum $1$, mean $1.2$). This width is outside''','''the region thins through a grazing band. No cell at $193$, $197$, or $199$
has positive delay at every retained seed. At $197$, $d=316$, eight of ten
seeds have delays of one to two epochs (minimum $0$, mean $0.9$). This width is outside''')
    change(principled,'''under that criterion. However, the five-seed extension at $d=316$ has
minimum delay $1$ epoch, so the exact cutoff is not confirmed. A held-out
check at $p = 149$ landed within $0.11$ dex.''',r'''under that criterion. However, the ten-seed extension at $d=316$ has eight
positive delays and a zero minimum, so absence of delayed generalisation in
every seed is not established. The historical held-out check at $p = 149$
landed within $0.11$ dex; its onset cell is not reproduced by the ten-seed
completion (\cref{sec:theory-prereg}).''')
    change(consequences,'''the additional $d=316$ cell has positive delay in all five seeds''','''the additional $d=316$ cell has positive delay in eight of ten seeds
and a zero minimum''')
    change(consequences,'''widths. The additional $p=197$, $d=316$ cell has delays $(1,1,1,2,1)$ epochs,
so it contradicts the literal empty-region prediction under the paper's
positive-delay definition. The registered-grid result remains valid; it''','''widths. The additional $p=197$, $d=316$ cell has delays
$(1,1,1,2,1,1,0,1,1,0)$ epochs, with zero minimum but eight observed positive
delays. The registered-grid result remains valid; it''')
    change(extra,r'''$p=197$, $\alpha=0.5$, $d=316$ cell has positive delay in all five seeds''',r'''$p=197$, $\alpha=0.5$, $d=316$ cell has positive delay in eight of ten seeds
and a zero minimum''')
    change(extra,'with one seed in five failing outright at three of four widths;','with one seed in five unresolved at the epoch cap at three of four widths;')
    for label in ['tab:hypothesis-tests','tab:crossing','tab:cone-replication']:assert tabular(old[main],label)==tabular(new[main],label),label
    for rel,text in new.items():
        assert 'positive delay in all five seeds' not in text
        assert 'five-seed extension' not in text
        assert 'unrecoverable' not in text.lower()
    diffs=[]
    for rel in TEX:
        diffs.extend(difflib.unified_diff(old[rel].splitlines(True),new[rel].splitlines(True),fromfile='before/'+rel,tofile='after/'+rel))
    (OUT/'paper_changes.diff').write_text(''.join(diffs))
    for rel,text in new.items():(PAPER/rel).write_text(text)
    for folder in [PAPER/'media/theory',ROOT/'figures/theory']:shutil.copy2(FIG/'cone_seedmin_fixed.pdf',folder/'cone_seedmin_fixed.pdf')
    write(OUT/'publication_source_hashes.json',{rel:sha(PAPER/rel) for rel in TEX})
def build_and_publish():
    a=json.loads((OUT/'analysis.json').read_text());audit=json.loads((OUT/'final_data_audit.json').read_text())
    assert a['total_runs']==audit['all_current_trace_hashes_and_first_passages_verified']==17360
    assert a['total_cells']==1736 and a['new_runs']==1495 and len(a['main_text_discrepancies'])==1
    # Exercise the usual plotting command on all final cells without touching the publication.
    reproduction=OUT/'full_plot_reproduction';reproduction.mkdir(exist_ok=True)
    for name in ['summary.json','cells.json']:shutil.copy2(FIG/name,reproduction/name)
    with (reproduction/'command.log').open('w') as f:subprocess.run([sys.executable,str(ROOT/'scripts/plot_cone_seedmin.py'),'--fixed-only','--reuse','--out',str(reproduction)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    assert sha(reproduction/'cone_seedmin_fixed.png')==sha(FIG/'cone_seedmin_fixed.png')
    build=OUT/'build';build.mkdir(exist_ok=True)
    cmd=['pdflatex','-interaction=nonstopmode','-halt-on-error',f'-output-directory={build}','paper-neurips-revised.tex']
    for i in range(1,4):
        with (build/f'pass{i}.log').open('w') as f:subprocess.run(cmd,cwd=PAPER,stdout=f,stderr=subprocess.STDOUT,check=True)
        if i==1:
            with (build/'bibtex.log').open('w') as f:subprocess.run(['bibtex','paper-neurips-revised'],cwd=build,env={**os.environ,'BIBINPUTS':f'{PAPER}:','BSTINPUTS':f'{PAPER}:'},stdout=f,stderr=subprocess.STDOUT,check=True)
    log=(build/'pass3.log').read_text();assert not re.search(r'(undefined references|undefined citations|Citation .* undefined|Reference .* undefined)',log,re.I)
    import pymupdf
    doc=pymupdf.open(build/'paper-neurips-revised.pdf');text=' '.join(' '.join(p.get_text().split()) for p in doc)
    for phrase in ['17,360','1,736','370','six seeds have positive delay and four have zero delay','5.64','5.75']:
        assert phrase in text,phrase
    for i,page in enumerate(doc):
        t=' '.join(page.get_text().split())
        if 'The measured grokking region' in t:page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(OUT/'paper_cone_page.png')
        if 'The ten-seed completion does not reproduce' in t:page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(OUT/'paper_discrepancy_page.png')
        if 'Ten-seed completion.' in t:page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(OUT/'paper_completion_page.png')
    shutil.copy2(build/'paper-neurips-revised.pdf',PAPER/'paper-neurips-revised.pdf')
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    result={'published_utc':now,'paper_pages':len(doc),'all_new_runs_CPU_verified':1495,'all_final_traces_independently_audited':17360,'ten_seed_cells':1736,'unresolved_seed_outcomes':370,'cells_with_unresolved_markers':102,'main_test_crossing_and_paired_replication_numeric_tables_unchanged':True,'historical_p149_discrepancy_explicit_in_main_text':True,'all_five_positive_p197_claim_removed':True,'canonical_plot_reproduction_pixels_match':True,'paper_sha256':sha(PAPER/'paper-neurips-revised.pdf'),'paper_source_sha256':sha(PAPER/'paper-neurips-revised.tex'),'figure_pdf_sha256':sha(FIG/'cone_seedmin_fixed.pdf'),'figure_png_sha256':sha(FIG/'cone_seedmin_fixed.png'),'undefined_references_or_citations':False,'overfull_boxes':re.findall(r'Overfull[^\n]+',log)}
    write(OUT/'publication_validation.json',result)
    v=json.loads((OUT/'validation.json').read_text());v.update({'paper_update_status':'Published after full data audit and explicit p149 discrepancy review.','figure_pdf_sha256':result['figure_pdf_sha256'],'figure_png_sha256':result['figure_png_sha256'],'runs_sha256':sha(FIG/'runs.json'),'cells_sha256':sha(FIG/'cells.json'),'publication_validation':'publication_validation.json','final_data_audit':'final_data_audit.json'});write(OUT/'validation.json',v)
    write(ROOT/'results/cone_latest.json',{'analysis_directory':str(FIG.relative_to(ROOT)),'validation':str((OUT/'validation.json').relative_to(ROOT)),'publication_validation':str((OUT/'publication_validation.json').relative_to(ROOT)),'figure_pdf':str((FIG/'cone_seedmin_fixed.pdf').relative_to(ROOT)),'figure_png':str((FIG/'cone_seedmin_fixed.png').relative_to(ROOT)),'paper':str((PAPER/'paper-neurips-revised.pdf').relative_to(ROOT)),'published_utc':now,'display_policy':'One primary cone figure, with ten distinct seeds at every cell; unresolved seeds explicitly marked even in zero-minimum cells. No five-seed figure.','seed_count_note':'All 1736 cells use seeds 42--51, for 17360 unique observations.','claim_flags':str((OUT/'claim_flags.txt').relative_to(ROOT))})
    src=OUT/'analysis_source';src.mkdir(exist_ok=True)
    for name in ['analyse_cone_ten_seeds.py','verify_cone_ten_seeds.py','finish_cone_ten_seeds.py','cone_data.py','verify_cone_h100_boundary.py','schedule_cone_ten_seeds_mps.py','benchmark_cone_mps.py','benchmark_cone_mps_wide.py','plot_cone_seedmin.py','publish_cone_ten_seeds.py']:shutil.copy2(ROOT/'scripts'/name,src/name)
    write(OUT/'analysis_source_manifest.json',{p.name:sha(p) for p in sorted(src.glob('*.py'))})
    (OUT/'report.txt').write_text('UNIFORM TEN-SEED CONE COMPLETE AND PUBLISHED\n\nAll 1495 planned runs completed on the two H100s (814 and 681 runs) and passed independent CPU checkpoint verification. Every one of the 17360 final traces was independently hash-checked and rescored. All 1736 cells use exactly seeds 42--51, with one hardware cohort per cell.\n\n370 seed outcomes remain unresolved; 102 cells carry explicit unresolved markers, including 101 cells with a known zero minimum. Missing validation98 is never classified as positive or zero.\n\nTwelve formerly positive-minimum cells now have zero minima; 53 minima changed in total. At p197,d316 the ten delays are [1,1,1,2,1,1,0,1,1,0], with mean0.9 and minimum0. No cell at p193,p197,p199 is all-ten-positive; individual positive delays remain.\n\nMAIN-TEXT DISCREPANCY: the historical p149 onset cell at d96 (log10P5.51) is now six positive and four zero delays. In the expanded cone, the first positive-minimum cell is d112 (5.64), while the sustained positive tail below d256 begins at d128 (5.75). These summaries are distinct. The historical main-test data and fitted laws were not silently refitted, and the paper explicitly flags the discrepancy.\n\nSee final_data_audit.json, publication_validation.json, figure/unresolved_runs.csv, changed_cells.csv, first_positive_comparison.csv, onset_comparison_max_width256.csv, and paired_A100_H100_runs.csv.\n\nReproduce analysis: python scripts/analyse_cone_ten_seeds.py\nRegenerate current figure: python scripts/plot_cone_seedmin.py --fixed-only --final\n\n'+json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--build-only',action='store_true');args=ap.parse_args()
    if not args.build_only:prepare()
    if not args.prepare_only:build_and_publish()
if __name__=='__main__':main()
