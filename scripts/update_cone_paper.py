#!/usr/bin/env python3
"""Publish audited cone counts, censoring conventions and scoped interpretation."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/cone_repair_20261104';PAPER=ROOT/'docs/papers/2026 Grokking Capacity'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--analysis',type=Path,default=OUT);ap.add_argument('--final',action='store_true');args=ap.parse_args()
    s=json.loads((args.analysis/'summary.json').read_text());assert s['cells']==1717 and s['unique_runs']==15825
    prior=json.loads((OUT/'publication_update.json').read_text()) if (OUT/'publication_update.json').exists() else {}
    before=OUT/'before'/'docs/papers/2026 Grokking Capacity'
    updates=[]
    for rel in ['paper-neurips-revised.tex','theory/theory-section.tex']:
        old=(before/rel).read_text();current=(PAPER/rel).read_text()
        if rel=='paper-neurips-revised.tex':
            start=old.index('    \\caption{\\textbf{The grokking region, every cell measured on the central')
            end=old.index('    \\label{fig:cone}',start)
            caption=r'''    \caption{\textbf{The measured grokking region at fixed learning rate.}
    Twenty-one primes, $1{,}717$ cells, $15{,}825$ distinct seed observations
    with matched model and optimiser settings. Each cell retains one hardware
    cohort and five or ten distinct seeds. Filled markers: positive observed
    delay at every retained seed; open markers: at least one zero-delay seed;
    crosses: unresolved. Colour: mean delay in filled cells. Unreached
    thresholds are retained as censored (\cref{sec:exp-cone}). The red curve
    is the onset crossing from the fitted timescale laws; the dashed line is
    their divergence prime $p^{*}=167.5$. The region thins beyond the fold.
    No measured cell at $p=193$ has positive delay at every seed; an extension
    cell at $p=197$ has delays of one to two epochs. The separate pre-registered
    $p=197$ grid has zero minimum delay at all eighteen widths. The upper
    closure near $10^{7}$ parameters is a fixed-learning-rate wall
    (\cref{sec:theory-region}).}
'''
            new=old[:start]+caption+old[end:]
            new=new.replace('\\includegraphics[width=0.6\\linewidth]{theory/cone_seedmin_fixed.pdf}','\\includegraphics[width=0.8\\linewidth]{theory/cone_seedmin_fixed.pdf}')
            anchor=r'For each prime $p$ in the 10-prime sweep and each model dimension $d '
            para_start=new.index(anchor);para_end=new.index('\n\n',para_start)
            methods=r'''

\paragraph{Cone cohorts and incomplete outcomes.}\label{sec:exp-cone}
For \cref{fig:cone}, we retain the A100 central cohort and H100 extension
campaigns under the model and optimiser settings above, including
initialisation scale $1$, with a random split and no per-layer learning-rate
scaling. Each $(p,d)$ cell uses one hardware cohort. Repeated observations
of a seed within that cohort count once: we select the largest planned epoch budget, then the
earliest run; all $45$ repeated observations have identical first-crossing
times. There are $1{,}448$ ten-seed cells and $269$ five-seed cells, with
planned budgets of $1{,}500$, $2{,}500$, or $5{,}000$ epochs.
We score delay at the first evaluation-mode validation-$98\%$ crossing
relative to the first online training-$99\%$ crossing. Validation crossing
before an unobserved training threshold establishes zero delay. If validation
has not crossed, we retain a right-censored outcome: it is never counted as
successful delayed generalisation. A cell is filled only when every retained
seed has an observed positive delay; any observed zero fixes its minimum at
zero; otherwise the cell is unresolved. Mean-delay colours use only complete
filled cells. Restricting the archive to the same five seeds ($42$--$46$)
in every cell changes $63$ of $1{,}717$ classifications; the all-seed region
therefore depends on the number of seeds and is not a seed-independent phase
boundary.
'''
            events=s['event_counts']
            methods+=f"\nThe retained cohort includes ${events.get('both_thresholds_unobserved',0)}$ runs with neither threshold observed and ${events.get('validation_right_censored',0)}$ with training observed but validation unresolved; ${events.get('validation_before_unobserved_training',0)}$ runs establish zero delay by reaching validation before an unobserved training threshold.\n"
            if args.final:
                replication=json.loads((ROOT/'results/cone_h100_boundary_20261104/analysis.json').read_text())
                methods+=r'''
For $24$ cells selected before the replication campaign (eight affected cells
and their nearest measured widths on either side), the figure uses $225$
fresh H100 runs with the same seed sets and a $5{,}000$-epoch cap. The archived
A100 observations are retained for the paired replication comparison,
separately from the plotted cell cohorts. Training stops after both required
accuracy thresholds have been observed. This targeted replication does not
refit the original central onset/crossing statistics.
'''
            if args.final:
                methods=methods.replace('$225$\nfresh H100',f"${replication['runs']}$\nfresh H100")
                pairs={(x['p'],x['dim']):x for x in replication['all_cell_comparisons']}
                manifest=json.loads((ROOT/'results/cone_h100_boundary_20261104/manifest.json').read_text())
                methods+=f"\nThe H100 replication agrees with the archived cell classification in ${replication['paired_cell_agreement']}$ of $24$ cells and with the individual-seed classification in ${replication['paired_run_agreement']}$ of ${replication['runs']}$ pairs. The same-five-seed restriction changes ${len(s['final_five_seed_classification_changes'])}$ cell classifications in the final plotted cohort. All preselected comparisons are reported in \\cref{{tab:cone-replication}}.\n"
                table=r'''
\begin{table}[t]
\centering
\small
\caption{Matched cone replication on one H100. Each row lists the affected
width in the middle and its two preselected neighbouring widths. Entries
are minimum observed delays (epochs) across the same seed IDs on each
machine; the three entries follow the listed width order. The archival
A100 measurements and fresh H100 measurements are compared separately.}
\label{tab:cone-replication}
\begin{tabular}{rllll}
\toprule
Prime & Widths & Seeds per cell & A100 minima & H100 minima \\
\midrule
'''
                for prime,target,widths,nseed in manifest['groups']:
                    fmt=lambda x:'--' if x is None else str(int(x))
                    aa=', '.join(fmt(pairs[prime,d]['A100_minimum_delay']) for d in widths)
                    hh=', '.join(fmt(pairs[prime,d]['H100_minimum_delay']) for d in widths)
                    seed_counts=[pairs[prime,d]['n_seeds'] for d in widths]
                    seed_label=str(seed_counts[0]) if len(set(seed_counts))==1 else ', '.join(map(str,seed_counts))
                    table+=f"{prime} & {', '.join(map(str,widths))} & {seed_label} & {aa} & {hh} \\\\\n"
                methods+=table+r'''\bottomrule
\end{tabular}
\end{table}
'''
            new=new[:para_end]+methods+new[para_end:]
        else:
            new=old.replace('rather than to the exact zero of every run; no cell was budget-censored.',
                'rather than to the exact zero of every run. All eighteen cell minima are\ndetermined, although three of the $180$ runs reach neither threshold within\nthe budget.')
            start=new.index('\\Cref{fig:cone} assembles every grokking cell measured on this protocol.')
            end=new.index('The region also has a top under this protocol:',start)
            new=new[:start]+r'''\Cref{fig:cone} assembles the matched cohorts with the seed and censoring
conventions of \cref{sec:exp-cone}. Its lower edge is the onset, and the
law-derived crossing runs along it from $97$ to $167$. Beyond the fold,
the region thins through a grazing band. No cell at $193$ has positive delay
at every retained seed, but the five-seed extension at $197$, $d=316$, has
delays of one to two epochs (minimum $1$, mean $1.2$). This width is outside
the pre-registered eighteen-width grid. The fold therefore marks the onset
of the thinning region; the data do not establish exact disappearance of
every positive delay at a single prime. '''+new[end:]
        permitted={hashlib.sha256(old.encode()).hexdigest()}
        permitted.update(r['after_sha256'] for r in prior.get('source_edits',[]) if r['file']==rel)
        assert hashlib.sha256(current.encode()).hexdigest() in permitted,f'Unexpected concurrent edits in {rel}'
        (PAPER/rel).write_text(new);updates.append({'file':rel,'before_sha256':hashlib.sha256(old.encode()).hexdigest(),'after_sha256':hashlib.sha256(new.encode()).hexdigest()})
    for dest in [ROOT/'figures/theory/cone_seedmin_fixed.pdf',PAPER/'media/theory/cone_seedmin_fixed.pdf']:shutil.copy2(args.analysis/'cone_seedmin_fixed.pdf',dest)
    (OUT/'publication_update.json').write_text(json.dumps({'stage':'final' if args.final else 'interim','analysis_directory':str(args.analysis),'source_edits':updates},indent=2)+'\n')
    print('Updated cone figure, caption, methods and the directly affected interpretation.')
if __name__=='__main__':main()
