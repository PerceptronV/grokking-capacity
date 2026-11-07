#!/usr/bin/env python3
"""Publish verified 2026-11-07 analyses into the revised manuscript.

Source snapshots preserve previous paper versions. Update generated tables and
figures in the CURRENT manuscript; never regenerate author prose from a historical
snapshot. Numerical changes in running prose require a separate editorial pass.
Run after the recomputation scripts, or reuse the existing validated analyses.
"""
from pathlib import Path
from paper_tables import threshold_table
from paper_sources import figure_assets, tex_sources
import json, math, re, shutil
import numpy as np
import pandas as pd
from plot_cone_matched_aggregations import ROOT, save_json, sha

PAPER=ROOT/'docs/papers/2026 Grokking Capacity'
REPORT=ROOT/'reports/paper_corrections_20261107'
BEFORE=REPORT/'source_before'
RESULTS=ROOT/'results/paper_revision_20261107'
PUBLISHED_CROSSING_PROTOCOLS=('fixed',)  # Current paper omits per-layer LR experiments.
SOURCES=['paper-neurips-revised.tex','theory/theory-section.tex','theory/consequences.tex','theory/extra.tex','theory/principled.tex','checklist.tex']
LABELS={'central':'hypothesis-tests','init_scale_sweep':'init-scale','weight_decay_sweep':'wd','lr_sweep':'lr','alpha_sweep':'alpha','depth_scaling':'depth','task_add':'task-add','task_mul':'task-mul'}
NAMES={'central':'Central sweep','init_scale_sweep':'Initialisation scale','weight_decay_sweep':'Weight decay','lr_sweep':'Learning rate','alpha_sweep':'Training fraction','depth_scaling':'Depth','task_add':'Addition and division','task_mul':'Multiplication and division'}
AXES={'init_scale_sweep':'init_scale','weight_decay_sweep':'weight_decay','lr_sweep':'lr','alpha_sweep':'train_fraction','depth_scaling':'depth','task_add':'operation','task_mul':'operation'}


def snapshot():
    BEFORE.mkdir(parents=True,exist_ok=True)
    paths=tex_sources(PAPER)+[PAPER/'references.bib', PAPER/'paper-neurips-revised.pdf']
    for figure in figure_assets(PAPER):
        for ext in ['.pdf','.png']:paths.append(figure.with_suffix(ext))
    paths=list(dict.fromkeys(paths))
    for path in paths:
        dest=BEFORE/path.relative_to(PAPER)
        if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    save_json(REPORT/'publication_before.json',{str(p.relative_to(PAPER)):sha(BEFORE/p.relative_to(PAPER)) for p in paths})


def number(x,d=3):
    if x is None or not np.isfinite(float(x)):return '---'
    return f'${float(x):.{d}f}$'


def pvalue(x):
    if x is None or not np.isfinite(float(x)):return '---'
    x=float(x)
    if x==0:return '$<10^{-300}$'
    if x>=.001:return f'${x:.3f}$'
    e=int(math.floor(math.log10(x)));a=x/10**e
    return rf'${a:.2f}\times10^{{{e}}}$'


def table(label,caption,columns,header,rows):
    return '\n'.join([r'\begin{table}[t]',r'\centering\small',rf'\caption{{{caption}}}',rf'\label{{{label}}}',
        rf'\begin{{tabular}}{{{columns}}}',r'\toprule',' & '.join(header)+r' \\',r'\midrule',
        *[' & '.join(row)+r' \\' for row in rows],r'\bottomrule',r'\end{tabular}',r'\end{table}'])


def replace_table(text,label,new):
    pattern=r'\\begin\{table\}(?:\[[^]]*\])?.*?\\end\{table\}'
    matches=[m for m in re.finditer(pattern,text,re.S) if rf'\label{{{label}}}' in m.group()]
    assert len(matches)==1,label
    m=matches[0];return text[:m.start()]+new+text[m.end():]


def pooled(name,test):
    pred,cal=test['predictiveness'],test['calibration'];ft=test['sufficiency']['ftests']
    ccc=cal['ccc'];jt=cal['joint_ftest'];wx=cal['wilcoxon']
    rows=[['Spearman $\\rho$',number(pred['spearman']['rho']),pvalue(pred['spearman']['p_permutation'])],
          ['Kendall $\\tau$',number(pred['kendall']['tau']),pvalue(pred['kendall']['p_analytic'])],
          ["Lin's CCC (95\\% interval)",rf'${ccc["value"]:.3f}\;[{ccc["ci_low"]:.3f},{ccc["ci_high"]:.3f}]$','---'],
          ['Log--log slope',number(jt['slope']),'---'],['Intercept',number(jt['intercept']),'---'],
          ['Calibration $F$ test',number(jt['f'],2),pvalue(jt['p'])],
          ['Median log-residual; Wilcoxon',number(wx['median_log_ratio']),pvalue(wx['p'])]]
    for key,label in [('M1_vs_M0','$M_1$ vs. $M_0$'),('M3_vs_M2','$M_3$ vs. $M_2$'),('M3_vs_M1','$M_3$ vs. $M_1$'),('M3c_vs_M2c','Crossing adds over $K/C$')]:
        rr=ft.get(key,{})
        rows.append([label,number(rr.get('f'),2),pvalue(rr.get('p'))])
    label='tab:hypothesis-tests' if name=='central' else f'tab:{LABELS[name]}-pooled'
    baseline='$p$' if name=='central' else 'the swept setting and $p$'
    caption=(f'{NAMES[name]}: {test["n_valid"]} identified onset/crossing pairs out of {test["n_cells"]} eligible prime/setting cases. '
        'Timescale curves use arithmetic means of observed events and actual lower crossings; unresolved and out-of-domain outcomes are excluded from point statistics, not scored as successes. '
        f'$M_0$ is intercept-only, $M_1$ adds log crossing, $M_2$ uses {baseline}, and $M_3$ combines them. '
        'Degrees of freedom use matrix rank. Permutation and CCC bootstrap counts are 10,000; nested tests are in-sample comparisons, not proofs of sufficiency.')
    if name == 'central':
        caption = caption.replace(
            'Timescale curves use arithmetic means of observed events and actual lower crossings; unresolved and out-of-domain outcomes are excluded from point statistics, not scored as successes. ',
            r'Timescale construction and crossing selection are described in \cref{sec:stats-methods}. ')
    return table(label,caption,'lll',['Test','Statistic','$p$-value'],rows)


def within_table(name,frame):
    dd=frame[frame.config==name];rows=[]
    for r in dd.itertuples():
        setting=str(r.setting).replace('_',r'\_')
        rows.append([setting,f'{r.valid}/{r.total}',number(r.median),number(r.sd),number(r.rho)])
    caption=(f'{NAMES[name]}: within-setting residuals for identified pairs. Denominators include every eligible prime; '
        'missing roots and unidentified onsets remain in the denominator. SD is the sample standard deviation and is undefined for fewer than two pairs; rank correlation is omitted below three. '
        'The seed pool and explicit central reference are given in the scope paragraph.')
    return table(f'tab:{LABELS[name]}-within',caption,'lcccc',['Setting','Pairs / eligible','Median log-ratio','SD','Spearman $\\rho$'],rows)


def crossing_table(which,frame):
    rows=[]
    for r in frame.itertuples():
        ci='---' if pd.isna(r.ci_lo) else rf'$[{r.ci_lo:.2f},{r.ci_hi:.2f}]$'
        onset=number(r.onset_log10P,2) if not pd.isna(r.onset_log10P) else 'unidentified'
        row=[str(r.p),number(r.crossing_log10P,2),ci,number(r.bootstrap_root_fraction,3),onset,f'{r.agreement_matches}/{r.scored_cells}']
        if which=='fixed':row.insert(1,number(r.law_crossing_log10P,2))
        rows.append(row)
    head=['$p$','Measured','95\\% interval','Root fraction','Onset','Agreement']
    if which=='fixed':head.insert(1,'Fitted law')
    caption=('Direct crossings under '+('fixed-rate' if which=='fixed' else 'per-layer learning-rate scaling')+
       ', in $\\log_{10}P$, recomputed from arithmetic means of observed events (five seeds per timescale cell). '
       'Intervals are percentile intervals among bootstrap draws with a crossing; root fraction reports how often one exists in 1,000 draws. '
       'A dash denotes no fitted-law root; unidentified onset is not zero. Agreement is matches / scoreable cells, with unresolved timescales or delays excluded explicitly. '
       'The onset uses the dense-grid upper-tail domain rule. This retrospective reanalysis is separate from the ten-seed cone.')
    return table('tab:crossing' if which=='fixed' else 'tab:crossing-mup',caption,'l'+'c'*(len(head)-1),head,rows)


def appendix_text(s,overview,tests,within):
    for name,test in tests.items():s=replace_table(s,'tab:hypothesis-tests' if name=='central' else f'tab:{LABELS[name]}-pooled',pooled(name,test))
    for name in AXES:s=replace_table(s,f'tab:{LABELS[name]}-within',within_table(name,within))
    for which in PUBLISHED_CROSSING_PROTOCOLS:
        new=crossing_table(which,pd.read_csv(RESULTS/'crossings'/f'crossing_{which}.csv'))
        if which=='fixed':s=replace_table(s,'tab:crossing',new)
        else:s=s.replace(r'\paragraph{Capacity constant.}',new+'\n\n'+r'\paragraph{Capacity constant.}',1)
    return s


def publish_assets():
    mapping={
       RESULTS/'models/cone_min_cellmeans.pdf':PAPER/'media/theory/cone_seedmin_fixed.pdf',
       RESULTS/'models/mem_inverse_capacity.pdf':PAPER/'media/saturation_epochs_vs_inverse_capacity_all_primes.pdf',
       RESULTS/'models/mem_capacity_fraction.pdf':PAPER/'media/saturation_time_vs_capacity_fraction_all_primes.pdf',
       **{RESULTS/'crossings'/f'crossing_{which}.pdf':PAPER/'media/theory'/f'crossing_{which}.pdf' for which in PUBLISHED_CROSSING_PROTOCOLS},
    }
    for prime in [97,113,139]:
        mapping[RESULTS/'supplement/primary_figures'/f'p={prime}.pdf']=PAPER/'media'/f'delay_with_speed_p{prime}.pdf'
    for name in AXES:
        for source in (RESULTS/'statistics'/name).glob('*.pdf'):
            mapping[source]=PAPER/'media'/name/'intersection'/source.name
    for src,dest in mapping.items():
        dest.parent.mkdir(parents=True,exist_ok=True)
        for ext in ['.pdf','.png']:shutil.copy2(src.with_suffix(ext),dest.with_suffix(ext))
    return {str(dest.relative_to(PAPER)): {'source':str(src.relative_to(ROOT)),'sha256':sha(dest)} for src,dest in mapping.items()}


def main():
    snapshot()
    overview=json.loads((RESULTS/'statistics/overview.json').read_text())
    tests={name:json.loads((RESULTS/'statistics'/name/'hypothesis_tests.json').read_text()) for name in LABELS}
    within=pd.read_csv(RESULTS/'statistics/within_setting.csv')
    manuscript=PAPER/'paper-neurips-revised.tex'
    current=manuscript.read_text()
    s=appendix_text(current,overview,tests,within)
    sensitivity=json.loads((RESULTS/'supplement/sensitivities.json').read_text())
    s=replace_table(s,'tab:threshold-sensitivity',threshold_table(sensitivity))
    # Author prose, captions, section placement and included theory are editorial
    # sources. Rebuilding must not reset them to BEFORE or replay prose rewrites.
    if s != current:
        manuscript.write_text(s)
    assets=publish_assets()
    save_json(REPORT/'publication_manifest.json',{'published_crossing_protocols':list(PUBLISHED_CROSSING_PROTOCOLS),'sources':{n:sha(PAPER/n) for n in SOURCES},'assets':assets,'status':'sources and figures published; build verification pending'})
    print('Published numerical tables and',len(assets),'PDF/PNG figure pairs; author prose preserved.')


if __name__=='__main__':main()
