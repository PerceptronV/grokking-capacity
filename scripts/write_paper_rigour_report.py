#!/usr/bin/env python3
"""Assemble evidence-backed audit; does not change the paper."""
from pathlib import Path
import csv,json,hashlib,re
import pandas as pd
R=Path(__file__).resolve().parents[1];O=R/'reports/paper_rigour_new_conventions_20261106';P='docs/papers/2026 Grokking Capacity/'
F=[]
def add(id,severity,title,locations,evidence,finding,action):
 F.append(dict(id=id,severity=severity,title=title,locations=locations,evidence=evidence,finding=finding,action=action))
def read(n):return json.loads((O/n).read_text())
def save(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
add('R01','high','Current clock fit has not reached the manuscript or publication pipeline',
 ['paper-neurips-revised.tex:127','paper-neurips-revised.tex:156','theory/theory-section.tex:22','theory/consequences.tex:93','theory/principled.tex:167'],
 ['law_raw_verification.json','current_law_forecasts.csv'],
 'The current all-observed-seed fit has ln(B)=1.092417962, gamma=15.372290226 at C=0.784932771; ln(A)=20.269441126, a=-0.814000014, b=0.946743826; p*=183.469907582, f_fold=0.061587689, P_fold=2,609,067.83. Paper and canonical cone still use b=1.109, a=-0.452, p*=167.5 and the old bootstrap interval [148,192]. Both current fits use A100 only. The latest preview is a separate artifact; the canonical pointer and forecast CLI still refer to the old estimator. No confidence interval has been calculated for this new fit in the adopted workflow.',
 'Propagate the chosen estimator and its coefficients, predictions, local slopes, calibration and figures together. Preserve original registered predictions explicitly as historical tests; do not relabel them as predictions from the new fit or transplant their confidence interval.')
add('R02','high','The new law fails the published 0.11-dex crossing-agreement claim',
 ['paper-neurips-revised.tex:127','theory/theory-section.tex:102','paper-neurips-revised.tex:1286'],
 ['current_law_vs_direct_crossings.csv','crossings/primes.csv'],
 'Current log10 law crossings at p=167,173,179 are 6.03430,6.12379,6.23461, versus published measured 6.23,6.25,6.89. The p=179 miss is -0.65539 dex (factor 4.52 in P); p=167 misses by -0.19570 dex. The new fold produces roots at primes where the original model had none. Comparing only the old five primes no longer checks the scope of the new law.',
 'Recompute and display all held-out crossings under the new fit, including failures. Do not carry over “within 0.11 dex” or exact agreement at 167.')
add('R03','high','Published sufficiency test uses an unidentifiable constant predictor',
 ['paper-neurips-revised.tex:422','paper-neurips-revised.tex:617','paper-neurips-revised.tex:656','src/grokking_capacity/analysis/stats.py:386'],
 ['historical_cohorts_rechecked.json','statistics/central_min_unresolved/hypothesis_tests.json'],
 'Dropout is constant. [1,log P_cross,dropout] has rank 2, not 3. M3 and M1 are the same model, so F≈0,p=1 cannot establish dropout invariance or sufficiency. The implementation counts columns rather than matrix rank for residual degrees of freedom. The current auto baseline requests p but the point table supplies slice_value, so M2/M3 are skipped instead of testing the advertised prime baseline. A separate valid capacity-scale comparison exists (F=29.5473,p=0.0009706 for adding crossing over K/C), but it is not the published dropout test.',
 'Use a genuinely varying, named baseline; pass the slice variable into its design matrix and use matrix rank. Report incremental predictive evidence without calling a failed-to-reject test proof of sufficiency.')
add('R04','high','Nearest approach is sometimes returned as a measured crossing',
 ['src/grokking_capacity/analysis/aggregate.py:187','paper-neurips-revised.tex:409','paper-neurips-revised.tex:904'],
 ['statistics/crossing_counterexample.json','statistics/crossing_signs.csv'],
 'find_intersection returns a finite answer for constant curves T_mem=1 and T_gen=2. Nine of 128 current sweep candidates have no sign change. Three such points remain in the uncensored weight-decay statistical sample. The code also interpolates T in raw P despite its log-y docstring; the dense crossing programme uses a different, actual log-ratio root.',
 'Require a bracketed root or classify the crossing as outside/unresolved on the observed domain. Name the interpolation convention. Recalculate affected statistics; do not interpret nearest approaches as successful onset forecasts.')
add('R05','high','Unresolved-delay conventions change the weight-decay table',
 ['paper-neurips-revised.tex:759','paper-neurips-revised.tex:784','paper-neurips-revised.tex:815'],
 ['statistics/overview.csv','statistics/weight_decay_sweep_min_unresolved/predictiveness.csv','statistics/crossing_signs.csv'],
 'The old 42-cell rho=0.899940 reproduces under the legacy delay-cap convention. Keeping missing validation outcomes unresolved leaves 33 identified onset/crossing pairs, rho=0.813861. Removing the three finite nearest-approach values that lack a root leaves 30 pairs, rho=0.785962. This last number is a diagnostic conditional sample, not a validated replacement table. It concerns the onset sweep, not the repaired lambda-scaling experiment whose -0.512335 result reproduces.',
 'Retain and label censored/width-limited outcomes in the complete sweep table. Define crossing eligibility before regenerating pooled and within-setting statistics.')
add('R06','high','Six appendix samples are not reproduced by current configuration selection',
 ['paper-neurips-revised.tex:712','paper-neurips-revised.tex:860','paper-neurips-revised.tex:934','paper-neurips-revised.tex:1017','paper-neurips-revised.tex:1100','paper-neurips-revised.tex:1171'],
 ['within_setting_rechecks.csv','statistics/overview.csv','statistics/init_scale_sweep_archived_points.json','statistics/lr_sweep_archived_points.json','statistics/depth_scaling_archived_points.json','statistics/task_add_archived_points.json'],
 'Even before changing censoring: init N15/rho.839 becomes N10/.79394; LR15/.972 becomes10/.98481; alpha31/.977 becomes29/.97882; depth9/.600 becomes4/-.200; addition10/.801 becomes5/1; multiplication10/.648 becomes5/.900. Archived point tables reproduce the published init/LR/depth/addition sample counts and rank statistics. The archived point tables support those historical rank statistics; the current cohort/pipeline mismatch, including missing inherited baseline groups, needs resolution. The alpha plot also retains large +~2-dex alpha=.5 residuals whereas the table explicitly substitutes main-sweep residuals near -.17.',
 'Pin the original eligible UUIDs and inherited baseline groups for every exhibit, then regenerate tables and figures from the same point table with current censoring. Reconcile alpha pooled/within tables and plot.')
add('R07','high','An all-seed cone is not an exact population cutoff',
 ['paper-neurips-revised.tex:156','theory/consequences.tex:72','theory/theory-section.tex:118'],
 ['cone_raw_verification.json','cone_protocol_check.json','cone_cells_independent.json'],
 'The current data are sound, but one complete p=191 cell has minimum delay 1 epoch beyond the new fitted terminal prime 183.47. All 31 measured p=197 cells have minimum zero; this means at least one seed has zero delay, not that all seeds do. There are 370 unresolved seeds in 102 cells; only one aggregate minimum is unresolved because other zeros identify the remaining minima. Restricting to five seeds changes 76 classifications. A finite-grid all-ten criterion cannot prove q=1 or q=0 in a seed population.',
 'Keep the explicit all-ten definition, observed fractions and unresolved marks. Preserve the p=191 counterexample when discussing the model fold. Distinguish crossing geometry from a proven empirical phase boundary.')
add('R08','high','The direct crossing programme still substitutes epoch caps for clock events',
 ['scripts/crossing_analysis.py:49','paper-neurips-revised.tex:1238','theory/theory-section.tex:90'],
 ['historical_cohorts_rechecked.json','crossings/summary.json','crossings/cells.csv'],
 'Fixed-rate measurements have 21 censored random-label runs and 3 censored generalisation runs; per-layer measurements have 88 censored random-label runs. Their legacy geometric-mean curves substitute caps, and the figure shows these as ordinary points. Changing --censoring affects delays only, not that clock substitution. Mechanism agreement is .7727 over110 fixed-rate cells (.7647 over102 fully observed clock cells), and .8444 over90 per-layer cells (.8056 over72 fully observed). These do not meet the registered .9 threshold.',
 'Distinguish historical cap-based scoring from observed-event curves; show censored clocks explicitly and report the complete eligible denominator. Do not silently present cap-derived curves as uncensored observations.')
add('R09','high','Directly measured T_gen does use task validation',
 ['theory/theory-section.tex:93'],['law_raw_verification.json'],
 '“Both timescales come from runs that never see the task validation set” contradicts T_gen being the first validation-99 event. Random-label T_mem is independent; the directly measured T_gen and delay use the same task validation traces. Training gradients do not use validation, which is a different claim.',
 'Separate held-out-prime forecasting from descriptive same-trace crossing measurements. Restrict independence claims to what is actually independent.')
add('R10','high','The calibration-offset explanation reverses the minimum order statistic',
 ['theory/extra.tex:179'],['cone_raw_verification.json'],
 'Positive minimum delay requires every seed to have positive delay. Under monotone seed onsets it selects the latest individual onset, not the earliest. The claimed -1.54 sigma contribution therefore does not follow. Lowering validation threshold from99 to98 cannot increase a given clipped delay and generally delays onset in width, contrary to the proposed negative contribution. The current law calibration offset is -0.208775 dex, not the old -0.224; neither is derived by this argument.',
 'Remove the claimed parameter-free decomposition unless it is replaced by a correct paired-threshold/seed model. Describe measured calibration as empirical.')
add('R11','high','The bilinear equations do not imply a square-root decay rate',
 ['theory/extra.tex:124','theory/principled.tex:139'],[],
 'The stated linear system has eigenvalues -lambda±s. Substituting s=c lambda gives (c-1)lambda, not sqrt(s0 lambda). extra.tex calls this a derivation of the half-power; principled.tex correctly says the bare factor-pair rate is linear and introduces a separate normalization mechanism. These are contradictory explanations.',
 'Keep the measured exponent and clearly state which additional assumptions generate a square-root clock. Delete or repair the invalid algebraic implication.')
add('R12','high','The cited lambda^0.53 measurement is post-grok, not the escape clock',
 ['theory/principled.tex:151','theory/extra.tex:161'],['secondary_numbers.json'],
 'The three rates 0.0010535,0.0011724,0.0035469 reproduce exponent0.533467, but all come from the window named grok+[0,300], after validation has grokked. This cannot directly measure the preceding plateau-escape clock. There is only one seed per lambda and the middle point is low. The independent alpha escape-rate exponent4.27995 and width-rate0.21537 do reproduce their result artifacts.',
 'Label the phase and sample size; treat this as post-transition growth evidence, or measure the escape interval before making an escape-clock claim.')
add('R13','medium','Fold expansion has the wrong remainder order',
 ['theory/consequences.tex:114'],['law_raw_verification.json'],
 'The next term is +(2/3)Delta R for both roots. Correctly x±=b±sqrt(2 b Delta R)+(2/3)Delta R+O(Delta R^(3/2)). The paper uses sqrt(Delta R)*(1+O(Delta R)), which omits an O(Delta R) term. Independent root calculations give omitted residual/Delta R tending to2/3.',
 'Add the common linear term or replace the relative remainder by O(sqrt(Delta R)). The leading square-root fold itself remains valid.')
add('R14','medium','Total memorisation need not be nonnegative',
 ['paper-neurips-revised.tex:288'],[],
 'sum(log2 V+log2 p_theta(y|x)) is negative when average assigned label probability is worse than uniform and can tend to minus infinity. The upper bound n log2 V is correct. No clipping is in the measured formula.',
 'State a baseline-relative score with only its valid upper bound, or explicitly define a different nonnegative statistic and recompute it. Do not silently add clipping.')
add('R15','medium','Achieved finite-grid capacity is presented as a representability bound',
 ['paper-neurips-revised.tex:300','paper-neurips-revised.tex:307','paper-neurips-revised.tex:318','paper-neurips-revised.tex:436'],
 ['mechanisms_verification.json','capacity_uncertainty_and_full_range.json'],
 'The estimator is an affine fit to the average of per-seed maxima over eight dataset sizes at six small widths, not an observed n→infinity limit or an architecture-wide upper bound. C=.784932771 and the interval reproduce. The intercept625.30 bits is12.76% of the smallest measured maximum; through-origin C=.830357693. These data support an achieved, protocol-specific normalization, not impossibility below K/C or guaranteed representability above it.',
 'Preserve the useful empirical normalization and distinguish it from an absolute capacity theorem. State the finite-grid/max estimator and small-model extrapolation.')
add('R16','medium','Capacity-related arithmetic and units remain inconsistent',
 ['theory/consequences.tex:269','theory/principled.tex:105','paper-neurips-revised.tex:349'],
 ['law_raw_verification.json','capacity_uncertainty_and_full_range.json','secondary_numbers.json'],
 'For P=100000,f=.275,C=.784932771, PCf ln2=14962.03 nats, not2200. gamma_inf=3.983 is not numerically coincident with an independently quoted spectral rate near10 without a coordinate conversion. The paper nominal full-range f ceiling.688 exceeds the actual largest uncensored point.513007 (3,324 observed points; full-range gamma10.914308,R2.930370); distinguish nominal and observed domains.',
 'Correct the arithmetic; align units before comparisons; label fit-domain and observed-data-domain bounds separately.')
add('R17','high','The claimed disappearance of mean delay at large width is false',
 ['theory/consequences.tex:215','theory/extra.tex:217','theory/theory-section.tex:137'],
 ['wide_width_mean_check.json','cone_raw_verification.json'],
 'The original five-seed fixed-rate cone_fill data already contradict a mean of zero above~10^7 parameters: p149,d1000 mean36.4 epochs; p151,d744 mean4.6; p163,d1000 mean3.2, all fully observed and with minimum0. A zero minimum must not be described as zero mean. Conversely finite per-layer measurements to d4000 do not establish delay growing “without limit.”',
 'Use the statistic actually measured and the finite width limit. Retain upper-branch model failure and optimizer effects as distinct observations.')
add('R18','medium','Curvature under truncation is asserted as a general lower-bound theorem',
 ['theory/consequences.tex:305'],[],
 'Discarding long runs lowers conditional mean log time, but monotonic missingness alone does not order second derivatives or fitted quadratic coefficients. Thus observed q>0 is not generally a lower bound on population curvature; a monotone bias can have either curvature. Current fits should be described as conditional on observed events.',
 'Remove the unconditional curvature/conservative-slope theorem or provide assumptions and proof sufficient for that inequality.')
add('R19','medium','Projection and free-constant claims exceed the stated model',
 ['theory/principled.tex:18','theory/principled.tex:46'],[],
 'The product projection law is exact only as an expectation under appropriate independent isotropic subspaces, not for the actual shared-block classifier subspaces. The same paragraph later introduces its invariant-block correction. “Two free constants in the entire account” also leaves b,A,a and load/drive assumptions outside the accounting.',
 'State expectation and independence assumptions before the formula, distinguish the stylized classifier from the Transformer, and inventory all measured/fitted constants.')
add('R20','medium','Mechanism exclusion table overextends the cited kernel result',
 ['theory/extra.tex:88'],['primary_source_checks.json'],
 'Mohamadi et al. establish a constant-fraction sample requirement for small population error in an early permutation-equivariant kernel regime. The present alpha=.5 dataset itself contains a constant fraction of all pairs. That lower bound alone does not imply “no generalisation at these n” for this division experiment.',
 'Provide a theorem with applicable task, constants and assumptions, or qualify this row as a mechanistic hypothesis rather than a ruled-out regime.')
add('R21','medium','The claimed 2% extreme-value formula agreement is overstated',
 ['theory/consequences.tex:292','theory/principled.tex:130'],['secondary_numbers.json'],
 'Across the stored cells, the analytic approximation differs from real-init means by up to6.54% and Gaussian-null means by up to3.89%; real vs Gaussian differs by up to2.49%. These are different comparisons. The multiplicative-character group also has a nontrivial real character, so not every non-DC mode has the same chi-square(2d) power law.',
 'Name the comparison and report its actual maximum/range; qualify the asymptotic formula and exceptional real modes.')
add('R22','medium','Small coefficient, estimator, and calibration discrepancies persist',
 ['paper-neurips-revised.tex:617','paper-neurips-revised.tex:653','theory/extra.tex:227'],
 ['statistics/central_min_unresolved/hypothesis_tests.json','secondary_numbers.json','mechanisms_verification.json','within_setting_rechecks.csv'],
 'Central rho=.9757576, tau=.9111111, CCC=.7394987 and median residual-.1614694 reproduce. Exact current OLS intercept is-.0754005 (paper-.076); joint F198.247 (paper199); M1-vs-null F565.454 (paper569). Within-setting tables also contain small drift: init=.5 sample SD=.265166 (paper.260), lambda=.01 SD=.132838 (paper.131), and lambda=3 SD=.104197 (paper.105). At depth6 the one-point sample has no estimable sample SD; reporting0 is misleading. The predictor overestimates onset by about45%, while empirical onset is about31% below prediction; those percentages are not interchangeable. extra.tex still says lambda^(-.45) after the matched -0.512335 update. The batch power-.47 is an older OLS summary, whereas the Tobit artifact gives-.377819; it needs its estimator named.',
 'Generate displayed values from a single result object, state percent-change denominator, and remove stale exponent copies. Do not confuse the central measured-curve test with the global-law test (new-law rho=.951515, raw CCC=.595237).')
add('R23','medium','Current paper cone has not adopted the requested plot conventions fully',
 ['paper-neurips-revised.tex:149','paper-neurips-revised.tex:518'],
 ['graphics.json','figure_render_checks.json','figures_contact_1.png'],
 'The included paper cone still has an undotted extrapolated lower curve, the old167.5 terminal line, and mean-delay colors in minimum-positive cells. The latest separate cone uses minimum colors, a139 fit boundary, dotted extrapolation,183.47, and a white background. All25 included figures have PDF and PNG files, no overall figure titles, and explanations/legends below. Per-prime identifiers inside small multiples remain useful panel labels.',
 'Publish the selected current cone and align caption/color semantics. Preserve fit-domain indication and unresolved crosses, and retain both PDF and PNG.')
add('R24','medium','Checklist and release claims do not describe the active manuscript',
 ['checklist.tex:37','checklist.tex:51','checklist.tex:71','checklist.tex:99','checklist.tex:116','checklist.tex:218'],
 ['inventory.json','pdf_build_comparison.json','relevant_tests.log'],
 'The checklist says there are no formal theorems/proofs, yet consequences.tex contains propositions, a corollary and proofs. It states mean-clock aggregation and all speed/grok quantities have10seeds, despite explicit4-,5-,3-,30-seed secondary studies. An anonymized release, aggregate~1000 GPU-hours, ethics-review assertion and the LLM-methodology declaration cannot be verified merely from the current source. No clean reproducible submission bundle was identified by this audit; local configs do not yet regenerate every published sample.',
 'Bring checklist answers into agreement with the final methods and proofs. Verify the actual release bundle and resource accounting; distinguish ten-seed primary cohorts from historical secondary cohorts. Have authors confirm declarations that require their knowledge.')
add('R25','medium','Architecture grouping omits per-layer learning-rate scaling',
 ['src/grokking_capacity/analysis/config_view.py:38','tests/test_figures.py:148'],
 ['relevant_tests.log'],
 'The relevant test suite has36passes and1failure: identifying field mup_base is absent from ARCH_KEY_FIELDS. This can pool distinct optimization protocols when both appear under otherwise matching groups. The freshly extracted current central clock fits are independently A100-only, and the cone has explicit mup_base=0 and one hardware cohort per cell, so this is not evidence of contamination in those verified cohorts.',
 'Include mup_base in architecture/cohort identity and verify group queries before regenerating extension figures.')
add('R26','medium','Historical registration and outcome-selection claims need a verifiable record',
 ['paper-neurips-revised.tex:1223','theory/theory-section.tex:109','theory/principled.tex:185'],
 ['historical_cohorts_rechecked.json','crossings/primes.csv'],
 'Dated local registrations exist, but the p197 public-comment artifact is explicitly a draft and does not prove a public pre-run timestamp. R1 records changing the crossing implementation when1085/1100fixed runs already existed; this is disclosed but is not a fully frozen analysis before data. The reported7-prime +.01/.13 offset excludes p191; the8available pairs give-.04279/.17792. A “well determined” inclusion rule needs explicit justification. New retrospective fits cannot inherit original preregistration status.',
 'Link immutable pre-run records if available, disclose the estimator amendment, and distinguish original forecasts from current sensitivity analyses. Define exclusions before aggregating agreement.')
add('R27','low','Two tables overflow the text block and some summaries are too absolute',
 ['paper-neurips-revised.tex:815','theory/extra.tex:82','paper-neurips-revised.tex:1249'],
 ['build/pass3.log','paper_contact_2.png','paper_contact_3.png','historical_cohorts_rechecked.json'],
 'Fresh pdflatex/bibtex build succeeds with identical46-page text and no undefined refs/citations. It reports table overflow34.705pt at the WD within-setting table and47.087pt at the mechanism table. The210-run, seven-cell30-seed means reproduce; claiming every underlying distribution is unimodal is stronger than a finite histogram establishes. “No proportional bias,” “perfect,” and “always predicts onset” also exceed the calibration failures and finite samples.',
 'Adjust table layout during the next edit and replace population-level/exact claims with the tested finite-sample statements.')
add('R28','medium','Global claims of shared clocks obscure distinct estimands',
 ['paper-neurips-revised.tex:349','paper-neurips-revised.tex:355','paper-neurips-revised.tex:494','paper-neurips-revised.tex:641','paper-neurips-revised.tex:1238'],
 ['law_raw_verification.json','statistics/overview.csv','crossings/summary.json'],
 'There are three distinct objects: current global OLS fits to individual observed log times; measured central curves of arithmetic seed means; dense crossing curves of geometric means with cap substitution. Exponentiated log OLS is not the arithmetic expected time without assumptions/correction. min delay is a paired per-seed statistic, not min(Tgen)-min(Tmem). Equation-level crossing geometry cannot by itself prove an all-ten positive-delay boundary.',
 'Use an explicit estimator/cohort ledger. Historical mean-curve tests can remain valid as separately labelled tests, but they must not be presented as automatic validation of the new global-law estimator.')

save('findings.json',F)
with (O/'findings.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(F[0]));w.writeheader();w.writerows([{**r,'locations':'; '.join(r['locations']),'evidence':'; '.join(r['evidence'])} for r in F])

# Every table, including inline appendix tables, receives a specific disposition.
tabs=read('all_tabulars.json');table_notes=[
 ('supported','235 paired raw traces reproduce24/24cell and234/235seed agreement; individual minima saved.',['hardware_replication_checks.json','cone_h100_boundary_20261104_cells.csv'],['R07']),
 ('partly_supported','Rank/CCC/median reproduce; minor numeric drift and invalid constant-dropout Ftests.',['statistics/central_min_unresolved/hypothesis_tests.json'],['R03','R22']),
 ('historical_points_only','Archived15point sample reproduces rho; current config selects10.',['statistics/init_scale_sweep_archived_points.json'],['R06']),
 ('historical_points_only','Within-setting rows describe old15point cohort; baseline provenance must be pinned.',['statistics/init_scale_sweep_archived_points.json'],['R06']),
 ('requires_recalculation','Legacy42cells reproduce; unresolved conventions leave33, including3nonroots.',['statistics/overview.csv'],['R04','R05']),
 ('requires_recalculation','Within-setting denominators change with censoring and valid roots; table overflows.',['statistics/weight_decay_sweep_min_unresolved/predictiveness.csv'],['R05','R27']),
 ('historical_points_only','Archived15point sample reproduces rho; current config selects10.',['statistics/lr_sweep_archived_points.json'],['R06']),
 ('historical_points_only','Baseline seed/cohort and unresolved widths need reconciliation.',['statistics/lr_sweep_min_unresolved/predictiveness.csv'],['R06']),
 ('inconsistent','31point paper versus29current; alpha=.5 substitutions differ from plot.',['statistics/alpha_sweep_min_unresolved/predictiveness.csv','figures_contact_2.png','figures_contact_3.png'],['R06']),
 ('inconsistent','Mixed11prime baseline/5prime secondary rows are labelled, but plot and pooled cohort need reconciliation.',['statistics/alpha_sweep_min_unresolved/predictiveness.csv'],['R06']),
 ('historical_points_only','Archived9point rho.6 reproduces; current config4point rho-.2.',['statistics/depth_scaling_archived_points.json'],['R06']),
 ('historical_points_only','Unresolved depths and inherited central baseline need explicit eligibility.',['statistics/depth_scaling_min_unresolved/predictiveness.csv'],['R06']),
 ('historical_points_only','Archived10point rho.801 reproduces; current config5addition points.',['statistics/task_add_archived_points.json'],['R06']),
 ('historical_points_only','Division baseline must use the declared original cohort.',['statistics/task_add_min_unresolved/predictiveness.csv'],['R06']),
 ('unreproduced_current_config','Current config5multiplication points; paper10includingdivision baseline.',['statistics/task_mul_min_unresolved/predictiveness.csv'],['R06']),
 ('unreproduced_current_config','Multiplication-only current offsets reproduce approximately; division baseline missing from current selection.',['statistics/task_mul_min_unresolved/predictiveness.csv'],['R06']),
 ('requires_recalculation','Measured roots historical; law row stale, p179omitted, censoring and eligibility need explicit labels.',['current_law_vs_direct_crossings.csv','crossings/primes.csv'],['R01','R02','R08','R26']),
 ('supported_with_scope','C and bootstrap interval reproduce; achieved capacity rather than absolute bound.',['mechanisms_verification.json','capacity_uncertainty_and_full_range.json'],['R15']),
 ('historical_misinterpretation','Wide-width minimum table is followed by corrections, but later mean-zero assertion remains false.',['wide_width_mean_check.json'],['R17']),
 ('conditional_model_claims','Mechanism table applies to named stylized assumptions; cannot exclude every extensive dynamics model.',['secondary_numbers.json'],['R18','R19']),
 ('contradicted','Bilinear half-power implication invalid; kernel lower bound insufficient for exclusion.',['secondary_numbers.json','primary_source_checks.json'],['R11','R12','R20'])]
assert len(tabs)==len(table_notes)==21
for i,(t,n) in enumerate(zip(tabs,table_notes),1):t.update(table_id=i,status=n[0],assessment=n[1],evidence=n[2],findings=n[3]);t.pop('text')
save('table_audit.json',tabs)

# Every included graphic is inspected in its actual PDF and the rebuilt manuscript.
graphics=read('graphics.json');ga=[]
for i,g in enumerate(graphics,1):
 if i==1:status,notes,ids='stale_fit','Canonical cone retains old fold, color metric and solid extrapolation. Latest separate preview is current.',['R01','R23']
 elif i in [2,3,4]:status,notes,ids='historical_estimator','Measured arithmetic means/min-delay figures remain separate from all-seed global fit.',['R28']
 elif i in [5,6]:status,notes,ids='supported_with_scope','Full-range empirical slope reproduced; separate from crossing-window fit. Lines connect seeds at repeated widths; discontinuities are not proof of mixed hardware.',['R16','R28']
 elif i in [7,8]:status,notes,ids='supported_with_scope','Matched six-size capacity, PDF and PNG agree; finite-grid estimator.',['R15']
 elif i==9:status,notes,ids='illustration','Training/validation illustration; does not independently establish exact threshold or universal delay.',['R28']
 elif i in [24,25]:status,notes,ids='requires_recalculation','Legacy clock-cap substitution and historical fit; missing censor marks.',['R01','R08','R23']
 else:status,notes,ids='cohort_reconciliation_required','Published sweep plot must use the same declared cohort and unresolved handling as its table.',['R05' if i in [12,13] else 'R06']
 ga.append({k:g[k] for k in ['source','line','include','resolved','sha256','png_exists']}|{'figure_id':i,'status':status,'assessment':notes,'findings':ids,'visual_review':'PDF contact sheet and rebuilt paper inspected; preview pair compared by rendered pixel differences; no overall title; legend/explanation below.'})
save('figure_audit.json',ga)

# Coverage means audited and disposition recorded, not that a contradicted claim is correct.
sections=pd.read_csv(O/'sections.csv');coverage=[]
for row in sections.to_dict('records'):
 src=row['source'];line=row['line'];ids=[]
 for f in F:
  for loc in f['locations']:
   name,_,ln=loc.rpartition(':')
   if src==P+name:
    nextlines=sections[(sections.source==src)&(sections.line>line)].line
    end=int(nextlines.min()) if len(nextlines) else 100000
    if line<=int(ln)<end:ids.append(f['id'])
 coverage.append(row|{'reviewed':True,'finding_ids':sorted(set(ids)),'disposition':'See identified findings and cohort-specific evidence; no blanket validation implied.' if ids else 'Definitions/context/protocol reviewed; no separate finding in this section. Empirical support is limited to the evidence scopes in the report.'})
save('section_coverage.json',coverage)
num=pd.read_csv(O/'numeric_text_inventory.csv');num['audit_status']='Reviewed in source context; use section disposition and linked evidence (not a claim that every literal is independently regenerated).'
num.to_csv(O/'numeric_claim_ledger.csv',index=False)

primary=[{'citation':'morris2025memorize','url':'https://arxiv.org/abs/2505.24832','assessment':'Primary abstract describes measured model capacity estimates; it does not establish a universal representability bound for this protocol.'},
 {'citation':'mohamadi2024why','url':'https://proceedings.mlr.press/v235/mohamadi24a.html','assessment':'Primary abstract specifies an early permutation-equivariant kernel regime and a constant fraction of all modular-addition pairs. It cannot alone rule out generalization at alpha=.5 here.'},
 {'citation':'vanlaarhoven2017l2','url':'https://arxiv.org/abs/1706.05350','assessment':'Primary abstract supports normalization affecting effective learning rate; extension to the measured task escape phase requires additional evidence.'},
 {'citation':'kosson2024rotational','url':'https://proceedings.mlr.press/v235/kosson24a.html','assessment':'Primary publication metadata and normalization/rotation context checked. PDF equation fetch failed; no claim of newly verifying the precise prefactor from that PDF.'}]
save('primary_source_checks.json',primary)

lines=['PAPER RIGOUR AND CONSISTENCY AUDIT — 2026-11-06','',
 'Outcome: the central observations and the repaired experiments retain support, but the current manuscript is not consistent with the adopted estimator and contains several substantive claim/analysis errors. This was an audit: paper wording, numerical claims, publication figures and the canonical PDF were not edited.',
 '', 'Scope and evidence',
 'Reviewed the active six-file LaTeX manuscript, all63 section entries, all21 tables (including inline tables), all25 included graphics, and the46-page rebuilt PDF. Structural references and bibliography keys resolve. Raw checks covered every17,360 cone run; all4,910 selected clock-fit runs including excluded censored cases;96 capacity runs;400 matched decay runs;30 norm-freezing runs;435 hardware pairs; and the historical crossing/30-seed cohorts. Eight sweep configurations were recomputed. Mechanistic summary numbers were recalculated from stored intervention/result rows; these checkpoint ablations and every ancillary historical training experiment were not rerun. Figures were visually reviewed and PDF/PNG rendering differences recorded. This is not an independent reproduction of training or proof of public preregistration.',
 '', 'Conventions checked',
 'Both current laws: equal-weight OLS on each uncensored seed log time; fit primes97,101,103,107,109,113,127,131,137,139, width<=256 and the original capacity-fraction window. Current-C window is[.220146,2.751828]. Onset/cone: minimum of paired nonnegative train99/validation98 delays over seeds42–51; retain unresolved outcomes. A100-only law fits; H100 matched capacity; one hardware cohort per cone cell. Paper PDF, preview PNG, no titles, explanation below, dotted prime extrapolation, white background.',
 '', 'What independently reproduces',
 '- Cone:1,736 ten-seed cells;1,102positive minima,633zero minima,1unknown minimum.370unknown seeds occur in102cells. No raw-event/hash disagreements or mixed-hardware cells.1,407A100cells and329H100cells. Protocol fields all match; budgets are explicit.',
 '- Current clock fits:674 observed memorisation runs and3,058 observed generalisation runs;966 and212 censored runs excluded. Coefficients and p*=183.4699076 reproduce from fresh registry selection and raw99% events.',
 '- Central measured-mean-curve test:rho=.975758,CCC=.739499,slope=.984889,median offset=-.161469dex. This is distinct from the current globally fitted law (rho=.951515,raw CCC=.595237). Calibration to identity is rejected; high rank signal survives.',
 '- Capacity:C=.784932771; the six-size affine fit and20,000-draw bootstrap interval reproduce. The repaired matched-A100 decay slope is-.512335. Paired-H100 norm-freezing slopes are-.611148 frozen and-.658990 unfrozen, supporting preserved qualitative decay dependence rather than an unchanged clock.',
 '- Initial hardware replication:234/235seed states and24/24cell states agree. Independent later sample:192/200seed states and19/20cell states agree; p127,d74 changes minimum1to0. This supports robustness on sampled central cells, not hardware identity or a validated high-prime cutoff.',
 '', 'Priority findings']
for f in F:
 lines.extend(['',f"{f['id']} [{f['severity']}] {f['title']}",'Locations: '+'; '.join(f['locations']),f['finding'],'Required correction: '+f['action'],'Evidence: '+('; '.join(f['evidence']) if f['evidence'] else 'Direct algebra/definition check against the cited source lines.')])
lines.extend(['','Full table and figure coverage','Every table has a disposition in table_audit.json; every included graphic in figure_audit.json. section_coverage.json and numeric_claim_ledger.csv locate the reviewed text. Archived/current sample differences are not treated as permission to silently replace published tables with a different cohort.',
 '', 'Limits requiring author or external evidence','Public pre-run timestamps, actual anonymized submission contents, aggregate GPU-hours, and declarations in the checklist remain unverified. Stored mechanism intervention summaries establish reported arithmetic, not a fresh rerun of all interventions. Conditional bootstrap intervals do not include all model/seed/selection uncertainty. Historical5-seed studies remain distinguishable from the current10-seed cone; they have not been silently rewritten.',
 '', 'Reproducibility','From the repository root, audit-only commands:',
 '  python scripts/audit_paper_current_conventions.py',
 '  python scripts/audit_paper_raw_evidence.py',
 '  python scripts/audit_paper_statistics.py',
 '  python scripts/audit_paper_secondary_evidence.py',
 '  python scripts/write_paper_rigour_report.py',
 'Build evidence: build/result.json and build/pass3.log. Existing relevant tests:36passed,1failed (mup_base grouping); unrelated system pytest plugin disabled. Additional one-off audits and exact commands are preserved in this task record; their CSV/JSON results and source hashes accompany the report.',
 '', 'Suggested correction order','First align the manuscript with the selected law and keep historical forecasts labelled; then fix true-root/censoring/cohort selection and regenerate the statistical exhibits; then correct the offset and mechanism derivations, overclaims and checklist. Finally regenerate all publication assets and repeat the build/traceability checks. These are recommended follow-on changes, not changes made during this audit.'])
report_text='\n'.join(lines)+'\n'
for old,new in {
'all63':'all 63','all21':'all 21','all25':'all 25','the46':'the 46',
'every17,360':'every 17,360','all4,910':'all 4,910',';96':'; 96',';400':'; 400',';30':'; 30',';435':'; 435',
'primes97':'primes 97','seeds42':'seeds 42','is[.220146':'is [.220146',
'Cone:1,736':'Cone: 1,736',';1,102positive':'; 1,102 positive',',633zero':', 633 zero',',1unknown':', 1 unknown',
'.370unknown':'. 370 unknown','in102cells':'in 102 cells','.1,407A100cells and329H100cells':'. 1,407 A100 cells and 329 H100 cells',
'fits:674':'fits: 674','and3,058':'and 3,058',';966 and212':'; 966 and 212',
'curve test:rho':'curve test: rho','Capacity:C':'Capacity: C','and20,000':'and 20,000',
'replication:234/235seed':'replication: 234/235 seed','and24/24cell':'and 24/24 cell','sample:192/200seed':'sample: 192/200 seed','and19/20cell':'and 19/20 cell','minimum1to0':'minimum 1 to 0',
'over110':'over 110','over102':'over 102','over90':'over 90','over72':'over 72',
'with10seeds':'with 10 seeds','have10seeds':'have 10 seeds','all25':'all 25','all10':'all 10',
'tests:36passed,1failed':'tests: 36 passed, 1 failed','36passes and1failure':'36 passes and 1 failure',
'1085/1100fixed':'1085/1100 fixed','exponent0.533467':'exponent 0.533467','exponent4.27995':'exponent 4.27995',
'width-rate0.21537':'width-rate 0.21537','by up to6.54%':'by up to 6.54%','by up to3.89%':'by up to 3.89%','by up to2.49%':'by up to 2.49%',
}.items():report_text=report_text.replace(old,new)
(O/'report.txt').write_text(report_text)
print('Wrote',len(F),'findings;',len(tabs),'table dispositions;',len(ga),'figure dispositions;',len(coverage),'section entries')
if __name__=='__main__':pass
