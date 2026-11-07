#!/usr/bin/env python3
"""Write the scientific assessment without editing paper prose or outputs."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/seed_aggregation_sensitivity_20261106'

def main():
    cone=json.loads((OUT/'cone/summary.json').read_text())
    main=pd.read_csv(OUT/'statistics/central_main_ten/overview.csv')
    common=pd.read_csv(OUT/'statistics/common_observed_cohort.csv')
    laws=json.loads((OUT/'laws/comparison.json').read_text())
    forecast=pd.read_csv(OUT/'laws/recalibrated_forecasts.csv')
    hardware=pd.DataFrame(json.loads((OUT/'hardware/summary.json').read_text()))
    baseline=pd.read_csv(OUT/'statistics/paper_baseline_reproduction.csv')
    crossing=pd.DataFrame(json.loads((OUT/'crossings/summary.json').read_text())['comparisons'])
    parts=['''SEED AGGREGATION: SCIENTIFIC ASSESSMENT (2026-11-06)

Recommendation

Use the median as a descriptive summary of a typical seed's delay, accompanied
by the fraction of seeds with positive observed delay and unresolved-outcome
counts. Retain minimum as an explicit all-seeds robustness statistic and as
the statistic defining the registered test. Do not mechanically replace every
minimum by a mean, or assume a common reducer makes the theoretical crossing
an identity. Different reducers ask different scientific questions.

The empirical cone expands under median and mean, especially at high primes.
Rank predictiveness on the ten-prime main test remains high, but uncalibrated
agreement with the unchanged crossing predictor becomes worse. Recalibration
helps some held-out predictions; it must be redone and reported for the new
estimand. This audit is exploratory and does not overwrite a registered test.

What was held fixed

The cone comparison uses exactly the frozen 17,360 seed observations at 1,736
cells, ten distinct seeds per cell, one hardware cohort per cell, with the
original width grids and epoch budgets. The validation threshold remains 98%
and online training threshold 99%. No new training was needed. Unobserved
validation is never assigned the epoch cap, an observed zero, or successful
positive delay. All 102 cells containing an unresolved seed remain marked.

Only the per-seed DELAY reducer changes in the primary comparison. A separate
factorial analysis changes TIMESCALE aggregation, and separately changes the
eligibility rule from any observed crossing to all ten crossings. These must
not be conflated.

Meaning of the three statistics

Let D_s = max(0, T_val98,s - T_train99,s). On a complete ten-seed cell:
  minimum > 0  iff 10/10 seeds have positive delay;
  mean > 0     iff at least 1/10 seeds has positive delay;
  median > 0   iff at least 5/10 seeds has positive delay under the usual
                  even-sample median, which averages positions five and six.
The last condition is NOT a strict majority. For a strict majority use the
positive-seed fraction > 0.5, rather than silently redefining the median.
Mean magnitude can be dominated by a slow outlier; mean positivity is an
any-seed event and is not evidence of typical delayed generalisation.
Minimum is sensitive to one zero-delay seed and depends strongly on seed count.
For independent seeds with positive-delay probability q, P(minimum > 0)=q^n
and P(mean > 0)=1-(1-q)^n. Neither is a seed-count-independent phase boundary.
The median is more stable as a central summary, although n=10 still leaves
substantial sampling variability. All-ten success does not establish q=1.

Cone occupancy (positive / zero / unresolved aggregate)
''']
    for m in ['min','median','mean']:
        x=cone['counts'][m];parts.append(f"{m:7s}: all cells {x['all_cells']}; same 1,634 complete cells {x['common_complete_cells']}\n")
    parts.append('''
An aggregate may be exactly known while some seeds are unresolved: any zero
fixes the minimum at zero; enough observed zeros fix a median at zero. Mean
requires every delay. This explains 1 unresolved minimum, 34 unresolved
medians and 102 unresolved means without changing the underlying data.

At p=193 / 197 / 199, minimum has 0 / 0 / 0 positive cells; median has 5 / 5 / 3;
mean has 11 / 14 / 7. All identified medians there are at most one epoch, and
all fully observed means there are at most 1.6 epochs. Unresolved means are
not bounded by that observed maximum. The p=197,d=316 delays have minimum 0,
median 1 and mean 0.9: the previous open point becomes positive under both
alternatives. No fully observed mean or median at these primes exceeds five
epochs. A five-epoch effect-size threshold would therefore produce a different
picture, but choosing it now would be a new exploratory definition, not a
retroactive confirmation of an exact zero-delay cutoff.

The registered p=197 grid and the expanded grid are distinct

The registered 18-width grid has:
  minimum: zero in all 18 cells; no onset;
  median: positive in three cells (d=268,300,364), but no sustained-positive-tail
          onset because the widest cells have median zero;
  mean: nine positive cells, eight zero cells and one unresolved aggregate.
        The registered last-zero rule gives d=188, P=1,206,772, log10 P=6.0816.
This mean onset falls within the old rival's reported [5.83,6.10] band. That
old band was itself defined using minimum; it is not a valid forecast band for
a newly chosen mean estimator. Refitting the rival to the central mean onsets
gives log10 P_197=5.6434 (prime-bootstrap fit interval [5.4896,5.8344], excluding
seed-noise uncertainty). Therefore the old rejection cannot simply be carried
over, and the numerical overlap with the old band does not confirm the rival.

On the full expanded p=197 grid, even the mean has no positive tail because
later widths have zero mean. First-positive, positive-tail onset on a fixed
bounded domain, and onset after trimming a terminal zero region are different
statistics. Their distinction is essential in a re-entrant cone. The audit
reports all-width and d<=256 results separately and preserves the registered
terminal-zero trimming rule in the measured-crossing comparison.

Ten-prime main test: p=97--139, same measured-timescale predictor

These ten primes reproduce the paper's current minimum result (rho=0.976,
CCC=0.74, slope=0.985, median offset=-0.16 dex). The eleven-prime pipeline
outputs include the held-out p=149 and are reported separately. The comparison
below does not silently add that held-out point to the main test.
''')
    parts.append(main.to_string(index=False,float_format=lambda v:f'{v:.6g}')+'\n')
    parts.append('''
Here median_all also changes both measured timescale curves to medians. It
does not improve the ten-prime calibration over changing delay alone.
All three delay summaries retain strong rank signal. The larger negative
offsets mean median/mean onsets occur at smaller parameter counts. CCC and
raw MAE quantify calibration to the unchanged crossing, so those worsen.
Even after removing a constant offset, the central residual standard deviation
increases from 0.024 dex (minimum) to 0.043 (median) and 0.053 (mean).
These observations do not justify selecting minimum just because its fit looks
best: the appropriate statistic is determined by the scientific claim.

Recalibration and held-out p=149

Holding the original fitted laws fixed and estimating a NEW calibration offset
from the ten central primes gives delta=-0.2242 / -0.3374 / -0.4697 dex for
minimum / median / mean. On the uniform ten-seed p=149 grid at d<=256, onset is
log10 P=5.7510 / 5.5111 / 5.4391. The corresponding calibrated predictions are
5.6244 / 5.5113 / 5.3789, with residuals +0.1265 / -0.0002 / +0.0601 dex.
Thus the median improves this particular held-out numerical comparison after
recalibration, even though it worsens uncalibrated central agreement. Since
we are choosing the metric after seeing p=149, this is sensitivity evidence,
not a new preregistered success. Full combinations of delay metric and law fit
are in laws/recalibrated_forecasts.csv.

Mathematical theory: what changes and what does not

The algebraic crossing T_mem(P)=T_gen(P), and the fold condition for an
exponential versus power-law crossing, do not contain the delay aggregation
operator. With coefficients held fixed, changing min to median/mean cannot
move p*=167.5125, or p*=203.8445 for the earlier complete-ten fit.

The link from a timescale crossing to an empirical grokking boundary changes.
In general:
  E[max(0, T_val-T_train)] != max(0, E[T_val]-E[T_train]);
  median(T_val-T_train) != median(T_val)-median(T_train).
A rare positive tail can make mean delay positive even where average signed
delay is negative. The random-label T_mem proxy is also measured in independent
runs, and is not identical by construction to the task's T_train. A theory of
an all-seeds, typical-seed or any-seed boundary requires the corresponding
seed-distribution assumption or calibration, not merely equal reducer names.

The current offset explanation in theory/extra.tex:179--194 uses an earliest-
seed order statistic to explain positive minimum delay. That does not follow
from the defined statistic: minimum delay > 0 requires every seed positive.
Under monotone individual onsets it corresponds to the latest onset, not the
earliest. The threshold-asymmetry direction in that paragraph also needs
review: raising the validation threshold delays its crossing and cannot reduce
a given seed's clipped delay. These passages require a mathematical correction
regardless of which summary becomes primary; the paper has not been edited.

Changing TIMESCALE estimators as well

The original memorisation fit uses per-run log OLS; the original generalisation
fit uses log arithmetic seed means, with incomplete observed cells allowed.
The following fits instead give each eligible cell equal weight and reduce
both timescales with the same named metric. The window and source hardware
remain fixed. This is a different estimator, not just a new cone colouring.
''')
    for name in ['observed_mean','observed_median','complete_mean','complete_median']:
        x=laws['unified_cell_reducers'][name];ci=x['fold_prime_bootstrap_95']
        parts.append(f"{name:18s}: fold={x['fold']:.3f}, paired prime-bootstrap 95% [{ci[0]:.3f}, {ci[1]:.3f}], mem cells={x['mem']['n_observations']}, gen cells={x['gen']['n_observations']}\n")
    parts.append('''
These intervals use 2,000 prime-cluster resamples with eligibility fixed.
Complete-cell selection still conditions on all ten crossings before the cap;
it does not correct censoring. Neither a later fold nor a higher fit R-squared
by itself establishes a better physical model. Capacity C is held fixed in a
consistent convention; its rescaling is not the cause of these changes.

Measured fixed-rate and muP crossing experiments

The registered predictor uses geometric mean timescales; censored times are
cap-substituted in that historical predictor. The complete_times_score below
excludes every cell with any unresolved timescale. All delay outcomes use
actual validation crossings. Alternative onset domains and scores retain
explicit denominators instead of treating unknown outcomes as false.
''')
    chosen=crossing[crossing.time_metric=='geometric_mean']
    parts.append(chosen[['setting','delay_metric','score','n_score','complete_times_score','n_complete_times_score','mean_onset_crossing_offset_dex']].to_string(index=False,float_format=lambda v:f'{v:.5g}')+'\n')
    parts.append('''
The fixed-rate score worsens from 0.773 to 0.700 to 0.533 for minimum, median
and mean. On the same 102 cells with both timescales fully observed, it is
0.765 / 0.686 / 0.520. The muP score is unchanged for median and slightly worse
for mean. Therefore improved robustness of median is not a uniform improvement
in agreement with the mechanism test. Capped historical time summaries are
not asserted to be eventual means or medians.

Other sweeps, on common observed-onset cohorts

The table excludes left-censored onsets at the narrowest width and uses the
same usable cells for all three metrics. In this audit depth has fewer than
three such common cells and is omitted from this matched comparison.
''')
    parts.append(common.to_string(index=False,float_format=lambda v:f'{v:.5g}')+'\n')
    parts.append('''
Uncalibrated MAE increases under median and mean in every recoverable matched
sweep. Rank changes have no uniform direction. Paired prime-cluster bootstrap
intervals for the MAE changes are in statistics/paired_MAE_changes.json.
The full original hypothesis suite, including nested-model sufficiency and
axis robustness tests, was run per configuration/figure/variant and is stored
in each hypothesis_tests.json. Main-table hypotheses compare prime as the
configured baseline; the paper's constant-dropout baseline should not be
mistaken for a nontrivial test against prime. Exploratory by-prime-axis results
can be degenerate (constant empirical onset); undefined rank statistics remain
undefined. Legacy tables retaining boundary points are not the common-observed
cohort table above.

Baseline reproduction limitation

The ten-prime main test and weight-decay rank/sample-count baseline reproduce.
Several frozen appendix tables use cohorts not reproduced by the current YAML
selectors. This discrepancy exists before changing aggregation and must not
be attributed to mean/median. In particular, current selectors yield empty or
unresolved baseline groups in several sweeps, and alpha's frozen table uses a
different inclusion convention. No unverified replacement paper numbers were
published. The code audit keeps these provenance issues separate.
''')
    parts.append(baseline.drop(columns='note').to_string(index=False)+'\n')
    parts.append('''
Stability with seed count and hardware

Enumerating all 252 five-of-ten subsets in the same 1,634 complete cells gives
mean five-vs-ten classification disagreement of 4.47% (minimum), 2.19% (median),
and 1.70% (mean). This is a finite-sample sensitivity calculation, not a
population error rate. The mean's apparently stable positivity reflects its
any-positive-seed criterion; it is not a typical-seed guarantee.

On the 20 preselected paired A100/H100 replication cells:
''')
    parts.append(hardware.to_string(index=False)+'\n')
    parts.append('''
Median preserves 20/20 cell classifications; minimum and mean preserve 19/20.
This small, deliberately selected sample does not establish universal hardware
equivalence or identify GPU architecture separately from runtime differences.

Recommended scientific presentation

1. Show typical delay (median) and positive-seed fraction, with explicit
   uncertainty/unresolved marks. Retain mean as expected-delay sensitivity.
2. Keep minimum as the all-ten-seeds robustness endpoint and retain its exact
   historical registration. Do not relabel that endpoint as a typical model.
3. Report the high-prime result as very small observed delays, with finite-grid
   and censoring limits, rather than infer a universal zero-delay cutoff.
4. If changing the primary estimator, redefine onset/domain, recalibrate the
   crossing and rival, recompute tests/intervals, and test on genuinely new data.
   Do not pick an effect-size threshold or reducer solely for the best fit.
5. Keep expected timescale laws and empirical summary choices separate. A
   stochastic race model predicting a positive-delay probability or quantile
   would connect the theory to these observables more directly.

Implementation and remaining scope

The shared API accepts min/median/mean/max/quantiles and Python callables,
with safe censoring and explicit legacy behavior for historical reproduction.
Delay and timescale reducers are independently configurable. Main plots,
onsets, statistics, cone rendering, forecast calibration/resampling and the
registered crossing CLI use the shared layer. Existing minimum-named fields
retain their meaning; they are not silently repurposed for another aggregate.

The default cone PNG was regenerated in an isolated directory and is byte-
identical to the published PNG. The original 1,736 minimum cell summaries are
also reproduced exactly. Scientific tests cover censored means/medians,
even-sample ties, missing traces, custom/quantile reducers, threshold effects,
seed-count dependence and forecast metric propagation. One unrelated existing
architecture-key test failure (mup_base missing from ARCH_KEY_FIELDS) was
confirmed in the saved pre-refactor source and is recorded separately.

Some specialized historical figure/report scripts remain explicitly tied to
the registered minimum or named archived summaries; see code_audit.csv. A
global string replacement would corrupt those historical meanings and would
also alter mathematical minima unrelated to seed aggregation.

Reproduction

  python scripts/analyse_seed_aggregation.py cone
  python scripts/analyse_seed_aggregation.py statistics
  python scripts/analyse_seed_aggregation.py laws
  python scripts/analyse_seed_aggregation_crossings.py
  python scripts/summarise_seed_aggregation.py
  python scripts/report_seed_aggregation.py

statistics reuses completed per-suite results; use a new output directory or
remove only this audit's completed artifacts for a fresh recomputation. No
paper source, published figure, fitted-law artifact or latest pointer is
modified by these commands. Current source snapshots, inputs, validation and
all numerical tables accompany this report.
''')
    (OUT/'report.txt').write_text(''.join(parts))
    print('Wrote',OUT/'report.txt')
if __name__=='__main__':main()
