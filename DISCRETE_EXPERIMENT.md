# Discrete latent grokking experiment

Branch: `idea/discrete`. Initial registration: 2026-10-24. The p97 dropout rematch was registered 2026-10-25 before its training.

Active follow-up: a logarithmic dataset-size sweep will resize both C and B
from the calibrated capacity constants to match each dataset's complexity.
The previous fixed-pair proposal was not launched. See the
[dataset-matched sweep registration](#dataset-matched-logarithmic-sweep-registration).

Latest completed result: the capacity-constant matched scaling sweep is complete: ten
C/B size pairs and ten fresh seeds per pair, 200 arithmetic runs. Capacity is
estimated from parameter count using separately calibrated constants,
kappa_C = 0.729117 and kappa_B = 0.310917 bits/parameter.
C is faster in 86/100 paired comparisons, B in 13, and 1 pair is censored
for both. First held-out 99% crossings are C 99/100 and B 84/100;
final learning checks are C 99/100 and B 81/100. On the six common
sizes with every crossing observed, mean-time log-log slopes are C -0.509 and
B -1.043; the paired 95% interval for B minus C is [-1.300, 0.256], so a
slope difference is not resolved. The all-size capped-time curve is steeper
for B, which has more low-capacity failures. All 235 checkpoint audits passed.
See the [completed scaling sweep](#completed-capacity-constant-matched-scaling-sweep),
[scaling plot](reports/p97_capacity_scaling/scaling.png),
[all 200 runs](reports/p97_capacity_scaling/confirmation_runs.csv), and
[validated summary](reports/p97_capacity_scaling/summary.json).

Previous result: the C-first categorical follow-up is complete. Ordinary continuous
C60/FF560 and categorical B60/FF240 have exactly 242,580 parameters each.
C passed the paper-style positive-delay criterion on ten search seeds and ten
fresh validation seeds. On the paired validation seeds, C's minimum delay is
150 updates; B's is zero, with eight positive and two zero delays. Both learn
arithmetic on all ten seeds. Mean first held-out 99% times are C 2,559 and
B 4,559 updates. Mean matched-size random memorisation time is C 1,230 updates;
all ten B times are censored beyond 50,000 under the online/dropout-active 99%
criterion, despite final dropout-free random-label accuracy of 98.26–99.38%.
All 50 new checkpoints passed audits. This is a C-first comparison, with no new
A or claim that B matches dataset capacity. See the
[completed follow-up](#completed-c-first-categorical-grokking-test),
[paired outcomes](reports/p97_categorical_groks/paired_outcomes.csv), and
[validated summary](reports/p97_categorical_groks/summary.json).

Previous result: both new p97/dropout-0.2 experiments are complete across ten
fresh seeds. The fixed dataset was chosen first, A/B were validated and matched
to its 30,866-bit target, and ordinary continuous C was constructed only after
A/B were frozen. Eight-level B reaches sustained held-out 99% on 5/10 seeds
versus C's 10/10; its all-seed median is >49,205 updates versus C's 9,390.
Categorical B and C both succeed on 10/10, with medians 4,035 and 3,235 updates.
Shared A succeeds on 8/10 with median 11,660. Categorical B improves on A, but
C wins nine of ten paired 99% comparisons. Neither discrete model eliminates
positive measured delay. All 115 new checkpoints passed independent audits.
See the [completed comparison](#completed-p97-eight-level-and-categorical-matched-experiments),
[all outcomes](reports/p97_multilevel_dropout/confirmation_runs.csv), and
[validated summary](reports/p97_multilevel_dropout/summary.json).

Previous result: the capacity-matched p97 A/B/C rerun with dropout 0.2 is complete
across ten fresh seeds per model. A and binary B match the fixed dataset within
10% on both capacity seeds; B/C have exactly 117,832 parameters. Sustained
held-out 99% success is A 9/10, B 5/10, C 10/10. C is fastest on every paired
seed at both 95% and 99%. Median updates to sustained 99% across all ten seeds
are A 8,310, B >30,215 (a censoring lower bound), C 2,965. The minimum observed
paper-style delay is zero for all three models, with undefined delays reported
separately. Using eval-mode training accuracy instead, the minima are
A 0, B 0, C 140 updates. A still has positive eval-mode delay on 7/9 defined
runs; a zero minimum is not a claim that every seed has zero delay.
See the [completed results](#completed-p97-capacity-matched-abc-with-dropout-02),
[all thirty outcomes](reports/p97_dropout_match/confirmation_runs.csv),
[speed comparison](reports/p97_dropout_match/speed_comparison.png), and
[capacity evidence](reports/p97_dropout_match/selection.json).

Earlier fixed-A follow-up: the fixed-A dropout rerun and local capacity calibration are
complete. Across ten p97 seeds at dropout 0.2, four delays are observed and all
are zero; six runs never reach online training 99% within 5,000 epochs. Five
reach held-out 99% at least once, and four finish with both accuracies >=99%.
The fresh local matched-protocol capacity slope is 0.807907 bits/parameter,
versus 0.816713 on the earlier seed and 2.095545 under historical defaults.
Direct A capacity is now 21,663–21,820 bits, only 70.2–70.7% of the fixed dataset
size: **A no longer meets the required capacity match under this protocol**.
All 74 new final checkpoints passed independent audits. See the
[ten-seed outcomes](reports/p97_dropout_followup/A_runs.csv),
[local calibration](reports/p97_dropout_followup/summary.json), and completed
follow-up sections at the end of this record. The paper-style rerun changes
measurement, shuffling, optimizer implementation and stopping as well as dropout;
it is not a dropout-only causal ablation.

Earlier follow-up: the p=97 binary A/B/C replication is complete and audited.
A26/FF104 has 26,910 parameters; binary B72/FF62 and ordinary continuous C
have exactly 82,872 each. A/B capacities match the fixed dataset within 10%
on both calibration seeds. Median updates to sustained 95% are A 7,500,
B 15,200, C 2,100. A/C reach sustained 99% on all three seeds; B on two.
Paper-style delays are A 3,200/400/400, B 0/0/0, C 200/100/100 updates.
Thus empirical capacity matching did not force A's measured delay to zero.
See the [p97 results](reports/p97_quantized_v9/confirmation.csv),
[learning curves](reports/p97_quantized_v9/learning_curves.png), and
[capacity match](reports/p97_development/accepted_capacity_match.json).
A later [check of the original paper](reports/paper_p97/provenance.json) confirms
that these delays are not directly comparable to its baseline: dropout, the
training-accuracy measurement, seed aggregation, and capacity calibration differ.

Previous follow-up: pre-softmax logit LayerNorm is implemented and tested. Both
normalized seed-42 pilots completed 120k updates without passing arithmetic
validation (19.34% at temperature 0.1; 88.75% at temperature 1). See the
[normalization evidence](reports/categorical_slot_normalization_v7/comparison.csv)
and the exact binary speed/grokking comparison near the end of this record.

Question: can a large transformer with discrete internal representations match a
small continuous model's empirical memorisation capacity, generalise sooner,
and avoid the generalisation delay of an equally large continuous control?
A positive result is not assumed. Lack of learning is not evidence of reduced
capacity, and failure to grok is not success unless generalisation succeeds.

## Completed p149 results: both full-width discrete families

Both registered A/B/C comparisons are complete on branch `idea/discrete`.
All 18 arithmetic checkpoints passed independent audits. Both selected B
recipes passed their own arithmetic gate before any random-label training,
then learned a separate 2,048-example random-label validation set. The
categorical gate reached 99.2472% hard-mode held-out accuracy and its random
validation reached 100% hard-mode (99.4141% sampled). The binary scalar gate
reached 99.2019% held-out accuracy and its random validation ended at 99.7559%
(best 100%). These development successes did not guarantee confirmation success.

The fixed p=149 division training set contains 11,026 examples; its registered
label-information target is 79,810.65 bits. A's empirical capacity estimates are
80,552.69/80,438.20 bits. Scalar B gives 79,986.88/79,816.66 bits; categorical B
gives 87,313.55/87,428.37 bits. Each B matches A and the target within 10% on both
calibration seeds, under the frozen two-size, 120k-update protocol. These are
finite-budget measurements, not intrinsic capacity bounds or a theorem that A
cannot grok.

| Comparison | A parameters | B parameters | Ordinary C parameters | B seeds reaching 95% | B seeds reaching 99% |
|---|---:|---:|---:|---:|---:|
| Binary scalar quantization | 75,460 | 229,360 | 229,360 | 3/3 | 2/3 |
| Dreamer-style categorical | 75,460 | 253,146 | 253,240 | 2/3 | 0/3 |

A and C reach sustained 99% on all three seeds in both comparisons. At 95%,
scalar median update counts are A 2,800/B 19,300/C 1,800. Categorical A/C medians
are 2,800/1,300; categorical B takes 30,600 and 5,500 updates on seeds 73/75 and
does not reach 95% on seed 74. No observed-only B median is presented as a
three-seed median. B does not beat A or C on any paired 95% comparison.
The same three A seeds are shared across families, not six independent trials.
Scalar B/C have identical parameter shapes and initialization. Categorical C
is an ordinary transformer with 0.0371% more parameters and different shapes.

Neither tested discrete recipe supports the proposed faster-generalization
benefit. Both reduce the transient train-test gap relative to C, but scalar B
is slower and categorical B has substantial learning instability. No run meets
the registered train>=99%/held-out<90% large-gap condition. Under the separate
paper-style first-train 99/first-held-out 98 measure, C has a 100-update delay on
every seed, A has 0, scalar B has 0, and categorical B has missing/missing/0.
Missing thresholds are not evidence of successful grokking avoidance. The
100-update evaluation spacing cannot resolve shorter delays; dropout 0 here
also differs from the paper's dropout 0.2. These are results for the frozen
recipes, not evidence that all discrete transformers must behave this way.

Full evidence: [scalar results](reports/fullwidth_quantized_v6/confirmation.csv),
[scalar curves](reports/fullwidth_quantized_v6/learning_curves.png),
[categorical results](reports/fullwidth_categorical_v5_ff216/confirmation.csv),
[categorical curves](reports/fullwidth_categorical_v5_ff216/learning_curves.png).
Frozen configurations, capacity traces, independent checkpoint audits and
artifact hashes accompany each report. Timing in seconds is descriptive on
shared GPUs; primary comparisons use actual optimizer updates.
The [completion audit](reports/fullwidth_development/completion_audit.json)
verifies all 182 exported artifacts and 36 supporting checkpoints, including
the arithmetic gates and selected small random-label validations.

## Architecture correction and development record

**The user rejected the 14-coordinate architecture. Every earlier learned
low-dimensional channel experiment, including confirmation-p149-v1/v2, is
superseded as evidence for the requested discrete transformer.** Preserve these
runs as historical attempts. Their gate and capacity match do not validate the
new architectures. The unfinished old B seed was scope-stopped, not declared a
learning failure. The ordinary continuous controls remain ordinary-model data.

Current full-width evidence: the normalized binary scalar width80/FF320 model
has an accepted capacity match to A44 and the fixed dataset. All nine scalar
A/B/C confirmations on registered seeds73/74/75 are complete and independently
audited. B is slower than A and C on every seed at95% held-out accuracy; median
update counts are A2,800, B19,300, C1,800. At99%, A and C succeed on3/3 seeds,
B on2/3. B reduces the transient train-test gap relative to C, but does not
provide the proposed speed advantage. No run exhibits the registered
train>=99%/held-out<90% memorization condition. The supplementary paper-style
first train99/held-out98 delay is100 updates for C on every seed and0 for A/B;
this distinguishes a short measured delay from a large memorization plateau.
See the completed scalar result section and reports/fullwidth_quantized_v6 for
the full evidence. A has75,460
parameters; B and unrestricted C each have229,360 with identical initial tensors.
All checkpoints supporting this scalar comparison are independently audited.

Categorical width58/K16,width62/K8 and width72/K4 have independently audited
above-target checkpoint witnesses of92,238.31,88,381.03 and88,283.57 bits.
Width62/FF232/K8 completed its full four-cell grid and failed the fixed-dataset
criterion: recorded per-seed maxima88,205.31 and88,119.97 bits exceed the target
by10.52% and10.41%, respectively. All final checkpoints passed independent
audit, but the historical peak weights were not retained; distinguish these
recorded estimates from the independently reloaded lower-bound witnesses.
Its queued C control was cancelled before any training. This recipe did pass
arithmetic and its own small random-label learning checks. Width62/FF160/K8
completed120k below the arithmetic gate (audited75.06% hard-mode held-out),
so it has no random-label runs. Width62/FF216/K8 passed its own arithmetic
checkpoint audit at14,900 updates (99.2472% hard-mode held-out,98.0863% sampled)
and its own10k-update small random-label check (100% hard-mode,99.4141% sampled).
Its four120k capacity cells are complete and independently audited; the fixed
capacity assessment accepts this recipe. This B has
253,146 parameters (3.3547x A); ordinary C width104/FF216 has253,240 (+0.0371%).
Width70/FF288/K4 and both width60/K8 initializations also completed120k below
their arithmetic gates and have no random-label runs.

Other audited scalar candidates, including normalized four-level width80/FF320,
exceed the fixed capacity target. These rejection witnesses are lower bounds,
not saturated capacity estimates. The categorical capacity match is accepted;
the categorical A/B/C confirmation is now complete, as summarized above. Seeds43/44/45 belong to the superseded narrow-channel
experiments; corrected full-width confirmations use73/74/75.

A44's four120k calibration cells are complete and independently audited. The
per-seed estimates are80,552.69 and80,438.20 bits, each within1% of the79,810.65-bit
target; every cell has late relative drift below1.5%. The v4 A training is reused
under the explicitly verified v4/v5 baseline source equivalence below. See
`reports/fullwidth_development` for all observations and audit summaries.

The fixed task remains **p149 modular division, 50% training split** and a
79,810.65-bit label-information target. New A/B/C comparisons are required for
BOTH full-width categorical and full-width scalar-quantized models. A is an
ordinary small continuous transformer. B keeps a full-width discrete state at
every complete transformer block boundary. C is an ordinary unrestricted
continuous transformer; soft categorical probabilities are not the C control.

Categorical B follows DreamerV2 Algorithm 1: one categorical variable per hidden
coordinate, K logits per variable, categorical sampling, hard one-hot forward,
and softmax-probability surrogate gradient. At width 96 it has **96 variables**,
not 96/K or 14. A learned full-width readout produces 96*K logits; concatenated
one-hots feed a learned linear embedding into the next stage. No other state
crosses the boundary. Count both logits and embedding matrices in B's parameters.
Training samples hard categories. Source-v5 primary evaluation uses hard argmax;
sampled evaluation is retained separately. Earlier pilots used sampled-primary
evaluation and keep that original interpretation. Evaluation RNG is isolated from training RNG. No Dreamer
world-model prior, KL objective, or deterministic recurrent state is added.

Quantized B rounds all native residual coordinates onto a fixed uniform grid after
each complete block. Initial pilots use clip range [-1,1] and identity STE
through rounding/clipping. There is no learned compression, expansion, or
input-dependent scale. This B and its ordinary C have identical parameter
shapes, counts, and initial values. For categorical B, choose an ordinary C width
whose total parameter count is closest (record the difference explicitly).

Validation before launch: 35 focused checks passed, including exact hard forward
values, softmax/identity surrogate gradients, no bypass at either boundary,
full-width shapes, and exact interrupted/resumed sampled training. Independent
checkpoint audit smoke checks passed for both families on CPU.

Arithmetic-only development registration: width 96, depth 2, one head, p149,
50% split, seed 42, lr 0.001, AdamW betas (0.9,0.98), weight decay 1, dropout 0,
batch 512, budget 60,000 optimizer updates, evaluation every 100 updates,
checkpoint every 2,000, held-out >=99% for three consecutive evaluations.
Quantized pilot levels: 2, 4, 16. Categorical pilot alphabets: 2, 4, 16. Primary
categorical arithmetic gate uses sampled evaluation, not just argmax. Failed
recipes remain excluded from random-label testing. Each new recipe needs its
own gate and checkpoint audit before calibration. Capacity is measured with
random labels, never inferred from the finite state alphabet. Parameter ratio,
dataset match, and paired fresh-seed requirements below remain in force.

Sources checked against official implementation:
[DreamerV2 Algorithm 1 and Appendix D](https://arxiv.org/pdf/2010.02193),
[OneHotDist](https://github.com/danijar/dreamerv2/blob/main/dreamerv2/common/dists.py),
[latent readout and embedding](https://github.com/danijar/dreamerv2/blob/main/dreamerv2/common/nets.py).
Dreamer's published latent has 32 variables with 32 categories each; the
mechanism is adapted here, rather than copying its dimensions or recurrent model.

The internal reference is
`language-regularisation/src/language_regularisation/common/interfaces.py`:
its `straight_through` uses hard argmax forward with a softmax surrogate.
`models/symbolic/shared/channel.py` sends hard values to consumer-owned
embeddings; its probability diagnostics are separate from that interface.
Our categorical adaptation uses DreamerV2 sampling during training, preserves
the softmax surrogate, and places the hard state at every complete Transformer
block boundary. The internal NEO quantizer uses a learned normalized codebook
and an identity surrogate; our scalar family instead uses a fixed finite grid
on every residual coordinate. Neither internal architecture is copied wholesale.


## Mandatory order

1. Verify finite forward values, STE gradients, matched B/C parameters, and no
   bypass around channels. Demonstrate that a discrete model learns modular
   arithmetic to >=99% held-out accuracy for three consecutive evaluations.
   **No random-label training before this gate passes.** This order is the
   user's explicit instruction of 24 October.
2. For successful discrete recipes, measure random-label memorisation against
   small continuous models and the paired large continuous model on this host.
   Use multiple dataset sizes, independent seeds, and horizon extensions where
   capacity is not saturated. Distinguish failed optimisation, dataset-limited
   measurements, finite-budget estimates and demonstrated plateaus.
3. Select A/B by calibration only: require each model's empirical capacity within
   10% of the fixed modular training-set label complexity AND within 10% of each
   other, with B/A parameter ratio >=3 for the source-v5 follow-up (originally >=4).
   Matching A to B alone is insufficient.
   Keep the training split fixed at 50%; tune models to the dataset. Freeze selection
   before fresh modular seed comparisons. C is an ordinary unrestricted continuous transformer. Quantized B/C share
   parameters and initialization; categorical B/C match total parameter count
   as closely as the ordinary architecture permits, with the difference reported. If no defensible match
   is found, report that and continue calibration rather than claiming a match.
4. Compare A/B/C on identical modular train/held-out splits and batch orders for
   >=3 fresh seeds. Report steps, examples seen, epochs and wall seconds to
   sustained 95% and 99% accuracy, train-to-held-out delay and maximal gap.
   Never replace censored runs with a fabricated threshold-crossing time.

## Historical architecture and estimator — superseded

Start with the repository's two-block, one-head transformer. After each complete
residual block, replace the entire residual stream by a learned scalar channel:
RMSNorm -> linear encoder -> tanh -> uniform finite-level rounding -> linear
decoder. B uses exact quantised activations in training and evaluation; backprop
uses identity STE with respect to the bounded latent (tanh's derivative remains).
C retains the same bounded latent without rounding and has identical tensors.
A is the unmodified continuous transformer at a smaller width. The last output
reads only the last channel; there is no continuous skip around a bottleneck.
No relaxation warmup, quantiser auxiliary loss, or target-derived codes.

The channel carries at most latent_dim * log2(levels) bits per token per layer.
This does NOT bound bits stored in the learned encoder's weights: even a code
with p symbols could encode any learned p-way label function. Empirical capacity
is therefore measured, never inferred from the message bit count. Repeated
quantisation may also alter optimisation; random-label failures alone cannot
establish a representational ceiling.

## Historical development and evidence — superseded

Start on modular division, p=97, random 50% split, depth=2, width=96 for B/C,
AdamW lr=1e-3, betas=(0.9,0.98), weight decay=1, batch=512. Use no dropout in
this initial architecture diagnostic, matched across conditions; this differs
from the paper's central dropout=0.2 protocol. Check bottlenecks of 8/16/32
coordinates with 2/4/16 levels. Initial seed 42 is development only. Extend or
amend based on development evidence, preserving all failed/censored runs.

New runs live in `results/discrete/`, separate from the legacy registry because
its identifying tuple cannot distinguish these new architectures. Each run
stores resolved config, source hashes/snapshots, dataset digest, trace, full
checkpoint including optimiser/RNG state, and terminal summary. Runs refuse
silent overwrite; resume requires the same scientific source/config. Training
budgets and selection changes must be recorded here before new comparisons.
GPU wall times on this shared machine include contention and are secondary to
optimizer updates; record hardware and background-process snapshots.

Capacity uses M = n*(log2(V) - cross_entropy/ln(2)), preserving negative estimates
rather than clipping them. A bits/parameter number is descriptive unless a
capacity plateau and a linear scaling fit have actually been established.

Reference implementation inspected: language-regularisation common/interfaces.py
(hard forward categorical channel and surrogate gradient), symbolic/shared/channel.py
(no probability side channel), and symbolic/neo/quantizer.py (identity STE).
This experiment adapts the mechanism to internal scalar latents.

Method references: [Morris et al.](https://arxiv.org/abs/2505.24832),
[Bengio et al.](https://arxiv.org/abs/1308.3432). Their results do not establish
capacity equivalence for this new architecture.

## First arithmetic evidence — 24 October 2026

`results/discrete/learning-v1/B_l32_q16` passed the arithmetic gate. Width 96,
32-coordinate/16-level channels, 326,880 parameters, seed 42: sustained held-out
99% crossing at update 1900; final update 2100 train 99.9785%, held-out 99.7423%.
The paired continuous model crossed at update 1700; final update 1900 train 100%,
held-out 100%. Train-to-held-out 99% delays were 800 and 700 updates respectively.
This initial bottleneck learns but does not demonstrate the proposed advantage.
It has not yet been capacity-matched to a small model.

An independent CPU reload reproduced both accuracies. Initial tensor digests and
dataset digests are exactly equal between B/C. Every inspected discrete latent
was on the finite grid; the 128-bit final-position code used 4656 distinct values
on 4656 held-out examples. Thus this permissive channel has not forced a compact
answer code. Evidence: `results/discrete/learning-v1/arithmetic-audit.json`.

The development queue remains arithmetic-only while tighter bottlenecks are
assessed. Eleven focused checks pass, including interruption/recovery equivalence
with nonzero dropout. The system's unrelated libtmux pytest plugin is incompatible
with its pytest version; checks use `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

Arithmetic-only follow-up `configs/discrete_learning_v2.yaml` is registered
before launch. The initial 16-coordinate/4-level model approaches 99% but jitters
below the sustained gate; test more resolution at that width, more coordinates
at low resolution, and a matched lower learning rate (3e-4). Test an 8-coordinate,
16-level channel to distinguish width from quantisation resolution. All are
seed 42 development; preserve the initial failures and the 99% gate. The width 32
continuous control is exactly the initial C_l32_q16 model for every level count
because levels do not enter its forward computation. Lower-rate controls are
run separately. This expansion uses arithmetic evidence only.

Full regression result: 48 passed, 1 failed. The existing legacy figure test
`test_arch_key_fields_cover_all_identifying_minus_seed_dim_n_samples_p` fails
because ARCH_KEY_FIELDS omits mup_base; its model/registry/test files are unchanged
from HEAD. New experiment checks: 11/11 pass. Broad pytest discovery was interrupted
while scanning the artifact tree; the completed suite explicitly targeted tests/.

The initial B_l16_q4 run exhausted 30,000 updates: final train 99.0120%, held-out
98.2603%; maximum held-out 99.4416%, but no three-evaluation 99% crossing. It is
excluded from capacity tests. It sustained 95% train and held-out simultaneously
at update 8600, with maximum train/held-out gap 20.4 percentage points. This is
promising only as a reduced-gap observation; it fails the registered 99% gate.

The v2 arithmetic queue runs concurrently with the remaining initial arithmetic
pilots on GPU 1. These exploratory elapsed times include shared-device contention;
final matched comparisons will use serial runs and report timing limitations.
Continuous controls differing only in levels are mathematically identical, so
v2 reuses C_l32_q16/C_l16_q4/C_l8_q2 for the matching latent widths at lr 1e-3.
Only the new lr 3e-4 condition needs a new continuous control. No random-label
training has started at this point.

## Prospective capacity screening

Only the independently verified B_l32_q16 recipe is initially admitted. The
first calibration (`configs/discrete_capacity_pilot.yaml`) uses a fresh random
seed 61, 8192 unique random four-token inputs, independent uniform 99-way targets,
and 20,000 updates per model. Test small A widths 16/20/24, successful B_l32_q16,
and its identical-parameter C. Optimiser, dropout, data and batch order are matched.
The uniform-label dataset contains 54,307 bits versus 30,866 bits for the modular
training set under the paper's log2(p+2) convention (rounded figures).

This first screen is only a finite-budget measurement. Do not call its largest
memorisation value a saturated capacity or infer a constant from one data size.
Report tail-mean bits, tail drift and fraction of the label-information ceiling.
A candidate match needs at least two dataset sizes beyond the interpolation
regime, independent calibration seeds, and horizon extension if still improving.
Require <=10% relative A/B capacity difference and >=4x parameter ratio, with
uncertainty and remaining optimisation dependence reported. Measure more sizes
or reject the match if either model interpolates the calibration data. A failed
random-label optimiser is not a proved representational limit.

Run this screen only after arithmetic evidence exists, using its explicit --gate.
The narrow models that failed the arithmetic gate remain excluded. A later
successful narrow recipe requires its own gate evidence before calibration.

The v2 B_l16_q8 recipe passed its arithmetic gate and independent checkpoint
reload: 320,736 parameters, 48 bits/token, sustained training 99% at update 1200,
held-out 99% at 1300, final held-out 99.4201% at 1500. Its paired continuous control
C_l16_q4 (levels is inactive in continuous mode) reached training 99% at 1300 and
held-out 99% at 1600. Both initial tensor and dataset digests match exactly.
The observed delays are 100 vs 300 updates, with 100-update measurement resolution.
This is a single development seed selected among several attempted recipes, not
confirmation. The final code still has 4563 distinct values on4656 held-out rows.

After this successful audit, register `configs/discrete_capacity_narrow.yaml`:
paired B_l16_q8/C runs at 8192/16384/32768 random examples, seed 61, 20,000 updates
each with the same optimiser. These are initial screens; larger data sizes have
fewer epochs at the same update budget, and increasing curves must be extended
before any capacity match is accepted. The initial A measurements use identical
8192-example data. All launches pass their particular successful arithmetic gate.

## Current continuation state

The objective is **not complete**. Capacity-equivalent A/B models have not been
selected; no fresh-seed matched A/B/C comparison or architecture capacity-constant
fit is claimed. The remaining work is to finish calibration, extend unresolved
curves, choose the empirical match, then run and analyse fresh paired modular
seeds. Preserve unsuccessful recipes and censoring.

Historical queues at registration (use scripts/discrete_status.py for current activity):

| Queue | Worker PID | Execution session | Configuration |
| --- | ---: | ---: | --- |
| Initial arithmetic | 1120695 | 2090 | discrete_learning.yaml |
| Follow-up arithmetic | 1129595 | 65875 | discrete_learning_v2.yaml (selected runs) |
| Initial capacity screen | 1132330 | 39965 | discrete_capacity_pilot.yaml |
| Narrow candidate capacity | 1138743 | 44291 | discrete_capacity_narrow.yaml |

All use GPU 1. Their outputs are respectively results/discrete/learning-v1,
learning-v2, capacity-pilot-v1, capacity-narrow-v1; corresponding .log files are
under results/discrete/. Checkpoints contain optimiser, batch permutation/cursor,
and RNG states. No completed run should be restarted. Queues snapshot scientific
source at interpreter startup. Wall times are training/evaluation/checkpoint loop
times, exclude setup, and are confounded by concurrent shared-GPU use. The saved
CUDA peak metric is a process-wide upper bound across sequential runs, not an
isolated per-model memory benchmark.

At the latest initial capacity screen, A width 16/20 at 8192 random examples and
20,000 updates had tail-mean memorisation 10,997/17,132 bits (11,440/16,860 params),
with late-window drift 391/551 bits. These are finite-budget values, not verified
saturated capacities. B_l16_q8 already exceeded 49,000 bits before its screen
finished, so these two small A models are not plausible matches. Await the
larger-dataset curves and select wider A candidates from capacity evidence.

B_l8_q16 subsequently passed its arithmetic gate and independent checkpoint
audit; consult its summary/audit before considering a new capacity screen.
It is not automatically included in the already running calibration queues.
The strongest binary model B_l8_q2 failed arithmetic (final held-out 26.95%);
B_l32_q4 also failed the sustained 99% gate (final held-out 98.80%). Neither is
eligible for random-label training.

The direct candidate/control plot is
results/discrete/narrow_development_comparison.pdf (and PNG). Refresh full
learning summaries with scripts/analyse_discrete.py and calibration summaries
with scripts/analyse_discrete_capacity.py. The test result remains 11/11 focused
checks and 48/49 full tests, with the unchanged legacy mup_base grouping failure.

## Calibration continuation — 24 October 2026

The previous goal turn made progress: implemented/audited discrete learning,
recorded both failures and successes, and launched gated capacity screens. On
continuation all four workers were verified live; none was restarted.

Initial arithmetic A_d24 reached sustained held-out 99% at update 8900; A_d16
exhausted 30,000 updates at 79.51% held-out. These cannot yet be contrasted as
capacity-equivalent with B. The initial B_l16_q8 screen ended at 50,412 bits on
8192 random examples, above the roughly 24,000-bit developing A_d24 screen.
Register wider continuous capacity probes at dimensions 28/32/40/46, n=16384,
seed 61, 20,000 updates, keeping the optimiser fixed. The largest of these still
has fewer than one quarter of B_l16_q8's parameters. Selection uses capacity
results, not their as-yet-unseen modular performance.

`scripts/extend_discrete.py` permits longer calibration from an exact terminal
checkpoint in a new output directory. Only max_steps changes; model, optimiser,
batch order/cursor, RNG, and trace are preserved. It stores the original scientific
declaration and checkpoint/manifest/summary hashes. The original run is immutable;
the extension is explicitly adaptive and does not retroactively change its budget.
This retains the unchanged training implementation and arithmetic-gate source
binding. Capacity curves still increasing at 20k updates will be extended to 60k
before being used to claim a match.

Capacity-estimator clarification before matching: retain the **best achieved**
M_T as a lower bound in addition to late-window averages/drift. C_l16_q8 reached
100% random-label training accuracy and about53,756 of54,308 available bits, then
sometimes forgot labels under continued AdamW training. A lower tail mean must
not be called a smaller capacity if the model already demonstrated a larger fit.
The capacity summary now flags dataset limitation using best achieved bits,
rather than terminal or tail bits alone. This follows the original upper-envelope
interpretation; it is not evidence that the best noisy observation is a saturated
capacity. Selection will require repeated seeds, larger datasets and horizon checks.

The fixed-rate checkpoint extension test passed with exact final weights, batch
state, RNG, and non-timing traces versus uninterrupted training, with nonzero
dropout; the parent checkpoint bytes remained unchanged. Focused tests:12 passed.
Register 20k ->60k extensions for initial B_l16_q8 at n8192 and small A_d24 at
n8192 as convergence checks, retaining their original runs. These are diagnostics,
not a claim that this particular pair is capacity-matched.

The independently audited B_l8_q16 recipe is now admitted to its own screen:
8192/16384/32768 examples, seed61,20,000 updates; include its continuous control at
16384 examples. This tests a stronger candidate bottleneck without conflating
failure to learn arithmetic with reduced capacity. It runs on GPU0, whose available
memory was rechecked; capacity-step comparisons do not use cross-GPU wall time.
The final fresh-seed A/B/C timing comparison remains serial on one device.

## Task-size adaptation from capacity evidence

Before seeing any p163 arithmetic outcome, register a p163 modular-division
learning gate for the fast B_l16_q8 recipe and its paired C. At p97/50% split,
the training labels contain30,866 bits, whereas this B already demonstrated
66,611 bits on n16384 random examples and is still improving. A capacity match
at that level would not test the intended near-dataset-capacity regime.
At p163/50%,13203 training examples carry about97k bits, closer to the developing
capacity estimates. Every A/B/C comparison will use one identical prime/split;
p97 results remain exploratory and are not pooled with p163. This is task-size
selection based on capacity, not selection on unseen p163 generalisation results.
The p163 model has333,408 parameters; a width46 continuous A has83,122 (ratio4.01).
Widths and capacity matches remain provisional. No random p163 experiment may
start until that specific discrete architecture passes its new arithmetic gate.
YAML configuration: configs/discrete_learning_p163.yaml, fresh task development
seed42, at most60,000 updates with the same constant-rate optimiser.

The stronger B_l8_q16 screen at n8192 ended near37,988 bits, while A_d32 at
n16384 is developing toward a similar scale. This is a possible near-dataset-size
match at p97 (dataset30,866 bits) and would retain an8x parameter ratio. However,
B_l8_q16 learns fewer bits at n16384 under the short budget, indicating unresolved
optimisation/data-size dependence. Do not accept this match yet. Extend its
n8192 and n16384 screens to60k updates before choosing A. The p163 exploration
remains separate; it does not replace this potentially informative p97 comparison.
A negative result (large discrete B learns slower than matched A) remains a valid
answer to the hypothesis; no selection rule requires a positive result.

## Candidate-match confirmation registration

The n8192 B_l16_q8 continuation completed 60k updates, reaching at most 51,131
bits, versus 50,412 at 20k. Its n16384 screen demonstrated 69,419 bits, so the
smaller-data plateau is not its overall capacity. The p163 arithmetic probe is
still unsuccessful and remains excluded from random-label tests.

For the stronger B_l8_q16 bottleneck, the n8192 extension has reached about
39,544 bits, while A_d32 achieved 37,035 on n16384 at 20k. Its parameter ratio is
317,664 / 39,264 = 8.09. Register a focused calibration comparison at 60k updates,
seeds 61 and 62, and n8192/16384 for both models. Extend the existing seed61
checkpoints, and use configs/discrete_capacity_match.yaml for missing cells.
The primary empirical estimate is the maximum achieved memorisation across the
registered dataset sizes, reported separately per seed and with tail drift.
Both sizes are beyond interpolation for the candidate pair. This remains an
optimiser- and budget-dependent estimate, not a proved representational bound.

Accept the proposed width32 match only if the relative A/B difference is at most
10% on each calibration seed, using A as the denominator. Increasing late curves
or strong seed dependence will be reported as unresolved; no universal bits per
parameter constant will be inferred. Do not pick a width using new modular
results. After calibration freezes the choice, test paired modular seeds43/44/45
with a 60k-update budget, using the same data, order, and optimiser across A/B/C.
Run these comparisons serially on one GPU; report sustained 95/99 thresholds,
train/test delay, maximum accuracy gap, loop wall time, and censored runs.

The earlier proposed A_d24 60k diagnostic is dropped before launch: that width
is far below either candidate's observed memory, and its extension cannot resolve
the selected pair. Original results are retained. This is a documented scope
reduction, not a changed result. The higher-capacity B_l16_q8 and p163 screens
remain exploratory and do not replace this potentially negative comparison.

## User correction: mandatory three-way match

The user clarified that capacity(A), capacity(B), and modular training-set label
complexity must all match; A is intended to be in the regime without memorisation
followed by grokking. The user explicitly chose to keep the dataset fixed and tune
the models, rather than change the split to accommodate the measured capacities.
The 50% split is fixed. No final width32/B_l8_q16 comparison is accepted at p97:
40k bits is materially above the 30,866-bit dataset target. The preceding
candidate-match registration is superseded; its partial results remain diagnostics.

Under the existing log2(p+2) convention, p97 requires 27,780–33,953 measured bits
per model (10% tolerance around 30,866). Additionally require the A/B difference
within 10%. All stopped workers retain checkpoints/traces and scope_stop.json;
they are scope interruptions, not budget failures or evidence of saturation.
The exploratory p163 task adaptation was stopped, and no p163 random tests ran.

The user then suggested considering p147. Since 147=3*7^2, the current division
implementation's Fermat inverse is invalid for that modulus. Prime149 with the
same division task was recommended; p147 addition is the alternative presented.
At p149/50%, 11,026 examples contain 79,811 bits, so the admissible per-model range
would be 71,830–87,792 bits. Any new modulus needs its own discrete arithmetic
gate before random-label training and recalibration with its actual vocabulary.
The modulus/task choice is pending the user's response; the split remains 50%.
Capacity matching does not mathematically guarantee absent grokking; A's curves
must verify that intended experimental condition on the fresh seeds.

Status tooling now distinguishes scope-stopped experiments and --active-only
shows actual live workers. Extension manifests now record the continuation device
and start time rather than copying those fields from the parent. For earlier
extensions, the manifest's device could be inherited; command-line provenance in
scope_stop.json identifies the actual continuation device. No timing comparison
uses those exploratory runs.

## Confirmed task: p149 modular division, fixed 50% split

The user selected prime149 with modular division. This supersedes p97 as the
final comparison task; the training fraction remains fixed at 0.5. The target is
79,810.6507 bits under the registered label-information metric. A and B must each
be within 10% of this target and within 10% of each other. No arithmetic outcome
at p149 has been observed when making this selection.

Register configs/discrete_learning_p149.yaml: width96, two layers, seed42,
16-coordinate channels with 8 or 16 levels; identical bounded-continuous control
(the levels setting is inactive for C). All use lr0.001, weight decay1, dropout0,
60k maximum updates, and the same sustained 99% gate. The eight-level candidate
was the fast development recipe at p97; the sixteen-level candidate tests whether
more resolution restores learning on the larger task. There is no random-label
training at p149 before a corresponding discrete arithmetic gate and audit pass.
If neither recipe learns, preserve the failures and adapt using arithmetic
results only; matching cannot be inferred from unsuccessful training.

Based only on p97 calibration, width46 is a plausible small A starting point:
71,642 demonstrated bits on that smaller vocabulary. This does not establish its
p149 capacity. Recalibration will use the actual p149 vocabulary, at least two
non-interpolated dataset sizes, two independent calibration seeds and horizon
checks. Fresh modular confirmation seeds43/44/45 remain unused. The proposed
width32/B_l8_q16 p97 match is abandoned, with its evidence retained.

Prepared, but not launched before the gate: configs/discrete_capacity_p149.yaml.
First screens use n16384 and seed61 for A widths44/46, successful B recipes, and
C_l16. All have a 20k-update exploratory budget with horizon extension required
before matching. A44/A46 have 75,460/81,834 parameters; B and C have 330,720, so
both preserve the >=4x parameter-ratio criterion. Candidate screening uses the
79,811-bit target; candidates already demonstrating more than87,792 bits are
excluded, rather than matched to a larger A. A low short-budget estimate is not
accepted as evidence of small capacity. A final candidate still needs n32768,
independent seed62, and 60k horizon checks on both dataset sizes.

`scripts/assess_discrete_match.py` enforces the fixed-target comparison from
completed calibration artifacts. It requires paired seed/size cells, consistent
scientific sources and optimisation settings, at least two seeds and sizes,
60k updates, no dataset-limited calibration, and at most 5% absolute relative
late-window drift. This numerical drift rule is registered before any p149
capacity result. It checks both target matching and A/B matching on every seed,
plus the >=4x parameter ratio. Passing means an empirical fixed-budget match,
not a proof of finite representational capacity or absence of grokking. Tests
explicitly reject an equal A/B pair that exceeds dataset complexity.

Arithmetic-only precision follow-up: after both p149 eight- and sixteen-level
runs slowed near 90% held-out beyond 13k updates, register a 64-level variant at
the same 16-coordinate width, seed42 and all other settings unchanged. This is a
precision diagnostic, not a capacity match. It needs its own sustained 99% gate and
independent audit before random labels. The 79,811-bit target remains unchanged;
a more accurate model that exceeds the target will be rejected for the final
comparison. Initial recipes retain their 60k budgets and are not declared failed
based on an intermediate plateau. Config: discrete_learning_p149_precision.yaml.

Prepare the same n16384/seed61 capacity screen for B_l16_q64, contingent on its
own gate. It has the same 330,720 parameters and initial tensors as the other
16-coordinate candidates. At around 4k updates its held-out accuracy is near 99%,
but the sustained gate has not yet passed. No calibration is launched on this
intermediate result. The capacity-assessment tests and existing experiment tests
all pass: 15 focused checks.

## p149 arithmetic gate passed before calibration

B_l16_q64_p149_s42 completed at 5600 updates with train 99.7551% and held-out 99.5465%.
Its three-evaluation 99% crossings were train 2700 and held-out 5400 (delay 2700),
and maximum train/held-out accuracy gap was 20.01 percentage points. The independent
CPU checkpoint audit reproduced both accuracies, source/data/initial-weight
hashes, and exact finite grid values at both layers. All 11,026 held-out final
codes were distinct; the 96-bit message alphabet does not imply compact weight
memory. This is an arithmetic-success gate, not evidence that grokking vanished.

Only after that audit, launch the p149 n16384/seed61 random-label screens for
B_l16_q64, the paired continuous C, and smaller A widths44/46. Gate evidence is
results/discrete/learning-p149-v1/B_l16_q64_p149_s42. Eight- and sixteen-level
recipes remain in arithmetic training and are not admitted to random labels.
The target stays 79,810.6507 bits; no match or fresh-seed outcome is claimed yet.

The p149 B_l16_q64 learning curve does not show a clear long memorisation plateau:
95% train/test crossings are1800/2100, and both rise together. The2700-update
99%-threshold delay reflects fluctuations near the threshold: between2700 and
5400, training dips below99% at10 of28 observations (minimum97.22%), and held-out
accuracy never falls below96.50%. Do not equate that threshold statistic alone
with strong evidence of classical grokking. Figure:
results/discrete/p149_discrete_arithmetic_gate.pdf (also PNG).

A44's first n16384/seed61 screen completed 20k updates: best72,817.61 bits, final
training accuracy64.03%, late relative drift2.59%. It is inside the dataset target
band but is not accepted yet. Launch the registered20k->60k continuation in
results/discrete/capacity-p149-extensions-v1/A_d44_n16384_s61_60k. Selection still
requires larger data, independent seeds and the discrete candidate's results.

The p149 control C_l16 completed at 2000 updates with 100% train/held-out accuracy;
sustained 99% crossings were both 1800 (zero threshold delay), and 95% train/test
crossings 1300/1800. Its dataset and initial-tensor hashes are identical to B_l16_q64
(levels is inactive in C). This development seed does not support discretisation
as the source of faster learning or reduced grokking: C is faster, with no 99%
threshold delay. The large-versus-small capacity-matched comparison is still open.
Keep this negative result; do not select or discard models based on achieving the
hoped-for C-groks/B-does-not pattern. Paired plot:
results/discrete/p149_development_comparison.pdf (also PNG).

The eight- and sixteen-level p149 recipes exhausted 60k updates at 87.82% and 87.54%
held-out, respectively. Both failed the arithmetic gate and remain excluded from
random-label testing. Their failure to grok is not evidence of useful regularisation.

## Continuation checkpoint after p149 arithmetic validation

Objective remains active and incomplete. No random-label jobs for failed p149
B_l16_q8/B_l16_q16 were launched. B_l16_q64 and C_l16 both have successful independent
CPU audits, matching data and initial tensors. Fifteen focused tests passed.

Live work (verify using manifests/status tool; never restart a live worker):
- capacity-p149-v1: A44 n16384 seed61 completed 20k; A46 follows in that queue.
- capacity-p149-v1: C16 n16384 seed61 completed 20k; B16q64 follows in that queue.
- capacity-p149-extensions-v1: A44 n16384 seed61 continuing 20k->60k.

At an intermediate checkpoint A44's extension had demonstrated 75,589 bits,
A46's initial screen 77,920, and B16q64's initial screen 83,039. These are not
saturated capacities. The target is fixed at 79,810.65 bits. Complete/extend the
candidate screens, reject any B that demonstrates >87,791.72 bits, and choose A
using capacity evidence only. A44 and A46 modular confirmation outcomes remain
unseen. The final candidate requires n32768, seed62, 60k horizons at both sizes,
and assessment by scripts/assess_discrete_match.py before fresh modular seeds.
If B16q64 exceeds the target, a narrower latent width may be probed, but it must
first pass its own p149 arithmetic gate; do not interpret optimisation failure
as reduced capacity. Preserve the negative C-vs-B development result regardless.

## p149 calibration horizon continuation

The preceding goal turn made concrete progress: selected the user-approved fixed
p149 task, completed and independently audited discrete arithmetic learning,
recorded failed recipes and the stronger continuous-control result, implemented
and tested the three-way capacity criterion, and started gated calibration.
All current process identities were revalidated before continuation.

Completed n16384/seed61, 20k-update peaks: A44=72,817.61, A46=78,435.56,
B16q64=85,090.63, C16=87,698.29 bits. These are finite-budget lower bounds.
Their late relative drifts are2.59%,2.73%,1.87%,2.56%, respectively.
A46 and B16q64 provisionally differ by8.48%, with both inside the dataset band,
but a longer horizon may invalidate that match. Extend A46 and B16q64 from their
exact20k checkpoints to60k, as already required by the protocol. A44's existing
extension continues. No new modular outcomes from A44/A46 are inspected.
Do not launch fresh-seed confirmation merely because the initial20k match passes.

## Wider-channel arithmetic diagnostic from capacity evidence

At the same n16384/seed61/20k budget, the continuous16-coordinate model demonstrated
87,698 bits versus85,091 for 64-level B, only about3% apart. Together with C's fast
arithmetic result, this suggests the narrow continuous channel itself accounts
for much of the effective-capacity reduction. It does not establish an advantage
from discretisation. Retain and finish current horizon checks rather than hiding
this negative control.

Before inspecting any fresh modular seeds, register a wider64-coordinate channel
with2 or4 levels at the fixed p149 task, plus its exact continuous control. This
tests whether coarse quantisation can limit memory while the corresponding
continuous channel retains more capacity. The model implementation is unchanged;
only latent_dim/levels differ. Both candidates require the same sustained 99%
arithmetic gate and CPU audit before random labels. Seed42,60k budget, all other
optimizer/data settings unchanged. Config: discrete_learning_p149_wide.yaml.
The79,811-bit target and final selection rules remain fixed. These are exploratory
arithmetic diagnostics, not assumed capacity matches or claimed positive results.

At38400 updates, B16q64's continuation demonstrated87,959.88 bits, exceeding the
fixed upper bound87,791.72. This recipe is rejected for the final match even
though the excess is small; do not relax the registered tolerance. Finish its
existing horizon as a diagnostic, but do not run independent-seed confirmation
for this rejected match. A44 finished60k with78,516.19 bits, close to the fixed
79,810.65-bit target; A46's extension is still running.

Register a capacity-directed14-coordinate/64-level arithmetic probe and its exact
continuous control, all else unchanged at p149,seed42. This narrows the latent
width in response to excess measured memory, not in response to fresh modular
outcomes. It must pass the same arithmetic gate and independent audit before any
random labels. Config: discrete_learning_p149_l14.yaml. The wider coarse-channel
diagnostics continue separately; no successful learning or capacity match is
assumed for either family.

A44 is the preferred small-model candidate for further calibration because its
completed 60k estimate78,516 bits is close to, and just below, the fixed dataset
target. This selection uses capacity only; its p149 modular outcomes are unseen.
Register its remaining n32768/seed61 and n16384/32768 seed62 cells, each60k updates,
in configs/discrete_capacity_p149_a44_confirm.yaml. These small-model calibration
runs use the already verified p149 B16q64 arithmetic gate; they do not admit any
untested discrete architecture. A46's existing extension is retained as a width
scaling diagnostic. No fresh modular confirmation begins before a valid A/B match.

Before fresh-seed comparisons, add a descriptive memorisation-span statistic to
scripts/analyse_discrete.py: consecutive sampled evaluations with train>=99% and
held-out<90%, reporting the longest observed span, observation count and whether
the span reaches the last evaluation. A single observation has zero observed
span. Report this alongside95/99 crossing delays and full curves; it is not a
binary definition of grokking. It prevents interpreting near99% jitter alone as
an extended memorisation plateau. No unobserved threshold time is imputed for
censored runs. Fresh seeds43/44/45 remain unseen.

After B14q64 reached isolated >99% held-out observations but not a sustained gate
through16,900 updates, register a128-level precision follow-up at the same14
coordinates and all other settings unchanged. This addresses arithmetic stability;
it does not relax the gate or assume a smaller memory estimate. Either successful
recipe must still be calibrated against the fixed target and may be rejected if
its memory is too large. Config: discrete_learning_p149_l14_precision.yaml.
The initial64-level run retains its60k budget; both results remain in the record.

B14q64 completed its arithmetic gate at24,500 updates: final train99.8368%,
held-out99.8549%, and maximum accuracy gap8.59 percentage points. Sustained95%
train/test crossings were2100/2300; sustained 99% crossings were24300/9500, yielding
a negative99%-threshold delay from near-threshold fluctuations. This is not a
memorisation-then-generalisation plateau. The independent CPU checkpoint audit
must pass before launching its calibration. The more precise128-level arithmetic
diagnostic and its continuous14-coordinate control continue separately.

Register the B14q64 calibration grid: n16384/32768, seeds61/62,60k updates, identical
optimiser/data settings to A44. Config: discrete_capacity_p149_l14_confirm.yaml.
Start the seed61 sizes after the audit, retain the unchanged79,811-bit target and
all prior rejection rules, and require the full repeated grid before accepting a
match. Gate path: results/discrete/learning-p149-l14-v1/B_l14_q64_p149_s42.

The completed 60k horizon checks are A44=78,516.19, A46=83,675.86, B16q64=89,670.24
bits. Late-window relative drift is approximately 1.37%,1.23%,1.25%, respectively.
Thus the original A46/B16 pair is still close to each other but fails the user's
three-way requirement because B exceeds dataset complexity by 12.35%.

B14q64's CPU audit passed with all checked latent values on the finite grid;
final accuracies agree within 1e-6. Only after that audit were the seed61 random
capacity runs started. The analysis/integrity tests now pass 17/17. The analysis
adds no training-source changes, so source-bound arithmetic gates remain valid.

Complete A44 n16384 seed62 at60k:78,755.65 bits (late relative drift1.33%), closely
matching seed61's78,516.19. A44 n32768 seed61 completed at68,912.07 bits (drift1.86%).
The lower larger-data result is retained as optimisation/budget dependence; the
registered primary estimate remains the maximum across tested sizes per seed.
All remaining A44/B14q64 seed/size cells are now launched with 60k budgets.

Prepare descriptive C14 capacity controls at n16384/32768,seed61,60k updates using
the matching B14 arithmetic gate. Config: discrete_capacity_p149_c14.yaml. These
measure whether the continuous control actually has more effective memory;
that is not assumed from its continuous representations. C measurements do not
determine which A/B pair is selected. Final modular confirmation will rotate
A/B/C order across seeds43/44/45 (ABC,BCA,CAB), serially on one GPU after matching.

## Full calibration grid in progress

A44's completed peak estimates are78,516.19 bits at n16384/seed61,
78,755.65 at n16384/seed62, and68,912.07 at n32768/seed61. Its remaining n32768/seed62
cell is running. All four B14q64 calibration cells are running with 60k budgets.
Intermediate n16384 peaks were80,991/80,811 bits on seeds61/62; these remain
provisional. The fixed target and acceptance rules have not changed.
C14's descriptive n16384/32768 seed61 capacity controls have also started, after
the B14 arithmetic gate and audit. They do not determine A/B selection.

Both wide coarse discrete models exhausted 60k arithmetic updates and failed:
64-coordinate binary held-out19.21%, four-level21.19%. They are excluded from
random-label tests. Their exact continuous control reached99.74% held-out at
24k updates, with95% train/test crossings800/2000 and 99% crossings4100/23800.
Interpret its99% delay alongside the full curve, not as proof of a long
memorisation plateau. The14-coordinate continuous and128-level precision
arithmetic diagnostics remain live; preserve all outcomes, including failures.

After the calibration grid completes, run scripts/assess_discrete_match.py with
the four exact A44 directories and four B14q64 directories. If it passes, freeze
those configurations and run modular seeds43/44/45 in the registered rotating
order on one GPU, without overlapping our other workers on that GPU. Report
successful and censored runs, all 95/99 metrics, sampled memorisation spans,
parameter/capacity ratios, paired curves, and shared-hardware timing limitations.
No A44 modular outcome or fresh confirmation seed has been inspected yet.

## Accepted empirical match and frozen confirmation

All eight A44/B14q64 calibration cells completed their 60k budgets and passed
scripts/assess_discrete_match.py without exceptions. Result:
results/discrete/p149_match_assessment.json. Seed61 A/B=78,516.19/81,896.76 bits;
seed62=78,755.65/82,972.08. B exceeds A by 4.31% and 5.35%; each is within 4% of the
79,810.65-bit target. Parameter ratio 329,952/75,460=4.37254. All late drifts are
below the pre-registered 5% threshold; both dataset sizes are non-interpolated.
The empirical estimate is the registered maximum across tested sizes per seed.

The current candidate is frozen using capacity evidence only. Before launch,
none of A44's modular outcomes or confirmation seeds43/44/45 were inspected.
`scripts/confirm_discrete.py` recomputes acceptance, validates the original discrete
arithmetic gate/audit and training-source identity, records hashes of all input
artifacts, and freezes exact model configurations. It waits for our other workers
on the chosen GPU, then runs ABC/BCA/CAB order for seeds43/44/45 with 60k limits.
Failed or censored runs are retained. It supports explicit provenance-checked
resume and refuses concurrent duplicate workers. Entry point:
`PYTHONPATH=src python -m scripts.confirm_discrete`.
Eighteen focused checks pass, including fresh-data pairing and rotation order.

Development outcomes retained: B14q128 exhausted 60k at 82.58% held-out; C14 at 90.11%.
Both final checkpoints were independently reloaded, and their data/initial-tensor
hashes match successful B14q64. More quantisation levels did not monotonically
improve learning. These development outcomes do not alter the frozen models,
optimizer, dataset, seed set or capacity criterion.

## Confirmation running after the accepted match

The confirmation worker is live in results/discrete/confirmation-p149-v1;
PID 1205174, session 87318, device cuda:1 (verify PID/start identity before action).
Its orchestrator.json is authoritative; selection.json freezes the accepted
assessment, all input-artifact hashes and all nine model/seed configurations.
Only this worker uses our confirmation GPU; descriptive control calibration ran
on GPU0. The worker automatically executes the remaining frozen runs and refuses
to substitute a different model after seeing fresh outcomes.

A_s43 completed at 3100 updates, final held-out 99.7279%, with sustained 95% at 2700
and 99% at 2900. B_s43 exhausted its 60k budget: final train 97.1250%, held-out
96.8076%, sustained 95% train/held-out at 4900, and no sustained held-out 99%
crossing. Independent CPU reloads reproduce both final results, and B's internal
codes are exactly on the finite grid. C_s43 is training; the remaining fresh seeds
are still pending. This early result favours A but is not an aggregate conclusion.
Retain the registered budget and remaining seeds; failed generalisation is not
counted as successful avoidance of grokking.

Remaining work: let all nine confirmation runs complete; independently audit
final checkpoints and paired data/initial tensors; recompute capacity acceptance;
analyse all 95/99 crossing metrics, sampled memorisation spans, timing/censoring,
and paired seed differences; produce reviewable plots/tables and a final report.
Do not mark the objective complete while any of this is missing. Implementation
and focused analysis checks: 20 passed. Training scientific sources remain unchanged.

`scripts/report_discrete.py` prepares the final numerical evidence and plots only
after all nine runs finish and their independent checkpoint audits exist. It
revalidates the frozen configurations, calibration hashes, capacity acceptance,
paired datasets/initial tensors, and arithmetic-before-random ordering. It exports
all crossings (including examples/epochs/shared-hardware seconds), missing values
for censored crossings, explicitly labelled observed-only medians, paired ratios,
full curves and compact run evidence. A retrospective ordering check found all
17 discrete random-label runs followed a successful gate for their exact model;
none of the failed arithmetic recipes entered calibration.


## Paper comparison requested during confirmation

The reported discrete-run “steps” are actual optimiser updates. At p149 and a
50% split, 11,026 training rows with batch512 give22 updates per complete epoch.
A_s43's sustained99% crossing at2900 updates is131.836 data passes (approximately
132 epochs), not2900 epochs. The paper uses first-passage epochs rather than the
new experiment's three consecutive observations at100-update intervals.

The published paper's Appendix C.2 includes p149 in the weight-decay sweep.
For its lambda1 baseline, `scripts/check_p149_paper.py` reads the original registry
without mutation and verifies all80 generalisation crossings against saved NPZ
traces. Settings: dropout0.2, depth2,heads1, lr0.001, batch512, fraction0.5,
seeds42–46, host147-224-250-73. Complete exports and figures are in
[reports/paper_p149](reports/paper_p149); every row includes its trace hash.

Mean first99% validation times (all5 seeds successful at these widths):
d40=309.4 epochs, d48=267.6, d64=107.4, d96=71.4, d128=57.0, d256=42.4.
Thus the original ordinary-transformer result does support faster generalisation
with increasing size. Width96 takes about1571 updates on average, compared with
our narrow-channel B/C models' much slower seed43 outcomes. The paper-defined
minimum delay (train99 to validation98, clipped at zero) is0 through width80,
1 epoch atwidth96,3 atwidth128 and7 atwidth256. Smaller failed runs stay in the
export: widths20/24 reached99% in0/5 seeds,28 in1/5 and32 in2/5.

This corrects any suggestion that the current result overturns the usual width
trend. B/C use a new14-coordinate bottleneck after every block, and the current
comparison uses dropout0. The new result concerns that channel architecture and
matched finite-budget capacity; it is not a pure width-scaling replication.
Reference: https://arxiv.org/pdf/2605.09724v2 (Appendices A.2,A.6,C.2).

## User correction: C must have no channel bottleneck

Before the remaining confirmation runs completed, the user clarified that C must
be an unrestricted continuous model. The original C was continuous but still
passed through14 coordinates after every block. It is retained as an auxiliary
quantisation-only control, not the intended ordinary-large-transformer baseline.
The user explicitly chose a plain width96 transformer (324,384 parameters),
accepting its1.69% smaller parameter count relative to B's329,952. No channel or
learned compression is added to this corrected C. A/B remain unchanged.

Register confirmation-p149-v2 with the same modular data, optimiser, dropout0,
seeds43/44/45 and60k update budgets. Completed A/B results are inherited verbatim
with file hashes; the active B_s44 is allowed to finish before the old queue stops
at its next C run. All other old C attempts remain in the record. Corrected C's
shared base parameter tensors initialise identically to B's under each paired
seed, but total parameter counts and whole-state hashes intentionally differ.
This is an explicit amendment after observing partial original results, not an
unchanged original registration. A/B calibration and model selection stay frozen.
Launcher: scripts/confirm_discrete_unrestricted.py.

Also register one descriptive memory check of unrestricted C: n16384,seed61,60k
updates, on the same random data and optimiser as A/B. This tests whether C can
store more than the modular dataset's79,811 bits. If it interpolates, report the
measurement as a lower bound; no saturated C capacity or universal constant is
inferred. Config: configs/discrete_capacity_p149_plain_c.yaml. The existing B14q64
arithmetic gate precedes this check. Original compressed-C capacity remains an
auxiliary measurement and must not be attributed to unrestricted C.

## Full-width arithmetic development follow-up

The first width96/K2 pilots have not passed by 24k updates. Quantized identity
STE shows severe clipping and large pre-quantizer activations (checkpoint14k:
85%/99.98% clipped in blocks1/2). Categorical K2 is learning but remains far below
the gate; its checkpoint16k mean maximum categorical probabilities are 0.85/0.87,
and mode evaluation is worse than sampled evaluation. These are optimization
diagnostics, not low-capacity evidence. The full registered pilot budgets remain.
Before launch, register an additional categorical K16 arithmetic pilot with
weight decay0.1 (other settings unchanged, seed42, 60k updates). Lower decay may
permit more confident categorical choices; this is a hypothesis to test, not a
claimed fix. If this optimizer is selected, A/B/C capacity and arithmetic runs
must all use the same decay0.1. No random labels are admitted by this amendment.

Register scalar-quantizer surrogate follow-up before launch: keep the same hard
forward grid and full residual width, but use the derivative of clipping (zero
outside [-1,1]) and identity through rounding. The v1 identity-through-clipping
variant remains available and its original runs are preserved. This addresses
the observed unbounded pre-quantizer activations without changing the state
architecture. Evaluate levels16,4,2 in that order, seed42 and unchanged optimizer.
The scientific source revision changes provenance for all newly launched runs;
old source-bound pilots continue as recorded, but any selected categorical
recipe must pass again under the revised source before new-source calibration.

Full-width scalar q16 with clipping-aware STE passed its p149 arithmetic gate:
first sustained held-out99% crossing at2,200 updates (=100 data passes), confirmed
at2,400; final train99.9637%, held-out99.4196%. An independent CPU reload reproduced
accuracy and verified every coordinate on the grid before any new random labels.
Register a q16 capacity screen at n=16,384/32,768, seed61, 60k updates each, plus
new-source A44 calibration at both sizes and seeds61/62. Both use identical
optimizer/data protocol and the existing 79,810.65-bit target. q16 is only a
learning-validated candidate, not a capacity-matched B yet. Stop original-source
pilot queues after their binary runs finish; unstarted follow-ups move to the
current source. No categorical recipe has passed the gate at this registration.

The q16 capacity screen already exceeds the target upper bound in a saved
checkpoint. Stop that candidate's screen and cancel its remaining cells: the
max-achieved memory statistic cannot decrease, so more training cannot make
this candidate match the fixed dataset. Preserve checkpoint, full trace, and
`screen_rejection.json`; do not call it a completed 60k run or a saturated capacity
estimate. This deterministic above-target rejection rule also applies to later
screens. A44 calibration continues. q4 passed arithmetic and is being audited;
q2 with the clipped surrogate is still arithmetic-only.

q4 passed its independent arithmetic audit: final held-out 99.083984%; sustained99% at update 11700. Register the same two-size, seed61 capacity screen as q16, with early rejection only when a saved checkpoint demonstrates memory above87,791.72 bits.

q4 screen rejected by the same pre-recorded rule: its saved checkpoint establishes memory above the fixed target upper bound. Check `fullwidth-q4-capacity-v2/B_q4_d96_n16384_s61/screen_rejection.json` for the exact witness. Both q16 and q4 have learned arithmetic and random labels, but neither is a capacity-matched B. Binary quantization and categorical learning remain open.

## Arithmetic tuning with frozen scalar source

The scalar experiment's scientific source is frozen at commit2d328cf in
`results/discrete/source-worktrees/fullwidth-scalar-v2`. All subsequent scalar
gates/calibration/confirmation use that source. This permits categorical-only
changes without altering the meaning or provenance of ongoing scalar runs.
The main worktree remains on idea/discrete.

Categorical temperature1 with decay1 remains below20% held-out at40k updates;
decay0.1 has nearly memorized training labels in mode evaluation but held-out
accuracy remains near chance. Neither passes the arithmetic gate. Register
K16 temperatures0.3/0.1 at width96 with the original decay1, otherwise unchanged,
seed42/60k updates. Sampling and the surrogate softmax use the SAME temperature;
forward values remain hard one-hot. Lower temperature tests whether categorical
noise is preventing arithmetic learning. This is an adaptation, not the exact
DreamerV2 unit-temperature configuration. Any capacity tests of this family use
the revised source and their own successful recipe gate.

Register binary scalar lr0.0003 with the frozen scalar source, otherwise unchanged
(clip1, clipped derivative, width96, decay1, seed42, 60k updates). The lr0.001 run
still jitters below99%; a smaller optimizer step tests whether stable arithmetic
learning is possible. If selected, A/B/C all use the same learning rate.

Temperature0.1 improves sampled categorical arithmetic learning substantially
in the first5k updates (held-out75.7%, mode82.5%, still below the gate). Register
a full-width58/K16/temperature0.1 arithmetic candidate, with the same optimizer
and seed42. It has340,750 parameters, 4.52 times A44; its ordinary count-matched
C is width98 with337,414 parameters (0.98% fewer). This is a uniformly width58
transformer with58 categorical variables per block, not a narrower learned
channel inside a width96 transformer. It needs its own arithmetic gate before
calibration. Selection remains based on measured capacity, not seed42 speed.

If the binary scalar lr0.001 pilot exhausts60k below the gate, extend that
completed run to120k updates without changing any optimizer or architecture
setting. It has approached99% and merits a bounded continuation. The parent
remains immutable; extension provenance records the parent checkpoint and exact
optimizer/RNG/batch-position continuation. No random labels before a passed gate.

Before any categorical random-label training, strengthen the measurement against
a noise-only explanation: categorical empirical capacity is the maximum learned
label information observed under sampled OR hard-argmax evaluation. Both use
finite one-hot states. Record both traces separately and evaluate late drift of
their per-observation maximum. Do not accept an80k sampled estimate if hard mode
already stores more than the dataset target tolerance. The arithmetic gate and
primary speed metrics continue to use sampled evaluation, with mode auxiliary.

Further arithmetic-only registration: temperature0.03, categorical K16, widths96
and58, original optimizer and60k updates. At temperature0.1 the wide model has
reached sampled95.6% and mode97.3% at20.2k, while the narrower candidate is still
learning. The lower temperature tests the remaining sampling error; it does not
change the alphabet, parameter count, or one-hot forward representation.

Also register a four-block scalar model, full width80 throughout, four levels
after every block, clipped surrogate, original optimizer, seed42/60k updates,
using frozen scalar source2d328cf. This tests the goal's suggestion of enough
discrete layers after two-block q4/q16 exceeded capacity. There is no internal
narrow projection. If it passes and is selected, its A and C controls will also
have four blocks, with A calibrated to the same fixed p149 dataset and C ordinary
width80 (434,480 parameters, exactly equal to B).

Binary scalar q2 passed after the registered horizon extension: sustained held-out
99% began at100,200 updates, confirmed at100,400; final train99.6282%, held-out
99.3561%. Independent CPU reload reproduced accuracy and verified binary values
at all96 coordinates in both blocks. Register n16,384/32,768 seed61 capacity
screening at120k updates (not just60k, since arithmetic required over100k). Early
above-target rejection remains permitted only with a saved checkpoint witness.
If q2 is near the target, extend all paired A/B calibration cells to the same
120k horizon before assessing a match. No mixed-horizon match is permitted.

Capacity audit: q16's reloaded weights independently store113,880.62 bits at10k.
q4's original GPU reproduces its recorded loss exactly; CPU loss differs by
1.72e-5 nats with identical accuracy, consistent with finite-grid sensitivity to
floating-point arithmetic. Capacity audits now require original-device loss
agreement within1e-5 AND CPU agreement within1e-4, with accuracy within1e-6.
This numerical tolerance represents under5 bits at the current dataset sizes,
far below the matching tolerance. Above-target rejection must hold for both
the CPU and original-device checkpoint measurements.

A selected120k calibration uses a120k fresh A/B/C arithmetic budget for all arms,
with the existing success stop and censoring rules. This avoids imposing a60k
comparison limit on a recipe whose arithmetic development required100k. All
capacity checkpoint audits are required before final full-width reporting; plots
use the actual registered horizons rather than a fixed60k axis.

The binary q2 width96/two-block model also exceeds the fixed target: its saved
checkpoint at28000 updates stores96432.71 bits. Stop its
screen by the registered above-target rule, preserve all artifacts, and cancel
remaining cells. Thus all tested width96/two-block scalar alphabets (2/4/16) can
learn arithmetic, but all store too many random labels to be the requested B.
The hypothesis has not been tested with a valid full-width A/B capacity match yet.
The four-block candidate remains arithmetic-only. Categorical temperature0.1
width96 exhausted60k below the sampled99% gate; no categorical random labels have
been trained. Temperature0.03 diagnostics reveal state collapse for width96 and
limited learning for width58; these are optimization failures, not capacity
measurements.

## Full-width architecture tuning, source v4

Both earlier scientific sources are frozen: scalar2d328cf and categorical31c5175
in source-worktrees. Ongoing old runs remain bound to those versions.

All tested two-block width96 scalar precisions exceed the target, so tune the
ordinary transformer's native residual width and feed-forward hidden width
independently. Register two-block (residual72, FF576) at2/4 levels and
(residual80, FF480) at2 levels, clip1/clipped surrogate, original optimizer,
seed42/120k arithmetic updates. Counts are312,408 and306,160, respectively,
versus A44's75,460 (ratios4.14 and4.06). These are full-width ordinary transformer
blocks with standard wider feed-forward networks: every residual coordinate
is quantized, and there is no learned narrow channel between blocks. Each C
uses exactly B's residual/FF widths, parameter shapes, and initial weights with
quantization disabled. A keeps its independently calibrated smaller architecture.
Default FF size remains4*residual_width, preserving existing model behavior.

Categorical diagnostics showed near-uniform first-layer states and collapsed
second-layer states in the sharp-temperature wide pilot. Register parameter-free
RMS normalization of the full block output before categorical logits; only the
resulting hard one-hots cross the boundary, with no norm/scale side channel.
Test width96/K16/temperature0.3 and width58/K16/temperature0.1, original optimizer,
seed42/120k updates. There are still96 and58 categorical variables respectively,
and parameter counts are unchanged by normalization. This tests optimization,
not a reduction in coordinate count. The 99% sampled arithmetic gate remains.

All new recipes need their own v4 arithmetic audit before random-label testing.
The fixed p149 split, target bits, ±10% matches, >=4 parameter ratio, multiple
calibration sizes/seeds, and fresh paired A/B/C comparison remain unchanged.
Gate matching now explicitly includes optimizer settings and the FF/normalization
settings as well as all discrete architecture settings.

The v4 scalar candidates (72,FF576,q4) and(80,FF480,q2) passed arithmetic at
10,600/11,500 terminal updates, with held-out99.2019%/99.6009%, respectively.
After independent audits, screen each at n16,384/32,768 seed61 with120k updates
and the existing above-target stopping rule. Recalibrate A44 under v4 at both
sizes and seeds61/62, also120k, so final comparisons cannot mix code or horizons.
Calibration cells may run concurrently; final paired modular comparisons remain
serial and primarily measured in optimizer updates, with shared-machine timing
limitations recorded. No capacity match is claimed before completed evidence.

Normalized categorical pilots learn much better: both exceeded96% held-out by
9k updates, but remain below the gate around21k. Register sharper normalized
pilots at(width96,temperature0.1) and(width58,temperature0.03), alphabet16,
original optimizer, seed42/120k. All states remain sampled hard one-hots; no
random-label tests are admitted by this arithmetic-only amendment.

The first v4 scalar screens also exceeded87,791.72 bits. Both were stopped by
the registered rejection rule; each exact witness is preserved separately as
`rejection_checkpoint.pt` with its SHA256, so a concurrent newer checkpoint
cannot erase the observed lower bound. Independent audits use that witness.
The user's preference is pending on two explicit protocol clarifications:
primary hard-mode versus sampled categorical evaluation, and whether the
agent-added >=4x parameter cutoff should remain or permit >=3x. Until answered,
existing acceptance criteria remain in force; no categorical random labels run.

The width80/FF480 binary capacity witness reproduces its original GPU loss and
accuracy exactly. CPU differs on one of16,384 examples: block2's pre-quantizer
coordinate is0 on CPU and3.576e-7 on GPU, straddling the binary rounding boundary.
The saved `cpu_gpu_diagnostic.json` identifies the checkpoint, example, and
coordinate. CPU loss differs by3.23e-5 nats (under1 bit of measured capacity).
For quantized checkpoints, the CPU audit now permits at most one prediction
of accuracy difference while retaining loss tolerance1e-4 and the strict
original-device checks. Both CPU and GPU must independently exceed the target
for rejection. This numerical tolerance amendment follows the observed
cross-device discrepancy; it changes no training, source model, or match bound.

## Explicit follow-up: deterministic categorical evaluation and 3x size ratio

The sampled-only arithmetic criterion and >=4x parameter threshold were agent
choices, not user requirements. After reporting the two alternatives and allowing
response time, proceed with hard argmax as the primary categorical inference
policy and >=3x parameters for the next comparison. This follows the user's
requested finite categorical activations and much-larger B. It is an explicit
post-development amendment, not a claim that old sampled gates passed. Original
sampled-primary results and >=4x assessments remain unchanged. Scalar inference
is still deterministic. Neither the fixed p149 dataset nor any capacity tolerance
changes. Training remains sampled, with the same Dreamer-style estimator.

Register source-v5 arithmetic pilots before launch: categorical width58/K16,
RMS normalization, temperature0.1, hard-mode primary; scalar width80/FF320,
binary clipped-STE. Original optimizer, seed42,120k budget, eval100. The latter
has229,360 parameters versus A44's75,460 (3.04x), with an exact ordinary C.
The categorical model has340,750 parameters, with ordinary C width98 (337,414).
Both need fresh source-v5 successful checkpoints and independent audits before
random labels. Old v4 sources are frozen at99f4637 for ongoing work.

Summary/gate/success stopping now explicitly use the registered evaluation
policy. Raw sampled and hard-mode traces remain distinct; both sets of crossing
metrics are exported. Success stopping requires the latest three observations
of both train and held-out accuracy to exceed99%, and a passed gate requires
its saved final checkpoint to retain held-out99%. Categorical capacity continues
to use max(sampled,hard-mode) information, so inference noise cannot create a
spurious capacity match. Independent audits reproduce sampled GPU results and
CPU hard-mode results, and apply99% to the registered primary policy.

Register 120k capacity calibration at sizes16,384/32,768 and seeds61/62 after
each source-v5 arithmetic gate. A44 is repeated under the same source. Keep
late drift<=5%, all A/B/dataset comparisons within10%, and a >=3x parameter
ratio. Exclude candidates early only with an audited saved above-target witness.
Freeze selected architectures before fresh paired arithmetic seeds43/44/45.

After launching the source-v5 pilots, stop the four still-live v4 categorical
pilots to concentrate compute on the new qualification policy. Preserve exact
checkpoint snapshots and all observations with scope_stop.json. Their original
120k budgets are censored by this scope change; no completed-failure conclusion
is drawn. The four v4 A capacity cells continue to their registered horizon.

Add a four-level arithmetic pilot at the same native width80/FF320 and all
other v5 settings before launch. It tests scalar precision without changing
parameter count. If both pass arithmetic and match capacity, select by capacity
agreement with the fixed dataset and A, before observing fresh-seed arithmetic
results. Binary width80 remains below60% held-out at13.5k; no failure is assumed.

Baseline source-equivalence audit: the v4/v5 model and data files are byte-identical.
After removing the new pure policy validation and the unreachable modular-only
stopping branch, the random-training program is AST-identical, including data,
evaluation, initialization, optimizer updates, batching and checkpoint state.
A real64-update CUDA replay under each version produced exactly equal model,
optimizer, batch order/cursor, CPU/CUDA RNG states, and all non-timing observations.
See reports/fullwidth_development/source_equivalence.json. Therefore reuse the
registered v4 A44 calibration rather than rerunning identical optimization.
Each A checkpoint remains audited under its original frozen source. Acceptance
permits only this exact pair of runner hashes with every model/data hash equal;
B calibration and fresh confirmation must use the same v5 source. The result
records that baseline source equivalence was used. This explicitly supersedes
the unnecessary v5 A rerun registration before any such rerun was launched.

The first categorical random-label screen exceeds the target under hard mode,
although sampled label information is lower. Preserve and independently audit
its exact above-target checkpoint and cancel remaining cells. This validates
categorical random-label learning but rejects width58/K16 as a capacity match.
Before launch, register a width48/K16/RMS/temperature0.1/hard-mode arithmetic
pilot with the same optimizer and120k budget. It has235,920 parameters,3.13x A44.
Also register binary width80/FF400 (267,760 parameters,3.55x A44): it interpolates
between the learning-successful but over-capacity FF480 and the slower-learning
FF320 candidate. These are architecture calibration attempts; fresh comparison
seeds43/44/45 remain untouched. Each recipe must pass its own arithmetic audit.

The categorical width58 capacity witness independently reproduces both original
GPU evaluations exactly. CPU hard mode differs on one of16,384 predictions due
to a near-tie between categories2 and13 in block2: competing-logit margins are
8.94e-8 on CPU and-2.98e-8 on GPU. The CPU memory estimate differs by0.223 bits.
Apply the same one-example cross-device accuracy tolerance to categorical hard
states as to scalar quantizers, retaining loss tolerance1e-4 and exact original
GPU agreement. The saved cpu_gpu_diagnostic.json identifies the exact cause;
both devices independently exceed the fixed target tolerance.

Register one alternative categorical alphabet before launch: native width62,
K8, RMS normalization, temperature0.1, sampled training/hard-mode primary,
unchanged optimizer and120k arithmetic budget. It has265,050 parameters (3.51x
A44), with nearest ordinary C width86 (263,074;0.75% fewer). This tests whether
more coordinates with fewer categories learn more reliably than width48/K16,
which remains below50% held-out mode accuracy at15.9k. Its total hard-state
alphabet budget is186 bits/token versus192 for width48/K16 and232 for width58/K16.
No capacity conclusion follows from those state sizes; this recipe still needs
its own arithmetic gate and independent audit before any random labels.

Complete the native width80/FF320 scalar precision pilot grid with8 and16
levels, original optimizer/seed42/120k budget, registered before launch. The
four-level run remains around98.4% held-out at65k; higher precision tests whether
its remaining arithmetic errors are caused by the coarse grid. Every candidate
has229,360 parameters and its own arithmetic gate before random labels. Larger
alphabets are not assumed to meet capacity; above-target screens remain rejected.

Complete a bounded categorical alphabet grid with(width72,K4) and(width84,K2),
RMS normalization,temperature0.1,mode-primary,sampled training,original optimizer,
seed42/120k arithmetic budget. Counts are270,936 and308,028 (3.59x and4.08x A44);
nearest ordinary controls are widths88 and94,within1.5% of B counts. Alongside
(width58,K16) and(width62,K8), this tests fewer categories with more native
coordinates, while retaining the entire discrete state at every block boundary.
The eight-category recipe has passed its fresh arithmetic gate at12,200 updates;
its capacity screen starts only after its independent audit. The two new recipes
are arithmetic-only until separately qualified. Selection remains based on
capacity calibration, never on fresh-seed confirmation speed.

Register explicit small random-label fitting checks after the respective
arithmetic audits: N2048,seed61,10k updates,eval100,original optimizer for
categorical width58/K16 (v5,mode-primary) and scalar width80/FF480/q2 (frozen v4).
These check near-complete memorization on a dataset below measured capacity,
separately from the larger calibration screens. They are learning-validation
runs, not capacity estimates or model-selection evidence. Export both categorical
inference policies; no assumption is made that sampled and hard-mode fits agree.

The source-v5 scalar width80/FF320/q16 arithmetic run reached sustained99% and
stopped at19,100 updates with99.69% held-out accuracy. After its independent
audit, run the registered120k, two-size/two-seed calibration grid, beginning
with n16,384 seed61. The same saved-checkpoint above-target rejection rule applies.
The q4 candidate's capacity config is prepared but remains prohibited until
its own arithmetic gate and independent audit pass.

Both small random-label validations completed10k updates and passed independent
checkpoint audits: scalar binary width80/FF480 fits100% of2048 random labels;
categorical width58/K16 fits100% in hard mode and99.46% under the recorded sampled
evaluation. These demonstrate learning, not saturated capacity. See
reports/fullwidth_development/random_learning.json.

The width80/FF320/q16 capacity screen is rejected by an independently audited
6k checkpoint storing106,059 bits, above the fixed tolerance. The q4 recipe has
now passed arithmetic at91,300 updates (99.08% held-out); its own audit precedes
its screen. The categorical width62/K8 two-size screen remains live and has not
yet established a completed capacity estimate or plateau.

The scalar width80/FF320/q4 screen exceeds the target in a saved checkpoint
and is stopped for independent audit; it cannot be selected. Categorical
width62/K8 is still within tolerance around40k, so start its already registered
second seed at both sizes. It still needs completed budgets and drift checks.
Categorical width72/K4 passed arithmetic at27,100 updates (hard-mode held-out
99.23%); after its independent audit, screen n16,384 seed61 before continuing
its registered two-seed/two-size calibration. No fresh confirmation seeds run
until a complete capacity match is accepted.

## Scalar normalization follow-up, source v6

Freeze v5 at d428975 for all existing categorical calibration, audits, and
confirmation. Source v6 changes only FullWidthQuantizer by adding rms_clipped:
normalize the entire block output by its per-token RMS, then use the existing
fixed grid, clipping derivative, and straight-through derivative through rounding.
The RMS scale is never passed across the boundary. There are no new parameters
or reduced-coordinate states. For binary codes, the mathematical forward sign
is unchanged by positive normalization; the surrogate gradient and clipping mask
change. Existing identity/clipped behavior and categorical classes are unchanged.

Register native width80/FF320 at2 and4 levels with rms_clipped, seed42,
original optimizer,120k arithmetic budget before launch. Both have229,360
parameters and exact ordinary continuous controls. Each needs its own arithmetic
gate/audit before any random labels. Keep old runs bound to their original source.
A baseline-source equivalence check is required before reusing its calibration
with this new scalar code; categorical v5 comparisons need no new source change.

The v5/v6 baseline-equivalence audit passed: all source files except the
FullWidthQuantizer class are unchanged, and baseline never instantiates that
class. A64-update baseline replay gives exact model/optimizer/RNG/batch state
and non-timing observations. Reuse A calibration only for this pinned file-hash
pair, combined with the earlier verified v4/v5 runner equivalence. No arithmetic
gate is transferred to rms_clipped; the new surrogate still requires its own
successful arithmetic checkpoint. Discrete calibration and confirmation retain
exact within-arm source matching. Evidence is saved in
reports/fullwidth_development/scalar_normalization_source_equivalence.json.

The width62/K8 categorical screen reaches87,283 bits by65.2k, still below the
87,791.72-bit ceiling but near its upper edge. Register a local native-width
refinement to60/K8, keeping RMS,temperature0.1,mode-primary,seed42,optimizer,
and120k arithmetic budget unchanged. It has248,820 parameters (3.30x A44), and
ordinary C width84 has251,580 (+1.11%). This moves capacity toward the center of
the fixed tolerance if the wider candidate ultimately exceeds it. It uses the
frozen v5 categorical source and needs its own arithmetic audit. No confirmation
speed is observed or used in this calibration choice.

The scalar width80/FF400/q2 pilot completed120k below the99% arithmetic gate
(final held-out98.64%); no random labels are permitted for that recipe. Stop the
binary categorical width84/K2 development pilot after remaining near chance
beyond60k to prioritize qualified capacity grids. Preserve its exact checkpoint
and trace; label its original120k budget censored by scope change, not completed.

Normalized binary width80/FF320 passed the v6 arithmetic gate at18,900 updates,
with99.20% held-out accuracy. Its unnormalized counterpart completed120k at57.06%
on the same development seed. This is development evidence, not the fresh-seed
A/B/C comparison. After its independent audit, run the registered120k capacity
grid at n16,384/32,768 and seeds61/62, beginning with seed61. The exact ordinary
C has229,360 parameters. A's reuse is covered by the pinned baseline-equivalence
audits; no calibration is inferred from the hard-state bit count.

Width62/K8 eventually exceeded the fixed target ceiling in a saved96k checkpoint:
88,381.03 bits versus87,791.72. Preserve that witness for audit and cancel its
other three calibration cells with explicit scope-stop records. Do not relax the
fixed tolerance or infer a plateau from partial cells.

Register one additional development initialization for width60/K8 before launch:
seed41 with data_seed42, preserving exactly the existing arithmetic train/test
split, optimizer and architecture. Seed42 remains in the record. A successful
seed41 run can establish arithmetic learnability, but is not confirmation speed
evidence; fresh paired seeds43/44/45 remain untouched and will reveal reliability.
This is a bounded initialization check after the width60 seed42 optimization
stall, not a change to the capacity criterion or permission for unqualified
random-label training.

The normalized four-level width80/FF320 model also passed arithmetic at37,200
updates (held-out99.12%). After independent audit, screen its registered120k
capacity grid, starting n16,384 seed61. Both normalized scalar candidates share
an exact ordinary continuous C (width80/FF320,229,360 parameters). Register that
C's descriptive random-label control now: n16,384,seed61,120k updates,identical
optimizer/data protocol. This control can serve either scalar candidate if a
capacity match passes; its results cannot alter B selection. A dataset-limited
C result is a lower bound, not a saturated capacity estimate.

Verified that every launched v5/v6 capacity cell and the scalar continuous
control uses exactly A's paired random dataset for its(seed,size), via saved
dataset hashes. For any calibration following the width60 seed41/data42 arithmetic
gate, do not inherit data_seed42 into random calibration: use data seeds61/62
paired with A. The arithmetic-only data override keeps the development split
fixed and is not part of the architecture. Pairing is required before acceptance.

The normalized four-level scalar screen is now rejected: its saved20k checkpoint
independently reproduces98,485.70 bits, above the87,791.72-bit ceiling. Preserve
rejection_checkpoint.pt and its SHA; the remaining three registered cells are
cancelled. This is a demonstrated above-target lower bound, not a completed
120k calibration. Normalized binary calibration and the common continuous C
control continue. The width62/K8 categorical96k rejection audit also passed.

The width72/K4 categorical calibration is rejected by an independently audited
90k checkpoint storing88,283.57 bits. Preserve that witness and cancel the
other three cells with exact checkpoint snapshots. Register a local refinement
to width70/K4 before launch, keeping the native feed-forward width at288,
temperature0.1, RMS normalization, sampled training, hard-mode primary
evaluation, seed42,120k arithmetic budget and original optimizer unchanged.
It has260,050 parameters (3.45x A44); ordinary C is width94,FF288
with261,978 parameters (+0.74%). The width70/default-FF280 preflight
could not meet the existing1.7% ordinary-C count tolerance and did not start
training; preserving FF288 meets that tolerance. Run the registered recipe
under frozen v5 source. It must pass its own arithmetic gate and independent
checkpoint audit before any random labels. If qualified, use the same120k,
two-size(n16384/32768),two-seed(61/62) capacity grid and unchanged acceptance
rules. No fresh confirmation seeds inform this development choice.

Register an additional small random-label learning validation for the qualified
normalized binary width80/FF320 recipe: n2048,seed61,10k updates,unchanged
optimizer and v6 source. Its own successful arithmetic audit already precedes
this run. This checks complete memorization below the measured capacity and
is not a capacity estimate or model-selection criterion. Preserve the earlier
un-normalized binary learning check as separate evidence.

Final C configuration validation ignores only the descriptive purpose field
(capacity_control or absent); all scientific configuration fields remain exact.
This fixes the independently registered control label without changing any
training run, source binding, or parameter/data pairing.

The normalized binary learning validation completed10k updates and passed its
independent checkpoint audit: final random-label accuracy99.7559% (2043/2048),
with100% observed earlier. The final checkpoint result, not the best trace
accuracy, is the audited outcome. This is learning evidence rather than a
capacity estimate.

## Confirmation-seed correction before selection

An audit of the actual manifests found that43/44/45 were used by the superseded
14-coordinate confirmations. Earlier statements calling these seeds untouched
were incorrect. Register73/74/75 for BOTH full-width families before either
capacity selection or new confirmation. The manifest audit confirms no existing
discrete-experiment run uses these seeds; retain the audit and registration in
reports/fullwidth_development/confirmation_seed_registration.json. Use rotation
ABC/BCA/CAB, unchanged paired datasets/batches, and all nine runs. No result
from these seeds has informed development or capacity selection. Keep legacy
43/44/45 artifacts unchanged. All prior plans to use43/44/45 for full-width
confirmation are superseded by this correction.

New full-width confirmation commands must pass --confirmation-seeds 73 74 75. The default43/44/45 remains only for
legacy compatibility. Pairing validation and plots derive seed indices from
the frozen configuration. Nineteen focused confirmation/report/matching tests
pass, and synthetic plotting with73/74/75 renders without indexing errors.

Register a second local categorical capacity refinement while the width70/K4
arithmetic run completes: retain the already successful width62/K8 residual
state, but reduce the native FF width from248 to232. The wider-FF recipe has
an audited88,381-bit capacity witness, just outside tolerance; reducing native
FF parameters may move the estimate inside the fixed band without changing
categorical width. Width60/K8 seed42 completed120k below the arithmetic gate,
and its second initialization also remains below the gate late in training.
The new width62/FF232/K8 recipe has259,098 parameters (3.43x A44);
ordinary C is width102/FF232 with256,530 parameters (-0.99%).
Keep seed42,120k arithmetic budget,RMS,temperature0.1,sampled training,
hard-mode primary evaluation,original optimizer and frozen v5 source.
It requires its own arithmetic gate and audit. If qualified, use the registered
120k capacity grid at n16384/32768,seeds61/62. Capacity matching remains the
selection criterion; no confirmation result informs this change.

## Accepted scalar match and fresh confirmation

The normalized binary width80/FF320 model completed all four120k capacity
cells and passed independent final-checkpoint audits. Per-seed empirical
capacities are79,986.88 and79,816.66 bits versus A44's80,552.69 and80,438.20;
the fixed target is79,810.65. All four B late drifts are1.10--1.26%, below5%.
The assessment accepts both dataset matching and A/B matching. A has75,460
parameters; B and its unrestricted ordinary C each have229,360 (B/A3.0395).
These are finite-budget empirical estimates, not representational upper bounds.

Freeze this scalar selection before all new arithmetic results. Run seeds
73/74/75 in the registered ABC/BCA/CAB order, original optimizer and p149/50%
task,120k budget, and identical B/C initial tensors. Wait for the live scalar
C capacity control to release cuda:0, then run the nine confirmation cells
sequentially. Each completed arithmetic checkpoint will be independently
audited, and failed/unobserved crossings remain censored. The categorical
family is still developing; no categorical capacity match is claimed.

Width62/FF232/K8 completed arithmetic at18,200 updates and passed independent
audit before its first random-label screen. The audited hard-mode held-out
accuracy is99.00236%; recorded sampled accuracy is96.50826%. Start the registered
n16384/seed61 capacity screen under frozen v5 source. Preserve both inference
policies, and use max(sampled,hard-mode) for capacity. The other three cells
remain registered, pending the first screen.

The width70/FF288/K4 arithmetic run completed120k without the sustained99%
gate (final hard-mode held-out98.2677%); no random labels are allowed for it.
The qualified width62/FF232/K8 screen is about80,920 bits at15.4k, so launch
its remaining three already registered capacity cells. All four use120k,
paired data seeds61/62, and n16384/32768. Above-target saved-checkpoint
rejection remains in force across the grid.

Register the width62/FF232/K8 recipe's own small random-label learning check:
n2048,seed61,10k updates,unchanged optimizer,v5 source,after its independent
arithmetic audit. It is learning validation, not capacity selection evidence.
Also register ordinary C width102/FF232 (256,530 parameters) at n16384,seed61,
120k updates,as a descriptive capacity control. Queue this C on cuda:0 only
after the scalar confirmation and its report worker finish; it must not compete
with those timed arithmetic runs. Its capacity can be a dataset-limited lower
bound and cannot influence B selection.

The width62/FF232/K8 small random-label check completed10k updates and passed
its independent checkpoint audit:100% hard-mode accuracy and99.609375% sampled
accuracy on2048 labels. This establishes learning on a small random set;
capacity matching still requires the separate complete four-cell grid.

The scalar seed73 comparison is complete and independently audited: sustained
99% held-out accuracy first occurs at3,000 actual optimizer updates for A,
19,700 for B, and2,200 for C. This seed does not support a speedup from the
capacity-matched discrete model. The other two registered seeds continue with
no changes to the frozen selection. Report95% and99% thresholds and observed
train>=99%/held-out<90% intervals separately; intermittent99% crossings near
98--99% generalization must not be presented as a classical memorization plateau.

The width62/FF232/K8 grid completed all four120k budgets. Both n16384 traces
record maxima above the fixed87,791.72-bit ceiling:88,205.31(seed61) and
88,119.97(seed62). The final saved checkpoints are below the ceiling; historical
peak weights were overwritten by later checkpoints. Therefore record this as
a failed completed maximum-over-trace capacity assessment, not an independently
reloaded above-target checkpoint witness. Independently audit all final weights
and preserve all traces. Cancel the unstarted ordinary-C waiter with its process
identity and explicit training_started=false record; no C training was run.

Register a follow-up before launch: categorical width62, native FF160, K8,
RMS,temperature0.1,sampled training,hard-mode primary evaluation,seed42,
original optimizer,120k arithmetic budget,frozen v5 source. This preserves all
62 discrete variables and changes only the native FF expansion;160 remains
larger than the residual width. A larger FF reduction gives more capacity
margin than another small move near the upper tolerance edge. It has232,314
parameters (3.0786x A44); ordinary C is width108/FF160 with230,148
parameters (-0.932%). The arithmetic gate and independent audit remain
mandatory before random labels. If qualified, use the same120k,two-size,
two-seed capacity grid and unchanged10% matching rules. The scalar confirmation
continues under its frozen selection; none of its results changes this target.

## Completed full-width scalar comparison

The normalized binary model does not provide the proposed speedup under this recipe. All nine registered runs on seeds 73/74/75 are complete and independently audited. A and B match the fixed dataset capacity within 1%, while B has 3.0395x A's parameters. B is slower than both A and C on every seed at the 95% threshold.

All times below are actual optimizer updates. A crossing requires three consecutive evaluations, spaced 100 updates apart. Training uses p=149 division, 50% split, AdamW lr=0.001, betas=(0.9,0.98), weight decay=1, dropout=0, batch=512, and a 120k-update ceiling.

| Seed | A to 95% | B to 95% | C to 95% | A to 99% | B to 99% | C to 99% |
|---|---:|---:|---:|---:|---:|---:|
| 73 | 2,900 | 19,300 | 1,800 | 3,000 | 19,700 | 2,200 |
| 74 | 2,700 | 5,800 | 1,900 | 2,900 | Not reached by 120,000 | 2,100 |
| 75 | 2,800 | 98,400 | 1,700 | 4,100 | 101,800 | 1,800 |

| Model | Parameters | Median updates to 95% | Median epochs to 95% | Median seconds to 95% | Seeds reaching 99% |
|---|---:|---:|---:|---:|---:|
| A | 75,460 | 2,800 | 127.28 | 13.15 | 3/3 |
| B | 229,360 | 19,300 | 877.28 | 110.48 | 2/3 |
| C | 229,360 | 1,800 | 81.84 | 8.31 | 3/3 |

The median per-run maximum train-minus-held-out gap is A:24.02, B:17.39, C:50.81 percentage points. B reduces this transient gap relative to C, but loses the generalization speed advantage. None of the nine runs has a recorded observation with train>=99% and held-out<90%; the intended strong contrast between C grokking and B avoiding grokking therefore did not appear under this criterion. B's long low-accuracy plateaus precede simultaneous improvements in training and held-out accuracy, rather than near-perfect memorization.

B seed74 ended at 98.957% held-out accuracy. Its 99% crossing remains unobserved; the budget is not substituted as a crossing time. No aggregate 99% median across all three B seeds is claimed.

C reaches an observed peak of 117,696.50 bits in the descriptive random-label control; its independently reloaded final checkpoint stores 116,994.05 bits. That run is limited by its dataset, so these are lower bounds, not saturated C capacity estimates. The peak weights were not retained separately; the final-checkpoint number is the independently reproduced value. A/B matching uses the registered random-label protocol and 120k budget; it does not prove an intrinsic information bound or guarantee absent grokking. Wall times were recorded sequentially on a shared GPU and are descriptive.

Full evidence: [confirmation table](reports/fullwidth_quantized_v6/confirmation.csv), [learning curves](reports/fullwidth_quantized_v6/learning_curves.png), [capacity curves](reports/fullwidth_quantized_v6/capacity_curves.png), [validation](reports/fullwidth_quantized_v6/validation.json). The categorical comparison is reported separately below.


Register an intermediate categorical candidate before launch: native width62,
FF216, K8, RMS normalization, temperature0.1, sampled training and hard-mode
primary evaluation. Retain seed42, the original optimizer, frozen v5 source,
and the120k arithmetic budget. FF232 learned arithmetic but its completed
capacity estimates narrowly exceeded the ceiling. FF160 remains below the
arithmetic gate near95k, suggesting the larger FF reduction may impair
optimization. The intermediate candidate has253,146 parameters (3.3547x A44);
ordinary C width104/FF216 has253,240 (+0.0371%). This changes only the native
FF expansion and preserves all62 categorical variables. Its own arithmetic
gate and independent checkpoint audit must precede random labels. If qualified,
use the unchanged120k capacity grid at n16384/32768,seeds61/62, paired with A.
The FF160 run will finish its registered budget; its prepared random runs
remain prohibited until it qualifies. No confirmation result determines this
capacity refinement, and no matching threshold is changed.

Register FF216's own n2048,seed61,10k random-label learning validation,
conditional on its arithmetic gate and independent audit. Its purpose is
learning_validation, so it cannot enter the capacity estimate or selection.
Also prepare ordinary C width104/FF216 at n16384,seed61,120k updates as a
descriptive control. Launch that C only if B's full capacity grid is accepted.
The small validation and C control use the unchanged original optimizer.

The FF160 arithmetic run completed120k and passed an independent audit of its
failed result: hard-mode held-out75.05895%, sampled65.48159%. Its prepared
capacity runs remain unstarted. FF216 passed the arithmetic gate at14,900
updates and independent CPU/original-GPU checkpoint audit before any random
training: hard-mode held-out99.24724%, sampled98.08635%. Its small random-label
validation completed10k updates and independently reproduces100% hard-mode
accuracy and99.4140625% sampled accuracy. This is learning evidence, not a
capacity estimate. After its first n16384/seed61 screen reached the target
band, start the remaining three registered capacity cells. All four use
paired A datasets and the unchanged120k budgets and matching criteria.


## Supplementary delay measure aligned with the paper

The strict train>=99%/held-out<90% interval is not the paper's definition of
generalization delay. Add a separate descriptive measure, computed from the
unchanged saved traces: max(0, first held-out>=98% time minus first train>=99%
time), as in [the paper, section4.3](https://arxiv.org/html/2605.09724v2#S4.SS3).
This supplement is introduced after scalar confirmation and before categorical
confirmation. It changes no capacity selection, stopping rule, or registered
sustained95/99 outcome. The original signed sustained99 delay remains available.

| Scalar seed | A delay (updates) | B delay (updates) | C delay (updates) |
|---|---:|---:|---:|
| 73 | 0 | 0 | 100 |
| 74 | 0 | 0 | 100 |
| 75 | 0 | 0 | 100 |

Thus C has a small positive measured delay under the paper's thresholds, while
A and B have zero measured delay. The desired contrast appears in this limited
sense; B still fails to provide the proposed generalization speed advantage.
The100-update delay is about4.5 epochs and equals one evaluation interval.
Sub-100-update delays are unresolved, so zero measured delay does not prove
simultaneous crossings. The paper evaluated every epoch and used dropout0.2
in grokking runs; these registered comparisons use dropout0. Five reporting
tests pass, including censoring and separation of first and sustained crossings.
The regenerated scalar report validates all nine checkpoints and all exported
artifact hashes. Both delay definitions will also be reported for categorical B.


## Accepted categorical match and registered confirmation

Width62/FF216/K8 completed all four120k capacity cells, and all final checkpoints
passed independent audits. Per-seed maximum-over-trace estimates are87,313.55
and87,428.37 bits, versus A44's80,552.69 and80,438.20. B is9.40% and9.54%
above the fixed79,810.65-bit target and8.39% and8.69% above A, within both
registered10% tolerances. Late drift is0.77%,0.74%,1.23%,0.91% across the four
cells. The n32768 maxima are69,821.10 and69,589.67 bits; selection uses the
maximum over sizes per seed, consistently with A and the scalar comparison.
All saved final checkpoint traces exactly equal their exported traces, and
checkpoint hashes, source hashes, configs, datasets and120k budgets verify.

Freeze this accepted selection before categorical confirmation. Run the
registered73/74/75 seeds in ABC/BCA/CAB order on cuda:1, with the original
optimizer,p149 division,50% split and120k ceiling. B has253,146 parameters
(3.3547x A44); ordinary unrestricted C width104/FF216 has253,240 (+0.0371%).
C and B have different tensor shapes, so no identical-initial-tensor claim
applies to this categorical comparison. Dataset and minibatch orders remain
paired. Run the registered descriptive C capacity control on cuda:0. Preserve
failed/censored confirmation outcomes and report both registered sustained
95/99 results and the supplementary paper-style99/98 first-passage delay.
No confirmation outcome can change this frozen candidate.

The categorical selection is now frozen in
results/discrete/fullwidth-confirmation-v5-categorical-ff216/selection.json.
At registration, the nine-run confirmation and independent audit/report worker
were launched, with the descriptive ordinary-C capacity control on the other
GPU. The frozen configuration was retained through all nine completed runs.

The categorical comparison's ordinary-C capacity control completed120k updates
and passed independent CPU/original-GPU checkpoint audit. Its observed peak is
117,657.87 bits; the independently reloaded final checkpoint reproduces
117,581.61 bits and99.993896% accuracy (16,383/16,384 random labels). The final
saved trace exactly matches the exported trace. This establishes a descriptive
lower bound above matched A/B; the control is dataset limited and does not
estimate a saturated C capacity. The peak weights were not retained separately.
All nine registered arithmetic confirmations are now complete without configuration changes.


## First audited categorical confirmation seed

Seed73 is complete and all three checkpoints passed independent audit. These are actual optimizer updates to three consecutive qualifying evaluations; the other two registered seeds remain pending.

| Arm | Sustained95% | Sustained99% | Final primary held-out |
|---|---:|---:|---:|
| A | 2,900 | 3,000 | 99.6191% |
| B | 30,600 | Not reached by120,000 | 95.1206% |
| C | 1,400 | 1,900 | 100.0000% |

B ended at96.0729% hard-mode training accuracy and95.1206% hard-mode held-out accuracy; sampled held-out accuracy is90.9759%. It never reaches either first train99 or first held-out98 in this trace, so the supplementary paper-style delay remains unobserved. This is not evidence of successful grokking avoidance. A has zero measured paper-style delay and C has100 updates. The frozen B and remaining seeds are unchanged.


## Completed full-width categorical comparison

All nine registered categorical-family runs and their independent audits are
complete. B uses 62 categorical variables per token at each block boundary,
K=8, native FF 216, RMS normalization and temperature 0.1. Training samples
hard one-hot codes with the DreamerV2 softmax STE. Primary evaluation is hard
argmax; sampled evaluation is recorded separately. RMS, temperature and the
primary inference rule are adaptations, not claims about DreamerV2 defaults.

All crossings below require three consecutive evaluations at 100-update
spacing. Values are actual optimizer updates, not epochs. All three B runs
reached the frozen 120,000-update budget without sustained 99% held-out accuracy.

| Seed | A to 95% | B to 95% | C to 95% | A to 99% | B to 99% | C to 99% |
|---|---:|---:|---:|---:|---:|---:|
| 73 | 2,900 | 30,600 | 1,400 | 3,000 | Not reached | 1,900 |
| 74 | 2,700 | Not reached | 1,200 | 2,900 | Not reached | 1,400 |
| 75 | 2,800 | 5,500 | 1,300 | 4,100 | Not reached | 1,600 |

| B seed | Final hard-mode train | Final hard-mode held-out | Final sampled held-out |
|---|---:|---:|---:|
| 73 | 96.0729% | 95.1206% | 90.9759% |
| 74 | 52.2220% | 48.5126% | 48.1952% |
| 75 | 98.6668% | 98.1952% | 96.7441% |

The median per-run maximum train-minus-held-out gap is A 24.02, B 11.54,
C 49.79 percentage points. Every run has zero recorded observations satisfying
train>=99% and held-out<90%. C's paper-style delay is 100 updates on all seeds;
B's is missing/missing/0. Seed 75 first reaches held-out 98 at 8,600 and train 99
at 9,900, so its positive-part delay is 0, but it still never reaches sustained
held-out 99. Seed 74's joint learning failure is not a successful capacity limit.

Categorical B is 3.3547 times A's parameter count. Its measured capacity is
9.40%/9.54% above the dataset target and 8.39%/8.69% above paired A estimates,
within the registered 10% tolerance. The ordinary C control has 253,240
parameters versus B's 253,146. C's random-label run is dataset limited: the
observed peak 117,657.87 bits is a lower bound; the independently reloaded final
checkpoint reproduces 117,581.61 bits. Peak weights were not retained separately.

Together with the completed scalar comparison, this is a negative result for
the proposed speed advantage under the selected recipes. Successful arithmetic
and random-label development validation established that both discrete forms
can learn; confirmation exposed slower learning and, especially for categorical
B, poor robustness across seeds. No configuration was changed using these
confirmation outcomes.

Evidence: [table](reports/fullwidth_categorical_v5_ff216/confirmation.csv),
[learning curves](reports/fullwidth_categorical_v5_ff216/learning_curves.png),
[capacity curves](reports/fullwidth_categorical_v5_ff216/capacity_curves.png),
[validation](reports/fullwidth_categorical_v5_ff216/validation.json).


## Rechecking the final evidence

The original checkpoints remain under `results/discrete/`. Exported reports
contain the frozen configurations, traces, independent audits, and artifact
hashes. Report generation first validates all nine completed runs and refuses
an unfinished queue. Both queues are complete. Regenerate and validate the
two reports from this repository root with:

```bash
PYTHONPATH=results/discrete/source-worktrees/fullwidth-v6/src:. python scripts/report_discrete.py \
  results/discrete/fullwidth-confirmation-v6-quantized-rms \
  --output reports/fullwidth_quantized_v6 \
  --c-runs results/discrete/fullwidth-control-capacity-v6/C_baseline_d80_ff320_n16384_s61

PYTHONPATH=results/discrete/source-worktrees/fullwidth-v5/src:. python scripts/report_discrete.py \
  results/discrete/fullwidth-confirmation-v5-categorical-ff216 \
  --output reports/fullwidth_categorical_v5_ff216 \
  --c-runs results/discrete/fullwidth-control-capacity-v5/C_baseline_d104_ff216_n16384_s61
```

The categorical worktree is pinned to
`d428975b217ecc566e82b45a4eaeeec36d1f1706`; it already exists on this host.
On a new checkout, recreate that detached source worktree at the same relative
path before using the categorical command. The original run/checkpoint
directories are also required for independent revalidation; cloning the Git
repository alone supplies the exported evidence, not the ignored checkpoints.
The scalar source is frozen at commit `4b29d26` in `results/discrete/source-worktrees/fullwidth-v6`; recreate that detached worktree on a new checkout. Per-run source hashes and snapshots,
rather than the launch commit alone, bind the actual training implementation.

A cross-family verification confirms identical A configurations, dataset hashes,
initial weights and every non-timing trace value for each of73/74/75. All six
arms across the two families share the same modular dataset for each seed.
These are the same three A seeds used in two comparisons, not six independent
A seeds; wall times remain separate measurements. Evidence:
reports/fullwidth_development/shared_a_confirmation.json.


## Pre-softmax categorical normalization follow-up — 24 October 2026

The user requested the score normalization from language-regularisation.
Its `experiments/object_selection/model.py:ScaledSlotLayerNorm` normalizes
across a slot's alphabet using LayerNorm with epsilon 1e-4 and no affine
parameters. This differs from our existing RMS normalization of the residual
before the logits projection. Source v7 adds `slot_layernorm` and
`rms_slot_layernorm` to the existing categorical normalization option. The
latter retains residual RMS and then normalizes each K-logit vector, before
temperature scaling and both categorical sampling and the softmax STE.
No continuous state bypasses the discrete boundary and no parameters are added.

Register a development arithmetic run using width 62 / FF216 / K8, seed 42,
p149 division, 50% training split, the existing optimizer, temperature 0.1,
hard-mode primary evaluation and a 120k-update ceiling. Only score
normalization changes from the previous seed-42 recipe. Temperature remains
0.1 for this controlled test; the reference's temperature-1, deterministic
training recipe is not copied. Score normalization controls raw logit scale,
but does not guarantee non-saturated probabilities or useful gradients at this
low temperature. Configuration: `configs/discrete_fullwidth_v7_categorical_slotnorm.yaml`.
This is development validation, not a fresh confirmation or a new capacity
match. Any random-label run for this variant requires its own successful
arithmetic gate and independent audit first. Preserve all previous results.


## Exact binary scalar A/B/C speed and grokking comparison

These are the completed p149 division, 50%-split, seeds 73/74/75 results.
A is the 75,460-parameter ordinary model; B and ordinary C each have 229,360
parameters and identical initial tensors. B quantizes every residual coordinate
to two fixed values after RMS normalization at each block boundary.

| Seed | A updates to 95% | B updates to 95% | C updates to 95% | A updates to 99% | B updates to 99% | C updates to 99% |
|---|---:|---:|---:|---:|---:|---:|
| 73 | 2,900 | 19,300 | 1,800 | 3,000 | 19,700 | 2,200 |
| 74 | 2,700 | 5,800 | 1,900 | 2,900 | Not reached by 120,000 | 2,100 |
| 75 | 2,800 | 98,400 | 1,700 | 4,100 | 101,800 | 1,800 |

A threshold time is the first observation in three consecutive qualifying
held-out evaluations, sampled every 100 optimizer updates. These are actual
updates, not epochs. C is faster than A, and A faster than B, on every seed at
95%. Median 95% times are A 2,800, B 19,300 and C 1,800 updates: ratios of these
medians make B 6.892857 times A and 10.722222 times C. The paired B/A ratios
are 6.655172, 2.148148 and 35.142857; paired B/C ratios are 10.722222,
3.052632 and 57.882353. B's missing 99% crossing is censored and is not replaced
with the budget or included in an unconditional three-seed median.

For grokking, use the paper's delay: max(0, first held-out 98% update minus
first training 99% update). These are first observations, not sustained
thresholds. The complete underlying times are:

| Model | Seed 73: train99 / held-out98 / delay | Seed 74: train99 / held-out98 / delay | Seed 75: train99 / held-out98 / delay | Interpretation |
|---|---:|---:|---:|---|
| A | 2,900 / 2,900 / 0 | 2,800 / 2,800 / 0 | 3,200 / 3,200 / 0 | No measured grokking delay |
| B | 19,300 / 19,300 / 0 | 6,900 / 6,400 / 0 | 99,000 / 99,000 / 0 | No measured grokking delay |
| C | 1,700 / 1,800 / 100 | 1,900 / 2,000 / 100 | 1,600 / 1,700 / 100 | Short positive delay on every seed |

C has a positive delay under this operational definition, but it is only one
100-update evaluation interval (about 4.5 epochs), not a long memorization
plateau. All nine runs have zero observed points with training >=99% and
held-out <90%. B's late transitions occur after long periods of underlearning
on both splits. Slower generalization alone is not grokking. The separate,
more threshold-sensitive sustained99-to-sustained99 delays are A 100/100/600,
B 400/missing/1,500 and C 500/200/200 updates; they are not the paper's definition.
The comparison uses dropout 0 versus the paper's central setting of 0.2.

Sources: [complete measurements](reports/fullwidth_quantized_v6/confirmation.csv),
[paper section 4.3](https://arxiv.org/html/2605.09724v2#S4.SS3).


At the normalized temperature-0.1 pilot's 20,000-update checkpoint, a hard-mode
probe of the first 512 held-out examples found mean maximum categorical
probabilities 0.999074/0.935338 in layers 1/2. Fractions above 99% confidence
were 0.988077/0.618574; entropy was 0.005665/0.232319 bits. Normalized score
magnitudes stayed below 2.623, so normalization controls scale but the retained
low temperature still permits sharp softmax saturation. This diagnostic does
not prove that saturation is the sole cause of poor learning.

Before launching another run, register a temperature-1 counterpart in
`configs/discrete_fullwidth_v7_categorical_slotnorm_t1.yaml`. It shares seed 42,
initial tensors, data, optimizer, architecture and budget with the normalization-
only run, and retains Dreamer-style categorical sampling. Temperature 1 matches
the reference's softmax scale; it does not copy its deterministic/no-noise
forward estimator. Both pilot outcomes will be retained. Neither the old
capacity match nor the old arithmetic gate applies to either new recipe.


## Completed normalization follow-up

The per-slot pre-softmax LayerNorm implementation passed 32 focused tests.
All three comparison checkpoints (the earlier RMS-only seed 42 and both new
normalized pilots) passed independent audits. The two new runs completed all
120,000 updates. Both failed sustained 95% and 99% held-out thresholds, so no
random-label training or new capacity matching was performed for them.

| Seed42 recipe | Final updates | Hard-mode train | Hard-mode held-out | Sampled held-out | Arithmetic gate |
|---|---:|---:|---:|---:|---|
| Original residual RMS, temperature 0.1 | 14,900 | 99.4014% | 99.2472% | 98.0863% | Passed |
| Residual RMS + logit LayerNorm, temperature 0.1 | 120,000 | 33.0673% | 19.3361% | 19.2726% | Failed |
| Residual RMS + logit LayerNorm, temperature 1.0 | 120,000 | 89.6064% | 88.7539% | 80.5913% | Failed |

The new pilots' best recorded hard-mode held-out accuracies were 39.0259% and
92.2184%; neither reached 95% even transiently. The original run's sustained
99% crossing starts at 14,700 and is confirmed by its final 14,900 evaluation.
All three have 253,146 parameters, identical initial tensors and identical
data. Only categorical normalization differs between the original and the
new temperature-0.1 run. Temperature is the only configuration difference
between the two new runs. They retain sampled training and hard-mode primary
evaluation. The temperature-1 variant was registered after the intermediate
temperature-0.1 saturation diagnostic; these are development trials, not three
fresh confirmation seeds or a general conclusion about logit normalization.

The final checkpoint diagnostics retain the first 512 held-out examples and
all token positions. With temperature 0.1, 98.17%/57.35% of categorical slots
in layers 1/2 exceed 99% confidence. At temperature 1, none do, and mean maximum
probabilities are 0.6765/0.4864. Thus the higher temperature removes the observed
softmax saturation, but successful arithmetic learning still did not follow
within the declared budget. The intermediate 20k checkpoint used for the earlier
diagnostic was not separately retained; final diagnostics are bound to saved
checkpoint hashes and are reproducible.

The implementation remains available through
`categorical_normalization: rms_slot_layernorm`, or `slot_layernorm` without
residual RMS. The old `none` and `rms` behavior remains available for reproducing
previous runs. Full-width hard states and the softmax surrogate through the
normalization are preserved; no learned affine parameters are introduced.

Evidence: [comparison](reports/categorical_slot_normalization_v7/comparison.csv),
[curves](reports/categorical_slot_normalization_v7/learning_curves.png),
[provenance and audits](reports/categorical_slot_normalization_v7/comparison.json),
[final diagnostics](reports/categorical_slot_normalization_v7/normalization_diagnostics.json).
All 17 exported artifacts were checksum-verified. The completed binary report
also passed source-bound revalidation using its new frozen v6 worktree; its
training results are unchanged.


## Registered p=97 binary scalar replication (2026-10-24)

The user requested the same experiment at p=97 with architecture sizes matched
to the dataset. Retain modular division and the fixed 50% split: 4,656 training
examples and 30,866.284423 bits under the existing N*log2(p+2) convention.
Repeat the selected full-width binary RMS/clipped-STE family. A is ordinary and
small; B is fully binary at every block boundary; C is ordinary continuous with
exactly B's parameter tensors and paired initialization. The old p149 capacity
match does not transfer to p97.

Before any new training, register initial candidates A width28/FF112 (30,772
parameters), B/C width52/FF208 (97,084 each), depth2/head1, dropout0. All other
training settings match the p149 experiment: AdamW lr0.001, betas(0.9,0.98),
weight decay1, batch512; arithmetic max120k updates, evaluation every100,
three consecutive observations for threshold success. Development seed42 is
kept separate from registered confirmation seeds73/74/75. The B seed42
arithmetic run must pass its own independent checkpoint audit before any
random-label task. First validate random-label learning on 2,048 examples for
10k updates; do not interpret inability to learn as a capacity reduction.

Capacity calibration keeps the same seeds61/62, random dataset sizes16384/32768,
120k updates per cell, evaluations every200, peak-over-trace estimate, late
drift <=5%, and rejection of dataset-limited estimates. Each A/B estimate must
be within10% of the p97 target and each other, with B/A parameters >=3. The
initial widths are scaling-based candidates, not claimed capacity matches.
Tune only sizes using development evidence; freeze the accepted architectures
before nine paired confirmation runs in ABC/BCA/CAB order. Record failed
candidates and missing thresholds. Report sustained95/99 speeds in updates,
epochs and descriptive shared-GPU seconds, plus the same first-train99 / first-
heldout98 grokking delay. No universal no-grokking guarantee is assumed.

Configuration: `configs/discrete_p97_v1_learning.yaml`.

Before any p97 random-label training, amend the calibration dataset sizes to
8,192 and 16,384 (seeds61/62 unchanged), scaling the old powers-of-two grid down
for the smaller information target. The smaller calibration set still contains
1.76 times the modular training examples, versus 1.49 times for p149; keeping
16,384/32,768 would instead change the overload to3.52/7.04 times. The target,
10% acceptance, 120k horizon, peak estimate and drift rules remain unchanged.
This amendment is made before observing random-label outcomes. C's descriptive
capacity control uses the paired n8192/seed61 cell. Configurations are
`configs/discrete_p97_v1_capacity_{a,b,c}.yaml` and
`configs/discrete_p97_v1_random_learning.yaml`.

The confirmation generator now accepts prime division moduli and rejects
mismatched A/B tasks/optimizers. Reporting derives the target and plot label
from the frozen modulus, and checks the orchestrator hash against the selected
source checkout so the old p149 frozen sources remain reproducible. The46
focused architecture, capacity-match and reporting tests pass.


During arithmetic development, A28 passed sustained99 and stopped at10,000
updates; C52 stopped at3,000 with100% held-out accuracy. Both checkpoints were
independently reloaded. B52 remained underlearning on both splits at18,100
updates; it continues to its registered120k budget. Before launching a second
candidate, register B64/FF128 (94,912 parameters) at the same seed42/settings.
This trades feed-forward width for residual width while keeping approximately
the same raw parameter count and B/A>=3. It tests whether retaining more binary
coordinates improves arithmetic learning. Neither candidate may train on random
labels before its own successful arithmetic gate and audit. Config:
`configs/discrete_p97_v2_learning.yaml`. Both outcomes will remain in the record.
Both original p149 reports revalidate (nine runs each); an exact archived copy
of their orchestrator is retained at `scripts/frozen/confirm_discrete_p149.py`.


The staged driver `scripts/run_discrete_replication.py` enforces independent
arithmetic audit -> small random learning and audit -> paired A/B capacity
calibration and audits -> capacity match -> frozen confirmations and C capacity
control -> final audits/report. A failed stage stops advancement. Separate
processes isolate RNG state across GPU workers. Its pre-random audit guard is
covered in the47 passing focused tests.


Before any random labels, register B64/FF128's own random-learning/capacity
configs (`configs/discrete_p97_v2_random_learning.yaml`,
`configs/discrete_p97_v2_capacity_{b,c}.yaml`), with the same A28 capacity grid,
seeds, horizons and acceptance criteria. Development candidates may enter
calibration once their own arithmetic/audit gate succeeds; final selection
still requires the complete fixed-dataset capacity match before confirmation.
At16,200 updates B64 had reached99.68% best held-out accuracy but had not yet
satisfied sustained stopping; a single high point is not sufficient.


B64/FF128 completed its arithmetic gate at18,200 updates, with99.2483% held-out
accuracy (the exported audit supplies exact values). Its independent checkpoint
audit passed before any random-label launch. Start its registered small random
validation and, conditional on passing sustained99 plus final>=99, its four
capacity cells. A28 calibration waits for the other arithmetic development job
to release its GPU. The initial B52 candidate is still running; retain its
outcome rather than calling an unfinished pilot a failure.


B64's 2,048-example random-learning run completed10k updates with
99.707031% final training accuracy and a sustained99 crossing;
its checkpoint audit passed (SHA256 `b640d8ccef8a72e2720b941a36843f23be7dd82267eecc1d76529edd452609ed`). Its capacity
calibration is now running under `results/discrete/p97-v2/capacity-B`.
As in the earlier development screens, a candidate can be rejected early if
an independently reloaded saved checkpoint itself exceeds110% of the target
(33,952.912865 bits). Such a result is a lower-bound rejection witness, not a
completed120k capacity estimate. A low partial value never establishes a match.


The initial B52/FF208 arithmetic pilot exhausted120k updates at60.7174% held-out
accuracy (65.8290% training); it failed its arithmetic gate and has no random
runs. B64/FF128's seed61/n8192 capacity checkpoint at34,000 updates independently
stores33990.332108 bits, above the33,952.912865-bit ceiling.
The fixed saved checkpoint is retained as `above_target_checkpoint.pt` with
SHA256 `b12493c90ab77781612a75240d203b20562460c4584d3da5bd6aebc1f7555482` and a passing `audit_capacity.json`.
This rejects B64 as oversized, even before its full120k budget; no final-capacity
estimate is claimed. Its worker was stopped after the independent witness audit.
A28 calibration continues independently and can be paired with a later B.

Before new training, register B60/FF144 (92,820 parameters,3.0164xA) as the next
size candidate. It reduces residual width while staying above the registered
B/A parameter ratio; optimizer, STE, seed42 and arithmetic budget are unchanged.
Config: `configs/discrete_p97_v3_learning.yaml`. Require its own arithmetic gate
and random-learning check before capacity calibration.

B60's potential follow-up grids are registered before results in
`configs/discrete_p97_v3_{random_learning,capacity_b,capacity_c}.yaml`, using the
same dataset sizes, seeds and horizons. They remain conditional on its own
successful arithmetic audit and small random-label learning validation.


B60/FF144 passed its arithmetic gate at18,700 updates with
99.742270% held-out and99.871135% training accuracy.
Its independent audit passed before starting its own random-learning check.
The conditional random-learning -> capacity queue is recorded under
`results/discrete/p97-v3`; A28's existing independent grid remains under
`results/discrete/p97-v2/capacity-A`.


A28/FF112 is also oversized: its independently reloaded58,000-update
n8192/seed61 checkpoint stores34397.401757 bits, above
the same10% ceiling. The saved witness SHA256 is`9b6dfa57b5a39d957638832ea71e7a87220cad54615101e7c949096263770bcb`;
its audit passed before stopping the worker. Retain this as an early rejection,
not a completed capacity grid. Before new training, register A28/FF100
(28,756 parameters) in `configs/discrete_p97_v4_learning_a.yaml` and
`configs/discrete_p97_v4_capacity_a.yaml`. Check arithmetic learning first,
then repeat the same four capacity cells. B60's independent grid continues.

A28/FF100 passed its modular learning check at9,600 updates with
99.570447% held-out accuracy, independently audited. Its four-cell
capacity grid starts at `results/discrete/p97-v4/capacity-A`, using the already
successful B60 arithmetic gate for the random-task authorization check.


B60/FF144 is rejected by its independently audited52,000-update n8192/seed61
checkpoint: 34432.875994 bits, above the fixed ceiling.
Witness SHA256:`50dac3a7fd322f9ec91f875aa4e933f8d41a2caa119461c750eed107f1e95563`. Its capacity worker stopped after the
passing witness audit, with no completed-grid claim. Before new training,
register B64/FF108 (87,232 parameters,3.033523x resized A), returning
to a residual width already shown to learn arithmetic while reducing the FF
width from128 to108. The approximately8% parameter reduction gives more margin
below the fixed capacity ceiling. A28/FF100's grid continues unchanged.
Configs:`configs/discrete_p97_v5_{learning,random_learning,capacity_b,capacity_c}.yaml`.
The new recipe must pass its own arithmetic audit and small random validation.


A28/FF100's98,000-update checkpoint independently stores
34039.721871 bits; its late crossing exceeds the fixed
ceiling. Retained witness SHA256:`5f52103effa31bb00e0d04f2680a4de88059c1d9c5f9f560c840129f257b1a1f`. Stop its capacity
worker after the successful witness audit. Register A26/FF104 (26,910
parameters), a further residual-width reduction, with its own seed42 arithmetic
check and the same capacity grid in `configs/discrete_p97_v6_{learning_a,capacity_a}.yaml`.

B64/FF108 is still below its arithmetic gate after57k updates (best held-out
98.60%). Before launching another candidate, register B60/FF128 (87,060
parameters) to retain more FF width at a similar total count. It reduces the
already successful B60/FF144 candidate by16 FF units. Configs:
`configs/discrete_p97_v6_{learning_b,random_learning,capacity_b,capacity_c}.yaml`.
B64/FF108 continues to its original budget; retain both development outcomes.
The first viable candidate still needs the complete capacity match before
confirmation. Independent small calibration jobs may share a GPU to reduce
elapsed calibration time; RNG stays isolated by process and primary timing
comparisons remain optimizer updates, with seconds descriptive only.


B64/FF108 still lacks sustained99 at87.6k updates (best held-out99.33%);
B60/FF128 remains underlearning at17.8k. Register an additional arithmetic
candidate B72/FF64 (83,736 parameters,3.111706xA26). It keeps72 binary
coordinates at every block boundary while reducing the internal feed-forward
width. There is no learned compression between blocks and no continuous bypass.
This explores whether a wider discrete state can improve arithmetic learning
at approximately the same raw parameter budget. Configs:
`configs/discrete_p97_v7_{learning,random_learning,capacity_b,capacity_c}.yaml`.
Its random stages remain conditional on its own audited arithmetic success.
Earlier pilots continue to their registered stopping rule or120k budget.


B72/FF64 passed its arithmetic gate at11,300 updates with
99.656355% held-out and99.763745% training accuracy.
Its independent checkpoint audit passed before random-label training. Its
small random validation and conditional two-lane capacity queue start under
`results/discrete/p97-v7`. The two calibration dataset sizes run in separate
processes on cuda:0, each processing seeds61 then62. No stopping, optimizer,
data, or acceptance criterion changes. A26 is still completing its arithmetic
check; B64/FF108 and B60/FF128 retain their original arithmetic budgets.


A26/FF104 completed its arithmetic check at41,600 updates with
99.527490% held-out accuracy and passed independent checkpoint audit.
Its capacity grid starts at `results/discrete/p97-v6/capacity-A` in two separate
processes on cuda:1, one per dataset size, using B72's successful arithmetic gate.
B72's small random validation completed10k updates at99.804688% accuracy and
passed audit before its two-lane capacity grid began. B64/FF108 exhausted120k
updates at95.6615% held-out accuracy, failed its gate, and has no random runs;
its checkpoint was independently audited. B60/FF128 is still running its
registered arithmetic budget. Capacity matching remains unconfirmed.


Status discovery now includes nested replication directories. The development
exporter accepts `--prefix p97-`, preserving its existing full-width default.
`reports/p97_development` records arithmetic outcomes and all random runs,
including rejected checkpoints, without implying a capacity match. The48
focused tests pass, including nested-run status/export coverage. Refresh with:
`PYTHONPATH=src:. python scripts/report_fullwidth_development.py --prefix p97- --output reports/p97_development`.


B60/FF128 exhausted120k arithmetic updates at80.8634% held-out accuracy and
failed its gate; it has no random-label runs. All arithmetic development jobs
are now complete. Current compute is the four paired capacity lanes for A26
and B72. A staged finalizer waits for all eight calibration cells, audits them,
and requires the fixed target/cross-arm/drift checks before launching the
registered nine confirmations plus the paired C capacity control. Its command
log and status are in `results/discrete/p97-v7/finalizer.json`.


At89.4k updates, B72/FF64's current peak is33,714.5 bits, still accepted on the
pointwise target test but close to the33,952.9-bit ceiling. Before observing any
new outcome, register a contingency B72/FF60 (82,008 parameters,3.047492xA26)
arithmetic pilot with the same seed42 and settings. Four fewer FF units preserve
the full72-coordinate binary state. Configs:
`configs/discrete_p97_v8_{learning,random_learning,capacity_b,capacity_c}.yaml`.
The incumbent full calibration continues. The contingency's random stages are
conditional on its own audited learning gates and are needed only if the
incumbent fails matching. Do not switch candidates based on confirmation data.


B72/FF64's completed seed61/n8192 run records a peak33,953.867427 bits, exceeding
the33,952.912865-bit ceiling by0.954562 bits. The final checkpoint stores
33,689.424526 bits; peak weights were not separately retained. This is a
completed full-budget recorded-peak rejection, not a reloaded above-bound
checkpoint witness. Checkpoint trace equals the exported trace, and its full
120k summary reproduces the peak. Both seed61 cells completed; remaining seed62
cells were scope-stopped because they cannot undo the monotone peak rejection.
Their incomplete traces are censored. The finalizer stopped before confirmation.
Do not relax the tolerance or switch to final-only capacity for this near miss.

A26's first seed completed with max-over-size31,608.686332 bits and continues
seed62. Its evidence is independent of which B candidate is eventually selected.
B72/FF60 seed42 remains in arithmetic development. Register one additional
development seed41 for exactly the same recipe, before observing its outcome,
in `configs/discrete_p97_v8_learning_s41.yaml`. Single-seed arithmetic training
has shown instability across the size candidates. Retain both outcomes; a
successful gate establishes learnability, not robust multi-seed performance.
The confirmation seeds remain73/74/75 and have not been run or used for selection.


Before any outcomes for the intermediate size, register B72/FF62 (82,872
parameters,3.079599xA26) on development seeds42 and41 together. B72/FF64
passed learning but missed the capacity cutoff by0.954562 bits; FF60 seed42 is
still underlearning at39.8k and its seed41 is ongoing. FF62 bisects that size
change while preserving72 binary coordinates and every optimizer/STE setting.
Configs:`configs/discrete_p97_v9_{learning,random_learning,capacity_b,capacity_c}.yaml`.
Use the first completed, independently audited successful development gate for
any later random tasks. Keep all development outcomes visible. No p97
confirmation has begun, and confirmation outcomes cannot influence selection.


B72/FF62 passed arithmetic on both development seeds: seed42 stopped at13,300
updates with99.8926% held-out accuracy, seed41 at19,400 with99.0550%. Both
checkpoints passed independent audits. Seed42 completed first and is the
registered gate for its own random tasks. Start its small random validation
under `results/discrete/p97-v9`; conditional on success, run n8192 seeds61/62
concurrently on cuda:0, then n16384 seeds61/62 on cuda:1 once A's grid completes.
This schedules the unchanged full four-cell grid using available resources.
A staged finalizer again requires all complete audits and the strict capacity
match before any confirmation. Its scripts/status are retained in the results
folder. B72/FF60 development outcomes remain separate and cannot substitute
for B72/FF62's gate or capacity evidence.

B72/FF62 completed its own10k-update small random validation at
99.414062% training accuracy, with a sustained99 crossing and
passing independent audit. Checkpoint SHA256:`2fc1ecb09e2dce5810a42c0df37563469646c3702b623c2dd881a0075d0f52a9`.
Its two n8192 capacity seeds are now running; both n16384 seeds are queued for
the GPU released when A26's full grid completes. Finalizer status and command
history are in `results/discrete/p97-v9/finalizer.json`.


A26's four120k capacity cells completed. Per-seed max-over-size estimates are
31,608.686332 bits (seed61) and31,579.711267 bits (seed62). They remain within10% of the30,866.284423-bit target. Final
independent grid audits and the A/B cross-match are still required. B72/FF62's
four cells are now all running, with both n16384 seeds launched after A released
cuda:1. B72/FF60 seed42 exhausted120k below its arithmetic gate and has no
random runs; seed41 remains in development.

Before confirmation, preserve the scientific code/config checkout at commit
`aa70459` in `results/discrete/source-worktrees/p97-v9`. This detached snapshot
leaves the working branch `idea/discrete` unchanged and supports later report
revalidation without relying on future model-code edits.


## Accepted p97 architecture match and frozen confirmation

The full paired grid passed with target 30866.284423 bits and raw B/A parameter ratio 3.079599. All eight final checkpoints were independently audited before confirmation.

| Calibration seed | A capacity (bits) | B capacity (bits) | A / target | B / target |
|---|---:|---:|---:|---:|
| 61 | 31608.686332 | 33446.009186 | 1.024052 | 1.083577 |
| 62 | 31579.711267 | 33425.687474 | 1.023113 | 1.082919 |

A is width26/FF104, 26,910 parameters. B is width72/FF62 with full-width binary
RMS/clipped-STE states at each block boundary, 82,872 parameters. C is an ordinary
continuous width72/FF62 transformer with exactly the same parameter tensors
and paired initialization as B. The FF width is internal to each residual
block; the state crossing a block boundary has72 coordinates.

The parameter count and capacity match were frozen before any confirmation.
Run seeds73/74/75 in ABC/BCA/CAB order with the unchanged optimizer, split and
120k-update maximum. These configurations are in
`results/discrete/p97-confirmation-v9-quantized-rms/selection.json` and
`configuration.yaml`. C's paired n8192/seed61 random-label capacity control is
under `results/discrete/p97-v9/capacity-C`. Final reporting must wait for all
nine outcomes and the independent audits, with missing crossings preserved.

The p97 report identifies the first-passage grokking metric as specified before
this replication, and limits development history to its p97 runs, including
nested calibration folders. This metadata change does not alter thresholds,
selection, training, or delay calculations.


## Completed p97 binary replication

All nine frozen confirmation runs and the continuous capacity control completed.
All nine arithmetic checkpoints passed independent reload/evaluation audits;
the eight selected A/B capacity checkpoints and C control also passed their
independent audits. The report revalidates against the scientific sources in
`results/discrete/source-worktrees/p97-v9` (commit `aa70459`). The current reporter
uses the actual capacity sizes for its plot axes and explicitly labels sustained
99% crossings. These presentation changes do not alter any training or outcome.

The task remains modular division at p=97, 50% training, 4,656 training examples,
and a registered label-information target of 30,866.284423 bits. The accepted
architectures and two-seed capacity table above were frozen before confirmation.
B passed arithmetic development on seeds 42 and 41 before its random tasks and then
passed its own 2,048-example random-label validation: final 99.4140625%,
best 99.90234375%. Confirmation outcomes were not used to change the model.

All entries below are optimizer updates, in seed order **73 / 74 / 75**.
A crossing requires three consecutive evaluations, spaced 100 updates apart.
At these evaluation points, 100 updates equals 10 epochs (ten minibatches per
epoch, including a final 48-example minibatch). A missing crossing remains
missing; it is not replaced with 120,000.

| Model | Sustained held-out 95% | Sustained held-out 99% | Paper-style delay |
|---|---|---|---|
| A, continuous 26/FF104 | 7,000 / 34,000 / 7,500 | 38,600 / 46,900 / 9,600 | 3,200 / 400 / 400 |
| B, binary 72/FF62 | 15,200 / 17,400 / 12,700 | missing by 120k / 20,500 / 17,200 | 0 / 0 / 0 |
| C, continuous 72/FF62 | 2,100 / 1,800 / 2,400 | 2,800 / 2,200 / 2,600 | 200 / 100 / 100 |

At 95%, all runs cross, so the three-seed medians are A 7,500, B 15,200,
C 2,100 updates. The ratio of medians is B/A 2.026667 and B/C 7.238095.
B beats A on seed 74 but loses on 73/75; C is fastest on every seed.
At 99%, A/C three-seed medians are 38,600/2,600. B has only two observed
crossings; their 18,850-update median is conditional on success and must not
be presented as the full three-seed median or evidence that B is faster overall.
B seed 73 finishes 120k at 95.317870% training and 94.029212% held-out accuracy.

The paper-style delay is `max(0, first held-out 98% - first train 99%)`, using
first observations rather than sustained crossings. Every A seed has a positive
delay despite the capacity match. B has zero observed delay on all three seeds,
but its failed seed is still a learning failure. None of the nine runs meets the
separate train>=99%/held-out<90% memorization-gap condition. These observations
do not establish that B can never grok or that empirical capacity matching is
an intrinsic capacity bound. Delays shorter than 100 updates remain unresolved.

Measured median wall time to 95% was A 37.930296s, B 84.466698s, C 10.578991s.
These are descriptive shared-H100 measurements; optimizer updates are the
primary speed comparison. Exact per-run seconds, examples, epochs, paired
ratios, thresholds, and final accuracies are retained in the CSV/JSON report.

C's paired n=8,192/seed 61 control records 51,329.060604 bits at 120k; the final
checkpoint independently reproduces the same value. It does not trigger the
registered 95%-of-label-entropy data-limit flag. This one tested cell gives a
finite-budget lower bound, not a precisely measured intrinsic maximum. It is
well above the A/B estimates and the dataset target, while B/C have identical
parameter shapes and initialization and only B quantizes block-boundary states.

For context, the completed p149 scalar medians to 95% were A 2,800, B 19,300,
C 1,800 updates, with B also reaching sustained 99% on two of three seeds.
At p97 the median ranking remains C, A, B, although B now beats A on one seed.
A's paper-style delays change from 0/0/0 to 3,200/400/400; B remains 0/0/0.
This is a resized, capacity-matched replication, not an isolated causal test of
modulus: the residual and feed-forward dimensions were recalibrated.

The [completion audit](reports/p97_development/completion_audit.json) binds 23
selected/supporting checkpoints to their exact traces, configs, source snapshots,
and existing evaluation audits. It verifies selection and all A/B calibration
audits preceded confirmation, and arithmetic/random-learning gates preceded
capacity training. All task controllers have finished. The full development
export retains 14 arithmetic and 21 random/capacity runs, including unsuccessful
sizes and censored cells. Its [capacity rejection record](reports/p97_development/capacity_rejections.json)
distinguishes four saved/reloaded above-bound witnesses from the v7 completed
recorded-peak near miss whose peak weights were not retained. No tolerance was
relaxed to accept that near miss.

Revalidate without changing gate audits:

```bash
PYTHONPATH=results/discrete/source-worktrees/p97-v9/src:. python results/discrete/p97_completion_audit.py
```

The audit script is also preserved under `reports/p97_development/`.
The earlier 48 focused checks passed; the six reporting tests also passed after
the capacity-axis generalization. Both exported figures were visually checked.
Report and development artifacts have SHA256 manifests; raw checkpoints remain
in the ignored results tree. No p97 categorical comparison is claimed here.


## Verification against the original p97 paper baseline (2026-10-25)

The published paper reports immediate generalization near its estimated capacity
threshold and delayed generalization at larger sizes. This is an empirical trend,
not a theorem that every capacity-matched run must have zero delay. Section 4.3
reports the minimum delay across ten seeds; a plotted zero alone does not imply
that every individual seed has zero delay. Reference:
https://arxiv.org/html/2605.09724v2#S4.SS3 and section 5.3.

`scripts/check_p97_paper.py` independently recomputes crossings from 680 original
saved traces (68 registered widths, ten seeds each, original host and central
hyperparameters). The registry is opened read-only. Every trace's modulus,
training-set size, parameter count, dimension, and first 99% validation epoch
are checked against the registry; SHA256 hashes are recorded. Missing threshold
crossings remain null. The [per-width counts](reports/paper_p97/sizes.csv) include:

| Original width | Parameters | Published capacity / dataset (2.16P/K) | Zero delays | Positive delays | Incomplete delays |
|---|---:|---:|---:|---:|---:|
| 20 | 16,860 | 1.180 | 0 | 0 | 10 |
| 22 | 19,954 | 1.396 | 5 | 0 | 5 |
| 26 | 26,910 | 1.883 | 3 | 0 | 7 |
| 34 | 43,894 | 3.072 | 9 | 1 | 0 |
| 54 | 104,274 | 7.297 | 0 | 10 | 0 |

The smallest checked width (20), closest to the published capacity threshold,
never reaches training 99% on any seed; no successful-generalization inference
should be drawn from its absence of measured delays. At width 26, exactly the
same parameter count as our A, the three fully observed delays are all zero.
The other seven lack a training 99% crossing; five of the ten runs do reach
held-out 99%. Width 54 is the first size after which all ten seeds at every
checked larger central-grid width through 250 have positive delays. Its delays
are 2–25 epochs. This verifies the small-model no-observed-delay trend, without
turning censored runs into successful examples or a guarantee across settings.

There are material differences from our binary comparison:

- The original modular baseline uses dropout 0.2. Source at its recorded commit
  `f6b0754`, `src/torch_grokking/experiments/groks.py`, accumulates training accuracy
  from training-mode minibatch predictions during the epoch; held-out accuracy
  is measured with dropout off after the epoch. Our discrete experiment uses
  dropout 0 and evaluates both entire sets at a fixed checkpoint. The numerical
  99/98 thresholds coincide, but the training-accuracy observables do not.
- Original evaluations are every epoch, with a 5,000-epoch limit and ten seeds.
  Ours are every ten epochs, with a 12,000-epoch limit and three fresh seeds.
- The published capacity convention is 2.16 bits per parameter. Applying that
  convention to A gives 58,125.6 bits, or 1.883 times the 30,866.284423-bit target,
  rather than our directly measured 31.6k-bit estimate under the new protocol.
  This arithmetic is not a validation or transfer of the published constant.
  The existing internal audit at
  `docs/papers/2026 Grokking Capacity/submissions/logs/p01a_capacity_provenance.md`
  records that the historical 2.16 fit could not be recovered from its available
  records. This verification checks the delay traces, not that missing fit.
  The completed follow-up below verifies separate September local calibration
  runs; those do not recover the provenance of the original published 2.16 fit.

The new A's positive first-passage delays remain correct for its three runs.
Describing them as a direct contradiction of the original paper, or describing
our protocol as an exact replication of its no-grokking regime, would be too
strong. Differences in training accuracy measurement and calibration must be
controlled before such a claim. The present evidence does not identify which
protocol difference causes the discrepancy, and no new training was run for
this verification.


## Registered dropout follow-up and local capacity calibration (2026-10-25)

Before fresh outcomes, freeze `configs/discrete_p97_dropout_followup.yaml`.
Keep A at width 26/FF104, 26,910 parameters, p97 division and the same 50% split.
Run ten seeds 73–82 with dropout 0.2 and the original `GrokkingTrainer`: training
accuracy is accumulated during minibatch updates, held-out accuracy is evaluated
at the end of every epoch. Use the paper's 5,000-epoch limit (50,000 updates),
early stop only when both train and held-out accuracy reach 99%, and report the
minimum first-train 99%/first-held-out 98% delay with all per-seed outcomes and
censoring counts. This differs from the prior 12,000-epoch ceiling and must remain
explicit. No architecture or hyperparameter tuning based on these outcomes.

Complete local calibration data were found in the separately stored
`/home/ubuntu/capacity_defaults_runs.db` and `capacity_matched_runs.db`: each has
56 completed random-label runs from this host, seven widths 10–22 and eight
sizes 1000–9300, p113, seed 42, trained 2026-09-26. Verify saved traces, registry
identities, recorded source commits (including dirty flags), formulas, and hashes
before quoting a fit. The defaults use dropout 0/weight decay 0.01; the matched
regime uses dropout 0.2/weight decay 1. These differ in two regularization settings,
so their difference is not an isolated dropout effect.

A fresh independent seed 61 will repeat the full 56-cell matched calibration grid
with saved final checkpoints. Eight additional random-label cells measure A
itself at p97: n2048/4096/8192/16384, seeds 61/62, dropout 0.2/weight decay 1.
These direct measurements avoid assuming the p113 fitted constant transfers
exactly to width 26 or vocabulary 99. Use the original capacity patience 100 epochs,
minimum loss improvement 1e-4, and 5,000-epoch ceiling. Fit maximum *final* learned
bits over dataset sizes at each parameter count, with a free intercept and
through-origin sensitivity fit; retain all traces, report saturation and data
limitations, and do not force agreement with 2.16. Bootstrap over widths is fit
sensitivity, not a multi-seed confidence interval. Compare the two matched seed
fits explicitly rather than silently pooling seed maxima.

`run_p97_dropout_followup.py` uses four worker slots, two per GPU, with isolated
registries and output paths; it leaves unrelated GPU jobs alone. Scientific
sources/configs are snapshotted and hashed before dispatch. The only change to
the capacity training module is an optional `--save-model` output for final
checkpoint audits; training updates, stopping and measurements are unchanged.
No new categorical or binary B experiment is implied by this A-only follow-up.


## Completed ten-seed A dropout follow-up (2026-10-25)

The registered arithmetic rerun is complete. With dropout 0.2, fixed A26/FF104
(26,910 parameters), and the original paper-style training/measurement protocol,
four seeds have fully observed delays and all four are **zero**. Six seeds never
reach online training 99% within 5,000 epochs. None has an observed positive delay.
Five seeds reach held-out 99% at least once, but only four finish with both train
and held-out accuracy at least 99%. Thus the positive-minimum grokking criterion
is not met; this is not evidence that all ten seeds generalize successfully.

The minimum over the four observed delays is zero. A complete ten-seed delay
vector is unavailable: missing crossings remain null rather than being imputed
as zero. One epoch is ten optimizer updates; the ceiling is 50,000 updates.

| Seed | First online train 99% epoch | First held-out 98% epoch | First held-out 99% epoch | Delay (epochs) | Final train / held-out (%) |
|---|---:|---:|---:|---:|---:|
| 73 | — | — | — | — | 96.78 / 96.80 |
| 74 | — | — | — | — | 54.83 / 45.12 |
| 75 | — | 4975 | — | — | 96.59 / 98.05 |
| 76 | 1768 | 1542 | 1832 | 0 | 99.29 / 99.25 |
| 77 | 1665 | 682 | 965 | 0 | 99.01 / 99.74 |
| 78 | 2243 | 866 | 1022 | 0 | 99.12 / 99.44 |
| 79 | 1005 | 917 | 920 | 0 | 99.05 / 100.00 |
| 80 | — | — | — | — | 86.68 / 83.46 |
| 81 | — | — | — | — | 86.10 / 84.28 |
| 82 | — | 1509 | 1854 | — | 97.08 / 98.37 |

Seed 82 reaches held-out 99% transiently and finishes at 98.37%; it is included
in the five first-crossing successes, not the four final joint successes.
Independent CPU checkpoint evaluation reproduces all ten final held-out
accuracies, and all ten saved initial states exactly match same-seed reconstruction.
The final eval-mode training accuracies are exported separately as diagnostics;
they do not replace the primary online training observable.

For the three seeds shared with the earlier dropout-free experiment, data hashes
and initial-weight hashes match exactly. However, this rerun uses the original
`GrokkingTrainer` to match the paper: shuffling uses the global CPU RNG instead
of a separate generator, AdamW is not explicitly fused, training accuracy is
measured online with dropout, evaluation is every epoch rather than every ten,
and stopping uses one joint 99% epoch with a 5,000-epoch maximum rather than
three sustained evaluations with a 12,000-epoch maximum. Therefore this is not
a controlled dropout-only causal ablation. Dropout here acts on each block
feedforward hidden activation, as implemented by the original transformer.

Artifacts: [all ten outcomes](reports/p97_dropout_followup/A_runs.csv),
[learning curves](reports/p97_dropout_followup/A_accuracy.png),
[checkpoint audits](reports/p97_dropout_followup/checkpoint_audits.json), and
[paired protocol comparison](reports/p97_dropout_followup/protocol_comparison.json).
The reproducible analysis is `scripts/report_p97_dropout_followup.py`.

### Direct capacity of fixed A with dropout

All eight direct p97 random-label tests are complete: n2048/4096/8192/16384
on seeds 61 and 62, with dropout 0.2, weight decay 1, patience 100 epochs and
minimum online-loss improvement 1e-4. The estimate is the maximum final
eval-mode learned bits over the four dataset sizes, not a maximum over epochs.

| Capacity seed | Maximum final bits | Fraction of 30,866.284-bit target | Winning random dataset size | Late 100-vs-100 epoch drift |
|---|---:|---:|---:|---:|
| 61 | 21,663.493 | 70.1850% | 8,192 | 0.605% |
| 62 | 21,819.783 | 70.6913% | 8,192 | 0.409% |

Both maxima are at n8192, below the largest tested size; both n16384 values
are lower. Neither winning run reaches the 5,000-epoch ceiling or the label
entropy limit. Both late drifts are below 1%. These checks support the registered
finite-training estimate; they do not establish an intrinsic capacity bound.

Fixed A is therefore **not capacity-matched to the dataset under this new
protocol**: both estimates are about 30% below the target, outside the required
10% tolerance. A was deliberately not resized after seeing the outcomes.
The prior dropout-free 31.6k-bit match remains correct for its own frozen
protocol; changing dropout, random-data generation and stopping requires
remeasurement. This follow-up does not establish a new matched A/B/C setup.

All eight final checkpoints reproduce their learned-bit totals in independent
CPU evaluation. See [capacity rows](reports/p97_dropout_followup/capacity_runs.csv),
[direct A curves](reports/p97_dropout_followup/A_capacity.png), and
[summary](reports/p97_dropout_followup/summary.json).


### Completed local capacity-constant calibration

Both historical 56-cell grids and the fresh 56-cell confirmation are verified.
All were run on this H100 host (`68-209-74-145`). Each grid uses p113, random
tokens and labels, seven widths 10–22, and eight dataset sizes 1000–9300. At
each width, take the maximum **final** learned bits over dataset sizes, then
fit `M = C * P + b` with a free intercept. The through-origin slope is a
separate sensitivity analysis; the two matched seeds are never pooled by
selecting the more favorable seed at each width.

| Protocol | Seed | C (bits/parameter) | Intercept (bits) | R² | Through-origin slope |
|---|---:|---:|---:|---:|---:|
| Historical defaults: dropout 0, WD 0.01 | 42 | 2.095545 | 303.536 | 0.998226 | 2.116457 |
| Matched: dropout 0.2, WD 1 | 42 | 0.816713 | 372.861 | 0.998241 | 0.842402 |
| Fresh matched: dropout 0.2, WD 1 | 61 | 0.807907 | 498.515 | 0.998577 | 0.842253 |

The independent matched-seed slopes differ by 1.08%; the through-origin
slopes are both about 0.842. The fresh slope's 95% width-bootstrap interval is
[0.7840, 0.8402]. This resamples seven width points and measures fit
sensitivity, not population uncertainty across independent training seeds.

No selected maximum in any of the three grids reaches 95% of label entropy,
occurs at the largest tested dataset size, reaches the epoch ceiling, or has
more than 1% relative drift between the last two 100-epoch windows. For the
fresh grid, selected-maxima drift ranges from 0.26% to 0.68%. The alternative
maximum-over-training-trace fit has slope 0.808611, close to the primary 0.807907.
These are finite-training estimates for these data and optimization settings.

The two historical grids were collected on 2026-09-26 in separate local
registries. All 112 saved traces agree with their registry identities and
learned-bit formulas. The default grid has clean recorded source state. The
matched grid has 44 rows with a dirty-source flag, so its full uncommitted
source cannot be reconstructed. Recorded commits have an identical
`CapacityTrainer` AST and byte-identical data module; the model change only
adds optional feedforward width with the same default. The fresh confirmation
uses the registered, snapshotted scientific source at `0c30587`, checked before
every launch and at completion, and saves every final checkpoint. Working-tree
dirty flags caused by report work do not change those verified source hashes.

The historical July audit remains correct about not recovering the published
2.16 fit. These newly located September data provide an independent local fit,
2.095545 under the documented defaults, not the original missing fit. The
matched-protocol value is about 0.81. Both grids use this same machine, while
dropout and weight decay both differ between regimes; their difference cannot
be attributed to hardware or dropout alone. A fitted p113 slope also does not
replace direct p97 calibration, which found fixed A below the dataset target.

All 74 new final checkpoints and ten initial arithmetic checkpoints passed
independent checks. The largest capacity reconstruction error is
0.001118 bits; held-out accuracy matches to floating-point
precision. The 186 traces (112 historical plus 74 new) and saved weights/source
are covered by 277 input-artifact hashes. All registered jobs exited zero,
the controller is complete, and no controller process remains. Shared GPUs
were used, so elapsed time is not a controlled speed benchmark.

Artifacts: [capacity fits](reports/p97_dropout_followup/capacity_fits.png),
[all dataset-size curves](reports/p97_dropout_followup/capacity_grid.png),
[all capacity rows](reports/p97_dropout_followup/capacity_runs.csv),
[registry provenance](reports/p97_dropout_followup/registry_rows.csv),
[source audit](reports/p97_dropout_followup/historical_source_audit.json), and
[completion validation](reports/p97_dropout_followup/validation.json). PDF
versions of all figures are saved alongside the PNGs. Recompute with:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/report_p97_dropout_followup.py --audit --plots
```


## p=97 A/B/C rematching with dropout 0.2 (registered 2026-10-25)

The new experiment is registered in `configs/discrete_p97_dropout_match.yaml`.
The fixed modular-division training set remains 4,656 examples, or
30,866.2844 label-information bits. All three models use dropout 0.2,
AdamW learning rate 0.001, weight decay 1, batch size 512, and the original
optimizer implementation. A and binary B are resized using direct random-label
capacity estimates on seeds 61/62; both must match the dataset and each other
within 10%. C is the ordinary continuous transformer with exactly B's
architecture size, parameter count and same-seed initialization.

Each candidate must first pass a same-recipe arithmetic learning gate and then
learn 2,048 random labels before larger capacity datasets are authorized.
The capacity grid is 2,048/4,096/8,192/16,384 samples with the original
CapacityTrainer. Selected maxima are checked for dataset limits, epoch limits
and late drift. The initial candidates are A width 32/FF128 and B
width 72/FF62, 80/FF104, 88/FF128; all development attempts are retained.
Only width/feedforward dimensions may change during matching.

Sizes will be frozen before ten fresh confirmation seeds 101–110 are run.
Each arithmetic run has a 5,000-epoch ceiling (50,000 optimizer updates),
and stops after three consecutive held-out 99% epochs when the current
online training accuracy is also at least 99%. Paper-style first-passage
delay uses online training 99% and held-out 98%; eval-mode training produces
a separately reported sensitivity result. Missing crossings remain missing.
Speed uses sustained held-out 95%/99% in optimizer updates. All three models
share this protocol. CPU parity tests verify identical parameters and RNG
state against the original GrokkingTrainer for continuous and binary models.

Artifacts will be kept under `results/discrete/p97-dropout-match-v1`, with
source snapshots, immutable stage configurations, initial/final checkpoints
and independent final audits. The completed results are recorded below. Old dropout-free matching
and the earlier under-capacity A dropout experiment do not substitute for this run.

Matching diagnostic recorded before confirmation: binary width 88/FF72,
seed 62, N=8,192 stopped at epoch 864 with 32,090.53 bits and 1.339%
late drift. The registered convergence check therefore requires an extension.
The diagnostic replays the original run and requires bitwise identical weights
and traces at its original stopping point. It then continues the same optimizer
and RNG in segments of at most 200 epochs, retaining the original 5,000-epoch
total ceiling. It stops after at least 200 additional epochs and late drift
at most 1%. The primary checkpoint/result remains unchanged; reports retain
both values and use the extended final measurement for matching if validated.
This is the registered convergence extension, before any confirmation outcome.

C's same-grid measurements establish excess capacity directly. If its maximum
is still at a data boundary or drifting, it is reported as a lower bound;
precisely locating C's upper capacity is not required to establish that it
exceeds A/B. A/B matching requires converged, interior maxima.

The architecture selection is now frozen in
`reports/p97_dropout_match/selection.json`, before confirmation seeds 101–110:

| Model | Width / FF width | Parameters | Capacity, seeds 61 / 62 (bits) |
|---|---:|---:|---:|
| A: ordinary continuous | 32 / 128 | 39,264 | 31,288.33 / 31,254.55 |
| B: full-width binary | 88 / 72 | 117,832 | 33,071.14 / 32,316.36 |
| C: ordinary continuous | 88 / 72 | 117,832 | at least 59,578.93 / 59,272.47 |

A/B match the 30,866.28-bit dataset and each other within 10% on both seeds.
B/C have exactly the same parameter count and will use identical same-seed
initial weights. B's seed-62 continuation reproduced the original 864-epoch
trajectory bit for bit, then trained 200 further epochs with the same optimizer
state. Its late drift fell from 1.339% to 0.644%; the primary estimate remains
archived at 32,090.53 bits. Both accepted A/B maxima are interior N=8,192
points and pass all convergence flags. C's maxima are at the largest tested
N=16,384, so they establish lower bounds rather than a saturated upper capacity.

Checkpoint verification retains CPU/GPU differences at hard binary thresholds.
One N=4,096 binary run changes a single prediction on CPU: a first-layer input
is -9.31e-10 on CPU versus +1.86e-9 on GPU. Its independently reloaded GPU
checkpoint reproduces the saved accuracy and information. The continuation
checkpoint also has a small CPU information difference (0.186 bits); its GPU
recheck agrees. No accuracy tolerance is relaxed; these discrepancies are
recorded in the audit artifacts. Confirmation outcomes are recorded below.


## Completed p97 capacity-matched A/B/C with dropout 0.2

All thirty fresh arithmetic runs (seeds 101–110) are complete. The selection
was frozen and committed before these seeds ran; no architecture or optimizer
was changed after confirmation started. Each model used the same original
AdamW implementation, learning rate 0.001, weight decay 1, betas (0.9, 0.98),
batch size 512, dropout 0.2, and a 5,000-epoch ceiling. The fixed modular-division
training set is 4,656 examples per seed: one epoch is exactly ten optimizer
updates, and the ceiling is 50,000 updates. A/B/C share each seed's training
and held-out data; B/C also share identical initial weights. Speed below means
optimizer updates, not wall-clock time on shared GPUs.

A is the ordinary width-32/FF128 transformer (39,264 parameters). B is the
width-88/FF72 full-width binary transformer (117,832 parameters), with fixed
±1 activations after both blocks and the RMS-clipped STE. C is the ordinary
continuous width-88/FF72 transformer, also exactly 117,832 parameters. It has
no quantizers or compression adapters. Dropout operates inside each block;
binary boundaries remain exactly ±1 in both train and eval modes.

The target is 30,866.2844 bits. Accepted capacities are A
31,288.33/31,254.55 and B 33,071.14/32,316.36 bits on seeds 61/62. Both models
match the dataset and each other within 10% on both seeds. The registered
convergence extension for B seed 62 is documented above and retained beside
the primary measurement. C demonstrates at least 59,578.93/59,272.47 bits on
the same grid. Those are lower bounds because its maximum occurs at the
largest tested dataset; establishing excess capacity does not require locating
C's exact upper capacity. These remain empirical, protocol-dependent estimates.

### Learning speed and success

Thresholds require three consecutive held-out evaluations; time is the first
epoch of that window. The arithmetic stopping rule additionally requires the
current online training accuracy to reach 99%.

| Model | Sustained 95% | Sustained 99% | Median updates to 95%, all ten seeds | Median updates to 99%, all ten seeds |
|---|---:|---:|---:|---:|
| A | 10/10 | 9/10 | 5,275 | 8,310 |
| B | 5/10 | 5/10 | >30,065 | >30,215 |
| C | 10/10 | 10/10 | 2,385 | 2,965 |

B's medians are **lower bounds**, not observed crossing times. Five B runs never
reach even sustained 95% within 50,000 updates. For 99%, B's fifth observed time
is 10,430 updates and its sixth is unknown but exceeds 50,000, so the ten-seed
sample median exceeds (10,430 + 50,000)/2 = 30,215. The successful-run-only
medians are B 8,700 updates to 95% and 9,650 to 99%; these conditional values
must not replace the all-seed comparison. A's 99% median remains identifiable
despite one censored run because nine crossings are observed.

C precedes both A and B on every paired seed at both thresholds, including
seeds where A or B never crosses within budget. A precedes B on 9/10 seeds at
95%; B precedes A on the other seed (106). At 99%, A precedes B on seven seeds,
B precedes A on two (101 and 106), and neither reaches the threshold on seed
109. A uses 2.21× C's median updates to 95% and 2.80× to 99%. B's median lower
bounds are approximately 5.70× A's median at 95% and 3.64× at 99%; those
bounds retain all five failures instead of conditioning on successful runs.

| Seed | A: 95% / 99% updates | B: 95% / 99% updates | C: 95% / 99% updates |
|---:|---:|---:|---:|
| 101 | 5,370 / 12,080 | 8,920 / 9,960 | 1,730 / 2,640 |
| 102 | 3,900 / 5,380 | 5,670 / 6,400 | 3,360 / 4,270 |
| 103 | 3,830 / 4,010 | >50,000 / >50,000 | 2,500 / 3,120 |
| 104 | 5,180 / 7,690 | 10,130 / 10,430 | 2,700 / 2,890 |
| 105 | 4,720 / 6,010 | >50,000 / >50,000 | 2,760 / 3,150 |
| 106 | 11,090 / 13,180 | 8,230 / 8,710 | 1,820 / 2,080 |
| 107 | 6,100 / 8,930 | >50,000 / >50,000 | 2,270 / 2,560 |
| 108 | 3,880 / 5,150 | 8,700 / 9,650 | 2,760 / 3,120 |
| 109 | 22,740 / >50,000 | >50,000 / >50,000 | 2,090 / 3,040 |
| 110 | 9,260 / 14,840 | >50,000 / >50,000 | 2,150 / 2,520 |

The final unsuccessful runs are A seed 109 (93.96% held-out accuracy), and B
seeds 103/105/107/109/110 (37.91%, 37.11%, 43.60%, 63.17%, 27.41%). A109 had
earlier reached sustained 95%. These are learning failures within the stated
budget, not successful zero-delay generalization.

### Grokking delay and the requested minimum across ten seeds

The primary paper-style delay is
`max(0, first held-out 98% epoch - first online training 99% epoch)`.
Online training accuracy includes dropout and parameter changes during the
epoch. The secondary measure substitutes end-of-epoch eval-mode training
accuracy, with dropout disabled, and leaves held-out evaluation unchanged.
Both metrics were registered before the confirmation runs.

| Model | Primary: positive / zero / undefined seeds | Minimum observed primary delay (updates) | Eval-mode: positive / zero / undefined seeds | Minimum observed eval-mode delay (updates) |
|---|---:|---:|---:|---:|
| A | 0 / 9 / 1 | 0 | 7 / 2 / 1 | 0 |
| B | 0 / 5 / 5 | 0 | 0 / 5 / 5 | 0 |
| C | 3 / 7 / 0 | 0 | 10 / 0 / 0 | 140 |

The requested online/minimum statistic is zero for A, B **and C**. It does not
separate A/B from C in this experiment. With eval-mode training accuracy, the
minimum is zero for A/B and 140 updates for C; all ten C runs have a positive
eval-mode delay. A nevertheless has a positive eval-mode delay in seven of
nine defined runs, up to 5,200 updates. Thus a zero minimum does not establish
that A never has delayed generalization. B's five successful runs have zero
delay under both measurements; the five failures have undefined delays and
are never imputed as zero.

| Seed | A primary / eval-mode delay, updates | B primary / eval-mode delay, updates | C primary / eval-mode delay, updates |
|---:|---:|---:|---:|
| 101 | 0 / 5,200 | 0 / 0 | 0 / 280 |
| 102 | 0 / 0 | 0 / 0 | 240 / 770 |
| 103 | 0 / 0 | undefined / undefined | 0 / 480 |
| 104 | 0 / 430 | 0 / 0 | 30 / 220 |
| 105 | 0 / 310 | undefined / undefined | 0 / 290 |
| 106 | 0 / 460 | 0 / 0 | 0 / 140 |
| 107 | 0 / 320 | undefined / undefined | 0 / 220 |
| 108 | 0 / 180 | 0 / 0 | 40 / 460 |
| 109 | undefined / undefined | undefined / undefined | 0 / 360 |
| 110 | 0 / 1,330 | undefined / undefined | 0 / 340 |

### Development checks and reproducibility

Every capacity recipe first passed its own arithmetic gate at dropout 0.2.
On 2,048 random labels, selected A and C reached 100% eval-mode accuracy on
both capacity seeds. Selected B reached 99.8047% and 99.9023%, passing the
registered 99% criterion on both seeds. Passing a development gate did not
guarantee confirmation success. All attempted gates are retained:

| Development recipe | Seed | Arithmetic gate | Optimizer updates | Final held-out accuracy |
|---|---:|---|---:|---:|
| A32f128 | 41 | passed | 5,120 | 99.21% |
| B72f62 | 41 | failed within budget | 50,000 | 90.29% |
| B80f104 | 41 | failed within budget | 50,000 | 98.11% |
| B80f104 | 42 | passed | 13,520 | 99.18% |
| B88f128 | 41 | passed | 3,390 | 99.36% |
| B88f48 | 41 | failed within budget | 50,000 | 81.46% |
| B88f72 | 41 | passed | 4,970 | 99.08% |
| C80f104 | 41 | passed | 2,280 | 99.61% |
| C88f72 | 41 | passed | 3,040 | 99.21% |

The width-88/FF128 binary candidate was over capacity (40,611.92 bits on its
first sweep). Width-88/FF72 was accepted from learning gates and capacity,
before seeing any confirmation outcomes. Unselected candidates and all failed
attempts remain in the report and provenance.

The report preserves initial/final checkpoints, source hashes, frozen stage
configurations, the selection commit, both calibration seeds, all thirty
confirmation traces, and the continuation's exact replay proof. The common
arithmetic loop passed CPU parity tests against the original GrokkingTrainer
for both continuous and binary models, including identical final weights and
RNG state. An independent NumPy implementation verifies all thirty first
crossings, sustained windows, delay counts, censoring bounds and medians.
Checkpoint audits recreate every dataset and initialization, check hard binary
boundaries with dropout on/off, and independently recompute final metrics. All
69 final checkpoints (68 primary runs plus the continuation) and 68 initial
checkpoints pass their checks; all ten B/C initial-weight and dataset pairings
are verified. Thirty targeted model/protocol tests passed. The largest capacity
error after independent verification is 0.002654 bits.
CPU/GPU numerical discrepancies are retained and checked on GPU rather than
hidden by relaxed accuracy tolerances; see the audit artifacts for exact cases.

This rerun uses one common protocol across A/B/C. Comparison with the earlier
zero-dropout experiment is not a dropout-only causal ablation: capacity
rematching changes architecture sizes, and the earlier experiment also used a
different optimizer implementation, shuffling and stopping protocol. Neither
finite-budget failure nor a zero minimum over ten seeds establishes that a
model can never grok.

Artifacts: [summary](reports/p97_dropout_match/summary.json),
[capacity cells](reports/p97_dropout_match/capacity_cells.csv),
[all arithmetic outcomes](reports/p97_dropout_match/arithmetic_runs.csv),
[confirmation outcomes](reports/p97_dropout_match/confirmation_runs.csv),
[frozen selection](reports/p97_dropout_match/selection.json),
[stage provenance](reports/p97_dropout_match/provenance.json),
[checkpoint audits](reports/p97_dropout_match/checkpoint_audits.json),
[independent metric checks](reports/p97_dropout_match/independent_metrics_validation.json),
[learning curves](reports/p97_dropout_match/confirmation_accuracy.png),
[speed comparison](reports/p97_dropout_match/speed_comparison.png), and
[capacity curves](reports/p97_dropout_match/capacity_curves.png).
Each plot also has a PDF export in the same directory.

Recompute the report and audits with:

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src python scripts/analyze_p97_dropout_match.py --require-complete --audit --plots
python scripts/verify_p97_dropout_results.py
```

## Prospective p97 eight-level and categorical matched experiments — 25 October 2026

The user requests two new A/B/C comparisons and explicitly fixes the order:
choose p97 data first, find capacity-matched A/B, and only then construct
continuous C from the selected B. The dataset remains modular division at
50% training fraction: 4,656 training examples and 30,866.284423 bits. Common
training remains dropout0.2, two blocks, one head, AdamW1e-3/WD1/betas0.9,0.98,
batch512 and at most5,000 epochs (50,000 optimizer updates for arithmetic).
Registration: `configs/discrete_p97_multilevel_dropout.yaml`.

The scalar family uses eight evenly spaced fixed levels in [-1,1] at every
full-width block boundary, with the existing RMS-clipped STE. The categorical
family uses eight symbols per slot and L=d slots per token at every block:
learned d-to-(L*8) logits, nonaffine per-slot LayerNorm eps1e-4, hard argmax
one-hots, softmax STE at temperature1, then learned (L*8)-to-d re-embedding.
No continuous feature bypasses the hard state. Categorical training is
**deterministic**, following the language-regularisation interface; this is
explicitly different from the earlier sampled Dreamer-style runs. Earlier
sampled normalized p149 failures remain valid and are not overwritten.

A32/FF128 (39,264 parameters) already passed this unchanged arithmetic and
random-label protocol, with seed61/62 capacity31,288.3301/31,254.5454 bits.
Reuse those calibration artifacts with verified source/config/trace/checkpoint
hashes rather than estimate A from a fitted constant. Ten new A confirmation
seeds201–210 will be shared across the two experiments, counted as ten runs.
No prior confirmation outcome is used to tune either B.

Initial scalar development candidates are d36/FF144 and d44/FF176. Initial
categorical candidates are d36/FF144 and d48/FF192. Every exact candidate must
first reach sustained hard-eval held-out99% and final training-eval99%, then
learn N2048 random labels to final99% before any larger random-label cell.
Development seeds41/42; calibration seeds61/62 and N2048/4096/8192/16384.
N8192 seed61 may screen size before the rest of the grid. Retain all attempts.
Resize width/FF from development evidence only. Require both A/B within10%
of dataset and each other on both seeds. The selected capacity maxima must
be interior, below95% label entropy, under the epoch cap, and have <=1% late
drift. Preserve original patience-stop result and continue the exact live
optimizer/RNG in200-epoch segments when late drift exceeds1%, staying within
5,000 total epochs. B must be larger than A; do not require the binary
experiment's particular parameter ratio for the more expressive channels.

Freeze A/B before constructing or training any C. Scalar C removes quantizers
and exactly matches B's shapes/initial tensors. Categorical C is an ordinary
continuous transformer at the selected B width/depth; its FF width increases
by round(2*alphabet*dim/3), accounting for both categorical projection
matrices per block. This is exact when d is divisible by3, as in the initial
candidates; otherwise require <=0.1% parameter difference. It has no
categorical adapters, softmax channel or artificial bottleneck. Differing FF
shapes mean categorical B/C cannot have identical complete initial weights.
C is measured on the same random-label grid after construction, retaining
lower-bound flags and explicitly reporting whether excess capacity exists;
C outcomes never resize B. Confirmation uses all ten seeds201–210 for each
B/C and the shared ten A runs. Speed, sustained windows, paper/eval delay,
minimum-over-seeds and right-censoring rules match the completed binary study.

The new worker `multilevel_dropout.py` and launcher
`run_p97_multilevel_stage.py` preserve all earlier scientific source hashes.
The 38 targeted tests pass, including exact scalar training/RNG parity with
the completed protocol, eight-level and one-hot boundary checks in train/eval,
nonzero categorical gradients, deterministic hard evaluation, embedding-sum
identity, and unrestricted parameter-matched C construction. GPU1 is shared
with unrelated jobs; start with four small workers and monitor resources.
No scientific result from these new experiments exists at registration.

Development update (before capacity selection): categorical d48/FF192 passed
its arithmetic gate after356 epochs/3,560 updates, with100% eval-mode training
and99.6349% held-out accuracy; independent checkpoint reload verified both.
Its seed61 N2048 random-learning gate is running. The d36 categorical and
initial d36/d44 scalar pilots remain active, with substantially lower current
accuracy; none is declared a completed failure or a capacity result.

Register two additional shape probes using those development trajectories:
Q8 d64/FF64 and categorical d42/FF96, all channel/optimizer settings unchanged.
They test whether a wider residual stream with a smaller FF learns more
reliably at a similar parameter budget. Their seeds are41 and maximum budgets
remain5,000 epochs; retain the original pilots to their registered endpoints.
GPU0 has about22GiB allocated and8% sampled utilization from an unrelated job;
run only these two small additional workers there, preserving that process.
The gate/selection planner and report now have independent NumPy metric
recomputation and explicitly censored all-seed medians;39 targeted tests pass.

Further development: Q8 d44/FF176 passed at1,057 epochs/10,570 updates
(final train-eval99.6993%, held-out99.0550%). Categorical d42/FF96 passed at
369 epochs/3,690 updates (train-eval99.8282%, held-out99.3342%). The first
categorical d48/FF192 N2048 random gate passed at final99.8535%; its N8192
capacity screening and Q8 d44's N2048 random gate have started. These are
learning-gate outcomes, not matched-model confirmations.

`advance_p97_multilevel.py` advances only registered candidates. It independently
checks completed arithmetic checkpoints before random gates, enforces each
seed's N2048 gate, screens N8192, completes candidate grids, and freezes A/B
before constructing C. It never invents architecture candidates or changes
confirmed recipes. Capacity already above target+10% rejects a size; failed
random/arithmetic gates and incomplete/nonconverged matches request development
review. It uses at most four workers on GPU1 and two on GPU0 and preserves
unrelated processes. All preceding attempted runs finish before the fifty-run
confirmation queue (ten shared A plus twenty B and twenty C). Its source is
recorded at launch, and each immutable stage retains exact submitted jobs.

Additional development probes, registered before launch: intermediate N8192
traces for the successful initial B sizes are already around or above the
30,866-bit target, although none is yet a final capacity estimate. Add Q8
**d56/FF48** and categorical **d42/FF32**, retaining all channel/training
settings. These test smaller FF budgets while preserving enough residual
width to learn; they do not change the data or any confirmation choice.
Both receive the same seed41 arithmetic gate before random labels. Keep all
ongoing calibration to its original stop and report final values separately
from these provisional observations. GPU0 uses about24GiB of80GiB; permit
these two extra small pilots, temporarily bounding our concurrency at four
workers per GPU. Shared-device wall times remain descriptive only.

Capacity screening finished for the larger eight-level candidates: Q8
44/FF176 gives42,825.695 bits and64/FF64 gives41,331.828 bits at N8192,
seed61; both exceed the fixed target+10% and are rejected. Categorical
48/FF192 gives47,205.891 bits, also rejected. Q8 56/FF48 passed arithmetic
at1,524 epochs (train-eval99.9141%, held-out99.5919%) and its N2048 gate at
100%; categorical42/FF32 passed arithmetic at610 epochs and N2048 at99.4629%.

The latter two N8192 screening traces are approaching the upper matching
boundary before their final stops. Register additional smaller-FF probes
Q8 **d56/FF32** and categorical **d42/FF16**, with every other setting and
learning gate unchanged. Their purpose is to bracket a match without changing
p, split, dropout, alphabet or estimator. This is prospective adaptation from
development traces, not a final estimate or a confirmation outcome. No C has
been constructed. Two available worker slots keep total concurrency bounded.

The categorical42/FF32 final N8192 seed61 estimate is34,730.807 bits: above
the allowed33,952.913-bit upper bound, so it is excluded despite successful
learning. Its smaller42/FF16 arithmetic pilot has now passed and is entering
random validation. Q8 56/FF48's ongoing N8192 trace is also near the upper
bound, while the56/FF32 arithmetic pilot is still learning. Register Q8
**d56/FF40**, seed41, as an intermediate FF probe to bracket capacity without
reducing residual width. All prior runs continue and all protocol settings
remain fixed. Use one currently free GPU0 worker slot.

Interim audited evidence (not architecture acceptance):33 completed jobs
have passed independent checkpoint and metric checks. Q8 d56/FF48's N8192
screens are33,909.836/32,825.2 bits on seeds61/62, while its largest-N cells
remain active. Q8 d56/FF32's first N8192 screen is29,520.3 bits. Categorical
d42/FF16's first N8192 screen is33,582.2 bits. All these candidates first
passed their own arithmetic and random N2048 gates. Two larger-N Q8 cells
and the categorical seed61 largest-N cell use available GPU0 slots, bounded
at four workers there; only our queue controller was briefly paused to avoid
duplicate submission, with all training processes left running. The interim
report explicitly has `complete: false`, no A/B selection and zero
confirmation runs. Continuous C remains unconstructed until the full
registered calibration passes.

Selection publication recovery: the Q8 full grid passed every matching and
convergence check, but the controller failed while writing the selection
because a NumPy boolean was not JSON serializable. No C had been constructed
and no confirmation had started. Preserve the partial JSON, exception, old
controller log/launch and exact planner sources under
`diagnostics/selection_publication_recovery/`. All training jobs continued
unchanged. The planner now casts diagnostic values to native JSON types and
publishes selections, C construction and confirmation bindings atomically.
A regression test reproduces and protects publication; all40 targeted tests
pass. No scientific worker, optimizer, architecture or data source changed.
The validated Q8 A/B selection was then republished from the same immutable
capacity artifacts, before construction of C.

The publication regression now also exercises the complete confirmation-plan
binding with both families and exactly ten shared A runs (fifty trajectories
in total). It caught the same NumPy-boolean risk in the future C-excess flag;
that flag is converted to native bool before publication as well. All41
focused tests pass. Restart only the idle queue controller to load the tested
planner; preserve its previous launch/log and leave every training process
running. This is a proactive bookkeeping repair, not a training restart or
change in selection criteria.

## Frozen eight-level and categorical A/B selections

Both full N2048/4096/8192/16384 sweeps at seeds61/62 passed the registered
capacity criteria. Every selected maximum is at N8192, with <1% late drift,
no epoch-cap flag and no entropy-boundary flag. Every selected B passed
arithmetic and N2048 random-label learning first. The fixed p97 dataset still
contains4,656 training examples and30,866.284423 label-information bits.
A is the shared ordinary width32/FF128 transformer (39,264 parameters), with
measured capacity31,288.330057 and31,254.545389 bits.

| Family | B residual/FF width | B parameters | B capacity seed61 / seed62 (bits) | Constructed ordinary C residual/FF width | C parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eight-level scalar | 56 / 48 | 52,584 | 33,909.836404 / 32,825.219095 | 56 / 48 | 52,584 |
| Eight-symbol categorical | 42 / 16 | 83,118 | 33,582.242776 / 33,264.716043 | 42 / 240 | 83,118 |

Each C was constructed strictly after its own A/B selection was frozen.
Q8 C removes B's parameter-free quantizers and exactly matches its parameter
shapes and paired initial weights. Categorical B retains **42 categorical
slots**, each with8 symbols, at each token/block boundary; FF16 is only the
within-block feedforward hidden width, not a16-coordinate interlayer latent.
Categorical C remains an ordinary full-width42 continuous transformer. Its
FF240 accounts for the parameters in B's two learned interface projections.
Thus that comparison changes parameter allocation as well as discreteness;
complete categorical B/C initial tensors cannot be identical.

The selected categorical forward is deterministic hard argmax with normalized
softmax STE, temperature1, per-slot nonaffine LayerNorm and learned one-hot
re-embedding. The scalar grid is exactly8 evenly spaced values in[-1,1] with
RMS-clipped STE. No continuous bypass is present in either B. All models use
dropout0.2 and the same original optimizer protocol. The C models now undergo
their arithmetic, random-learning and full capacity checks; any lack of
measured excess C capacity will be reported explicitly and will not resize B.
The subsequent ten-seed runs201–210 remain unseen at this selection point.
The same ten new A trajectories will serve both comparisons.

Immutable selection/construction evidence has been copied into
`reports/p97_multilevel_dropout/{q8,categorical}_{AB_selection,C_construction}.json`.
`fill_p97_control_slots.py` can fill available worker slots with only the
already-authorized C calibration cells, after their own arithmetic/random
gates. It never constructs or resizes a model, reserves slots for pending
stages, and stops adding calibration jobs once confirmation is submitted.


Both C controls subsequently passed arithmetic and random-label gates. Their
capacity sweeps finished before the confirmation bindings were published. Q8 C
measures34,822.639505/34,140.472513 bits, with interior N8192 maxima; this is only
2.69%/4.01% above its paired B estimates, and12.82%/10.61% above the fixed dataset.
Categorical C measures at least55,972.625079/57,582.807941 bits. Both maxima are
at the largest tested N16384, so these are lower bounds rather than saturated
capacity estimates. The fifty fresh trajectories201–210 are now running with
frozen selections: ten shared A, ten of each B and ten of each C. There will be
no resizing, seed replacement or recipe changes based on these outcomes.


## Completed p97 eight-level and categorical matched experiments

Both experiments are complete on `idea/discrete`, with dropout 0.2 and fresh
seeds 201–210. The order was fixed dataset, validated/capacity-matched A and B,
frozen A/B selection, construction of ordinary continuous C from B, C validation
and capacity measurement, then confirmation. The independent ordering audit
verified all 115 job manifests and all 51 completed stages, including all 49
random-label cells' prerequisite learning gates. No confirmation outcome was
used to choose or resize a model. All 50 confirmation trajectories are retained;
the ten A trajectories are shared between comparisons, not independent duplicates.

The fixed task is p=97 modular division, with a 50% split and 4,656 training
examples, giving the registered target of 30,866.284423 bits = 4,656 log2(99).
Each seed has a paired split shared by A/B/C; the ten seeds are not the same
individual split. Every selected discrete recipe first passed its own arithmetic
gate, then its N=2,048 random-label gate at both calibration seeds. The full
capacity grid uses N=2,048/4,096/8,192/16,384 and seeds 61/62. All A/B maxima
are interior at N=8,192, have late drift below 1%, and match both the target and
each other within 10% on each seed. A's same-protocol calibration was reused
with explicit source/configuration/artifact verification.

| Model | Residual / FF width | Parameters | Capacity seed 61 / seed 62 (bits) |
| --- | ---: | ---: | ---: |
| A, ordinary continuous (shared) | 32 / 128 | 39,264 | 31,288.33 / 31,254.55 |
| Eight-level B | 56 / 48 | 52,584 | 33,909.84 / 32,825.22 |
| Eight-level C | 56 / 48 | 52,584 | 34,822.64 / 34,140.47 |
| Categorical B | 42 / 16 | 83,118 | 33,582.24 / 33,264.72 |
| Categorical C | 42 / 240 | 83,118 | ≥55,972.63 / ≥57,582.81 |

Eight-level B quantizes every native residual coordinate at each block boundary
to eight equally spaced values in [-1,1], using the registered RMS-clipped STE.
Its C removes those parameter-free quantizers; B/C have exactly the same
parameter shapes and paired initial tensors. Their measured capacities differ
by only 2.69% and 4.01%, so this is a weak capacity contrast.

Categorical B uses 42 categorical slots per token at each block boundary, each
with eight symbols: learned logits, per-slot nonaffine LayerNorm before softmax,
hard argmax one-hots in the forward pass, and softmax STE in the backward pass.
The learned 336-to-42 consumer is the sum of slot-specific symbol embeddings.
There is no continuous bypass and no 16-coordinate latent: FF16 is only the
within-block feedforward hidden width. Training is deterministic argmax with
normalized-softmax STE, not categorical sampling. C is an ordinary continuous
width-42 transformer with FF240, accounting for B's categorical projection
parameters. This comparison changes parameter allocation as well as
discreteness, and complete B/C initial tensors are not identical. C's largest-N
maxima are lower bounds; they already establish substantially higher capacity.

All times below are **optimizer updates**, with ten updates per epoch. Sustained
accuracy means three consecutive end-of-epoch held-out evaluations. Each run
stops when that held-out 99% condition and current dropout-active training 99%
hold, otherwise at 5,000 epochs / 50,000 updates.

| Model | Sustained 95% successes | Median updates to 95%, all ten | Sustained 99% successes | Median updates to 99%, all ten |
| --- | ---: | ---: | ---: | ---: |
| A (shared) | 9/10 | 6,775 | 8/10 | 11,660 |
| Eight-level B | 7/10 | 33,915 | 5/10 | >49,205 (censored) |
| Eight-level C | 10/10 | 4,935 | 10/10 | 9,390 |
| Categorical B | 10/10 | 3,165 | 10/10 | 4,035 |
| Categorical C | 10/10 | 2,925 | 10/10 | 3,235 |

For eight-level B, five 99% crossings are observed and five are right-censored
at 50,000 updates. The fifth observed crossing is 48,410 updates; the sixth is
unobserved, so the conventional ten-seed sample median is >49,205 updates,
not a success-only median. Final held-out 99% counts equal the sustained-success
counts for all five model groups.

At 99%, eight-level C beats B on all ten paired seeds: C is faster in the five
pairs where both succeed and succeeds in the other five where B does not. A
is faster than B on four mutually successful pairs; A alone succeeds on four,
B alone on one, and neither on one. A versus eight-level C is mixed: A is faster
on three, C on five, and C alone succeeds on two.

Categorical B reaches 99% before A on seven mutually successful pairs, A wins
one, and B alone succeeds on two. Categorical C beats B on nine seeds, with B
faster on one; C beats A on all eight mutually successful seeds and succeeds
on the two A failures. At 95%, B beats categorical C on three seeds and C wins
seven. Thus categorical B improves speed and reliability relative to matched A,
but does not improve on its ordinary continuous C control. Eight-level B is
substantially slower and less reliable than C. These are capacity-rematched
architectures, not a precision-only ablation of the earlier binary result.

The registered paper-style delay is max(0, first held-out 98% update minus
first dropout-active online training 99% update). The secondary delay uses
end-of-epoch dropout-free training accuracy. These first-crossing delays are
distinct from the sustained-accuracy speed metric. Missing thresholds stay
undefined. The requested minimum across ten seeds is reported alongside the
positive / zero / undefined counts, so a zero minimum does not imply that
every seed has zero delay or that an unsuccessful model avoids grokking.

| Model | Paper-style minimum (updates) | Positive / zero / undefined | Eval-training minimum (updates) | Positive / zero / undefined |
| --- | ---: | ---: | ---: | ---: |
| A (shared) | 0 | 1 / 8 / 1 | 0 | 7 / 2 / 1 |
| Eight-level B | 0 | 1 / 4 / 5 | 380 | 6 / 0 / 4 |
| Eight-level C | 0 | 0 / 10 / 0 | 120 | 10 / 0 / 0 |
| Categorical B | 0 | 8 / 2 / 0 | 110 | 10 / 0 / 0 |
| Categorical C | 0 | 7 / 3 / 0 | 40 | 10 / 0 / 0 |

A still has positive measured delay on one paper-style seed and seven
eval-training seeds. Categorical B has positive paper-style delay on eight
seeds and eval-training delay on all ten, with a minimum of 110 updates under
the latter definition. Eight-level B has positive eval-training delay on all
six defined seeds, minimum 380 updates, with four undefined. Neither discrete
recipe supports a universal claim of never grokking. These finite-budget
capacity estimates and threshold delays also do not establish intrinsic
capacity bounds or a theorem that dataset-matched A cannot grok.

All 115 new final checkpoints passed independent reload/evaluation audits,
including all 65 development/calibration jobs and 50 confirmations. Every
initial checkpoint and dataset was reconstructed; all 66 arithmetic traces
had thresholds recomputed with an independent NumPy implementation, and all
20 family/seed pairings passed data, parameter, initialization, and frozen
configuration checks. The largest capacity recomputation error was 0.035624
bits; CPU/GPU discrepancies are retained and verified on GPU at the original
strict tolerances. The 41 focused tests passed. Failed pilots and rejected
capacity candidates remain in the exported arithmetic/capacity tables. The
previously documented metadata publication repairs changed no scientific
worker, model, optimizer, dataset or training trajectory.

Artifacts: [all 50 outcomes](reports/p97_multilevel_dropout/confirmation_runs.csv),
[full summary](reports/p97_multilevel_dropout/summary.json),
[checkpoint validation](reports/p97_multilevel_dropout/validation.json), and
[ordering/stage audit](reports/p97_multilevel_dropout/ordering_validation.json).
Figures are exported as both PNG and PDF:
[eight-level speed](reports/p97_multilevel_dropout/q8_speed.png),
[categorical speed](reports/p97_multilevel_dropout/categorical_speed.png),
[eight-level capacity](reports/p97_multilevel_dropout/q8_capacity.png),
[categorical capacity](reports/p97_multilevel_dropout/categorical_capacity.png),
[eight-level train/test](reports/p97_multilevel_dropout/q8_train_test.png), and
[categorical train/test](reports/p97_multilevel_dropout/categorical_train_test.png).


## Registered C-first categorical grokking test (2026-10-26)

The user now requests a continuous C that groks by the paper's minimum-across-
ten-seeds criterion, followed by its categorical B equivalent. This is a new
C-first selection question; the previous capacity-matched ABC results remain
complete and unchanged. Keep p=97 modular division, the 50% split, dropout 0.2,
and the original optimizer/batch/epoch protocol. There is no new A or claim that
the selected B matches the dataset's empirical capacity.

Prospectively search C widths 60, 72, 96, 120, sequentially. For width d, C is
an ordinary continuous transformer with FF=28d/3. Its eventual categorical B
has the same width/depth and FF=4d, plus the existing eight-symbol normalized
hard-argmax/softmax-STE interfaces. These configurations have exactly equal
parameter counts (242,580; 346,392; 609,312; 945,960 respectively). The parameter
allocation and full initial tensor shapes differ; this remains an ordinary C
comparison rather than an identical-backbone intervention.

Each C must first pass all ten search seeds 301–310, then all ten fresh
validation seeds 321–330: successful arithmetic learning plus a strictly
positive, defined online-train99 to held-out98 delay on every seed. Retain all
attempts and choose the first candidate passing both groups. Freeze C and the
corresponding B architecture before any B training; B results cannot resize C.
This conditional search is not an unbiased estimate of how often arbitrary
continuous controls grok. The B comparison uses the same validation seeds,
paired arithmetic splits and unchanged settings.

After arithmetic validation, measure both models' memorisation time on exactly
4,656 uniform-random input/label examples at seeds 321–330. Stop at the first
dropout-active online training accuracy >=99%, or 5,000 epochs; do not apply
capacity-loss patience. The new loop reuses original CapacityTrainer updates,
shuffling and optimizer state, with a regression test requiring exact agreement
of parameters, accuracy traces and RNG with the original multi-epoch loop.
A B recipe that fails every arithmetic gate gets no random-label jobs; failure
to learn cannot establish successful avoidance of grokking. All missing
thresholds remain censored/undefined.

The preceding categorical-versus-eight-level comparison does not isolate why
categorical B was faster: its parameter count was 83,118 versus 52,584, its
parameter allocation differed, and its continuous C was also faster. Learned
symbol embeddings and the normalized-softmax gradient are plausible mechanisms,
but their individual effects have not been experimentally isolated.

Registration: `configs/discrete_p97_categorical_groks.yaml`; worker:
`src/grokking_capacity/experiments/categorical_groks.py`; durable campaign:
`scripts/run_p97_categorical_groks.py`. Earlier scientific sources remain frozen.


## Completed C-first categorical grokking test

The registered C-first follow-up is complete on `idea/discrete`. The first
candidate, ordinary continuous C60/FF560, passed all ten search seeds 301–310
and all ten fresh validation seeds 321–330. No wider candidate was needed.
C and its B equivalent were frozen before any B training. Both use depth two,
one attention head, dropout 0.2, batch 512, and the registered AdamW settings
(lr 0.001, weight decay 1, betas 0.9/0.98). The task remains p=97 modular
division with a 50% split: 4,656 training examples and 4,656 held-out examples.
All times below are optimizer updates; one epoch is ten updates.

| Model | Residual / FF width | Parameters | Boundary representation |
| --- | ---: | ---: | --- |
| C | 60 / 560 | 242,580 | Ordinary fully continuous transformer |
| B | 60 / 240 | 242,580 | 60 categorical slots, each with eight symbols |

B uses learned logits, per-slot nonaffine LayerNorm before softmax, hard argmax
one-hots in the forward pass, and normalized-softmax STE in the backward pass.
The flattened 60-by-8 one-hot state is mapped back to width 60 by a learned sum
of slot-specific symbol embeddings. No continuous state bypasses this boundary.
This uses deterministic argmax, not sampled categories. C has no categorical
interface or bottleneck. Exact parameter matching reallocates the interface
parameters into C's feedforward layers; the full initial tensor shapes differ.
This is not an identical-backbone intervention. Initialization is identical
across arithmetic/random tasks within each role/seed; C/B use paired datasets
within each task. There is no A or new dataset-capacity match claim.

The primary delay follows the previously verified paper convention:
max(0, first held-out 98% update minus first dropout-active online training 99%
update). The criterion here is successful arithmetic learning and a strictly
positive, defined delay on all ten seeds, so the minimum across ten is positive.
The [paper, sections 4.2–4.3](https://arxiv.org/html/2605.09724v2) distinguishes
these first crossings from sustained-accuracy measurements.

| Validation metric, seeds 321–330 | C | B |
| --- | ---: | ---: |
| Successful arithmetic learning | 10/10 | 10/10 |
| Mean first held-out 99% update | 2,559 | 4,559 |
| Median first held-out 99% update | 2,365 | 3,625 |
| Mean sustained held-out 99% update | 2,578 | 5,002 |
| Median sustained held-out 99% update | 2,365 | 3,645 |
| Paper-style minimum delay | 150 | 0 |
| Paper-style positive / zero / undefined | 10 / 0 / 0 | 8 / 2 / 0 |
| Paper-style mean delay | 319 | 459 |
| Paper-style median delay | 270 | 200 |
| Secondary eval-training minimum delay | 200 | 150 |
| Secondary positive / zero / undefined | 10 / 0 / 0 | 10 / 0 / 0 |

The selected C therefore meets the requested paper-style criterion. Its search
group also has ten positive delays, minimum 190 updates. B fails the minimum
criterion because seeds 324 and 326 have zero delay, but B still has positive
paper-style delay on eight seeds. Its mean delay is larger than C's. Using
end-of-epoch dropout-free training accuracy instead, both models have positive
delay on all ten seeds. The categorical interface does not universally remove
grokking, and the zero minimum should not conceal the other eight outcomes or
the dependence on the training-accuracy measurement. These are small positive
threshold delays, not evidence of a long post-memorisation plateau. C was
selected for this behavior; the experiment does not estimate grokking frequency
for arbitrary, unselected continuous models.

Only after all arithmetic validation finished were the matched random-label
experiments run. Each has exactly 4,656 original uniform-random input/label
examples over 99 symbols, the same label-information target of 30,866.284423
bits. Both roles use the same ten seeds 321–330 and fresh optimization from
initial weights. The stop is first dropout-active online training accuracy
>=99%, otherwise 5,000 epochs / 50,000 updates. There is no loss-patience stop.

| Random-label measurement | C | B |
| --- | ---: | ---: |
| Online 99% crossings | 10/10 | 0/10 |
| Mean updates to online 99% | 1,230 | >50,000, all ten censored |
| Observed crossing range | 1,190–1,290 | None |
| Final online accuracy range | 99.01–99.21% | 91.62–94.65% |
| Final dropout-free accuracy range | 99.85–100% | 98.26–99.38% |
| Final dropout-free accuracy >=99% | 10/10 | 6/10 |

C memorises the matched random task before its mean arithmetic generalisation
crossing. For B, the required online threshold is unobserved, so neither a
finite mean of 50,000 nor an infinite time is claimed. B has learned most
random labels and six checkpoints exceed 99% without dropout. These results
measure online memorisation under dropout; they do not prove that B cannot
memorise or establish an intrinsic capacity bound. No claim that B's empirical
capacity equals the dataset size follows from this test.

Why the earlier categorical model outperformed eight-level scalar quantisation
remains unresolved. Categorical B had 83,118 parameters versus scalar B's
52,584, and the residual widths, feedforward allocations, boundary encoders,
and gradient estimators differed. The corresponding categorical C was also
faster, showing that the surrounding architecture matters. At a linear
consumer, one scalar slot contributes q[k] times a single learned vector;
one categorical slot can select an independently learned vector e[k]. The
latter avoids imposing the scalar grid's ordering and collinearity on the
symbols. The normalized-softmax STE also differs from the scalar RMS-clipped
STE. These are plausible explanations, not experimentally isolated causes.
The new C/B pair tests a selected continuous control against its categorical
equivalent, not categorical versus scalar causality.

All 50 final checkpoints passed independent reload/evaluation audits, including
initial-weight and dataset reconstruction and train/eval boundary checks.
All 30 arithmetic traces and 20 memorisation crossing times were recomputed;
all 250 input artifact hashes and the frozen scientific sources were verified.
The stage order is C search, fresh C validation, frozen selection, B validation,
then matched memorisation. All 43 focused tests passed before training, including
exact optimizer/RNG/trajectory equivalence of the one-epoch memorisation loop.
No training source, hyperparameter, seed, or stopping rule changed during the
campaign. The analysis independently checks stage ordering, paired datasets,
parameter counts, initial weights, and every retained seed.

Artifacts: [all arithmetic runs](reports/p97_categorical_groks/arithmetic_runs.csv),
[paired validation outcomes](reports/p97_categorical_groks/paired_outcomes.csv),
[random-label outcomes](reports/p97_categorical_groks/memorisation_runs.csv),
[summary](reports/p97_categorical_groks/summary.json), and
[validation](reports/p97_categorical_groks/validation.json).
Figures are exported as PNG and PDF:
[arithmetic learning curves](reports/p97_categorical_groks/train_test.png),
[first-crossing times](reports/p97_categorical_groks/learning_times.png),
[per-seed delays](reports/p97_categorical_groks/grokking_delays.png), and
[random-label learning curves](reports/p97_categorical_groks/random_learning.png).

## Registered capacity-constant matched scaling sweep (2026-10-26)

The user requested ten model pairs to explore how generalisation-time scaling
differs between continuous C and categorical B, then specified empirical
memorisation-capacity matching via parameter count and a capacity constant.
This is a new scaling question. Use one separately calibrated constant per
family, estimated capacity K_hat = kappa times parameter count, rather than
equal parameter counts or a fresh capacity search at every sweep point.

Keep p=97 division, 50% training split, depth two, one head, dropout 0.2,
batch 512, AdamW lr 0.001 / weight decay 1 / betas 0.9 and 0.98. Both families
have FF width 4d throughout, avoiding the earlier comparisons' varying FF
proportions. C is ordinary and fully continuous. B retains full-width,
eight-symbol categorical states with nonaffine per-slot pre-softmax LayerNorm,
deterministic argmax/normalized-softmax STE, and learned embedding sums.

Calibrate C at widths 32 and 48, and B at widths 48 and 60, using random-data
seeds 61/62. Reuse only compatible audited C32 and B48 cells after rechecking
frozen sources, configurations, data, initial tensors and checkpoint hashes.
C32 is the earlier ordinary A32 model, explicitly reused under role C for this
new comparison. Each recipe must pass arithmetic before random-label training;
each model/seed must learn N=2,048 before larger random datasets. The new grid
is N=2,048/8,192/16,384/32,768; the existing C32 grid additionally has N=4,096
and already has an interior maximum at N=8,192. Extend a largest-N maximum by
doubling N, at most to 131,072. A capacity point must be an interior maximum,
below 95% label entropy, below the 5,000-epoch ceiling, with late drift <=1%.
Use the original online-loss patience and live optimizer/RNG continuation.

Fit each family's constant through the origin to its four architecture/seed
maxima: kappa = sum(P*K)/sum(P^2). Keep both seeds; do not select the better
seed. Report affine-fit diagnostics, residuals, per-seed constant sensitivity,
and the calibrated parameter range. This replaces extrapolating the earlier
mostly p=113/small-model continuous calibration and supplies the missing
categorical constant for the fixed FF4d architecture family.

C widths are fixed at 32, 36, 40, 44, 48, 52, 56, 60, 64 and 68. Choose each
B width from even integers 24–160 by nearest estimated capacity; require ten
distinct increasing B widths and <=5% predicted mismatch. Freeze constants,
all ten pairs and their calibration evidence before confirmation. These are
estimated empirical-capacity matches under a constant approximation. They do
not establish directly measured capacity equality at all ten points. Report
fit errors and extrapolation, including residuals over 10%; do not hide them.

Run fresh seeds 401–410 for both roles at all ten sizes: 200 arithmetic runs.
Every role/size shares the same seed's data split. Different model shapes mean
full initial tensors and subsequent shuffle/dropout RNG are not identical.
No selection on grokking or generalisation outcomes is allowed. Use the
original joint stopping rule: three held-out 99% epochs plus current online
training 99%, otherwise 5,000 epochs / 50,000 optimizer updates.

Primary t_gen is the first held-out 99% crossing, in optimizer updates. Retain
all censored outcomes; means are complete only when all ten crossings are
observed, otherwise report lower bounds. Also export medians, sustained 99%,
success counts, online/eval-training delays and all individual traces. Plot
against both estimated capacity and actual parameter count. Fit descriptive
per-seed log-log slopes only on common size points with all ten C and B times
observed (minimum three points); use paired seed bootstrap for slope intervals
and their difference. Explicitly label this complete-range conditioning. A
separate all-size capped-time analysis uses min(t_gen, 50,000) and must not be
labelled an uncensored power law.

Registration: `configs/discrete_p97_capacity_scaling.yaml`; worker:
`src/grokking_capacity/experiments/capacity_scaling.py`; durable controller:
`scripts/run_p97_capacity_scaling.py`. The 20 focused tests passed before
launch. A descriptive YAML scalar needed quoting; the first startup failed
before source freezing or any training. Its empty run root and log are preserved
in `results/discrete/p97-capacity-scaling-v1-registration-parse-failure`. GPU0 runs unrelated PID1637741 and is left alone; use up to eight
small workers on the idle GPU1. Prior scientific sources and results remain
unchanged. Run root: `results/discrete/p97-capacity-scaling-v1`.


Before pair selection or confirmation, the analysis was extended to also fit
log of the arithmetic-mean t_gen at each size, with paired seed bootstrap
before averaging. The registered per-seed log-slope analysis describes the
geometric-mean trend and remains unchanged; both are needed when tail behavior
varies with size. The timestamped addition is saved in the run root as
`analysis_plan_extension.json`. Four focused analysis tests passed, including
a case where arithmetic-mean and geometric-mean scaling differ.


Calibration finished with all 32 cells and eight acceptable interior maxima.
During checkpoint auditing, a CPU recomputation for B48/N32768/seed61 differed
by 0.207 bit from the recorded 30,846.312 bits; on the original GPU the
difference was 0.000874 bit and the full audit passed. The controller stopped
before selection. A separate recovery script audited the remaining calibration
checkpoints on GPU1 and executed the original frozen controller's remaining
statements. No training was restarted or changed. The original error and
recovery hashes are retained in the run's `audit_recovery` directory.
Constants and all ten pairs were then frozen before confirmation began.

## Completed capacity-constant matched scaling sweep

The ten C/B pairs were frozen from calibration before seeds 401-410 were
trained. All 200 confirmation runs finished on p=97 modular division with a
50% split (4,656 training examples, 30,866.284 bits under the registered
log2(99) convention), dropout 0.2, depth two, one head, FF width 4d and the
registered AdamW settings. C is an ordinary fully continuous transformer.
B uses a full-width eight-symbol categorical representation, per-slot
pre-softmax LayerNorm, hard one-hot forward values, normalized-softmax STE
backward gradients and learned embedding sums.

Capacity matching uses K_hat = kappa * P, with one constant per family fitted
through the origin to four architecture/seed maxima. Every calibration recipe
passed arithmetic before random-label training; all 32 calibration cells
finished, and each of the eight selected maxima was interior, below the
registered entropy/epoch limits and stable to <=1% late drift. The constants
are empirical under this optimizer/dropout protocol, not intrinsic capacity
bounds. The ten sweep points are predicted matches, not direct capacity
measurements at every size.

| Family | Bits per parameter | Calibrated parameter range | Maximum relative fit residual |
| --- | ---: | ---: | ---: |
| C | 0.729116935 | 39,264-83,472 | 9.29% |
| B | 0.310917024 | 157,200-242,580 | 5.09% |

B requires approximately 2.345 times C's parameter count for an equal prediction.
All ten pooled-constant matches are within 4.06%; using the two
seed-specific constant estimates gives a worst mismatch of 5.00%.
These rounding/sensitivity figures do not include model misspecification.
Only C44/B50 and C48/B54 lie inside both calibration ranges; the other pairs
use extrapolation for at least one member. Affine diagnostics, individual
maxima and seed-specific constants are retained in the report.

| C/B widths | C parameters | B parameters | C estimated bits | B estimated bits | B relative mismatch |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32/36 | 39,264 | 90,252 | 28,628 | 28,061 | -1.98% |
| 36/40 | 48,780 | 110,520 | 35,566 | 34,363 | -3.38% |
| 40/46 | 59,320 | 144,762 | 43,251 | 45,009 | +4.06% |
| 44/50 | 70,884 | 170,150 | 51,683 | 52,903 | +2.36% |
| 48/54 | 83,472 | 197,586 | 60,861 | 61,433 | +0.94% |
| 52/58 | 97,084 | 227,070 | 70,786 | 70,600 | -0.26% |
| 56/62 | 111,720 | 258,602 | 81,457 | 80,404 | -1.29% |
| 60/66 | 127,380 | 292,182 | 92,875 | 90,844 | -2.19% |
| 64/72 | 144,064 | 346,392 | 105,040 | 107,699 | +2.53% |
| 68/76 | 161,772 | 385,092 | 117,951 | 119,732 | +1.51% |

Primary t_gen is the first held-out 99% crossing, measured in actual optimizer
updates: ten updates per epoch. The training stop additionally requires three
consecutive held-out 99% epochs and current dropout-active online training
99%; the ceiling is 5,000 epochs / 50,000 updates. Missing crossings are
right-censored. A > sign denotes a lower bound on the all-seed mean or median;
no successful-seed-only average is substituted for an all-seed mean.

| C/B widths | Mean t_gen C | Mean t_gen B | Median C | Median B | First 99% C/B (of 10 each) | Final learning C/B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32/36 | >20,553 | >43,703 | 15,030 | >50,000 | 9/2 | 9/1 |
| 36/40 | 5,762 | >26,496 | 4,485 | 23,430 | 10/7 | 10/5 |
| 40/46 | 5,224 | >14,874 | 3,220 | 6,340 | 10/8 | 10/8 |
| 44/50 | 3,395 | >18,938 | 3,190 | 6,915 | 10/7 | 10/7 |
| 48/54 | 3,304 | 7,551 | 3,065 | 4,950 | 10/10 | 10/10 |
| 52/58 | 3,022 | 3,715 | 2,810 | 3,750 | 10/10 | 10/10 |
| 56/62 | 2,688 | 4,430 | 2,705 | 3,935 | 10/10 | 10/10 |
| 60/66 | 2,557 | 3,616 | 2,530 | 3,440 | 10/10 | 10/10 |
| 64/72 | 2,440 | 3,281 | 2,460 | 2,910 | 10/10 | 10/10 |
| 68/76 | 2,379 | 3,168 | 2,335 | 3,030 | 10/10 | 10/10 |

C has 99/100 first crossings and 99/100 final learning checks passed;
B has 84/100 first crossings and 81/100 final learning checks passed.
The stricter joint stopping rule was met by C 99/100 and B 81/100.
C is faster in 86 paired comparisons, B in 13; 1 pair is censored for
both and cannot be ordered. The true means at the smallest pair cannot be
ordered from their lower bounds alone. At the largest pair, mean times are
C 2,379 and B 3,168, with C faster on nine of ten paired seeds.

Three B runs crossed 99% but did not pass the final learning check. Their
observed t_gen remains in the primary analysis, and their final failures are
reported separately:

| Model / seed | First 99% updates | Final held-out accuracy | Final dropout-free training accuracy |
| --- | ---: | ---: | ---: |
| B36 / 405 | 29,550 | 96.18% | 97.79% |
| B40 / 407 | 35,310 | 95.25% | 96.80% |
| B40 / 410 | 48,240 | 97.21% | 98.17% |

The scaling fit t_gen proportional to K_hat^beta includes an intercept in log
space. The six common fully observed points are pairs 4-9, with C capacity
targets 60,861-117,951 bits. All ten seeds are retained at each point. The
additional arithmetic-mean analysis was recorded before pair selection and
confirmation; the originally registered per-seed log-slope analysis is also
reported. Intervals use 10,000 paired seed bootstrap draws, seed 9001.

| Analysis | C beta [95% interval] | B beta [95% interval] | B minus C [95% interval] |
| --- | ---: | ---: | ---: |
| Mean t_gen, six fully observed sizes | -0.509 [-0.707, -0.302] | -1.043 [-1.714, -0.374] | -0.534 [-1.300, 0.256] |
| Per-seed log t_gen, same six sizes | -0.483 [-0.673, -0.293] | -0.781 [-1.314, -0.361] | -0.298 [-0.916, 0.242] |
| Mean capped time, all ten sizes | -1.216 [-1.411, -0.970] | -1.955 [-2.180, -1.644] | -0.739 [-1.056, -0.367] |
| Per-seed log capped time, all ten sizes | -1.085 [-1.269, -0.920] | -1.661 [-1.960, -1.391] | -0.576 [-0.900, -0.236] |

Both families become faster across the six fully observed sizes. B's fitted
mean-time decline is steeper, but the paired interval for the slope difference
includes zero; these ten seeds do not resolve different uncensored exponents.
The registered per-seed log analysis gives the same uncertainty conclusion.
B remains slower in mean at every one of these six matched-capacity points.
Its mean is more affected by slow runs, including B54/seed404 at 30,180 updates;
the mean and geometric-mean trends therefore differ.

Across all ten sizes, the capped-time curve is clearly steeper for B. This is
a descriptive result for min(t_gen, 50,000), reflecting more failures and long
runs at small sizes. It is not an uncensored power law for the missing times.
Neither categorical mean times nor medians decrease at every adjacent size.
Bootstrap intervals condition on the fitted constants and frozen architectures;
calibration error and extrapolation limit the accuracy of the matching.
Since each family uses a single constant, its slope versus estimated capacity
is identical to its slope versus parameter count.

The same seed's data split is shared across every role and size. Different
model shapes consume initialization RNG differently, so full initial weights
and subsequent shuffle/dropout RNG streams are not identical. Selection used
capacity predictions before confirmation, with no generalisation-based choice
of C or post-result resizing. t_gen is not the paper-style grokking delay:
online-train99-to-held-out98 and dropout-free counterparts, including zero and
undefined delays, are exported for every run. An observed minimum with missing
seeds must not be described as a ten-seed minimum-delay result.

All 235 final checkpoints passed reload/evaluation audits, including initial
weights, dataset reconstruction, dropout and categorical-boundary checks. The
report recomputes 203 arithmetic traces and verifies 1223 input artifact hashes.
The independent verifier checks all 200 first crossings, final learning
outcomes and first stopping epochs, all eight calibration maxima, both scalar
constant fits, all ten nearest-capacity choices and both forms of log-log fit.
Twenty focused training/protocol tests passed before launch; four statistical
report tests passed, including censoring and arithmetic/geometric mean cases.

The earlier CPU-only calibration audit discrepancy was resolved on the
training GPU before selection, with no restarted training. Its original
failure, exact numerical discrepancy, recovery source hash and timestamps
are retained in [audit recovery](reports/p97_capacity_scaling/audit_recovery.json).
The frozen scientific sources match commit `56ed10f`; final artifact and
analysis hashes are recorded in
[completion provenance](reports/p97_capacity_scaling/completion_provenance.json).

Artifacts: [all 200 confirmation runs](reports/p97_capacity_scaling/confirmation_runs.csv),
[ten-point summary](reports/p97_capacity_scaling/size_summary.csv),
[paired outcomes](reports/p97_capacity_scaling/paired_outcomes.csv),
[32 calibration cells](reports/p97_capacity_scaling/capacity_cells.csv),
[complete summary](reports/p97_capacity_scaling/summary.json), and
[independent verification](reports/p97_capacity_scaling/independent_validation.json).
Figures are available as PNG and PDF:
[scaling](reports/p97_capacity_scaling/scaling.png),
[capacity calibration](reports/p97_capacity_scaling/calibration.png),
[crossings and paired speeds](reports/p97_capacity_scaling/success_and_pairs.png),
and [all seed trajectories](reports/p97_capacity_scaling/individual_times.png).

## Dataset-matched logarithmic sweep registration

Registered on 2026-10-26 before training. The user requested checking whether
continuous C and categorical B generalisation-time curves cross with
logarithmically sampled dataset sizes, then explicitly requested resizing both
models to each dataset via the existing capacity constants. The fixed-pair
proposal was never launched. This experiment grows the training dataset and
predicted model capacity together on p=97 modular division.

For each N, target D=N*log2(99) bits, then select the even model width closest
to kappa_role*P_role=D. The independently calibrated constants remain
kappa_C=0.7291169353321082 and kappa_B=0.31091702370607593 bits/parameter.
The unchanged depth-two, one-head, FF=4d model families have
P_C=32d^2+203d and P_B=64d^2+203d. C is ordinary and fully continuous;
B has d categorical slots per token at every block boundary, eight symbols,
per-slot pre-softmax LayerNorm, hard argmax forward, softmax STE backward,
and learned embedding sums with no bypass.

The endpoints 928 and 7,449 permit ten rounded logarithmic dataset sizes
while keeping both role-to-dataset and B-to-C prediction errors below 10%
without altering FF ratios. Endpoints and model widths were chosen from
parameter arithmetic only, before observing any outcomes. The unlaunched
512-example endpoint could not meet that tolerance for C with even widths.

| Train N | Dataset bits | C width | B width | C prediction error | B prediction error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 928 | 6,152 | 14 | 16 | +8.02% | -0.78% |
| 1,170 | 7,756 | 16 | 18 | +7.54% | -2.23% |
| 1,474 | 9,772 | 18 | 20 | +4.63% | -5.63% |
| 1,858 | 12,317 | 20 | 24 | -0.20% | +5.35% |
| 2,342 | 15,526 | 22 | 26 | -6.29% | -2.79% |
| 2,952 | 19,570 | 26 | 30 | +0.26% | +1.19% |
| 3,720 | 24,661 | 30 | 34 | +3.15% | +1.98% |
| 4,689 | 31,085 | 34 | 38 | +2.96% | +0.15% |
| 5,910 | 39,179 | 38 | 42 | +0.35% | -3.64% |
| 7,449 | 49,382 | 42 | 48 | -4.07% | -1.02% |

The maximum model-to-dataset prediction error is 8.02%; the maximum B-to-C
prediction mismatch is 9.80%. These are capacity predictions, not direct
empirical measurements at every width. All C widths below 32 and B widths
below 48 extrapolate below the calibration ranges. The calibration residuals
remain 9.29% for C and 5.09% for B; rounding error does not include this model
uncertainty. No model is selected based on arithmetic success, and no new
random-label training is needed to apply the existing constants.

Seeds 501-510 each define one permutation of all 9,312 modular input pairs.
The first N examples form nested training subsets; the last 1,863 examples
form one fixed held-out set shared by every role and size within that seed.
Unused training-pool entries are excluded from evaluation. Every size/role
gets all ten fresh seeds, giving 200 runs. Dropout remains 0.2, batch size
512, original AdamW lr0.001/wd1/betas(0.9,0.98), with reshuffling each epoch
and partial final batches retained.

Evaluate every ten actual optimizer updates, independently of epoch
boundaries. Online training metrics cover the preceding ten updates weighted
by examples. Stop at the first evaluation with three consecutive held-out99
checks and current online training99, otherwise exactly 50,000 updates.
Primary t_gen is first held-out99; also retain sustained99, final learning,
all-seed/minimum paper-style positive-delay results, examples processed and
epoch equivalents. Fresh test splits and evaluation windows mean these times
must not be pooled with the previous 50%-split campaign.

An observed mean-curve crossing requires opposite signs of mean(B)-mean(C)
at adjacent sizes with all twenty first crossings observed. Any log-N
interpolation is descriptive and stays inside that bracket. Missing crossings
remain right-censored; capped times min(t_gen,50000) are a separate sensitivity
analysis, never substituted for uncensored means. Ten-seed paired bootstrap
(10,000 resamples, seed9002) retains cross-size pairing, produces pointwise95%
intervals and simultaneous95% centered maximum-deviation bands across all ten
capped contrasts. A resolved crossing requires opposite signs of simultaneous
bands at fully observed sizes. Failure to observe a crossing is limited to
this range, these capacity predictions and the update budget.

Scientific code, selection, registry and calibration evidence will be hashed
before launch. Every final checkpoint is independently reloaded, data and
initialization reproduced, representation boundaries audited, and update,
example and stopping counts checked. The scheduler uses four workers on GPU1
and leaves other workloads untouched. Configuration:
[discrete_p97_dataset_scaling.yaml](configs/discrete_p97_dataset_scaling.yaml).
