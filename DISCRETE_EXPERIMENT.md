# Discrete latent grokking experiment

Branch: `idea/discrete`. Registered 2026-10-24 before new training.

Latest follow-up: pre-softmax logit LayerNorm is implemented and tested. Both
normalized seed-42 pilots completed 120k updates without passing arithmetic
validation (19.34% at temperature 0.1; 88.75% at temperature 1). See the
[normalization evidence](reports/categorical_slot_normalization_v7/comparison.csv)
and the exact binary speed/grokking comparison near the end of this record.

Question: can a large transformer with discrete internal representations match a
small continuous model's empirical memorisation capacity, generalise sooner,
and avoid the generalisation delay of an equally large continuous control?
A positive result is not assumed. Lack of learning is not evidence of reduced
capacity, and failure to grok is not success unless generalisation succeeds.

## Completed results: both full-width discrete families

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
