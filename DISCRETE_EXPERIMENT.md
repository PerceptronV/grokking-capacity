# Discrete latent grokking experiment

Branch: `idea/discrete`. Registered 2026-10-24 before new training.

Question: can a large transformer with discrete internal representations match a
small continuous model's empirical memorisation capacity, generalise sooner,
and avoid the generalisation delay of an equally large continuous control?
A positive result is not assumed. Lack of learning is not evidence of reduced
capacity, and failure to grok is not success unless generalisation succeeds.

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
3. Select A/B by calibration only: aim for capacity within 10%, target capacity
   near dataset label complexity, B/A parameter ratio >=4. Freeze selection
   before fresh modular seed comparisons. C has B's architecture/parameter
   tensors/initial weights, with quantisation disabled. If no defensible match
   is found, report that and continue calibration rather than claiming a match.
4. Compare A/B/C on identical modular train/held-out splits and batch orders for
   >=3 fresh seeds. Report steps, examples seen, epochs and wall seconds to
   sustained 95% and 99% accuracy, train-to-held-out delay and maximal gap.
   Never replace censored runs with a fabricated threshold-crossing time.

## Architecture and estimator

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

## Development and evidence

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

Active queues (verify process start identities from each manifest before acting):

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
