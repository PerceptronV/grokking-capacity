# Discrete latent grokking experiment

Branch: `idea/discrete`. Registered 2026-10-24 before new training.

Question: can a large transformer with discrete internal representations match a
small continuous model's empirical memorisation capacity, generalise sooner,
and avoid the generalisation delay of an equally large continuous control?
A positive result is not assumed. Lack of learning is not evidence of reduced
capacity, and failure to grok is not success unless generalisation succeeds.

## Current experiment status

The user selected **p149 modular division with a fixed 50% split**. The target is
79,810.65 bits, measured as training examples times log2(p+2). Both A and B must
match this target and each other within 10%, with B/A parameter ratio at least 4.
Matching A to B alone is explicitly insufficient. Earlier p97/p163 explorations
remain in the history and do not redefine the final task.

The current small-model candidate is A44: 75,460 parameters and 78,516 demonstrated
bits after 60k updates on n16384,seed61. Repeated sizes and seeds are running.
B16q64 is rejected: its longer calibration reached 89,670 bits, above 87,792.
The narrower B14q64 (329,952 parameters) passed its own arithmetic gate and
independent CPU audit at 99.85% held-out. Its seed61 capacity checks are running
at both n16384/32768. **No capacity match or fresh-seed comparison is accepted.**

The B16/C16 development comparison favoured C. B14's continuous control and the
wider coarse-channel arithmetic diagnostics remain separate development runs.
Failed arithmetic recipes are excluded from random labels. Use
`python scripts/discrete_status.py --active-only` for verified live progress.

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
   other, with B/A parameter ratio >=4. Matching A to B alone is insufficient.
   Keep the training split fixed at 50%; tune models to the dataset. Freeze selection
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
only latent_dim/levels differ. Both candidates require the same sustained99%
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
train/test crossings were2100/2300; sustained99% crossings were24300/9500, yielding
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
