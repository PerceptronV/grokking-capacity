#!/usr/bin/env python3
"""Summarize the numerical update and flag discrepancies without rewriting prose."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/capacity_h100_matched_20261103';PAPER=ROOT/'docs/papers/2026 Grokking Capacity'
S=json.loads((OUT/'summary.json').read_text());I=json.loads((OUT/'invariance.json').read_text());P=json.loads((OUT/'plot_manifest.json').read_text());V=json.loads((OUT/'checkpoint_verification.json').read_text())
r=I['ratio'];s=I['f_scale'];C=S['C']
lines=[f'''# Matched H100 capacity update — 2026-11-03

This update changes numerical values, plots, and measurement counts in `paper-neurips-revised.tex` and its included appendices. References to the unrecoverable capacity run and the superseded normalization comparison were removed as requested. Other scientific wording was retained, including claims flagged by the earlier review. Other top-level manuscript sources and archived forecast/registration artifacts were not rewritten. The theory includes are shared files, so other drafts that include them also inherit their updated numerical values.

## Fresh measurement

- Hardware: NVIDIA H100 80 GB; float32; the manifest records PyTorch/CUDA and numerical settings.
- Architecture: the existing two-layer, one-head Transformer; widths {S['dims']}.
- Training: random inputs/labels at p=113; dropout 0.2; AdamW learning rate 0.001, weight decay 1.0, betas (0.9,0.98), batch size 512; maximum 5,000 epochs; capacity loss patience 100 and minimum improvement 0.0001.
- Eight dataset sizes per width, two fresh seeds {S['seeds']}: {S['n_runs']} completed fits.
- Estimator: each seed's maximum achieved bits over the fixed dataset-size grid, averaged across the two seeds at each width, then an affine fit against parameter count.
- **C = {C:.6f} bits/parameter**, 95% bootstrap interval **[{S['CI95'][0]:.6f}, {S['CI95'][1]:.6f}]**; intercept {S['intercept']:.2f} bits; R²={S['r2']:.6f}.
- Per-seed slopes: {', '.join(f'{k}: {v["C"]:.6f}' for k,v in S['per_seed_fits'].items())}.
- Bootstrap: width-cluster resampling plus paired seed resampling across widths, 20,000 draws. This is finite-grid empirical uncertainty with only two seeds; it does not certify an asymptotic representational bound.
- Runs reaching the 5,000-epoch horizon: {S['horizon_reached']}. Per-seed maxima at the largest tested dataset size: {S['plateau_peak_at_largest_n']}. Maximum late-trace relative drift over all runs: {S['max_late_drift']:.3%}.
- Twelve selected checkpoints were independently evaluated on CPU after reconstructing their seeded random datasets. Maximum difference from saved capacity: {V['max_error_bits']:.6f} bits; all checks passed.

## Numerical propagation

The calculation uses the unrounded estimate; the paper displays rounded values.

| Quantity | Before | After |
|---|---:|---:|
| C | 2.16 | {C:.6f} |
| Capacity fraction / inverse capacity multiplier | 1 | {s:.6f} |
| Crossing-window gamma | {I['gamma_old']:.6f} | {I['gamma_new']:.6f} |
| Invariant gamma/C | {I['q']:.6f} | {I['q']:.6f} |
| Lower fit-window boundary | 0.08 | {.08*s:.6f} |
| Full-range gamma | 30.034299 | {30.034299*r:.6f} |
| Fold capacity fraction | 0.026213 | {.02621300454057341*s:.6f} |
| Fold prime (unrounded root) | {I['fold_old']:.6f} | {I['fold_new']:.6f} |

Linear coefficients in f were multiplied by C/2.16; quadratic coefficients by its square. All retained fit observations are unchanged, so fit-window bounds move with f. The maximum difference in the compared log-times is {I['max_log_time_difference']:.3g}. Parameter-space plots without capacity axes/labels remain byte-identical.

## Main-test checks and flags

**No change in the central measured-crossing/onset test is induced by the new constant.** The raw-data audit was rerun and reproduces both the original central results and the dense crossing results exactly. Spearman rho remains {I['main_test_after']['spearman']:.9f}, Kendall tau {I['main_test_after']['kendall']:.9f}, fitted slope {I['main_test_after']['slope']:.9f}, and median log residual {I['main_test_after']['median_log_residual']:.9f} dex. The existing hypothesis-test table was retained. This update does not resolve its previously flagged rank/degree-of-freedom and calibration issues.

1. **Capacity-only comparisons move.** P_mem=K/C increases by {s:.4f}. The central observed onsets remain above the revised reference, but the multiplicative separation is smaller. This should be considered when wording is revised later.
2. **The old main-text capacity-fraction figure was not a simple coordinate-only update.** Its embedded fit was gamma=25.14, intercept 12.5 epochs, R²=0.978. Replotting the current canonical ten-prime, {P['speed_rows']}-run cohort gives gamma={P['speed_fit_gamma']:.6f}, intercept {P['speed_fit_b']:.6f}, R²={P['speed_fit_r2']:.6f}. In the former capacity units this gamma is {P['speed_fit_gamma']/r:.6f}. The difference in R² and intercept is a pre-existing figure/data inconsistency, not an effect of C. The new plot matches the archived full-range canonical fit.
3. **Nominal versus observed f range.** The prose's nominal upper bound 0.25 was rescaled to {.25*s:.3f}; the largest uncensored canonical data point is actually {.18642423421563978*s:.6f}. Its existing wording about the exact budget boundary remains flagged; no new fit range was selected.
4. **The new slope does not remove existing prediction misses.** The held-out p=149 interval miss, depth/learning-rate failures, original p=197 censoring, and large-prime per-layer failures in the prior review are unchanged. These are not new discrepancies caused by the capacity update.
5. **Existing nonnumerical interpretation is deferred.** The prose still calls the achieved capacity a representational threshold, contains the redundant phrase “except with dropout 0.2” after matching the protocol, and uses qualitative plateau/stability language. These were not rewritten. Two fresh seeds and the diagnostics above now support assessment of stability.
6. **Noise-floor / spectral-clock comparison needs a units check.** The empirical gamma-infinity changes from 11 to {10.960686837450506*r:.4f}, whereas the independently quoted perceptron spectral clock near 10 (10.3 in its derivation) has not been redefined in the new capacity coordinate. The retained phrase ‘numerically coincident’ in `theory/principled.tex` is therefore flagged. An explicit conversion is required before this comparison can be claimed; the spectral-clock number was not arbitrarily rescaled.
7. **Affine versus through-origin capacity.** The intercept is {S["intercept_fraction_smallest"]:.2%} of the smallest measured mean maximum. A through-origin fit gives C={S["C_through_origin"]:.6f}, versus the primary affine slope {C:.6f}. The existing “small intercept” / “close to zero” wording was retained; the numerical sensitivity is now explicit.
8. **Illustrative barrier arithmetic is pre-existing.** The example's f value was rescaled to preserve its physical load, but its quoted 2,200 nats remains inconsistent with PCf ln(2), approximately 14,972 nats. This unrelated correction is flagged rather than silently folded into a capacity update.

| Prime | Old P_mem | New P_mem | Observed onset | Onset / new P_mem |
|---:|---:|---:|---:|---:|''']
for x in I['capacity_only_thresholds']:lines.append(f"| {x['p']} | {x['P_mem_before']:.0f} | {x['P_mem_after']:.0f} | {x['P_onset']:.0f} | {x['onset_over_P_mem']:.3f} |")
lines.append('''
## Reproduction and artifacts

From the repository root:

```bash
python scripts/run_capacity_h100_matched.py
python scripts/update_matched_capacity_paper.py
python scripts/verify_matched_capacity_checkpoints.py
python scripts/report_matched_capacity_update.py
```

The run controller is resumable; completed registry entries are skipped. It uses GPU 0 and leaves GPU 1 alone. The update script requires all 96 completed measurements. The main-test recheck is generated with:

```bash
python 'docs/papers/2026 Grokking Capacity/review-audit.py' > results/capacity_h100_matched_20261103/main-test-recheck.json
```

- [Measurement manifest](../../../results/capacity_h100_matched_20261103/manifest.json)
- [Fit and diagnostics](../../../results/capacity_h100_matched_20261103/summary.json)
- [Per-run values](../../../results/capacity_h100_matched_20261103/cells.json)
- [Independent checkpoint checks](../../../results/capacity_h100_matched_20261103/checkpoint_verification.json)
- [Invariance and threshold checks](../../../results/capacity_h100_matched_20261103/invariance.json)
- [Transformed fitted coefficients](../../../results/capacity_h100_matched_20261103/forecast_coefficients.json)
- [Fresh raw-data main-test audit](../../../results/capacity_h100_matched_20261103/main-test-recheck.json)
- [Plot rows and fit](../../../results/capacity_h100_matched_20261103/plot_manifest.json)

The publication update reads its capacity directly from this campaign. Existing historical analysis defaults and frozen forecast files retain their original units; use the dedicated update command above to regenerate the revised publication assets. The `before/` directory preserves the source and included-figure snapshot used by the numerical update.
''')
(PAPER/'matched-capacity-update.md').write_text('\n'.join(lines))
print('Wrote matched-capacity-update.md')
