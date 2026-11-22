"""Numerical primitives shared by the post-hoc diagnostics.

Callers retain their own thresholds, epoch conventions, fit windows, and
minimum sample requirements. No run selection or plotting happens here.
"""
import numpy as np


def normalise_accuracy(values):
    """Accept traces stored as fractions or percentages, using the legacy rule."""
    values = np.asarray(values, dtype=float)
    return values / 100.0 if values.size and values.max() > 1.5 else values


def first_crossing(values, threshold, *, index_base=0):
    """First recorded threshold crossing, or None; no censor-time imputation."""
    above = np.where(np.asarray(values) >= threshold)[0]
    return int(above[0]) + index_base if above.size else None


def r_squared(observed, predicted):
    observed, predicted = np.asarray(observed), np.asarray(predicted)
    residual = float(np.sum((observed - predicted) ** 2))
    total = float(np.sum((observed - observed.mean()) ** 2))
    return 1.0 - residual / total if total > 0 else float('nan')


def linear_fit(x, y):
    """OLS y = slope*x + intercept; return slope, intercept, and R-squared."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    design = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(design, y, rcond=None)
    prediction = design @ np.array([slope, intercept])
    return float(slope), float(intercept), r_squared(y, prediction)


def ols(design, y):
    """Fit an explicit design matrix; preserve the timescale fit's RSS reduction."""
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    rss = float(residual @ residual)
    total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / total if total > 0 else float('nan')
    return coef, r2, rss


def permutation_slope_pvalue(x, y, observed, *, resamples, seed=0, min_samples=0):
    """Two-sided y-permutation test, including the finite-resampling correction."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < min_samples:
        return float('nan')
    if resamples < 1:
        raise ValueError('Require a positive number of permutations')
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(resamples):
        slope, _, _ = linear_fit(x, rng.permutation(y))
        count += abs(slope) >= abs(observed)
    return (int(count) + 1) / (resamples + 1)
