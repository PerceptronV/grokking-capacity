"""
Verify predictions from the toy interference model:
1. The slope `a` in T_mem = b * e^(a*f) is independent of threshold tau
2. The prefactor `b` scales as const + const' * log(1/(1-tau))

Uses the same methodology as primes --speed: aggregate across seeds,
compute f = dataset_bits / (C * P), fit log(epochs) = a*f + log(b).
"""
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy import stats
import consts


def compute_saturation_epoch_from_trace(acc_trace, threshold, steps_per_epoch, steps_trace=None):
    if len(acc_trace) > 0 and acc_trace.max() <= 1.0:
        acc_trace = acc_trace * 100
    if steps_trace is not None:
        for step, acc in zip(steps_trace, acc_trace):
            if acc >= threshold:
                return step / steps_per_epoch
    else:
        for step, acc in enumerate(acc_trace):
            if acc >= threshold:
                return step / steps_per_epoch
    return None


def load_all_speed_data(saturation_threshold, batch_size=512):
    """Load all speed data across all primes and seeds at a given threshold.

    Returns arrays of (f, saturation_epoch, prime) for all saturated runs.
    """
    speed_base_dir = 'data/speed'
    if not os.path.exists(speed_base_dir):
        raise FileNotFoundError("data/speed not found")

    all_f = []
    all_epochs = []
    all_primes = []

    # Find all prime directories
    pattern_re = re.compile(r'^p(\d+)_seed(\d+)$')
    dirs = [d for d in os.listdir(speed_base_dir)
            if pattern_re.match(d) and os.path.isdir(os.path.join(speed_base_dir, d))]

    # Group by prime
    prime_seed_dirs = {}
    for d in dirs:
        m = pattern_re.match(d)
        p = int(m.group(1))
        if p not in prime_seed_dirs:
            prime_seed_dirs[p] = []
        prime_seed_dirs[p].append(d)

    for p in sorted(prime_seed_dirs.keys()):
        # Group runs by (dim, param_count) to average across seeds
        configs = {}
        for seed_dir in prime_seed_dirs[p]:
            speed_dir = os.path.join(speed_base_dir, seed_dir)
            files = glob(os.path.join(speed_dir, 'speed_dim*.npz'))
            for sf in files:
                data = np.load(sf)
                dim = int(data['dim'])
                param_count = int(data['param_count'])
                n_samples = int(data['n_samples'])
                steps_per_epoch = (n_samples + batch_size - 1) // batch_size

                sat_epoch = None
                if 'train_acc_trace' in data:
                    acc_trace = data['train_acc_trace']
                    steps_trace = data['steps_trace'] if 'steps_trace' in data else None
                    sat_epoch = compute_saturation_epoch_from_trace(
                        acc_trace, saturation_threshold, steps_per_epoch, steps_trace)

                if sat_epoch is None:
                    continue

                key = (dim, param_count)
                if key not in configs:
                    configs[key] = {
                        'sat_epochs': [],
                        'dataset_bits': float(data['dataset_bits']),
                        'param_count': param_count,
                    }
                configs[key]['sat_epochs'].append(sat_epoch)

        # Average across seeds
        for key, cfg in configs.items():
            mean_epoch = float(np.mean(cfg['sat_epochs']))
            f = cfg['dataset_bits'] / (consts.C * cfg['param_count'])
            all_f.append(f)
            all_epochs.append(mean_epoch)
            all_primes.append(p)

    return np.array(all_f), np.array(all_epochs), np.array(all_primes)


def fit_exponential(f_values, epochs_values):
    """Fit epochs = b * exp(a * f) in log space. Returns (a, log_b, b, r_squared)."""
    log_epochs = np.log(epochs_values)
    a, log_b = np.polyfit(f_values, log_epochs, 1)
    b = np.exp(log_b)
    y_pred = a * f_values + log_b
    ss_res = np.sum((log_epochs - y_pred) ** 2)
    ss_tot = np.sum((log_epochs - np.mean(log_epochs)) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return a, log_b, b, r_sq


def main():
    thresholds = [70, 75, 80, 85, 90, 92, 94, 95, 96, 97, 98, 99]

    results = []
    for tau in thresholds:
        print(f"\n{'='*60}")
        print(f"Threshold tau = {tau}%")
        print(f"{'='*60}")
        f_vals, epoch_vals, primes = load_all_speed_data(tau)
        print(f"  {len(f_vals)} data points across {len(set(primes))} primes")

        if len(f_vals) < 3:
            print(f"  Skipping: not enough data")
            continue

        a_slope, log_b, b_prefactor, r_sq = fit_exponential(f_vals, epoch_vals)
        print(f"  Fit: epochs = {b_prefactor:.2f} * exp({a_slope:.3f} * f)")
        print(f"  R² = {r_sq:.4f}")
        print(f"  log(b) = {log_b:.4f}")
        print(f"  -log(1 - tau/100) = {-np.log(1 - tau/100):.4f}")

        results.append({
            'tau': tau,
            'a': a_slope,
            'log_b': log_b,
            'b': b_prefactor,
            'r_sq': r_sq,
            'n_points': len(f_vals),
            'f_min': float(f_vals.min()),
            'f_max': float(f_vals.max()),
            'f_values': f_vals,
            'epoch_values': epoch_vals,
        })

    if not results:
        print("No results!")
        return

    taus = np.array([r['tau'] for r in results])
    a_vals = np.array([r['a'] for r in results])
    log_b_vals = np.array([r['log_b'] for r in results])
    r_sq_vals = np.array([r['r_sq'] for r in results])
    neg_log_1_minus_tau = -np.log(1 - taus / 100)

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'tau':>6s} {'a (slope)':>12s} {'log(b)':>12s} {'b (prefact)':>12s} {'R²':>8s} {'n_pts':>6s} {'-log(1-τ)':>12s}")
    print(f"{'-'*80}")
    for r in results:
        tau = r['tau']
        print(f"{tau:>6.0f} {r['a']:>12.3f} {r['log_b']:>12.4f} {r['b']:>12.2f} {r['r_sq']:>8.4f} {r['n_points']:>6d} {-np.log(1-tau/100):>12.4f}")

    # Test 1: Is a independent of tau?
    print(f"\n{'='*60}")
    print("TEST 1: Is slope 'a' independent of tau?")
    print(f"{'='*60}")
    print(f"  Mean a  = {np.mean(a_vals):.3f}")
    print(f"  Std a   = {np.std(a_vals):.3f}")
    print(f"  CV(a)   = {np.std(a_vals)/np.mean(a_vals)*100:.1f}%")
    print(f"  Range   = [{np.min(a_vals):.3f}, {np.max(a_vals):.3f}]")

    # Linear regression of a on -log(1-tau) to check for trend
    slope_a, intercept_a, r_a, p_a, se_a = stats.linregress(neg_log_1_minus_tau, a_vals)
    print(f"  Regression of a on -log(1-tau): slope={slope_a:.4f}, p-value={p_a:.4f}")
    print(f"  (p > 0.05 means no significant dependence)")

    # Test 2: Does b (prefactor) scale linearly with -log(1-tau)?
    # Model: b(tau) = (1/r0) * [log(eps0) - log(1-tau)] = const + (1/r0)*(-log(1-tau))
    b_vals = np.array([r['b'] for r in results])
    print(f"\n{'='*60}")
    print("TEST 2: Does b ~ const + const' * (-log(1-tau))?")
    print(f"{'='*60}")
    slope_b, intercept_b, r_b, p_b, se_b = stats.linregress(neg_log_1_minus_tau, b_vals)
    print(f"  Regression: b = {intercept_b:.4f} + {slope_b:.4f} * (-log(1-tau))")
    print(f"  R² = {r_b**2:.4f}")
    print(f"  p-value = {p_b:.6f}")
    print(f"  (Model predicts linear relationship, slope ≈ 1/r0)")

    # Also check log(b) for completeness
    slope_lb, intercept_lb, r_lb, p_lb, se_lb = stats.linregress(neg_log_1_minus_tau, log_b_vals)
    print(f"\n  Also: log(b) = {intercept_lb:.4f} + {slope_lb:.4f} * (-log(1-tau))")
    print(f"  R² = {r_lb**2:.4f}")

    # Censoring check: how does the max f change with threshold?
    print(f"\n{'='*60}")
    print("CENSORING CHECK: Does the f-range shrink at higher tau?")
    print(f"{'='*60}")
    for r in results:
        print(f"  tau={r['tau']:>3.0f}%: {r['n_points']} points, f_max={r.get('f_max', 'N/A')}, f_range=[{r.get('f_min', 'N/A')}, {r.get('f_max', 'N/A')}]")

    # Censoring-controlled analysis: restrict to common f-range
    # Use the f-range of the highest threshold (most censored)
    f_max_common = min(r['f_max'] for r in results)
    f_min_common = max(r['f_min'] for r in results)
    print(f"\n{'='*60}")
    print(f"CENSORING-CONTROLLED FIT (f in [{f_min_common:.3f}, {f_max_common:.3f}])")
    print(f"{'='*60}")

    results_ctrl = []
    for r in results:
        mask = (r['f_values'] >= f_min_common) & (r['f_values'] <= f_max_common)
        f_sub = r['f_values'][mask]
        ep_sub = r['epoch_values'][mask]
        if len(f_sub) < 3:
            continue
        a_c, log_b_c, b_c, r_sq_c = fit_exponential(f_sub, ep_sub)
        print(f"  tau={r['tau']:>3.0f}%: a={a_c:.3f}, b={b_c:.2f}, R²={r_sq_c:.4f}, n={len(f_sub)}")
        results_ctrl.append({'tau': r['tau'], 'a': a_c, 'b': b_c, 'log_b': log_b_c, 'r_sq': r_sq_c})

    if len(results_ctrl) >= 3:
        a_ctrl = np.array([r['a'] for r in results_ctrl])
        b_ctrl = np.array([r['b'] for r in results_ctrl])
        taus_ctrl = np.array([r['tau'] for r in results_ctrl])
        neg_log_ctrl = -np.log(1 - taus_ctrl / 100)

        print(f"\n  Controlled slope a: mean={np.mean(a_ctrl):.3f}, std={np.std(a_ctrl):.3f}, CV={np.std(a_ctrl)/np.mean(a_ctrl)*100:.1f}%")
        slope_a_c, intercept_a_c, r_a_c, p_a_c, _ = stats.linregress(neg_log_ctrl, a_ctrl)
        print(f"  Regression a on -log(1-tau): slope={slope_a_c:.4f}, p={p_a_c:.4f}")

        slope_b_c, intercept_b_c, r_b_c, p_b_c, _ = stats.linregress(neg_log_ctrl, b_ctrl)
        print(f"  Regression b on -log(1-tau): b = {intercept_b_c:.2f} + {slope_b_c:.2f}*(-log(1-tau)), R²={r_b_c**2:.4f}")

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: a (slope) vs tau
    ax = axes[0, 0]
    ax.plot(taus, a_vals, 'o-', color='tab:blue', markersize=8)
    ax.axhline(np.mean(a_vals), color='red', linestyle='--', alpha=0.7,
               label=f'mean = {np.mean(a_vals):.2f}')
    ax.fill_between([taus.min()-1, taus.max()+1],
                    np.mean(a_vals) - np.std(a_vals),
                    np.mean(a_vals) + np.std(a_vals),
                    color='red', alpha=0.1, label=f'±1σ = {np.std(a_vals):.2f}')
    ax.set_xlabel('Threshold τ (%)', fontsize=12)
    ax.set_ylabel('Slope a (exponent)', fontsize=12)
    ax.set_title('Slope a vs threshold τ', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: a vs -log(1-tau)
    ax = axes[0, 1]
    ax.plot(neg_log_1_minus_tau, a_vals, 'o', color='tab:blue', markersize=8)
    x_line = np.linspace(neg_log_1_minus_tau.min(), neg_log_1_minus_tau.max(), 100)
    ax.plot(x_line, slope_a * x_line + intercept_a, '--', color='red', alpha=0.7,
            label=f'slope={slope_a:.3f}, p={p_a:.3f}')
    ax.set_xlabel('-log(1-τ)', fontsize=12)
    ax.set_ylabel('Slope a (exponent)', fontsize=12)
    ax.set_title('Slope a vs -log(1-τ)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: b (prefactor) vs -log(1-tau)
    ax = axes[1, 0]
    ax.plot(neg_log_1_minus_tau, b_vals, 'o', color='tab:green', markersize=8)
    ax.plot(x_line, slope_b * x_line + intercept_b, '--', color='red', alpha=0.7,
            label=f'b = {intercept_b:.2f} + {slope_b:.2f}·(-log(1-τ))\nR²={r_b**2:.4f}')
    ax.set_xlabel('-log(1-τ)', fontsize=12)
    ax.set_ylabel('b (prefactor)', fontsize=12)
    ax.set_title('Prefactor b vs -log(1-τ)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 4: R² vs tau
    ax = axes[1, 1]
    ax.plot(taus, r_sq_vals, 'o-', color='tab:orange', markersize=8)
    ax.set_xlabel('Threshold τ (%)', fontsize=12)
    ax.set_ylabel('R² of exponential fit', fontsize=12)
    ax.set_title('Fit quality vs threshold', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Verification of toy interference model predictions', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('media/tau_independence_verification.pdf', bbox_inches='tight', dpi=150)
    print(f"\nSaved plot to media/tau_independence_verification.pdf")
    plt.show()


if __name__ == '__main__':
    main()
