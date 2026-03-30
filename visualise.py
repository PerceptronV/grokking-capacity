"""
Utility script to view saved experiment results.

Subcommands:
    groks       - Grokking experiment visualizations
    capacity    - Model capacity (memorisation) experiment visualizations
    speed       - Learning speed experiment visualizations
    primes      - Multi-prime analysis visualizations
"""
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from typing import Dict, List, Optional, Tuple
from plotting import (
    plot_combined_curves,
    plot_separate_curves,
    plot_separate_curves_with_memorization,
    plot_grokking_delay as plot_delay_util,
    plot_grokking_integral,
    plot_grokking_time as plot_time_util,
    plot_delay_vs_memorization,
    plot_delay_and_memorization_vs_params,
    plot_grokking_critical_capacity,
    plot_capacity_curves,
    plot_capacity_estimation,
    estimate_capacity,
    plot_bits_vs_accuracy,
    plot_memorization_curves,
    plot_grokking_with_memorization,
    plot_max_memorization_vs_params,
    compute_critical_params,
    compute_critical_params_from_speed,
    plot_grokking_delay_with_speed,
    plot_learning_speed_curves,
    plot_speed_vs_model_size,
    plot_combined_speed_analysis,
    plot_saturation_time_vs_capacity_fraction,
    plot_saturation_epochs_vs_params,
    plot_saturation_epochs_vs_dataset_bits,
    plot_saturation_epochs_vs_inverse_capacity,
    plot_rate_vs_dataset_size,
    plot_delay_vs_capacity_fraction,
    plot_predicted_vs_empirical_grokking,
    plot_critical_params_vs_prime,
    plot_critical_params_vs_dataset_size,
    calculate_grokking_delay
)
from utils import compute_dataset_size_bits
import consts
from results import ResultsIndex
from matching import load_match_table
from cli_args import (
    add_vis_output_args,
    add_vis_file_selection_args,
    add_vis_model_filter_args,
    add_vis_optimizer_filter_args,
    add_vis_task_args,
    add_vis_dim_args,
)


def list_results(data_dir, pattern='grokking_dim*.npz'):
    """List all saved result files."""
    files = sorted(glob(os.path.join(data_dir, pattern)))
    
    if not files:
        print(f"No results found matching pattern: {pattern}")
        return []
    
    print(f"\nFound {len(files)} result files:")
    print("="*80)
    
    results = []
    for i, fname in enumerate(files):
        data = np.load(fname)
        dim = int(data['dim'])
        param_count = int(data['param_count'])
        depth = int(data['depth'])
        heads = int(data['heads'])
        final_train_acc = data['train_acc'][-1]
        final_val_acc = data['val_acc'][-1]
        
        results.append({
            'file': fname,
            'dim': dim,
            'param_count': param_count,
            'depth': depth,
            'heads': heads,
            'final_train_acc': final_train_acc,
            'final_val_acc': final_val_acc,
            'data': data
        })
        
        print(f"{i:2d}. {os.path.basename(fname)}")
        print(f"    dim={dim:3d}, depth={depth}, heads={heads}, params={param_count:8,}")
        print(f"    Final: Train={final_train_acc:.1f}%, Val={final_val_acc:.1f}%")
    
    print("="*80)
    return results


def plot_result(result_file):
    """Plot a single result file."""
    if not os.path.exists(result_file):
        print(f"File not found: {result_file}")
        return
    
    data = np.load(result_file)
    train_acc = data['train_acc']
    val_acc = data['val_acc']
    dim = int(data['dim'])
    param_count = int(data['param_count'])
    depth = int(data['depth'])
    heads = int(data['heads'])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(train_acc, label='Training Accuracy', color='#1b9e77', linewidth=2, linestyle='-')
    ax.plot(val_acc, label='Validation Accuracy', color='#d95f02', linewidth=2, linestyle='--')
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title(f'Grokking Curve: dim={dim}, depth={depth}, heads={heads}\n'
                 f'{param_count:,} parameters', fontsize=16, pad=20)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylim([0, 105])
    
    textstr = f'Final Train Acc: {train_acc[-1]:.1f}%\nFinal Val Acc: {val_acc[-1]:.1f}%'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


def plot_grokking_delay(result_files, threshold_train=99.0, threshold_val=99.0, 
                        threshold_params=None, save_path=None, show=True):
    """Plot grokking delay vs parameter count."""
    if not result_files:
        print("No files to compare")
        return
    
    # Load all data
    results = []
    for fname in result_files:
        if not os.path.exists(fname):
            print(f"Warning: File not found: {fname}")
            continue
            
        data = np.load(fname)
        results.append({
            'train_acc': data['train_acc'],
            'val_acc': data['val_acc'],
            'dim': int(data['dim']),
            'param_count': int(data['param_count'])
        })
    
    # Use shared plotting utility
    plot_delay_util(results, threshold_train=threshold_train, threshold_val=threshold_val, 
                    threshold_params=threshold_params, save_path=save_path, show=show)


def plot_grokking_time(result_files, threshold_val=99.0, save_path=None, show=True):
    """Plot absolute grokking time (epochs to reach threshold) vs parameter count."""
    if not result_files:
        print("No files to compare")
        return
    
    # Load all data
    results = []
    for fname in result_files:
        if not os.path.exists(fname):
            print(f"Warning: File not found: {fname}")
            continue
            
        data = np.load(fname)
        results.append({
            'val_acc': data['val_acc'],
            'dim': int(data['dim']),
            'param_count': int(data['param_count'])
        })
    
    # Use shared plotting utility
    plot_time_util(results, threshold_val=threshold_val, save_path=save_path, show=show)


def compare_results(result_files):
    """Compare multiple results on the same plot."""
    if not result_files:
        print("No files to compare")
        return
    
    # Load all data
    results = []
    for fname in result_files:
        if not os.path.exists(fname):
            print(f"Warning: File not found: {fname}")
            continue
            
        data = np.load(fname)
        results.append({
            'train_acc': data['train_acc'],
            'val_acc': data['val_acc'],
            'dim': int(data['dim']),
            'param_count': int(data['param_count'])
        })
    
    # Use shared plotting utility
    plot_combined_curves(results, show=True)


def extract_dim(fname):
    match = re.search(r'dim(\d+)', fname)
    return int(match.group(1)) if match else 0


def extract_samples(fname):
    match = re.search(r'samples(\d+)', fname)
    return int(match.group(1)) if match else 0


def compute_saturation_epoch_from_trace(
    acc_trace: np.ndarray,
    threshold: float,
    steps_per_epoch: int,
    steps_trace: Optional[np.ndarray] = None
) -> Optional[float]:
    """
    Compute saturation epoch from accuracy trace.

    Args:
        acc_trace: Array of accuracy values (0-100 scale or 0-1 scale)
        threshold: Accuracy threshold (0-100 scale)
        steps_per_epoch: Number of steps per epoch
        steps_trace: Optional array of step numbers corresponding to acc_trace.
                     If None, assumes acc_trace is indexed by step number directly.

    Returns:
        Saturation epoch (float) or None if threshold never reached
    """
    # Handle both 0-1 and 0-100 scales
    if len(acc_trace) > 0 and acc_trace.max() <= 1.0:
        acc_trace = acc_trace * 100

    if steps_trace is not None:
        # acc_trace and steps_trace are parallel arrays
        for step, acc in zip(steps_trace, acc_trace):
            if acc >= threshold:
                return step / steps_per_epoch
    else:
        # acc_trace is indexed by step number directly
        for step, acc in enumerate(acc_trace):
            if acc >= threshold:
                return step / steps_per_epoch

    return None  # Never reached threshold


def _legacy_filter(value, default):
    """Return a filter that accepts None (legacy records) or the exact value when value==default.

    This is used for backward compat with .meta.json sidecars that predate a field:
    e.g. legacy groks entries may not have 'operation' stored, which is treated as '/'.
    """
    if value == default:
        return lambda x: x is None or x == value
    return value


def _add_filter(filters, key, value):
    """Add key=value to filters dict only if value is not None (optional filter)."""
    if value is not None:
        filters[key] = value


def _nested_get(d, key):
    """Recursively search for key in nested dict d, returning value or None."""
    if key in d:
        return d[key]
    for v in d.values():
        if isinstance(v, dict):
            result = _nested_get(v, key)
            if result is not None:
                return result
    return None


def _build_groks_filters(args, p):
    """Build ResultsIndex query filters for grokking experiments."""
    filters = {
        'experiment_type': 'groks',
        'p': p,
        'depth': args.depth,
        'heads': args.heads,
        'seed': args.seed,
        'operation': _legacy_filter(args.op, '/'),
        'train_fraction': _legacy_filter(args.training_fraction, 0.5),
    }
    _add_filter(filters, 'weight_decay', getattr(args, 'weight_decay', None))
    _add_filter(filters, 'dropout', getattr(args, 'dropout', None))
    _add_filter(filters, 'init_scale', getattr(args, 'init_scale', None))
    return filters


def _build_speed_filters(args):
    """Build ResultsIndex query filters for speed experiments (without p)."""
    filters = {
        'experiment_type': 'speed',
        'depth': getattr(args, 'depth', 2),
        'heads': getattr(args, 'heads', 1),
        'operation': _legacy_filter(getattr(args, 'op', '/'), '/'),
        'train_fraction': _legacy_filter(getattr(args, 'training_fraction', 0.5), 0.5),
    }
    _add_filter(filters, 'weight_decay', getattr(args, 'weight_decay', None))
    _add_filter(filters, 'dropout', getattr(args, 'dropout', None))
    return filters


def _build_capacity_filters(args, p):
    """Build ResultsIndex query filters for capacity experiments."""
    op = getattr(args, 'op', 'random')
    filters = {
        'experiment_type': 'capacity',
        'p': p,
        'seed': args.seed,
        'depth': getattr(args, 'depth', 2),
        'heads': getattr(args, 'heads', 1),
    }
    # For capacity, operation='random' is the default; legacy entries may not have it
    if op != 'random':
        filters['operation'] = op
    _add_filter(filters, 'dropout', getattr(args, 'dropout', None))
    _add_filter(filters, 'weight_decay', getattr(args, 'weight_decay', None))
    return filters


def _apply_dim_filter(entries, args, index):
    """Post-filter entries by --dims or --dims-start/end/step."""
    if args.dims:
        dims_set = set(args.dims)
        return [e for e in entries if _nested_get(e, 'dim') in dims_set]
    if getattr(args, 'dims_start', None) is not None and getattr(args, 'dims_end', None) is not None:
        step = getattr(args, 'dims_step', None) or 1
        dims_range = set(range(args.dims_start, args.dims_end + 1, step))
        return [e for e in entries if _nested_get(e, 'dim') in dims_range]
    return entries


def _load_groks_results(args, p, index=None):
    """Load grokking results using ResultsIndex (or --files/--pattern fallback).

    Returns list of result dicts:
        {train_acc, val_acc, dim, param_count, mem_t_trace (opt), mem_u_trace (opt), mem_trace (opt)}
    """
    results = []

    # --files: explicit file list, bypass index
    if getattr(args, 'files', None):
        for fname in args.files:
            if not os.path.exists(fname):
                print(f"Warning: File not found: {fname}")
                continue
            data = np.load(fname, allow_pickle=True)
            result = {
                'train_acc': data['train_acc'],
                'val_acc': data['val_acc'],
                'dim': int(data['dim']),
                'param_count': int(data['param_count']),
            }
            if 'mem_t_trace' in data:
                result['mem_t_trace'] = data['mem_t_trace']
            if 'mem_u_trace' in data:
                result['mem_u_trace'] = data['mem_u_trace']
                result['mem_trace'] = data['mem_u_trace']
            elif 'mem_trace' in data:
                result['mem_u_trace'] = data['mem_trace']
                result['mem_trace'] = data['mem_trace']
            results.append(result)
        return sorted(results, key=lambda r: r['dim'])

    # --pattern: glob fallback
    if getattr(args, 'pattern', None):
        data_dir_sig = os.path.join(args.data_dir, f'p{p}_seed{args.seed}_split{getattr(args, "split_type", "random")}')
        pattern_path = os.path.join(data_dir_sig, args.pattern)
        for fname in sorted(glob(pattern_path), key=extract_dim):
            if not os.path.exists(fname):
                continue
            data = np.load(fname, allow_pickle=True)
            result = {
                'train_acc': data['train_acc'],
                'val_acc': data['val_acc'],
                'dim': int(data['dim']),
                'param_count': int(data['param_count']),
            }
            if 'mem_t_trace' in data:
                result['mem_t_trace'] = data['mem_t_trace']
            if 'mem_u_trace' in data:
                result['mem_u_trace'] = data['mem_u_trace']
                result['mem_trace'] = data['mem_u_trace']
            elif 'mem_trace' in data:
                result['mem_u_trace'] = data['mem_trace']
                result['mem_trace'] = data['mem_trace']
            results.append(result)
        return results

    # Default: use ResultsIndex
    if index is None:
        index = ResultsIndex(args.data_dir)

    entries = index.query(**_build_groks_filters(args, p))
    entries = _apply_dim_filter(entries, args, index)
    entries = sorted(entries, key=lambda e: _nested_get(e, 'dim') or 0)

    if not entries:
        return []

    print(f"Found {len(entries)} entries in index for p={p}, depth={args.depth}, heads={args.heads}, seed={args.seed}")

    for entry in entries:
        traces = index.load_traces(entry)
        result = {
            'train_acc': traces['train_acc'],
            'val_acc': traces['val_acc'],
            'dim': _nested_get(entry, 'dim'),
            'param_count': _nested_get(entry, 'param_count'),
        }
        if 'mem_t_trace' in traces:
            result['mem_t_trace'] = traces['mem_t_trace']
        if 'mem_u_trace' in traces:
            result['mem_u_trace'] = traces['mem_u_trace']
            result['mem_trace'] = traces['mem_u_trace']
        elif 'mem_trace' in traces:
            result['mem_u_trace'] = traces['mem_trace']
            result['mem_trace'] = traces['mem_trace']
        results.append(result)

    return results


def _load_capacity_results(args, index=None):
    """Load capacity results using ResultsIndex (or --files/--pattern fallback).

    Returns list of result dicts with: file, dim, n_samples, param_count, depth, heads,
                                        final_acc, bits_per_example, total_bits
    """
    p = args.p[0] if isinstance(args.p, list) else args.p
    results = []

    if getattr(args, 'files', None):
        fnames = args.files
    elif getattr(args, 'pattern', None):
        pattern_path = os.path.join(args.data_dir, args.pattern)
        fnames = sorted(glob(pattern_path), key=lambda f: (extract_dim(f), extract_samples(f)))
    else:
        fnames = None

    if fnames is not None:
        for fname in fnames:
            if not os.path.exists(fname):
                print(f"Warning: File not found: {fname}")
                continue
            data = np.load(fname, allow_pickle=True)
            results.append({
                'file': fname,
                'dim': int(data['dim']),
                'n_samples': int(data['n_samples']),
                'param_count': int(data['param_count']),
                'depth': int(data['depth']),
                'heads': int(data['heads']),
                'final_acc': float(data['final_acc']),
                'bits_per_example': float(data['final_bits_per_example']),
                'total_bits': float(data['total_bits_memorized'])
            })
        return sorted(results, key=lambda r: (r['dim'], r['n_samples']))

    # Default: use ResultsIndex
    if index is None:
        index = ResultsIndex(args.data_dir)

    entries = index.query(**_build_capacity_filters(args, p))

    if getattr(args, 'dims', None):
        dims_set = set(args.dims)
        entries = [e for e in entries if _nested_get(e, 'dim') in dims_set]
    if getattr(args, 'samples', None):
        samples_set = set(args.samples)
        entries = [e for e in entries if _nested_get(e, 'n_samples') in samples_set]

    entries = sorted(entries, key=lambda e: (_nested_get(e, 'dim') or 0, _nested_get(e, 'n_samples') or 0))

    if not entries:
        return []

    print(f"Found {len(entries)} capacity entries in index for p={p}, depth={getattr(args,'depth',2)}, heads={getattr(args,'heads',1)}, seed={args.seed}")

    for entry in entries:
        traces = index.load_traces(entry)
        results.append({
            'file': entry.get('_npz_path', ''),
            'dim': _nested_get(entry, 'dim'),
            'n_samples': _nested_get(entry, 'n_samples'),
            'param_count': _nested_get(entry, 'param_count'),
            'depth': _nested_get(entry, 'depth') or 2,
            'heads': _nested_get(entry, 'heads') or 1,
            'final_acc': float(traces.get('final_acc', 0)),
            'bits_per_example': float(traces.get('final_bits_per_example', 0)),
            'total_bits': float(traces.get('total_bits_memorized', 0)),
        })

    return results


def _load_speed_results_for_prime(p, index, speed_filters, batch_size=512, saturation_threshold=99.0, n_samples_filter=None):
    """Load speed results for one prime using ResultsIndex.

    Returns dict mapping dim -> list of result dicts (same structure as old load_speed_results).
    """
    filters = dict(speed_filters)
    filters['p'] = p
    if n_samples_filter is not None:
        filters['n_samples'] = n_samples_filter

    entries = index.query(**filters)

    all_results = {}
    for entry in entries:
        traces = index.load_traces(entry)
        dim = _nested_get(entry, 'dim')
        n_samples = _nested_get(entry, 'n_samples')
        param_count = _nested_get(entry, 'param_count')
        dataset_bits = _nested_get(entry, 'dataset_bits') or float(traces.get('dataset_bits', 0))

        if n_samples is None:
            continue
        steps_per_epoch = (n_samples + batch_size - 1) // batch_size

        saturation_epoch = None
        if 'train_acc_trace' in traces:
            saturation_epoch = compute_saturation_epoch_from_trace(
                traces['train_acc_trace'], saturation_threshold,
                steps_per_epoch, traces.get('steps_trace')
            )

        result = {
            'n_samples': n_samples,
            'dim': dim,
            'depth': _nested_get(entry, 'depth') or 2,
            'heads': _nested_get(entry, 'heads') or 1,
            'param_count': param_count,
            'p': p,
            'saturation_epoch': saturation_epoch,
            'final_acc': float(traces.get('final_acc', 0)),
            'dataset_bits': dataset_bits,
            'saturated': saturation_epoch is not None,
        }
        # also include train_acc_trace for plotting
        if 'train_acc_trace' in traces:
            result['train_acc_trace'] = traces['train_acc_trace']
        if 'steps_trace' in traces:
            result['steps_trace'] = traces['steps_trace']

        if dim not in all_results:
            all_results[dim] = []
        all_results[dim].append(result)

    return all_results


def groks(args):
    # Ensure args.p is a list and use only the first prime
    primes_list = args.p if isinstance(args.p, list) else [args.p]
    p = primes_list[0]

    if len(primes_list) > 1:
        print("Warning: groks command only uses the first prime. For multi-prime analysis, use 'primes' command.")

    plot_dir = os.path.join(args.plot_dir, f'p{p}_seed{args.seed}')
    print(f'Prime: p={p}, seed={args.seed}, depth={args.depth}, heads={args.heads}')

    # Determine show/save settings
    show = not args.no_show

    if args.list:
        index = ResultsIndex(args.data_dir)
        entries = index.query(**_build_groks_filters(args, p))
        print(f"\nFound {len(entries)} grokking results:")
        print("="*80)
        for e in sorted(entries, key=lambda e: _nested_get(e, 'dim') or 0):
            dim = _nested_get(e, 'dim')
            pc = _nested_get(e, 'param_count')
            wd = _nested_get(e, 'weight_decay')
            do = _nested_get(e, 'dropout')
            seed = _nested_get(e, 'seed')
            print(f"  dim={dim:3d}  params={pc:8,}  wd={wd}  dropout={do}  seed={seed}")
        print("="*80)

    if args.plot:
        # Check if --show-mem is also provided
        if args.show_mem:
            # Load file and plot with memorisation overlay
            if not os.path.exists(args.plot):
                print(f"File not found: {args.plot}")
                return
            data = np.load(args.plot)
            result = {
                'train_acc': data['train_acc'],
                'val_acc': data['val_acc'],
                'dim': int(data['dim']),
                'param_count': int(data['param_count'])
            }
            # Load both M_T and M_U traces
            if 'mem_t_trace' in data:
                result['mem_t_trace'] = data['mem_t_trace']
            if 'mem_u_trace' in data:
                result['mem_u_trace'] = data['mem_u_trace']
            # Legacy support for old 'mem_trace' field (was M_U)
            elif 'mem_trace' in data:
                result['mem_u_trace'] = data['mem_trace']

            save_path = None
            if args.save:
                os.makedirs(plot_dir, exist_ok=True)
                basename = os.path.splitext(os.path.basename(args.plot))[0]
                save_path = os.path.join(plot_dir, f'{basename}_with_mem.pdf')
            plot_grokking_with_memorization(result, save_path=save_path, show=show)
        else:
            plot_result(args.plot)

    # Load data
    index = ResultsIndex(args.data_dir)
    results = _load_groks_results(args, p, index)

    if not results:
        print("No results found matching the current filters.")
        print("  Try adjusting --weight-decay, --dropout, --depth, --heads, or --data-dir.")
        print("  Use --list to see what's available.")
        return

    # Calculate number of parameters to lower bound capacity of model to memorise all training data
    n, size = compute_dataset_size_bits(p, args.op, args.training_fraction)
    threshold_params = size / consts.C

    print(f"Found {len(results)} results to analyze:")
    for r in results:
        print(f"  - dim={r['dim']:3d}  params={r['param_count']:8,}")

    # Create plot directory if saving
    if args.save:
        os.makedirs(plot_dir, exist_ok=True)

    # Handle --show-mem when used standalone (without --plot)
    if args.show_mem:
        print(f"\nPlotting memorisation curves for {len(results)} results...")
        # Filter results with memorisation data (M_T or M_U)
        results_with_mem_t = [r for r in results if 'mem_t_trace' in r]
        results_with_mem_u = [r for r in results if 'mem_u_trace' in r]

        if not results_with_mem_t and not results_with_mem_u:
            print("No memorisation data found in selected files.")
            print("Run experiments with updated groks.py to get M_T data.")
            print("Run with --baseline flag to also get M_U data.")
        else:
            # Plot M_T curves over training (if available)
            if results_with_mem_t:
                save_path = os.path.join(plot_dir, f'mem_t_p{p}.pdf') if args.save else None
                plot_memorization_curves(results_with_mem_t, mem_key='mem_t_trace', title='Total Memorisation (M_T)',
                                        save_path=save_path, show=show)

                # Plot final M_T vs parameter count
                save_path = os.path.join(plot_dir, f'mem_t_vs_params_p{p}.pdf') if args.save else None
                plot_max_memorization_vs_params(results_with_mem_t, mem_key='mem_t_trace', title='Final M_T vs Parameters',
                                                 save_path=save_path, show=show)

            # Plot M_U curves over training (if available)
            if results_with_mem_u:
                save_path = os.path.join(plot_dir, f'mem_u_p{p}.pdf') if args.save else None
                plot_memorization_curves(results_with_mem_u, mem_key='mem_u_trace', title='Unintended Memorisation (M_U)',
                                        save_path=save_path, show=show)

                # Plot final M_U vs parameter count
                save_path = os.path.join(plot_dir, f'mem_u_vs_params_p{p}.pdf') if args.save else None
                plot_max_memorization_vs_params(results_with_mem_u, mem_key='mem_u_trace', title='Final M_U vs Parameters',
                                                 save_path=save_path, show=show)

    if args.separate:
        print(f"Plotting separate curves for {len(results)} results...")
        save_path = os.path.join(plot_dir, f'grokking_separate_p{p}.pdf') if args.save else None
        plot_separate_curves(results, save_path=save_path, show=show)
        # Check if we should overlay memorisation
        if args.show_mem:
            print(f"Plotting separate curves with memorisation for {len(results)} results...")
            save_path = os.path.join(plot_dir, f'grokking_separate_mem_p{p}.pdf') if args.save else None
            plot_separate_curves_with_memorization(results, save_path=save_path, show=show)

    if args.combined:
        print(f"Plotting combined curves for {len(results)} results...")
        save_path = os.path.join(plot_dir, f'grokking_combined_p{p}.pdf') if args.save else None
        plot_combined_curves(results, save_path=save_path, show=show)

    if args.delay:
        # Standard delay plot
        save_path = os.path.join(plot_dir, f'grokking_delay_p{p}.pdf') if args.save else None
        plot_delay_util(results, threshold_train=args.threshold_train, threshold_val=args.threshold_val,
                        save_path=save_path, show=show, threshold_params=threshold_params)
        # Check if we should also plot delay vs memorisation
        if args.show_mem:
            # Plot delay vs memorisation
            save_path = os.path.join(plot_dir, f'delay_vs_memorisation_p{p}.pdf') if args.save else None
            plot_delay_vs_memorization(results, threshold_train=args.threshold_train, threshold_val=args.threshold_val,
                                      save_path=save_path, show=show)
            # Plot delay and memorisation vs parameter count on same graph
            save_path = os.path.join(plot_dir, f'delay_and_mem_vs_params_p{p}.pdf') if args.save else None
            plot_delay_and_memorization_vs_params(results, threshold_train=args.threshold_train, threshold_val=args.threshold_val,
                                                 save_path=save_path, show=show)

    if args.integral:
        # Integral plot
        save_path = os.path.join(plot_dir, f'grokking_integral_p{p}.pdf') if args.save else None
        plot_grokking_integral(results, threshold_train=args.threshold_train, threshold_val=args.threshold_val,
                              save_path=save_path, show=show, threshold_params=threshold_params)

    if args.critical:
        save_path = os.path.join(plot_dir, f'critical_capacity_p{p}.pdf') if args.save else None
        critical_params = plot_grokking_critical_capacity(results, threshold_train=args.threshold_train,
                                                          threshold_val=args.threshold_val,
                                                          delay_threshold=args.delay_threshold,
                                                          save_path=save_path, show=show)
        if critical_params:
            print(f"\n{'='*80}")
            print(f"CRITICAL CAPACITY: {critical_params:,.0f} parameters")
            print(f"{'='*80}")

    if args.time:
        save_path = os.path.join(plot_dir, f'grokking_time_p{p}.pdf') if args.save else None
        plot_time_util(results, threshold_val=args.threshold_val, save_path=save_path, show=show)

    if args.speed:
        # Calculate dataset size
        n, size = compute_dataset_size_bits(p, args.op, args.training_fraction)
        threshold_params = size / consts.C

        # Load speed data using ResultsIndex
        speed_index = ResultsIndex(args.data_dir)
        speed_filters = _build_speed_filters(args)
        speed_filters['p'] = p
        speed_filters['n_samples'] = lambda x: x is not None and abs(x - n) <= 1
        speed_entries = speed_index.query(**speed_filters)

        speed_data = []
        for entry in speed_entries:
            traces = speed_index.load_traces(entry)
            n_samples = _nested_get(entry, 'n_samples')
            param_count = _nested_get(entry, 'param_count')
            if n_samples is None:
                continue
            steps_per_epoch = (n_samples + args.batch_size - 1) // args.batch_size
            saturation_epoch = None
            if 'train_acc_trace' in traces:
                saturation_epoch = compute_saturation_epoch_from_trace(
                    traces['train_acc_trace'], args.saturation_threshold,
                    steps_per_epoch, traces.get('steps_trace')
                )
            if saturation_epoch is not None:
                speed_data.append({
                    'dim': _nested_get(entry, 'dim'),
                    'param_count': param_count,
                    'saturation_epoch': saturation_epoch,
                    'n_samples': n_samples,
                    'saturated': True
                })

        if not speed_data:
            print(f"Warning: No speed data found for p={p} with n_samples≈{n}")

        # Plot the speed-grok intersect graph
        save_path = os.path.join(plot_dir, f'delay_with_speed_p{p}.pdf') if args.save else None
        plot_grokking_delay_with_speed(
            results,
            speed_data,
            threshold_train=args.threshold_train,
            threshold_val=args.threshold_val,
            saturation_threshold=args.saturation_threshold,
            threshold_params=threshold_params,
            batch_size=args.batch_size,
            n_train_samples=n,
            save_path=save_path,
            show=show
        )


# =============================================================================
# Capacity Experiment Visualization Functions
# =============================================================================

def list_capacity_results(data_dir: str, pattern: str = 'capacity_dim*.npz') -> List[Dict]:
    """List all saved capacity experiment results."""
    files = sorted(glob(os.path.join(data_dir, pattern)))
    
    if not files:
        print(f"No results found matching pattern: {pattern}")
        return []
    
    print(f"\nFound {len(files)} capacity result files:")
    print("="*80)
    
    results = []
    for i, fname in enumerate(files):
        data = np.load(fname)
        dim = int(data['dim'])
        n_samples = int(data['n_samples'])
        param_count = int(data['param_count'])
        depth = int(data['depth'])
        heads = int(data['heads'])
        final_acc = float(data['final_acc'])
        bits_per_example = float(data['final_bits_per_example'])
        total_bits = float(data['total_bits_memorized'])
        
        results.append({
            'file': fname,
            'dim': dim,
            'n_samples': n_samples,
            'param_count': param_count,
            'depth': depth,
            'heads': heads,
            'final_acc': final_acc,
            'bits_per_example': bits_per_example,
            'total_bits': total_bits,
            'data': data
        })
        
        print(f"{i:2d}. {os.path.basename(fname)}")
        print(f"    dim={dim:3d}, samples={n_samples:6d}, params={param_count:8,}")
        print(f"    Acc={final_acc:.1%}, bits/ex={bits_per_example:.2f}, total={total_bits:,.0f}")
    
    print("="*80)
    return results


def plot_capacity_result(result_file: str):
    """Plot training curves for a single capacity experiment."""
    if not os.path.exists(result_file):
        print(f"File not found: {result_file}")
        return
    
    data = np.load(result_file)
    train_loss = data['train_loss_trace']
    train_acc = data['train_acc_trace']
    bits_trace = data['bits_trace']
    dim = int(data['dim'])
    n_samples = int(data['n_samples'])
    param_count = int(data['param_count'])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Loss curve
    ax = axes[0]
    ax.plot(train_loss, color='#d95f02', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax.set_title('Training Loss', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Accuracy curve
    ax = axes[1]
    ax.plot(train_acc * 100, color='#1b9e77', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Training Accuracy', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Bits memorized
    ax = axes[2]
    ax.plot(bits_trace, color='#7570b3', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Bits per Example', fontsize=12)
    ax.set_title('Memorisation', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Capacity Experiment: dim={dim}, samples={n_samples}, params={param_count:,}', 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


# Capacity plotting functions are now imported from plotting.py


def capacity(args):
    """Handle capacity subcommand."""
    p = args.p[0] if isinstance(args.p, list) else args.p

    plot_dir = os.path.join(args.plot_dir, f'p{p}_seed{args.seed}')
    print(f'Data directory: {args.data_dir}')
    print(f'Plot directory: {plot_dir}')

    # List results
    if args.list:
        index = ResultsIndex(args.data_dir)
        entries = index.query(**_build_capacity_filters(args, p))
        print(f"\nFound {len(entries)} capacity results:")
        print("="*80)
        for e in sorted(entries, key=lambda e: (_nested_get(e,'dim') or 0, _nested_get(e,'n_samples') or 0)):
            dim = _nested_get(e, 'dim')
            ns = _nested_get(e, 'n_samples')
            pc = _nested_get(e, 'param_count')
            print(f"  dim={dim:3d}  n_samples={ns:6d}  params={pc:8,}")
        print("="*80)
        return

    # Plot single result
    if args.plot:
        plot_capacity_result(args.plot)
        return

    # Load results
    index = ResultsIndex(args.data_dir)
    results = _load_capacity_results(args, index)

    if not results:
        print("No capacity results found matching the current filters.")
        print("  Try adjusting --depth, --heads, --dropout, --weight-decay, or --data-dir.")
        return

    print(f"\nFound {len(results)} results to analyze:")
    for r in results:
        print(f"  - dim={r['dim']:3d}  n_samples={r['n_samples']:6d}  params={r['param_count']:8,}")

    # Group by dimension for curves plot
    all_results = {}
    for r in results:
        dim = r['dim']
        if dim not in all_results:
            all_results[dim] = []
        all_results[dim].append(r)

    # Generate requested plots
    os.makedirs(plot_dir, exist_ok=True)
    
    if args.curves:
        print("\nPlotting capacity curves (Morris et al. style)...")
        save_path = os.path.join(plot_dir, 'capacity_curves.pdf') if args.save else None
        saturation_points = plot_capacity_curves(
            all_results, p=p, save_path=save_path, show=not args.no_show
        )
        
        # Also plot estimation if we have enough points
        if len(saturation_points) >= 2:
            save_path = os.path.join(plot_dir, 'capacity_estimation.pdf') if args.save else None
            C, intercept, r_squared = plot_capacity_estimation(saturation_points, save_path=save_path, show=not args.no_show)
            print(f"\nCapacity: C = {C:.2f} bits/parameter (R² = {r_squared:.3f})")
            sign = '+' if intercept >= 0 else '−'
            print(f"Linear fit: bits = {C:.2f} × params {sign} {abs(intercept):.0f}")
    
    if args.accuracy:
        print("\nPlotting bits vs accuracy...")
        save_path = os.path.join(plot_dir, 'bits_vs_accuracy.pdf') if args.save else None
        plot_bits_vs_accuracy(results, save_path=save_path, show=not args.no_show)
    
    if args.summary:
        print("\n" + "="*70)
        print("CAPACITY SUMMARY")
        print("="*70)
        
        for dim in sorted(all_results.keys()):
            dim_results = all_results[dim]
            param_count = dim_results[0]['param_count']
            max_bits = max(r['total_bits'] for r in dim_results)
            max_acc = max(r['final_acc'] for r in dim_results)
            print(f"dim={dim:3d}: {param_count:8,} params | "
                  f"max bits: {max_bits:10,.0f} | "
                  f"bits/param: {max_bits/param_count:.2f} | "
                  f"max acc: {max_acc:.1%}")
        
        # Overall capacity estimate
        saturation_points = [(all_results[dim][0]['param_count'], 
                             max(r['total_bits'] for r in all_results[dim]))
                            for dim in all_results]
        C, intercept, r_squared = estimate_capacity(saturation_points)
        print("="*70)
        print(f"Capacity: C = {C:.2f} bits/parameter (R² = {r_squared:.3f})")
        sign = '+' if intercept >= 0 else '−'
        print(f"Linear fit: bits = {C:.2f} × params {sign} {abs(intercept):.0f}")
        print("="*70)


# =============================================================================
# Speed Experiment Visualization Functions
# =============================================================================

def speed(args):
    """Handle speed subcommand."""
    # Ensure args.p is a list
    primes_list = args.p if isinstance(args.p, list) else [args.p]
    show = not args.no_show

    index = ResultsIndex(args.data_dir)
    speed_filters = _build_speed_filters(args)

    # List results
    if args.list:
        for p in primes_list:
            all_results = _load_speed_results_for_prime(p, index, speed_filters,
                                                         args.batch_size, args.saturation_threshold)
            print(f'\n{"="*80}')
            print(f'Prime p={p}')
            print(f'{"="*80}')
            if not all_results:
                print("No speed results found.")
                continue
            for dim in sorted(all_results.keys()):
                for r in all_results[dim]:
                    print(f"  dim={dim:3d}  n_samples={r['n_samples']:6d}  params={r['param_count']:8,}  sat_epoch={r['saturation_epoch']}")
        return

    # For --fraction mode, collect data from all primes to plot together
    if args.fraction:
        print("\nDeprecation notice: 'speed --fraction' will be removed. Use 'primes --speed' instead (supports seed aggregation).")
        print("\nPlotting saturation time vs capacity fraction (separate line for each prime)...")

        # Collect data from all primes, keyed by (p, dim)
        combined_results = {}
        for p in primes_list:
            all_results = _load_speed_results_for_prime(p, index, speed_filters,
                                                         args.batch_size, args.saturation_threshold)
            if not all_results:
                print(f"Warning: No files found for p={p}")
                continue

            print(f"\nLoaded {sum(len(v) for v in all_results.values())} results for p={p}")

            for dim, dim_results in all_results.items():
                for result in dim_results:
                    if result.get('saturation_epoch') is None:
                        continue
                    key = (p, dim)
                    if key not in combined_results:
                        combined_results[key] = []
                    combined_results[key].append(result)

        if not combined_results:
            print("No data found for any prime")
            return

        # Plot all data together
        plot_dir = args.plot_dir
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, 'saturation_time_vs_capacity_fraction_all_primes.pdf') if args.save else None
        exponent, coefficient, r_squared = plot_saturation_time_vs_capacity_fraction(
            combined_results,
            C=consts.C,
            save_path=save_path,
            show=show
        )

        print("\n" + "="*60)
        print("CAPACITY FRACTION ANALYSIS (ALL PRIMES)")
        print("="*60)
        print(f"Power law fit: epochs = {coefficient:.1f} × f^{exponent:.2f}")
        print(f"Exponent: {exponent:.2f}")
        print(f"R²: {r_squared:.3f}")
        print(f"Capacity constant C = {consts.C:.2f} bits/param")
        print("="*60)
        return

    # For other modes (--curves, --rate, --combined, --summary), process each prime separately
    for p in primes_list:
        plot_dir_p = os.path.join(args.plot_dir, f'p{p}')

        print(f'\n{"="*80}')
        print(f'Processing prime p={p}')
        print(f'{"="*80}')

        # Load results
        all_results = _load_speed_results_for_prime(p, index, speed_filters,
                                                     args.batch_size, args.saturation_threshold)

        if not all_results:
            print("No speed results found for this prime with the current filters.")
            print("  Try adjusting --weight-decay, --dropout, --depth, --heads, or --data-dir.")
            continue

        print(f"Found {sum(len(v) for v in all_results.values())} results across {len(all_results)} dimensions")

        # Generate requested plots
        os.makedirs(plot_dir_p, exist_ok=True)
        plot_dir = plot_dir_p  # for save_path generation below

        if args.curves:
            print(f"\nPlotting learning speed curves for p={p}...")
            save_path = os.path.join(plot_dir, f'learning_speed_curves_p{p}.pdf') if args.save else None
            speed_estimates = plot_learning_speed_curves(
                all_results,
                p=p,
                save_path=save_path,
                show=show
            )

            # Plot epochs to saturation vs model parameters
            print(f"\nPlotting epochs to saturation vs model parameters for p={p}...")
            save_path = os.path.join(plot_dir, f'saturation_epochs_vs_params_p{p}.pdf') if args.save else None
            plot_saturation_epochs_vs_params(
                all_results,
                save_path=save_path,
                show=show
            )

            # Also plot speed vs model size if we have multiple model sizes
            if len(speed_estimates) >= 2:
                print(f"\nPlotting speed vs model size for p={p}...")
                save_path = os.path.join(plot_dir, f'speed_vs_model_size_p{p}.pdf') if args.save else None
                b, log_a, r_squared = plot_speed_vs_model_size(
                    speed_estimates,
                    save_path=save_path,
                    show=show
                )

                print("\n" + "="*60)
                print(f"SPEED SCALING (p={p})")
                print("="*60)
                print(f"Power law exponent: {b:.3f}")
                print(f"R²: {r_squared:.3f}")
                if b < 0:
                    print(f"Larger models learn FASTER (fewer epochs per bit)")
                else:
                    print(f"Larger models learn SLOWER (more epochs per bit)")

        if args.combined:
            print(f"\nPlotting combined speed analysis for p={p}...")
            save_path = os.path.join(plot_dir, f'speed_analysis_combined_p{p}.pdf') if args.save else None
            speed_estimates = plot_combined_speed_analysis(
                all_results,
                p=p,
                save_path=save_path,
                show=show
            )

        if args.rate:
            print(f"\nPlotting dT/dS vs dataset size for p={p} (k={args.rate_k})...")
            save_path = os.path.join(plot_dir, f'rate_vs_dataset_size_p{p}_k{args.rate_k}.pdf') if args.save else None
            rate_data = plot_rate_vs_dataset_size(
                all_results,
                k=args.rate_k,
                save_path=save_path,
                show=show
            )

            if rate_data:
                print("\n" + "="*60)
                print(f"RATE ESTIMATION (p={p}, k={args.rate_k} samples)")
                print("="*60)
                for dim in sorted(rate_data.keys()):
                    rates = rate_data[dim]
                    if rates:
                        avg_rate = np.mean([r[1] for r in rates])
                        print(f"dim={dim:3d}: avg dT/dS = {avg_rate:.2f} epochs/bit")
                print("="*60)
            else:
                print("No paired data points found. Run speed.py with --rate to generate paired data.")

        if args.summary:
            print("\n" + "="*70)
            print(f"SPEED SUMMARY (p={p})")
            print("="*70)

            for dim in sorted(all_results.keys()):
                results = all_results[dim]
                param_count = results[0]['param_count']

                # Average epochs per bit across dataset sizes
                saturated = [r for r in results if r.get('saturated', True)]
                if saturated:
                    avg_speed = np.mean([r['saturation_epoch'] / r['dataset_bits']
                                        for r in saturated if r['dataset_bits'] > 0])
                    print(f"dim={dim:3d}: {param_count:8,} params, "
                          f"avg speed: {avg_speed:.2f} epochs/bit")

            print("="*70)


# =============================================================================
# Primes Experiment Visualization Functions
# =============================================================================

def aggregate_grokking_results_across_seeds(
    p_prime: int,
    index,
    filters: dict,
    threshold_train: float = 99.0,
    threshold_val: float = 99.0,
    saturation_threshold: float = 99.0,
    use_min_delay: bool = False,
    dims=None,
) -> list:
    """Load and aggregate grokking results across all seeds for a given prime.

    Uses ResultsIndex to find all entries matching (p_prime, **filters).
    Seeds are implicit in the query results — no directory scanning needed.

    Returns:
        List of aggregated results, one per (dim, param_count). Each contains:
            delay, train_epoch, val_epoch, epochs_to_grok, dim, param_count, n_seeds
    """
    query_filters = dict(filters)
    query_filters['p'] = p_prime
    entries = index.query(experiment_type='groks', **{k: v for k, v in query_filters.items() if k != 'experiment_type'})

    if not entries:
        return []

    if dims is not None:
        dims_set = set(dims)
        entries = [e for e in entries if _nested_get(e, 'dim') in dims_set]

    seeds_seen = set()
    for e in entries:
        s = _nested_get(e, 'seed')
        if s is not None:
            seeds_seen.add(s)
    if seeds_seen:
        print(f"  Found {len(entries)} entries across {len(seeds_seen)} seed(s) for p={p_prime}: {sorted(seeds_seen)}")
    else:
        print(f"  Found {len(entries)} entries for p={p_prime}")

    # Collect delays and epoch info by (dim, param_count) key
    delays_by_config = {}

    for entry in entries:
        traces = index.load_traces(entry)
        dim = _nested_get(entry, 'dim')
        param_count = _nested_get(entry, 'param_count')
        key = (dim, param_count)

        # Calculate delay for this entry
        train_epoch, val_epoch, delay = calculate_grokking_delay(
            traces['train_acc'], traces['val_acc'], threshold_train, threshold_val
        )

        # Calculate epochs_to_grok using saturation_threshold (for the speed curve)
        epochs_to_grok = None
        val_acc = traces['val_acc']
        for epoch, acc in enumerate(val_acc):
            if acc >= saturation_threshold:
                epochs_to_grok = epoch
                break

        if key not in delays_by_config:
            delays_by_config[key] = {
                'results': [],
                'dim': dim,
                'param_count': param_count
            }

        if delay is not None:
            delays_by_config[key]['results'].append({
                'delay': delay,
                'train_epoch': train_epoch,
                'val_epoch': val_epoch,
                'epochs_to_grok': epochs_to_grok
            })

    # Aggregate delays across seeds
    aggregated_results = []
    for key, data in delays_by_config.items():
        if data['results']:  # Only include configs where at least one entry had valid delay
            delays = [r['delay'] for r in data['results']]

            if use_min_delay:
                aggregated_delay = min(delays)
                best_idx = delays.index(aggregated_delay)
                selected_result = data['results'][best_idx]
            else:
                aggregated_delay = np.mean(delays)
                closest_idx = np.argmin([abs(d - aggregated_delay) for d in delays])
                selected_result = data['results'][closest_idx]

            epochs_to_grok_values = [r['epochs_to_grok'] for r in data['results'] if r['epochs_to_grok'] is not None]
            avg_epochs_to_grok = float(np.mean(epochs_to_grok_values)) if epochs_to_grok_values else None

            aggregated_results.append({
                'delay': aggregated_delay,
                'train_epoch': selected_result['train_epoch'],
                'val_epoch': selected_result['val_epoch'],
                'epochs_to_grok': avg_epochs_to_grok,
                'dim': data['dim'],
                'param_count': data['param_count'],
                'n_seeds': len(data['results'])
            })

    return aggregated_results


def aggregate_speed_results_across_seeds(
    p_prime: int,
    n_prime: int,
    index,
    filters: dict,
    batch_size: int = 512,
    saturation_threshold: float = 99.0,
) -> list:
    """Load and aggregate speed results across all seeds for a given prime.

    Uses ResultsIndex. Filters by n_samples within ±1 of n_prime.

    Returns:
        List of aggregated results per (dim, param_count):
            {dim, param_count, saturation_epoch (mean), n_samples, saturated, n_seeds}
    """
    # Filter by n_samples with ±1 tolerance (rounding)
    n_filter = lambda x: x is not None and abs(x - n_prime) <= 1

    query_filters = dict(filters)
    query_filters['p'] = p_prime
    query_filters['n_samples'] = n_filter
    # Remove experiment_type from filters if present (we add it explicitly)
    query_filters.pop('experiment_type', None)

    entries = index.query(experiment_type='speed', **query_filters)

    if not entries:
        return []

    seeds_seen = set()
    for e in entries:
        s = _nested_get(e, 'seed')
        if s is not None:
            seeds_seen.add(s)
    print(f"  Found {len(entries)} speed entries across {len(seeds_seen)} seed(s) for p={p_prime} (n≈{n_prime})")

    # Collect saturation epochs by (dim, param_count) key
    results_by_config = {}

    for entry in entries:
        traces = index.load_traces(entry)
        dim = _nested_get(entry, 'dim')
        n_samples = _nested_get(entry, 'n_samples')
        param_count = _nested_get(entry, 'param_count')
        dataset_bits = _nested_get(entry, 'dataset_bits') or float(traces.get('dataset_bits', 0))

        if n_samples is None:
            continue
        steps_per_epoch = (n_samples + batch_size - 1) // batch_size

        saturation_epoch = None
        if 'train_acc_trace' in traces:
            saturation_epoch = compute_saturation_epoch_from_trace(
                traces['train_acc_trace'], saturation_threshold,
                steps_per_epoch, traces.get('steps_trace')
            )

        if saturation_epoch is None:
            continue

        key = (dim, param_count)
        if key not in results_by_config:
            results_by_config[key] = {
                'saturation_epochs': [],
                'dim': dim,
                'param_count': param_count,
                'n_samples': n_samples,
                'dataset_bits': dataset_bits,
            }
        results_by_config[key]['saturation_epochs'].append(saturation_epoch)

    aggregated_results = []
    for key, data in results_by_config.items():
        saturation_epoch = float(np.mean(data['saturation_epochs']))
        aggregated_results.append({
            'dim': data['dim'],
            'param_count': data['param_count'],
            'saturation_epoch': saturation_epoch,
            'n_samples': data['n_samples'],
            'dataset_bits': data['dataset_bits'],
            'saturated': True,
            'n_seeds': len(data['saturation_epochs'])
        })

    return aggregated_results


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def _resolve_primes(args) -> list:
    """Return sorted list of primes to analyse from --p or --min-prime/--max-prime.

    When --p is given, returns it directly. Otherwise generates all primes in
    [--min-prime, --max-prime] and filters to those with matching groks data.
    With --match-table: filters against entries in the match table by (op, train_fraction).
    Without --match-table: queries ResultsIndex for groks experiments matching
    (op, train_fraction, depth, heads).
    """
    if args.p:
        return sorted(args.p if isinstance(args.p, list) else [args.p])

    candidates = [n for n in range(args.min_prime, args.max_prime + 1) if _is_prime(n)]

    op = getattr(args, 'op', '/')
    tf = getattr(args, 'training_fraction', 0.5)

    if getattr(args, 'match_table', None):
        matches = load_match_table(args.match_table)
        valid_p = {
            m['p'] for m in matches
            if m.get('operation') == op
            and abs(m.get('train_fraction', tf) - tf) < 1e-9
        }
    else:
        index = ResultsIndex(getattr(args, 'data_dir', 'data'))
        depth = getattr(args, 'depth', 2)
        heads = getattr(args, 'heads', 1)
        # Legacy entries may omit operation (implying '/') or train_fraction (implying 0.5)
        op_filter = (lambda x: x is None or x == op) if op == '/' else op
        tf_filter = (lambda x: x is None or x == tf) if tf == 0.5 else tf
        entries = index.query(
            experiment_type='groks',
            operation=op_filter,
            train_fraction=tf_filter,
            depth=depth,
            heads=heads,
        )
        valid_p = set()
        for e in entries:
            p_val, found = index._get_nested(e, 'p')
            if found and p_val is not None:
                valid_p.add(p_val)

    found = sorted(p for p in candidates if p in valid_p)
    print(f"Auto-detected {len(found)} primes in [{args.min_prime}, {args.max_prime}]: {found}")
    return found


def _primes_from_match_table(args):
    """Load paired (groks, speed) data from a match table and run the requested analysis.

    This is an alternative to the glob-based loading in primes(). It reads
    ExperimentMatch dicts from the JSON produced by run_config.py or
    scripts/migrate_legacy_data.py, loads the corresponding .npz traces,
    and builds the same data structures used by the existing plotting functions.

    Currently supports: --speed, --delay, --correlation, --cv analysis modes.
    For --critical, falls back to the glob-based primes() path.
    """
    import numpy as np

    matches = load_match_table(args.match_table)
    index = ResultsIndex(args.data_dir)

    show = not args.no_show
    plot_dir = args.plot_dir
    os.makedirs(plot_dir, exist_ok=True)

    # Build per-prime dicts of groks and speed data.
    # groks_data[p] = list of {dim, param_count, train_acc, val_acc}
    # speed_data[p] = list of {param_count, n_samples, dataset_bits, saturation_epoch}
    groks_data = {}
    speed_data = {}

    for m in matches:
        p = m['p']
        # Load groks traces
        if os.path.exists(m['groks_npz_path']):
            g = np.load(m['groks_npz_path'], allow_pickle=True)
            groks_entry = {
                'dim': int(g['dim']),
                'param_count': m['param_count_groks'],
                'train_acc': g['train_acc'],
                'val_acc': g['val_acc'],
                'dataset_bits': m['dataset_bits'],
                'capacity_fraction': m['capacity_fraction'],
            }
            groks_data.setdefault(p, []).append(groks_entry)

        # Load speed traces
        if os.path.exists(m['speed_npz_path']):
            s = np.load(m['speed_npz_path'], allow_pickle=True)
            speed_entry = {
                'param_count': m['param_count_speed'],
                'n_samples': m['n_equiv'],
                'dataset_bits': m['dataset_bits'],
                'saturation_epoch': float(s['saturation_epoch']),
                'capacity_fraction': m['capacity_fraction'],
            }
            speed_data.setdefault(p, []).append(speed_entry)

    if not groks_data and not speed_data:
        print("No data loaded from match table — check that .npz paths exist.")
        return

    primes_list = sorted(set(list(groks_data.keys()) + list(speed_data.keys())))
    if args.p:
        filter_p = set(args.p if isinstance(args.p, list) else [args.p])
        primes_list = [p for p in primes_list if p in filter_p]
    elif getattr(args, 'min_prime', None) is not None:
        primes_list = [p for p in primes_list
                       if args.min_prime <= p <= args.max_prime]
        print(f"Auto-detected {len(primes_list)} primes in "
              f"[{args.min_prime}, {args.max_prime}]: {primes_list}")

    print(f"Match table loaded: {len(matches)} pairs across {len(primes_list)} primes")

    # Delegate to the same analysis functions used by the glob-based path.
    # We reconstruct a minimal args-like structure and call into the relevant
    # plotting utilities directly.
    if args.speed:
        all_speed_results = []
        for p in primes_list:
            for se in speed_data.get(p, []):
                all_speed_results.append({
                    'p': p,
                    'param_count': se['param_count'],
                    'dataset_bits': se['dataset_bits'],
                    'saturation_epoch': se['saturation_epoch'],
                    'capacity_fraction': se['capacity_fraction'],
                })
        if all_speed_results:
            plot_saturation_epochs_vs_params(
                all_speed_results,
                save=getattr(args, 'save', False),
                show=show,
                plot_dir=plot_dir,
            )
        else:
            print("No speed data found in match table.")

    elif args.delay:
        all_delay_results = []
        for p in primes_list:
            n_equiv, K_mem = compute_dataset_size_bits(p, getattr(args, 'op', '/'),
                                                       getattr(args, 'training_fraction', 0.5))
            for ge in groks_data.get(p, []):
                delay = calculate_grokking_delay(
                    ge['train_acc'], ge['val_acc'],
                    threshold_train=getattr(args, 'threshold_train', 99.0),
                    threshold_val=getattr(args, 'threshold_val', 97.0),
                )
                all_delay_results.append({
                    'p': p,
                    'param_count': ge['param_count'],
                    'capacity_fraction': ge['capacity_fraction'],
                    'delay': delay,
                })
        if all_delay_results:
            plot_delay_vs_capacity_fraction(
                all_delay_results,
                save=getattr(args, 'save', False),
                show=show,
                plot_dir=plot_dir,
            )
        else:
            print("No grokking data found in match table.")

    else:
        # For unsupported modes, warn and fall through to the glob-based path
        print("Note: --match-table is not yet implemented for this analysis mode. "
              "Falling back to glob-based loading.")
        primes(args)


def primes(args):
    """Handle primes subcommand for multi-prime analysis."""
    # Dispatch to match-table path when --match-table is provided
    if getattr(args, 'match_table', None):
        _primes_from_match_table(args)
        return

    if not args.p and (getattr(args, 'min_prime', None) is None
                       or getattr(args, 'max_prime', None) is None):
        raise SystemExit("error: specify either --p or both --min-prime and --max-prime")

    primes_list = _resolve_primes(args)

    if len(primes_list) < 2:
        print("Warning: primes command works best with multiple primes (--p p1 p2 p3 ...)")

    show = not args.no_show

    # Build ResultsIndex and shared filters once for all analysis modes
    index = ResultsIndex(args.data_dir)
    filters = {
        'operation': _legacy_filter(getattr(args, 'op', '/'), '/'),
        'train_fraction': _legacy_filter(getattr(args, 'training_fraction', 0.5), 0.5),
        'depth': getattr(args, 'depth', 2),
        'heads': getattr(args, 'heads', 1),
    }
    _add_filter(filters, 'weight_decay', getattr(args, 'weight_decay', None))
    _add_filter(filters, 'dropout', getattr(args, 'dropout', None))

    # Create plot directory
    plot_dir = args.plot_dir
    os.makedirs(plot_dir, exist_ok=True)

    # Process based on flags
    if args.critical:
        print("\n" + "="*80)
        print("EMPIRICAL CRITICAL PARAMETER COUNT VS PRIME")
        print("="*80)

        critical_data = []

        for p_prime in primes_list:
            print(f"\nProcessing prime p={p_prime}")

            # Calculate dataset size for this prime
            n_prime, size_prime = compute_dataset_size_bits(p_prime, args.op, args.training_fraction)

            # Aggregate grokking results across all seeds (computes minimum delay per config)
            prime_results = aggregate_grokking_results_across_seeds(
                p_prime, index, filters,
                args.threshold_train, args.threshold_val, args.saturation_threshold,
                use_min_delay=True, dims=getattr(args, 'dims', None)
            )

            if not prime_results:
                print(f"Warning: No grokking results found for p={p_prime}")
                continue

            # Filter by max_dim if specified
            if args.max_dim is not None:
                prime_results = [r for r in prime_results if r['dim'] <= args.max_dim]
                if not prime_results:
                    print(f"Warning: No results found for p={p_prime} with dim <= {args.max_dim}")
                    continue

            # Compute empirical critical params (first param size where minimum delay across seeds is non-zero)
            empirical_critical_params = None
            delay_data = []
            for result in prime_results:
                delay_data.append({
                    'param_count': result['param_count'],
                    'delay': result['delay']
                })

            # Sort by parameter count and find first param size where min_delay > 0
            # AND all subsequent param sizes also have min_delay > 0
            delay_data.sort(key=lambda x: x['param_count'])
            if delay_data:
                # Start from the end (largest params) and go backwards to find the LAST zero delay
                # This ensures all subsequent points have delay > 0
                last_zero_idx = -1
                for i in range(len(delay_data) - 1, -1, -1):
                    if delay_data[i]['delay'] == 0:
                        last_zero_idx = i
                        break

                # Critical point is the first param count after the last zero delay
                # (guarantees all subsequent points have delay > 0)
                if last_zero_idx >= 0 and last_zero_idx + 1 < len(delay_data):
                    empirical_critical_params = delay_data[last_zero_idx + 1]['param_count']
                # If no zero delay found, all points have delay > 0, so take the first
                elif last_zero_idx == -1 and delay_data[0]['delay'] > 0:
                    empirical_critical_params = delay_data[0]['param_count']

            if empirical_critical_params is not None:
                critical_data.append({
                    'p': p_prime,
                    'critical_params': empirical_critical_params,
                    'dataset_bits': size_prime
                })
                print(f"  Empirical critical params (first min delay > 0): {empirical_critical_params:,.0f}")
                print(f"  Dataset size: {size_prime:,.0f} bits")
            else:
                print(f"  Warning: Could not find empirical critical params for p={p_prime}")

        if critical_data:
            print("\n" + "="*80)
            print("Plotting empirical critical parameter count vs prime")
            print("="*80)

            save_path = os.path.join(plot_dir, 'empirical_critical_params_vs_prime.pdf') if args.save else None
            plot_critical_params_vs_prime(
                critical_data,
                title='Empirical Critical Parameter Count vs Prime',
                save_path=save_path,
                show=show
            )

            print("\n" + "="*80)
            print("Plotting empirical critical parameter count vs dataset size")
            print("="*80)

            save_path = os.path.join(plot_dir, 'empirical_critical_params_vs_dataset_size.pdf') if args.save else None
            slope, intercept, r_squared = plot_critical_params_vs_dataset_size(
                critical_data,
                title='Empirical Critical Parameter Count vs Dataset Size',
                save_path=save_path,
                show=show
            )

            if slope > 0:
                print(f"\nCapacity interpretation: C ≈ {1/slope:.2f} bits/parameter")
        else:
            print("\nWarning: No valid critical parameter data found for any prime")

    elif args.groks:
        print("\n" + "="*80)
        print("SPEED-BASED PREDICTED VS EMPIRICAL GROKKING POINTS")
        print("="*80)

        # Compute global exponential fit if --predicted-speed or --global-fit is specified
        speed_fit_params = None  # (a, b) for epochs = a * exp(b * f)
        global_fit_params = None  # Same format, for --global-fit
        if args.predicted_speed or args.global_fit:
            print("\nComputing overall exponential fit from all speed data...")
            
            # Collect all speed data across all primes
            all_speed_f = []
            all_speed_epochs = []
            
            for p in primes_list:
                speed_fit_entries = index.query(
                    experiment_type='speed', p=p,
                    **{k: v for k, v in filters.items() if k != 'experiment_type'}
                )
                for entry in speed_fit_entries:
                    traces = index.load_traces(entry)
                    if traces is None:
                        continue
                    n_samples = int(_nested_get(entry, 'n_samples') or 0)
                    param_count = int(_nested_get(entry, 'param_count') or 0)
                    if n_samples == 0 or param_count == 0:
                        continue
                    steps_per_epoch = (n_samples + args.batch_size - 1) // args.batch_size

                    # Compute saturation epoch dynamically
                    saturation_epoch = None
                    acc_trace = traces.get('train_acc_trace')
                    if acc_trace is not None:
                        steps_trace = traces.get('steps_trace')
                        saturation_epoch = compute_saturation_epoch_from_trace(
                            acc_trace, args.saturation_threshold, steps_per_epoch, steps_trace
                        )

                    if saturation_epoch is not None:
                        S = float(_nested_get(entry, 'dataset_bits') or 0)
                        if S == 0:
                            continue
                        P = param_count
                        f = S / (consts.C * P)
                        all_speed_f.append(f)
                        all_speed_epochs.append(saturation_epoch)
            
            if len(all_speed_f) >= 2:
                all_speed_f = np.array(all_speed_f)
                all_speed_epochs = np.array(all_speed_epochs)
                
                # Fit exponential: log(epochs) = b * f + log(a)
                log_epochs = np.log(all_speed_epochs)
                b, log_a = np.polyfit(all_speed_f, log_epochs, 1)
                a = np.exp(log_a)
                
                # Calculate R²
                y_pred_log = b * all_speed_f + log_a
                ss_res = np.sum((log_epochs - y_pred_log) ** 2)
                ss_tot = np.sum((log_epochs - np.mean(log_epochs)) ** 2)
                r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                
                if args.predicted_speed:
                    speed_fit_params = (a, b)
                if args.global_fit:
                    global_fit_params = (a, b)
                print(f"  Fit: epochs = {a:.1f} × exp({b:.2f} × f)")
                print(f"  R² = {r_squared:.3f}")
                print(f"  Based on {len(all_speed_f)} data points across all primes")
            else:
                print("  Warning: Not enough speed data to compute global fit.")

        all_grokking_points = []

        # Process each prime
        for prime_idx, p_prime in enumerate(primes_list):
            print(f"\n{'='*80}")
            print(f"Processing prime p={p_prime} ({prime_idx + 1}/{len(primes_list)})")
            print(f"{'='*80}")

            # Plot directory for this prime (aggregated across seeds)
            prime_plot_dir = os.path.join(args.plot_dir, f'p{p_prime}_aggregated')

            # Calculate dataset size for this prime
            n_prime, size_prime = compute_dataset_size_bits(p_prime, args.op, args.training_fraction)
            threshold_params_prime = size_prime / consts.C

            # Aggregate grokking results across all seeds (computes minimum delay per config)
            # Using minimum ensures critical param is where ALL seeds grok for that param count and above
            prime_results = aggregate_grokking_results_across_seeds(
                p_prime, index, filters,
                args.threshold_train, args.threshold_val, args.saturation_threshold,
                use_min_delay=True, dims=getattr(args, 'dims', None)
            )

            if not prime_results:
                print(f"Warning: No grokking results found for p={p_prime}")
                continue

            # Filter by max_dim if specified
            if args.max_dim is not None:
                prime_results = [r for r in prime_results if r['dim'] <= args.max_dim]
                if not prime_results:
                    print(f"Warning: No results found for p={p_prime} with dim <= {args.max_dim}")
                    continue

            # Get grokking curve data (epochs to grok vs param_count)
            grok_params = []
            grok_epochs = []
            for result in sorted(prime_results, key=lambda x: x['param_count']):
                etg = result.get('epochs_to_grok')
                if etg is not None and etg > 0:
                    grok_params.append(result['param_count'])
                    grok_epochs.append(etg)
            
            grok_params = np.array(grok_params) if grok_params else np.array([])
            grok_epochs = np.array(grok_epochs) if grok_epochs else np.array([])

            # Compute empirical grokking point (first param size after which everything has non-zero delay)
            empirical_critical_params = None
            delay_data = []
            for result in prime_results:
                delay_data.append({
                    'param_count': result['param_count'],
                    'delay': result['delay']
                })
            delay_data.sort(key=lambda x: x['param_count'])
            if delay_data:
                last_zero_idx = -1
                for i in range(len(delay_data) - 1, -1, -1):
                    if delay_data[i]['delay'] == 0:
                        last_zero_idx = i
                        break
                if last_zero_idx >= 0 and last_zero_idx + 1 < len(delay_data):
                    empirical_critical_params = delay_data[last_zero_idx + 1]['param_count']
                elif last_zero_idx == -1 and delay_data[0]['delay'] > 0:
                    empirical_critical_params = delay_data[0]['param_count']

            # Determine which intersection method to use and prepare speed data
            predicted_critical_params = None
            speed_data = []
            fit_params_used = None  # (a, b) for the exponential fit used
            
            # Get unique param counts from grokking results for generating fitted speed data
            param_counts = sorted(set(r['param_count'] for r in prime_results))
            
            if args.global_fit and global_fit_params is not None:
                # Use global exponential fit
                a, b = global_fit_params
                fit_params_used = (a, b)
                print(f"Using global exp fit: epochs = {a:.1f} × exp({b:.2f} × f)")
                
                # Generate speed data from global fit for plotting
                speed_data = []
                for P in param_counts:
                    f = size_prime / (consts.C * P)
                    predicted_epoch = a * np.exp(b * f)
                    speed_data.append({
                        'param_count': P,
                        'saturation_epoch': predicted_epoch,
                        'n_samples': n_prime,
                        'saturated': True
                    })
            
            elif args.prime_fit:
                # Fit per-prime exponential model from speed data
                speed_data_raw = aggregate_speed_results_across_seeds(p_prime, n_prime, index, filters, args.batch_size, args.saturation_threshold)
                
                if speed_data_raw and len(speed_data_raw) >= 2:
                    speed_params_arr = np.array([sp['param_count'] for sp in speed_data_raw if sp.get('saturation_epoch')])
                    speed_epochs_arr = np.array([sp['saturation_epoch'] for sp in speed_data_raw if sp.get('saturation_epoch')])
                    
                    if len(speed_params_arr) >= 2:
                        # Fit: log(epochs) = log(a) + b * f
                        speed_f = size_prime / (consts.C * speed_params_arr)
                        b_fit, log_a_fit = np.polyfit(speed_f, np.log(speed_epochs_arr), 1)
                        a_fit = np.exp(log_a_fit)
                        fit_params_used = (a_fit, b_fit)
                        print(f"Using per-prime exp fit: epochs = {a_fit:.1f} × exp({b_fit:.2f} × f)")
                        
                        # Generate speed data from per-prime fit for plotting
                        speed_data = []
                        for P in param_counts:
                            f = size_prime / (consts.C * P)
                            predicted_epoch = a_fit * np.exp(b_fit * f)
                            speed_data.append({
                                'param_count': P,
                                'saturation_epoch': predicted_epoch,
                                'n_samples': n_prime,
                                'saturated': True
                            })
                    else:
                        print(f"Warning: Not enough speed data for per-prime fit for p={p_prime}")
                else:
                    print(f"Warning: No speed data for per-prime fit for p={p_prime}")
            
            elif args.predicted_speed and speed_fit_params is not None:
                # Generate predicted speed data using the overall fit (legacy --predicted-speed)
                a, b = speed_fit_params
                print(f"Using predicted speed curve: epochs = {a:.1f} × exp({b:.2f} × f)")
                
                speed_data = []
                for P in param_counts:
                    f = size_prime / (consts.C * P)
                    predicted_epoch = a * np.exp(b * f)
                    speed_data.append({
                        'param_count': P,
                        'saturation_epoch': predicted_epoch,
                        'n_samples': n_prime,
                        'saturated': True
                    })
            
            else:
                # Aggregate actual speed data across all seeds (default behavior)
                print(f"Expecting speed data with {n_prime} samples for p={p_prime}, training fraction={args.training_fraction}.")
                speed_data = aggregate_speed_results_across_seeds(p_prime, n_prime, index, filters, args.batch_size, args.saturation_threshold)

            if not speed_data:
                print(f"Warning: No speed data found for p={p_prime}")

            # Determine filename suffix based on method
            if args.global_fit:
                filename_suffix = '_gf'
            elif args.prime_fit:
                filename_suffix = '_pf'
            else:
                filename_suffix = ''

            # Plot the speed-grok intersect graph for this prime
            os.makedirs(prime_plot_dir, exist_ok=True)
            save_path = os.path.join(prime_plot_dir, f'delay_with_speed_p{p_prime}{filename_suffix}.pdf') if args.save else None
            plot_grokking_delay_with_speed(
                prime_results,
                speed_data,
                threshold_train=args.threshold_train,
                threshold_val=args.threshold_val,
                saturation_threshold=args.saturation_threshold,
                threshold_params=threshold_params_prime,
                batch_size=args.batch_size,
                n_train_samples=n_prime,
                save_path=save_path,
                show=show
            )

            # Compute predicted grokking point (from speed intersection)
            predicted_critical_params = compute_critical_params_from_speed(
                prime_results,
                speed_data,
                threshold_train=args.threshold_train,
                threshold_val=args.threshold_val
            )

            # Collect grokking point data
            if predicted_critical_params is not None and empirical_critical_params is not None:
                all_grokking_points.append({
                    'p': p_prime,
                    'predicted': predicted_critical_params,
                    'empirical': empirical_critical_params
                })
                method_name = "global exp fit" if args.global_fit else ("per-prime exp fit" if args.prime_fit else "speed intersection")
                print(f"\nGrokking points for p={p_prime}:")
                print(f"  Predicted ({method_name}): {predicted_critical_params:,.0f} params")
                print(f"  Empirical (first delay > 0):    {empirical_critical_params:,.0f} params")
            elif predicted_critical_params is not None:
                method_name = "global exp fit" if args.global_fit else ("per-prime exp fit" if args.prime_fit else "speed intersection")
                print(f"\nWarning: Could not find empirical grokking point for p={p_prime} (no delay > 0)")
                print(f"  Predicted ({method_name}): {predicted_critical_params:,.0f} params")
            elif empirical_critical_params is not None:
                print(f"\nWarning: Could not compute predicted grokking point for p={p_prime}")
                print(f"  Empirical (first delay > 0):    {empirical_critical_params:,.0f} params")
            else:
                print(f"\nWarning: Could not compute grokking points for p={p_prime}")

        # Plot predicted vs empirical grokking points if we have multiple primes
        if len(all_grokking_points) > 1:
            print(f"\n{'='*80}")
            print("Plotting predicted vs empirical grokking points")
            print(f"{'='*80}")

            # Determine filename suffix based on method
            if args.global_fit:
                suffix = '_global_fit'
                title = 'Predicted (Global Exp Fit) vs Empirical Grokking Points'
            elif args.prime_fit:
                suffix = '_prime_fit'
                title = 'Predicted (Per-Prime Exp Fit) vs Empirical Grokking Points'
            else:
                suffix = ''
                title = None  # Use default title
            
            save_path = os.path.join(plot_dir, f'predicted_vs_empirical_grokking{suffix}.pdf') if args.save else None
            plot_predicted_vs_empirical_grokking(all_grokking_points, save_path=save_path, show=show, title=title)
        elif len(all_grokking_points) == 1:
            print("\nOnly one prime processed. Skipping predicted vs empirical comparison plot (requires multiple primes).")
        else:
            print("\nWarning: No valid grokking points found for any prime")

    elif args.speed:
        print("\nPlotting saturation time vs capacity fraction (separate line for each prime)...")

        # Collect and aggregate data from all primes across all seeds
        combined_results = {}
        for p in primes_list:
            print(f"\nLoading speed data for p={p} (aggregating across all seeds)")

            # Query all speed entries for this prime through the index
            speed_entries = index.query(
                experiment_type='speed', p=p,
                **{k: v for k, v in filters.items() if k != 'experiment_type'}
            )

            if not speed_entries:
                print(f"Warning: No speed data found for p={p}")
                continue

            seeds_seen = sorted(set(int(_nested_get(e, 'seed') or 0) for e in speed_entries))
            print(f"  Found data across {len(seeds_seen)} seed(s) for p={p}: {seeds_seen}")

            # Collect results by (dim, n_samples, param_count) key for averaging
            results_by_config = {}

            for entry in speed_entries:
                traces = index.load_traces(entry)
                if traces is None:
                    continue

                dim = int(_nested_get(entry, 'dim') or 0)
                n_samples = int(_nested_get(entry, 'n_samples') or 0)
                param_count = int(_nested_get(entry, 'param_count') or 0)
                if dim == 0 or n_samples == 0 or param_count == 0:
                    continue
                steps_per_epoch = (n_samples + args.batch_size - 1) // args.batch_size

                # Compute saturation epoch dynamically from train accuracy trace
                saturation_epoch = None
                acc_trace = traces.get('train_acc_trace')
                if acc_trace is not None:
                    steps_trace = traces.get('steps_trace')
                    saturation_epoch = compute_saturation_epoch_from_trace(
                        acc_trace, args.saturation_threshold, steps_per_epoch, steps_trace
                    )

                # Skip if saturation threshold never reached
                if saturation_epoch is None:
                    continue

                key = (dim, n_samples, param_count)

                if key not in results_by_config:
                    results_by_config[key] = {
                        'n_samples': n_samples,
                        'dim': dim,
                        'depth': int(_nested_get(entry, 'depth') or 0),
                        'heads': int(_nested_get(entry, 'heads') or 0),
                        'param_count': param_count,
                        'p': p,
                        'saturation_epochs': [],
                        'final_accs': [],
                        'dataset_bits': float(_nested_get(entry, 'dataset_bits') or 0)
                    }

                results_by_config[key]['saturation_epochs'].append(saturation_epoch)
                final_acc = traces.get('final_acc')
                results_by_config[key]['final_accs'].append(float(final_acc) if final_acc is not None else 0.0)

            # Average saturation epochs across seeds and add to combined results
            for key, rdata in results_by_config.items():
                saturation_epoch = float(np.mean(rdata['saturation_epochs']))

                result = {
                    'n_samples': rdata['n_samples'],
                    'dim': rdata['dim'],
                    'depth': rdata['depth'],
                    'heads': rdata['heads'],
                    'param_count': rdata['param_count'],
                    'p': rdata['p'],
                    'saturation_epoch': saturation_epoch,
                    'final_acc': float(np.mean(rdata['final_accs'])),
                    'dataset_bits': rdata['dataset_bits'],
                    'saturated': True
                }

                # Key by (p, dim) - each prime gets its own color/line in the plot
                combined_key = (p, rdata['dim'])
                if combined_key not in combined_results:
                    combined_results[combined_key] = []
                combined_results[combined_key].append(result)

        if not combined_results:
            print("No data found for any prime")
            return

        # Plot all data together
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, 'saturation_time_vs_capacity_fraction_all_primes.pdf') if args.save else None
        exponent, coefficient, r_squared = plot_saturation_time_vs_capacity_fraction(
            combined_results,
            C=consts.C,
            save_path=save_path,
            show=show
        )
        
        '''
        # Plot saturation epochs vs model parameters for comparison
        save_path = os.path.join(plot_dir, 'saturation_epochs_vs_params_all_primes.pdf') if args.save else None
        plot_saturation_epochs_vs_params(
            combined_results,
            save_path=save_path,
            show=show
        )

        # Plot saturation epochs vs dataset size (bits) with model sizes color coded
        save_path = os.path.join(plot_dir, 'saturation_epochs_vs_dataset_bits_all_primes.pdf') if args.save else None
        plot_saturation_epochs_vs_dataset_bits(
            combined_results,
            save_path=save_path,
            show=show
        )
        '''

        # Plot saturation epochs vs inverse capacity
        save_path = os.path.join(plot_dir, 'saturation_epochs_vs_inverse_capacity_all_primes.pdf') if args.save else None
        plot_saturation_epochs_vs_inverse_capacity(
            combined_results,
            C=consts.C,
            save_path=save_path,
            show=show
        )

        print("\n" + "="*60)
        print("CAPACITY FRACTION ANALYSIS (ALL PRIMES)")
        print("="*60)
        print(f"Power law fit: epochs = {coefficient:.1f} × f^{exponent:.2f}")
        print(f"Exponent: {exponent:.2f}")
        print(f"R²: {r_squared:.3f}")
        print(f"Capacity constant C = {consts.C:.2f} bits/param")
        print("="*60)

    elif args.delay:
        print("\n" + "="*80)
        print("GROKKING DELAY VS CAPACITY FRACTION")
        print("="*80)

        all_delay_data = []

        for p_prime in primes_list:
            print(f"\nProcessing prime p={p_prime}")

            # Calculate dataset size for this prime
            n_prime, size_prime = compute_dataset_size_bits(p_prime, args.op, args.training_fraction)

            # Aggregate grokking results across all seeds (computes minimum delay per config)
            prime_results = aggregate_grokking_results_across_seeds(
                p_prime, index, filters,
                args.threshold_train, args.threshold_val, args.saturation_threshold,
                use_min_delay=True, dims=getattr(args, 'dims', None)
            )

            if not prime_results:
                print(f"Warning: No grokking results found for p={p_prime}")
                continue

            # Filter by max_dim if specified
            if args.max_dim is not None:
                prime_results = [r for r in prime_results if r['dim'] <= args.max_dim]
                if not prime_results:
                    print(f"Warning: No results found for p={p_prime} with dim <= {args.max_dim}")
                    continue

            # Add delay data with dataset_bits
            for result in prime_results:
                all_delay_data.append({
                    'p': p_prime,
                    'param_count': result['param_count'],
                    'delay': result['delay'],
                    'dataset_bits': size_prime,
                    'dim': result['dim']
                })

        if not all_delay_data:
            print("No delay data found for any prime")
            return

        # Plot delay vs capacity fraction
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, 'delay_vs_capacity_fraction_all_primes.pdf') if args.save else None
        plot_delay_vs_capacity_fraction(
            all_delay_data,
            C=consts.C,
            save_path=save_path,
            show=show
        )

    elif args.correlation:
        print("\n" + "="*80)
        print("CORRELATION ANALYSIS: WHAT DETERMINES CRITICAL PARAMETER COUNT?")
        print("="*80)

        from scipy import stats
        from scipy.interpolate import interp1d
        import pandas as pd

        correlation_data = []

        # First, compute GLOBAL exponential fit from all primes' speed data
        print("\n" + "-"*60)
        print("Computing global exponential fit from all primes' speed data...")
        print("-"*60)
        
        all_f_values = []
        all_speed_epochs = []
        
        for p_prime in primes_list:
            n_prime, size_prime = compute_dataset_size_bits(p_prime, args.op, args.training_fraction)
            speed_data_temp = aggregate_speed_results_across_seeds(p_prime, n_prime, index, filters, args.batch_size, args.saturation_threshold)
            
            for sp in speed_data_temp:
                if sp.get('saturation_epoch') is not None and sp['saturation_epoch'] > 0:
                    P = sp['param_count']
                    f = size_prime / (consts.C * P)
                    all_f_values.append(f)
                    all_speed_epochs.append(sp['saturation_epoch'])
        
        global_a, global_b = None, None
        if len(all_f_values) >= 2:
            all_f_values = np.array(all_f_values)
            all_speed_epochs = np.array(all_speed_epochs)
            
            # Fit: log(epochs) = log(a) + b * f  =>  epochs = a * exp(b * f)
            global_b, global_log_a = np.polyfit(all_f_values, np.log(all_speed_epochs), 1)
            global_a = np.exp(global_log_a)
            
            # Calculate R²
            y_pred_log = global_b * all_f_values + global_log_a
            ss_res = np.sum((np.log(all_speed_epochs) - y_pred_log) ** 2)
            ss_tot = np.sum((np.log(all_speed_epochs) - np.mean(np.log(all_speed_epochs))) ** 2)
            global_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            print(f"  Global fit: epochs = {global_a:.2f} × exp({global_b:.2f} × f)")
            print(f"  R² = {global_r2:.4f}")
            print(f"  Based on {len(all_f_values)} data points across {len(primes_list)} primes")
        else:
            print("  Warning: Not enough speed data for global fit")

        print("\n" + "="*60)

        for p_prime in primes_list:
            print(f"\nProcessing prime p={p_prime}")

            # Calculate dataset size for this prime
            n_prime, size_prime = compute_dataset_size_bits(p_prime, args.op, args.training_fraction)

            # Aggregate grokking results across all seeds
            prime_results = aggregate_grokking_results_across_seeds(
                p_prime, index, filters,
                args.threshold_train, args.threshold_val, args.saturation_threshold,
                use_min_delay=True, dims=getattr(args, 'dims', None)
            )

            if not prime_results:
                print(f"  Warning: No grokking results found for p={p_prime}")
                continue

            # Filter by max_dim if specified
            if args.max_dim is not None:
                prime_results = [r for r in prime_results if r['dim'] <= args.max_dim]
                if not prime_results:
                    print(f"  Warning: No results found for p={p_prime} with dim <= {args.max_dim}")
                    continue

            # Get speed data (memorisation speed)
            speed_data = aggregate_speed_results_across_seeds(p_prime, n_prime, index, filters, args.batch_size, args.saturation_threshold)

            # Compute empirical critical params (first param size where min delay > 0 for all larger params)
            delay_data = sorted(prime_results, key=lambda x: x['param_count'])
            empirical_critical_params = None
            last_zero_idx = -1
            for i in range(len(delay_data) - 1, -1, -1):
                if delay_data[i]['delay'] == 0:
                    last_zero_idx = i
                    break
            if last_zero_idx >= 0 and last_zero_idx + 1 < len(delay_data):
                empirical_critical_params = delay_data[last_zero_idx + 1]['param_count']
            elif last_zero_idx == -1 and delay_data[0]['delay'] > 0:
                empirical_critical_params = delay_data[0]['param_count']

            if empirical_critical_params is None:
                print(f"  Warning: Could not find empirical critical params for p={p_prime}")
                continue

            print(f"  Empirical critical params: {empirical_critical_params:,.0f}")

            # Debug: show available param counts in speed data
            if speed_data:
                speed_param_counts = sorted(set(sp['param_count'] for sp in speed_data))
                print(f"  Speed data param counts: {speed_param_counts}")
            else:
                print(f"  Speed data: NONE LOADED")

            # Get grokking and memorisation speeds at the critical parameter count
            grok_speed_at_critical = None
            mem_speed_at_critical = None

            # Find grokking speed (epochs to val acc >= saturation_threshold) at critical point
            for result in prime_results:
                if result['param_count'] == empirical_critical_params:
                    grok_speed_at_critical = result.get('epochs_to_grok')
                    break

            # Find memorisation speed at critical point
            # First try exact match (±1 tolerance)
            for sp in speed_data:
                if abs(sp['param_count'] - empirical_critical_params) <= 1:
                    mem_speed_at_critical = sp.get('saturation_epoch')
                    break
            
            # If no exact match, use nearest param count (with warning)
            if mem_speed_at_critical is None and speed_data:
                nearest = min(speed_data, key=lambda sp: abs(sp['param_count'] - empirical_critical_params))
                diff = abs(nearest['param_count'] - empirical_critical_params)
                diff_pct = diff / empirical_critical_params * 100
                
                # Use nearest if within 50% of critical params
                if diff_pct <= 50:
                    mem_speed_at_critical = nearest.get('saturation_epoch')
                    print(f"  Using nearest mem_speed: param_count={nearest['param_count']:,} (diff: {diff:,}, {diff_pct:.1f}%)")
                else:
                    print(f"  WARNING: Nearest speed param_count too far: {nearest['param_count']:,} (diff: {diff:,}, {diff_pct:.1f}%) - skipping")

            # Compute intersection point (predicted critical params from curve intersection)
            intersection_params = None
            if len(prime_results) >= 2 and len(speed_data) >= 2:
                # Get grokking curve data
                grok_params = []
                grok_epochs = []
                for result in sorted(prime_results, key=lambda x: x['param_count']):
                    etg = result.get('epochs_to_grok')
                    if etg is not None and etg > 0:
                        grok_params.append(result['param_count'])
                        grok_epochs.append(etg)

                # Get speed curve data
                speed_params = []
                speed_epochs = []
                for sp in sorted(speed_data, key=lambda x: x['param_count']):
                    if sp.get('saturation_epoch') is not None and sp['saturation_epoch'] > 0:
                        speed_params.append(sp['param_count'])
                        speed_epochs.append(sp['saturation_epoch'])

                if len(grok_params) >= 2 and len(speed_params) >= 2:
                    grok_params = np.array(grok_params)
                    grok_epochs = np.array(grok_epochs)
                    speed_params = np.array(speed_params)
                    speed_epochs = np.array(speed_epochs)

                    try:
                        f_grok = interp1d(grok_params, grok_epochs, kind='linear', fill_value='extrapolate')
                        f_speed = interp1d(speed_params, speed_epochs, kind='linear', fill_value='extrapolate')

                        x_min = max(grok_params.min(), speed_params.min())
                        x_max = min(grok_params.max(), speed_params.max())

                        if x_min < x_max:
                            x_test = np.logspace(np.log10(x_min), np.log10(x_max), 1000)
                            y_grok_test = f_grok(x_test)
                            y_speed_test = f_speed(x_test)

                            # Handle potential negative values from extrapolation
                            valid_mask = (y_grok_test > 0) & (y_speed_test > 0)
                            if valid_mask.sum() > 0:
                                diff = np.abs(np.log(np.maximum(y_grok_test, 1e-6)) - np.log(np.maximum(y_speed_test, 1e-6)))
                                diff[~valid_mask] = np.inf
                                idx_closest = np.argmin(diff)
                                intersection_params = x_test[idx_closest]
                    except Exception as e:
                        print(f"  Warning: Could not compute intersection: {e}")

            # Compute intersection using exponential model for speed
            intersection_params_exp = None
            if len(grok_params) >= 2 and len(speed_params) >= 2:
                try:
                    # Fit exponential model to speed data: epochs = a * exp(b * f)
                    # where f = S / (C * P) is capacity fraction
                    speed_f = size_prime / (consts.C * speed_params)
                    log_speed_epochs = np.log(speed_epochs)
                    
                    # Linear fit in log space: log(epochs) = log(a) + b * f
                    b_fit, log_a_fit = np.polyfit(speed_f, log_speed_epochs, 1)
                    a_fit = np.exp(log_a_fit)
                    
                    # Function to get predicted speed epochs from param count
                    def speed_exp_model(P):
                        f = size_prime / (consts.C * P)
                        return a_fit * np.exp(b_fit * f)
                    
                    # Find intersection with grok curve
                    f_grok = interp1d(grok_params, grok_epochs, kind='linear', fill_value='extrapolate')
                    
                    x_min = grok_params.min()
                    x_max = grok_params.max()
                    x_test = np.logspace(np.log10(x_min), np.log10(x_max), 1000)
                    
                    y_grok_test = f_grok(x_test)
                    y_speed_exp_test = speed_exp_model(x_test)
                    
                    valid_mask = (y_grok_test > 0) & (y_speed_exp_test > 0)
                    if valid_mask.sum() > 0:
                        diff = np.abs(np.log(np.maximum(y_grok_test, 1e-6)) - np.log(np.maximum(y_speed_exp_test, 1e-6)))
                        diff[~valid_mask] = np.inf
                        idx_closest = np.argmin(diff)
                        intersection_params_exp = x_test[idx_closest]
                        
                except Exception as e:
                    print(f"  Warning: Could not compute exponential intersection: {e}")

            # Compute intersection using GLOBAL exponential model
            intersection_params_global = None
            if global_a is not None and global_b is not None and len(grok_params) >= 2:
                try:
                    # Use global fit: epochs = global_a * exp(global_b * f)
                    def speed_global_model(P):
                        f = size_prime / (consts.C * P)
                        return global_a * np.exp(global_b * f)
                    
                    # Find intersection with grok curve
                    f_grok = interp1d(grok_params, grok_epochs, kind='linear', fill_value='extrapolate')
                    
                    x_min = grok_params.min()
                    x_max = grok_params.max()
                    x_test = np.logspace(np.log10(x_min), np.log10(x_max), 1000)
                    
                    y_grok_test = f_grok(x_test)
                    y_speed_global_test = speed_global_model(x_test)
                    
                    valid_mask = (y_grok_test > 0) & (y_speed_global_test > 0)
                    if valid_mask.sum() > 0:
                        diff = np.abs(np.log(np.maximum(y_grok_test, 1e-6)) - np.log(np.maximum(y_speed_global_test, 1e-6)))
                        diff[~valid_mask] = np.inf
                        idx_closest = np.argmin(diff)
                        intersection_params_global = x_test[idx_closest]
                        
                except Exception as e:
                    print(f"  Warning: Could not compute global intersection: {e}")

            # Store data
            entry = {
                'p': p_prime,
                'dataset_bits': size_prime,
                'critical_params': empirical_critical_params,
                'grok_speed': grok_speed_at_critical,
                'mem_speed': mem_speed_at_critical,
                'intersection_params': intersection_params,
                'intersection_params_exp': intersection_params_exp,
                'intersection_params_global': intersection_params_global
            }
            correlation_data.append(entry)

            print(f"  Dataset size: {size_prime:,.0f} bits")
            print(f"  Grok speed at critical: {grok_speed_at_critical if grok_speed_at_critical else 'N/A'} epochs")
            print(f"  Mem speed at critical: {mem_speed_at_critical if mem_speed_at_critical else 'N/A'} epochs")
            print(f"  Intersection params (empirical): {intersection_params:,.0f}" if intersection_params else "  Intersection params (empirical): N/A")
            print(f"  Intersection params (per-prime exp): {intersection_params_exp:,.0f}" if intersection_params_exp else "  Intersection params (per-prime exp): N/A")
            print(f"  Intersection params (global exp): {intersection_params_global:,.0f}" if intersection_params_global else "  Intersection params (global exp): N/A")

        if len(correlation_data) < 3:
            print("\nError: Need at least 3 primes for meaningful correlation analysis")
            return

        # Create DataFrame for analysis
        df = pd.DataFrame(correlation_data)
        print("\n" + "="*80)
        print("DATA SUMMARY")
        print("="*80)
        print(df.to_string(index=False))

        # Compute correlations
        print("\n" + "="*80)
        print("CORRELATION ANALYSIS")
        print("="*80)

        variables = ['p', 'dataset_bits', 'grok_speed', 'mem_speed', 'intersection_params', 'intersection_params_exp', 'intersection_params_global']
        var_names = {
            'p': 'Prime (p)',
            'dataset_bits': 'Dataset size (bits)',
            'grok_speed': 'Grok speed at critical',
            'mem_speed': 'Mem speed at critical',
            'intersection_params': 'Intersection (empirical)',
            'intersection_params_exp': 'Intersection (per-prime exp)',
            'intersection_params_global': 'Intersection (global exp)'
        }

        print("\nPearson correlations with Critical Params:")
        print("-" * 60)

        correlations = []
        for var in variables:
            valid_mask = df['critical_params'].notna() & df[var].notna()
            if valid_mask.sum() >= 3:
                x = df.loc[valid_mask, var].values
                y = df.loc[valid_mask, 'critical_params'].values
                r, p_val = stats.pearsonr(x, y)
                correlations.append((var, r, p_val, valid_mask.sum()))
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"  {var_names[var]:30s}: r = {r:+.4f}, p = {p_val:.4f} {sig}  (n={valid_mask.sum()})")
            else:
                print(f"  {var_names[var]:30s}: Insufficient data")

        # Sort by absolute correlation
        print("\n" + "-"*60)
        print("Ranked by |r|:")
        correlations_sorted = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)
        for i, (var, r, p_val, n) in enumerate(correlations_sorted):
            print(f"  {i+1}. {var_names[var]:30s}: |r| = {abs(r):.4f}")

        # Spearman (rank) correlations for robustness
        print("\n" + "="*80)
        print("SPEARMAN (RANK) CORRELATIONS with Critical Params:")
        print("-" * 60)

        for var in variables:
            valid_mask = df['critical_params'].notna() & df[var].notna()
            if valid_mask.sum() >= 3:
                x = df.loc[valid_mask, var].values
                y = df.loc[valid_mask, 'critical_params'].values
                rho, p_val = stats.spearmanr(x, y)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"  {var_names[var]:30s}: ρ = {rho:+.4f}, p = {p_val:.4f} {sig}")

        # Test coefficient of variation for capacity hypothesis
        print("\n" + "="*80)
        print("CAPACITY HYPOTHESIS TEST")
        print("="*80)
        print("If critical params = S/C (where C is bits/param), then S/P* should be constant.")
        
        valid = df['dataset_bits'].notna() & df['critical_params'].notna()
        if valid.sum() >= 2:
            ratios = df.loc[valid, 'dataset_bits'] / df.loc[valid, 'critical_params']
            cv = ratios.std() / ratios.mean()
            print(f"\n  S/P* (implied C): mean = {ratios.mean():.2f}, std = {ratios.std():.2f}")
            print(f"  Coefficient of variation: {cv:.3f}")
            if cv < 0.1:
                print("  → LOW variation: Strong support for capacity hypothesis (P* ∝ S)")
            elif cv < 0.2:
                print("  → MODERATE variation: Some support for capacity hypothesis")
            else:
                print("  → HIGH variation: Capacity alone may not explain critical params")

        # Test if speed at critical point is constant
        print("\n" + "="*80)
        print("SPEED CONSTANCY TEST")
        print("="*80)
        print("If critical point is where speed = constant, then speed at P* should be similar.")

        for var, name in [('grok_speed', 'Grok speed'), ('mem_speed', 'Mem speed')]:
            valid = df[var].notna()
            if valid.sum() >= 2:
                speeds = df.loc[valid, var]
                cv = speeds.std() / speeds.mean()
                print(f"\n  {name} at critical: mean = {speeds.mean():.1f}, std = {speeds.std():.1f}")
                print(f"  Coefficient of variation: {cv:.3f}")
                if cv < 0.2:
                    print(f"  → LOW variation: {name} is roughly constant at critical point")
                elif cv < 0.4:
                    print(f"  → MODERATE variation")
                else:
                    print(f"  → HIGH variation: {name} is NOT constant at critical point")

        # Multiple regression to see which predictors matter
        print("\n" + "="*80)
        print("MULTIPLE REGRESSION: Critical Params ~ Predictors")
        print("="*80)

        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler

            # Prepare data - only use rows with all variables
            pred_vars = ['p', 'dataset_bits', 'grok_speed', 'mem_speed', 'intersection_params', 'intersection_params_exp', 'intersection_params_global']
            valid = df['critical_params'].notna()
            for v in pred_vars:
                valid = valid & df[v].notna()

            if valid.sum() >= 4:
                X = df.loc[valid, pred_vars].values
                y = df.loc[valid, 'critical_params'].values

                # Standardize for comparable coefficients
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                reg = LinearRegression()
                reg.fit(X_scaled, y)
                r2 = reg.score(X_scaled, y)

                print(f"\nR² (all predictors): {r2:.4f}")
                print("\nStandardized coefficients (higher |β| = more important):")
                coefs = list(zip(pred_vars, reg.coef_))
                coefs_sorted = sorted(coefs, key=lambda x: abs(x[1]), reverse=True)
                for var, beta in coefs_sorted:
                    print(f"  {var_names[var]:30s}: β = {beta:+.2f}")

                # Single-predictor R² values (using all available data for each variable)
                print("\nSingle-predictor R² values (all available data per variable):")
                for var in pred_vars:
                    var_valid = df['critical_params'].notna() & df[var].notna()
                    if var_valid.sum() >= 2:
                        X_single = df.loc[var_valid, [var]].values
                        y_single = df.loc[var_valid, 'critical_params'].values
                        X_single_scaled = StandardScaler().fit_transform(X_single)
                        reg_single = LinearRegression()
                        reg_single.fit(X_single_scaled, y_single)
                        r2_single = reg_single.score(X_single_scaled, y_single)
                        print(f"  {var_names[var]:30s}: R² = {r2_single:.4f}  (n={var_valid.sum()})")
                    else:
                        print(f"  {var_names[var]:30s}: Insufficient data")
            else:
                print("Insufficient data for multiple regression (need at least 4 complete rows)")

        except ImportError:
            print("sklearn not available for multiple regression analysis")

        # Plot correlation matrix
        os.makedirs(plot_dir, exist_ok=True)
        save_path_corr = os.path.join(plot_dir, 'correlation_matrix.pdf') if args.save else None

        import matplotlib.pyplot as plt

        # Create correlation matrix for available data
        plot_vars = ['critical_params', 'p', 'dataset_bits', 'grok_speed', 'mem_speed', 'intersection_params', 'intersection_params_exp', 'intersection_params_global']
        plot_labels = ['Grokking point', 'Prime', 'Dataset complexity', 'Gen. speed at point', 'Mem. speed at point', 'Intersect (empirical)', 'Intersect (per-prime exp)', 'Intersect (global exp)']

        # Only include variables with enough data
        valid_vars = []
        valid_labels = []
        for v, l in zip(plot_vars, plot_labels):
            if df[v].notna().sum() >= 3:
                valid_vars.append(v)
                valid_labels.append(l)

        if len(valid_vars) >= 2:
            corr_matrix = df[valid_vars].corr()

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)

            ax.set_xticks(range(len(valid_labels)))
            ax.set_yticks(range(len(valid_labels)))
            ax.set_xticklabels(valid_labels, rotation=45, ha='right', fontsize=11)
            ax.set_yticklabels(valid_labels, fontsize=11)

            # Add correlation values
            for i in range(len(valid_labels)):
                for j in range(len(valid_labels)):
                    val = corr_matrix.values[i, j]
                    color = 'white' if abs(val) > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=10)

            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Pearson r', fontsize=12)

            plt.tight_layout()

            if save_path_corr:
                plt.savefig(save_path_corr, bbox_inches='tight', dpi=300)
                print(f"\nSaved correlation matrix: {save_path_corr}")

            if show:
                plt.show()
            else:
                plt.close()

        print("\n" + "="*80)
        print("INTERPRETATION GUIDE")
        print("="*80)
        print("""
Significance levels: * p<0.05, ** p<0.01, *** p<0.001

If critical params is most strongly correlated with:
  - Dataset size: Critical P* is about storage capacity (P* ∝ S)
  - Prime: Other prime-dependent factors matter beyond just dataset size
  - Grok speed: Critical P* is where grokking becomes "fast enough"
  - Mem speed: Critical P* is where memorization becomes "fast enough"
  - Intersection: Your predicted critical point matches the empirical one

Low CV (coefficient of variation) for a ratio/value suggests that relationship
is roughly constant across primes, supporting that hypothesis.
""")

    elif args.cv:
        print("\n" + "="*80)
        print("CROSS-VALIDATION ANALYSIS: INCREMENTAL VALIDITY OF INTERSECTION")
        print("="*80)
        print("Testing whether intersection adds predictive value beyond dataset_bits")

        from scipy import stats
        from scipy.interpolate import interp1d
        import pandas as pd

        # =================================================================
        # STEP 1: Collect data (same as --correlation)
        # =================================================================
        cv_data = []

        # First, compute GLOBAL exponential fit from all primes' speed data
        print("\n" + "-"*60)
        print("Computing global exponential fit from all primes' speed data...")
        print("-"*60)

        all_f_values = []
        all_speed_epochs = []

        for p_prime in primes_list:
            n_prime, size_prime = compute_dataset_size_bits(p_prime, args.op, args.training_fraction)
            speed_data_temp = aggregate_speed_results_across_seeds(p_prime, n_prime, index, filters, args.batch_size, args.saturation_threshold)

            for sp in speed_data_temp:
                if sp.get('saturation_epoch') is not None and sp['saturation_epoch'] > 0:
                    P = sp['param_count']
                    f = size_prime / (consts.C * P)
                    all_f_values.append(f)
                    all_speed_epochs.append(sp['saturation_epoch'])

        global_a, global_b = None, None
        if len(all_f_values) >= 2:
            all_f_values = np.array(all_f_values)
            all_speed_epochs = np.array(all_speed_epochs)

            global_b, global_log_a = np.polyfit(all_f_values, np.log(all_speed_epochs), 1)
            global_a = np.exp(global_log_a)

            y_pred_log = global_b * all_f_values + global_log_a
            ss_res = np.sum((np.log(all_speed_epochs) - y_pred_log) ** 2)
            ss_tot = np.sum((np.log(all_speed_epochs) - np.mean(np.log(all_speed_epochs))) ** 2)
            global_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            print(f"  Global fit: epochs = {global_a:.2f} × exp({global_b:.2f} × f)")
            print(f"  R² = {global_r2:.4f}")
        else:
            print("  Warning: Not enough speed data for global fit")

        print("\n" + "-"*60)
        print("Collecting data for each prime...")
        print("-"*60)

        for p_prime in primes_list:
            print(f"\nProcessing prime p={p_prime}")

            n_prime, size_prime = compute_dataset_size_bits(p_prime, args.op, args.training_fraction)

            prime_results = aggregate_grokking_results_across_seeds(
                p_prime, index, filters,
                args.threshold_train, args.threshold_val, args.saturation_threshold,
                use_min_delay=True, dims=getattr(args, 'dims', None)
            )

            if not prime_results:
                print(f"  Warning: No grokking results found for p={p_prime}")
                continue

            if args.max_dim is not None:
                prime_results = [r for r in prime_results if r['dim'] <= args.max_dim]
                if not prime_results:
                    print(f"  Warning: No results found for p={p_prime} with dim <= {args.max_dim}")
                    continue

            speed_data = aggregate_speed_results_across_seeds(p_prime, n_prime, index, filters, args.batch_size, args.saturation_threshold)

            # Compute empirical critical params
            delay_data = sorted(prime_results, key=lambda x: x['param_count'])
            empirical_critical_params = None
            last_zero_idx = -1
            for i in range(len(delay_data) - 1, -1, -1):
                if delay_data[i]['delay'] == 0:
                    last_zero_idx = i
                    break
            if last_zero_idx >= 0 and last_zero_idx + 1 < len(delay_data):
                empirical_critical_params = delay_data[last_zero_idx + 1]['param_count']
            elif last_zero_idx == -1 and delay_data[0]['delay'] > 0:
                empirical_critical_params = delay_data[0]['param_count']

            if empirical_critical_params is None:
                print(f"  Warning: Could not find empirical critical params for p={p_prime}")
                continue

            # Get grok_speed and mem_speed at critical point
            grok_speed_at_critical = None
            mem_speed_at_critical = None

            for result in prime_results:
                if result['param_count'] == empirical_critical_params:
                    grok_speed_at_critical = result.get('epochs_to_grok')
                    break

            for sp in speed_data:
                if abs(sp['param_count'] - empirical_critical_params) <= 1:
                    mem_speed_at_critical = sp.get('saturation_epoch')
                    break

            # If no exact match for mem_speed, use nearest
            if mem_speed_at_critical is None and speed_data:
                nearest = min(speed_data, key=lambda sp: abs(sp['param_count'] - empirical_critical_params))
                diff_pct = abs(nearest['param_count'] - empirical_critical_params) / empirical_critical_params * 100
                if diff_pct <= 50:
                    mem_speed_at_critical = nearest.get('saturation_epoch')

            # Get grokking and speed curve data
            grok_params = []
            grok_epochs = []
            for result in sorted(prime_results, key=lambda x: x['param_count']):
                etg = result.get('epochs_to_grok')
                if etg is not None and etg > 0:
                    grok_params.append(result['param_count'])
                    grok_epochs.append(etg)

            speed_params = []
            speed_epochs = []
            for sp in sorted(speed_data, key=lambda x: x['param_count']):
                if sp.get('saturation_epoch') is not None and sp['saturation_epoch'] > 0:
                    speed_params.append(sp['param_count'])
                    speed_epochs.append(sp['saturation_epoch'])

            # Compute intersection (empirical - linear interpolation)
            intersection_params = None
            if len(grok_params) >= 2 and len(speed_params) >= 2:
                grok_params_arr = np.array(grok_params)
                grok_epochs_arr = np.array(grok_epochs)
                speed_params_arr = np.array(speed_params)
                speed_epochs_arr = np.array(speed_epochs)

                try:
                    f_grok = interp1d(grok_params_arr, grok_epochs_arr, kind='linear', fill_value='extrapolate')
                    f_speed = interp1d(speed_params_arr, speed_epochs_arr, kind='linear', fill_value='extrapolate')

                    x_min = max(grok_params_arr.min(), speed_params_arr.min())
                    x_max = min(grok_params_arr.max(), speed_params_arr.max())

                    if x_min < x_max:
                        x_test = np.logspace(np.log10(x_min), np.log10(x_max), 1000)
                        y_grok_test = f_grok(x_test)
                        y_speed_test = f_speed(x_test)

                        valid_mask = (y_grok_test > 0) & (y_speed_test > 0)
                        if valid_mask.sum() > 0:
                            diff = np.abs(np.log(np.maximum(y_grok_test, 1e-6)) - np.log(np.maximum(y_speed_test, 1e-6)))
                            diff[~valid_mask] = np.inf
                            idx_closest = np.argmin(diff)
                            intersection_params = x_test[idx_closest]
                except Exception as e:
                    print(f"  Warning: Could not compute intersection: {e}")

            # Compute intersection using per-prime exponential fit
            intersection_params_exp = None
            if len(grok_params) >= 2 and len(speed_params) >= 2:
                try:
                    speed_f = size_prime / (consts.C * speed_params_arr)
                    log_speed_epochs = np.log(speed_epochs_arr)
                    b_fit, log_a_fit = np.polyfit(speed_f, log_speed_epochs, 1)
                    a_fit = np.exp(log_a_fit)

                    def speed_exp_model(P):
                        f = size_prime / (consts.C * P)
                        return a_fit * np.exp(b_fit * f)

                    f_grok = interp1d(grok_params_arr, grok_epochs_arr, kind='linear', fill_value='extrapolate')
                    x_min = grok_params_arr.min()
                    x_max = grok_params_arr.max()
                    x_test = np.logspace(np.log10(x_min), np.log10(x_max), 1000)
                    y_grok_test = f_grok(x_test)
                    y_speed_exp_test = speed_exp_model(x_test)

                    valid_mask = (y_grok_test > 0) & (y_speed_exp_test > 0)
                    if valid_mask.sum() > 0:
                        diff = np.abs(np.log(np.maximum(y_grok_test, 1e-6)) - np.log(np.maximum(y_speed_exp_test, 1e-6)))
                        diff[~valid_mask] = np.inf
                        idx_closest = np.argmin(diff)
                        intersection_params_exp = x_test[idx_closest]
                except Exception as e:
                    print(f"  Warning: Could not compute exponential intersection: {e}")

            # Compute intersection using global exponential fit
            intersection_params_global = None
            if global_a is not None and global_b is not None and len(grok_params) >= 2:
                try:
                    def speed_global_model(P):
                        f = size_prime / (consts.C * P)
                        return global_a * np.exp(global_b * f)

                    f_grok = interp1d(grok_params_arr, grok_epochs_arr, kind='linear', fill_value='extrapolate')
                    x_min = grok_params_arr.min()
                    x_max = grok_params_arr.max()
                    x_test = np.logspace(np.log10(x_min), np.log10(x_max), 1000)
                    y_grok_test = f_grok(x_test)
                    y_speed_global_test = speed_global_model(x_test)

                    valid_mask = (y_grok_test > 0) & (y_speed_global_test > 0)
                    if valid_mask.sum() > 0:
                        diff = np.abs(np.log(np.maximum(y_grok_test, 1e-6)) - np.log(np.maximum(y_speed_global_test, 1e-6)))
                        diff[~valid_mask] = np.inf
                        idx_closest = np.argmin(diff)
                        intersection_params_global = x_test[idx_closest]
                except Exception as e:
                    print(f"  Warning: Could not compute global intersection: {e}")

            entry = {
                'p': p_prime,
                'dataset_bits': size_prime,
                'critical_params': empirical_critical_params,
                'grok_speed': grok_speed_at_critical,
                'mem_speed': mem_speed_at_critical,
                'intersection_params': intersection_params,
                'intersection_params_exp': intersection_params_exp,
                'intersection_params_global': intersection_params_global
            }
            cv_data.append(entry)
            print(f"  Critical params: {empirical_critical_params:,.0f}, Dataset bits: {size_prime:,.0f}")

        if len(cv_data) < 5:
            print(f"\nError: Need at least 5 primes for meaningful CV analysis (have {len(cv_data)})")
            return

        df = pd.DataFrame(cv_data)
        print("\n" + "="*80)
        print("DATA SUMMARY")
        print("="*80)
        print(df.to_string(index=False))

        # =================================================================
        # STEP 2: Define LOOCV helper functions
        # =================================================================

        def loocv_ols(X, y):
            """
            Perform leave-one-out cross-validation with OLS.
            Returns: predictions, residuals, LOOCV-R², LOOCV-MSE
            """
            n = len(y)
            predictions = np.zeros(n)
            residuals = np.zeros(n)

            for i in range(n):
                # Leave out point i
                X_train = np.delete(X, i, axis=0)
                y_train = np.delete(y, i)
                X_test = X[i:i+1]
                y_test = y[i]

                # Fit OLS: beta = (X'X)^-1 X'y
                try:
                    beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
                    pred = X_test @ beta
                    predictions[i] = pred[0]
                except np.linalg.LinAlgError:
                    predictions[i] = np.mean(y_train)

                residuals[i] = y_test - predictions[i]

            mse = np.mean(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum(residuals ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            return predictions, residuals, r2, mse

        def fit_ols(X, y):
            """Fit OLS and return R², coefficients."""
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                y_pred = X @ beta
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                return r2, beta
            except np.linalg.LinAlgError:
                return 0, None

        # =================================================================
        # STEP 3: Define all predictors
        # =================================================================

        all_predictors = {
            'dataset_bits': 'Dataset bits',
            'p': 'Prime (p)',
            'grok_speed': 'Grok speed',
            'mem_speed': 'Mem speed',
            'intersection_params': 'Intersection (empirical)',
            'intersection_params_exp': 'Intersection (per-prime exp)',
            'intersection_params_global': 'Intersection (global exp)'
        }

        # =================================================================
        # STEP 4: Univariate LOOCV-R² for all predictors
        # =================================================================

        print("\n" + "="*80)
        print("UNIVARIATE LOOCV-R² (single predictor models)")
        print("="*80)

        univariate_results = []
        for var, name in all_predictors.items():
            valid_mask = df[var].notna() & df['critical_params'].notna()
            n_valid = valid_mask.sum()

            if n_valid < 3:
                print(f"  {name:35s}: Insufficient data (n={n_valid})")
                continue

            y_uni = df.loc[valid_mask, 'critical_params'].values
            X_uni = np.column_stack([np.ones(n_valid), df.loc[valid_mask, var].values])

            r2_in, _ = fit_ols(X_uni, y_uni)
            _, _, r2_cv, mse_cv = loocv_ols(X_uni, y_uni)

            univariate_results.append((var, name, n_valid, r2_in, r2_cv, np.sqrt(mse_cv)))
            print(f"  {name:35s}: R²={r2_in:.4f}, LOOCV-R²={r2_cv:.4f}, RMSE={np.sqrt(mse_cv):,.0f} (n={n_valid})")

        # Rank by LOOCV-R²
        print("\n  Ranked by LOOCV-R²:")
        univariate_results.sort(key=lambda x: x[4], reverse=True)
        for i, (var, name, n, r2_in, r2_cv, rmse) in enumerate(univariate_results):
            print(f"    {i+1}. {name:35s}: LOOCV-R² = {r2_cv:.4f}")

        # =================================================================
        # STEP 5: Incremental validity for each baseline
        # =================================================================

        # Define baselines to test
        baselines = ['dataset_bits', 'p', 'grok_speed', 'mem_speed']
        baseline_names = {
            'dataset_bits': 'Dataset bits',
            'p': 'Prime (p)',
            'grok_speed': 'Grok speed',
            'mem_speed': 'Mem speed'
        }

        n_permutations = 10000
        rng = np.random.default_rng(42)

        # Store all results for summary
        all_incremental_results = {}

        for baseline_var in baselines:
            baseline_name = baseline_names[baseline_var]

            print("\n" + "="*80)
            print(f"INCREMENTAL VALIDITY BEYOND: {baseline_name}")
            print("="*80)
            print(f"Testing: critical_params ~ {baseline_name} + X")

            # Get valid mask for baseline
            baseline_valid = df[baseline_var].notna() & df['critical_params'].notna()
            if baseline_valid.sum() < 5:
                print(f"  Insufficient data for baseline {baseline_name}")
                continue

            incremental_results = []

            # Test each other predictor as incremental
            for inc_var, inc_name in all_predictors.items():
                if inc_var == baseline_var:
                    continue  # Skip if same as baseline

                # Get valid mask for this combination
                valid_mask = baseline_valid & df[inc_var].notna()
                n_valid = valid_mask.sum()

                if n_valid < 5:
                    continue

                y = df.loc[valid_mask, 'critical_params'].values
                x_base = df.loc[valid_mask, baseline_var].values
                x_inc = df.loc[valid_mask, inc_var].values

                X_baseline = np.column_stack([np.ones(n_valid), x_base])
                X_augmented = np.column_stack([np.ones(n_valid), x_base, x_inc])

                # In-sample analysis
                r2_base_in, _ = fit_ols(X_baseline, y)
                r2_aug_in, _ = fit_ols(X_augmented, y)
                delta_r2_in = r2_aug_in - r2_base_in

                # LOOCV analysis
                _, _, r2_base_cv, _ = loocv_ols(X_baseline, y)
                _, _, r2_aug_cv, mse_aug_cv = loocv_ols(X_augmented, y)
                delta_r2_cv = r2_aug_cv - r2_base_cv

                # Permutation test
                perm_deltas = np.zeros(n_permutations)
                for b in range(n_permutations):
                    x_inc_perm = rng.permutation(x_inc)
                    X_aug_perm = np.column_stack([np.ones(n_valid), x_base, x_inc_perm])
                    _, _, r2_aug_perm, _ = loocv_ols(X_aug_perm, y)
                    perm_deltas[b] = r2_aug_perm - r2_base_cv

                p_value_perm = (1 + np.sum(perm_deltas >= delta_r2_cv)) / (1 + n_permutations)

                incremental_results.append({
                    'var': inc_var,
                    'name': inc_name,
                    'n': n_valid,
                    'delta_r2_in': delta_r2_in,
                    'delta_r2_cv': delta_r2_cv,
                    'p_value': p_value_perm,
                    'rmse_cv': np.sqrt(mse_aug_cv)
                })

            # Sort by LOOCV ΔR²
            incremental_results.sort(key=lambda x: x['delta_r2_cv'], reverse=True)

            # Print results table
            print(f"\n  {'Predictor':<35s} {'ΔR²(in)':>10s} {'ΔR²(LOOCV)':>12s} {'p-value':>10s}")
            print("  " + "-"*70)
            for res in incremental_results:
                sig = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else ""
                print(f"  {res['name']:<35s} {res['delta_r2_in']:>+10.4f} {res['delta_r2_cv']:>+12.4f} {res['p_value']:>9.4f} {sig}")

            # Store for summary
            all_incremental_results[baseline_var] = incremental_results

        # =================================================================
        # STEP 6: Summary matrix
        # =================================================================

        print("\n" + "="*80)
        print("SUMMARY: BEST INCREMENTAL PREDICTOR FOR EACH BASELINE")
        print("="*80)

        print(f"\n  {'Baseline':<20s} {'Best Incremental Predictor':<35s} {'ΔR²(LOOCV)':>12s} {'p-value':>10s}")
        print("  " + "-"*80)

        for baseline_var in baselines:
            if baseline_var not in all_incremental_results:
                continue
            results = all_incremental_results[baseline_var]
            if not results:
                continue
            best = results[0]  # Already sorted by delta_r2_cv
            sig = "***" if best['p_value'] < 0.001 else "**" if best['p_value'] < 0.01 else "*" if best['p_value'] < 0.05 else ""
            print(f"  {baseline_names[baseline_var]:<20s} {best['name']:<35s} {best['delta_r2_cv']:>+12.4f} {best['p_value']:>9.4f} {sig}")

        # =================================================================
        # STEP 7: Consistency check - is intersection consistently best?
        # =================================================================

        print("\n" + "="*80)
        print("CONSISTENCY CHECK: HOW OFTEN IS EACH PREDICTOR THE BEST?")
        print("="*80)

        # Count how often each predictor is ranked #1
        rank1_counts = {}
        for baseline_var, results in all_incremental_results.items():
            if results:
                best_var = results[0]['var']
                rank1_counts[best_var] = rank1_counts.get(best_var, 0) + 1

        print("\n  Times ranked #1 (across all baselines):")
        for var, count in sorted(rank1_counts.items(), key=lambda x: -x[1]):
            print(f"    {all_predictors[var]:<35s}: {count}")

        # Check if any intersection type is consistently in top 2
        intersection_vars = ['intersection_params', 'intersection_params_exp', 'intersection_params_global']
        print("\n  Average rank of intersection predictors:")
        for int_var in intersection_vars:
            ranks = []
            for baseline_var, results in all_incremental_results.items():
                for i, res in enumerate(results):
                    if res['var'] == int_var:
                        ranks.append(i + 1)
                        break
            if ranks:
                avg_rank = np.mean(ranks)
                print(f"    {all_predictors[int_var]:<35s}: avg rank = {avg_rank:.1f} (across {len(ranks)} baselines)")

        # =================================================================
        # STEP 8: Final interpretation
        # =================================================================

        print("\n" + "="*80)
        print("INTERPRETATION")
        print("="*80)

        # Find overall best predictor (most consistent across baselines)
        if rank1_counts:
            best_overall = max(rank1_counts.items(), key=lambda x: x[1])
            print(f"\n  Most consistent best predictor: {all_predictors[best_overall[0]]}")
            print(f"  (Ranked #1 in {best_overall[1]} out of {len(all_incremental_results)} baselines)")

            # Check if it's an intersection type
            if best_overall[0] in intersection_vars:
                print("\n  → CONCLUSION: An intersection predictor is consistently the best")
                print("    incremental predictor across multiple baselines.")
            else:
                print(f"\n  → CONCLUSION: {all_predictors[best_overall[0]]} is a stronger predictor")
                print("    than intersection metrics across most baselines.")

        print("\n" + "="*80)

    elif args.nlcv:
        # =====================================================================
        # NON-LINEAR CROSS-VALIDATION (Kernel Ridge Regression)
        # =====================================================================
        print("\n" + "="*80)
        print("NON-LINEAR CROSS-VALIDATION (Kernel Ridge Regression)")
        print("="*80)
        print("Testing predictors with non-linear relationships")

        from scipy import stats
        from scipy.interpolate import interp1d
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import LeaveOneOut
        import pandas as pd

        # =================================================================
        # STEP 1: Collect data (same as --cv)
        # =================================================================
        nlcv_data = []

        # Compute global exponential fit
        print("\n" + "-"*60)
        print("Computing global exponential fit from speed data...")
        print("-"*60)

        all_f_values = []
        all_speed_epochs = []

        for p_prime in primes_list:
            n_prime, size_prime = compute_dataset_size_bits(p_prime, args.op, args.training_fraction)
            speed_data_temp = aggregate_speed_results_across_seeds(p_prime, n_prime, index, filters, args.batch_size, args.saturation_threshold)

            for sp in speed_data_temp:
                if sp.get('saturation_epoch') is not None and sp['saturation_epoch'] > 0:
                    P = sp['param_count']
                    f = size_prime / (consts.C * P)
                    all_f_values.append(f)
                    all_speed_epochs.append(sp['saturation_epoch'])

        global_a, global_b = None, None
        if len(all_f_values) >= 2:
            all_f_values = np.array(all_f_values)
            all_speed_epochs = np.array(all_speed_epochs)
            global_b, global_log_a = np.polyfit(all_f_values, np.log(all_speed_epochs), 1)
            global_a = np.exp(global_log_a)
            print(f"  Global fit: epochs = {global_a:.2f} × exp({global_b:.2f} × f)")

        print("\n" + "-"*60)
        print("Collecting data for each prime...")
        print("-"*60)

        for p_prime in primes_list:
            n_prime, size_prime = compute_dataset_size_bits(p_prime, args.op, args.training_fraction)

            prime_results = aggregate_grokking_results_across_seeds(
                p_prime, index, filters,
                args.threshold_train, args.threshold_val, args.saturation_threshold,
                use_min_delay=True, dims=getattr(args, 'dims', None)
            )

            if not prime_results:
                continue

            if args.max_dim is not None:
                prime_results = [r for r in prime_results if r['dim'] <= args.max_dim]
                if not prime_results:
                    continue

            speed_data = aggregate_speed_results_across_seeds(p_prime, n_prime, index, filters, args.batch_size, args.saturation_threshold)

            # Compute empirical critical params
            delay_data = sorted(prime_results, key=lambda x: x['param_count'])
            empirical_critical_params = None
            last_zero_idx = -1
            for i in range(len(delay_data) - 1, -1, -1):
                if delay_data[i]['delay'] == 0:
                    last_zero_idx = i
                    break
            if last_zero_idx >= 0 and last_zero_idx + 1 < len(delay_data):
                empirical_critical_params = delay_data[last_zero_idx + 1]['param_count']
            elif last_zero_idx == -1 and delay_data[0]['delay'] > 0:
                empirical_critical_params = delay_data[0]['param_count']

            if empirical_critical_params is None:
                continue

            # Get grok_speed and mem_speed at critical point
            grok_speed_at_critical = None
            mem_speed_at_critical = None

            for result in prime_results:
                if result['param_count'] == empirical_critical_params:
                    grok_speed_at_critical = result.get('epochs_to_grok')
                    break

            for sp in speed_data:
                if abs(sp['param_count'] - empirical_critical_params) <= 1:
                    mem_speed_at_critical = sp.get('saturation_epoch')
                    break

            if mem_speed_at_critical is None and speed_data:
                nearest = min(speed_data, key=lambda sp: abs(sp['param_count'] - empirical_critical_params))
                diff_pct = abs(nearest['param_count'] - empirical_critical_params) / empirical_critical_params * 100
                if diff_pct <= 50:
                    mem_speed_at_critical = nearest.get('saturation_epoch')

            # Get curve data for intersection computation
            grok_params = []
            grok_epochs = []
            for result in sorted(prime_results, key=lambda x: x['param_count']):
                etg = result.get('epochs_to_grok')
                if etg is not None and etg > 0:
                    grok_params.append(result['param_count'])
                    grok_epochs.append(etg)

            speed_params = []
            speed_epochs = []
            for sp in sorted(speed_data, key=lambda x: x['param_count']):
                if sp.get('saturation_epoch') is not None and sp['saturation_epoch'] > 0:
                    speed_params.append(sp['param_count'])
                    speed_epochs.append(sp['saturation_epoch'])

            # Compute intersections
            intersection_params = None
            intersection_params_exp = None
            intersection_params_global = None

            if len(grok_params) >= 2 and len(speed_params) >= 2:
                grok_params_arr = np.array(grok_params)
                grok_epochs_arr = np.array(grok_epochs)
                speed_params_arr = np.array(speed_params)
                speed_epochs_arr = np.array(speed_epochs)

                try:
                    f_grok = interp1d(grok_params_arr, grok_epochs_arr, kind='linear', fill_value='extrapolate')
                    f_speed = interp1d(speed_params_arr, speed_epochs_arr, kind='linear', fill_value='extrapolate')
                    x_min = max(grok_params_arr.min(), speed_params_arr.min())
                    x_max = min(grok_params_arr.max(), speed_params_arr.max())
                    if x_min < x_max:
                        x_test = np.logspace(np.log10(x_min), np.log10(x_max), 1000)
                        y_grok_test = f_grok(x_test)
                        y_speed_test = f_speed(x_test)
                        valid_mask = (y_grok_test > 0) & (y_speed_test > 0)
                        if valid_mask.sum() > 0:
                            diff = np.abs(np.log(np.maximum(y_grok_test, 1e-6)) - np.log(np.maximum(y_speed_test, 1e-6)))
                            diff[~valid_mask] = np.inf
                            intersection_params = x_test[np.argmin(diff)]
                except:
                    pass

                try:
                    speed_f = size_prime / (consts.C * speed_params_arr)
                    b_fit, log_a_fit = np.polyfit(speed_f, np.log(speed_epochs_arr), 1)
                    a_fit = np.exp(log_a_fit)
                    def speed_exp_model(P):
                        return a_fit * np.exp(b_fit * size_prime / (consts.C * P))
                    f_grok = interp1d(grok_params_arr, grok_epochs_arr, kind='linear', fill_value='extrapolate')
                    x_test = np.logspace(np.log10(grok_params_arr.min()), np.log10(grok_params_arr.max()), 1000)
                    y_grok_test = f_grok(x_test)
                    y_speed_exp_test = speed_exp_model(x_test)
                    valid_mask = (y_grok_test > 0) & (y_speed_exp_test > 0)
                    if valid_mask.sum() > 0:
                        diff = np.abs(np.log(np.maximum(y_grok_test, 1e-6)) - np.log(np.maximum(y_speed_exp_test, 1e-6)))
                        diff[~valid_mask] = np.inf
                        intersection_params_exp = x_test[np.argmin(diff)]
                except:
                    pass

                if global_a is not None and global_b is not None:
                    try:
                        def speed_global_model(P):
                            return global_a * np.exp(global_b * size_prime / (consts.C * P))
                        f_grok = interp1d(grok_params_arr, grok_epochs_arr, kind='linear', fill_value='extrapolate')
                        x_test = np.logspace(np.log10(grok_params_arr.min()), np.log10(grok_params_arr.max()), 1000)
                        y_grok_test = f_grok(x_test)
                        y_speed_global_test = speed_global_model(x_test)
                        valid_mask = (y_grok_test > 0) & (y_speed_global_test > 0)
                        if valid_mask.sum() > 0:
                            diff = np.abs(np.log(np.maximum(y_grok_test, 1e-6)) - np.log(np.maximum(y_speed_global_test, 1e-6)))
                            diff[~valid_mask] = np.inf
                            intersection_params_global = x_test[np.argmin(diff)]
                    except:
                        pass

            nlcv_data.append({
                'p': p_prime,
                'dataset_bits': size_prime,
                'critical_params': empirical_critical_params,
                'grok_speed': grok_speed_at_critical,
                'mem_speed': mem_speed_at_critical,
                'intersection_params': intersection_params,
                'intersection_params_exp': intersection_params_exp,
                'intersection_params_global': intersection_params_global
            })
            print(f"  p={p_prime}: critical_params={empirical_critical_params:,}")

        if len(nlcv_data) < 5:
            print(f"\nError: Need at least 5 primes for meaningful analysis (have {len(nlcv_data)})")
            return

        df = pd.DataFrame(nlcv_data)
        print("\n" + "="*80)
        print("DATA SUMMARY")
        print("="*80)
        print(df.to_string(index=False))

        # =================================================================
        # STEP 2: Define Kernel Ridge LOOCV helper
        # =================================================================

        def krr_loocv(X, y, gamma_values=None):
            """
            Perform LOOCV with Kernel Ridge Regression.
            Selects best gamma via inner LOOCV.
            Returns: LOOCV-R², LOOCV-MSE, best_gamma
            """
            if gamma_values is None:
                gamma_values = np.logspace(-2, 1, 10)  # Reduced grid for speed

            n = len(y)
            loo = LeaveOneOut()

            # Standardize features
            scaler = StandardScaler()

            best_gamma = None
            best_score = -np.inf

            # Select gamma via LOOCV
            for gamma in gamma_values:
                scores = []
                for train_idx, test_idx in loo.split(X):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    scaler_inner = StandardScaler()
                    X_train_scaled = scaler_inner.fit_transform(X_train)
                    X_test_scaled = scaler_inner.transform(X_test)

                    krr = KernelRidge(kernel='rbf', gamma=gamma, alpha=1.0)
                    krr.fit(X_train_scaled, y_train)
                    pred = krr.predict(X_test_scaled)
                    scores.append((y_test[0] - pred[0]) ** 2)

                mse = np.mean(scores)
                if -mse > best_score:
                    best_score = -mse
                    best_gamma = gamma

            # Final LOOCV with best gamma
            predictions = np.zeros(n)
            for train_idx, test_idx in loo.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                scaler_inner = StandardScaler()
                X_train_scaled = scaler_inner.fit_transform(X_train)
                X_test_scaled = scaler_inner.transform(X_test)

                krr = KernelRidge(kernel='rbf', gamma=best_gamma, alpha=1.0)
                krr.fit(X_train_scaled, y_train)
                predictions[test_idx] = krr.predict(X_test_scaled)

            residuals = y - predictions
            mse = np.mean(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum(residuals ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            return r2, mse, best_gamma, residuals

        def krr_loocv_fixed_gamma(X, y, gamma):
            """
            Fast LOOCV with fixed gamma (for permutation tests).
            """
            n = len(y)
            loo = LeaveOneOut()
            predictions = np.zeros(n)

            for train_idx, test_idx in loo.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train = y[train_idx]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                krr = KernelRidge(kernel='rbf', gamma=gamma, alpha=1.0)
                krr.fit(X_train_scaled, y_train)
                predictions[test_idx] = krr.predict(X_test_scaled)

            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - predictions) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            return r2

        # =================================================================
        # STEP 3: Define predictors
        # =================================================================

        all_predictors = {
            'dataset_bits': 'Dataset bits',
            'p': 'Prime (p)',
            'grok_speed': 'Grok speed',
            'mem_speed': 'Mem speed',
            'intersection_params': 'Intersection (empirical)',
            'intersection_params_exp': 'Intersection (per-prime exp)',
            'intersection_params_global': 'Intersection (global exp)'
        }

        # =================================================================
        # STEP 4: Univariate KRR LOOCV
        # =================================================================

        print("\n" + "="*80)
        print("UNIVARIATE KERNEL RIDGE REGRESSION (LOOCV)")
        print("="*80)

        y = df['critical_params'].values
        univariate_results = []

        for var, name in all_predictors.items():
            valid_mask = df[var].notna()
            n_valid = valid_mask.sum()

            if n_valid < 5:
                print(f"  {name:35s}: Insufficient data (n={n_valid})")
                continue

            X = df.loc[valid_mask, var].values.reshape(-1, 1)
            y_valid = df.loc[valid_mask, 'critical_params'].values

            r2_cv, mse_cv, best_gamma, _ = krr_loocv(X, y_valid)
            univariate_results.append((var, name, n_valid, r2_cv, np.sqrt(mse_cv), best_gamma))
            print(f"  {name:35s}: LOOCV-R²={r2_cv:.4f}, RMSE={np.sqrt(mse_cv):,.0f}, γ={best_gamma:.4f} (n={n_valid})")

        print("\n  Ranked by LOOCV-R²:")
        univariate_results.sort(key=lambda x: x[3], reverse=True)
        for i, (var, name, n, r2_cv, rmse, gamma) in enumerate(univariate_results):
            print(f"    {i+1}. {name:35s}: LOOCV-R² = {r2_cv:.4f}")

        # =================================================================
        # STEP 5: Incremental validity for each baseline (KRR)
        # =================================================================

        baselines = ['dataset_bits', 'p', 'grok_speed', 'mem_speed']
        baseline_names = {
            'dataset_bits': 'Dataset bits',
            'p': 'Prime (p)',
            'grok_speed': 'Grok speed',
            'mem_speed': 'Mem speed'
        }

        n_permutations = 100  # Reduced for KRR (use fixed gamma for speed)
        rng = np.random.default_rng(42)

        all_incremental_results = {}

        for baseline_var in baselines:
            baseline_name = baseline_names[baseline_var]

            print("\n" + "="*80)
            print(f"INCREMENTAL VALIDITY BEYOND: {baseline_name} (Kernel Ridge)")
            print("="*80)

            baseline_valid = df[baseline_var].notna() & df['critical_params'].notna()
            if baseline_valid.sum() < 5:
                print(f"  Insufficient data for baseline {baseline_name}")
                continue

            incremental_results = []

            for inc_var, inc_name in all_predictors.items():
                if inc_var == baseline_var:
                    continue

                valid_mask = baseline_valid & df[inc_var].notna()
                n_valid = valid_mask.sum()

                if n_valid < 5:
                    continue

                y_valid = df.loc[valid_mask, 'critical_params'].values
                X_base = df.loc[valid_mask, baseline_var].values.reshape(-1, 1)
                X_aug = np.column_stack([
                    df.loc[valid_mask, baseline_var].values,
                    df.loc[valid_mask, inc_var].values
                ])

                # LOOCV for baseline and augmented (with gamma selection)
                r2_base_cv, _, gamma_base, resid_base = krr_loocv(X_base, y_valid)
                r2_aug_cv, mse_aug_cv, gamma_aug, resid_aug = krr_loocv(X_aug, y_valid)
                delta_r2_cv = r2_aug_cv - r2_base_cv

                # Permutation test (use fixed gamma for speed)
                perm_deltas = []
                x_inc = df.loc[valid_mask, inc_var].values
                for _ in range(n_permutations):
                    x_inc_perm = rng.permutation(x_inc)
                    X_aug_perm = np.column_stack([
                        df.loc[valid_mask, baseline_var].values,
                        x_inc_perm
                    ])
                    r2_aug_perm = krr_loocv_fixed_gamma(X_aug_perm, y_valid, gamma_aug)
                    perm_deltas.append(r2_aug_perm - r2_base_cv)

                p_value = (1 + np.sum(np.array(perm_deltas) >= delta_r2_cv)) / (1 + n_permutations)

                incremental_results.append({
                    'var': inc_var,
                    'name': inc_name,
                    'n': n_valid,
                    'delta_r2_cv': delta_r2_cv,
                    'p_value': p_value,
                    'rmse_cv': np.sqrt(mse_aug_cv)
                })

            incremental_results.sort(key=lambda x: x['delta_r2_cv'], reverse=True)

            print(f"\n  {'Predictor':<35s} {'ΔR²(LOOCV)':>12s} {'p-value':>10s}")
            print("  " + "-"*60)
            for res in incremental_results:
                sig = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else ""
                print(f"  {res['name']:<35s} {res['delta_r2_cv']:>+12.4f} {res['p_value']:>9.4f} {sig}")

            all_incremental_results[baseline_var] = incremental_results

        # =================================================================
        # STEP 6: Summary
        # =================================================================

        print("\n" + "="*80)
        print("SUMMARY: BEST INCREMENTAL PREDICTOR (NON-LINEAR)")
        print("="*80)

        print(f"\n  {'Baseline':<20s} {'Best Predictor':<35s} {'ΔR²':>10s} {'p-value':>10s}")
        print("  " + "-"*78)

        rank1_counts = {}
        for baseline_var, results in all_incremental_results.items():
            if results:
                best = results[0]
                sig = "***" if best['p_value'] < 0.001 else "**" if best['p_value'] < 0.01 else "*" if best['p_value'] < 0.05 else ""
                print(f"  {baseline_names[baseline_var]:<20s} {best['name']:<35s} {best['delta_r2_cv']:>+10.4f} {best['p_value']:>9.4f} {sig}")
                rank1_counts[best['var']] = rank1_counts.get(best['var'], 0) + 1

        print("\n  Times ranked #1:")
        for var, count in sorted(rank1_counts.items(), key=lambda x: -x[1]):
            print(f"    {all_predictors[var]:<35s}: {count}")

        print("\n" + "="*80)

    elif args.ccv:
        # =====================================================================
        # CURVE-BASED CROSS-VALIDATION
        # =====================================================================
        print("\n" + "="*80)
        print("CURVE-BASED CROSS-VALIDATION")
        print("="*80)
        print("Testing: Does intersection beat full curve embeddings?")

        from scipy import stats
        from scipy.interpolate import interp1d
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import LeaveOneOut
        import pandas as pd

        # =================================================================
        # STEP 1: Collect curve data for all primes
        # =================================================================

        print("\n" + "-"*60)
        print("Collecting curve data for each prime...")
        print("-"*60)

        curve_data = []
        all_grok_param_counts = set()
        all_speed_param_counts = set()

        # First pass: collect all unique param counts
        for p_prime in primes_list:
            n_prime, size_prime = compute_dataset_size_bits(p_prime, args.op, args.training_fraction)

            prime_results = aggregate_grokking_results_across_seeds(
                p_prime, index, filters,
                args.threshold_train, args.threshold_val, args.saturation_threshold,
                use_min_delay=True, dims=getattr(args, 'dims', None)
            )

            if not prime_results:
                continue

            if args.max_dim is not None:
                prime_results = [r for r in prime_results if r['dim'] <= args.max_dim]

            speed_data = aggregate_speed_results_across_seeds(p_prime, n_prime, index, filters, args.batch_size, args.saturation_threshold)

            for result in prime_results:
                if result.get('epochs_to_grok') is not None and result['epochs_to_grok'] > 0:
                    all_grok_param_counts.add(result['param_count'])

            for sp in speed_data:
                if sp.get('saturation_epoch') is not None and sp['saturation_epoch'] > 0:
                    all_speed_param_counts.add(sp['param_count'])

        all_grok_param_counts = sorted(all_grok_param_counts)
        all_speed_param_counts = sorted(all_speed_param_counts)

        print(f"  Found {len(all_grok_param_counts)} unique grok param counts")
        print(f"  Found {len(all_speed_param_counts)} unique speed param counts")

        # Second pass: build curve embeddings
        for p_prime in primes_list:
            n_prime, size_prime = compute_dataset_size_bits(p_prime, args.op, args.training_fraction)

            prime_results = aggregate_grokking_results_across_seeds(
                p_prime, index, filters,
                args.threshold_train, args.threshold_val, args.saturation_threshold,
                use_min_delay=True, dims=getattr(args, 'dims', None)
            )

            if not prime_results:
                continue

            if args.max_dim is not None:
                prime_results = [r for r in prime_results if r['dim'] <= args.max_dim]
                if not prime_results:
                    continue

            speed_data = aggregate_speed_results_across_seeds(p_prime, n_prime, index, filters, args.batch_size, args.saturation_threshold)

            # Compute critical params
            delay_data = sorted(prime_results, key=lambda x: x['param_count'])
            empirical_critical_params = None
            last_zero_idx = -1
            for i in range(len(delay_data) - 1, -1, -1):
                if delay_data[i]['delay'] == 0:
                    last_zero_idx = i
                    break
            if last_zero_idx >= 0 and last_zero_idx + 1 < len(delay_data):
                empirical_critical_params = delay_data[last_zero_idx + 1]['param_count']
            elif last_zero_idx == -1 and delay_data[0]['delay'] > 0:
                empirical_critical_params = delay_data[0]['param_count']

            if empirical_critical_params is None:
                continue

            # Build grok curve embedding
            grok_by_param = {r['param_count']: r.get('epochs_to_grok') for r in prime_results}
            grok_embedding = []
            for pc in all_grok_param_counts:
                val = grok_by_param.get(pc)
                grok_embedding.append(val if val is not None and val > 0 else np.nan)

            # Build speed curve embedding
            speed_by_param = {sp['param_count']: sp.get('saturation_epoch') for sp in speed_data}
            speed_embedding = []
            for pc in all_speed_param_counts:
                val = speed_by_param.get(pc)
                speed_embedding.append(val if val is not None and val > 0 else np.nan)

            # Compute intersection
            grok_params = []
            grok_epochs = []
            for result in sorted(prime_results, key=lambda x: x['param_count']):
                etg = result.get('epochs_to_grok')
                if etg is not None and etg > 0:
                    grok_params.append(result['param_count'])
                    grok_epochs.append(etg)

            speed_params = []
            speed_epochs = []
            for sp in sorted(speed_data, key=lambda x: x['param_count']):
                if sp.get('saturation_epoch') is not None and sp['saturation_epoch'] > 0:
                    speed_params.append(sp['param_count'])
                    speed_epochs.append(sp['saturation_epoch'])

            intersection_params_exp = None
            if len(grok_params) >= 2 and len(speed_params) >= 2:
                try:
                    grok_params_arr = np.array(grok_params)
                    grok_epochs_arr = np.array(grok_epochs)
                    speed_params_arr = np.array(speed_params)
                    speed_epochs_arr = np.array(speed_epochs)

                    speed_f = size_prime / (consts.C * speed_params_arr)
                    b_fit, log_a_fit = np.polyfit(speed_f, np.log(speed_epochs_arr), 1)
                    a_fit = np.exp(log_a_fit)

                    def speed_exp_model(P):
                        return a_fit * np.exp(b_fit * size_prime / (consts.C * P))

                    f_grok = interp1d(grok_params_arr, grok_epochs_arr, kind='linear', fill_value='extrapolate')
                    x_test = np.logspace(np.log10(grok_params_arr.min()), np.log10(grok_params_arr.max()), 1000)
                    y_grok_test = f_grok(x_test)
                    y_speed_exp_test = speed_exp_model(x_test)
                    valid_mask = (y_grok_test > 0) & (y_speed_exp_test > 0)
                    if valid_mask.sum() > 0:
                        diff = np.abs(np.log(np.maximum(y_grok_test, 1e-6)) - np.log(np.maximum(y_speed_exp_test, 1e-6)))
                        diff[~valid_mask] = np.inf
                        intersection_params_exp = x_test[np.argmin(diff)]
                except:
                    pass

            curve_data.append({
                'p': p_prime,
                'critical_params': empirical_critical_params,
                'intersection_exp': intersection_params_exp,
                'grok_embedding': grok_embedding,
                'speed_embedding': speed_embedding
            })
            print(f"  p={p_prime}: critical_params={empirical_critical_params:,}")

        if len(curve_data) < 5:
            print(f"\nError: Need at least 5 primes (have {len(curve_data)})")
            return

        print(f"\n  Collected data for {len(curve_data)} primes")
        print(f"  Grok embedding dimension: {len(all_grok_param_counts)}")
        print(f"  Speed embedding dimension: {len(all_speed_param_counts)}")

        # =================================================================
        # STEP 2: Define KRR LOOCV with NaN handling
        # =================================================================

        def krr_loocv_with_nan(X, y, gamma_values=None):
            """
            KRR LOOCV that handles NaN by using only valid features per sample.
            Uses imputation: replace NaN with column mean from training set.
            """
            import warnings
            # Suppress expected warnings from nanmean on empty slices
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'Mean of empty slice')
                return _krr_loocv_with_nan_impl(X, y, gamma_values)

        def _krr_loocv_with_nan_impl(X, y, gamma_values):
            if gamma_values is None:
                gamma_values = np.logspace(-3, 2, 15)

            n = len(y)
            loo = LeaveOneOut()

            best_gamma = gamma_values[len(gamma_values) // 2]  # Default
            best_score = -np.inf

            # Select gamma via LOOCV
            for gamma in gamma_values:
                scores = []
                for train_idx, test_idx in loo.split(X):
                    X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
                    y_train, y_test = y[train_idx], y[test_idx]

                    # Impute NaN with column mean from training set
                    col_means = np.nanmean(X_train, axis=0)
                    for j in range(X_train.shape[1]):
                        mask = np.isnan(X_train[:, j])
                        X_train[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0
                        mask_test = np.isnan(X_test[:, j])
                        X_test[mask_test, j] = col_means[j] if not np.isnan(col_means[j]) else 0

                    # Remove columns that are all NaN
                    valid_cols = ~np.all(np.isnan(X_train) | (X_train == 0), axis=0)
                    if valid_cols.sum() == 0:
                        scores.append(np.inf)
                        continue

                    X_train = X_train[:, valid_cols]
                    X_test = X_test[:, valid_cols]

                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    krr = KernelRidge(kernel='rbf', gamma=gamma, alpha=1.0)
                    krr.fit(X_train_scaled, y_train)
                    pred = krr.predict(X_test_scaled)
                    scores.append((y_test[0] - pred[0]) ** 2)

                mse = np.mean(scores)
                if -mse > best_score:
                    best_score = -mse
                    best_gamma = gamma

            # Final LOOCV with best gamma
            predictions = np.zeros(n)
            for train_idx, test_idx in loo.split(X):
                X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
                y_train = y[train_idx]

                col_means = np.nanmean(X_train, axis=0)
                for j in range(X_train.shape[1]):
                    mask = np.isnan(X_train[:, j])
                    X_train[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0
                    mask_test = np.isnan(X_test[:, j])
                    X_test[mask_test, j] = col_means[j] if not np.isnan(col_means[j]) else 0

                valid_cols = ~np.all(np.isnan(X_train) | (X_train == 0), axis=0)
                if valid_cols.sum() == 0:
                    predictions[test_idx] = np.mean(y_train)
                    continue

                X_train = X_train[:, valid_cols]
                X_test = X_test[:, valid_cols]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                krr = KernelRidge(kernel='rbf', gamma=best_gamma, alpha=1.0)
                krr.fit(X_train_scaled, y_train)
                predictions[test_idx] = krr.predict(X_test_scaled)

            residuals = y - predictions
            mse = np.mean(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum(residuals ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            return r2, mse, best_gamma

        # =================================================================
        # STEP 3: Prepare feature matrices
        # =================================================================

        y = np.array([d['critical_params'] for d in curve_data])

        # Intersection (scalar)
        X_intersection = np.array([d['intersection_exp'] if d['intersection_exp'] is not None else np.nan
                                   for d in curve_data]).reshape(-1, 1)

        # Grok curve embedding
        X_grok = np.array([d['grok_embedding'] for d in curve_data])

        # Speed curve embedding
        X_speed = np.array([d['speed_embedding'] for d in curve_data])

        # Both curves
        X_both = np.hstack([X_grok, X_speed])

        # =================================================================
        # STEP 4: Compare models
        # =================================================================

        print("\n" + "="*80)
        print("MODEL COMPARISON (Kernel Ridge Regression)")
        print("="*80)

        models = [
            ('Intersection (scalar)', X_intersection),
            ('Grok curve (embedding)', X_grok),
            ('Speed curve (embedding)', X_speed),
            ('Both curves (embedding)', X_both)
        ]

        results = []
        for name, X in models:
            # Check for valid data
            valid_rows = ~np.all(np.isnan(X), axis=1)
            if valid_rows.sum() < 5:
                print(f"  {name:30s}: Insufficient valid data")
                continue

            X_valid = X[valid_rows]
            y_valid = y[valid_rows]

            r2_cv, mse_cv, best_gamma = krr_loocv_with_nan(X_valid, y_valid)
            results.append((name, valid_rows.sum(), r2_cv, np.sqrt(mse_cv), best_gamma))
            print(f"  {name:30s}: LOOCV-R²={r2_cv:.4f}, RMSE={np.sqrt(mse_cv):,.0f}, γ={best_gamma:.4f} (n={valid_rows.sum()})")

        print("\n  Ranked by LOOCV-R²:")
        results.sort(key=lambda x: x[2], reverse=True)
        for i, (name, n, r2, rmse, gamma) in enumerate(results):
            print(f"    {i+1}. {name:30s}: LOOCV-R² = {r2:.4f}")

        # =================================================================
        # STEP 5: Statistical comparison
        # =================================================================

        print("\n" + "="*80)
        print("KEY COMPARISONS")
        print("="*80)

        # Find results by name
        result_dict = {r[0]: r for r in results}

        if 'Intersection (scalar)' in result_dict and 'Grok curve (embedding)' in result_dict:
            r2_inter = result_dict['Intersection (scalar)'][2]
            r2_grok = result_dict['Grok curve (embedding)'][2]
            print(f"\n  Intersection vs Grok curve:")
            print(f"    Intersection: LOOCV-R² = {r2_inter:.4f}")
            print(f"    Grok curve:   LOOCV-R² = {r2_grok:.4f}")
            if r2_inter > r2_grok:
                print(f"    → Intersection is better by {r2_inter - r2_grok:.4f}")
            else:
                print(f"    → Grok curve is better by {r2_grok - r2_inter:.4f}")

        if 'Intersection (scalar)' in result_dict and 'Speed curve (embedding)' in result_dict:
            r2_inter = result_dict['Intersection (scalar)'][2]
            r2_speed = result_dict['Speed curve (embedding)'][2]
            print(f"\n  Intersection vs Speed curve:")
            print(f"    Intersection: LOOCV-R² = {r2_inter:.4f}")
            print(f"    Speed curve:  LOOCV-R² = {r2_speed:.4f}")
            if r2_inter > r2_speed:
                print(f"    → Intersection is better by {r2_inter - r2_speed:.4f}")
            else:
                print(f"    → Speed curve is better by {r2_speed - r2_inter:.4f}")

        if 'Both curves (embedding)' in result_dict:
            r2_both = result_dict['Both curves (embedding)'][2]
            print(f"\n  Both curves combined: LOOCV-R² = {r2_both:.4f}")

            if 'Grok curve (embedding)' in result_dict:
                r2_grok = result_dict['Grok curve (embedding)'][2]
                if r2_both > r2_grok:
                    print(f"    → Better than grok alone by {r2_both - r2_grok:.4f}")

            if 'Speed curve (embedding)' in result_dict:
                r2_speed = result_dict['Speed curve (embedding)'][2]
                if r2_both > r2_speed:
                    print(f"    → Better than speed alone by {r2_both - r2_speed:.4f}")

        # =================================================================
        # STEP 6: Interpretation
        # =================================================================

        print("\n" + "="*80)
        print("INTERPRETATION")
        print("="*80)

        if results:
            best = results[0]
            print(f"\n  Best model: {best[0]}")
            print(f"  LOOCV-R² = {best[2]:.4f}")

            if 'Intersection' in best[0]:
                print("\n  → CONCLUSION: The intersection point (a scalar) is the best predictor,")
                print("    outperforming full curve embeddings. This suggests the intersection")
                print("    captures the essential information from both curves.")
            elif 'Both' in best[0]:
                print("\n  → CONCLUSION: Using both curves together is best. The intersection")
                print("    may not fully capture all relevant curve information.")
            else:
                print(f"\n  → CONCLUSION: {best[0]} is the best predictor.")

        print("\n" + "="*80)

    else:
        print("Error: Please specify one of --critical, --speed, --groks, --delay, --correlation, --cv, --nlcv, or --ccv")
        print("  --critical:    Plot empirical critical parameter count vs prime")
        print("  --speed:       Plot saturation time vs capacity fraction for multiple primes")
        print("  --groks:       Plot critical capacity (from groks line-fitting) vs prime")
        print("  --delay:       Plot grokking delay vs capacity fraction for multiple primes")
        print("  --correlation: Correlation analysis of what determines critical params")
        print("  --cv:          Cross-validation test of intersection's incremental validity")
        print("  --nlcv:        Non-linear CV with kernel ridge regression")
        print("  --ccv:         Curve-based CV comparing intersection vs full curve embeddings")


def main():
    parser = argparse.ArgumentParser(
        description='View saved experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all grokking results
  python visualise.py groks --list

  # Plot capacity curves for all experiments
  python visualise.py capacity --all --curves

  # Get capacity summary
  python visualise.py capacity --all --summary
"""
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Experiment type"
    )

    # =========================================================================
    # Grokking subparser
    # =========================================================================
    grok_subparser = subparsers.add_parser('groks', help='Grokking experiments')
    grok_subparser.set_defaults(func=groks)

    add_vis_output_args(grok_subparser, plot_dir_default='media/groks')
    add_vis_file_selection_args(grok_subparser)
    add_vis_model_filter_args(grok_subparser)
    add_vis_optimizer_filter_args(grok_subparser)
    add_vis_task_args(grok_subparser, op_default='/')
    add_vis_dim_args(grok_subparser)

    grok_subparser.add_argument('--split-type', type=str, default='random',
                       help='Split type (default: random)',
                       choices=['random', 'sequential', 'alternating'])
    grok_subparser.add_argument('--plot', type=str,
                       help='Plot a specific result file')
    grok_subparser.add_argument('--combined', action='store_true',
                       help='Compare all available results (combined plot)')
    grok_subparser.add_argument('--separate', action='store_true',
                       help='Plot training and validation in separate subplots')
    grok_subparser.add_argument('--delay', action='store_true',
                       help='Plot grokking delay vs parameter count')
    grok_subparser.add_argument('--integral', action='store_true',
                       help='Plot grokking integral vs parameter count (sum of min(train_acc - val_acc, 0) after train reaches threshold)')
    grok_subparser.add_argument('--critical', action='store_true',
                       help='Find and plot critical model capacity where grokking delay first becomes positive (extrapolated from linear fit)')
    grok_subparser.add_argument('--time', action='store_true',
                       help='Plot absolute grokking time (epochs to reach threshold) vs parameter count')
    grok_subparser.add_argument('--speed', action='store_true',
                       help='Plot grokking delay with overlaid steps axis, showing steps to grok and steps to learn (from speed data) for a single prime. '
                            'For multi-prime speed analysis, use "primes --speed".')
    grok_subparser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size used in grokking experiments (default: 512)')
    grok_subparser.add_argument('--show-mem', action='store_true',
                       help='Show memorisation metrics (M_T and M_U). When used alone: plot memorisation curves for all selected files. '
                            'M_T (total memorisation) is always available; M_U (unintended) requires --baseline during training. '
                            'When used with --plot: overlay memorisation on top of train/val accuracy curves. '
                            'When used with --separate: overlay memorisation on both training and validation subplots. '
                            'When used with --delay: also plot delay vs maximum memorisation.')
    grok_subparser.add_argument('--threshold-train', type=float, default=99.0,
                       help='Accuracy threshold for training (default: 99.0)')
    grok_subparser.add_argument('--threshold-val', type=float, default=97.0,
                       help='Accuracy threshold for validation (default: 97.0)')
    grok_subparser.add_argument('--delay-threshold', type=float, default=0.5,
                       help='Minimum delay to include in critical capacity fit (default: 0.5)')
    grok_subparser.add_argument('--saturation-threshold', type=float, default=99.0,
                       help='Accuracy threshold for determining saturation epoch in speed experiments (default: 99.0)')

    # =========================================================================
    # Capacity subparser
    # =========================================================================
    cap_subparser = subparsers.add_parser(
        'capacity',
        help='Model capacity (memorisation) experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all capacity results
  python visualise.py capacity --list

  # Plot Morris et al. style capacity curves
  python visualise.py capacity --all --curves

  # Plot and save curves
  python visualise.py capacity --all --curves --save

  # Get summary statistics
  python visualise.py capacity --all --summary

  # Filter by model dimension
  python visualise.py capacity --dims 20 40 80 --curves

  # Plot single experiment
  python visualise.py capacity --plot data/capacity/capacity_dim40_samples1000.npz
"""
    )
    cap_subparser.set_defaults(func=capacity)

    add_vis_output_args(cap_subparser, plot_dir_default='media/capacity')
    add_vis_file_selection_args(cap_subparser)
    add_vis_model_filter_args(cap_subparser)
    add_vis_optimizer_filter_args(cap_subparser)
    add_vis_dim_args(cap_subparser)

    cap_subparser.add_argument('--dataset-type', type=str, default='random',
                               help='Dataset type (default: random)',
                               choices=['random', '+', '-', '*', '/'])
    cap_subparser.add_argument('--plot', type=str, metavar='FILE',
                               help='Plot training curves for a specific result file')
    cap_subparser.add_argument('--samples', nargs='+', type=int, metavar='N',
                               help='Filter by dataset sizes (e.g., --samples 100 1000)')
    cap_subparser.add_argument('--curves', action='store_true',
                               help='Plot memorisation vs dataset size curves (Morris et al. style)')
    cap_subparser.add_argument('--accuracy', action='store_true',
                               help='Plot bits memorized vs training accuracy')
    cap_subparser.add_argument('--summary', action='store_true',
                               help='Print summary statistics and capacity estimate')

    # =========================================================================
    # Speed subparser
    # =========================================================================
    speed_subparser = subparsers.add_parser(
        'speed',
        help='Learning speed experiment visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all speed results
  python visualise.py speed --list

  # Plot learning speed curves for all experiments
  python visualise.py speed --all --curves

  # Plot combined analysis (curves + speed vs model size)
  python visualise.py speed --all --combined

  # Plot saturation time vs capacity fraction
  python visualise.py speed --all --fraction

  # Get summary statistics
  python visualise.py speed --all --summary

  # Filter by model dimension
  python visualise.py speed --dims 20 28 --curves --save
"""
    )
    speed_subparser.set_defaults(func=speed)

    add_vis_output_args(speed_subparser, plot_dir_default='media/speed')
    add_vis_file_selection_args(speed_subparser)
    add_vis_model_filter_args(speed_subparser)
    add_vis_optimizer_filter_args(speed_subparser)
    add_vis_task_args(speed_subparser, op_default='/')
    add_vis_dim_args(speed_subparser)

    speed_subparser.add_argument('--curves', action='store_true',
                                 help='Plot learning speed curves (steps to saturation vs dataset size)')
    speed_subparser.add_argument('--combined', action='store_true',
                                 help='Plot combined analysis (curves + speed vs model size)')
    speed_subparser.add_argument('--fraction', action='store_true',
                                 help='Plot saturation time vs f where f=S/(CP) is capacity fraction')
    speed_subparser.add_argument('--rate', action='store_true',
                                 help='Plot dT/dS (rate of change of saturation time) vs dataset size S')
    speed_subparser.add_argument('--rate-k', type=int, default=10,
                                 help='Delta k for rate estimation: dT/dS = (T(n+k) - T(n)) / k (default: 10)')
    speed_subparser.add_argument('--batch-size', type=int, default=512,
                                 help='Batch size used in speed experiments (default: 512)')
    speed_subparser.add_argument('--saturation-threshold', type=float, default=99.0,
                                 help='Accuracy threshold for determining saturation epoch (default: 99.0)')
    speed_subparser.add_argument('--summary', action='store_true',
                                 help='Print summary statistics')

    # =========================================================================
    # Primes subparser
    # =========================================================================
    primes_subparser = subparsers.add_parser(
        'primes',
        help='Multi-prime analysis visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot empirical critical parameter count vs prime
  python visualise.py primes --p 97 113 127 --critical

  # Plot saturation time vs capacity fraction for multiple primes
  python visualise.py primes --p 97 113 127 --speed

  # Plot critical capacity from groks vs prime
  python visualise.py primes --p 97 113 127 --groks

  # Save plots
  python visualise.py primes --p 97 113 127 --critical --save
"""
    )
    primes_subparser.set_defaults(func=primes)

    add_vis_output_args(primes_subparser, plot_dir_default='media/primes')
    add_vis_model_filter_args(primes_subparser)
    add_vis_optimizer_filter_args(primes_subparser)
    add_vis_task_args(primes_subparser, op_default='/')
    add_vis_dim_args(primes_subparser)

    primes_subparser.add_argument('--p', nargs='+', type=int, default=None,
                                   help='Prime numbers (e.g., --p 97 113 127). '
                                        'Mutually exclusive with --min-prime/--max-prime.')
    primes_subparser.add_argument('--min-prime', type=int, default=None,
                                   help='Lower bound for prime auto-detection (inclusive). '
                                        'Use with --max-prime instead of --p.')
    primes_subparser.add_argument('--max-prime', type=int, default=None,
                                   help='Upper bound for prime auto-detection (inclusive). '
                                        'Use with --min-prime instead of --p.')
    primes_subparser.add_argument('--seed', type=int, default=42,
                                   help='Training seed (default: 42)')
    primes_subparser.add_argument('--split-type', type=str, default='random',
                                   help='Split type (default: random)',
                                   choices=['random', 'sequential', 'alternating'])
    primes_subparser.add_argument('--pattern', type=str,
                                   help='Glob pattern to match result files')
    primes_subparser.add_argument('--max-dim', type=int, default=None,
                                   help='Maximum dimension to consider when computing empirical grokking point '
                                        '(useful when speed data only exists for smaller models)')
    primes_subparser.add_argument('--match-table', type=str, default=None,
                                   help='Path to a match table JSON produced by run_config.py or '
                                        'scripts/migrate_legacy_data.py. When provided, loads paired '
                                        '(groks, speed) data from the match table instead of using '
                                        'glob-based file discovery.')

    # Analysis modes (mutually exclusive)
    analysis_group = primes_subparser.add_mutually_exclusive_group(required=True)
    analysis_group.add_argument('--critical', action='store_true',
                                help='Plot empirical critical parameter count vs prime (first param size where minimum delay across seeds is non-zero)')
    analysis_group.add_argument('--speed', action='store_true',
                                help='Plot saturation time vs capacity fraction for multiple primes')
    analysis_group.add_argument('--groks', action='store_true',
                                help='Plot critical capacity from groks (line-fitting method) vs prime')
    analysis_group.add_argument('--delay', action='store_true',
                                help='Plot grokking delay (min across seeds) vs capacity fraction for multiple primes')
    analysis_group.add_argument('--correlation', action='store_true',
                                help='Correlation analysis: what determines critical parameter count?')
    analysis_group.add_argument('--cv', action='store_true',
                                help='Cross-validation analysis: test if intersection adds predictive value beyond dataset size')
    analysis_group.add_argument('--nlcv', action='store_true',
                                help='Non-linear CV: same as --cv but with kernel ridge regression for non-linear relationships')
    analysis_group.add_argument('--ccv', action='store_true',
                                help='Curve CV: test if intersection beats full curve embeddings as predictors')

    # Threshold parameters
    primes_subparser.add_argument('--threshold-train', type=float, default=99.0,
                                   help='Accuracy threshold for training (default: 99.0)')
    primes_subparser.add_argument('--threshold-val', type=float, default=97.0,
                                   help='Accuracy threshold for validation (default: 97.0)')
    primes_subparser.add_argument('--delay-threshold', type=float, default=0.5,
                                   help='Minimum delay to include in critical capacity fit (default: 0.5)')
    primes_subparser.add_argument('--saturation-threshold', type=float, default=99.0,
                                   help='Accuracy threshold for determining saturation epoch in speed experiments '
                                        '(default: 99.0, only used for --speed and --groks)')
    primes_subparser.add_argument('--batch-size', type=int, default=512,
                                   help='Batch size used in grokking experiments (default: 512, only used for --speed)')
    primes_subparser.add_argument('--predicted-speed', action='store_true',
                                   help='For --groks: use overall exponential fit (from all speed data) to predict memorization speed curve instead of per-model data')
    primes_subparser.add_argument('--global-fit', action='store_true',
                                   help='For --groks: also compute and plot intersection using global exponential fit across all primes')
    primes_subparser.add_argument('--prime-fit', action='store_true',
                                   help='For --groks: also compute and plot intersection using per-prime exponential fit')

    # Run the appropriate function based on the subparser
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
