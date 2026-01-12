"""
Utility script to view saved experiment results.

Subcommands:
    groks       - Grokking experiment visualizations
    capacity    - Model capacity (memorisation) experiment visualizations
    speed       - Learning speed experiment visualizations
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
    plot_cross_exp_critical,
    plot_grokking_delay_with_speed,
    plot_learning_speed_curves,
    plot_speed_vs_model_size,
    plot_combined_speed_analysis,
    plot_saturation_time_vs_capacity_fraction,
    plot_saturation_steps_vs_params,
    plot_rate_vs_dataset_size
)
from utils import compute_dataset_size_bits
import consts


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


def plot_grokking_delay(result_files, threshold_train=99.0, threshold_val=97.0, 
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


def plot_grokking_time(result_files, threshold_val=97.0, save_path=None, show=True):
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


def groks(args):
    # Set data and plot directories
    signature = f'p{args.p}_seed{args.training_seed}_split{args.split_type}'
    data_dir = os.path.join(args.data_dir, signature)
    plot_dir = os.path.join(args.plot_dir, signature)

    print(f'Using signature {signature}')
    
    # Determine show/save settings
    show = not args.no_show
    
    if args.list:
        # Use default pattern when none is provided
        pattern = args.pattern or 'grokking_dim*.npz'
        list_results(data_dir, pattern)
    
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
    
    files = set()

    if args.files:
        files.update(args.files)

    if args.all:
        pattern = os.path.join(data_dir, f'grokking_dim*_depth{args.depth}_heads{args.heads}.npz')
        files.update(glob(pattern))

    if args.pattern:
        pattern = os.path.join(data_dir, args.pattern)
        files.update(glob(pattern))
    
    if args.dims:
        files.update([os.path.join(data_dir, f'grokking_dim{d}_depth{args.depth}_heads{args.heads}.npz') 
                      for d in args.dims])
    
    if args.dims_start and args.dims_end:
        dims = list(range(args.dims_start, args.dims_end + 1, args.dims_step))
        files.update([os.path.join(data_dir, f'grokking_dim{d}_depth{args.depth}_heads{args.heads}.npz') 
                      for d in dims])
    
    files = sorted(files, key=extract_dim)

    if not files:
        print("No files found. Use --list to see available results, or --all to select all.")
        return
    
    # Calculate number of parameters to lower bound capacity of model to memorise all training data
    n, size = compute_dataset_size_bits(args.p, args.op, args.training_fraction)
    threshold_params = size / consts.C
    
    print(f"Found {len(files)} files to analyze:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    # Create plot directory if saving
    if args.save:
        os.makedirs(plot_dir, exist_ok=True)
    
    # Load data with memorisation traces
    results = []
    for fname in files:
        if not os.path.exists(fname):
            print(f"Warning: File not found: {fname}")
            continue
        data = np.load(fname)
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
            result['mem_trace'] = data['mem_u_trace']
        # Legacy support for old 'mem_trace' field (was M_U)
        elif 'mem_trace' in data:
            result['mem_u_trace'] = data['mem_trace']
            result['mem_trace'] = data['mem_trace']
        results.append(result)

    # Handle --show-mem when used standalone (without --plot)
    if args.show_mem:
        print(f"\nPlotting memorisation curves for {len(files)} results...")
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
                save_path = os.path.join(plot_dir, f'mem_t_p{args.p}.pdf') if args.save else None
                plot_memorization_curves(results_with_mem_t, mem_key='mem_t_trace', title='Total Memorisation (M_T)',
                                        save_path=save_path, show=show)
                
                # Plot final M_T vs parameter count
                save_path = os.path.join(plot_dir, f'mem_t_vs_params_p{args.p}.pdf') if args.save else None
                plot_max_memorization_vs_params(results_with_mem_t, mem_key='mem_t_trace', title='Final M_T vs Parameters',
                                                 save_path=save_path, show=show)
            
            # Plot M_U curves over training (if available)
            if results_with_mem_u:
                save_path = os.path.join(plot_dir, f'mem_u_p{args.p}.pdf') if args.save else None
                plot_memorization_curves(results_with_mem_u, mem_key='mem_u_trace', title='Unintended Memorisation (M_U)',
                                        save_path=save_path, show=show)
                
                # Plot final M_U vs parameter count
                save_path = os.path.join(plot_dir, f'mem_u_vs_params_p{args.p}.pdf') if args.save else None
                plot_max_memorization_vs_params(results_with_mem_u, mem_key='mem_u_trace', title='Final M_U vs Parameters',
                                                 save_path=save_path, show=show)

    if args.separate:
        print(f"Plotting separate curves for {len(files)} results...")
        save_path = os.path.join(plot_dir, f'grokking_separate_p{args.p}.pdf') if args.save else None
        plot_separate_curves(results, save_path=save_path, show=show)
        # Check if we should overlay memorisation
        if args.show_mem:
            print(f"Plotting separate curves with memorisation for {len(files)} results...")
            save_path = os.path.join(plot_dir, f'grokking_separate_mem_p{args.p}.pdf') if args.save else None
            plot_separate_curves_with_memorization(results, save_path=save_path, show=show)
    
    if args.combined:
        print(f"Plotting combined curves for {len(files)} results...")
        save_path = os.path.join(plot_dir, f'grokking_combined_p{args.p}.pdf') if args.save else None
        plot_combined_curves(results, save_path=save_path, show=show)
    
    if args.delay:
        # Standard delay plot

        save_path = os.path.join(plot_dir, f'grokking_delay_p{args.p}.pdf') if args.save else None
        plot_grokking_delay(files, threshold_train=args.threshold_train, threshold_val=args.threshold_val,
                            save_path=save_path, show=show, threshold_params=threshold_params)
        # Check if we should also plot delay vs memorisation
        if args.show_mem:
            # Plot delay vs memorisation
            save_path = os.path.join(plot_dir, f'delay_vs_memorisation_p{args.p}.pdf') if args.save else None
            plot_delay_vs_memorization(results, threshold_train=args.threshold_train, threshold_val=args.threshold_val,
                                      save_path=save_path, show=show)
            # Plot delay and memorisation vs parameter count on same graph
            save_path = os.path.join(plot_dir, f'delay_and_mem_vs_params_p{args.p}.pdf') if args.save else None
            plot_delay_and_memorization_vs_params(results, threshold_train=args.threshold_train, threshold_val=args.threshold_val,
                                                 save_path=save_path, show=show)
    
    if args.critical:
        save_path = os.path.join(plot_dir, f'critical_capacity_p{args.p}.pdf') if args.save else None
        critical_params = plot_grokking_critical_capacity(results, threshold_train=args.threshold_train, 
                                                          threshold_val=args.threshold_val,
                                                          delay_threshold=args.delay_threshold,
                                                          save_path=save_path, show=show)
        if critical_params:
            print(f"\n{'='*80}")
            print(f"CRITICAL CAPACITY: {critical_params:,.0f} parameters")
            print(f"{'='*80}")
    
    if args.time:
        save_path = os.path.join(plot_dir, f'grokking_time_p{args.p}.pdf') if args.save else None
        plot_grokking_time(files, threshold_val=args.threshold_val, save_path=save_path, show=show)
    
    if args.speed:
        # Load speed data, prioritizing matching seed but allowing fallback
        speed_data = []
        speed_signature = f'p{args.p}_seed{args.training_seed}'
        speed_dir = os.path.join('data/speed', speed_signature)

        # First try to load with matching seed
        if os.path.exists(speed_dir):
            speed_files = glob(os.path.join(speed_dir, 'speed_dim*.npz'))
            if speed_files:
                for sf in speed_files:
                    data = np.load(sf)
                    if int(data['n_samples']) == int(n):
                        speed_data.append({
                            'dim': int(data['dim']),
                            'param_count': int(data['param_count']),
                            'saturation_step': int(data['saturation_step']),
                            'n_samples': int(data['n_samples'])
                        })
                print(f"Loaded {len(speed_data)} speed experiments from {speed_dir} (matching seed)")

        # If no matching seed found, search for any available seed with same p
        if not speed_data:
            speed_base_dir = os.path.join('data/speed')
            if os.path.exists(speed_base_dir):
                # Find all directories matching p{args.p}_seed*
                import re
                pattern = re.compile(rf'^p{args.p}_seed\d+$')
                available_dirs = [d for d in os.listdir(speed_base_dir)
                                 if pattern.match(d) and os.path.isdir(os.path.join(speed_base_dir, d))]

                if available_dirs:
                    # Sort to get consistent behavior (prefer lower seed numbers as fallback)
                    available_dirs = sorted(available_dirs, key=lambda x: int(re.search(r'seed(\d+)', x).group(1)))
                    fallback_dir = os.path.join(speed_base_dir, available_dirs[0])

                    speed_files = glob(os.path.join(fallback_dir, 'speed_dim*.npz'))
                    for sf in speed_files:
                        data = np.load(sf)
                        if int(data['n_samples']) == int(n):
                            speed_data.append({
                                'dim': int(data['dim']),
                                'param_count': int(data['param_count']),
                                'saturation_step': int(data['saturation_step']),
                                'n_samples': int(data['n_samples'])
                            })
                    print(f"Loaded {len(speed_data)} speed experiments from {fallback_dir} (fallback seed)")
                else:
                    print(f"Warning: No speed data found for p={args.p} with any seed for size {n}")
            else:
                print(f"Warning: Speed data directory not found at {speed_base_dir}")
        
        save_path = os.path.join(plot_dir, f'delay_with_speed_p{args.p}.pdf') if args.save else None
        plot_grokking_delay_with_speed(
            results,
            speed_data,
            threshold_train=args.threshold_train,
            threshold_val=args.threshold_val,
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
    symb_map = {
        'random': 'random',
        '+': 'add',
        '-': 'sub',
        '*': 'mul',
        '/': 'div'
    }
    signature = f'p{args.p}_seed{args.training_seed}_ds{symb_map[args.dataset_type]}'
    data_dir = os.path.join(args.data_dir, signature)
    plot_dir = os.path.join(args.plot_dir, signature)
    
    print(f'Data directory: {data_dir}')
    print(f'Plot directory: {plot_dir}')
    
    # List results
    if args.list:
        # Use default pattern when none is provided
        pattern = args.pattern or 'capacity_dim*.npz'
        list_capacity_results(data_dir, pattern)
        return
    
    # Plot single result
    if args.plot:
        plot_capacity_result(args.plot)
        return
    
    # Collect files based on selection criteria
    files = set()
    
    if args.files:
        files.update(args.files)
    
    if args.all:
        pattern = os.path.join(data_dir, 'capacity_dim*.npz')
        files.update(glob(pattern))
    
    if args.pattern:
        pattern = os.path.join(data_dir, args.pattern)
        files.update(glob(pattern))
    
    if args.dims:
        for d in args.dims:
            pattern = os.path.join(data_dir, f'capacity_dim{d}_samples*.npz')
            files.update(glob(pattern))
    
    if args.samples:
        for s in args.samples:
            pattern = os.path.join(data_dir, f'capacity_dim*_samples{s}.npz')
            files.update(glob(pattern))
    
    files = sorted(files, key=lambda f: (extract_dim(f), extract_samples(f)))
    
    if not files:
        print("No files found. Use --list to see available results, or --all to select all.")
        return
    
    print(f"\nFound {len(files)} files to analyze:")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    
    # Load all results
    results = []
    for fname in files:
        if not os.path.exists(fname):
            print(f"Warning: File not found: {fname}")
            continue
        data = np.load(fname)
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
            all_results, p=args.p, save_path=save_path, show=not args.no_show
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
# Cross-Experiment Visualization Functions
# =============================================================================

def parse_signature(signature: str) -> dict:
    """
    Parse a signature like 'p137_seed42_splitrandom' to extract parameters.
    
    Returns dict with keys: 'p', 'seed', 'split_type'
    """
    import re
    match = re.match(r'p(\d+)_seed(\d+)_split(\w+)', signature)
    if not match:
        raise ValueError(f"Invalid signature format: {signature}")
    return {
        'p': int(match.group(1)),
        'seed': int(match.group(2)),
        'split_type': match.group(3)
    }


def cross_exp(args):
    """Handle cross-exp subcommand."""
    if not args.sigs:
        print("Error: --sigs is required for cross-exp command")
        return
    
    if not args.critical:
        print("Error: --critical is required for cross-exp command (only --critical mode is supported)")
        return
    
    cross_exp_data = []
    
    for sig in args.sigs:
        print(f"\nProcessing signature: {sig}")
        
        # Parse signature to get prime p
        try:
            sig_params = parse_signature(sig)
        except ValueError as e:
            print(f"  Warning: {e}")
            continue
        
        p = sig_params['p']
        
        # Compute dataset size in bits
        _, dataset_bits = compute_dataset_size_bits(p, op=args.op, training_fraction=args.training_fraction)
        print(f"  p={p}, dataset size = {dataset_bits:,.0f} bits")
        
        # Find and load grokking results for this signature
        data_dir = os.path.join(args.data_dir, sig)
        if not os.path.exists(data_dir):
            print(f"  Warning: Data directory not found: {data_dir}")
            continue
        
        # Load all grokking files matching depth/heads
        pattern = os.path.join(data_dir, f'grokking_dim*_depth{args.depth}_heads{args.heads}.npz')
        files = sorted(glob(pattern), key=extract_dim)
        
        if not files:
            print(f"  Warning: No files found matching pattern: {pattern}")
            continue
        
        print(f"  Found {len(files)} result files")
        
        # Load results
        results = []
        for fname in files:
            data = np.load(fname)
            results.append({
                'train_acc': data['train_acc'],
                'val_acc': data['val_acc'],
                'dim': int(data['dim']),
                'param_count': int(data['param_count'])
            })

        # Compute critical parameter count (line-fitting method)
        critical_params = compute_critical_params(
            results,
            threshold_train=args.threshold_train,
            threshold_val=args.threshold_val,
            delay_threshold=args.delay_threshold
        )

        if critical_params is None:
            print(f"  Warning: Could not compute critical params for {sig}")
            continue

        model_capacity = consts.C * critical_params
        print(f"  Critical params (line-fitting): {critical_params:,.0f}")
        print(f"  Model capacity (line-fitting): {model_capacity:,.0f} bits (C={consts.C})")

        # Load speed data for intersection-based critical point
        speed_data = []
        speed_signature = f'p{p}_seed{sig_params["seed"]}'
        speed_dir = os.path.join('data/speed', speed_signature)

        # First try to load with matching seed
        if os.path.exists(speed_dir):
            speed_files = glob(os.path.join(speed_dir, 'speed_dim*.npz'))
            if speed_files:
                for sf in speed_files:
                    data = np.load(sf)
                    speed_data.append({
                        'dim': int(data['dim']),
                        'param_count': int(data['param_count']),
                        'saturation_step': int(data['saturation_step']),
                        'n_samples': int(data['n_samples'])
                    })

        # If no matching seed found, search for any available seed with same p
        if not speed_data:
            speed_base_dir = os.path.join('data/speed')
            if os.path.exists(speed_base_dir):
                # Find all directories matching p{p}_seed*
                pattern = re.compile(rf'^p{p}_seed\d+$')
                available_dirs = [d for d in os.listdir(speed_base_dir)
                                 if pattern.match(d) and os.path.isdir(os.path.join(speed_base_dir, d))]

                if available_dirs:
                    # Sort to get consistent behavior (prefer lower seed numbers as fallback)
                    available_dirs = sorted(available_dirs, key=lambda x: int(re.search(r'seed(\d+)', x).group(1)))
                    fallback_dir = os.path.join(speed_base_dir, available_dirs[0])

                    speed_files = glob(os.path.join(fallback_dir, 'speed_dim*.npz'))
                    for sf in speed_files:
                        data = np.load(sf)
                        speed_data.append({
                            'dim': int(data['dim']),
                            'param_count': int(data['param_count']),
                            'saturation_step': int(data['saturation_step']),
                            'n_samples': int(data['n_samples'])
                        })

        # Compute critical parameter count using intersection method if speed data available
        critical_params_speed = None
        if speed_data:
            critical_params_speed = compute_critical_params_from_speed(
                results,
                speed_data,
                threshold_train=args.threshold_train,
                threshold_val=args.threshold_val
            )

            if critical_params_speed is not None:
                model_capacity_speed = consts.C * critical_params_speed
                print(f"  Critical params (speed-based): {critical_params_speed:,.0f}")
                print(f"  Model capacity (speed-based): {model_capacity_speed:,.0f} bits")
            else:
                print(f"  Warning: Could not compute critical params via speed for {sig}")

        cross_exp_data.append({
            'signature': sig,
            'p': p,
            'critical_params': critical_params,
            'critical_params_speed': critical_params_speed,
            'model_capacity': model_capacity,
            'dataset_bits': dataset_bits
        })
    
    if not cross_exp_data:
        print("\nNo valid data points to plot")
        return
    
    # Print summary
    print("\n" + "="*80)
    print("CROSS-EXPERIMENT SUMMARY")
    print("="*80)
    print(f"{'Signature':<30} {'p':<6} {'Critical Params':<16} {'Model Cap (bits)':<18} {'Dataset (bits)':<16}")
    print("-"*80)
    for d in cross_exp_data:
        print(f"{d['signature']:<30} {d['p']:<6} {d['critical_params']:<16,.0f} {d['model_capacity']:<18,.0f} {d['dataset_bits']:<16,.0f}")
    print("="*80)
    
    # Create plot
    show = not args.no_show
    save_path = None
    if args.save:
        os.makedirs(args.plot_dir, exist_ok=True)
        save_path = os.path.join(args.plot_dir, 'cross_exp_critical.pdf')
    
    plot_cross_exp_critical(cross_exp_data, save_path=save_path, show=show)


# =============================================================================
# Speed Experiment Visualization Functions
# =============================================================================

def list_speed_results(data_dir: str, pattern: str = 'speed_dim*.npz') -> List[Dict]:
    """List all saved speed experiment results."""
    files = sorted(glob(os.path.join(data_dir, pattern)))

    if not files:
        print(f"No results found matching pattern: {pattern}")
        return []

    print(f"\nFound {len(files)} speed result files:")
    print("="*80)

    results = []
    for i, fname in enumerate(files):
        data = np.load(fname)
        dim = int(data['dim'])
        n_samples = int(data['n_samples'])
        param_count = int(data['param_count'])
        depth = int(data['depth'])
        heads = int(data['heads'])
        saturation_step = int(data['saturation_step'])
        final_acc = float(data['final_acc'])

        results.append({
            'file': fname,
            'dim': dim,
            'n_samples': n_samples,
            'param_count': param_count,
            'depth': depth,
            'heads': heads,
            'saturation_step': saturation_step,
            'final_acc': final_acc,
            'data': data
        })

        print(f"{i:2d}. {os.path.basename(fname)}")
        print(f"    dim={dim:3d}, samples={n_samples:6d}, params={param_count:8,}")
        print(f"    Saturation step={saturation_step:6d}, Acc={final_acc:.1f}%")

    print("="*80)
    return results


def speed(args):
    """Handle speed subcommand."""
    signature = f'p{args.p}_seed{args.seed}'
    data_dir = os.path.join(args.data_dir, signature)
    plot_dir = os.path.join(args.plot_dir, signature)

    print(f'Data directory: {data_dir}')
    print(f'Plot directory: {plot_dir}')

    # List results
    if args.list:
        # Use default pattern when none is provided
        pattern = args.pattern or 'speed_dim*.npz'
        list_speed_results(data_dir, pattern)
        return

    # Collect files based on selection criteria
    files = set()

    if args.files:
        files.update(args.files)

    if args.all:
        pattern = os.path.join(data_dir, 'speed_dim*.npz')
        files.update(glob(pattern))

    if args.pattern:
        pattern = os.path.join(data_dir, args.pattern)
        files.update(glob(pattern))

    if args.dims:
        for d in args.dims:
            pattern = os.path.join(data_dir, f'speed_dim{d}_samples*.npz')
            files.update(glob(pattern))

    files = sorted(files, key=lambda f: (extract_dim(f), extract_samples(f)))

    if not files:
        print("No files found. Use --list to see available results, or --all to select all.")
        return

    print(f"\nFound {len(files)} files to analyze:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    # Load all results and group by dimension
    all_results = {}
    for fname in files:
        if not os.path.exists(fname):
            print(f"Warning: File not found: {fname}")
            continue
        data = np.load(fname)

        dim = int(data['dim'])
        result = {
            'n_samples': int(data['n_samples']),
            'dim': dim,
            'depth': int(data['depth']),
            'heads': int(data['heads']),
            'param_count': int(data['param_count']),
            'p': int(data['p']),
            'saturation_step': int(data['saturation_step']),
            'final_acc': float(data['final_acc']),
            'dataset_bits': float(data['dataset_bits']),
            'saturated': bool(data.get('saturated', True))
        }

        if dim not in all_results:
            all_results[dim] = []
        all_results[dim].append(result)

    # Generate requested plots
    os.makedirs(plot_dir, exist_ok=True)

    show = not args.no_show

    if args.curves:
        print("\nPlotting learning speed curves...")
        save_path = os.path.join(plot_dir, 'learning_speed_curves.pdf') if args.save else None
        speed_estimates = plot_learning_speed_curves(
            all_results,
            p=args.p,
            save_path=save_path,
            show=show
        )

        # Plot steps to saturation vs model parameters
        print("\nPlotting steps to saturation vs model parameters...")
        save_path = os.path.join(plot_dir, 'saturation_steps_vs_params.pdf') if args.save else None
        plot_saturation_steps_vs_params(
            all_results,
            save_path=save_path,
            show=show
        )

        # Also plot speed vs model size if we have multiple model sizes
        if len(speed_estimates) >= 2:
            print("\nPlotting speed vs model size...")
            save_path = os.path.join(plot_dir, 'speed_vs_model_size.pdf') if args.save else None
            b, log_a, r_squared = plot_speed_vs_model_size(
                speed_estimates,
                save_path=save_path,
                show=show
            )

            print("\n" + "="*60)
            print("SPEED SCALING")
            print("="*60)
            print(f"Power law exponent: {b:.3f}")
            print(f"R²: {r_squared:.3f}")
            if b < 0:
                print(f"Larger models learn FASTER (fewer steps per bit)")
            else:
                print(f"Larger models learn SLOWER (more steps per bit)")

    if args.combined:
        print("\nPlotting combined speed analysis...")
        save_path = os.path.join(plot_dir, 'speed_analysis_combined.pdf') if args.save else None
        speed_estimates = plot_combined_speed_analysis(
            all_results,
            p=args.p,
            save_path=save_path,
            show=show
        )

    if args.fraction:
        print("\nPlotting saturation time vs capacity fraction...")
        save_path = os.path.join(plot_dir, 'saturation_time_vs_capacity_fraction.pdf') if args.save else None
        exponent, coefficient, r_squared = plot_saturation_time_vs_capacity_fraction(
            all_results,
            C=consts.C,
            save_path=save_path,
            show=show
        )

        print("\n" + "="*60)
        print("CAPACITY FRACTION ANALYSIS")
        print("="*60)
        print(f"Power law fit: steps = {coefficient:.1f} × f^{exponent:.2f}")
        print(f"Exponent: {exponent:.2f}")
        print(f"R²: {r_squared:.3f}")
        print(f"Capacity constant C = {consts.C:.2f} bits/param")
        print("="*60)

    if args.rate:
        print(f"\nPlotting dT/dS vs dataset size (k={args.rate_k})...")
        save_path = os.path.join(plot_dir, f'rate_vs_dataset_size_k{args.rate_k}.pdf') if args.save else None
        rate_data = plot_rate_vs_dataset_size(
            all_results,
            k=args.rate_k,
            save_path=save_path,
            show=show
        )

        if rate_data:
            print("\n" + "="*60)
            print(f"RATE ESTIMATION (k={args.rate_k} samples)")
            print("="*60)
            for dim in sorted(rate_data.keys()):
                rates = rate_data[dim]
                if rates:
                    avg_rate = np.mean([r[1] for r in rates])
                    print(f"dim={dim:3d}: avg dT/dS = {avg_rate:.2f} steps/bit")
            print("="*60)
        else:
            print("No paired data points found. Run speed.py with --rate to generate paired data.")

    if args.summary:
        print("\n" + "="*70)
        print("SPEED SUMMARY")
        print("="*70)

        for dim in sorted(all_results.keys()):
            results = all_results[dim]
            param_count = results[0]['param_count']

            # Average steps per bit across dataset sizes
            saturated = [r for r in results if r.get('saturated', True)]
            if saturated:
                avg_speed = np.mean([r['saturation_step'] / r['dataset_bits']
                                    for r in saturated if r['dataset_bits'] > 0])
                print(f"dim={dim:3d}: {param_count:8,} params, "
                      f"avg speed: {avg_speed:.2f} steps/bit")

        print("="*70)


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

    grok_subparser.add_argument('--training-seed', type=int, default=42, help='training seed')
    grok_subparser.add_argument('--training-fraction', type=float, default=0.5, help='training fraction')
    grok_subparser.add_argument('--op', type=str, default='/', help='operation', choices=['*', '/', '+', '-'])
    grok_subparser.add_argument('--split-type', type=str, default='random', help='split type', choices=['random', 'sequential', 'alternating'])
    grok_subparser.add_argument('--p', type=int, default=97, help='prime number')
    grok_subparser.add_argument('--data-dir', type=str, default='data/groks', help='data directory')
    grok_subparser.add_argument('--plot-dir', type=str, default='media/groks', help='plot directory')

    # Actions to perform
    grok_subparser.add_argument('--list', action='store_true', 
                       help='List all available results')
    grok_subparser.add_argument('--plot', type=str,
                       help='Plot a specific result file')
    grok_subparser.add_argument('--files', nargs='+',
                       help='Compare multiple result files')
    grok_subparser.add_argument('--pattern', type=str,
                       help='Pattern to match result files')
    grok_subparser.add_argument('--dims', nargs='+', type=int,
                       help='Compare specific dimensions (e.g., --dims 20 40 60)')
    grok_subparser.add_argument('--dims-start', type=int, default=None,
                       help='Starting dimension for dimension comparison')
    grok_subparser.add_argument('--dims-end', type=int, default=None,
                       help='Ending dimension for dimension comparison')
    grok_subparser.add_argument('--dims-step', type=int, default=None,
                       help='Step size for dimension comparison')
    grok_subparser.add_argument('--all', action='store_true',
                       help='Compare all available results (combined plot)')
    grok_subparser.add_argument('--combined', action='store_true',
                       help='Compare all available results (combined plot)')
    grok_subparser.add_argument('--separate', action='store_true',
                       help='Plot training and validation in separate subplots')
    grok_subparser.add_argument('--delay', action='store_true',
                       help='Plot grokking delay vs parameter count')
    grok_subparser.add_argument('--critical', action='store_true',
                       help='Find and plot critical model capacity where grokking delay first becomes positive (extrapolated from linear fit)')
    grok_subparser.add_argument('--time', action='store_true',
                       help='Plot absolute grokking time (epochs to reach threshold) vs parameter count')
    grok_subparser.add_argument('--speed', action='store_true',
                       help='Plot grokking delay with overlaid steps axis, showing steps to grok and steps to learn (from speed data)')
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
                       help='Minimum delay to include in critical capacity fit (default: 1.0)')
    grok_subparser.add_argument('--depth', type=int, default=2,
                       help='Depth for dimension comparison')
    grok_subparser.add_argument('--heads', type=int, default=1,
                       help='Heads for dimension comparison')
    
    # Output options
    grok_subparser.add_argument('--save', action='store_true',
                       help='Save plots to plot-dir')
    grok_subparser.add_argument('--no-show', action='store_true',
                       help='Do not display plots (only save)')
    
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
    
    # Directory configuration
    cap_subparser.add_argument('--data-dir', type=str, default='data/capacity',
                               help='Data directory (default: data/capacity)')
    cap_subparser.add_argument('--plot-dir', type=str, default='media/capacity',
                               help='Plot output directory (default: media/capacity)')
    cap_subparser.add_argument('--p', type=int, default=97,
                               help='Prime number (default: 97)')
    cap_subparser.add_argument('--training-seed', type=int, default=42,
                               help='Training seed (default: 42)')
    cap_subparser.add_argument('--dataset-type', type=str, default='random',
                               help='Dataset type (default: random)',
                               choices=['random', '+', '-', '*', '/'])
    
    # File selection
    cap_subparser.add_argument('--list', action='store_true',
                               help='List all available capacity results')
    cap_subparser.add_argument('--plot', type=str, metavar='FILE',
                               help='Plot training curves for a specific result file')
    cap_subparser.add_argument('--files', nargs='+', metavar='FILE',
                               help='Analyze specific result files')
    cap_subparser.add_argument('--pattern', type=str,
                               help='Glob pattern to match result files')
    cap_subparser.add_argument('--all', action='store_true',
                               help='Select all available results')
    cap_subparser.add_argument('--dims', nargs='+', type=int, metavar='DIM',
                               help='Filter by model dimensions (e.g., --dims 20 40 80)')
    cap_subparser.add_argument('--samples', nargs='+', type=int, metavar='N',
                               help='Filter by dataset sizes (e.g., --samples 100 1000)')
    
    # Plot types
    cap_subparser.add_argument('--curves', action='store_true',
                               help='Plot memorisation vs dataset size curves (Morris et al. style)')
    cap_subparser.add_argument('--accuracy', action='store_true',
                               help='Plot bits memorized vs training accuracy')
    cap_subparser.add_argument('--summary', action='store_true',
                               help='Print summary statistics and capacity estimate')
    
    # Output options
    cap_subparser.add_argument('--save', action='store_true',
                               help='Save plots to plot-dir')
    cap_subparser.add_argument('--no-show', action='store_true',
                               help='Do not display plots (only save)')

    # =========================================================================
    # Cross-Experiment subparser
    # =========================================================================
    cross_exp_subparser = subparsers.add_parser(
        'cross-exp',
        help='Cross-experiment analysis (compare across multiple signatures)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot model capacity vs dataset size for multiple experiments
  python visualise.py cross-exp --sigs p97_seed42_splitrandom p127_seed42_splitrandom p137_seed42_splitrandom --critical
  
  # Save the plot
  python visualise.py cross-exp --sigs p97_seed42_splitrandom p127_seed42_splitrandom --critical --save
"""
    )
    cross_exp_subparser.set_defaults(func=cross_exp)
    
    # Signature selection
    cross_exp_subparser.add_argument('--sigs', nargs='+', metavar='SIG', required=True,
                                     help='List of experiment signatures (e.g., p137_seed42_splitrandom)')
    
    # Analysis type
    cross_exp_subparser.add_argument('--critical', action='store_true',
                                     help='Plot model capacity (C × critical params) vs dataset size (bits)')
    
    # Experiment parameters
    cross_exp_subparser.add_argument('--op', type=str, default='/',
                                     help='Operation (default: /)', choices=['*', '/', '+', '-'])
    cross_exp_subparser.add_argument('--training-fraction', type=float, default=0.5,
                                     help='Training fraction (default: 0.5)')
    cross_exp_subparser.add_argument('--depth', type=int, default=2,
                                     help='Model depth (default: 2)')
    cross_exp_subparser.add_argument('--heads', type=int, default=1,
                                     help='Number of attention heads (default: 1)')
    
    # Threshold parameters
    cross_exp_subparser.add_argument('--threshold-train', type=float, default=99.0,
                                     help='Accuracy threshold for training (default: 99.0)')
    cross_exp_subparser.add_argument('--threshold-val', type=float, default=97.0,
                                     help='Accuracy threshold for validation (default: 97.0)')
    cross_exp_subparser.add_argument('--delay-threshold', type=float, default=0.5,
                                     help='Minimum delay to include in critical capacity fit (default: 1.0)')
    
    # Directory configuration
    cross_exp_subparser.add_argument('--data-dir', type=str, default='data/groks',
                                     help='Data directory (default: data/groks)')
    cross_exp_subparser.add_argument('--plot-dir', type=str, default='media/cross_exp',
                                     help='Plot output directory (default: media/cross_exp)')
    
    # Output options
    cross_exp_subparser.add_argument('--save', action='store_true',
                                     help='Save plots to plot-dir')
    cross_exp_subparser.add_argument('--no-show', action='store_true',
                                     help='Do not display plots (only save)')

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

    # Directory configuration
    speed_subparser.add_argument('--data-dir', type=str, default='data/speed',
                                 help='Data directory (default: data/speed)')
    speed_subparser.add_argument('--plot-dir', type=str, default='media/speed',
                                 help='Plot output directory (default: media/speed)')
    speed_subparser.add_argument('--p', type=int, default=97,
                                 help='Prime number (default: 97)')
    speed_subparser.add_argument('--seed', type=int, default=42,
                                 help='Random seed (default: 42)')

    # File selection
    speed_subparser.add_argument('--list', action='store_true',
                                 help='List all available speed results')
    speed_subparser.add_argument('--files', nargs='+', metavar='FILE',
                                 help='Analyze specific result files')
    speed_subparser.add_argument('--pattern', type=str,
                                 help='Glob pattern to match result files')
    speed_subparser.add_argument('--all', action='store_true',
                                 help='Select all available results')
    speed_subparser.add_argument('--dims', nargs='+', type=int, metavar='DIM',
                                 help='Filter by model dimensions (e.g., --dims 20 28 32)')

    # Plot types
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
    speed_subparser.add_argument('--summary', action='store_true',
                                 help='Print summary statistics')

    # Output options
    speed_subparser.add_argument('--save', action='store_true',
                                 help='Save plots to plot-dir')
    speed_subparser.add_argument('--no-show', action='store_true',
                                 help='Do not display plots (only save)')

    # Run the appropriate function based on the subparser
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
