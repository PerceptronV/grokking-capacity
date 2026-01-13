"""
Shared plotting utilities for grokking and capacity experiments.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple, Optional


def calculate_grokking_delay(train_acc, val_acc, threshold_train=99.0, threshold_val=99.0):
    """
    Calculate grokking delay: difference in epochs between train and val reaching threshold_train and threshold_val respectively.
    Returns (train_epoch, val_epoch, delay) or (None, None, None) if either doesn't reach threshold.
    """
    train_epoch = None
    val_epoch = None
    
    # Find when training reaches threshold
    for epoch, acc in enumerate(train_acc):
        if acc >= threshold_train:
            train_epoch = epoch
            break
    
    # Find when validation reaches threshold
    for epoch, acc in enumerate(val_acc):
        if acc >= threshold_val:
            val_epoch = epoch
            break
    
    if train_epoch is not None and val_epoch is not None:
        delay = val_epoch - train_epoch
        return train_epoch, val_epoch, max(0, delay)
    if train_epoch is None and val_epoch is not None:
        return None, val_epoch, 0
    else:
        return None, None, None


def plot_combined_curves(results, title='Training (solid) vs Validation (dashed) Curves', 
                        save_path=None, show=True):
    """
    Plot combined training and validation curves for multiple experiments.
    
    Args:
        results: List of dicts with keys 'train_acc', 'val_acc', 'dim', 'param_count'
        title: Plot title
        save_path: Path to save the plot (optional)
        show: Whether to show the plot
    """
    if not results:
        print("No results to plot")
        return
    
    # Sort by dimension
    results_sorted = sorted(results, key=lambda x: x['dim'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Use crest colormap for both training and validation
    crest_cmap = sns.color_palette('crest', as_cmap=True)
    colors = crest_cmap(np.linspace(0, 1, len(results_sorted)))
    
    for i, result in enumerate(results_sorted):
        train_acc = result['train_acc']
        val_acc = result['val_acc']
        dim = result['dim']
        
        # Training: solid line
        ax.plot(train_acc, color=colors[i], linewidth=2, linestyle='-', alpha=0.7)
        # Validation (grokking): dashed line
        ax.plot(val_acc, color=colors[i], linewidth=2, linestyle='--', alpha=0.7)
        
        # Add label at the end of the grokking (validation) curve
        y_pos = val_acc[-1] + 1
        ax.text(len(val_acc) + 2, y_pos, f'{dim}', 
                color=colors[i], fontsize=8, va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.5))
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_xscale('log')
    ax.set_title(title, fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylim([0, 105])
    
    # Add colorbar showing dimension gradient
    dims = [r['dim'] for r in results_sorted]
    crest_cmap = sns.color_palette('crest', as_cmap=True)
    sm = plt.cm.ScalarMappable(cmap=crest_cmap, 
                               norm=plt.Normalize(vmin=min(dims), vmax=max(dims)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Dimension', pad=0.02)
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Combined plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_separate_curves(results, title_val='Validation Accuracy Curves', 
                         title_train='Training Accuracy Curves',
                         save_path=None, show=True):
    """
    Plot training and validation curves in separate side-by-side subplots.
    
    Args:
        results: List of dicts with keys 'train_acc', 'val_acc', 'dim', 'param_count'
        title_val: Title for validation subplot
        title_train: Title for training subplot
        save_path: Path to save the plot (optional)
        show: Whether to show the plot
    """
    if not results:
        print("No results to plot")
        return
    
    # Sort by dimension
    results_sorted = sorted(results, key=lambda x: x['dim'])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Use crest colormap for different dimensions
    crest_cmap = sns.color_palette('crest', as_cmap=True)
    colors = crest_cmap(np.linspace(0, 1, len(results_sorted)))
    
    dims = [r['dim'] for r in results_sorted]
 
    # Plot training accuracy for comparison
    for i, result in enumerate(results_sorted):
        train_acc = result['train_acc']
        dim = result['dim']
        ax1.plot(train_acc, color=colors[i], linewidth=2, alpha=0.8)
        
        # Add label at the end of training curve
        y_pos = train_acc[-1] + 1
        ax1.text(len(train_acc) + 2, y_pos, f'{dim}', 
                color=colors[i], fontsize=8, va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.5))
    
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Training Accuracy (%)', fontsize=14)
    ax1.set_xscale('log')
    ax1.set_title(title_train, fontsize=16, pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.set_ylim([0, 105])
    
    # Plot validation accuracy (main grokking plot)
    for i, result in enumerate(results_sorted):
        val_acc = result['val_acc']
        dim = result['dim']
        ax2.plot(val_acc, color=colors[i], linewidth=2, alpha=0.8)
        
        # Add label at the end of validation curve
        y_pos = val_acc[-1] + 1
        ax2.text(len(val_acc) + 2, y_pos, f'{dim}', 
                color=colors[i], fontsize=8, va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.5))
    
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Validation Accuracy (%)', fontsize=14)
    ax2.set_xscale('log')
    ax2.set_title(title_val, fontsize=16, pad=20)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.set_ylim([0, 105])
    
    # Adjust layout first to get proper spacing
    plt.tight_layout()
    
    # Add a shared colorbar on the right
    # Create a mappable for the colorbar
    crest_cmap = sns.color_palette('crest', as_cmap=True)
    sm = plt.cm.ScalarMappable(cmap=crest_cmap, 
                               norm=plt.Normalize(vmin=min(dims), vmax=max(dims)))
    sm.set_array([])
    
    # Make room for colorbar and add it
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, label='Dimension')
    cbar.ax.tick_params(labelsize=12)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Separate plots saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_separate_curves_with_memorization(
    results, 
    mem_key: str = 'mem_u_trace',
    title_val='Validation Accuracy with Memorisation', 
    title_train='Training Accuracy with Memorisation',
    save_path=None, 
    show=True
):
    """
    Plot training and validation curves in separate side-by-side subplots with memorisation overlaid.
    
    Args:
        results: List of dicts with keys 'train_acc', 'val_acc', and mem_key, 'dim', 'param_count'
        mem_key: Key to access memorisation trace ('mem_t_trace', 'mem_u_trace', or legacy 'mem_trace')
        title_val: Title for validation subplot
        title_train: Title for training subplot
        save_path: Path to save the plot (optional)
        show: Whether to show the plot
    """
    if not results:
        print("No results to plot")
        return
    
    # Filter results that have memorisation traces
    results_with_mem = [r for r in results if mem_key in r and r[mem_key] is not None]
    # Legacy support
    if not results_with_mem and mem_key == 'mem_u_trace':
        results_with_mem = [r for r in results if 'mem_trace' in r and r['mem_trace'] is not None]
        mem_key = 'mem_trace'
    
    if not results_with_mem:
        print(f"No results with memorisation data found (key: {mem_key}), falling back to standard separate curves")
        return plot_separate_curves(results, title_val=title_val, title_train=title_train, 
                                    save_path=save_path, show=show)
    
    # Determine label based on mem_key
    if 'mem_t' in mem_key:
        mem_label = 'M_T'
    elif 'mem_u' in mem_key:
        mem_label = 'M_U'
    else:
        mem_label = 'Memorisation'
    
    # Sort by dimension
    results_sorted = sorted(results_with_mem, key=lambda x: x['dim'])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
    
    # Create secondary axes for memorisation
    ax1_mem = ax1.twinx()
    ax2_mem = ax2.twinx()
    
    # Use crest colormap for different dimensions
    crest_cmap = sns.color_palette('crest', as_cmap=True)
    colors = crest_cmap(np.linspace(0, 1, len(results_sorted)))
    
    dims = [r['dim'] for r in results_sorted]
    
    # Find global memorisation range for consistent scaling
    all_mem = np.concatenate([r[mem_key] for r in results_sorted])
    mem_min, mem_max = all_mem.min(), all_mem.max()
    mem_range = mem_max - mem_min
    mem_padding = mem_range * 0.05
    
    if mem_min >= 0:
        mem_ylim = [0, mem_max * 1.05]
    else:
        mem_ylim = [mem_min - mem_padding, mem_max + mem_padding]
    
    # Plot training accuracy and memorisation
    for i, result in enumerate(results_sorted):
        train_acc = result['train_acc']
        mem_trace = result[mem_key]
        dim = result['dim']
        
        # Plot training accuracy (solid line)
        ax1.plot(train_acc, color=colors[i], linewidth=2, alpha=0.7, linestyle='-')
        
        # Plot memorisation (dashed line, slightly thinner)
        ax1_mem.plot(mem_trace, color=colors[i], linewidth=1.5, alpha=0.5, linestyle='--')
        
        # Add label at the end of training curve
        y_pos = train_acc[-1] + 1
        ax1.text(len(train_acc) + 2, y_pos, f'{dim}', 
                color=colors[i], fontsize=8, va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.5))
    
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Training Accuracy (%)', fontsize=14, color='black')
    ax1.set_xscale('log')
    ax1.set_title(title_train, fontsize=16, pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='black', labelsize=12)
    ax1.set_ylim([0, 105])
    
    ax1_mem.set_ylabel(f'{mem_label} (bits)', fontsize=14, color='#7570b3')
    ax1_mem.tick_params(axis='y', labelcolor='#7570b3', labelsize=12)
    ax1_mem.set_ylim(mem_ylim)
    ax1_mem.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    # Plot validation accuracy and memorisation
    for i, result in enumerate(results_sorted):
        val_acc = result['val_acc']
        mem_trace = result[mem_key]
        dim = result['dim']
        
        # Plot validation accuracy (solid line)
        ax2.plot(val_acc, color=colors[i], linewidth=2, alpha=0.7, linestyle='-')
        
        # Plot memorisation (dashed line, slightly thinner)
        ax2_mem.plot(mem_trace, color=colors[i], linewidth=1.5, alpha=0.5, linestyle='--')
        
        # Add label at the end of validation curve
        y_pos = val_acc[-1] + 1
        ax2.text(len(val_acc) + 2, y_pos, f'{dim}', 
                color=colors[i], fontsize=8, va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.5))
    
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Validation Accuracy (%)', fontsize=14, color='black')
    ax2.set_xscale('log')
    ax2.set_title(title_val, fontsize=16, pad=20)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor='black', labelsize=12)
    ax2.set_ylim([0, 105])
    
    ax2_mem.set_ylabel(f'{mem_label} (bits)', fontsize=14, color='#7570b3')
    ax2_mem.tick_params(axis='y', labelcolor='#7570b3', labelsize=12)
    ax2_mem.set_ylim(mem_ylim)
    ax2_mem.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    # Add legend to explain line styles
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linewidth=2, linestyle='-', label='Accuracy'),
        Line2D([0], [0], color='gray', linewidth=1.5, linestyle='--', alpha=0.5, label='M_U (bits)')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=11)
    
    # Adjust layout first to get proper spacing
    plt.tight_layout()
    
    # Add a shared colorbar on the right
    crest_cmap = sns.color_palette('crest', as_cmap=True)
    sm = plt.cm.ScalarMappable(cmap=crest_cmap, 
                               norm=plt.Normalize(vmin=min(dims), vmax=max(dims)))
    sm.set_array([])
    
    # Make room for colorbar and add it
    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, label='Dimension')
    cbar.ax.tick_params(labelsize=12)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Separate plots with memorisation saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_grokking_delay(results, threshold_train=99.0, threshold_val=99.0, threshold_params=None, title=None, save_path=None, show=True):
    """
    Plot grokking delay vs parameter count.
    
    Args:
        results: List of dicts with keys 'train_acc', 'val_acc', 'dim', 'param_count'
        threshold_train: Accuracy threshold for training
        threshold_val: Accuracy threshold for validation
        threshold_params: Threshold for parameter count
        title: Plot title (auto-generated if None)
        save_path: Path to save the plot (optional)
        show: Whether to show the plot
    """
    if not results:
        print("No results to plot")
        return
    
    # Calculate delays for all results
    delay_data = []
    for result in results:
        train_acc = result['train_acc']
        val_acc = result['val_acc']
        dim = result['dim']
        param_count = result['param_count']
        
        train_epoch, val_epoch, delay = calculate_grokking_delay(train_acc, val_acc, threshold_train, threshold_val)
        
        if delay is not None:
            delay_data.append({
                'dim': dim,
                'param_count': param_count,
                'train_epoch': train_epoch,
                'val_epoch': val_epoch,
                'delay': delay
            })
        else:
            print(f"Warning: dim={dim} did not reach val={threshold_val}% accuracy")
    
    if not delay_data:
        print(f"No results reached val={threshold_val}% accuracy threshold")
        return
    
    # Sort by parameter count
    delay_data.sort(key=lambda x: x['param_count'])
    
    # Extract data for plotting
    dims = [item['dim'] for item in delay_data]
    param_counts = [item['param_count'] for item in delay_data]
    delays = [item['delay'] for item in delay_data]
    
    # Remove anomalies: points with positive delay but both neighbors have non-positive delay
    delays_arr = np.array(delays)
    is_grokking = delays_arr > 0
    anomaly_mask = np.ones(len(delays), dtype=bool)  # Start by keeping all points
    
    for i in range(1, len(delays) - 1):
        if is_grokking[i] and not is_grokking[i-1] and not is_grokking[i+1]:
            anomaly_mask[i] = False
            print(f"Filtering out anomaly: dim={dims[i]}, delay={delays[i]:.1f} (neighbors not grokking)")
    
    # Apply anomaly filter
    dims = [d for d, keep in zip(dims, anomaly_mask) if keep]
    param_counts = [pc for pc, keep in zip(param_counts, anomaly_mask) if keep]
    delays = [d for d, keep in zip(delays, anomaly_mask) if keep]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Scatter plot with color-coded dimensions
    crest_cmap = sns.color_palette('crest', as_cmap=True)
    scatter = ax.scatter(param_counts, delays, c=dims, cmap=crest_cmap, 
                        s=80, alpha=0.7, edgecolors='none')
    
    # Add labels for each point
    for i, (pc, delay, dim) in enumerate(zip(param_counts, delays, dims)):
        ax.annotate(f'{dim}', (pc, delay), xytext=(5, 5), 
                   textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Parameter Count', fontsize=14)
    ax.set_ylabel(f'Grokking Delay (epochs)', fontsize=14)
    ax.set_xscale('log')
    
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Dimension')
    cbar.ax.tick_params(labelsize=12)
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Add dotted vertical line at threshold_params, if provided
    if threshold_params is not None:
        ax.axvline(x=threshold_params, color='black', linestyle=':', linewidth=2, alpha=0.7)
        # Annotate its value (with thousands separator)
        # Place annotation slightly above the largest delay
        y_annot = max(delays) if len(delays) > 0 else 0
        ax.annotate(f"{int(threshold_params):,}", 
                    xy=(threshold_params, y_annot), 
                    xycoords='data',
                    xytext=(10, -25), textcoords='offset points',  # moved down by changing y offset from 0 to -25
                    fontsize=16, color='black',
                    va='bottom', ha='left')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Delay plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"GROKKING DELAY SUMMARY (train={threshold_train}%, val={threshold_val}%)")
    print(f"{'='*80}")
    for item in delay_data:
        train_str = f"{item['train_epoch']:3d}" if item['train_epoch'] is not None else "N/A"
        val_str = f"{item['val_epoch']:3d}" if item['val_epoch'] is not None else "N/A"
        print(f"dim={item['dim']:3d}: {item['param_count']:8,} params, "
              f"train@{train_str}, val@{val_str}, delay={item['delay']:4d} epochs")
    print(f"{'='*80}")
    
    return delay_data


def plot_grokking_delay_with_speed(
    results: List[Dict],
    speed_data: List[Dict],
    threshold_train: float = 99.0,
    threshold_val: float = 99.0,
    threshold_params: Optional[float] = None,
    batch_size: int = 512,
    n_train_samples: int = 4656,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot grokking delay vs parameter count with an overlaid steps axis.

    Args:
        results: List of dicts with keys 'train_acc', 'val_acc', 'dim', 'param_count'
        speed_data: List of dicts with keys 'param_count', 'dim', 'saturation_step'
        threshold_train: Accuracy threshold for training
        threshold_val: Accuracy threshold for validation
        threshold_params: Threshold for parameter count (vertical line)
        batch_size: Batch size used in grokking experiments
        n_train_samples: Number of training samples
        title: Plot title (auto-generated if None)
        save_path: Path to save the plot (optional)
        show: Whether to show the plot
    """
    if not results:
        print("No results to plot")
        return

    # Calculate steps per epoch
    steps_per_epoch = (n_train_samples + batch_size - 1) // batch_size

    # Calculate delays and grokking epochs for all results
    delay_data = []
    for result in results:
        train_acc = result['train_acc']
        val_acc = result['val_acc']
        dim = result['dim']
        param_count = result['param_count']

        train_epoch, val_epoch, delay = calculate_grokking_delay(train_acc, val_acc, threshold_train, threshold_val)

        if delay is not None and val_epoch is not None:
            delay_data.append({
                'dim': dim,
                'param_count': param_count,
                'train_epoch': train_epoch,
                'val_epoch': val_epoch,
                'delay': delay,
                'steps_to_grok': val_epoch * steps_per_epoch
            })

    if not delay_data:
        print(f"No results reached val={threshold_val}% accuracy threshold")
        return

    # Sort by parameter count
    delay_data.sort(key=lambda x: x['param_count'])

    # Extract data for plotting
    dims = [item['dim'] for item in delay_data]
    param_counts = np.array([item['param_count'] for item in delay_data])
    delays = np.array([item['delay'] for item in delay_data])
    steps_to_grok = np.array([item['steps_to_grok'] for item in delay_data])

    # Remove anomalies: points with positive delay but both neighbors have non-positive delay
    is_grokking = delays > 0
    anomaly_mask = np.ones(len(delays), dtype=bool)

    for i in range(1, len(delays) - 1):
        if is_grokking[i] and not is_grokking[i-1] and not is_grokking[i+1]:
            anomaly_mask[i] = False
            print(f"Filtering out anomaly: dim={dims[i]}, delay={delays[i]:.1f} (neighbors not grokking)")

    # Apply anomaly filter
    dims = [d for d, keep in zip(dims, anomaly_mask) if keep]
    param_counts = param_counts[anomaly_mask]
    delays = delays[anomaly_mask]
    steps_to_grok = steps_to_grok[anomaly_mask]

    # Process speed data - create lookup by param_count
    speed_by_params = {}
    for sd in speed_data:
        pc = sd['param_count']
        if pc not in speed_by_params:
            speed_by_params[pc] = sd['saturation_step']

    # Match speed data to grokking param counts
    speed_steps = []
    speed_params = []
    for pc in param_counts:
        if pc in speed_by_params:
            speed_steps.append(speed_by_params[pc])
            speed_params.append(pc)

    speed_steps = np.array(speed_steps)
    speed_params = np.array(speed_params)

    # Create plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 7))

    # Left y-axis: Grokking delay (epochs)
    crest_cmap = sns.color_palette('crest', as_cmap=True)
    scatter = ax1.scatter(param_counts, delays, c=dims, cmap=crest_cmap,
                         s=80, alpha=0.7, edgecolors='none', label='Grokking delay')

    ax1.set_xlabel('Parameter Count', fontsize=14)
    ax1.set_ylabel('Grokking Delay (epochs)', fontsize=14, color='#1b7a3d')
    ax1.set_xscale('log')
    ax1.tick_params(axis='y', labelcolor='#1b7a3d')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=12)

    # Add horizontal line at y=0
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1, label='Dimension', pad=0.12)
    cbar.ax.tick_params(labelsize=10)

    # Right y-axis: Steps with logarithmic scale
    ax2 = ax1.twinx()

    # Plot steps to grok
    line1, = ax2.plot(param_counts, steps_to_grok, '-', color='#d95f02', linewidth=2,
             markersize=6, alpha=0.8, label='Steps to grok')

    # Plot speed data (steps to memorise) if available
    line2 = None
    if len(speed_steps) > 0:
        line2, = ax2.plot(speed_params, speed_steps, '-', color='#7570b3', linewidth=2,
                 markersize=6, alpha=0.8, label='Steps to memorise')

    ax2.set_ylabel('Steps', fontsize=14, color='#d95f02')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='#d95f02')
    ax2.tick_params(axis='both', which='major', labelsize=12)

    # Find intersection of the two curves if both exist
    intersection_point = None
    if line2 is not None and len(speed_steps) > 0 and len(steps_to_grok) > 0:
        # Interpolate both curves to find intersection
        from scipy.interpolate import interp1d

        # Create interpolation functions for both curves
        f_grok = interp1d(param_counts, steps_to_grok, kind='linear', fill_value='extrapolate')
        f_speed = interp1d(speed_params, speed_steps, kind='linear', fill_value='extrapolate')

        # Find the intersection by looking for where they're closest
        # Use the common x range
        x_min = max(param_counts.min(), speed_params.min())
        x_max = min(param_counts.max(), speed_params.max())

        if x_min < x_max:
            x_test = np.logspace(np.log10(x_min), np.log10(x_max), 1000)
            y_grok_test = f_grok(x_test)
            y_speed_test = f_speed(x_test)

            # Find where the difference is minimum
            diff = np.abs(np.log(y_grok_test) - np.log(y_speed_test))
            idx_closest = np.argmin(diff)

            intersection_x = x_test[idx_closest]
            intersection_y = (y_grok_test[idx_closest] + y_speed_test[idx_closest]) / 2

            intersection_point = (intersection_x, intersection_y)

            # Plot dotted lines to both axes (darker translucent red)
            ax2.axvline(x=intersection_x, color='#8B0000', linestyle=':', linewidth=1.5, alpha=0.5)
            ax2.axhline(y=intersection_y, color='#8B0000', linestyle=':', linewidth=1.5, alpha=0.5)

            # Add text labels on the axes using annotate
            # X-axis label: place on the x-axis with offset above
            ax1.annotate(f'{intersection_x:,.0f}',
                    xy=(intersection_x, ax1.get_ylim()[0]),
                    xytext=(0, -16), textcoords='offset points',
                    ha='center', va='bottom', fontsize=11, color='#8B0000')

            # Y-axis label: place on the y-axis with offset to the right
            ax2.annotate(f'{intersection_y:,.0f}',
                    xy=(ax2.get_xlim()[1], intersection_y),
                    xytext=(-5, 5), textcoords='offset points',
                    ha='right', va='bottom', fontsize=11, color='#8B0000')

            # Print intersection info
            print(f"\nIntersection of curves found:")
            print(f"  Parameter count: {intersection_x:,.0f}")
            print(f"  Steps: {intersection_y:,.0f}")

    # Add vertical line at threshold_params if provided
    if threshold_params is not None:
        ax1.axvline(x=threshold_params, color='black', linestyle=':', linewidth=2, alpha=0.7)
        y_annot = max(delays) if len(delays) > 0 else 0
        ax1.annotate(f"{int(threshold_params):,}",
                    xy=(threshold_params, y_annot),
                    xycoords='data',
                    xytext=(10, -25), textcoords='offset points',
                    fontsize=14, color='black',
                    va='bottom', ha='left')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Delay with speed plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return delay_data


def plot_delay_vs_memorization(
    results: List[Dict],
    mem_key: str = 'mem_u_trace',
    threshold_train: float = 99.0,
    threshold_val: float = 99.0,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot grokking delay vs maximum memorisation.
    
    Args:
        results: List of dicts with keys 'train_acc', 'val_acc', and mem_key, 'dim', 'param_count'
        mem_key: Key to access memorisation trace ('mem_t_trace', 'mem_u_trace', or legacy 'mem_trace')
        threshold_train: Accuracy threshold for training
        threshold_val: Accuracy threshold for validation
        title: Plot title (auto-generated if None)
        save_path: Path to save the plot (optional)
        show: Whether to show the plot
    """
    if not results:
        print("No results to plot")
        return
    
    # Filter results that have memorisation traces
    results_with_mem = [r for r in results if mem_key in r and r[mem_key] is not None and len(r[mem_key]) > 0]
    # Legacy support
    if not results_with_mem and mem_key == 'mem_u_trace':
        results_with_mem = [r for r in results if 'mem_trace' in r and r['mem_trace'] is not None and len(r['mem_trace']) > 0]
        mem_key = 'mem_trace'
    
    if not results_with_mem:
        print(f"No results with memorisation data found (key: {mem_key})")
        return
    
    # Calculate delays and extract max memorisation
    delay_mem_data = []
    for result in results_with_mem:
        train_acc = result['train_acc']
        val_acc = result['val_acc']
        mem_trace = result[mem_key]
        dim = result['dim']
        param_count = result['param_count']
        
        train_epoch, val_epoch, delay = calculate_grokking_delay(train_acc, val_acc, threshold_train, threshold_val)
        max_mem = mem_trace.max()
        
        if delay is not None:
            delay_mem_data.append({
                'dim': dim,
                'param_count': param_count,
                'train_epoch': train_epoch,
                'val_epoch': val_epoch,
                'delay': delay,
                'max_mem': max_mem
            })
        else:
            print(f"Warning: dim={dim} did not reach val={threshold_val}% accuracy")
    
    if not delay_mem_data:
        print(f"No results reached val={threshold_val}% accuracy threshold")
        return
    
    # Sort by max memorisation
    delay_mem_data.sort(key=lambda x: x['max_mem'])
    
    # Extract data for plotting
    dims = [item['dim'] for item in delay_mem_data]
    max_mems = [item['max_mem'] for item in delay_mem_data]
    delays = [item['delay'] for item in delay_mem_data]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot with color-coded dimensions
    crest_cmap = sns.color_palette('crest', as_cmap=True)
    scatter = ax.scatter(max_mems, delays, c=dims, cmap=crest_cmap, 
                        s=40, alpha=0.7, edgecolors='none')
    
    # Add labels for each point
    for i, (mem, delay, dim) in enumerate(zip(max_mems, delays, dims)):
        ax.annotate(f'{dim}', (mem, delay), xytext=(2, 2), 
                   textcoords='offset points', fontsize=7)
    
    # Determine label based on mem_key
    if 'mem_t' in mem_key:
        mem_label = 'M_T'
    elif 'mem_u' in mem_key:
        mem_label = 'M_U'
    else:
        mem_label = 'Memorisation'
    
    ax.set_xlabel(f'Maximum {mem_label} (bits)', fontsize=14)
    ax.set_ylabel(f'Grokking Delay (epochs)', fontsize=14)
    
    if title is None:
        title = f'Grokking Delay vs Maximum Memorisation\n(Delay = epochs for val to reach {threshold_val}% - epochs for train to reach {threshold_train}%)'
    ax.set_title(title, fontsize=16, pad=20)
    
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Dimension')
    cbar.ax.tick_params(labelsize=12)
    
    # Add horizontal line at delay=0 for reference
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Delay vs memorisation plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"DELAY vs MEMORIZATION SUMMARY (train={threshold_train}%, val={threshold_val}%)")
    print(f"{'='*80}")
    for item in delay_mem_data:
        train_str = f"{item['train_epoch']:3d}" if item['train_epoch'] is not None else "N/A"
        val_str = f"{item['val_epoch']:3d}" if item['val_epoch'] is not None else "N/A"
        print(f"dim={item['dim']:3d}: max M_U={item['max_mem']:8,.1f} bits, "
              f"delay={item['delay']:4d} epochs (train@{train_str}, val@{val_str})")
    print(f"{'='*80}")
    
    return delay_mem_data


def _compute_critical_capacity_data(
    results: List[Dict],
    threshold_train: float = 99.0,
    threshold_val: float = 99.0,
    delay_threshold: float = 0.5,
    verbose: bool = True
) -> Optional[Dict]:
    """
    Compute delay data and linear fit for critical capacity analysis.
    
    Args:
        results: List of dicts with keys 'train_acc', 'val_acc', 'dim', 'param_count'
        threshold_train: Accuracy threshold for training
        threshold_val: Accuracy threshold for validation
        delay_threshold: Minimum delay to include in fit (filters out near-zero delays)
        verbose: Whether to print status messages
    
    Returns:
        Dict with keys: 'dims', 'param_counts', 'delays', 'mask', 'a', 'b', 'critical_param_count'
        or None if computation fails
    """
    if not results:
        return None
    
    # Calculate delays for all results
    delay_data = []
    for result in results:
        train_acc = result['train_acc']
        val_acc = result['val_acc']
        dim = result['dim']
        param_count = result['param_count']
        
        train_epoch, val_epoch, delay = calculate_grokking_delay(train_acc, val_acc, threshold_train, threshold_val)
        
        if delay is not None:
            delay_data.append({
                'dim': dim,
                'param_count': param_count,
                'delay': delay
            })
    
    if not delay_data:
        return None
    
    # Sort by parameter count
    delay_data.sort(key=lambda x: x['param_count'])
    
    # Extract data
    dims = [item['dim'] for item in delay_data]
    param_counts = np.array([item['param_count'] for item in delay_data])
    delays = np.array([item['delay'] for item in delay_data])
    
    # Remove anomalies: points with positive delay but both neighbors have non-positive delay
    is_grokking = delays > 0
    anomaly_mask = np.ones(len(delays), dtype=bool)
    
    for i in range(1, len(delays) - 1):
        if is_grokking[i] and not is_grokking[i-1] and not is_grokking[i+1]:
            anomaly_mask[i] = False
            if verbose:
                print(f"Filtering out anomaly: dim={dims[i]}, delay={delays[i]:.1f} (neighbors not grokking)")
    
    # Apply anomaly filter
    dims = [d for d, keep in zip(dims, anomaly_mask) if keep]
    param_counts = param_counts[anomaly_mask]
    delays = delays[anomaly_mask]
    
    # Filter to only include delays above threshold
    mask = delays > delay_threshold
    if mask.sum() < 2:
        return None
    
    # Remove outliers using IQR on delays above threshold
    delays_for_iqr = delays[mask]
    q1, q3 = np.percentile(delays_for_iqr, [25, 75])
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    
    # Update mask to exclude high outliers
    outlier_mask = delays <= upper_bound
    combined_mask = mask & outlier_mask
    
    # Report filtered outliers
    if verbose:
        outliers = mask & ~outlier_mask
        if outliers.any():
            for i, is_outlier in enumerate(outliers):
                if is_outlier:
                    print(f"Filtering out outlier: dim={dims[i]}, delay={delays[i]:.1f} (above IQR upper bound {upper_bound:.1f})")
    
    if combined_mask.sum() < 2:
        return None
    
    log_params_fit = np.log10(param_counts[combined_mask])
    delays_fit = delays[combined_mask]
    
    # Fit linear model: delay = a * log10(params) + b
    a, b = np.polyfit(log_params_fit, delays_fit, 1)
    
    # Find x-intercept: 0 = a * log10(x) + b  =>  log10(x) = -b/a  =>  x = 10^(-b/a)
    if a == 0:
        return None
    
    log_critical = -b / a
    critical_param_count = 10 ** log_critical
    
    return {
        'dims': dims,
        'param_counts': param_counts,
        'delays': delays,
        'mask': combined_mask,
        'a': a,
        'b': b,
        'critical_param_count': critical_param_count,
        'delay_threshold': delay_threshold
    }


def plot_grokking_critical_capacity(
    results: List[Dict],
    threshold_train: float = 99.0,
    threshold_val: float = 99.0,
    delay_threshold: float = 1.0,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Find and plot the critical model size where grokking delay first becomes positive.
    Fits a line to delays > threshold on log x scale and finds x-intercept.
    
    Args:
        results: List of dicts with keys 'train_acc', 'val_acc', 'dim', 'param_count'
        threshold_train: Accuracy threshold for training
        threshold_val: Accuracy threshold for validation
        delay_threshold: Minimum delay to include in fit (filters out near-zero delays)
        title: Plot title (auto-generated if None)
        save_path: Path to save the plot (optional)
        show: Whether to show the plot
    
    Returns:
        critical_param_count: The parameter count where delay crosses zero
    """
    if not results:
        print("No results to plot")
        return None
    
    # Compute delay data and fit
    data = _compute_critical_capacity_data(
        results, threshold_train, threshold_val, delay_threshold, verbose=True
    )
    
    if data is None:
        print(f"Could not compute critical capacity (not enough data points with delay > {delay_threshold})")
        return None
    
    dims = data['dims']
    param_counts = data['param_counts']
    delays = data['delays']
    mask = data['mask']
    a = data['a']
    b = data['b']
    critical_param_count = data['critical_param_count']
    
    print(f"\nCritical capacity analysis:")
    print(f"  Fitted line: delay = {a:.3f} * log10(params) + {b:.3f}")
    print(f"  Critical parameter count (delay = 0): {critical_param_count:,.0f}")
    print(f"  Used {mask.sum()} points with delay > {delay_threshold}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot with color-coded dimensions
    crest_cmap = sns.color_palette('crest', as_cmap=True)
    scatter = ax.scatter(param_counts, delays, c=dims, cmap=crest_cmap, 
                        s=80, alpha=0.7, edgecolors='none')
    
    # Mark points used in fit with larger, darker markers
    ax.scatter(param_counts[mask], delays[mask], 
              s=120, alpha=0.3, edgecolors='red', facecolors='none', linewidths=2,
              label=f'Points used in fit (delay > {delay_threshold})')
    
    # Plot fitted line
    log_param_range = np.linspace(np.log10(param_counts.min() * 0.5), 
                                   np.log10(param_counts.max() * 2), 100)
    param_range = 10 ** log_param_range
    delay_fit_line = a * log_param_range + b
    ax.plot(param_range, delay_fit_line, '--', color='red', linewidth=2, alpha=0.7,
            label=f'Fit: delay = {a:.2f} x log10(params) + {b:.1f}')
    
    # Mark critical point
    ax.axvline(x=critical_param_count, color='green', linestyle='--', linewidth=2, alpha=0.7,
              label=f'Critical capacity: {critical_param_count:,.0f} params')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Parameter Count', fontsize=14)
    ax.set_ylabel(f'Grokking Delay (epochs)', fontsize=14)
    ax.set_xscale('log')
    
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Dimension')
    cbar.ax.tick_params(labelsize=12)
    
    ax.legend(fontsize=12, loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Critical capacity plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return critical_param_count


def plot_delay_and_memorization_vs_params(
    results: List[Dict],
    mem_key: str = 'mem_u_trace',
    threshold_train: float = 99.0,
    threshold_val: float = 99.0,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot both grokking delay and maximum memorisation vs parameter count on the same graph.
    
    Args:
        results: List of dicts with keys 'train_acc', 'val_acc', and mem_key, 'dim', 'param_count'
        mem_key: Key to access memorisation trace ('mem_t_trace', 'mem_u_trace', or legacy 'mem_trace')
        threshold_train: Accuracy threshold for training
        threshold_val: Accuracy threshold for validation
        title: Plot title (auto-generated if None)
        save_path: Path to save the plot (optional)
        show: Whether to show the plot
    """
    if not results:
        print("No results to plot")
        return
    
    # Filter results that have memorisation traces
    results_with_mem = [r for r in results if mem_key in r and r[mem_key] is not None and len(r[mem_key]) > 0]
    # Legacy support
    if not results_with_mem and mem_key == 'mem_u_trace':
        results_with_mem = [r for r in results if 'mem_trace' in r and r['mem_trace'] is not None and len(r['mem_trace']) > 0]
        mem_key = 'mem_trace'
    
    if not results_with_mem:
        print(f"No results with memorisation data found (key: {mem_key})")
        return
    
    # Calculate delays and extract max memorisation
    data_points = []
    for result in results_with_mem:
        train_acc = result['train_acc']
        val_acc = result['val_acc']
        mem_trace = result[mem_key]
        dim = result['dim']
        param_count = result['param_count']
        
        train_epoch, val_epoch, delay = calculate_grokking_delay(train_acc, val_acc, threshold_train, threshold_val)
        max_mem = mem_trace.max()
        
        if delay is not None:
            data_points.append({
                'dim': dim,
                'param_count': param_count,
                'delay': delay,
                'max_mem': max_mem
            })
        else:
            print(f"Warning: dim={dim} did not reach val={threshold_val}% accuracy")
    
    if not data_points:
        print(f"No results reached val={threshold_val}% accuracy threshold")
        return
    
    # Sort by parameter count
    data_points.sort(key=lambda x: x['param_count'])
    
    # Extract data for plotting
    dims = [item['dim'] for item in data_points]
    param_counts = [item['param_count'] for item in data_points]
    delays = [item['delay'] for item in data_points]
    max_mems = [item['max_mem'] for item in data_points]
    
    # Determine label based on mem_key
    if 'mem_u' in mem_key:
        mem_label = 'M_U'
        legend_label = 'Max M_U (bits)'
    else:
        mem_label = 'Memorisation'
        legend_label = 'Max Memorisation (bits)'
    
    # Create plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    
    # Use crest colormap for dimensions
    crest_cmap = sns.color_palette('crest', as_cmap=True)
    
    # Plot delay on first axis (circles)
    scatter1 = ax1.scatter(param_counts, delays, c=dims, cmap=crest_cmap, 
                          s=80, alpha=0.7, edgecolors='none', marker='o', label='Grokking Delay')
    
    # Plot max memorisation on second axis (triangles)
    scatter2 = ax2.scatter(param_counts, max_mems, c=dims, cmap=crest_cmap, 
                          s=80, alpha=0.7, edgecolors='none', marker='^', label=legend_label)
    
    # Add labels for delay points (circles)
    for i, (pc, delay, dim) in enumerate(zip(param_counts, delays, dims)):
        ax1.annotate(f'{dim}', (pc, delay), xytext=(-5, -15), 
                   textcoords='offset points', fontsize=8)
    
    # Configure axes
    ax1.set_xlabel('Parameter Count', fontsize=14)
    ax1.set_ylabel('Grokking Delay (epochs)', fontsize=14)
    ax1.tick_params(axis='y')
    ax1.set_xscale('log')
    
    ax2.set_ylabel(f'Maximum {mem_label} (bits)', fontsize=14)
    ax2.tick_params(axis='y')
    
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, 
               label='Grokking Delay'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, 
               label='Max M_U (bits)')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Delay and memorisation vs params plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return data_points


def plot_grokking_time(results, threshold_val=99.0, max_epochs=None, title=None, save_path=None, show=True):
    """
    Plot when each model reaches a threshold accuracy (grokking time).
    
    Args:
        results: List of dicts with keys 'val_acc', 'dim', 'param_count'
        threshold_val: Accuracy threshold for validation
        max_epochs: Maximum epochs if threshold not reached
        title: Plot title (auto-generated if None)
        save_path: Path to save the plot (optional)
        show: Whether to show the plot
    """
    if not results:
        print("No results to plot")
        return
    
    dims = [r['dim'] for r in results]
    param_counts = [r['param_count'] for r in results]
    
    # Find epoch where validation accuracy reaches threshold
    time_data = []
    for result in results:
        val_acc = result['val_acc']
        dim = result['dim']
        param_count = result['param_count']
        
        grokking_epoch = None
        for epoch, acc in enumerate(val_acc):
            if acc >= threshold_val:
                grokking_epoch = epoch
                break
        
        if grokking_epoch is None:
            if max_epochs is None:
                max_epochs = len(val_acc)
            print(f"Warning: dim={dim} did not reach {threshold_val}% accuracy")
        else:
            time_data.append({
                'dim': dim,
                'param_count': param_count,
                'grokking_epoch': grokking_epoch
            })
    
    if not time_data:
        print(f"No results reached val={threshold_val}% accuracy threshold")
        return
    
    # Sort by parameter count
    time_data.sort(key=lambda x: x['param_count'])
    
    # Extract data for plotting
    dims_plot = [item['dim'] for item in time_data]
    param_counts_plot = [item['param_count'] for item in time_data]
    grokking_epochs = [item['grokking_epoch'] for item in time_data]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot with color-coded dimensions
    crest_cmap = sns.color_palette('crest', as_cmap=True)
    scatter = ax.scatter(param_counts_plot, grokking_epochs, c=dims_plot, cmap=crest_cmap, 
                        s=80, alpha=0.7, edgecolors='none')
    
    # Add labels for each point
    for i, (pc, ge, d) in enumerate(zip(param_counts_plot, grokking_epochs, dims_plot)):
        ax.annotate(f'{d}', (pc, ge), xytext=(5, 5), 
                   textcoords='offset points', fontsize=10)
    
    ax.set_xlabel('Parameter Count', fontsize=14)
    ax.set_ylabel(f'Epochs to val={threshold_val}% Accuracy', fontsize=14)

    if title is None:
        title = f'Grokking Time vs Model Size\n(Epochs for validation to reach val={threshold_val}%)'
    ax.set_title(title, fontsize=16, pad=20)
    
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Dimension')
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Grokking time plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"GROKKING TIME SUMMARY (val={threshold_val}%)")
    print(f"{'='*80}")
    for item in time_data:
        print(f"dim={item['dim']:3d}: {item['param_count']:8,} params, "
              f"grokked at epoch {item['grokking_epoch']:3d}")
    print(f"{'='*80}")
    
    return time_data


# =============================================================================
# Capacity Experiment Plotting Functions
# =============================================================================

def estimate_capacity(saturation_points: List[Tuple[int, float]]) -> Tuple[float, float, float]:
    """
    Estimate capacity from saturation points using linear regression.
    
    Fits: bits = C * params + intercept
    
    Args:
        saturation_points: List of (param_count, saturated_bits) tuples
    
    Returns:
        Tuple of (C, intercept, r_squared)
        - C: Bits per parameter (slope of linear fit)
        - intercept: y-intercept of the fit
        - r_squared: R² goodness of fit
    """
    if len(saturation_points) < 2:
        return 0.0, 0.0, 0.0

    print(saturation_points)
    
    params = np.array([p for p, _ in saturation_points])
    bits = np.array([b for _, b in saturation_points])
    
    # Linear regression: bits = C * params + intercept
    C, intercept = np.polyfit(params, bits, 1)
    
    # Calculate R²
    y_pred = C * params + intercept
    ss_res = np.sum((bits - y_pred) ** 2)
    ss_tot = np.sum((bits - np.mean(bits)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return C, intercept, r_squared


def plot_capacity_curves(
    all_results: Dict[int, List[Dict]],
    p: int = 97,
    save_path: Optional[str] = None,
    show: bool = True
) -> List[Tuple[int, float]]:
    """
    Plot memorisation vs dataset size curves for different model sizes.
    
    Args:
        all_results: Dict mapping dimension to list of result dicts with keys:
                     'n_samples', 'total_bits_memorized', 'param_count'
        p: Prime number for reference line
        save_path: Path to save the plot (optional)
        show: Whether to show the plot
    
    Returns:
        saturation_points: List of (param_count, saturated_bits) tuples
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set log-log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Sort dims for consistent legend ordering
    dims = sorted(all_results.keys())
    
    # Get colors from seaborn crest colormap
    colors = sns.color_palette("crest", n_colors=len(dims))
    
    # For estimating C, collect saturation points
    saturation_points = []  # (param_count, saturated_bits)
    
    for idx, dim in enumerate(dims):
        results = all_results[dim]
        
        # Sort by dataset size
        results = sorted(results, key=lambda x: x['n_samples'])
        
        dataset_sizes = [r['n_samples'] for r in results]
        total_bits = [r.get('total_bits_memorized', r.get('total_bits', 0)) for r in results]
        param_count = results[0]['param_count']
        
        # Format parameter count for legend
        if param_count >= 1e6:
            param_str = f'{param_count/1e6:.1f}M'
        elif param_count >= 1e3:
            param_str = f'{param_count/1e3:.0f}K'
        else:
            param_str = str(param_count)
        
        ax.plot(
            dataset_sizes,
            total_bits,
            marker='o',
            markersize=6,
            linewidth=2,
            color=colors[idx],
            label=param_str
        )
        
        # Find saturation point (where curve flattens)
        max_bits = max(total_bits)
        saturation_points.append((param_count, max_bits))
    
    # Plot reference line showing dataset size in bits
    all_sizes = []
    for dim in dims:
        all_sizes.extend([r['n_samples'] for r in all_results[dim]])
    if all_sizes:
        x_min, x_max = min(all_sizes) * 0.5, max(all_sizes) * 2
        x_range = np.logspace(np.log10(x_min), np.log10(x_max), 50)
        bits_per_example = np.log2(p + 2)
        dataset_bits = x_range * bits_per_example
        ax.plot(x_range, dataset_bits, '--', color='gray', alpha=0.5, label='Dataset size')
        
    ax.set_xlabel('Dataset size\n(number of datapoints)', fontsize=14)
    ax.set_ylabel('Memorisation\n(bits)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Legend
    ax.legend(
        title='Parameters',
        loc='upper left',
        fontsize=11,
        title_fontsize=12
    )
    
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Add rotated text label for the diagonal (after tight_layout so transform is accurate)
    if all_sizes:
        mid_x = np.sqrt(x_min * x_max)
        mid_y = mid_x * bits_per_example * 1.5
        
        # Calculate rotation angle from actual display coordinates
        p1 = ax.transData.transform([x_min, x_min * bits_per_example])
        p2 = ax.transData.transform([x_max, x_max * bits_per_example])
        angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
        
        ax.text(
            mid_x, mid_y,
            'Dataset size (bits)',
            rotation=angle,
            rotation_mode='anchor',
            color='gray',
            fontsize=12,
            alpha=0.8
        )
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved capacity plot: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return saturation_points


def plot_capacity_estimation(
    saturation_points: List[Tuple[int, float]],
    save_path: Optional[str] = None,
    show: bool = True
) -> Tuple[float, float, float]:
    """
    Plot saturation memorisation vs parameters with linear fit.
    
    Args:
        saturation_points: List of (param_count, saturated_bits) tuples
        save_path: Path to save the plot (optional)
        show: Whether to show the plot
    
    Returns:
        Tuple of (C, intercept, r_squared)
    """
    if len(saturation_points) < 2:
        print("Not enough data points for capacity estimation plot")
        return 0.0, 0.0, 0.0
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    params = np.array([p for p, _ in saturation_points])
    bits = np.array([b for _, b in saturation_points])
    
    ax.scatter(params, bits, s=100, zorder=5, label='Data')
    
    # Fit linear model: bits = C * params + intercept
    C, intercept, r_squared = estimate_capacity(saturation_points)
    
    # Plot fitted line
    x_fit = np.linspace(0, params.max() * 1.1, 100)
    y_fit = C * x_fit + intercept
    
    # Format fit equation
    sign = '+' if intercept >= 0 else '−'
    abs_intercept = abs(intercept)
    ax.plot(x_fit, y_fit, '--', linewidth=2, color='C1',
            label=f'Fit: bits = {C:.2f} × params {sign} {abs_intercept:.0f}')
    
    ax.set_xlabel('Number of Parameters', fontsize=14)
    ax.set_ylabel('Saturation Memorisation (bits)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add text box with stats
    textstr = f'C: {C:.2f} bits/param\n'
    textstr += f'Intercept: {intercept:.0f} bits\n'
    textstr += f'R²: {r_squared:.3f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved capacity estimation plot: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return C, intercept, r_squared


def plot_bits_vs_accuracy(
    results: List[Dict],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot bits memorized vs training accuracy for capacity experiments.
    
    Args:
        results: List of result dicts with keys 'final_acc', 'bits_per_example', 'n_samples'
        save_path: Path to save the plot (optional)
        show: Whether to show the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    accuracies = [r['final_acc'] * 100 if r['final_acc'] < 1.5 else r['final_acc'] for r in results]
    bits = [r['bits_per_example'] for r in results]
    sizes = [r['n_samples'] for r in results]
    
    scatter = ax.scatter(accuracies, bits, c=sizes, cmap='viridis', 
                         s=80, alpha=0.7, norm=plt.matplotlib.colors.LogNorm())
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Dataset Size', fontsize=12)
    
    ax.set_xlabel('Training Accuracy (%)', fontsize=14)
    ax.set_ylabel('Bits Memorized per Example', fontsize=14)
    ax.set_title('Memorisation vs Accuracy', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved bits vs accuracy plot: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# =============================================================================
# Unintended Memorisation (M_U) Plotting Functions
# =============================================================================

def plot_memorization_curves(
    results: List[Dict],
    mem_key: str = 'mem_trace',
    title: str = 'Memorisation vs Training Epoch',
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot memorisation curves for multiple experiments with crest colorbar depicting model size.
    
    Args:
        results: List of dicts with keys [mem_key], 'dim', 'param_count'
        mem_key: Key to access memorisation trace ('mem_t_trace', 'mem_u_trace', or legacy 'mem_trace')
        title: Plot title
        save_path: Path to save the plot (optional)
        show: Whether to show the plot
    """
    if not results:
        print("No results to plot")
        return
    
    # Filter results that have memorisation traces
    results_with_mem = [r for r in results if mem_key in r and r[mem_key] is not None]
    if not results_with_mem:
        print(f"No results with memorisation data found (key: {mem_key})")
        return
    
    # Sort by dimension
    results_sorted = sorted(results_with_mem, key=lambda x: x['dim'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use crest colormap
    crest_cmap = sns.color_palette('crest', as_cmap=True)
    colors = crest_cmap(np.linspace(0, 1, len(results_sorted)))
    
    for i, result in enumerate(results_sorted):
        mem_trace = result[mem_key]
        dim = result['dim']
        
        ax.plot(mem_trace, color=colors[i], linewidth=2, alpha=0.8)
        
        # Add label at the end of the curve
        y_pos = mem_trace[-1]
        ax.text(len(mem_trace) + 2, y_pos, f'{dim}', 
                color=colors[i], fontsize=8, va='center', ha='left',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.5))
    
    # Determine y-axis label based on mem_key
    if 'mem_t' in mem_key:
        ylabel = 'Total Memorisation M_T (bits)'
    elif 'mem_u' in mem_key:
        ylabel = 'Unintended Memorisation M_U (bits)'
    else:
        ylabel = 'Memorisation (bits)'
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title(title, fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add colorbar showing dimension gradient
    dims = [r['dim'] for r in results_sorted]
    sm = plt.cm.ScalarMappable(cmap=crest_cmap, 
                               norm=plt.Normalize(vmin=min(dims), vmax=max(dims)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Dimension', pad=0.02)
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Memorisation plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_max_memorization_vs_params(
    results: List[Dict],
    mem_key: str = 'mem_trace',
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot maximum memorisation vs parameter count.
    
    Args:
        results: List of dicts with keys [mem_key], 'dim', 'param_count'
        mem_key: Key to access memorisation trace ('mem_t_trace', 'mem_u_trace', or legacy 'mem_trace')
        title: Plot title (auto-generated if None)
        save_path: Path to save the plot (optional)
        show: Whether to show the plot
    """
    if not results:
        print("No results to plot")
        return
    
    # Filter results that have memorisation traces
    results_with_mem = [r for r in results if mem_key in r and r[mem_key] is not None and len(r[mem_key]) > 0]
    if not results_with_mem:
        print(f"No results with memorisation data found (key: {mem_key})")
        return
    
    # Sort by parameter count
    results_sorted = sorted(results_with_mem, key=lambda x: x['param_count'])
    
    # Extract data
    dims = [r['dim'] for r in results_sorted]
    param_counts = [r['param_count'] for r in results_sorted]
    max_mems = [r[mem_key].max() for r in results_sorted]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot with color-coded dimensions
    crest_cmap = sns.color_palette('crest', as_cmap=True)
    scatter = ax.scatter(param_counts, max_mems, c=dims, cmap=crest_cmap, 
                        s=40, alpha=0.7, edgecolors='none')
    
    # Add labels for each point
    for i, (pc, mem, dim) in enumerate(zip(param_counts, max_mems, dims)):
        ax.annotate(f'{dim}', (pc, mem), xytext=(2, 2), 
                   textcoords='offset points', fontsize=7)
    
    if 'mem_u' in mem_key:
        ylabel = 'Maximum Unintended Memorisation M_U (bits)'
        default_title = 'Maximum M_U vs Model Size'
    else:
        ylabel = 'Maximum Memorisation (bits)'
        default_title = 'Maximum Memorisation vs Model Size'
    
    ax.set_xlabel('Parameter Count', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xscale('log')
    
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Dimension')
    cbar.ax.tick_params(labelsize=12)
    
    # Add reference line at M_U=0
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Maximum memorisation vs params plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # Print summary
    mem_label = 'M_T' if 'mem_t' in mem_key else 'M_U' if 'mem_u' in mem_key else 'Mem'
    print(f"\n{'='*80}")
    print(f"MAXIMUM MEMORIZATION SUMMARY ({mem_label})")
    print(f"{'='*80}")
    for item in results_sorted:
        mem_val = item[mem_key].max()
        print(f"dim={item['dim']:3d}: {item['param_count']:8,} params, "
              f"max {mem_label}={mem_val:10,.1f} bits, "
              f"{mem_label}/param={mem_val/item['param_count']:.3f} bits/param")
    print(f"{'='*80}")
    
    return results_sorted


def plot_grokking_with_memorization(
    result: Dict,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot grokking curves (train/val accuracy) with memorisation overlaid on a secondary y-axis.
    
    Supports both M_T (total memorisation) and M_U (unintended memorisation).
    
    Args:
        result: Dict with keys 'train_acc', 'val_acc', and optionally 'mem_t_trace', 'mem_u_trace', 'dim', 'param_count'
        title: Plot title (auto-generated if None)
        save_path: Path to save the plot (optional)
        show: Whether to show the plot
    """
    has_mem_t = 'mem_t_trace' in result and result['mem_t_trace'] is not None
    has_mem_u = 'mem_u_trace' in result and result['mem_u_trace'] is not None
    
    if not has_mem_t and not has_mem_u:
        print("No memorisation data in result")
        # Fall back to simple grokking plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(result['train_acc'], label='Training Accuracy', color='#1b9e77', linewidth=2, linestyle='-')
        ax.plot(result['val_acc'], label='Validation Accuracy', color='#d95f02', linewidth=2, linestyle='--')
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('Accuracy (%)', fontsize=14)
        dim = result.get('dim', '?')
        param_count = result.get('param_count', 0)
        ax.set_title(f'Grokking Curve: dim={dim}, {param_count:,} params', fontsize=16, pad=20)
        ax.legend(fontsize=12, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        return
    
    train_acc = result['train_acc']
    val_acc = result['val_acc']
    dim = result.get('dim', '?')
    param_count = result.get('param_count', 0)
    
    # Create figure with primary and secondary y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    
    # Plot accuracy curves on primary axis
    line1, = ax1.plot(train_acc, label='Training Accuracy', color='#1b9e77', linewidth=2, linestyle='-')
    line2, = ax1.plot(val_acc, label='Validation Accuracy', color='#d95f02', linewidth=2, linestyle='--')
    
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Accuracy (%)', fontsize=14, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim([0, 105])
    ax1.grid(True, alpha=0.3)
    
    # Collect all memorisation traces to find y-limits
    all_mem_values = []
    lines = [line1, line2]
    textstr_parts = [f'Final Train Acc: {train_acc[-1]:.1f}%',
                     f'Final Val Acc: {val_acc[-1]:.1f}%']
    
    # Plot M_T if available
    if has_mem_t:
        mem_t = result['mem_t_trace']
        all_mem_values.extend(mem_t)
        line_t, = ax2.plot(mem_t, label='M_T (bits)', color='#e7298a', linewidth=2.5, linestyle='-.')
        lines.append(line_t)
        textstr_parts.append(f'Final M_T: {mem_t[-1]:.1f} bits')
    
    # Plot M_U if available
    if has_mem_u:
        mem_u = result['mem_u_trace']
        all_mem_values.extend(mem_u)
        line_u, = ax2.plot(mem_u, label='M_U (bits)', color='#7570b3', linewidth=2.5, linestyle=':')
        lines.append(line_u)
        textstr_parts.append(f'Final M_U: {mem_u[-1]:.1f} bits')
    
    # Set y-axis label based on what's available
    if has_mem_t and has_mem_u:
        ax2.set_ylabel('Memorisation (bits)', fontsize=14, color='#7570b3')
        ax2.set_yscale('log')
    elif has_mem_t:
        ax2.set_ylabel('Total Memorisation M_T (bits)', fontsize=14, color='#e7298a')
    else:
        ax2.set_ylabel('Unintended Memorisation M_U (bits)', fontsize=14, color='#7570b3')
    ax2.tick_params(axis='y', labelcolor='#7570b3')
    
    # Calculate appropriate limits for memorisation axis
    mem_min = min(all_mem_values)
    mem_max = max(all_mem_values)
    mem_range = mem_max - mem_min if mem_max != mem_min else 1
    
    # Add some padding (5% on top, and ensure 0 is included)
    if mem_min >= 0:
        ax2.set_ylim([0, mem_max * 1.05])
    else:
        padding = mem_range * 0.05
        ax2.set_ylim([mem_min - padding, mem_max + padding])
    
    # Add horizontal line at 0 (same style as other reference lines)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Combined legend
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=12, loc='center left')
    
    if title is None:
        title = f'dim={dim}, params={param_count:,}'
    ax1.set_title(title, fontsize=16, pad=20)
    
    # Add text annotation
    textstr = '\n'.join(textstr_parts)
    ax1.text(0.8, 0.15, textstr, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Grokking with memorisation plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def compute_critical_params(
    results: List[Dict],
    threshold_train: float = 99.0,
    threshold_val: float = 99.0,
    delay_threshold: float = 1.0,
) -> Optional[float]:
    """
    Compute the critical parameter count where grokking delay first becomes positive.
    Fits a line to delays > threshold on log x scale and finds x-intercept.
    
    Args:
        results: List of dicts with keys 'train_acc', 'val_acc', 'dim', 'param_count'
        threshold_train: Accuracy threshold for training
        threshold_val: Accuracy threshold for validation
        delay_threshold: Minimum delay to include in fit (filters out near-zero delays)
    
    Returns:
        critical_param_count: The parameter count where delay crosses zero, or None if cannot be computed
    """
    data = _compute_critical_capacity_data(
        results, threshold_train, threshold_val, delay_threshold, verbose=False
    )
    
    if data is None:
        return None
    
    return data['critical_param_count']


def compute_critical_params_from_speed(
    results: List[Dict],
    speed_data: List[Dict],
    threshold_train: float = 99.0,
    threshold_val: float = 99.0,
) -> Optional[float]:
    """
    Compute the critical parameter count using empirical intersection of curves.

    Finds where the "steps to grok" curve intersects with the "steps to memorise" curve
    by matching grokking results with speed data.

    Args:
        results: List of dicts with keys 'train_acc', 'val_acc', 'dim', 'param_count'
        speed_data: List of dicts with keys 'param_count', 'dim', 'saturation_step', 'n_samples'
        threshold_train: Accuracy threshold for training
        threshold_val: Accuracy threshold for validation

    Returns:
        critical_param_count: The parameter count at intersection of curves, or None if cannot be computed
    """
    if not results or not speed_data:
        return None

    # Calculate delays and grokking epochs for all results
    delay_data = []
    for result in results:
        train_acc = result['train_acc']
        val_acc = result['val_acc']
        param_count = result['param_count']

        train_epoch, val_epoch, delay = calculate_grokking_delay(train_acc, val_acc, threshold_train, threshold_val)

        if delay is not None and val_epoch is not None:
            delay_data.append({
                'param_count': param_count,
                'val_epoch': val_epoch,
                'epochs_to_grok': val_epoch
            })

    if not delay_data:
        return None

    # Sort by parameter count
    delay_data.sort(key=lambda x: x['param_count'])

    # Extract data for plotting
    param_counts = np.array([item['param_count'] for item in delay_data])
    epochs_to_grok = np.array([item['epochs_to_grok'] for item in delay_data])

    # Process speed data - create lookup by param_count
    speed_by_params = {}
    for sd in speed_data:
        pc = sd['param_count']
        if pc not in speed_by_params:
            speed_by_params[pc] = sd['saturation_step']

    # Match speed data to grokking param counts
    speed_steps = []
    speed_params = []
    for pc in param_counts:
        if pc in speed_by_params:
            speed_steps.append(speed_by_params[pc])
            speed_params.append(pc)

    speed_steps = np.array(speed_steps)
    speed_params = np.array(speed_params)

    # Find intersection of the two curves if both exist
    if len(speed_steps) > 0 and len(epochs_to_grok) > 0:
        # Interpolate both curves to find intersection
        from scipy.interpolate import interp1d

        # Create interpolation functions for both curves
        f_grok = interp1d(param_counts, epochs_to_grok, kind='linear', fill_value='extrapolate')
        f_speed = interp1d(speed_params, speed_steps, kind='linear', fill_value='extrapolate')

        # Find the intersection by looking for where they're closest
        # Use the common x range
        x_min = max(param_counts.min(), speed_params.min())
        x_max = min(param_counts.max(), speed_params.max())

        if x_min < x_max:
            x_test = np.logspace(np.log10(x_min), np.log10(x_max), 1000)
            y_grok_test = f_grok(x_test)
            y_speed_test = f_speed(x_test)

            # Find where the difference is minimum
            diff = np.abs(np.log(y_grok_test) - np.log(y_speed_test))
            idx_closest = np.argmin(diff)

            intersection_x = x_test[idx_closest]

            return intersection_x

    return None


def plot_cross_exp_critical(
    cross_exp_data: List[Dict],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot line-fitting based critical params vs speed-based (intersection) critical params.

    Compares the empirical critical point obtained via line-fitting method with that obtained
    via the intersection method used in visualize.py -> groks -> --speed.

    Args:
        cross_exp_data: List of dicts with keys 'signature', 'critical_params', 'critical_params_speed'
        title: Plot title (auto-generated if None)
        save_path: Path to save the plot (optional)
        show: Whether to show the plot
    """
    if not cross_exp_data:
        print("No cross-experiment data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract data for plotting
    signatures = [d['signature'] for d in cross_exp_data]
    critical_params_line = [d['critical_params'] for d in cross_exp_data]
    critical_params_speed = [d['critical_params_speed'] for d in cross_exp_data]

    # Create scatter plot
    colors = sns.color_palette('husl', len(cross_exp_data))

    for i, (cp_line, cp_speed, sig) in enumerate(zip(critical_params_line, critical_params_speed, signatures)):
        if cp_speed is not None:
            ax.scatter(cp_line, cp_speed, c=[colors[i]], s=150, alpha=0.8, label=sig,
                      edgecolors='white', linewidth=2)

    # Fit line of best fit between line-fitting and speed-based critical points
    # Only use points where both are available
    valid_indices = [i for i, cp_speed in enumerate(critical_params_speed) if cp_speed is not None]
    if len(valid_indices) >= 2:
        cp_line_arr = np.array([critical_params_line[i] for i in valid_indices])
        cp_speed_arr = np.array([critical_params_speed[i] for i in valid_indices])

        # Linear fit: y = mx + b
        m, b = np.polyfit(cp_line_arr, cp_speed_arr, 1)

        # Calculate R²
        y_pred = m * cp_line_arr + b
        ss_res = np.sum((cp_speed_arr - y_pred) ** 2)
        ss_tot = np.sum((cp_speed_arr - np.mean(cp_speed_arr)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Plot line of best fit
        x_line = np.linspace(cp_line_arr.min() * 0.95, cp_line_arr.max() * 1.05, 100)
        y_line = m * x_line + b
        ax.plot(x_line, y_line, 'k--', linewidth=2, alpha=0.6,
                label=f'Fit: y = {m:.3f}x + {b:.0f}\n$R^2$ = {r_squared:.3f}')

        # Print R² to console
        print(f"\nLine of best fit:")
        print(f"  y = {m:.3f}x + {b:.0f}")
        print(f"  R² = {r_squared:.3f}")

    ax.set_xlabel('Critical Params (line-fitting)', fontsize=14)
    ax.set_ylabel('Critical Params (speed-based / intersection)', fontsize=14)
    
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Cross-experiment plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


# =============================================================================
# Learning Speed Plotting Functions
# =============================================================================

def plot_learning_speed_curves(
    all_results: Dict[int, List[Dict]],
    p: int = 97,
    save_path: Optional[str] = None,
    show: bool = True
) -> Dict[int, Tuple[float, float, int]]:
    """
    Plot saturation steps vs dataset size (in bits) for different model sizes.

    Args:
        all_results: Dict mapping dimension to list of result dicts
        p: Prime number for computing bits
        save_path: Path to save the plot (optional)
        show: Whether to show the plot

    Returns:
        speed_estimates: Dict mapping param_count to (slope, intercept, dim) of linear fit
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Sort dims for consistent legend ordering
    dims = sorted(all_results.keys())

    # Get colors from seaborn crest colormap
    colors = sns.color_palette("crest", n_colors=len(dims))

    speed_estimates = {}

    for idx, dim in enumerate(dims):
        results = all_results[dim]

        # Filter to only saturated experiments
        saturated_results = [r for r in results if r.get('saturated', True)]
        if not saturated_results:
            print(f"Warning: No saturated results for dim={dim}")
            continue

        # Sort by dataset size
        saturated_results = sorted(saturated_results, key=lambda x: x['n_samples'])

        dataset_bits = [r['dataset_bits'] for r in saturated_results]
        saturation_steps = [r['saturation_step'] for r in saturated_results]
        param_count = saturated_results[0]['param_count']

        # Format parameter count for legend
        if param_count >= 1e6:
            param_str = f'{param_count/1e6:.1f}M'
        elif param_count >= 1e3:
            param_str = f'{param_count/1e3:.0f}K'
        else:
            param_str = str(param_count)

        ax.plot(
            dataset_bits,
            saturation_steps,
            marker='o',
            markersize=8,
            linewidth=2,
            color=colors[idx],
            label=f'{param_str} params (dim={dim})'
        )

        # Fit linear model for speed estimate
        if len(dataset_bits) >= 2:
            slope, intercept = np.polyfit(dataset_bits, saturation_steps, 1)
            speed_estimates[param_count] = (slope, intercept, dim)

    ax.set_xlabel('Dataset Size (bits)', fontsize=14)
    ax.set_ylabel('Steps to Saturation', fontsize=14)
    ax.set_title('Learning Speed: Steps to Memorize Random Data', fontsize=16, pad=20)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Log scale for both axes
    ax.set_yscale('log')

    # Legend
    ax.legend(
        title='Model Size',
        loc='upper left',
        fontsize=10,
        title_fontsize=11
    )

    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved learning speed plot: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return speed_estimates


def plot_speed_vs_model_size(
    speed_estimates: Dict[int, Tuple[float, float, int]],
    save_path: Optional[str] = None,
    show: bool = True
) -> Tuple[float, float, float]:
    """
    Plot learning speed (steps per bit) vs model parameter count.

    Args:
        speed_estimates: Dict mapping param_count to (slope, intercept, dim)
        save_path: Path to save the plot (optional)
        show: Whether to show the plot

    Returns:
        Tuple of (slope_of_log_fit, intercept_of_log_fit, r_squared)
    """
    if len(speed_estimates) < 2:
        print("Not enough data points for speed vs model size plot")
        return 0.0, 0.0, 0.0

    fig, ax = plt.subplots(figsize=(10, 8))

    # Sort by parameter count
    sorted_items = sorted(speed_estimates.items())
    param_counts = np.array([p for p, _ in sorted_items])
    slopes = np.array([s[0] for _, s in sorted_items])  # steps per bit
    dims = [s[2] for _, s in sorted_items]

    # Use crest colormap
    crest_cmap = sns.color_palette('crest', as_cmap=True)

    scatter = ax.scatter(
        param_counts, slopes,
        c=dims, cmap=crest_cmap,
        s=100, alpha=0.8, edgecolors='black', linewidths=1
    )

    # Add labels for each point
    for pc, slope, dim in zip(param_counts, slopes, dims):
        ax.annotate(f'dim={dim}', (pc, slope), xytext=(5, 5),
                   textcoords='offset points', fontsize=9)

    # Fit power law: slope = a * params^b  =>  log(slope) = log(a) + b * log(params)
    log_params = np.log10(param_counts)
    log_slopes = np.log10(slopes)

    b, log_a = np.polyfit(log_params, log_slopes, 1)
    a = 10 ** log_a

    # Calculate R²
    y_pred = log_a + b * log_params
    ss_res = np.sum((log_slopes - y_pred) ** 2)
    ss_tot = np.sum((log_slopes - np.mean(log_slopes)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Plot fitted curve
    x_fit = np.logspace(np.log10(param_counts.min() * 0.5),
                        np.log10(param_counts.max() * 2), 100)
    y_fit = a * x_fit ** b
    ax.plot(x_fit, y_fit, '--', color='red', linewidth=2, alpha=0.7,
            label=f'Fit: speed = {a:.2e} × params^{b:.2f}')

    ax.set_xlabel('Model Parameters', fontsize=14)
    ax.set_ylabel('Learning Speed (steps per bit)', fontsize=14)
    ax.set_title('Learning Speed vs Model Size', fontsize=16, pad=20)
    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.set_xscale('log')
    ax.set_yscale('log')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Dimension')
    cbar.ax.tick_params(labelsize=11)

    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')

    # Add text box with stats
    textstr = f'Power law exponent: {b:.3f}\n'
    textstr += f'Coefficient: {a:.2e}\n'
    textstr += f'R²: {r_squared:.3f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved speed vs model size plot: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return b, log_a, r_squared


def plot_combined_speed_analysis(
    all_results: Dict[int, List[Dict]],
    p: int = 97,
    save_path: Optional[str] = None,
    show: bool = True
) -> Dict[int, Tuple[float, float, int]]:
    """
    Create a combined figure with both learning speed curves and speed vs model size.

    Args:
        all_results: Dict mapping dimension to list of result dicts
        p: Prime number for computing bits
        save_path: Path to save the plot (optional)
        show: Whether to show the plot

    Returns:
        speed_estimates: Dict mapping param_count to (slope, intercept, dim)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Sort dims for consistent legend ordering
    dims = sorted(all_results.keys())

    # Get colors from seaborn crest colormap
    colors = sns.color_palette("crest", n_colors=len(dims))

    speed_estimates = {}

    # Left plot: Steps to saturation vs dataset bits
    for idx, dim in enumerate(dims):
        results = all_results[dim]

        # Filter to only saturated experiments
        saturated_results = [r for r in results if r.get('saturated', True)]
        if not saturated_results:
            continue

        saturated_results = sorted(saturated_results, key=lambda x: x['n_samples'])

        dataset_bits = [r['dataset_bits'] for r in saturated_results]
        saturation_steps = [r['saturation_step'] for r in saturated_results]
        param_count = saturated_results[0]['param_count']

        if param_count >= 1e6:
            param_str = f'{param_count/1e6:.1f}M'
        elif param_count >= 1e3:
            param_str = f'{param_count/1e3:.0f}K'
        else:
            param_str = str(param_count)

        ax1.plot(
            dataset_bits,
            saturation_steps,
            marker='o',
            markersize=8,
            linewidth=2,
            color=colors[idx],
            label=f'{param_str}'
        )

        # Fit linear model for speed estimate
        if len(dataset_bits) >= 2:
            slope, intercept = np.polyfit(dataset_bits, saturation_steps, 1)
            speed_estimates[param_count] = (slope, intercept, dim)

    ax1.set_xlabel('Dataset Size (bits)', fontsize=14)
    ax1.set_ylabel('Steps to Saturation', fontsize=14)
    ax1.set_title('Learning Speed Curves', fontsize=16, pad=10)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(title='Parameters', loc='upper left', fontsize=10, title_fontsize=11)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.tick_params(axis='both', which='major', labelsize=12)

    # Right plot: Speed vs model size
    if len(speed_estimates) >= 2:
        sorted_items = sorted(speed_estimates.items())
        param_counts = np.array([p for p, _ in sorted_items])
        slopes = np.array([s[0] for _, s in sorted_items])
        dims_plot = [s[2] for _, s in sorted_items]

        crest_cmap = sns.color_palette('crest', as_cmap=True)

        scatter = ax2.scatter(
            param_counts, slopes,
            c=dims_plot, cmap=crest_cmap,
            s=100, alpha=0.8, edgecolors='black', linewidths=1
        )

        for pc, slope, dim in zip(param_counts, slopes, dims_plot):
            ax2.annotate(f'{dim}', (pc, slope), xytext=(5, 5),
                        textcoords='offset points', fontsize=9)

        # Fit power law
        log_params = np.log10(param_counts)
        log_slopes = np.log10(slopes)
        b, log_a = np.polyfit(log_params, log_slopes, 1)
        a = 10 ** log_a

        y_pred = log_a + b * log_params
        ss_res = np.sum((log_slopes - y_pred) ** 2)
        ss_tot = np.sum((log_slopes - np.mean(log_slopes)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        x_fit = np.logspace(np.log10(param_counts.min() * 0.5),
                            np.log10(param_counts.max() * 2), 100)
        y_fit = a * x_fit ** b
        ax2.plot(x_fit, y_fit, '--', color='red', linewidth=2, alpha=0.7,
                label=f'speed ∝ params^{b:.2f}')

        ax2.set_xlabel('Model Parameters', fontsize=14)
        ax2.set_ylabel('Learning Speed (steps/bit)', fontsize=14)
        ax2.set_title('Speed vs Model Size', fontsize=16, pad=10)
        ax2.set_xscale('log')
        ax2.set_yscale('log')

        cbar = plt.colorbar(scatter, ax=ax2, label='Dimension')
        cbar.ax.tick_params(labelsize=11)

        ax2.legend(fontsize=11, loc='upper right')
        ax2.grid(True, alpha=0.3, which='both')
        ax2.tick_params(axis='both', which='major', labelsize=12)

        textstr = f'Exponent: {b:.3f}\nR²: {r_squared:.3f}'
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved combined speed analysis plot: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return speed_estimates


def plot_saturation_time_vs_capacity_fraction(
    all_results: Dict[int, List[Dict]],
    C: float,
    save_path: Optional[str] = None,
    show: bool = True
) -> Tuple[float, float, float]:
    """
    Plot saturation time vs f where f = S/(CP) is the fraction of total capacity.
    Uses log y-scale and fits an exponential curve: steps = a * exp(b * f)

    Args:
        all_results: Dict mapping dimension to list of result dicts, OR
                     Dict mapping (prime, dimension) tuples to list of result dicts (for multi-prime plots)
        C: Capacity constant (bits per parameter)
        save_path: Path to save the plot (optional)
        show: Whether to show the plot

    Returns:
        Tuple of (exponent, coefficient, r_squared) from exponential fit
    """
    # Detect if we have multi-prime data (keys are tuples) or single-prime (keys are ints)
    first_key = next(iter(all_results.keys()))
    is_multi_prime = isinstance(first_key, tuple)

    # Collect all data points across all dimensions and dataset sizes
    f_values = []
    saturation_steps_values = []
    dims = []
    primes = []
    param_counts = []

    for key, results in all_results.items():
        if is_multi_prime:
            p, dim = key
        else:
            p = None
            dim = key

        for result in results:
            P = result['param_count']
            S = result['dataset_bits']
            saturation_step = result['saturation_step']

            # Get prime from result if not from key
            if p is None:
                p = result.get('p', 0)

            # Calculate fraction of capacity: f = S / (C * P)
            f = S / (C * P)

            f_values.append(f)
            saturation_steps_values.append(saturation_step)
            dims.append(dim)
            primes.append(p)
            param_counts.append(P)

    if len(f_values) < 2:
        print("Not enough data points for capacity fraction analysis")
        return 0.0, 0.0, 0.0

    f_values = np.array(f_values)
    saturation_steps_values = np.array(saturation_steps_values)
    dims = np.array(dims)
    primes = np.array(primes)

    # Fit exponential in semi-log space: log(steps) = b * f + log(a)
    # This gives us: steps = a * exp(b * f)
    log_steps = np.log(saturation_steps_values)

    b, log_a = np.polyfit(f_values, log_steps, 1)
    a = np.exp(log_a)

    # Calculate R² in log space
    y_pred_log = b * f_values + log_a
    ss_res = np.sum((log_steps - y_pred_log) ** 2)
    ss_tot = np.sum((log_steps - np.mean(log_steps)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    if is_multi_prime:
        # Color by prime for multi-prime plots
        unique_primes = sorted(set(primes))
        colors = sns.color_palette("Set2", n_colors=len(unique_primes))
        prime_to_color = {p: colors[i] for i, p in enumerate(unique_primes)}

        # Plot data points colored by prime
        for p in unique_primes:
            mask = primes == p
            ax.scatter(
                f_values[mask],
                saturation_steps_values[mask],
                c=[prime_to_color[p]],
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidths=1,
                label=f'p={p}'
            )

            # Fit and plot separate curve for this prime
            if np.sum(mask) >= 2:
                f_prime = f_values[mask]
                steps_prime = saturation_steps_values[mask]
                log_steps_prime = np.log(steps_prime)

                b_prime, log_a_prime = np.polyfit(f_prime, log_steps_prime, 1)
                a_prime = np.exp(log_a_prime)

                # Plot fitted curve for this prime
                x_fit_prime = np.linspace(f_prime.min() * 0.95, f_prime.max() * 1.05, 100)
                y_fit_prime = a_prime * np.exp(b_prime * x_fit_prime)
                ax.plot(x_fit_prime, y_fit_prime, '--', color=prime_to_color[p],
                       linewidth=2, alpha=0.8)
    else:
        # Color by dimension for single-prime plots
        unique_dims = sorted(set(dims))
        colors = sns.color_palette("crest", n_colors=len(unique_dims))
        dim_to_color = {d: colors[i] for i, d in enumerate(unique_dims)}

        for dim in unique_dims:
            mask = dims == dim
            ax.scatter(
                f_values[mask],
                saturation_steps_values[mask],
                c=[dim_to_color[dim]],
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidths=1,
                label=f'dim={dim}'
            )

        # Plot overall fitted exponential curve for single prime
        x_fit = np.linspace(f_values.min() * 0.95, f_values.max() * 1.05, 100)
        y_fit = a * np.exp(b * x_fit)
        ax.plot(x_fit, y_fit, '--', color='red', linewidth=2, alpha=0.7,
                label=f'Fit: steps = {a:.1f} × exp({b:.2f} × f)')

    ax.set_xlabel('f = S/(CP) (Capacity Fraction)', fontsize=14)
    ax.set_ylabel('Steps to Saturation', fontsize=14)

    if is_multi_prime:
        ax.set_title(f'Saturation Time vs Capacity Fraction - Multiple Primes (C={C:.2f} bits/param)',
                    fontsize=16, pad=20)
    else:
        ax.set_title(f'Saturation Time vs Capacity Fraction (C={C:.2f} bits/param)',
                    fontsize=16, pad=20)

    ax.tick_params(axis='both', which='major', labelsize=12)

    # Use log y-scale only
    ax.set_yscale('log')

    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, which='both')

    # Add text box with stats
    if is_multi_prime:
        textstr = f'Overall fit: steps = {a:.1f} × exp({b:.2f} × f)\n'
        textstr += f'Exponent: {b:.2f}\n'
        textstr += f'R²: {r_squared:.3f}\n'
        textstr += f'C = {C:.2f} bits/param'
    else:
        textstr = f'Exponential: steps = {a:.1f} × exp({b:.2f} × f)\n'
        textstr += f'Exponent: {b:.2f}\n'
        textstr += f'R²: {r_squared:.3f}\n'
        textstr += f'C = {C:.2f} bits/param'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved capacity fraction plot: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return b, a, r_squared


def plot_saturation_steps_vs_params(
    all_results: Dict[int, List[Dict]],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot steps to saturation vs model parameter count for different dataset sizes.

    Args:
        all_results: Dict mapping dimension to list of result dicts
        save_path: Path to save the plot (optional)
        show: Whether to show the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Collect all unique dataset sizes across all dimensions
    all_dataset_sizes = set()
    for results in all_results.values():
        for result in results:
            all_dataset_sizes.add(result['n_samples'])

    # Sort dataset sizes for consistent ordering
    sorted_dataset_sizes = sorted(all_dataset_sizes)

    # Get colors from seaborn crest colormap
    colors = sns.color_palette("crest", n_colors=len(sorted_dataset_sizes))
    dataset_size_to_color = {size: colors[i] for i, size in enumerate(sorted_dataset_sizes)}

    # Group by dataset size
    dataset_size_groups = {}
    for dim, results in all_results.items():
        for result in results:
            n_samples = result['n_samples']
            if n_samples not in dataset_size_groups:
                dataset_size_groups[n_samples] = []
            dataset_size_groups[n_samples].append(result)

    # Plot each dataset size as a separate series
    for n_samples in sorted_dataset_sizes:
        if n_samples not in dataset_size_groups:
            continue

        results = dataset_size_groups[n_samples]
        # Sort by parameter count
        results = sorted(results, key=lambda x: x['param_count'])

        param_counts = [r['param_count'] for r in results]
        saturation_steps = [r['saturation_step'] for r in results]

        ax.plot(
            param_counts,
            saturation_steps,
            marker='o',
            markersize=8,
            linewidth=2,
            color=dataset_size_to_color[n_samples],
            label=f'{n_samples} samples'
        )

    ax.set_xlabel('Model Parameters', fontsize=14)
    ax.set_ylabel('Steps to Saturation', fontsize=14)
    ax.set_title('Steps to Saturation vs Model Size', fontsize=16, pad=20)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Use log scale for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Legend
    ax.legend(
        title='Dataset Size',
        loc='best',
        fontsize=10,
        title_fontsize=11
    )

    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved saturation steps vs params plot: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_rate_vs_dataset_size(
    all_results: Dict[int, List[Dict]],
    k: int = 10,
    save_path: Optional[str] = None,
    show: bool = True
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Plot dT/dS (rate of change of saturation time) vs dataset size S in bits.

    For each dimension, finds pairs of data points (n, n+k) and computes:
        dT/dS = (T(n+k) - T(n)) / (S(n+k) - S(n))
    where S is the dataset size in bits.

    Args:
        all_results: Dict mapping dimension to list of result dicts.
                     Each result dict should have 'n_samples', 'saturation_step',
                     and 'dataset_bits'.
        k: Delta in number of samples for rate estimation (default: 10)
        save_path: Path to save the plot (optional)
        show: Whether to show the plot

    Returns:
        Dict mapping dimension to list of (dataset_bits, dT_dS) tuples
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get colors from seaborn crest colormap
    dims = sorted(all_results.keys())
    colors = sns.color_palette("crest", n_colors=len(dims))
    dim_to_color = {dim: colors[i] for i, dim in enumerate(dims)}

    rate_data = {}

    for dim in dims:
        results = all_results[dim]

        # Build mappings from n_samples to saturation_step and dataset_bits
        samples_to_step = {r['n_samples']: r['saturation_step'] for r in results}
        samples_to_bits = {r['n_samples']: r['dataset_bits'] for r in results}

        # Find all pairs (n, n+k) and compute dT/dS
        rates = []
        for n_samples in sorted(samples_to_step.keys()):
            if (n_samples + k) in samples_to_step:
                T_n = samples_to_step[n_samples]
                T_n_k = samples_to_step[n_samples + k]
                S_n = samples_to_bits[n_samples]
                S_n_k = samples_to_bits[n_samples + k]
                delta_S = S_n_k - S_n
                if delta_S > 0:
                    dT_dS = (T_n_k - T_n) / delta_S
                    rates.append((S_n, dT_dS))

        if not rates:
            continue

        rate_data[dim] = rates

        # Extract data for plotting
        x_vals = [r[0] for r in rates]
        y_vals = [r[1] for r in rates]

        param_count = results[0]['param_count']
        ax.plot(
            x_vals,
            y_vals,
            marker='o',
            markersize=8,
            linewidth=2,
            color=dim_to_color[dim],
            label=f'dim={dim} ({param_count:,} params)'
        )

    ax.set_xlabel('Dataset Size S (bits)', fontsize=14)
    ax.set_ylabel('dT/dS (steps per bit)', fontsize=14)
    ax.set_title(f'Rate of Change of Saturation Time (k={k} samples)', fontsize=16, pad=20)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Add legend
    ax.legend(
        title='Model Size',
        loc='best',
        fontsize=10,
        title_fontsize=11
    )

    ax.grid(True, alpha=0.3)

    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved rate plot: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return rate_data
