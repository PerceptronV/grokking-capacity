"""
Single grokking experiment with model saving and loading utilities.

This script runs a single grokking experiment and saves the trained model
along with all hyperparameters needed to recreate it. It can also be imported
to use the load_model() function for hassle-free model loading.

Example usage:
    # Run a single experiment
    python single.py --dim 128 --depth 2 --heads 4 --epochs 200
    
    # Load the model in another script
    from single import load_model
    model, metadata = load_model('data/single/model.pt')
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import torch.optim as optim_torch

from models import TransformerTorch
from data import grokking_data_torch
from groks import GrokkingTrainer
from utils import save_model


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_results(train_acc, val_acc, args, param_count, save_dir):
    """Create and save a plot of the training results."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(train_acc, label='Training Accuracy', color='#1b9e77', linewidth=2, linestyle='-')
    ax.plot(val_acc, label='Validation Accuracy', color='#d95f02', linewidth=2, linestyle='--')
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title(f'Grokking Curve: dim={args.dim}, depth={args.depth}, heads={args.heads}\n'
                 f'{param_count:,} parameters', fontsize=16, pad=20)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylim([0, 105])
    
    # Add text annotation with final accuracies
    textstr = f'Final Train Acc: {train_acc[-1]:.1f}%\nFinal Val Acc: {val_acc[-1]:.1f}%'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, 'grokking_curve.pdf')
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()


def run_experiment(args):
    """Run a single grokking experiment."""
    print(f"\n{'='*60}")
    print(f"Running single grokking experiment")
    print(f"{'='*60}")
    
    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Prepare data
    print(f"\nPreparing data...")
    print(f"  Prime: {args.p}")
    print(f"  Operation: {args.op}")
    print(f"  Train fraction: {args.train_fraction}")
    print(f"  Split type: {args.split_type}")
    
    Xtrain_torch, Ttrain_torch, Xtest_torch, Ttest_torch = grokking_data_torch(
        args.p, op=args.op, split_type=args.split_type, 
        train_fraction=args.train_fraction, device='cpu')
    
    print(f"  Training samples: {len(Xtrain_torch)}")
    print(f"  Validation samples: {len(Xtest_torch)}")
    
    # Build model
    kwargs = {
        'depth': args.depth,
        'dim': args.dim,
        'heads': args.heads,
        'n_tokens': args.p + 2,
        'seq_len': 4,
        'dropout': args.dropout
    }
    
    device = 'cpu'
    if not args.cpu:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
    
    print(f"\nBuilding model...")
    print(f"  Architecture: depth={args.depth}, dim={args.dim}, heads={args.heads}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Device: {device}")
    
    model = TransformerTorch(**kwargs).to(device)
    param_count = count_parameters(model)
    print(f"  Parameters: {param_count:,}")
    
    # Setup optimizer
    optimizer = optim_torch.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )
    
    print(f"\nOptimizer settings:")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Betas: ({args.beta1}, {args.beta2})")
    
    # Setup trainer
    trainer = GrokkingTrainer(
        model=model,
        optimizer=optimizer,
        n_tokens=args.p + 2,
        batch_size=args.batch_size,
        device=device,
        baseline_model=None
    )
    
    # Set early stopping threshold (None if disabled)
    early_stop_thresh = None if args.no_early_stopping else args.early_stopping_threshold

    print(f"\nTraining settings:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Early stopping: {early_stop_thresh if early_stop_thresh is not None else 'Disabled'}")
    if args.patience >= 0:
        print(f"  Patience: {args.patience} epochs (min_delta: {args.min_delta})")
    else:
        print(f"  Patience: Disabled")

    # Train
    results = trainer.train(
        train_data=(Xtrain_torch, Ttrain_torch),
        val_data=(Xtest_torch, Ttest_torch),
        epochs=args.epochs,
        shuffle=True,
        early_stopping_threshold=early_stop_thresh,
        patience=args.patience,
        min_delta=args.min_delta
    )
    
    train_acc = results['train_acc'] * 100
    val_acc = results['val_acc'] * 100
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}")
    print(f"Final training accuracy: {train_acc[-1]:.2f}%")
    print(f"Final validation accuracy: {val_acc[-1]:.2f}%")
    
    # Print memorisation stats if available
    if 'mem_t_trace' in results and results['mem_t_trace'] is not None:
        print(f"Final M_T (total memorisation): {results['mem_t_trace'][-1]:.1f} bits")
    if 'mem_u_trace' in results and results['mem_u_trace'] is not None:
        print(f"Final M_U (unintended memorisation): {results['mem_u_trace'][-1]:.1f} bits")
    
    # Prepare metadata
    metadata = {
        # Model hyperparameters
        'depth': args.depth,
        'dim': args.dim,
        'heads': args.heads,
        'n_tokens': args.p + 2,
        'seq_len': 4,
        'dropout': args.dropout,
        'param_count': param_count,
        
        # Data parameters
        'p': args.p,
        'op': args.op,
        'train_fraction': args.train_fraction,
        'split_type': args.split_type,
        
        # Training parameters
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'beta1': args.beta1,
        'beta2': args.beta2,
        'batch_size': args.batch_size,
        'epochs': len(train_acc),  # Actual number of epochs trained
        'patience': args.patience,
        'min_delta': args.min_delta,
        'seed': args.seed,
        
        # Results
        'train_acc': train_acc,
        'val_acc': val_acc,
        'final_train_acc': float(train_acc[-1]),
        'final_val_acc': float(val_acc[-1]),
    }
    
    # Add memorisation data if available
    if 'train_log_probs' in results:
        metadata['train_log_probs'] = results['train_log_probs']
    if 'val_log_probs' in results:
        metadata['val_log_probs'] = results['val_log_probs']
    if 'mem_t_trace' in results and results['mem_t_trace'] is not None:
        metadata['mem_t_trace'] = results['mem_t_trace']
    if 'mem_u_trace' in results and results['mem_u_trace'] is not None:
        metadata['mem_u_trace'] = results['mem_u_trace']
    
    # Save model
    model_path = os.path.join(args.save_dir, 'model.pt')
    save_model(model, metadata, model_path)
    
    # Save plot
    plot_results(train_acc, val_acc, args, param_count, args.save_dir)
    
    # Save raw data as numpy file for easy access
    data_path = os.path.join(args.save_dir, 'results.npz')
    np.savez(data_path, **metadata)
    print(f"Results saved to: {data_path}")
    
    return model, metadata


def main():
    parser = argparse.ArgumentParser(
        description='Run a single grokking experiment and save the trained model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data args
    parser.add_argument('--p', type=int, default=97, 
                       help='Prime number (determines vocabulary size)')
    parser.add_argument('--op', type=str, default='/', 
                       help='Arithmetic operation', 
                       choices=['*', '/', '+', '-'])
    parser.add_argument('--train-fraction', type=float, default=0.5, 
                       help='Fraction of data to use for training')
    parser.add_argument('--split-type', type=str, default='sequential', 
                       help='Type of train/test split', 
                       choices=['random', 'sequential', 'alternating'])
    
    # Model args
    parser.add_argument('--depth', type=int, default=2, 
                       help='Number of transformer layers')
    parser.add_argument('--dim', type=int, default=128, 
                       help='Model dimension')
    parser.add_argument('--heads', type=int, default=1, 
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2, 
                       help='Dropout rate')
    
    # Optimizer args
    parser.add_argument('--lr', type=float, default=1e-3, 
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1, 
                       help='Weight decay for AdamW')
    parser.add_argument('--beta1', type=float, default=0.9, 
                       help='Adam beta1 parameter')
    parser.add_argument('--beta2', type=float, default=0.98, 
                       help='Adam beta2 parameter')
    
    # Training args
    parser.add_argument('-b', '--batch-size', type=int, default=512,
                       help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=200,
                       help='Maximum number of training epochs')
    parser.add_argument('--early-stopping-threshold', type=float, default=1.0,
                       help='Stop training when validation accuracy reaches this threshold')
    parser.add_argument('--no-early-stopping', action='store_true',
                       help='Disable early stopping and train for full epochs')
    parser.add_argument('--patience', type=int, default=-1,
                       help='Patience for early stopping (epochs without improvement). If negative, never stop early due to patience')
    parser.add_argument('--min-delta', type=float, default=1e-4,
                       help='Minimum validation accuracy improvement to reset patience')
    
    # Save/load args
    parser.add_argument('--save-dir', type=str, default='data/single', 
                       help='Directory to save model and results')
    
    # Misc args
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    parser.add_argument('--cpu', action='store_true', 
                       help='Force CPU usage (disable GPU/MPS)')
    
    args = parser.parse_args()
    
    # Run experiment
    signature = f'p{args.p}_seed{args.seed}_split{args.split_type}'
    run_name = f'dim{args.dim}_depth{args.depth}_heads{args.heads}'
    args.save_dir = os.path.join(args.save_dir, signature, run_name)

    os.makedirs(args.save_dir, exist_ok=True)
    model, metadata = run_experiment(args)
    
    print(f"\n{'='*60}")
    print(f"Experiment complete!")
    print(f"{'='*60}")
    print(f"Model and results saved to: {args.save_dir}")
    print(f"\nTo load this model in another script:")
    print(f"  from single import load_model")
    print(f"  model, metadata = load_model('{os.path.join(args.save_dir, 'model.pt')}')")


if __name__ == '__main__':
    main()
