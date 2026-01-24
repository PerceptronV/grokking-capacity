"""
Measure the learning speed of different model architectures on random data.

Learning speed is defined as the number of training steps required to saturate
model memory (reach near-perfect accuracy on random data). By comparing
saturation steps across different dataset sizes and model architectures,
we can estimate learning speed in steps per bit.
"""

import argparse
import numpy as np
import os
from tqdm import tqdm
from typing import Tuple, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import TransformerTorch
from data import random_target_data_torch

import matplotlib.pyplot as plt
from plotting import (
    plot_learning_speed_curves,
    plot_speed_vs_model_size,
    plot_combined_speed_analysis
)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SpeedTrainer:
    """
    Trainer for measuring learning speed (steps to saturation).
    
    Trains until accuracy reaches a saturation threshold and counts
    the total number of training steps.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        batch_size: int = 512,
        device: str = 'cpu'
    ):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.device = device
        
        # Traces
        self.train_loss_trace = []
        self.train_acc_trace = []
        self.steps_trace = []  # Track cumulative steps at each epoch
    
    def _make_batches(self, X: torch.Tensor, T: torch.Tensor):
        """Yield batches from data."""
        bs = self.batch_size if self.batch_size != -1 else X.shape[0]
        for i in range(0, X.shape[0], bs):
            yield X[i:i+bs], T[i:i+bs]
    
    def train(
        self,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        max_epochs: int = 10000,
        saturation_threshold: float = 99.5,
        patience: int = 0,
        verbose: bool = True
    ) -> Dict:
        """
        Train until accuracy saturates (reaches threshold).
        
        Args:
            train_data: Tuple of (X, T) tensors
            max_epochs: Maximum number of epochs
            saturation_threshold: Accuracy threshold to consider saturated (%)
            patience: Number of epochs to confirm saturation
            verbose: Whether to show progress bar
        
        Returns:
            Dictionary with training results including steps to saturation
        """
        X_train, T_train = train_data
        n_samples = X_train.shape[0]
        
        # Calculate number of steps per epoch
        bs = self.batch_size if self.batch_size != -1 else n_samples
        steps_per_epoch = (n_samples + bs - 1) // bs
        
        total_steps = 0
        saturation_step = None
        epochs_above_threshold = 0
        
        epoch_iter = tqdm(range(max_epochs), desc='Training', unit='epoch') if verbose else range(max_epochs)
        
        for epoch in epoch_iter:
            self.model.train()
            
            # Shuffle data
            perm = torch.randperm(n_samples)
            X_shuffled = X_train[perm]
            T_shuffled = T_train[perm]
            
            total_loss = 0.0
            total_correct = 0
            
            for Xb, Tb in self._make_batches(X_shuffled, T_shuffled):
                Xb = Xb.to(self.device)
                Tb = Tb.to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(Xb)
                loss = F.cross_entropy(logits, Tb)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * Xb.size(0)
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == Tb).sum().item()
                
                total_steps += 1
            
            avg_loss = total_loss / n_samples
            avg_acc = (total_correct / n_samples) * 100  # Convert to percentage
            
            self.train_loss_trace.append(avg_loss)
            self.train_acc_trace.append(avg_acc)
            self.steps_trace.append(total_steps)
            
            if verbose:
                epoch_iter.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{avg_acc:.2f}%',
                    'steps': total_steps
                })
            
            # Check for saturation
            if avg_acc >= saturation_threshold:
                epochs_above_threshold += 1
                if saturation_step is None:
                    saturation_step = total_steps
                
                # Confirm saturation after patience epochs
                if epochs_above_threshold >= patience:
                    if verbose:
                        print(f"\nSaturation reached at step {saturation_step} "
                              f"(epoch {epoch + 1 - patience}), acc={avg_acc:.2f}%")
                    break
            else:
                epochs_above_threshold = 0
                saturation_step = None
        
        # If we didn't reach saturation, use final values
        if saturation_step is None:
            saturation_step = total_steps
            if verbose:
                print(f"\nDid not reach saturation threshold ({saturation_threshold}%). "
                      f"Final acc: {avg_acc:.2f}%")
        
        return {
            'epochs_trained': epoch + 1,
            'total_steps': total_steps,
            'saturation_step': saturation_step,
            'final_loss': avg_loss,
            'final_acc': avg_acc,
            'train_loss_trace': np.array(self.train_loss_trace),
            'train_acc_trace': np.array(self.train_acc_trace),
            'steps_trace': np.array(self.steps_trace),
            'saturated': avg_acc >= saturation_threshold
        }


def run_speed_experiment(
    n_samples: int,
    dim: int,
    depth: int,
    heads: int,
    p: int,
    max_epochs: int,
    saturation_threshold: float,
    patience: int,
    args,
    verbose: bool = True
) -> Dict:
    """
    Run a single speed experiment with given dataset size and model config.
    
    Returns:
        Dictionary with experiment results
    """
    n_tokens = p + 2  # Full vocabulary size (p digits + operator + equals)
    
    # Generate random target data
    X_train, T_train = random_target_data_torch(n_samples, p, seq_len=4, device='cpu')
    
    # Build model
    model_kwargs = {
        'depth': depth,
        'dim': dim,
        'heads': heads,
        'n_tokens': n_tokens,
        'seq_len': 4,
        'dropout': args.dropout
    }
    
    # Select device
    if args.device is not None:
        device = args.device
    elif args.cpu:
        device = 'cpu'
    else:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    model = TransformerTorch(**model_kwargs).to(device)
    param_count = count_parameters(model)
    
    if verbose:
        print(f"  Dataset size: {n_samples}, Model dim: {dim}, Parameters: {param_count:,}")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )
    
    trainer = SpeedTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=args.batch_size,
        device=device
    )
    
    results = trainer.train(
        (X_train, T_train),
        max_epochs=max_epochs,
        saturation_threshold=saturation_threshold,
        patience=patience,
        verbose=verbose
    )
    
    # Add experiment metadata
    results['n_samples'] = n_samples
    results['dim'] = dim
    results['depth'] = depth
    results['heads'] = heads
    results['param_count'] = param_count
    results['p'] = p
    
    # Compute dataset bits
    bits_per_example = np.log2(n_tokens)
    results['dataset_bits'] = n_samples * bits_per_example
    results['bits_per_example'] = bits_per_example
    
    # Compute saturation_epoch from saturation_step
    bs = args.batch_size if args.batch_size != -1 else n_samples
    steps_per_epoch = (n_samples + bs - 1) // bs
    results['saturation_epoch'] = results['saturation_step'] / steps_per_epoch
    
    return results


def save_results(results: Dict, args):
    """Save individual experiment results."""
    os.makedirs(args.data_dir, exist_ok=True)
    
    fname = os.path.join(
        args.data_dir,
        f'speed_dim{results["dim"]}_samples{results["n_samples"]}.npz'
    )
    
    np.savez(
        fname,
        n_samples=results['n_samples'],
        dim=results['dim'],
        depth=results['depth'],
        heads=results['heads'],
        param_count=results['param_count'],
        p=results['p'],
        epochs_trained=results['epochs_trained'],
        total_steps=results['total_steps'],
        saturation_step=results['saturation_step'],
        saturation_epoch=results['saturation_epoch'],
        final_loss=results['final_loss'],
        final_acc=results['final_acc'],
        dataset_bits=results['dataset_bits'],
        bits_per_example=results['bits_per_example'],
        saturated=results['saturated'],
        train_loss_trace=results['train_loss_trace'],
        train_acc_trace=results['train_acc_trace'],
        steps_trace=results['steps_trace']
    )
    
    return fname


def load_or_run_experiment(
    n_samples: int,
    dim: int,
    args,
    force: bool = False,
    verbose: bool = True
) -> Dict:
    """Load existing results or run a new experiment."""
    fname = os.path.join(
        args.data_dir,
        f'speed_dim{dim}_samples{n_samples}.npz'
    )
    
    if os.path.exists(fname) and not force:
        if verbose:
            print(f"  Loading existing results: {fname}")
        data = np.load(fname)
        result = {key: data[key].item() if data[key].ndim == 0 else data[key] 
                  for key in data.files}
        # Compute saturation_epoch if not present (for old files)
        if 'saturation_epoch' not in result:
            n_samples = int(result['n_samples'])
            bs = args.batch_size if args.batch_size != -1 else n_samples
            steps_per_epoch = (n_samples + bs - 1) // bs
            result['saturation_epoch'] = result['saturation_step'] / steps_per_epoch
        return result
    
    # Run experiment
    result = run_speed_experiment(
        n_samples=n_samples,
        dim=dim,
        depth=args.depth,
        heads=args.heads,
        p=args.p,
        max_epochs=args.epochs,
        saturation_threshold=args.saturation_threshold,
        patience=args.patience,
        args=args,
        verbose=verbose
    )
    
    # Save results
    save_results(result, args)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Measure learning speed (steps to memorisation) for different architectures'
    )
    
    # Data args
    parser.add_argument('--p', type=int, default=97, 
                        help='Prime number (determines vocabulary size)')
    
    # Model args
    parser.add_argument('--depth', type=int, default=2, help='Transformer depth')
    parser.add_argument('--heads', type=int, default=1, help='Attention heads')
    parser.add_argument('--dropout', type=float, default=0.2, 
                        help='Dropout (0.2 to match training)')
    parser.add_argument('--dims', type=int, nargs='+', 
                        default=[20, 24, 28],
                        help='List of model dimensions to test')
    
    # Dataset size args
    parser.add_argument('--samples-start', type=int, default=100,
                        help='Starting dataset size')
    parser.add_argument('--samples-end', type=int, default=1000,
                        help='Ending dataset size')
    parser.add_argument('--samples-steps', type=int, default=4,
                        help='Number of dataset sizes to test (log spaced)')
    
    # Optimizer args
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, 
                        help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.98, help='Adam beta2')
    
    # Training args
    parser.add_argument('-b', '--batch-size', type=int, default=512, 
                        help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=5000, 
                        help='Maximum epochs')
    parser.add_argument('--saturation-threshold', type=float, default=99.0,
                        help='Accuracy threshold to consider saturated (%%)')
    parser.add_argument('--patience', type=int, default=0,
                        help='Epochs to confirm saturation')
    
    # Output args
    parser.add_argument('--data-dir', type=str, default='data/speed',
                        help='Data output directory')
    parser.add_argument('--plot-dir', type=str, default='media/speed',
                        help='Plot output directory')
    
    # Misc args
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cpu', action='store_true', help='Force CPU only')
    parser.add_argument('--device', type=str, default=None, 
                        help='Device to use (e.g., "cuda:0", "cpu", "mps")')
    parser.add_argument('--force', action='store_true',
                        help='Force re-run even if results exist')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots (just save)')
    parser.add_argument('--rate', action='store_true',
                        help='Generate additional runs at n+k samples for rate estimation (dT/dS)')
    parser.add_argument('--rate-k', type=int, default=50,
                        help='Delta for rate estimation: measure at n and n+k samples (default: 10)')
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directories with signature
    signature = f'p{args.p}_seed{args.seed}'
    args.data_dir = os.path.join(args.data_dir, signature)
    args.plot_dir = os.path.join(args.plot_dir, signature)
    
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    
    # Generate log-spaced dataset sizes
    dataset_sizes = np.logspace(
        np.log10(args.samples_start),
        np.log10(args.samples_end),
        args.samples_steps
    ).astype(int)
    dataset_sizes = np.unique(dataset_sizes)  # Remove duplicates

    # If --rate is set, add n+k sizes for each n
    if args.rate:
        rate_sizes = dataset_sizes + args.rate_k
        dataset_sizes = np.unique(np.concatenate([dataset_sizes, rate_sizes]))
    
    n_tokens = args.p + 2
    bits_per_example = np.log2(n_tokens)
    
    print(f"Model dimensions: {args.dims}")
    print(f"Dataset sizes: {list(dataset_sizes)}")
    if args.rate:
        print(f"Rate estimation: enabled (k={args.rate_k})")
    print(f"p: {args.p}")
    print(f"Full vocab size: {n_tokens}")
    print(f"Bits per example: {bits_per_example:.2f}")
    print(f"Saturation threshold: {args.saturation_threshold}%")
    print()
    
    # Run experiments
    all_results = {}
    
    for dim in args.dims:
        print(f"\n{'='*60}")
        print(f"Model dimension: {dim}")
        print(f"{'='*60}")
        
        all_results[dim] = []
        
        for n_samples in dataset_sizes:
            print(f"\n  Dataset size: {n_samples}")
            
            result = load_or_run_experiment(
                n_samples=int(n_samples),
                dim=dim,
                args=args,
                force=args.force,
                verbose=True
            )
            
            all_results[dim].append(result)
            
            dataset_bits = n_samples * bits_per_example
            steps_per_bit = result['saturation_step'] / dataset_bits if dataset_bits > 0 else 0
            
            print(f"    Final accuracy: {result['final_acc']:.2f}%")
            print(f"    Saturation steps: {result['saturation_step']:,}")
            print(f"    Dataset bits: {dataset_bits:.0f}")
            print(f"    Steps per bit: {steps_per_bit:.2f}")
    
    # Generate plots
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    # Main learning speed curves plot
    curves_path = os.path.join(args.plot_dir, 'learning_speed_curves.pdf')
    speed_estimates = plot_learning_speed_curves(
        all_results,
        p=args.p,
        save_path=curves_path,
        show=not args.no_show
    )
    
    # Speed vs model size plot
    if speed_estimates:
        speed_path = os.path.join(args.plot_dir, 'speed_vs_model_size.pdf')
        b, log_a, r_squared = plot_speed_vs_model_size(
            speed_estimates,
            save_path=speed_path,
            show=not args.no_show
        )
    
    # Combined analysis plot
    combined_path = os.path.join(args.plot_dir, 'speed_analysis_combined.pdf')
    plot_combined_speed_analysis(
        all_results,
        p=args.p,
        save_path=combined_path,
        show=not args.no_show
    )
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
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
    
    if speed_estimates:
        print("\n" + "="*60)
        print("LEARNING SPEED ESTIMATES (steps per bit)")
        print("="*60)
        
        for param_count in sorted(speed_estimates.keys()):
            slope, intercept, dim = speed_estimates[param_count]
            print(f"dim={dim:3d}: {param_count:8,} params, "
                  f"slope={slope:.2f} steps/bit, intercept={intercept:.0f} steps")
        
        if len(speed_estimates) >= 2:
            print("\n" + "="*60)
            print("SPEED SCALING")
            print("="*60)
            print(f"Power law exponent: {b:.3f}")
            print(f"R²: {r_squared:.3f}")
            if b < 0:
                print(f"Larger models learn FASTER (fewer steps per bit)")
            else:
                print(f"Larger models learn SLOWER (more steps per bit)")


if __name__ == '__main__':
    main()

