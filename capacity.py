"""
Measure the information capacity C of a model architecture.

C is defined as the number of bits the model can memorize about its training set
per parameter. Following Morris et al. "How Much Do Language Models Memorize?",
we empirically determine C using information theory:

1. Generate datasets with random uniform target outputs
2. Train models to saturation (memorisation)
3. Measure bits memorized = log_2(N) - L, where N is vocab size and L is avg log prob
4. Find C as the slope of saturation memorisation vs. number of parameters
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
from data import random_target_data_torch, grokking_data_torch
from plotting import plot_capacity_curves, plot_capacity_estimation, estimate_capacity
from experiment import ExperimentConfig, save_run
from utils import get_device


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CapacityTrainer:
    """
    Trainer for measuring model memorisation capacity.
    
    Trains until loss saturates (stops decreasing) and measures
    the bits memorized about the training set.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        n_tokens: int,
        batch_size: int = 512,
        device: str = 'cpu'
    ):
        self.model = model
        self.optimizer = optimizer
        self.n_tokens = n_tokens  # Vocabulary size for computing bits
        self.batch_size = batch_size
        self.device = device
        
        # Traces
        self.train_loss_trace = []
        self.train_acc_trace = []
        self.bits_per_example_trace = []
    
    def _make_batches(self, X: torch.Tensor, T: torch.Tensor):
        """Yield batches from data."""
        bs = self.batch_size if self.batch_size != -1 else X.shape[0]
        for i in range(0, X.shape[0], bs):
            yield X[i:i+bs], T[i:i+bs]
    
    def compute_memorization(self, X: torch.Tensor, T: torch.Tensor) -> Tuple[float, float, float]:
        """
        Compute memorisation metrics for the dataset.
        
        Returns:
            avg_loss: Average cross-entropy loss (in nats)
            avg_log_prob: Average log probability (in bits, log base 2)
            bits_per_example: Bits memorized per example = log2(N) - avg_log_prob
        """
        self.model.eval()
        total_log_prob = 0.0
        total_loss = 0.0
        n_samples = X.shape[0]
        
        with torch.no_grad():
            for Xb, Tb in self._make_batches(X, T):
                Xb = Xb.to(self.device)
                Tb = Tb.to(self.device)
                
                logits = self.model(Xb)
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Get log probability of correct class
                correct_log_probs = log_probs.gather(1, Tb.unsqueeze(1)).squeeze(1)
                
                # Convert from nats to bits (divide by ln(2))
                correct_log_probs_bits = correct_log_probs / np.log(2)
                
                total_log_prob += correct_log_probs_bits.sum().item()
                total_loss += F.cross_entropy(logits, Tb, reduction='sum').item()
        
        avg_loss = total_loss / n_samples
        avg_log_prob = total_log_prob / n_samples  # This is negative (log of probability < 1)
        
        # Maximum entropy (uniform random guessing) in bits
        max_entropy = np.log2(self.n_tokens)
        
        # Bits memorized = max_entropy - (-avg_log_prob) = max_entropy + avg_log_prob
        # Note: avg_log_prob is negative, so we add it
        bits_per_example = max_entropy + avg_log_prob
        
        return avg_loss, avg_log_prob, bits_per_example
    
    def train(
        self,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        max_epochs: int = 10000,
        patience: int = 50,
        min_delta: float = 1e-4,
        verbose: bool = True
    ) -> Dict:
        """
        Train until loss saturates (early stopping based on loss plateau).
        
        Args:
            train_data: Tuple of (X, T) tensors
            max_epochs: Maximum number of epochs
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change in loss to qualify as an improvement
            verbose: Whether to show progress bar
        
        Returns:
            Dictionary with training results and memorisation metrics
        """
        X_train, T_train = train_data
        n_samples = X_train.shape[0]
        
        best_loss = float('inf')
        epochs_without_improvement = 0
        
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
            
            avg_loss = total_loss / n_samples
            avg_acc = total_correct / n_samples
            
            self.train_loss_trace.append(avg_loss)
            self.train_acc_trace.append(avg_acc)
            
            # Compute bits memorized
            _, _, bits = self.compute_memorization(X_train, T_train)
            self.bits_per_example_trace.append(bits)
            
            if verbose:
                epoch_iter.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{avg_acc:.3f}',
                    'bits': f'{bits:.2f}'
                })
            
            # Early stopping check based on loss saturation
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"\nEarly stopping: loss saturated at {avg_loss:.4f} after {epoch + 1} epochs")
                break
        
        # Final memorisation measurement
        final_loss, final_log_prob, final_bits = self.compute_memorization(X_train, T_train)
        total_bits = final_bits * n_samples
        
        return {
            'epochs_trained': epoch + 1,
            'final_loss': final_loss,
            'final_acc': self.train_acc_trace[-1],
            'final_bits_per_example': final_bits,
            'total_bits_memorized': total_bits,
            'train_loss_trace': np.array(self.train_loss_trace),
            'train_acc_trace': np.array(self.train_acc_trace),
            'bits_trace': np.array(self.bits_per_example_trace)
        }


def get_dataset(
    n_samples: int,
    p: int,
    dataset_type: str = 'random',
    seq_len: int = 4,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get a dataset of random target data.
    """
    if dataset_type == 'random':
        return random_target_data_torch(n_samples, p, seq_len=seq_len, device=device)
    elif dataset_type in ('+', '-', '*', '/'):
        size = p * (p - 1) if dataset_type == '/' else p * p
        if n_samples > size:
            raise ValueError(f"Required # samples: {n_samples} is greater than the available size of the dataset: {size}")
        train_fraction = n_samples / size
        return grokking_data_torch(
            p, op=dataset_type, split_type='random',
            train_fraction=train_fraction, device=device
        )
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

def run_capacity_experiment(
    n_samples: int,
    dim: int,
    depth: int,
    heads: int,
    p: int,
    max_epochs: int,
    patience: int,
    args,
    verbose: bool = True,
    dataset_type: str = 'random'
) -> Dict:
    """
    Run a single capacity experiment with given dataset size and model config.
    
    Returns:
        Dictionary with experiment results
    """
    n_tokens = p + 2  # Full vocabulary size (p digits + operator + equals)
    
    # Generate random target data
    X_train, T_train = get_dataset(n_samples, p, dataset_type=dataset_type, seq_len=4, device='cpu')
    
    # Build model
    model_kwargs = {
        'depth': depth,
        'dim': dim,
        'heads': heads,
        'n_tokens': p + 2,  # Full vocabulary including op tokens
        'seq_len': 4,
        'dropout': args.dropout
    }
    
    device = get_device(args.device, args.cpu)
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
    
    trainer = CapacityTrainer(
        model=model,
        optimizer=optimizer,
        n_tokens=n_tokens,
        batch_size=args.batch_size,
        device=device
    )
    
    results = trainer.train(
        (X_train, T_train),
        max_epochs=max_epochs,
        patience=patience,
        min_delta=args.min_delta,
        verbose=verbose
    )
    
    results['n_samples'] = n_samples
    results['dim'] = dim
    results['depth'] = depth
    results['heads'] = heads
    results['param_count'] = param_count
    
    return results


def save_results(results: Dict, args, signature: str):
    """Save individual experiment results."""
    os.makedirs(args.data_dir, exist_ok=True)
    
    fname = os.path.join(
        args.data_dir,
        f'capacity_dim{results["dim"]}_depth{results["depth"]}_heads{results["heads"]}_wd{args.weight_decay}_samples{results["n_samples"]}.npz'
    )

    np.savez(
        fname,
        n_samples=results['n_samples'],
        dim=results['dim'],
        depth=results['depth'],
        heads=results['heads'],
        param_count=results['param_count'],
        epochs_trained=results['epochs_trained'],
        final_loss=results['final_loss'],
        final_acc=results['final_acc'],
        final_bits_per_example=results['final_bits_per_example'],
        total_bits_memorized=results['total_bits_memorized'],
        train_loss_trace=results['train_loss_trace'],
        train_acc_trace=results['train_acc_trace'],
        bits_trace=results['bits_trace']
    )

    config = ExperimentConfig(
        experiment_type="capacity",
        p=args.p,
        operation=args.dataset_type,
        n_samples=int(results['n_samples']),
        dim=int(results['dim']),
        depth=int(results['depth']),
        heads=int(results['heads']),
        dropout=args.dropout,
        param_count=int(results['param_count']),
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        seed=args.seed,
    )
    results_summary = {
        "final_acc": float(results['final_acc']),
        "final_loss": float(results['final_loss']),
        "total_bits_memorized": float(results['total_bits_memorized']),
        "epochs_trained": int(results['epochs_trained']),
    }
    save_run(config, results_summary, fname)

    return fname


def load_or_run_experiment(
    n_samples: int,
    dim: int,
    args,
    force: bool = False,
    verbose: bool = True,
    dataset_type: str = 'random'
) -> Dict:
    """Load existing results or run a new experiment."""
    fname = os.path.join(
        args.data_dir,
        f'capacity_dim{dim}_depth{args.depth}_heads{args.heads}_wd{args.weight_decay}_samples{n_samples}.npz'
    )

    if os.path.exists(fname) and not force:
        if verbose:
            print(f"  Loading existing results: {fname}")
        data = np.load(fname)
        return {key: data[key].item() if data[key].ndim == 0 else data[key] 
                for key in data.files}
    
    # Run experiment
    result = run_capacity_experiment(
        n_samples=n_samples,
        dim=dim,
        depth=args.depth,
        heads=args.heads,
        p=args.p,
        max_epochs=args.epochs,
        patience=args.patience,
        args=args,
        verbose=verbose
    )
    
    # Save results
    save_results(result, args, '')
    
    return result


# Plotting functions are now in plotting.py and imported at the top


def main():
    parser = argparse.ArgumentParser(
        description='Measure model information capacity (bits per parameter)'
    )
    
    # Data args
    parser.add_argument('--p', type=int, default=97, 
                        help='Prime number')
    
    # Model args
    parser.add_argument('--depth', type=int, default=2, help='Transformer depth')
    parser.add_argument('--heads', type=int, default=1, help='Attention heads')
    parser.add_argument('--dropout', type=float, default=0.0, 
                        help='Dropout (typically 0 for memorisation)')
    parser.add_argument('--dims', type=int, nargs='+', 
                        default=[10, 12, 14, 16, 18, 20, 22],
                        help='List of model dimensions to test')
    
    # Dataset size args
    parser.add_argument('--samples-start', type=int, default=1000,
                        help='Starting dataset size')
    parser.add_argument('--samples-end', type=int, default=9300,
                        help='Ending dataset size')
    parser.add_argument('--samples-steps', type=int, default=8,
                        help='Number of dataset sizes to test (log spaced)')
    parser.add_argument('--dataset-type', type=str, default='random',
                        help='Type of dataset to use',
                        choices=['random', '+', '-', '*', '/'])
    
    # Optimizer args
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, 
                        help='Weight decay (lower for memorisation)')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.98, help='Adam beta2')
    
    # Training args
    parser.add_argument('-b', '--batch-size', type=int, default=512, 
                        help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=5000, 
                        help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=100,
                        help='Patience for early stopping (epochs without improvement)')
    parser.add_argument('--min-delta', type=float, default=1e-4,
                        help='Minimum loss improvement to reset patience')
    
    # Output args
    parser.add_argument('--data-dir', type=str, default='data/capacity',
                        help='Data output directory')
    parser.add_argument('--plot-dir', type=str, default='media/capacity',
                        help='Plot output directory')
    
    # Misc args
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cpu', action='store_true', help='Force CPU only')
    parser.add_argument('--device', type=str, default=None, 
                        help='device to use (e.g., "cuda:0", "cuda:1", "cpu", "mps"). Overrides --cpu flag if specified.')
    parser.add_argument('--force', action='store_true', 
                        help='Force re-run even if results exist')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots (just save)')
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directories
    symb_map = {
        'random': 'random',
        '+': 'add',
        '-': 'sub',
        '*': 'mul',
        '/': 'div'
    }
    signature = f'p{args.p}_op_{symb_map[args.dataset_type]}_seed{args.seed}'
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
    
    print(f"Model dimensions: {args.dims}")
    print(f"Dataset sizes: {list(dataset_sizes)}")
    print(f"p: {args.p}")
    print(f"Full vocab size: {args.p + 2}")
    print(f"Max bits per example: {np.log2(args.p + 2):.2f}")
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
                verbose=True,
                dataset_type=args.dataset_type
            )
            
            all_results[dim].append(result)
            
            print(f"    Final accuracy: {result['final_acc']:.3f}")
            print(f"    Bits per example: {result['final_bits_per_example']:.2f}")
            print(f"    Total bits memorized: {result['total_bits_memorized']:.0f}")
    
    # Plot results
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    # Main capacity curves plot
    curves_path = os.path.join(args.plot_dir, 'capacity_curves.pdf')
    saturation_points = plot_capacity_curves(
        all_results,
        p=args.p,
        save_path=curves_path,
        show=not args.no_show
    )
    
    # Capacity estimation plot
    estimation_path = os.path.join(args.plot_dir, 'capacity_estimation.pdf')
    C, intercept, r_squared = plot_capacity_estimation(
        saturation_points,
        save_path=estimation_path,
        show=not args.no_show
    )
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for dim in sorted(all_results.keys()):
        results = all_results[dim]
        param_count = results[0]['param_count']
        max_bits = max(r['total_bits_memorized'] for r in results)
        print(f"dim={dim:3d}: {param_count:8,} params, "
              f"max bits memorized: {max_bits:,.0f}, "
              f"bits/param: {max_bits/param_count:.2f}")
    
    print("="*60)
    print("CAPACITY ESTIMATION")
    print("="*60)
    sign = '+' if intercept >= 0 else '−'
    print(f"Linear fit: bits = {C:.2f} × params {sign} {abs(intercept):.0f}")
    print(f"Capacity C: {C:.2f} bits/parameter")
    print(f"Intercept: {intercept:.0f} bits")
    print(f"R²: {r_squared:.3f}")


if __name__ == '__main__':
    main()
