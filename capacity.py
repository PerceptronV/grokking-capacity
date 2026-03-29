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
from cli_args import add_model_args, add_optimizer_args, add_device_args, add_io_args
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
        'dropout': args.dropout,
        'init_scale': args.init_scale,
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


def save_results(results: Dict, args):
    """Save individual experiment results."""
    os.makedirs(args.data_dir, exist_ok=True)
    
    _is_suffix = f'_is{args.init_scale}' if args.init_scale != 1.0 else ''
    _do_suffix = f'_do{args.dropout}' if args.dropout != 0.0 else ''
    fname = os.path.join(
        args.data_dir,
        f'capacity_dim{results["dim"]}_depth{results["depth"]}_heads{results["heads"]}_wd{args.weight_decay}_samples{results["n_samples"]}{_is_suffix}{_do_suffix}.npz'
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
        init_scale=args.init_scale,
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
    _is_suffix = f'_is{args.init_scale}' if args.init_scale != 1.0 else ''
    _do_suffix = f'_do{args.dropout}' if args.dropout != 0.0 else ''
    fname = os.path.join(
        args.data_dir,
        f'capacity_dim{dim}_depth{args.depth}_heads{args.heads}_wd{args.weight_decay}_samples{n_samples}{_is_suffix}{_do_suffix}.npz'
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
    save_results(result, args)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Measure model capacity for a single (dim, n_samples) pair'
    )

    add_model_args(parser, dropout_default=0.0)
    add_optimizer_args(parser, weight_decay_default=0.01, epochs_default=5000)
    add_device_args(parser)
    add_io_args(parser, data_dir='data/capacity')

    # Dataset args
    parser.add_argument('--n-samples', type=int, required=True,
                        help='Dataset size')
    parser.add_argument('--dataset-type', type=str, default='random',
                        help='Type of dataset to use',
                        choices=['random', '+', '-', '*', '/'])

    # Training args
    parser.add_argument('--patience', type=int, default=100,
                        help='Patience for early stopping (epochs without improvement)')
    parser.add_argument('--min-delta', type=float, default=1e-4,
                        help='Minimum loss improvement to reset patience')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    symb_map = {'random': 'random', '+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}
    signature = f'p{args.p}_op_{symb_map[args.dataset_type]}_seed{args.seed}'
    args.data_dir = os.path.join(args.data_dir, signature)
    os.makedirs(args.data_dir, exist_ok=True)

    print(f"=== capacity  p={args.p} type={args.dataset_type} seed={args.seed} | "
          f"dim={args.dim} n={args.n_samples} depth={args.depth} heads={args.heads} "
          f"dropout={args.dropout} wd={args.weight_decay} lr={args.lr} ===")

    result = load_or_run_experiment(
        n_samples=args.n_samples,
        dim=args.dim,
        args=args,
        force=args.force,
        verbose=True,
        dataset_type=args.dataset_type
    )

    print(f"  final acc: {result['final_acc']:.3f}, "
          f"bits/example: {result['final_bits_per_example']:.2f}, "
          f"total bits: {result['total_bits_memorized']:.0f}")


if __name__ == '__main__':
    main()
