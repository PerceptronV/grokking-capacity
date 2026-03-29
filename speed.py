"""
Measure the learning speed of different model architectures on random data.

Learning speed is defined as the number of training steps required to saturate
model memory (reach near-perfect accuracy on random data). By comparing
saturation steps across different dataset sizes and model architectures,
we can estimate learning speed in steps per bit.
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
from data import random_target_data_torch
from experiment import ExperimentConfig, save_run
from utils import get_device



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
    
    _is_suffix = f'_is{args.init_scale}' if args.init_scale != 1.0 else ''
    _do_suffix = f'_do{args.dropout}' if args.dropout != 0.2 else ''
    fname = os.path.join(
        args.data_dir,
        f'speed_dim{results["dim"]}_depth{args.depth}_heads{args.heads}_wd{args.weight_decay}_samples{results["n_samples"]}{_is_suffix}{_do_suffix}.npz'
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

    config = ExperimentConfig(
        experiment_type="speed",
        p=int(results['p']),
        operation=args.operation,
        train_fraction=args.train_fraction,
        split_type=args.split_type,
        n_samples=int(results['n_samples']),
        dataset_bits=float(results['dataset_bits']),
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
        saturation_threshold=args.saturation_threshold,
    )
    results_summary = {
        "saturated": bool(results['saturated']),
        "saturation_epoch": float(results['saturation_epoch']),
        "final_acc": float(results['final_acc']),
        "final_loss": float(results['final_loss']),
    }
    save_run(config, results_summary, fname)

    return fname


def load_or_run_experiment(
    n_samples: int,
    dim: int,
    args,
    force: bool = False,
    verbose: bool = True
) -> Dict:
    """Load existing results or run a new experiment."""
    _is_suffix = f'_is{args.init_scale}' if args.init_scale != 1.0 else ''
    _do_suffix = f'_do{args.dropout}' if args.dropout != 0.2 else ''
    fname = os.path.join(
        args.data_dir,
        f'speed_dim{dim}_depth{args.depth}_heads{args.heads}_wd{args.weight_decay}_samples{n_samples}{_is_suffix}{_do_suffix}.npz'
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
        description='Measure learning speed for a single (dim, n_samples) pair'
    )

    add_model_args(parser, dropout_default=0.2)
    add_optimizer_args(parser, weight_decay_default=0.01, epochs_default=5000)
    add_device_args(parser)
    add_io_args(parser, data_dir='data/speed')

    # Task context args (informational; speed uses random data)
    parser.add_argument('--operation', type=str, default='/',
                        choices=['*', '/', '+', '-'],
                        help='Operation this speed run is matched to (informational)')
    parser.add_argument('--train-fraction', type=float, default=0.5,
                        help='Training fraction this run is matched to (informational)')
    parser.add_argument('--split-type', type=str, default='random',
                        choices=['random', 'sequential', 'alternating'],
                        help='Split type (informational; passed through to sidecar)')

    # Dataset size
    parser.add_argument('--n-samples', type=int, required=True,
                        help='Dataset size')

    # Training args
    parser.add_argument('--saturation-threshold', type=float, default=99.0,
                        help='Accuracy threshold to consider saturated (%%)')
    parser.add_argument('--patience', type=int, default=0,
                        help='Epochs to confirm saturation')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    _op_safe = args.operation.replace('/', 'div').replace('*', 'mul').replace('+', 'add').replace('-', 'sub')
    signature = f'p{args.p}_op_{_op_safe}_seed{args.seed}'
    args.data_dir = os.path.join(args.data_dir, signature)
    os.makedirs(args.data_dir, exist_ok=True)

    n_tokens = args.p + 2
    bits_per_example = np.log2(n_tokens)
    print(f"=== speed  p={args.p} op={args.operation} seed={args.seed} | "
          f"dim={args.dim} n={args.n_samples} depth={args.depth} heads={args.heads} "
          f"dropout={args.dropout} wd={args.weight_decay} lr={args.lr} ===")

    result = load_or_run_experiment(
        n_samples=args.n_samples,
        dim=args.dim,
        args=args,
        force=args.force,
        verbose=True
    )

    dataset_bits = args.n_samples * bits_per_example
    steps_per_bit = result['saturation_step'] / dataset_bits if dataset_bits > 0 else 0
    print(f"  final acc: {result['final_acc']:.2f}%, "
          f"saturation steps: {result['saturation_step']:,}, "
          f"steps/bit: {steps_per_bit:.2f}")


if __name__ == '__main__':
    main()

