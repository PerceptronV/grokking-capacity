"""
Run grokking experiments with varying model dimensions.

Tracks two memorisation metrics throughout training:
- M_T (total memorisation): log_2(|num_tokens|) + log_2(p_model_correct) per data point
- M_U (unintended memorisation): requires a baseline model, measures bits beyond baseline

where |num_tokens| = p + 2 for modular arithmetic tasks (p outputs + 2 special tokens).

Stores log probs on all data points at each epoch in the npz files,
allowing reconstruction of memorisation against other models/baselines.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import TransformerTorch
from data import grokking_data_torch
from plotting import plot_combined_curves, plot_separate_curves, plot_grokking_time
from utils import load_model


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GrokkingTrainer:
    """
    Trainer for grokking experiments with memorisation tracking.
    
    Tracks two memorisation metrics:
    - M_T (total memorisation): log_2(|num_tokens|) + log_2(p_model_correct) per data point
    - M_U (unintended memorisation): when baseline provided, measures bits beyond baseline
    
    where |num_tokens| = p + 2 (p outputs + 2 special tokens for op and equals).
    
    Also stores log probs on all data points at each epoch to allow
    reconstruction of memorisation against other models/baselines.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        n_tokens: int,
        batch_size: int = 512,
        device: str = 'cpu',
        baseline_model: Optional[nn.Module] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.n_tokens = n_tokens  # Number of output tokens (p + 2)
        self.batch_size = batch_size
        self.device = device
        self.baseline_model = baseline_model
        
        # Move baseline to device if provided
        if self.baseline_model is not None:
            self.baseline_model = self.baseline_model.to(device)
            self.baseline_model.eval()
        
        # Traces
        self.train_acc_trace = []
        self.train_loss_trace = []
        self.val_acc_trace = []
        self.val_loss_trace = []
        self.train_log_probs_trace = []  # Log2 probs for all training points at each epoch
        self.val_log_probs_trace = []  # Log2 probs for all validation points at each epoch
        self.mem_t_trace = []  # Total memorisation M_T at each epoch
        self.mem_u_trace = []  # Unintended memorisation M_U at each epoch (if baseline provided)
    
    def _make_batches(self, X: torch.Tensor, T: torch.Tensor):
        """Yield batches from data."""
        bs = self.batch_size if self.batch_size != -1 else X.shape[0]
        for i in range(0, X.shape[0], bs):
            yield X[i:i+bs], T[i:i+bs]
    
    def _log_probs(self, model: nn.Module, X: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Compute log2 probabilities (in bits) for each sample using the given model.
        
        Args:
            model: The model to use for computing log probs
            X: Input tensor
            T: Target tensor
        
        Returns:
            Tensor of shape (N,) with log2 probabilities of the correct answer
        """
        model.eval()
        all_log_probs = []
        
        with torch.no_grad():
            for Xb, Tb in self._make_batches(X, T):
                Xb = Xb.to(self.device)
                Tb = Tb.to(self.device)
                
                logits = model(Xb)
                log_probs = F.log_softmax(logits, dim=-1)
                correct_log_probs = log_probs.gather(1, Tb.unsqueeze(1)).squeeze(1)
                # Convert from natural log to log2 (bits)
                log_probs_batch = correct_log_probs / np.log(2)
                all_log_probs.append(log_probs_batch.cpu())
        
        return torch.cat(all_log_probs)
    
    def compute_mem_t(self, log_probs: torch.Tensor) -> float:
        """
        Compute total memorisation M_T from log probs.
        
        M_T per data point = log_2(|num_tokens|) + log_2(p_model_correct)
        
        - Random baseline: p_model_correct = 1/n, so M_T = log_2(n) + log_2(1/n) = 0
        - Perfect model: p_model_correct = 1, so M_T = log_2(n) + 0 = log_2(n)
        
        Args:
            log_probs: Tensor of log2 probabilities for correct answers
        
        Returns:
            Total memorisation M_T in bits (sum over all data points)
        """
        entropy_bits = np.log2(self.n_tokens)  # log_2(|num_tokens|)
        # log_probs are log_2(p_model_correct), which are negative or zero
        per_sample_mem = entropy_bits + log_probs  # Each is in [0, log_2(n_tokens)]
        return per_sample_mem.sum().item()
    
    def compute_mem_u(self, X: torch.Tensor, T: torch.Tensor) -> float:
        """
        Compute unintended memorisation M_U on the given data.
        
        M_U = sum_i (H_baseline(i) - H_joint(i)) where H is negative log2 probability,
        and the joint distribution takes the max probability between model and baseline.
        
        Returns:
            Total unintended memorisation M_U in bits
        """
        if self.baseline_model is None:
            return 0.0
        
        log_probs_model = self._log_probs(self.model, X, T)
        log_probs_baseline = self._log_probs(self.baseline_model, X, T)
        # Joint takes max probability = max log prob
        log_probs_joint = torch.max(log_probs_model, log_probs_baseline)
        
        # M_U = H_baseline - H_joint = (-log_probs_baseline) - (-log_probs_joint)
        per_sample_mem = (-log_probs_baseline) - (-log_probs_joint)
        return per_sample_mem.sum().item()
    
    def evaluate(self, X: torch.Tensor, T: torch.Tensor) -> Tuple[float, float]:
        """Evaluate model on data, returning (loss, accuracy)."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        
        with torch.no_grad():
            for Xb, Tb in self._make_batches(X, T):
                Xb = Xb.to(self.device)
                Tb = Tb.to(self.device)
                
                outputs = self.model(Xb)
                loss = F.cross_entropy(outputs, Tb)
                total_loss += loss.item() * Xb.size(0)
                
                preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == Tb).sum().item()
        
        avg_loss = total_loss / X.shape[0]
        avg_acc = total_correct / X.shape[0]
        return avg_loss, avg_acc
    
    def train(
        self,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        val_data: Tuple[torch.Tensor, torch.Tensor],
        epochs: int = 200,
        shuffle: bool = True,
        early_stopping_train: Optional[float] = None,
        early_stopping_val: Optional[float] = None,
        patience: int = -1,
        min_delta: float = 1e-4
    ) -> Dict:
        """
        Train the model with memorisation tracking.

        Args:
            train_data: Tuple of (X_train, T_train)
            val_data: Tuple of (X_val, T_val)
            epochs: Maximum number of epochs
            shuffle: Whether to shuffle training data each epoch
            early_stopping_train: Stop when train accuracy reaches this (e.g., 0.99) and val accuracy is also reached
            early_stopping_val: Stop when val accuracy reaches this (e.g., 0.99) and train accuracy is also reached
            patience: Number of epochs to wait for improvement before stopping. If negative, never stop early due to patience.
            min_delta: Minimum change in validation accuracy to qualify as an improvement

        Returns:
            Dictionary with training results including log probs traces
        """
        X_train, T_train = train_data
        X_val, T_val = val_data
        n_train = X_train.shape[0]

        # Initialize patience tracking
        best_val_acc = 0.0
        epochs_without_improvement = 0
        use_patience = patience >= 0

        epoch_bar = tqdm(range(epochs), desc='Training', unit='epoch')
        
        for epoch in epoch_bar:
            self.model.train()
            
            # Shuffle training data
            if shuffle:
                perm = torch.randperm(n_train)
                X_train_epoch = X_train[perm]
                T_train_epoch = T_train[perm]
            else:
                X_train_epoch = X_train
                T_train_epoch = T_train
            
            # Training loop
            total_loss = 0.0
            total_correct = 0
            
            for Xb, Tb in self._make_batches(X_train_epoch, T_train_epoch):
                Xb = Xb.to(self.device)
                Tb = Tb.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(Xb)
                loss = F.cross_entropy(outputs, Tb)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * Xb.size(0)
                preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == Tb).sum().item()
            
            train_loss = total_loss / n_train
            train_acc = total_correct / n_train
            
            # Evaluate on validation set
            val_loss, val_acc = self.evaluate(X_val, T_val)
            
            # Record traces
            self.train_loss_trace.append(train_loss)
            self.train_acc_trace.append(train_acc)
            self.val_loss_trace.append(val_loss)
            self.val_acc_trace.append(val_acc)
            
            # Compute log probs for all data points (preserves original order)
            train_log_probs = self._log_probs(self.model, X_train, T_train)
            val_log_probs = self._log_probs(self.model, X_val, T_val)
            self.train_log_probs_trace.append(train_log_probs.numpy())
            self.val_log_probs_trace.append(val_log_probs.numpy())
            
            # Compute total memorisation M_T
            mem_t = self.compute_mem_t(train_log_probs)
            self.mem_t_trace.append(mem_t)
            
            # Compute unintended memorisation M_U if baseline provided
            postfix = {
                'train_acc': f'{train_acc:.3f}',
                'val_acc': f'{val_acc:.3f}',
                'M_T': f'{mem_t:.1f}'
            }
            
            if self.baseline_model is not None:
                mem_u = self.compute_mem_u(X_train, T_train)
                self.mem_u_trace.append(mem_u)
                postfix['M_U'] = f'{mem_u:.1f}'
            
            epoch_bar.set_postfix(postfix)

            # Early stopping based on threshold
            if ((early_stopping_val is not None and val_acc >= early_stopping_val)
                and (early_stopping_train is not None and train_acc >= early_stopping_train)):
                print(f"\nEarly stopping: val acc {val_acc:.3f} >= {early_stopping_val:.3f} and train acc {train_acc:.3f} >= {early_stopping_train:.3f}")
                break

            # Early stopping based on patience
            if use_patience:
                if val_acc > best_val_acc + min_delta:
                    best_val_acc = val_acc
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(f"\nEarly stopping: validation accuracy did not improve for {patience} epochs")
                    break
        
        # Stack log probs: shape (epochs, n_samples)
        train_log_probs_array = np.stack(self.train_log_probs_trace, axis=0)
        val_log_probs_array = np.stack(self.val_log_probs_trace, axis=0)
        
        return {
            'train_acc': np.array(self.train_acc_trace),
            'val_acc': np.array(self.val_acc_trace),
            'train_loss': np.array(self.train_loss_trace),
            'val_loss': np.array(self.val_loss_trace),
            'train_log_probs': train_log_probs_array,  # (epochs, n_train)
            'val_log_probs': val_log_probs_array,  # (epochs, n_val)
            'mem_t_trace': np.array(self.mem_t_trace),  # M_T at each epoch
            'mem_u_trace': np.array(self.mem_u_trace) if self.mem_u_trace else None,  # M_U at each epoch
        }


def save_individual_results(dim, train_acc, val_acc, param_count, args,
                           train_log_probs=None, val_log_probs=None,
                           mem_t_trace=None, mem_u_trace=None, baseline_path=None):
    """Save individual experiment results (plot and data).
    
    Args:
        dim: Model dimension
        train_acc: Training accuracy trace (epochs,)
        val_acc: Validation accuracy trace (epochs,)
        param_count: Number of model parameters
        args: Command line arguments
        train_log_probs: Log2 probs on training data at each epoch (epochs, n_train)
        val_log_probs: Log2 probs on validation data at each epoch (epochs, n_val)
        mem_t_trace: Total memorisation M_T at each epoch
        mem_u_trace: Unintended memorisation M_U at each epoch (if baseline provided)
        baseline_path: Path to baseline model (if provided)
    """
    os.makedirs(args.plot_dir, exist_ok=True)
    
    # Create individual plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(train_acc, label='Training Accuracy', color='#1b9e77', linewidth=2, linestyle='-')
    ax.plot(val_acc, label='Validation Accuracy', color='#d95f02', linewidth=2, linestyle='--')
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title(f'Grokking Curve: dim={dim}, depth={args.depth}, heads={args.heads}\n'
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
    plot_fname = os.path.join(args.plot_dir, f'grokking_dim{dim}_depth{args.depth}_heads{args.heads}.pdf')
    plt.savefig(plot_fname, bbox_inches='tight')
    print(f"  Saved plot: {plot_fname}")
    plt.close()
    
    # Prepare data dict
    data_dict = {
        'train_acc': train_acc, 
        'val_acc': val_acc, 
        'dim': dim,
        'param_count': param_count,
        'depth': args.depth,
        'heads': args.heads,
        'epochs': args.epochs,
        'p': args.p,
        'n_tokens': args.p + 2,  # Store n_tokens explicitly
        'op': args.op,
        'train_fraction': args.train_fraction,
        'n_train': train_log_probs.shape[1] if train_log_probs is not None else 0,
        'n_val': val_log_probs.shape[1] if val_log_probs is not None else 0,
    }
    
    # Add log probs for memorisation reconstruction
    if train_log_probs is not None:
        data_dict['train_log_probs'] = train_log_probs
    if val_log_probs is not None:
        data_dict['val_log_probs'] = val_log_probs
    
    # Add memorisation traces
    if mem_t_trace is not None:
        data_dict['mem_t_trace'] = mem_t_trace
    if mem_u_trace is not None:
        data_dict['mem_u_trace'] = mem_u_trace
        data_dict['baseline_path'] = baseline_path
    
    # Save raw data
    data_fname = os.path.join(args.data_dir, f'grokking_dim{dim}_depth{args.depth}_heads{args.heads}.npz')
    np.savez(data_fname, **data_dict)
    print(f"  Saved data: {data_fname}")


def run_experiment(dim, args, baseline_model=None, baseline_path=None):
    """Run a single experiment with the given dimension."""
    print(f"\n{'='*60}")
    print(f"Running experiment with dim={dim}")
    if baseline_model is not None:
        print(f"Computing M_U against baseline: {baseline_path}")
    print(f"{'='*60}")
    
    # Check if results already exist
    data_fname = os.path.join(args.data_dir, f'grokking_dim{dim}_depth{args.depth}_heads{args.heads}.npz')
    if os.path.exists(data_fname) and not args.force:
        print(f"Results already exist for dim={dim}, loading from {data_fname}")
        data = np.load(data_fname)
        result = {
            'dim': int(data['dim']),
            'param_count': int(data['param_count']),
            'train_acc': data['train_acc'],
            'val_acc': data['val_acc']
        }
        if 'train_log_probs' in data:
            result['train_log_probs'] = data['train_log_probs']
        if 'val_log_probs' in data:
            result['val_log_probs'] = data['val_log_probs']
        if 'mem_t_trace' in data:
            result['mem_t_trace'] = data['mem_t_trace']
        if 'mem_u_trace' in data:
            result['mem_u_trace'] = data['mem_u_trace']
        return result
    
    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Prepare data
    Xtrain, Ttrain, Xtest, Ttest = grokking_data_torch(
        args.p, op=args.op, split_type=args.split_type, 
        train_fraction=args.train_fraction, device='cpu'
    )
    
    n_tokens = args.p + 2
    
    # Build model
    model_kwargs = {
        'depth': args.depth,
        'dim': dim,
        'heads': args.heads,
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
    print(f"Device: {device}")
    print(f"Model parameters: {param_count:,}")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )
    
    # Create trainer with optional baseline
    trainer = GrokkingTrainer(
        model=model,
        optimizer=optimizer,
        n_tokens=n_tokens,
        batch_size=args.batch_size,
        device=device,
        baseline_model=baseline_model
    )
    
    # Set early stopping threshold
    early_stop_train = None if args.no_early_stopping else args.early_stopping_train
    early_stop_val = None if args.no_early_stopping else args.early_stopping_val

    # Train
    results = trainer.train(
        train_data=(Xtrain, Ttrain),
        val_data=(Xtest, Ttest),
        epochs=args.epochs,
        shuffle=True,
        early_stopping_train=early_stop_train,
        early_stopping_val=early_stop_val,
        patience=args.patience,
        min_delta=args.min_delta
    )
    
    # Convert to percentages for consistency
    train_acc = results['train_acc'] * 100
    val_acc = results['val_acc'] * 100
    train_log_probs = results['train_log_probs']
    val_log_probs = results['val_log_probs']
    mem_t_trace = results['mem_t_trace']
    mem_u_trace = results['mem_u_trace']
    
    # Save results
    save_individual_results(dim, train_acc, val_acc, param_count, args,
                           train_log_probs=train_log_probs, val_log_probs=val_log_probs,
                           mem_t_trace=mem_t_trace, mem_u_trace=mem_u_trace,
                           baseline_path=baseline_path)
    
    result = {
        'dim': dim,
        'param_count': param_count,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'train_log_probs': train_log_probs,
        'val_log_probs': val_log_probs,
        'mem_t_trace': mem_t_trace
    }
    if mem_u_trace is not None:
        result['mem_u_trace'] = mem_u_trace
    
    return result


def plot_results(results, args):
    """Plot overlaid grokking curves for all dimensions."""
    os.makedirs(args.plot_dir, exist_ok=True)
    
    # Create separate subplots for validation and training using shared utility
    fname_separate = os.path.join(args.plot_dir, f'grokking_vs_dimension_depth{args.depth}_heads{args.heads}.pdf')
    plot_separate_curves(results,
                        title_val='Grokking Curves vs. Model Dimension',
                        title_train='Training Curves vs. Model Dimension',
                        save_path=fname_separate, show=not args.no_show)

    # Create a combined plot with both train and val on same axes using shared utility
    fname_combined = os.path.join(args.plot_dir, f'grokking_combined_depth{args.depth}_heads{args.heads}.pdf')
    plot_combined_curves(results, save_path=fname_combined, show=not args.no_show)

    # Create grokking time plot using shared utility
    fname = os.path.join(args.plot_dir, f'grokking_time_vs_params_{args.depth}-{args.heads}.pdf')
    plot_grokking_time(results, threshold_val=97.0, max_epochs=args.epochs,
                      save_path=fname, show=not args.no_show)


def compute_mem_t_from_log_probs(log_probs: np.ndarray, n_tokens: int) -> float:
    """
    Compute total memorisation M_T from log probs array.
    
    Args:
        log_probs: Array of log2 probabilities (epochs, n_samples) or (n_samples,)
        n_tokens: Number of output tokens (p + 2)
    
    Returns:
        Total memorisation M_T in bits
    """
    baseline_bits = np.log2(n_tokens)
    if log_probs.ndim == 1:
        return (baseline_bits + log_probs).sum()
    else:
        # Return memorisation at final epoch
        return (baseline_bits + log_probs[-1]).sum()


def main():
    parser = argparse.ArgumentParser(description='Run grokking experiments with varying model dimensions')
    
    # Data args
    parser.add_argument('--p', type=int, default=97, help='prime number')
    parser.add_argument('--op', type=str, default='/', help='operation', choices=['*', '/', '+', '-'])
    parser.add_argument('--train-fraction', type=float, default=0.5, help='train fraction')
    parser.add_argument('--split-type', type=str, default='random', help='split type', choices=['random', 'sequential', 'alternating'])
    
    # Model args (dim will be varied)
    parser.add_argument('--depth', type=int, default=2, help='depth')
    parser.add_argument('--heads', type=int, default=1, help='heads')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--dim-start', type=int, default=20, help='starting dimension')
    parser.add_argument('--dim-end', type=int, default=240, help='ending dimension')
    parser.add_argument('--dim-step', type=int, default=20, help='dimension step size')
    
    # Optimizer args
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=1, help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1')
    parser.add_argument('--beta2', type=float, default=0.98, help='beta2')
    
    # Training args
    parser.add_argument('-b', '--batch-size', type=int, default=512, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--early-stopping-train', type=float, default=0.99,
                       help='stop training when train accuracy reaches this threshold (default: 0.99), if val accuracy is also reached, stop training')
    parser.add_argument('--early-stopping-val', type=float, default=0.99,
                       help='stop training when val accuracy reaches this threshold (default: 0.99), if train accuracy is also reached, stop training')
    parser.add_argument('--no-early-stopping', action='store_true',
                       help='disable early stopping and train for full epochs')
    parser.add_argument('--patience', type=int, default=-1,
                       help='patience for early stopping (epochs without improvement). If negative, never stop early due to patience (default: -1)')
    parser.add_argument('--min-delta', type=float, default=1e-4,
                       help='minimum validation accuracy improvement to reset patience (default: 1e-4)')
    parser.add_argument('--data-dir', type=str, default='data/groks', help='data output directory')
    parser.add_argument('--plot-dir', type=str, default='media/groks', help='plot output directory')
    
    # Memorisation tracking
    parser.add_argument('--baseline', type=str, default=None, 
                       help='Path to baseline model directory (e.g., "p97_seed42_splitrandom/dim24_depth2_heads1"). '
                            'When provided, computes unintended memorisation M_U at each epoch on training set.')
    
    # Misc args
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--cpu', action='store_true', help='use cpu only')
    parser.add_argument('--device', type=str, default=None,
                       help='device to use (e.g., "cuda:0", "cuda:1", "cpu", "mps"). Overrides --cpu flag if specified.')
    parser.add_argument('--force', action='store_true', help='force re-run even if results exist')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots (just save)')
    
    args = parser.parse_args()

    # Set data and plot directories
    signature = f'p{args.p}_seed{args.seed}_split{args.split_type}'
    args.data_dir = os.path.join(args.data_dir, signature)
    args.plot_dir = os.path.join(args.plot_dir, signature)

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    
    # Load baseline model if provided
    baseline_model = None
    baseline_path = None
    if args.baseline:
        baseline_path = args.baseline
        # Construct full path to model file
        model_file = os.path.join('data/single', baseline_path, 'model.pt')
        if not os.path.exists(model_file):
            print(f"Error: Baseline model not found at {model_file}")
            return
        
        print(f"\nLoading baseline model from: {model_file}")
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
        baseline_model, baseline_metadata = load_model(model_file, device=device)
        print(f"Baseline model loaded: {baseline_metadata['param_count']:,} parameters")
    
    n_tokens = args.p + 2
    
    # Generate dimension values
    dims = list(range(args.dim_start, args.dim_end + 1, args.dim_step))
    print(f"Running experiments with dimensions: {dims}")
    print(f"Prime p={args.p}, n_tokens={n_tokens}, max bits per sample = log_2({n_tokens}) = {np.log2(n_tokens):.2f}")
    
    # Run experiments
    results = []
    for dim in dims:
        result = run_experiment(dim, args, baseline_model=baseline_model, baseline_path=baseline_path)
        results.append(result)
    
    # Plot results
    plot_results(results, args)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for result in results:
        summary_str = (f"dim={result['dim']:3d}: {result['param_count']:8,} params, "
                      f"final train acc={result['train_acc'][-1]:.1f}%, "
                      f"final val acc={result['val_acc'][-1]:.1f}%, "
                      f"final M_T={result['mem_t_trace'][-1]:.1f} bits")
        if 'mem_u_trace' in result:
            summary_str += f", final M_U={result['mem_u_trace'][-1]:.1f} bits"
        print(summary_str)


if __name__ == '__main__':
    main()
