"""
Investigate whether memorisation occurs in the lazy (NTK) regime.

This module provides diagnostics to determine if neural networks are operating
in the "lazy" regime (where the NTK stays approximately constant during training)
or the "feature learning" regime (where the network's representations evolve).

Key metrics tracked:
1. NTK evolution: How much the empirical NTK changes during training
2. Parameter displacement: ||θ(t) - θ(0)|| / ||θ(0)||
3. Linearization error: Difference between actual network and its linearization

In the lazy regime:
- NTK stays approximately constant
- Parameters move O(1/width) from initialization
- Linearization accurately predicts network dynamics

In the feature learning regime:
- NTK evolves significantly
- Parameters move O(1) from initialization
- Linearization fails to capture network dynamics
"""

import argparse
import numpy as np
import os
from tqdm import tqdm
from typing import Tuple, Dict, List, Optional
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import TransformerTorch
from data import random_target_data_torch


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_flat_params(model: nn.Module) -> torch.Tensor:
    """Get all model parameters as a single flat vector."""
    return torch.cat([p.view(-1) for p in model.parameters() if p.requires_grad])


def set_flat_params(model: nn.Module, flat_params: torch.Tensor):
    """Set model parameters from a flat vector."""
    offset = 0
    for p in model.parameters():
        if p.requires_grad:
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset + numel].view_as(p))
            offset += numel


def compute_jacobian(
    model: nn.Module,
    X: torch.Tensor,
    device: str = 'cpu',
    max_samples: int = 100
) -> torch.Tensor:
    """
    Compute the Jacobian of the model output w.r.t. parameters.

    Returns:
        Jacobian matrix of shape (n_samples * n_outputs, n_params)
    """
    model.eval()
    n_samples = min(X.shape[0], max_samples)
    X_subset = X[:n_samples].to(device)

    # Get number of parameters and output dimension
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Forward pass to get output shape
    with torch.no_grad():
        out = model(X_subset)
    n_outputs = out.shape[-1]

    # Compute Jacobian row by row
    jacobian_rows = []

    for i in range(n_samples):
        x_i = X_subset[i:i+1]

        for j in range(n_outputs):
            model.zero_grad()
            out = model(x_i)
            out[0, j].backward(retain_graph=True)

            # Collect gradients
            grads = []
            for p in model.parameters():
                if p.requires_grad:
                    if p.grad is not None:
                        grads.append(p.grad.view(-1).clone())
                    else:
                        grads.append(torch.zeros(p.numel(), device=device))

            jacobian_rows.append(torch.cat(grads))

    jacobian = torch.stack(jacobian_rows)
    return jacobian


def compute_ntk(
    model: nn.Module,
    X: torch.Tensor,
    device: str = 'cpu',
    max_samples: int = 50
) -> torch.Tensor:
    """
    Compute the empirical Neural Tangent Kernel.

    NTK(x, x') = J(x) @ J(x')^T where J is the Jacobian of outputs w.r.t. parameters.

    Returns:
        NTK matrix of shape (n_samples * n_outputs, n_samples * n_outputs)
    """
    jacobian = compute_jacobian(model, X, device, max_samples)
    ntk = jacobian @ jacobian.T
    return ntk


def compute_ntk_change(ntk_init: torch.Tensor, ntk_current: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics for how much the NTK has changed.

    Returns:
        Dictionary with:
        - relative_frobenius: ||K_t - K_0||_F / ||K_0||_F
        - relative_spectral: ||K_t - K_0||_2 / ||K_0||_2
        - cosine_similarity: cos(K_0, K_t) treating as vectors
    """
    diff = ntk_current - ntk_init

    # Frobenius norm change
    fro_init = torch.norm(ntk_init, p='fro')
    fro_diff = torch.norm(diff, p='fro')
    relative_fro = (fro_diff / fro_init).item() if fro_init > 0 else 0.0

    # Spectral norm change (largest singular value)
    try:
        spec_init = torch.linalg.norm(ntk_init, ord=2)
        spec_diff = torch.linalg.norm(diff, ord=2)
        relative_spec = (spec_diff / spec_init).item() if spec_init > 0 else 0.0
    except:
        relative_spec = relative_fro  # Fallback

    # Cosine similarity (flatten to vectors)
    flat_init = ntk_init.flatten()
    flat_current = ntk_current.flatten()
    cos_sim = F.cosine_similarity(flat_init.unsqueeze(0), flat_current.unsqueeze(0)).item()

    return {
        'relative_frobenius': relative_fro,
        'relative_spectral': relative_spec,
        'cosine_similarity': cos_sim
    }


def compute_parameter_displacement(
    theta_init: torch.Tensor,
    theta_current: torch.Tensor
) -> Dict[str, float]:
    """
    Compute metrics for parameter displacement from initialization.

    Returns:
        Dictionary with:
        - relative_l2: ||θ_t - θ_0||_2 / ||θ_0||_2
        - relative_linf: ||θ_t - θ_0||_∞ / ||θ_0||_∞
        - cosine_similarity: cos(θ_0, θ_t)
    """
    diff = theta_current - theta_init

    # L2 norm
    l2_init = torch.norm(theta_init, p=2)
    l2_diff = torch.norm(diff, p=2)
    relative_l2 = (l2_diff / l2_init).item() if l2_init > 0 else 0.0

    # L-infinity norm
    linf_init = torch.norm(theta_init, p=float('inf'))
    linf_diff = torch.norm(diff, p=float('inf'))
    relative_linf = (linf_diff / linf_init).item() if linf_init > 0 else 0.0

    # Cosine similarity
    cos_sim = F.cosine_similarity(theta_init.unsqueeze(0), theta_current.unsqueeze(0)).item()

    return {
        'relative_l2': relative_l2,
        'relative_linf': relative_linf,
        'cosine_similarity': cos_sim
    }


def compute_linearization_error(
    model: nn.Module,
    model_init: nn.Module,
    theta_init: torch.Tensor,
    jacobian_init: torch.Tensor,
    f_init: torch.Tensor,
    X: torch.Tensor,
    device: str = 'cpu',
    max_samples: int = 100
) -> Dict[str, float]:
    """
    Compute the error between the actual network and its linear approximation.

    Linear approximation: f_lin(x; θ) = f(x; θ_0) + J(x; θ_0) @ (θ - θ_0)

    Args:
        model: Current model
        model_init: Initial model (inference only, no grads)
        theta_init: Initial parameters (flattened)
        jacobian_init: Pre-computed Jacobian at initialization
        f_init: Pre-computed initial outputs f(x; θ_0)
        X: Input data (must match what was used for jacobian_init)
        device: Device
        max_samples: Max samples to use

    Returns:
        Dictionary with:
        - output_mse: MSE between actual and linearized outputs
        - output_relative_error: ||f - f_lin||_F / ||f||_F
        - prediction_agreement: Fraction of matching predictions
    """
    model.eval()

    n_samples = min(X.shape[0], max_samples)
    X_subset = X[:n_samples].to(device)

    # Get current parameters
    theta_current = get_flat_params(model)
    param_diff = theta_current - theta_init.to(theta_current.device)

    # Compute current outputs
    with torch.no_grad():
        f_current = model(X_subset)    # f(x; θ_t)

    # Reshape f_init to match jacobian output (n_samples * n_outputs)
    f_init_flat = f_init.reshape(-1)

    # Linear approximation: f_lin = f_0 + J_0 @ (θ - θ_0)
    f_lin_flat = f_init_flat + jacobian_init @ param_diff
    f_lin = f_lin_flat.reshape(f_current.shape)

    # Compute errors
    mse = F.mse_loss(f_current, f_lin).item()

    # Relative error
    f_norm = torch.norm(f_current, p='fro')
    diff_norm = torch.norm(f_current - f_lin, p='fro')
    relative_error = (diff_norm / f_norm).item() if f_norm > 0 else 0.0

    # Prediction agreement
    pred_current = torch.argmax(f_current, dim=-1)
    pred_lin = torch.argmax(f_lin, dim=-1)
    agreement = (pred_current == pred_lin).float().mean().item()

    return {
        'output_mse': mse,
        'output_relative_error': relative_error,
        'prediction_agreement': agreement
    }


class NTKTrainer:
    """
    Trainer that tracks NTK-related metrics during memorisation.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        batch_size: int = 512,
        device: str = 'cpu',
        ntk_samples: int = 50,
        track_interval: int = 100
    ):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.device = device
        self.ntk_samples = ntk_samples
        self.track_interval = track_interval

        # Store initial state
        self.theta_init = get_flat_params(model).clone().detach()

        # Keep a copy of initial model for linearization error computation
        # This copy has requires_grad=False for inference only
        self.model_init_inference = copy.deepcopy(model)
        self.model_init_inference.eval()
        for p in self.model_init_inference.parameters():
            p.requires_grad = False

        # Keep a separate copy with requires_grad=True for NTK computation
        # We'll only use this once to compute the initial NTK
        self.model_init_for_ntk = copy.deepcopy(model)
        self.model_init_for_ntk.eval()
        # Keep requires_grad=True for backprop in Jacobian computation

        # Traces
        self.train_loss_trace = []
        self.train_acc_trace = []
        self.steps_trace = []

        # NTK tracking
        self.ntk_change_trace = []
        self.param_displacement_trace = []
        self.linearization_error_trace = []
        self.tracking_steps = []

        # Store initial NTK and Jacobian (computed lazily)
        self.ntk_init = None
        self.jacobian_init = None
        self.f_init = None
        self.X_ntk = None

    def _make_batches(self, X: torch.Tensor, T: torch.Tensor):
        """Yield batches from data."""
        bs = self.batch_size if self.batch_size != -1 else X.shape[0]
        for i in range(0, X.shape[0], bs):
            yield X[i:i+bs], T[i:i+bs]

    def _track_metrics(self, X: torch.Tensor, T: torch.Tensor, total_steps: int):
        """Track NTK-related metrics."""
        # Use subset for NTK computation
        if self.X_ntk is None:
            self.X_ntk = X[:self.ntk_samples].to(self.device)

        # Compute initial NTK and Jacobian if not done yet (using model with grads enabled)
        if self.ntk_init is None:
            print("  Computing initial NTK and Jacobian...")
            # Compute Jacobian (NTK = J @ J^T)
            self.jacobian_init = compute_jacobian(
                self.model_init_for_ntk, self.X_ntk, self.device, self.ntk_samples
            )
            self.ntk_init = self.jacobian_init @ self.jacobian_init.T

            # Compute initial outputs for linearization error
            with torch.no_grad():
                self.f_init = self.model_init_inference(self.X_ntk)

            # We can now delete this model to save memory
            del self.model_init_for_ntk
            self.model_init_for_ntk = None

        # Current NTK
        ntk_current = compute_ntk(self.model, self.X_ntk, self.device, self.ntk_samples)
        ntk_change = compute_ntk_change(self.ntk_init, ntk_current)
        self.ntk_change_trace.append(ntk_change)

        # Parameter displacement
        theta_current = get_flat_params(self.model)
        param_disp = compute_parameter_displacement(self.theta_init, theta_current)
        self.param_displacement_trace.append(param_disp)

        # Linearization error (uses pre-computed Jacobian and initial outputs)
        lin_error = compute_linearization_error(
            self.model, self.model_init_inference, self.theta_init,
            self.jacobian_init, self.f_init,
            self.X_ntk, self.device, self.ntk_samples
        )
        self.linearization_error_trace.append(lin_error)

        self.tracking_steps.append(total_steps)

        return ntk_change, param_disp, lin_error

    def train(
        self,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        max_epochs: int = 10000,
        saturation_threshold: float = 99.5,
        patience: int = 50,
        verbose: bool = True
    ) -> Dict:
        """
        Train until accuracy saturates, tracking NTK metrics.
        """
        X_train, T_train = train_data
        n_samples = X_train.shape[0]

        # Calculate steps per epoch
        bs = self.batch_size if self.batch_size != -1 else n_samples
        steps_per_epoch = (n_samples + bs - 1) // bs

        total_steps = 0
        saturation_step = None
        epochs_above_threshold = 0

        # Track initial metrics
        if verbose:
            print("\nTracking initial metrics...")
        self._track_metrics(X_train, T_train, 0)

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
            avg_acc = (total_correct / n_samples) * 100

            self.train_loss_trace.append(avg_loss)
            self.train_acc_trace.append(avg_acc)
            self.steps_trace.append(total_steps)

            # Track NTK metrics periodically
            if (epoch + 1) % self.track_interval == 0 or avg_acc >= saturation_threshold:
                ntk_change, param_disp, lin_error = self._track_metrics(X_train, T_train, total_steps)

                if verbose:
                    epoch_iter.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'acc': f'{avg_acc:.1f}%',
                        'NTK_Δ': f'{ntk_change["relative_frobenius"]:.3f}',
                        'θ_Δ': f'{param_disp["relative_l2"]:.3f}',
                        'lin_err': f'{lin_error["output_relative_error"]:.3f}'
                    })
            elif verbose:
                epoch_iter.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{avg_acc:.1f}%'
                })

            # Check for saturation
            if avg_acc >= saturation_threshold:
                epochs_above_threshold += 1
                if saturation_step is None:
                    saturation_step = total_steps

                if epochs_above_threshold >= patience:
                    if verbose:
                        print(f"\nSaturation reached at step {saturation_step}")
                    break
            else:
                epochs_above_threshold = 0
                saturation_step = None

        # Final tracking
        if len(self.tracking_steps) == 0 or self.tracking_steps[-1] != total_steps:
            self._track_metrics(X_train, T_train, total_steps)

        if saturation_step is None:
            saturation_step = total_steps
            if verbose:
                print(f"\nDid not reach saturation. Final acc: {avg_acc:.2f}%")

        return {
            'epochs_trained': epoch + 1,
            'total_steps': total_steps,
            'saturation_step': saturation_step,
            'final_loss': avg_loss,
            'final_acc': avg_acc,
            'train_loss_trace': np.array(self.train_loss_trace),
            'train_acc_trace': np.array(self.train_acc_trace),
            'steps_trace': np.array(self.steps_trace),
            'saturated': avg_acc >= saturation_threshold,
            # NTK metrics
            'ntk_change_trace': self.ntk_change_trace,
            'param_displacement_trace': self.param_displacement_trace,
            'linearization_error_trace': self.linearization_error_trace,
            'tracking_steps': np.array(self.tracking_steps)
        }


def run_ntk_experiment(
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
    Run a single NTK diagnostics experiment.
    """
    n_tokens = p + 2

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
        print(f"  Device: {device}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )

    trainer = NTKTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=args.batch_size,
        device=device,
        ntk_samples=args.ntk_samples,
        track_interval=args.track_interval
    )

    results = trainer.train(
        (X_train, T_train),
        max_epochs=max_epochs,
        saturation_threshold=saturation_threshold,
        patience=patience,
        verbose=verbose
    )

    # Add metadata
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

    return results


def save_results(results: Dict, args):
    """Save experiment results."""
    os.makedirs(args.data_dir, exist_ok=True)

    fname = os.path.join(
        args.data_dir,
        f'ntk_dim{results["dim"]}_samples{results["n_samples"]}.npz'
    )

    # Convert trace lists to arrays
    ntk_fro = [t['relative_frobenius'] for t in results['ntk_change_trace']]
    ntk_spec = [t['relative_spectral'] for t in results['ntk_change_trace']]
    ntk_cos = [t['cosine_similarity'] for t in results['ntk_change_trace']]

    param_l2 = [t['relative_l2'] for t in results['param_displacement_trace']]
    param_linf = [t['relative_linf'] for t in results['param_displacement_trace']]
    param_cos = [t['cosine_similarity'] for t in results['param_displacement_trace']]

    lin_mse = [t['output_mse'] for t in results['linearization_error_trace']]
    lin_rel = [t['output_relative_error'] for t in results['linearization_error_trace']]
    lin_agree = [t['prediction_agreement'] for t in results['linearization_error_trace']]

    np.savez(
        fname,
        # Metadata
        n_samples=results['n_samples'],
        dim=results['dim'],
        depth=results['depth'],
        heads=results['heads'],
        param_count=results['param_count'],
        p=results['p'],
        dataset_bits=results['dataset_bits'],
        # Training results
        epochs_trained=results['epochs_trained'],
        total_steps=results['total_steps'],
        saturation_step=results['saturation_step'],
        final_loss=results['final_loss'],
        final_acc=results['final_acc'],
        saturated=results['saturated'],
        # Traces
        train_loss_trace=results['train_loss_trace'],
        train_acc_trace=results['train_acc_trace'],
        steps_trace=results['steps_trace'],
        tracking_steps=results['tracking_steps'],
        # NTK change
        ntk_relative_frobenius=np.array(ntk_fro),
        ntk_relative_spectral=np.array(ntk_spec),
        ntk_cosine_similarity=np.array(ntk_cos),
        # Parameter displacement
        param_relative_l2=np.array(param_l2),
        param_relative_linf=np.array(param_linf),
        param_cosine_similarity=np.array(param_cos),
        # Linearization error
        lin_output_mse=np.array(lin_mse),
        lin_output_relative_error=np.array(lin_rel),
        lin_prediction_agreement=np.array(lin_agree)
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
        f'ntk_dim{dim}_samples{n_samples}.npz'
    )

    if os.path.exists(fname) and not force:
        if verbose:
            print(f"  Loading existing results: {fname}")
        data = np.load(fname)

        # Reconstruct trace dicts
        results = {}
        for key in data.files:
            val = data[key]
            results[key] = val.item() if val.ndim == 0 else val

        # Reconstruct structured traces
        n_tracking = len(results['tracking_steps'])
        results['ntk_change_trace'] = [
            {
                'relative_frobenius': results['ntk_relative_frobenius'][i],
                'relative_spectral': results['ntk_relative_spectral'][i],
                'cosine_similarity': results['ntk_cosine_similarity'][i]
            }
            for i in range(n_tracking)
        ]
        results['param_displacement_trace'] = [
            {
                'relative_l2': results['param_relative_l2'][i],
                'relative_linf': results['param_relative_linf'][i],
                'cosine_similarity': results['param_cosine_similarity'][i]
            }
            for i in range(n_tracking)
        ]
        results['linearization_error_trace'] = [
            {
                'output_mse': results['lin_output_mse'][i],
                'output_relative_error': results['lin_output_relative_error'][i],
                'prediction_agreement': results['lin_prediction_agreement'][i]
            }
            for i in range(n_tracking)
        ]

        return results

    # Run experiment
    result = run_ntk_experiment(
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
        description='Investigate whether memorisation occurs in the lazy (NTK) regime'
    )

    # Data args
    parser.add_argument('--p', type=int, default=97,
                        help='Prime number (determines vocabulary size)')

    # Model args
    parser.add_argument('--depth', type=int, default=2, help='Transformer depth')
    parser.add_argument('--heads', type=int, default=1, help='Attention heads')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout')
    parser.add_argument('--dim-list', type=int, nargs='+',
                        default=[16, 24, 32],
                        help='List of model dimensions to test')

    # Dataset size args
    parser.add_argument('--samples-start', type=int, default=100,
                        help='Starting dataset size')
    parser.add_argument('--samples-end', type=int, default=500,
                        help='Ending dataset size')
    parser.add_argument('--samples-steps', type=int, default=3,
                        help='Number of dataset sizes to test')

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
                        help='Accuracy threshold to consider saturated')
    parser.add_argument('--patience', type=int, default=10,
                        help='Epochs to confirm saturation')

    # NTK tracking args
    parser.add_argument('--ntk-samples', type=int, default=50,
                        help='Number of samples for NTK computation')
    parser.add_argument('--track-interval', type=int, default=100,
                        help='Epochs between NTK tracking')

    # Output args
    parser.add_argument('--data-dir', type=str, default='data/ntk',
                        help='Data output directory')
    parser.add_argument('--plot-dir', type=str, default='media/ntk',
                        help='Plot output directory')

    # Misc args
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cpu', action='store_true', help='Force CPU only')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use')
    parser.add_argument('--force', action='store_true',
                        help='Force re-run even if results exist')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots')

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

    # Generate dataset sizes
    dataset_sizes = np.linspace(
        args.samples_start,
        args.samples_end,
        args.samples_steps
    ).astype(int)
    dataset_sizes = np.unique(dataset_sizes)

    print(f"Model dimensions: {args.dim_list}")
    print(f"Dataset sizes: {list(dataset_sizes)}")
    print(f"NTK samples: {args.ntk_samples}")
    print(f"Track interval: {args.track_interval} epochs")
    print()

    # Import plotting functions
    from plotting import (
        plot_ntk_evolution,
        plot_param_displacement,
        plot_linearization_error,
        plot_ntk_summary
    )

    # Run experiments
    all_results = {}

    for dim in args.dim_list:
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

            # Print summary for this run
            final_ntk = result['ntk_change_trace'][-1]
            final_param = result['param_displacement_trace'][-1]
            final_lin = result['linearization_error_trace'][-1]

            print(f"    Saturation step: {result['saturation_step']:,}")
            print(f"    Final NTK change (Frobenius): {final_ntk['relative_frobenius']:.4f}")
            print(f"    Final param displacement (L2): {final_param['relative_l2']:.4f}")
            print(f"    Final linearization error: {final_lin['output_relative_error']:.4f}")

    # Generate plots
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)

    show = not args.no_show

    # NTK evolution plot
    ntk_path = os.path.join(args.plot_dir, 'ntk_evolution.pdf')
    plot_ntk_evolution(all_results, save_path=ntk_path, show=show)

    # Parameter displacement plot
    param_path = os.path.join(args.plot_dir, 'param_displacement.pdf')
    plot_param_displacement(all_results, save_path=param_path, show=show)

    # Linearization error plot
    lin_path = os.path.join(args.plot_dir, 'linearization_error.pdf')
    plot_linearization_error(all_results, save_path=lin_path, show=show)

    # Summary plot
    summary_path = os.path.join(args.plot_dir, 'ntk_summary.pdf')
    plot_ntk_summary(all_results, save_path=summary_path, show=show)

    # Print final summary
    print("\n" + "="*60)
    print("NTK REGIME ANALYSIS SUMMARY")
    print("="*60)

    for dim in sorted(all_results.keys()):
        results = all_results[dim]
        param_count = results[0]['param_count']

        # Average final metrics
        avg_ntk_change = np.mean([r['ntk_change_trace'][-1]['relative_frobenius'] for r in results])
        avg_param_disp = np.mean([r['param_displacement_trace'][-1]['relative_l2'] for r in results])
        avg_lin_error = np.mean([r['linearization_error_trace'][-1]['output_relative_error'] for r in results])

        regime = "LAZY" if avg_ntk_change < 0.1 and avg_param_disp < 0.1 else "FEATURE LEARNING"

        print(f"\ndim={dim:3d}: {param_count:8,} params")
        print(f"  Avg NTK change:        {avg_ntk_change:.4f}")
        print(f"  Avg param displacement: {avg_param_disp:.4f}")
        print(f"  Avg linearization error: {avg_lin_error:.4f}")
        print(f"  Likely regime: {regime}")

    print("\n" + "="*60)
    print("Interpretation:")
    print("  - NTK change < 0.1: Kernel approximately constant (lazy regime)")
    print("  - Param displacement < 0.1: Parameters near initialization (lazy regime)")
    print("  - High linearization error: Features are learning (not lazy)")
    print("="*60)


if __name__ == '__main__':
    main()
