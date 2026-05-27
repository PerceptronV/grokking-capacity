"""Measure the information capacity C of a model architecture.

Trains on random-target data until loss saturates, then reports bits-per-example
memorised. C is later fit as the slope of total bits vs param count across many
dims (see analysis/capacity_constant.py).
"""
from __future__ import annotations

import argparse
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from ..data import grokking_data_torch, random_target_data_torch
from ..models import TransformerTorch, count_parameters
from ..registry import build_identifying, run_lifecycle, AlreadyCompleted
from ..utils.cli_args import (
    add_model_args, add_optimizer_args, add_device_args, add_io_args, add_registry_args,
)
from ..utils.device import get_device


class CapacityTrainer:
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, n_tokens: int,
                 batch_size: int = 512, device: str = 'cpu'):
        self.model = model
        self.optimizer = optimizer
        self.n_tokens = n_tokens
        self.batch_size = batch_size
        self.device = device
        self.train_loss_trace: list[float] = []
        self.train_acc_trace: list[float] = []
        self.bits_per_example_trace: list[float] = []

    def _make_batches(self, X, T):
        bs = self.batch_size if self.batch_size != -1 else X.shape[0]
        for i in range(0, X.shape[0], bs):
            yield X[i:i + bs], T[i:i + bs]

    def compute_memorization(self, X, T) -> Tuple[float, float, float]:
        self.model.eval()
        total_log_prob = 0.0
        total_loss = 0.0
        n = X.shape[0]
        with torch.no_grad():
            for Xb, Tb in self._make_batches(X, T):
                Xb, Tb = Xb.to(self.device), Tb.to(self.device)
                logits = self.model(Xb)
                log_probs = F.log_softmax(logits, dim=-1)
                correct = log_probs.gather(1, Tb.unsqueeze(1)).squeeze(1)
                total_log_prob += (correct / np.log(2)).sum().item()
                total_loss += F.cross_entropy(logits, Tb, reduction='sum').item()
        avg_loss = total_loss / n
        avg_log_prob = total_log_prob / n
        bits_per_example = np.log2(self.n_tokens) + avg_log_prob
        return avg_loss, avg_log_prob, bits_per_example

    def train(self, train_data, *, max_epochs: int, patience: int, min_delta: float, verbose: bool):
        X_train, T_train = train_data
        n = X_train.shape[0]
        best_loss = float('inf')
        epochs_without_improvement = 0

        epoch_iter = tqdm(range(max_epochs), desc='Training', unit='epoch') if verbose else range(max_epochs)
        for epoch in epoch_iter:
            self.model.train()
            perm = torch.randperm(n)
            X_sh, T_sh = X_train[perm], T_train[perm]

            total_loss = 0.0
            total_correct = 0
            for Xb, Tb in self._make_batches(X_sh, T_sh):
                Xb, Tb = Xb.to(self.device), Tb.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(Xb)
                loss = F.cross_entropy(logits, Tb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * Xb.size(0)
                total_correct += (torch.argmax(logits, dim=1) == Tb).sum().item()

            avg_loss = total_loss / n
            avg_acc = total_correct / n
            self.train_loss_trace.append(avg_loss)
            self.train_acc_trace.append(avg_acc)
            _, _, bits = self.compute_memorization(X_train, T_train)
            self.bits_per_example_trace.append(bits)

            if verbose:
                epoch_iter.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{avg_acc:.3f}', 'bits': f'{bits:.2f}'})

            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"\nEarly stopping after {epoch + 1} epochs (loss saturated at {avg_loss:.4f})")
                break

        final_loss, _, final_bits = self.compute_memorization(X_train, T_train)
        return {
            'epochs_trained': epoch + 1,
            'final_loss': final_loss,
            'final_acc': self.train_acc_trace[-1],
            'final_bits_per_example': final_bits,
            'total_bits_memorized': final_bits * n,
            'train_loss_trace': np.array(self.train_loss_trace),
            'train_acc_trace': np.array(self.train_acc_trace),
            'bits_trace': np.array(self.bits_per_example_trace),
        }


def _get_dataset(n_samples: int, p: int, dataset_type: str, *, seq_len: int = 4, device: str = 'cpu'):
    if dataset_type == 'random':
        return random_target_data_torch(n_samples, p, seq_len=seq_len, device=device)
    if dataset_type in ('+', '-', '*', '/'):
        size = p * (p - 1) if dataset_type == '/' else p * p
        if n_samples > size:
            raise ValueError(f"n_samples={n_samples} exceeds available {dataset_type} dataset size {size}")
        train_fraction = n_samples / size
        return grokking_data_torch(p, op=dataset_type, split_type='random',
                                    train_fraction=train_fraction, device=device)[:2]
    raise ValueError(f"unknown dataset_type {dataset_type!r}")


def _train_capacity(args, *, device: str) -> Dict:
    n_tokens = args.p + 2
    X_train, T_train = _get_dataset(args.n_samples, args.p, args.dataset_type_raw, seq_len=4, device='cpu')
    model = TransformerTorch(
        depth=args.depth, dim=args.dim, heads=args.heads,
        n_tokens=n_tokens, seq_len=4, dropout=args.dropout, init_scale=args.init_scale,
    ).to(device)
    param_count = count_parameters(model)
    print(f"  n_samples={args.n_samples} dim={args.dim} params={param_count:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                             betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    trainer = CapacityTrainer(model=model, optimizer=optimizer, n_tokens=n_tokens,
                               batch_size=args.batch_size, device=device)
    results = trainer.train(
        (X_train, T_train),
        max_epochs=args.epochs, patience=args.patience, min_delta=args.min_delta, verbose=True,
    )
    results['param_count'] = param_count
    return results


def main():
    parser = argparse.ArgumentParser(description='Measure model capacity for one (dim, n_samples).')
    add_model_args(parser, dropout_default=0.0)
    add_optimizer_args(parser, weight_decay_default=0.01, epochs_default=5000)
    add_device_args(parser)
    add_io_args(parser, data_dir='data/capacity')
    add_registry_args(parser)
    parser.add_argument('--n-samples', type=int, required=True, help='Dataset size')
    parser.add_argument('--dataset-type', dest='dataset_type_raw', type=str, default='random',
                        choices=['random', '+', '-', '*', '/'],
                        help="Dataset kind (capacity-random or one of the modular ops)")
    parser.add_argument('--patience', type=int, default=100,
                        help='Epochs without improvement before early stop')
    parser.add_argument('--min-delta', type=float, default=1e-4)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device(args.device, args.cpu)

    # The wallow-identifying `dataset_type` collapses '+'/'-'/'*'/'/' into 'modular'
    # for capacity (the actual op is captured by the `operation` field).
    if args.dataset_type_raw == 'random':
        dataset_type, operation = 'random', '/'   # operation is a placeholder when dataset_type='random'
    else:
        dataset_type, operation = 'modular', args.dataset_type_raw

    identifying = build_identifying(
        experiment_type='capacity',
        p=args.p, operation=operation, train_fraction=0.5, split_type='random',
        dataset_type=dataset_type, n_samples=args.n_samples,
        dim=args.dim, depth=args.depth, heads=args.heads,
        dropout=args.dropout, init_scale=args.init_scale,
        lr=args.lr, weight_decay=args.weight_decay,
        beta1=args.beta1, beta2=args.beta2, batch_size=args.batch_size,
        max_epochs=args.epochs, seed=args.seed,
    )

    print(f"=== capacity p={args.p} type={args.dataset_type_raw} seed={args.seed} | "
          f"dim={args.dim} n={args.n_samples} depth={args.depth} heads={args.heads} "
          f"dropout={args.dropout} wd={args.weight_decay} ===")

    try:
        with run_lifecycle(
            identifying,
            force=args.force,
            db_path=args.db_path,
            device=device,
            node_rank=args.node_rank,
        ) as h:
            results = _train_capacity(args, device=device)
            np.savez(
                h.npz_path,
                n_samples=args.n_samples, dim=args.dim, depth=args.depth, heads=args.heads,
                param_count=results['param_count'],
                epochs_trained=results['epochs_trained'],
                final_loss=results['final_loss'], final_acc=results['final_acc'],
                final_bits_per_example=results['final_bits_per_example'],
                total_bits_memorized=results['total_bits_memorized'],
                train_loss_trace=results['train_loss_trace'],
                train_acc_trace=results['train_acc_trace'],
                bits_trace=results['bits_trace'],
            )
            h.finalise(results={
                'param_count': int(results['param_count']),
                'epochs_trained': int(results['epochs_trained']),
                'final_acc': float(results['final_acc']),
                'final_loss': float(results['final_loss']),
                'final_bits_per_example': float(results['final_bits_per_example']),
                'total_bits_memorized': float(results['total_bits_memorized']),
            })
            print(f"  final acc: {results['final_acc']:.3f}, "
                  f"bits/example: {results['final_bits_per_example']:.2f}, "
                  f"total bits: {results['total_bits_memorized']:.0f}")
    except AlreadyCompleted as e:
        print(f"  Skip: run {e.run.uuid} already completed (use --force to re-run).")


if __name__ == '__main__':
    main()
