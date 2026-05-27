"""Measure memorisation speed (steps to saturation) on random-target data."""
from __future__ import annotations

import argparse
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from ..data import random_target_data_torch
from ..models import TransformerTorch, count_parameters
from ..registry import build_identifying, run_lifecycle, AlreadyCompleted
from ..utils.cli_args import (
    add_model_args, add_optimizer_args, add_device_args, add_io_args, add_registry_args,
)
from ..utils.device import get_device


class SpeedTrainer:
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer,
                 batch_size: int = 512, device: str = 'cpu'):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.device = device
        self.train_loss_trace: list[float] = []
        self.train_acc_trace: list[float] = []
        self.steps_trace: list[int] = []

    def _make_batches(self, X, T):
        bs = self.batch_size if self.batch_size != -1 else X.shape[0]
        for i in range(0, X.shape[0], bs):
            yield X[i:i + bs], T[i:i + bs]

    def train(self, train_data, *, max_epochs: int, saturation_threshold: float,
              patience: int, verbose: bool) -> Dict:
        X_train, T_train = train_data
        n = X_train.shape[0]
        total_steps = 0
        saturation_step: int | None = None
        epochs_above_threshold = 0

        epoch_iter = tqdm(range(max_epochs), desc='Training', unit='epoch') if verbose else range(max_epochs)
        avg_loss = float('inf')
        avg_acc = 0.0
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
                total_steps += 1
            avg_loss = total_loss / n
            avg_acc = (total_correct / n) * 100.0
            self.train_loss_trace.append(avg_loss)
            self.train_acc_trace.append(avg_acc)
            self.steps_trace.append(total_steps)
            if verbose:
                epoch_iter.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{avg_acc:.2f}%', 'steps': total_steps})

            if avg_acc >= saturation_threshold:
                epochs_above_threshold += 1
                if saturation_step is None:
                    saturation_step = total_steps
                if epochs_above_threshold >= patience:
                    if verbose:
                        print(f"\nSaturation at step {saturation_step} (epoch {epoch + 1 - patience})")
                    break
            else:
                epochs_above_threshold = 0
                saturation_step = None

        if saturation_step is None:
            saturation_step = total_steps
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
        }


def _train_speed(args, *, device: str) -> Dict:
    n_tokens = args.p + 2
    X_train, T_train = random_target_data_torch(args.n_samples, args.p, seq_len=4, device='cpu')
    model = TransformerTorch(
        depth=args.depth, dim=args.dim, heads=args.heads,
        n_tokens=n_tokens, seq_len=4, dropout=args.dropout, init_scale=args.init_scale,
    ).to(device)
    param_count = count_parameters(model)
    print(f"  n_samples={args.n_samples} dim={args.dim} params={param_count:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                             betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    trainer = SpeedTrainer(model=model, optimizer=optimizer,
                           batch_size=args.batch_size, device=device)
    results = trainer.train(
        (X_train, T_train),
        max_epochs=args.epochs, saturation_threshold=args.saturation_threshold,
        patience=args.patience, verbose=True,
    )
    results['param_count'] = param_count
    bs = args.batch_size if args.batch_size != -1 else args.n_samples
    steps_per_epoch = (args.n_samples + bs - 1) // bs
    results['saturation_epoch'] = float(results['saturation_step']) / steps_per_epoch
    bits_per_example = float(np.log2(n_tokens))
    results['dataset_bits'] = args.n_samples * bits_per_example
    return results


def main():
    parser = argparse.ArgumentParser(description='Measure learning speed for one (dim, n_samples).')
    add_model_args(parser, dropout_default=0.2)
    add_optimizer_args(parser, weight_decay_default=0.01, epochs_default=5000)
    add_device_args(parser)
    add_io_args(parser, data_dir='data/speed')
    add_registry_args(parser)
    parser.add_argument('--operation', type=str, default='/', choices=['*', '/', '+', '-'],
                        help='Modular op this speed run is matched to (informational).')
    parser.add_argument('--train-fraction', type=float, default=0.5,
                        help='Train fraction this run is matched to (informational).')
    parser.add_argument('--split-type', type=str, default='random',
                        choices=['random', 'sequential', 'alternating'])
    parser.add_argument('--n-samples', type=int, required=True, help='Dataset size')
    parser.add_argument('--saturation-threshold', type=float, default=99.0,
                        help='Accuracy threshold (%%) to consider saturated')
    parser.add_argument('--patience', type=int, default=0,
                        help='Epochs to confirm saturation')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device(args.device, args.cpu)

    identifying = build_identifying(
        experiment_type='speed',
        p=args.p, operation=args.operation, train_fraction=args.train_fraction,
        split_type=args.split_type, n_samples=args.n_samples,
        dim=args.dim, depth=args.depth, heads=args.heads,
        dropout=args.dropout, init_scale=args.init_scale,
        lr=args.lr, weight_decay=args.weight_decay,
        beta1=args.beta1, beta2=args.beta2, batch_size=args.batch_size,
        max_epochs=args.epochs, seed=args.seed,
    )

    print(f"=== speed p={args.p} op={args.operation} seed={args.seed} | "
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
            results = _train_speed(args, device=device)
            np.savez(
                h.npz_path,
                n_samples=args.n_samples, p=args.p, dim=args.dim,
                depth=args.depth, heads=args.heads,
                param_count=results['param_count'],
                epochs_trained=results['epochs_trained'],
                total_steps=results['total_steps'],
                saturation_step=results['saturation_step'],
                saturation_epoch=results['saturation_epoch'],
                final_loss=results['final_loss'], final_acc=results['final_acc'],
                dataset_bits=results['dataset_bits'],
                saturated=results['saturated'],
                train_loss_trace=results['train_loss_trace'],
                train_acc_trace=results['train_acc_trace'],
                steps_trace=results['steps_trace'],
            )
            h.finalise(results={
                'param_count': int(results['param_count']),
                'epochs_trained': int(results['epochs_trained']),
                'saturation_step': int(results['saturation_step']),
                'saturation_epoch': float(results['saturation_epoch']),
                'final_acc': float(results['final_acc']),
                'final_loss': float(results['final_loss']),
                'dataset_bits': float(results['dataset_bits']),
                'saturated': bool(results['saturated']),
            })
            print(f"  saturated: {results['saturated']}, "
                  f"saturation_step={results['saturation_step']:,}, "
                  f"final acc={results['final_acc']:.2f}%")
    except AlreadyCompleted as e:
        print(f"  Skip: run {e.run.uuid} already completed (use --force to re-run).")


if __name__ == '__main__':
    main()
