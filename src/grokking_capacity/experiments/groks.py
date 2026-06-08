"""Run a grokking experiment on modular arithmetic data."""
from __future__ import annotations

import argparse
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from ..data import grokking_data_torch
from ..models import TransformerTorch, count_parameters
from ..registry import build_identifying, run_lifecycle, AlreadyCompleted
from ..utils.cli_args import (
    add_model_args, add_optimizer_args, add_device_args, add_io_args, add_registry_args,
)
from ..utils.checkpoints import load_model
from ..utils.device import get_device


@torch.no_grad()
def squared_param_norm(model) -> float:
    """``V_t = ‖θ‖²`` — whole-model squared parameter norm at the current step."""
    return float(sum((p.detach() ** 2).sum() for p in model.parameters()
                     if p.requires_grad))


class GrokkingTrainer:
    def __init__(self, model, optimizer, n_tokens, batch_size=512, device='cpu',
                 baseline_model=None, norm_log_every=0):
        self.model = model
        self.optimizer = optimizer
        self.n_tokens = n_tokens
        self.batch_size = batch_size
        self.device = device
        self.baseline_model = baseline_model
        if self.baseline_model is not None:
            self.baseline_model = self.baseline_model.to(device)
            self.baseline_model.eval()
        # Norm-contraction channel (V_t logged every `norm_log_every` optimiser
        # steps; 0 disables). steps_per_epoch lets the analysis convert the
        # per-epoch accuracy thresholds into the per-step index the norm uses.
        self.norm_log_every = norm_log_every
        self.norm_steps_trace: list[int] = []
        self.norm_value_trace: list[float] = []
        self.global_step = 0
        self.steps_per_epoch = 0
        self.train_acc_trace: list[float] = []
        self.train_loss_trace: list[float] = []
        self.val_acc_trace: list[float] = []
        self.val_loss_trace: list[float] = []
        self.train_log_probs_trace: list[np.ndarray] = []
        self.val_log_probs_trace: list[np.ndarray] = []
        self.mem_t_trace: list[float] = []
        self.mem_u_trace: list[float] = []

    def _make_batches(self, X, T):
        bs = self.batch_size if self.batch_size != -1 else X.shape[0]
        for i in range(0, X.shape[0], bs):
            yield X[i:i + bs], T[i:i + bs]

    def _log_probs(self, model, X, T) -> torch.Tensor:
        model.eval()
        out = []
        with torch.no_grad():
            for Xb, Tb in self._make_batches(X, T):
                Xb, Tb = Xb.to(self.device), Tb.to(self.device)
                lp = F.log_softmax(model(Xb), dim=-1)
                correct = lp.gather(1, Tb.unsqueeze(1)).squeeze(1) / np.log(2)
                out.append(correct.cpu())
        return torch.cat(out)

    def compute_mem_t(self, log_probs: torch.Tensor) -> float:
        return float((np.log2(self.n_tokens) + log_probs).sum().item())

    def compute_mem_u(self, X, T) -> float:
        if self.baseline_model is None:
            return 0.0
        lp_m = self._log_probs(self.model, X, T)
        lp_b = self._log_probs(self.baseline_model, X, T)
        lp_j = torch.max(lp_m, lp_b)
        return float(((-lp_b) - (-lp_j)).sum().item())

    def evaluate(self, X, T) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        with torch.no_grad():
            for Xb, Tb in self._make_batches(X, T):
                Xb, Tb = Xb.to(self.device), Tb.to(self.device)
                logits = self.model(Xb)
                total_loss += F.cross_entropy(logits, Tb).item() * Xb.size(0)
                total_correct += (torch.argmax(logits, dim=1) == Tb).sum().item()
        return total_loss / X.shape[0], total_correct / X.shape[0]

    def train(self, train_data, val_data, *, epochs: int,
              early_stopping_train: Optional[float], early_stopping_val: Optional[float],
              patience: int, min_delta: float, ignore_memorisation: bool,
              post_grok_epochs: int = 0):
        X_train, T_train = train_data
        X_val, T_val = val_data
        n_train = X_train.shape[0]
        best_val_acc = 0.0
        epochs_without_improvement = 0
        use_patience = patience >= 0
        bs = self.batch_size if self.batch_size != -1 else n_train
        self.steps_per_epoch = (n_train + bs - 1) // bs
        stop_pending = False
        grok_stop_epoch = 0

        bar = tqdm(range(epochs), desc='Training', unit='epoch')
        for epoch in bar:
            self.model.train()
            perm = torch.randperm(n_train)
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
                self.global_step += 1
                if self.norm_log_every and self.global_step % self.norm_log_every == 0:
                    self.norm_steps_trace.append(self.global_step)
                    self.norm_value_trace.append(squared_param_norm(self.model))
                total_loss += loss.item() * Xb.size(0)
                total_correct += (torch.argmax(logits, dim=1) == Tb).sum().item()
            train_loss = total_loss / n_train
            train_acc = total_correct / n_train
            val_loss, val_acc = self.evaluate(X_val, T_val)
            self.train_loss_trace.append(train_loss)
            self.train_acc_trace.append(train_acc)
            self.val_loss_trace.append(val_loss)
            self.val_acc_trace.append(val_acc)

            postfix = {'train_acc': f'{train_acc:.3f}', 'val_acc': f'{val_acc:.3f}'}
            if not ignore_memorisation:
                tlp = self._log_probs(self.model, X_train, T_train)
                vlp = self._log_probs(self.model, X_val, T_val)
                self.train_log_probs_trace.append(tlp.numpy())
                self.val_log_probs_trace.append(vlp.numpy())
                mt = self.compute_mem_t(tlp)
                self.mem_t_trace.append(mt)
                postfix['M_T'] = f'{mt:.1f}'
                if self.baseline_model is not None:
                    mu = self.compute_mem_u(X_train, T_train)
                    self.mem_u_trace.append(mu)
                    postfix['M_U'] = f'{mu:.1f}'
            bar.set_postfix(postfix)

            grok_reached = (early_stopping_val is not None and val_acc >= early_stopping_val
                            and early_stopping_train is not None
                            and train_acc >= early_stopping_train)
            if grok_reached and not stop_pending:
                if post_grok_epochs <= 0:
                    print(f"\nEarly stopping at epoch {epoch + 1}: "
                          f"train={train_acc:.3f}, val={val_acc:.3f}")
                    break
                # Keep training the plateau so the analysis sees a settled V_post.
                stop_pending = True
                grok_stop_epoch = epoch + post_grok_epochs
                print(f"\nGrok at epoch {epoch + 1}; training {post_grok_epochs} "
                      f"more epoch(s) to settle the V_post plateau.")
            if stop_pending and epoch >= grok_stop_epoch:
                print(f"\nStopping at epoch {epoch + 1} ({post_grok_epochs} past grok).")
                break

            if use_patience:
                if val_acc > best_val_acc + min_delta:
                    best_val_acc = val_acc
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"\nEarly stopping: val acc plateau for {patience} epochs")
                    break

        results: Dict = {
            'train_acc': np.array(self.train_acc_trace),
            'val_acc': np.array(self.val_acc_trace),
            'train_loss': np.array(self.train_loss_trace),
            'val_loss': np.array(self.val_loss_trace),
        }
        if self.norm_steps_trace:
            results['norm_steps'] = np.array(self.norm_steps_trace, dtype=np.int64)
            results['norm_values'] = np.array(self.norm_value_trace, dtype=float)
            results['steps_per_epoch'] = int(self.steps_per_epoch)
        if self.train_log_probs_trace:
            results['train_log_probs'] = np.stack(self.train_log_probs_trace, axis=0)
            results['val_log_probs'] = np.stack(self.val_log_probs_trace, axis=0)
            results['mem_t_trace'] = np.array(self.mem_t_trace)
        if self.mem_u_trace:
            results['mem_u_trace'] = np.array(self.mem_u_trace)
        return results


def _grokking_epoch(val_acc: np.ndarray, threshold: float) -> Optional[int]:
    """First 1-indexed epoch where val_acc reaches threshold (in [0, 1])."""
    above = np.where(val_acc >= threshold)[0]
    return int(above[0]) + 1 if len(above) else None


def main():
    parser = argparse.ArgumentParser(description='Run a single grokking experiment')
    add_model_args(parser, dropout_default=0.2)
    add_optimizer_args(parser, weight_decay_default=1.0, epochs_default=200)
    add_device_args(parser)
    add_io_args(parser, data_dir='data/groks')
    add_registry_args(parser)
    parser.add_argument('--operation', type=str, default='/', choices=['*', '/', '+', '-'])
    parser.add_argument('--train-fraction', type=float, default=0.5)
    parser.add_argument('--split-type', type=str, default='random',
                        choices=['random', 'sequential', 'alternating'])
    parser.add_argument('--early-stopping-train', type=float, default=0.99)
    parser.add_argument('--early-stopping-val', type=float, default=0.99)
    parser.add_argument('--no-early-stopping', action='store_true')
    parser.add_argument('--patience', type=int, default=-1)
    parser.add_argument('--min-delta', type=float, default=1e-4)
    parser.add_argument('--ignore-memorisation', action='store_true')
    parser.add_argument('--baseline', type=str, default=None,
                        help='Path to baseline model .pt for M_U computation')
    parser.add_argument('--save-model', action='store_true',
                        help='Save the trained weights into the artefacts directory')
    parser.add_argument('--norm-log-every', type=int, default=20,
                        help='Log the squared parameter norm V_t every N optimiser steps '
                             '(0 disables). The norm-contraction (gc-contraction) channel.')
    parser.add_argument('--post-grok-epochs', type=int, default=0,
                        help='After the early-stop trigger, train this many more epochs to '
                             'settle the post-grok norm plateau V_post (0 = stop at grok).')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device(args.device, args.cpu)

    identifying = build_identifying(
        experiment_type='groks',
        p=args.p, operation=args.operation, train_fraction=args.train_fraction,
        split_type=args.split_type,
        dim=args.dim, depth=args.depth, heads=args.heads,
        dropout=args.dropout, init_scale=args.init_scale,
        lr=args.lr, weight_decay=args.weight_decay,
        beta1=args.beta1, beta2=args.beta2, batch_size=args.batch_size,
        max_epochs=args.epochs, seed=args.seed,
    )

    print(f"=== groks p={args.p} op={args.operation} seed={args.seed} | "
          f"dim={args.dim} depth={args.depth} heads={args.heads} "
          f"dropout={args.dropout} wd={args.weight_decay} ===")

    baseline_model = None
    if args.baseline:
        if not os.path.exists(args.baseline):
            print(f"Error: baseline not found at {args.baseline}")
            return
        baseline_model, _ = load_model(args.baseline, device=device)

    try:
        with run_lifecycle(
            identifying,
            force=args.force,
            db_path=args.db_path,
            device=device,
            node_rank=args.node_rank,
        ) as h:
            Xtr, Ttr, Xva, Tva = grokking_data_torch(
                args.p, op=args.operation, split_type=args.split_type,
                train_fraction=args.train_fraction, device='cpu',
            )
            n_tokens = args.p + 2
            model = TransformerTorch(
                depth=args.depth, dim=args.dim, heads=args.heads,
                n_tokens=n_tokens, seq_len=4, dropout=args.dropout, init_scale=args.init_scale,
            ).to(device)
            param_count = count_parameters(model)
            print(f"  Device: {device}, params: {param_count:,}")

            optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                                     betas=(args.beta1, args.beta2),
                                     weight_decay=args.weight_decay)
            trainer = GrokkingTrainer(model=model, optimizer=optimizer, n_tokens=n_tokens,
                                       batch_size=args.batch_size, device=device,
                                       baseline_model=baseline_model,
                                       norm_log_every=args.norm_log_every)

            early_stop_t = None if args.no_early_stopping else args.early_stopping_train
            early_stop_v = None if args.no_early_stopping else args.early_stopping_val

            results = trainer.train(
                (Xtr, Ttr), (Xva, Tva), epochs=args.epochs,
                early_stopping_train=early_stop_t,
                early_stopping_val=early_stop_v,
                patience=args.patience, min_delta=args.min_delta,
                ignore_memorisation=args.ignore_memorisation,
                post_grok_epochs=args.post_grok_epochs,
            )
            train_acc_pct = results['train_acc'] * 100.0
            val_acc_pct = results['val_acc'] * 100.0

            data_dict = {
                'train_acc': train_acc_pct, 'val_acc': val_acc_pct,
                'dim': args.dim, 'param_count': param_count,
                'depth': args.depth, 'heads': args.heads, 'epochs': args.epochs,
                'p': args.p, 'n_tokens': n_tokens,
                'op': args.operation, 'train_fraction': args.train_fraction,
                'n_train': Xtr.shape[0], 'n_val': Xva.shape[0],
            }
            if 'norm_steps' in results:
                data_dict['norm_steps'] = results['norm_steps']
                data_dict['norm_values'] = results['norm_values']
                data_dict['steps_per_epoch'] = results['steps_per_epoch']
            if 'train_log_probs' in results:
                data_dict['train_log_probs'] = results['train_log_probs']
                data_dict['val_log_probs'] = results['val_log_probs']
                data_dict['mem_t_trace'] = results['mem_t_trace']
            if 'mem_u_trace' in results:
                data_dict['mem_u_trace'] = results['mem_u_trace']
                data_dict['baseline_path'] = args.baseline

            np.savez(h.npz_path, **data_dict)

            if args.save_model:
                model_path = os.path.join(h.artefacts_dir, 'model.pt')
                torch.save({'model_state_dict': model.state_dict(),
                            'metadata': {'dim': args.dim, 'depth': args.depth,
                                         'heads': args.heads, 'dropout': args.dropout,
                                         'p': args.p, 'n_tokens': n_tokens, 'seq_len': 4,
                                         'param_count': param_count}},
                           model_path)
                print(f"  Saved model: {model_path}")

            grok_threshold = args.early_stopping_val if not args.no_early_stopping else 0.99
            grok_epoch = _grokking_epoch(results['val_acc'], grok_threshold)

            h.finalise(results={
                'param_count': int(param_count),
                'epochs_trained': int(len(train_acc_pct)),
                'final_train_acc': float(train_acc_pct[-1]),
                'final_val_acc': float(val_acc_pct[-1]),
                'grokking_epoch': int(grok_epoch) if grok_epoch is not None else None,
                'grokking_threshold': float(grok_threshold),
            })
            summary = (f"  dim={args.dim}: {param_count:,} params, "
                       f"final train={train_acc_pct[-1]:.1f}%, val={val_acc_pct[-1]:.1f}%, "
                       f"grok@{grok_epoch}")
            print(summary)
    except AlreadyCompleted as e:
        print(f"  Skip: run {e.run.uuid} already completed (use --force to re-run).")


if __name__ == '__main__':
    main()
