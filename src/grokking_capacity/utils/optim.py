"""Shared optimiser construction for capacity / speed / groks.

`mup_base > 0` selects lr-only per-layer scaling in the manner of Yang et al.
(2021): embedding and normalisation parameters keep the base learning rate;
every other (matrix) parameter uses lr * mup_base / dim. Initialisation is
left standard. With mup_base == 0 this is exactly the previous plain AdamW.
"""
from __future__ import annotations

import torch.nn as nn
import torch.optim as optim


def split_mup_groups(model: nn.Module):
    vec, mat = [], []
    for name, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        (vec if ('embedding' in name or 'norm' in name) else mat).append(prm)
    return vec, mat


def build_adamw(model: nn.Module, *, lr: float, betas: tuple[float, float],
                weight_decay: float, mup_base: int = 0, dim: int | None = None) -> optim.Optimizer:
    if mup_base and mup_base > 0:
        if dim is None:
            raise ValueError("mup_base > 0 requires dim")
        vec, mat = split_mup_groups(model)
        return optim.AdamW(
            [{'params': vec, 'lr': lr},
             {'params': mat, 'lr': lr * mup_base / dim}],
            betas=betas, weight_decay=weight_decay)
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
