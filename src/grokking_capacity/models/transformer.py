# Author Attribution
# Adapted from Amund Tveit, https://github.com/atveit/torch_grokking (MIT)
# itself a PyTorch port of Jason Stock's MLX code, https://github.com/stockeh/mlx-grokking
# RotaryPositionalEmbeddings vendored verbatim from torchtune v0.1.1
#       https://github.com/pytorch/torchtune (BSD-style license)

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10_000) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._rope_init()

    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        seq_len = x.size(1)
        rope_cache = self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        return x_out.flatten(3).type_as(x)


class RoPETorch(nn.Module):
    """Wraps RotaryPositionalEmbeddings; expects (b, seq, heads, dim_head)."""

    def __init__(self, dim_head, base=1e6):
        super().__init__()
        self.rope = RotaryPositionalEmbeddings(dim_head, base=base)

    def forward(self, x, input_pos=None):
        return self.rope(x, input_pos=input_pos)


class RMSNormTorch(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return normed * self.weight


class AttentionTorch(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = RMSNormTorch(dim)
        self.drop = nn.Dropout(dropout)

        self.wq = nn.Linear(dim, inner_dim, bias=False)
        self.wk = nn.Linear(dim, inner_dim, bias=False)
        self.wv = nn.Linear(dim, inner_dim, bias=False)
        self.wo = nn.Linear(inner_dim, dim, bias=False)

        self.project_out = not (heads == 1 and dim_head == dim)
        if self.project_out:
            self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        else:
            self.to_out = nn.Identity()

        self.rope = RoPETorch(dim_head, base=1e6)

    def forward(self, x, mask=None):
        b, n, d = x.shape
        x = self.norm(x)

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.reshape(b, n, self.heads, -1).transpose(1, 2)
        k = k.reshape(b, n, self.heads, -1).transpose(1, 2)
        v = v.reshape(b, n, self.heads, -1).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        scores = torch.einsum('bhqd,bhkd->bhqk', q, k) * self.scale
        if mask is not None:
            scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum('bhqk,bhkd->bhqd', attn, v)

        out = out.transpose(1, 2).reshape(b, n, -1)
        out = self.wo(out)
        if self.project_out:
            out = self.to_out(out)
        return out


class FeedForwardTorch(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = RMSNormTorch(dim)
        self.drop = nn.Dropout(dropout)
        self.w1 = nn.Linear(dim, mlp_dim, bias=False)
        self.w2 = nn.Linear(mlp_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, mlp_dim, bias=False)

    def forward(self, x):
        x_norm = self.norm(x)
        x_silu = F.silu(self.w1(x_norm))
        x2 = self.drop(x_silu * self.w3(x_norm))
        return self.w2(x2)


class BlockTorch(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, seq_len, dropout):
        super().__init__()
        self.attn = AttentionTorch(dim, heads, dim_head, dropout)
        self.ff = FeedForwardTorch(dim, mlp_dim, dropout)
        self.register_buffer("_mask", self._causal_mask(seq_len), persistent=False)

    @staticmethod
    def _causal_mask(n):
        mask = torch.triu(torch.full((n, n), float('-inf')), diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        mask = self._mask
        x = x + self.attn(x, mask=mask)
        x = x + self.ff(x)
        return x


class TransformerTorch(nn.Module):
    def __init__(
        self,
        depth,
        dim,
        heads,
        n_tokens,
        seq_len,
        dropout=0.,
        pool='cls',
        init_scale: float = 1.0,
    ):
        super().__init__()
        assert pool in {'cls', 'mean'}
        self.pool = pool

        self.embedding = nn.Embedding(n_tokens, dim)
        self.layers = nn.ModuleList([
            BlockTorch(dim, heads, dim // heads, dim * 4, seq_len, dropout)
            for _ in range(depth)
        ])
        self.norm = RMSNormTorch(dim)
        self.out = nn.Linear(dim, n_tokens, bias=False)

        if init_scale != 1.0:
            with torch.no_grad():
                for param in self.parameters():
                    param.mul_(init_scale)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        if self.pool == 'mean':
            x = x.mean(dim=1)
        else:
            x = x[:, -1]
        return self.out(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(
    *,
    depth: int,
    dim: int,
    heads: int,
    p: int,
    dropout: float,
    init_scale: float = 1.0,
    seq_len: int = 4,
    device: str = 'cpu',
) -> TransformerTorch:
    """Build a TransformerTorch and move it to device."""
    return TransformerTorch(
        depth=depth,
        dim=dim,
        heads=heads,
        n_tokens=p + 2,
        seq_len=seq_len,
        dropout=dropout,
        init_scale=init_scale,
    ).to(device)
