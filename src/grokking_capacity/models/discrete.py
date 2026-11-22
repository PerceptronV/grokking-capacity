"""Finite scalar channels between transformer blocks, with identity STE.

The continuous control has exactly the same parameters and bounded latent; only
rounding differs. No residual path crosses a channel. This is a bound on the
per-example message alphabet, NOT a bound on memorisation in model weights.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from .transformer import TransformerTorch, RMSNormTorch, _scale_parameters


class ScalarChannel(nn.Module):
    def __init__(self, dim: int, latent_dim: int, levels: int, discrete: bool):
        super().__init__()
        if latent_dim < 1 or levels < 2:
            raise ValueError("latent_dim >= 1 and levels >= 2 required")
        self.latent_dim = latent_dim
        self.levels = levels
        self.discrete = discrete
        self.norm = RMSNormTorch(dim)
        self.encode = nn.Linear(dim, latent_dim, bias=False)
        self.decode = nn.Linear(latent_dim, dim, bias=False)

    @property
    def bits_per_token(self):
        return self.latent_dim * math.log2(self.levels)

    def latent(self, x):
        z = torch.tanh(self.encode(self.norm(x)))
        if not self.discrete:
            return z
        hard = ((z + 1) * ((self.levels - 1) / 2)).round()
        hard = hard * (2 / (self.levels - 1)) - 1
        # Exact hard forward values, identity surrogate wrt bounded z.
        # Parentheses matter: z - z.detach() is exactly zero in the forward.
        return hard.detach() + (z - z.detach())

    def forward(self, x):
        return self.decode(self.latent(x))


class ChannelTransformer(TransformerTorch):
    def __init__(self, *, latent_dim: int, levels: int = 2,
                 discrete: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.channels = nn.ModuleList([
            ScalarChannel(kwargs["dim"], latent_dim, levels, discrete)
            for _ in self.layers
        ])
        # Apply init_scale to added parameters consistently with the base model.
        _scale_parameters(kwargs.get("init_scale", 1.0), self.channels)

    def forward(self, tokens):
        x = self.embedding(tokens)
        for layer, channel in zip(self.layers, self.channels):
            x = channel(layer(x))
        return self._readout(x)


def make_model(config: dict):
    spec = dict(depth=config.get("depth", 2), dim=config["dim"],
                heads=config.get("heads", 1), n_tokens=config["p"] + 2,
                seq_len=4, dropout=config.get("dropout", 0.0), mlp_dim=config.get("mlp_dim"))
    if config["model"] == "baseline":
        return TransformerTorch(**spec)
    if config["model"] in {"quantized", "categorical"}:
        from .fullwidth_discrete import (
            FullWidthQuantizedTransformer, FullWidthCategoricalTransformer,
        )
        if "latent_dim" in config:
            raise ValueError("Full-width models do not accept a compressed latent_dim")
        if config["model"] == "quantized":
            return FullWidthQuantizedTransformer(
                **spec, levels=config["levels"], quantizer_clip=config.get("quantizer_clip", 1.0),
                quantizer_surrogate=config.get("quantizer_surrogate", "identity"))
        return FullWidthCategoricalTransformer(**spec, alphabet=config["alphabet"],
            categorical_temperature=config.get("categorical_temperature", 1.0),
            categorical_normalization=config.get("categorical_normalization", "none"))
    if config["model"] not in {"discrete", "continuous"}:
        raise ValueError("Unknown model")
    if config["latent_dim"] * math.log2(config["levels"]) < math.log2(config["p"]):
        raise ValueError("Final code alphabet cannot represent all arithmetic answers")
    return ChannelTransformer(**spec, latent_dim=config["latent_dim"],
                              levels=config["levels"],
                              discrete=config["model"] == "discrete")
