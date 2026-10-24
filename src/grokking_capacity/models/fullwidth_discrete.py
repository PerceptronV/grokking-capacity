"""Full-width discrete states at every transformer block boundary.

Categorical states use DreamerV2's sampled one-hot forward and softmax backward
estimator (Algorithm 1, https://arxiv.org/abs/2010.02193). There are `dim`
categorical variables, NOT `dim // alphabet` variables. The next linear map is
an embedding of their concatenated one-hots. No continuous state crosses a
boundary. This adapts the categorical mechanism, not Dreamer's recurrent model.

Scalar quantization keeps every residual coordinate and adds no parameters.
The old learned low-dimensional channel lives separately for reproducibility.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .transformer import TransformerTorch


def categorical_straight_through(logits, *, sample=True):
    probabilities = logits.softmax(dim=-1)
    if sample:
        # Exponential races draw exactly from the categorical distribution.
        # Gradients come only from probabilities, not this discrete draw.
        with torch.no_grad():
            noise = torch.empty_like(probabilities).exponential_()
            indices = (probabilities / noise).argmax(dim=-1)
    else:
        indices = logits.argmax(dim=-1)
    hard = F.one_hot(indices, logits.shape[-1]).to(logits)
    # Parentheses preserve exactly hard forward values, including in float32.
    return hard + (probabilities - probabilities.detach())


class FullWidthQuantizer(nn.Module):
    def __init__(self, dim, levels, clip, surrogate='identity'):
        super().__init__()
        if not isinstance(levels, int) or levels < 2:
            raise ValueError('Quantization requires at least two integer levels')
        if not math.isfinite(clip) or clip <= 0:
            raise ValueError('Quantization requires a positive finite fixed clip')
        if surrogate not in {'identity', 'clipped'}:
            raise ValueError('Unknown quantizer surrogate')
        self.surrogate = surrogate
        self.dim, self.levels, self.clip = dim, levels, float(clip)

    @property
    def bits_per_token(self):
        return self.dim * math.log2(self.levels)

    def forward(self, x):
        if x.shape[-1] != self.dim:
            raise ValueError('Quantize the entire residual width')
        indices = ((x.detach().clamp(-self.clip, self.clip) + self.clip)
                   * ((self.levels - 1) / (2 * self.clip))).round()
        hard = indices * (2 * self.clip / (self.levels - 1)) - self.clip
        # Both estimators ignore rounding. The clipped variant keeps the
        # clamp's derivative (zero outside the range), preventing unbounded
        # activation growth driven by a surrogate insensitive to saturation.
        # Range is fixed: no input-dependent scale crosses the boundary.
        soft = x.clamp(-self.clip, self.clip) if self.surrogate == 'clipped' else x
        return hard + (soft - soft.detach())


class FullWidthQuantizedTransformer(TransformerTorch):
    def __init__(self, *, levels, quantizer_clip=1.0, quantizer_surrogate='identity', **kwargs):
        super().__init__(**kwargs)
        self.quantizers = nn.ModuleList([
            FullWidthQuantizer(kwargs['dim'], levels, quantizer_clip, quantizer_surrogate)
            for _ in self.layers
        ])

    def forward(self, tokens):
        x = self.embedding(tokens)
        for layer, quantize in zip(self.layers, self.quantizers):
            x = quantize(layer(x))
        x = self.norm(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, -1]
        return self.out(x)


class FullWidthCategoricalState(nn.Module):
    def __init__(self, dim, alphabet, temperature=1.0):
        super().__init__()
        if not isinstance(alphabet, int) or alphabet < 2:
            raise ValueError('Categorical alphabet must contain at least two symbols')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Categorical temperature must be positive and finite')
        self.temperature = float(temperature)
        self.dim, self.alphabet = dim, alphabet
        self.to_logits = nn.Linear(dim, dim * alphabet, bias=False)

    @property
    def bits_per_token(self):
        return self.dim * math.log2(self.alphabet)

    def forward(self, x, *, sample=True):
        logits = self.to_logits(x).unflatten(-1, (self.dim, self.alphabet))
        return categorical_straight_through(logits / self.temperature, sample=sample)


class FullWidthCategoricalTransformer(TransformerTorch):
    def __init__(self, *, alphabet, categorical_temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        dim = kwargs['dim']
        self.categorical_states = nn.ModuleList([
            FullWidthCategoricalState(dim, alphabet, categorical_temperature) for _ in self.layers
        ])
        # Applied only to a hard one-hot state. These matrices implement the
        # next stage's embedding lookup/sum; they have no other input.
        self.state_embeddings = nn.ModuleList([
            nn.Linear(dim * alphabet, dim, bias=False) for _ in self.layers
        ])
        scale = kwargs.get('init_scale', 1.0)
        if scale != 1:
            with torch.no_grad():
                for module in [self.categorical_states, self.state_embeddings]:
                    for parameter in module.parameters():
                        parameter.mul_(scale)

    def forward(self, tokens, *, sample=True):
        x = self.embedding(tokens)
        for layer, categorical, embed in zip(
                self.layers, self.categorical_states, self.state_embeddings):
            state = categorical(layer(x), sample=sample)
            x = embed(state.flatten(-2))
        x = self.norm(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, -1]
        return self.out(x)
