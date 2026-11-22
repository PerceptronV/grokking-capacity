"""Shared model plumbing preserves the archived training/checkpoint protocol."""
import sys
import types
import zipfile
from pathlib import Path

import pytest
import torch

from grokking_capacity.models import discrete, fullwidth_discrete, transformer


@pytest.fixture(scope='module')
def originals():
    """Load the immutable pre-refactor models under a separate package name."""
    prefix = '_model_protocol_before_refactor'
    package = types.ModuleType(prefix)
    package.__path__ = []
    modules = {prefix: package}
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    archive = (Path(__file__).resolve().parents[1] / 'src/grokking_capacity'
               / 'experiments/legacy_v1/sources.zip')
    try:
        sys.modules[prefix] = package
        with zipfile.ZipFile(archive) as source:
            for name in ['transformer', 'fullwidth_discrete', 'discrete']:
                qualified = prefix + '.' + name
                module = types.ModuleType(qualified)
                module.__package__ = prefix
                modules[qualified] = module
                sys.modules[qualified] = module
                setattr(package, name, module)
                path = 'src/grokking_capacity/models/' + name + '.py'
                exec(compile(source.read(path), '<archived ' + path + '>', 'exec'),
                     module.__dict__)
        yield package
    finally:
        for name in modules:
            sys.modules.pop(name, None)
        torch.set_num_threads(previous_threads)


def assert_tensors_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        if actual[name] is None or expected[name] is None:
            assert actual[name] is expected[name], name
        else:
            assert torch.equal(actual[name], expected[name]), name


@pytest.mark.parametrize('family', ['baseline', 'discrete', 'continuous', 'quantized', 'categorical'])
@pytest.mark.parametrize('heads,pool,scale', [(1, 'cls', 1.0), (2, 'mean', .37)])
def test_model_initialization_forward_backward_and_rng_match_archive(originals, family, heads, pool, scale):
    constructors = {
        'baseline': ('transformer', 'TransformerTorch', {}),
        'discrete': ('discrete', 'ChannelTransformer', {'latent_dim': 4, 'levels': 4}),
        'continuous': ('discrete', 'ChannelTransformer', {'latent_dim': 4, 'levels': 4, 'discrete': False}),
        'quantized': ('fullwidth_discrete', 'FullWidthQuantizedTransformer',
                      {'levels': 4, 'quantizer_surrogate': 'rms_clipped'}),
        'categorical': ('fullwidth_discrete', 'FullWidthCategoricalTransformer',
                        {'alphabet': 4, 'categorical_normalization': 'rms_slot_layernorm'}),
    }
    module, name, options = constructors[family]
    config = dict(depth=2, dim=8, heads=heads, n_tokens=9, seq_len=4,
                  dropout=.2, pool=pool, init_scale=scale, mlp_dim=18, **options)
    old_type = getattr(getattr(originals, module), name)
    new_type = getattr({'transformer': transformer, 'discrete': discrete,
                        'fullwidth_discrete': fullwidth_discrete}[module], name)
    torch.manual_seed(891)
    old = old_type(**config)
    old_rng = torch.get_rng_state()
    torch.manual_seed(891)
    new = new_type(**config)
    assert torch.equal(torch.get_rng_state(), old_rng)
    assert_tensors_equal(new.state_dict(), old.state_dict())
    assert_tensors_equal(dict(new.named_buffers()), dict(old.named_buffers()))
    tokens = torch.tensor([[0, 7, 1, 8], [2, 7, 5, 8], [3, 7, 6, 8]])
    sampling_options = [{'sample': False}, {'sample': True}] if family == 'categorical' else [{}]
    for training in [False, True]:
        new.train(training)
        old.train(training)
        for kwargs in sampling_options:
            old.zero_grad()
            new.zero_grad()
            torch.manual_seed(895)
            expected = old(tokens, **kwargs)
            expected.square().sum().backward()
            old_rng = torch.get_rng_state()
            torch.manual_seed(895)
            actual = new(tokens, **kwargs)
            actual.square().sum().backward()
            assert torch.equal(actual, expected)
            assert torch.equal(torch.get_rng_state(), old_rng)
            assert_tensors_equal({n: p.grad for n, p in new.named_parameters()},
                                 {n: p.grad for n, p in old.named_parameters()})
