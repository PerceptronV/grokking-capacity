"""Invariants that distinguish the requested models from old bottlenecks."""
import json

import pytest
import torch
from torch.nn import functional as F

from grokking_capacity.models.discrete import make_model
from grokking_capacity.models.fullwidth_discrete import (
    FullWidthQuantizer, categorical_straight_through,
)
from grokking_capacity.models.transformer import count_parameters
from grokking_capacity.experiments.discrete import (
    DEFAULTS, check_gate, run, source_hashes, summarise,
)


def config(**overrides):
    return {**DEFAULTS, 'model': 'categorical', 'p': 7, 'dim': 8,
            'alphabet': 4, **overrides}


@pytest.mark.parametrize('sample', [False, True])
def test_one_hot_forward_softmax_backward(sample):
    torch.manual_seed(3)
    logits = torch.randn(3, 5, 4, dtype=torch.float64, requires_grad=True)
    weights = torch.randn_like(logits)
    hard = categorical_straight_through(logits, sample=sample)
    assert torch.all((hard == 0) | (hard == 1))
    assert torch.equal(hard.sum(-1), torch.ones(3, 5, dtype=hard.dtype))
    expected = torch.autograd.grad((logits.softmax(-1) * weights).sum(), logits)[0]
    actual = torch.autograd.grad((hard * weights).sum(), logits)[0]
    assert torch.equal(actual, expected)
    if not sample:
        assert torch.equal(hard.argmax(-1), logits.argmax(-1))


def test_sample_distribution_is_categorical_not_argmax():
    torch.manual_seed(4)
    probabilities = torch.tensor([0.1, 0.2, 0.7])
    samples = categorical_straight_through(probabilities.log().expand(100000, -1))
    assert torch.allclose(samples.mean(0), probabilities, atol=0.005, rtol=0)


@pytest.mark.parametrize('levels', [2, 4, 16])
def test_quantization_has_fixed_finite_grid_and_identity_gradient(levels):
    quantize = FullWidthQuantizer(8, levels, 1.5)
    x = torch.linspace(-10, 10, 800, dtype=torch.double).reshape(100, 8).requires_grad_()
    z = quantize(x)
    grid = torch.arange(levels, dtype=x.dtype) * (3 / (levels - 1)) - 1.5
    assert torch.all(torch.isin(z, grid))
    assert z.shape == x.shape
    assert len(list(quantize.parameters())) == 0
    assert torch.equal(torch.autograd.grad(z.sum(), x)[0], torch.ones_like(x))


def test_quantized_and_ordinary_control_have_identical_initial_parameters():
    torch.manual_seed(6)
    b = make_model(config(model='quantized', levels=4))
    torch.manual_seed(6)
    c = make_model(config(model='baseline'))
    assert count_parameters(b) == count_parameters(c)
    assert b.state_dict().keys() == c.state_dict().keys()
    assert all(torch.equal(v, c.state_dict()[k]) for k, v in b.state_dict().items())


@pytest.mark.parametrize('family', ['categorical', 'quantized'])
def test_every_full_width_boundary_is_hard_and_has_no_bypass(family):
    torch.manual_seed(7)
    model = make_model(config(model=family, levels=4))
    tokens = torch.randint(9, (6, 4))
    boundaries = model.categorical_states if family == 'categorical' else model.quantizers
    outputs = []
    handles = [boundary.register_forward_hook(lambda _m, _a, out: outputs.append(out))
               for boundary in boundaries]
    kwargs = {'sample': False} if family == 'categorical' else {}
    model(tokens, **kwargs).square().sum().backward()
    for handle in handles:
        handle.remove()
    assert len(outputs) == 2
    for output in outputs:
        assert output.shape == ((6, 4, 8, 4) if family == 'categorical' else (6, 4, 8))
        if family == 'categorical':
            assert torch.all((output == 0) | (output == 1))
            assert torch.all(output.sum(-1) == 1)
    assert model.embedding.weight.grad.norm() > 0
    for boundary_index, boundary in enumerate(boundaries):
        def constant_state(_m, _a, output):
            if family == 'categorical':
                indices = torch.zeros(output.shape[:-1], dtype=torch.long)
                return F.one_hot(indices, output.shape[-1]).to(output)
            return torch.ones_like(output)
        handle = boundary.register_forward_hook(constant_state)
        try:
            assert torch.equal(model(tokens, **kwargs), model((tokens + 1) % 9, **kwargs))
        finally:
            handle.remove()


def test_concatenated_one_hots_linear_map_equals_sum_of_embeddings():
    model = make_model(config())
    indices = torch.randint(4, (2, 8))
    state = F.one_hot(indices, 4).float()
    matrix = model.state_embeddings[0]
    tables = matrix.weight.T.reshape(8, 4, 8)
    expected = torch.stack([tables[i, indices[:, i]] for i in range(8)]).sum(0)
    assert torch.allclose(matrix(state.flatten(-2)), expected, atol=1e-7)


@pytest.mark.parametrize('family', ['categorical', 'quantized'])
def test_full_width_rejects_compressed_latent_config(family):
    with pytest.raises(ValueError, match='compressed latent_dim'):
        make_model(config(model=family, levels=4, latent_dim=2))


def test_gate_checks_family_and_alphabet(tmp_path):
    cfg = config()
    trace = [dict(step=i, examples=i, epochs=i, wall_seconds=i,
                  train_acc=1., val_acc=1., train_bits=10.) for i in range(3)]
    summary = summarise(cfg, trace, 100)
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    (tmp_path / 'manifest.json').write_text(json.dumps({'source_hashes': source_hashes()}))
    (tmp_path / 'trace.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in trace))
    check_gate(tmp_path, cfg)
    for candidate in [config(alphabet=2), config(model='quantized', levels=4)]:
        with pytest.raises(ValueError, match='architecture'):
            check_gate(tmp_path, candidate)


def test_categorical_checkpoint_resume_preserves_sampled_training(tmp_path, monkeypatch):
    import grokking_capacity.experiments.discrete as runner
    torch.set_num_threads(1)
    cfg = config(max_steps=6, eval_every=2, checkpoint_every=2)
    full, interrupted = tmp_path / 'full', tmp_path / 'interrupted'
    run(cfg, full, torch.device('cpu'))
    original = runner.evaluate
    calls = 0
    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 9:
            raise RuntimeError('interrupt after step 2 checkpoint')
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, 'evaluate', interrupt)
    with pytest.raises(RuntimeError, match='interrupt'):
        run(cfg, interrupted, torch.device('cpu'))
    monkeypatch.setattr(runner, 'evaluate', original)
    path = interrupted / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['pid'] = 999999999
    path.write_text(json.dumps(manifest))
    run(cfg, interrupted, torch.device('cpu'), resume=True)
    a = torch.load(full / 'checkpoint.pt', weights_only=False)
    b = torch.load(interrupted / 'checkpoint.pt', weights_only=False)
    assert all(torch.equal(v, b['model'][k]) for k, v in a['model'].items())
    assert all({k: v for k, v in x.items() if k != 'wall_seconds'} ==
               {k: v for k, v in y.items() if k != 'wall_seconds'}
               for x, y in zip(a['trace'], b['trace']))


def test_clipped_surrogate_has_no_gradient_outside_fixed_range():
    x = torch.tensor([[-2., -.5, .1, 2.]], requires_grad=True)
    identity = FullWidthQuantizer(4, 4, 1., 'identity')(x)
    clipped = FullWidthQuantizer(4, 4, 1., 'clipped')(x)
    assert torch.equal(identity, clipped)
    assert torch.equal(torch.autograd.grad(clipped.sum(), x)[0],
                       torch.tensor([[0., 1., 1., 0.]]))


def test_temperature_uses_the_same_distribution_in_forward_and_backward():
    from grokking_capacity.models.fullwidth_discrete import FullWidthCategoricalState
    state = FullWidthCategoricalState(2, 3, temperature=0.3).double()
    x = torch.tensor([[.2, -.5]], dtype=torch.float64, requires_grad=True)
    logits = state.to_logits(x).unflatten(-1, (2, 3)) / .3
    weights = torch.tensor([[[1., -2., 4.], [3., 5., -1.]]], dtype=x.dtype)
    expected = torch.autograd.grad((logits.softmax(-1) * weights).sum(), x)[0]
    hard = state(x, sample=False)
    actual = torch.autograd.grad((hard * weights).sum(), x)[0]
    assert torch.equal(hard.argmax(-1), logits.argmax(-1))
    assert torch.equal(expected, actual)
    assert torch.all((hard == 0) | (hard == 1))
