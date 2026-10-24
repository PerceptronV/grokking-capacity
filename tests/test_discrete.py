"""Scientific invariants for the discrete-latent experiment."""
import json
import math

import pytest
import torch
from torch import nn

from grokking_capacity.models.discrete import ScalarChannel, make_model
from grokking_capacity.models.transformer import count_parameters
from grokking_capacity.experiments.discrete import (
    DEFAULTS, check_gate, dataset, run, sustained_crossing, tensor_digest,
)


@pytest.mark.parametrize('levels', [2, 4, 16])
def test_exact_hard_forward_and_identity_surrogate(levels):
    channel = ScalarChannel(4, 4, levels, True).double()
    channel.norm = nn.Identity()
    channel.encode = nn.Identity()
    x = torch.tensor([[-1.2, -0.1, 0.4, 1.7]], dtype=torch.double, requires_grad=True)
    hard = channel.latent(x)
    expected = ((torch.tanh(x.detach()) + 1) * (levels - 1) / 2).round()
    expected = expected * (2 / (levels - 1)) - 1
    assert torch.equal(hard, expected)
    weights = torch.tensor([[1., -2., 3., 4.]], dtype=torch.double)
    (hard * weights).sum().backward()
    assert torch.allclose(x.grad, weights * (1 - torch.tanh(x.detach()).square()))
    channel.eval()
    assert torch.equal(channel.latent(x), expected)


def config(**kwargs):
    return {**DEFAULTS, 'model': 'discrete', 'p': 7, 'dim': 8,
            'latent_dim': 4, 'levels': 2, **kwargs}


def test_bc_identical_parameters_and_initialisation():
    torch.manual_seed(11)
    b = make_model(config())
    torch.manual_seed(11)
    c = make_model(config(model='continuous'))
    assert count_parameters(b) == count_parameters(c)
    assert b.state_dict().keys() == c.state_dict().keys()
    assert all(torch.equal(v, c.state_dict()[k]) for k, v in b.state_dict().items())


def test_no_input_bypass_around_final_channel():
    b = make_model(config())
    x = torch.randint(9, (3, 4))
    # Clamp the final message. Changing every input must then change no output.
    handle = b.channels[-1].register_forward_hook(lambda _m, _a, out: torch.ones_like(out))
    try:
        assert torch.equal(b(x), b((x + 1) % 9))
    finally:
        handle.remove()
    b(x).sum().backward()
    assert b.channels[0].encode.weight.grad.norm() > 0
    assert b.embedding.weight.grad.norm() > 0


def test_insufficient_final_code_rejected():
    with pytest.raises(ValueError, match='alphabet'):
        make_model(config(p=97, latent_dim=4))


def test_paired_data_and_arithmetic_answers():
    a, b = dataset(config()), dataset(config(model='continuous'))
    assert tensor_digest(a) == tensor_digest(b)
    x, y, xv, yv = a
    assert not set(map(tuple, x.tolist())) & set(map(tuple, xv.tolist()))
    for xx, yy in [(x, y), (xv, yv)]:
        assert torch.equal(yy, (xx[:, 0] * torch.tensor([pow(int(v), 5, 7) for v in xx[:, 2]])) % 7)


def test_threshold_requires_consecutive_observations():
    trace = [dict(step=i, examples=i * 2, epochs=i, wall_seconds=i, val_acc=v)
             for i, v in enumerate([0.5, 1., 0.9, 1., 1., 1.])]
    assert sustained_crossing(trace, 'val_acc', .99)['step'] == 3
    assert sustained_crossing(trace[:-1], 'val_acc', .99) is None


def test_random_training_refused_without_arithmetic_gate(tmp_path):
    with pytest.raises(ValueError, match='require --gate'):
        run(config(task='random', n_samples=20), tmp_path / 'forbidden', torch.device('cpu'))
    assert not (tmp_path / 'forbidden').exists()


def test_cpu_training_checkpoint_and_metrics(tmp_path):
    torch.set_num_threads(1)
    output = tmp_path / 'train'
    run(config(max_steps=4, eval_every=2, checkpoint_every=2), output, torch.device('cpu'))
    result = json.loads((output / 'summary.json').read_text())
    saved = torch.load(output / 'checkpoint.pt', weights_only=False)
    assert saved['step'] == 4
    assert saved['optimizer']['state']
    assert saved['scientific']['dataset_sha256']
    assert math.isfinite(result['final']['train_loss'])
    with pytest.raises(ValueError, match='has not passed'):
        check_gate(output)
    # A changed run must not overwrite an existing result.
    with pytest.raises(ValueError, match='differs'):
        run(config(max_steps=5, eval_every=2, checkpoint_every=2), output, torch.device('cpu'))


def test_interrupted_resume_preserves_weights_and_batch_order(tmp_path, monkeypatch):
    import grokking_capacity.experiments.discrete as runner
    torch.set_num_threads(1)
    cfg = config(max_steps=8, eval_every=2, checkpoint_every=2, dropout=0.2)
    full, interrupted = tmp_path / 'full', tmp_path / 'interrupted'
    run(cfg, full, torch.device('cpu'))
    original_evaluate = runner.evaluate
    calls = 0

    def interrupted_evaluate(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 7:
            raise RuntimeError('simulated interruption after checkpoint at step 4')
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(runner, 'evaluate', interrupted_evaluate)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        run(cfg, interrupted, torch.device('cpu'))
    monkeypatch.setattr(runner, 'evaluate', original_evaluate)
    # Simulate the terminated worker identity in this temporary test run only.
    manifest_path = interrupted / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['pid'] = 999999999
    manifest_path.write_text(json.dumps(manifest))
    run(cfg, interrupted, torch.device('cpu'), resume=True)
    a = torch.load(full / 'checkpoint.pt', weights_only=False)
    b = torch.load(interrupted / 'checkpoint.pt', weights_only=False)
    assert all(torch.equal(v, b['model'][k]) for k, v in a['model'].items())
    assert a['examples'] == b['examples']
    for x, y in zip(a['trace'], b['trace']):
        assert {k: v for k, v in x.items() if k != 'wall_seconds'} == {
            k: v for k, v in y.items() if k != 'wall_seconds'}
