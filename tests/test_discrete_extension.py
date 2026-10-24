"""Budget-fork integrity on tiny CPU fixtures, not scientific capacity estimates."""
import hashlib
import json

import pytest
import torch

import grokking_capacity.experiments.discrete as runner
from scripts.extend_discrete import extend


@pytest.mark.parametrize('task', ['random', 'modular'])
def test_budget_extension_matches_uninterrupted_training(tmp_path, monkeypatch, task):
    torch.set_num_threads(1)
    # The production gate is covered by test_discrete. This mock isolates state
    # continuation from external experiment artifacts; it never writes a gate.
    monkeypatch.setattr(runner, 'check_gate', lambda *_args: dict(path=str(tmp_path / 'fixture-gate'), summary_sha256='test'))
    cfg = dict(model='baseline', task=task, p=7, dim=8, seed=12, n_samples=40,
               batch_size=16, dropout=.2, stop_on_success=task == 'modular',
               max_steps=4, eval_every=2, checkpoint_every=2)
    parent, child, full = (tmp_path / x for x in ['parent', 'extended', 'full'])
    runner.run(cfg, parent, torch.device('cpu'), gate=tmp_path / 'fixture-gate')
    parent_bytes = (parent / 'checkpoint.pt').read_bytes()
    extend(parent, child, 8, torch.device('cpu'))
    runner.run({**cfg, 'max_steps': 8}, full, torch.device('cpu'), gate=tmp_path / 'fixture-gate')
    a = torch.load(child / 'checkpoint.pt', weights_only=False)
    b = torch.load(full / 'checkpoint.pt', weights_only=False)
    assert all(torch.equal(v, b['model'][k]) for k, v in a['model'].items())
    assert torch.equal(a['batch_rng'], b['batch_rng'])
    assert torch.equal(a['torch_rng'], b['torch_rng'])
    assert a['cursor'] == b['cursor']
    assert a['examples'] == b['examples']
    assert (parent / 'checkpoint.pt').read_bytes() == parent_bytes
    provenance = json.loads((child / 'extension.json').read_text())
    assert provenance['parent_checkpoint_sha256'] == hashlib.sha256(parent_bytes).hexdigest()
    assert provenance['changes'] == {'max_steps': [4, 8]}
    for x, y in zip(a['trace'], b['trace']):
        assert {k: v for k, v in x.items() if k != 'wall_seconds'} == {
            k: v for k, v in y.items() if k != 'wall_seconds'}
    with pytest.raises(FileExistsError):
        extend(parent, child, 8, torch.device('cpu'))
    with pytest.raises(ValueError, match='greater'):
        extend(parent, tmp_path / 'invalid', 4, torch.device('cpu'))
