"""Utility helpers preserve saved models and independently frozen workflows."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from grokking_capacity.models import TransformerTorch
from grokking_capacity.utils.artifacts import clean, save_json
from grokking_capacity.utils.checkpoints import load_model, save_model
from grokking_capacity.utils.workflow_paths import (
    ROOT, SOURCES, catalog, frozen_source, package_sources, script_source,
)


def test_metadata_and_workflow_imports_do_not_load_training_dependencies():
    code = '''import sys
import grokking_capacity.utils as utils
from grokking_capacity.utils import compute_dataset_size_bits
from grokking_capacity.utils.workflow_paths import ROOT, SOURCES
from grokking_capacity.workflows import catalog
assert 'torch' not in sys.modules
assert compute_dataset_size_bits(13)[0] == 78
assert SOURCES['character_growth'] == catalog()['mechanisms']['character_growth']['path']
assert 'load_model' in dir(utils)
from grokking_capacity.utils import load_model
from grokking_capacity.utils.checkpoints import load_model as direct
assert load_model is direct
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize('optional', [{}, {'pool': 'mean', 'mlp_dim': 18}])
@pytest.mark.parametrize('filename', ['model.pt', Path('nested/model.pt')])
def test_checkpoint_roundtrip_retains_architecture_and_predictions(tmp_path, monkeypatch, optional, filename):
    monkeypatch.chdir(tmp_path)
    metadata = dict(depth=1, dim=8, heads=2, n_tokens=9, seq_len=4, dropout=0., **optional)
    model = TransformerTorch(**metadata).eval()
    tokens = torch.tensor([[0, 7, 1, 8], [2, 7, 5, 8]])
    expected = model(tokens)
    save_model(model, metadata, filename)
    restored, saved = load_model(filename)
    assert saved == metadata
    assert not restored.training
    assert restored.state_dict().keys() == model.state_dict().keys()
    assert torch.equal(restored(tokens), expected)


def test_json_supports_numpy_scalars_arrays_and_string_paths(tmp_path):
    data = {'count': np.array(3), 'ok': np.array(True),
            'missing': np.array(np.nan), 'values': np.array([1., np.inf])}
    expected = {'count': 3, 'ok': True, 'missing': None, 'values': [1., None]}
    assert clean(data) == expected
    path = tmp_path / 'values.json'
    save_json(str(path), data)
    assert path.read_text() == json.dumps(expected, indent=2, allow_nan=False) + '\n'


def test_source_aliases_follow_catalog_and_keep_historical_names():
    for commands in catalog().values():
        for name, entry in commands.items():
            assert SOURCES[name] == entry['path']
            assert script_source(name + '.py') == ROOT / entry['path']
    assert script_source('cone_data.py').name == 'cone_data.py'
    assert script_source('discrete_source_compatibility.py').name == 'compatibility.py'
    with pytest.raises(KeyError):
        script_source('not_a_workflow.py')


def test_frozen_worker_never_uses_live_code_even_if_it_exists(tmp_path):
    name = 'run_cone_ten_seeds.py'
    assert script_source(name).is_file()
    with pytest.raises(FileNotFoundError):
        frozen_source(tmp_path, name)
    historical = tmp_path / 'source/scripts' / name
    historical.parent.mkdir(parents=True)
    historical.write_text('original source')
    assert frozen_source(tmp_path, name) == historical


def test_frozen_package_has_catalog_and_archived_source_evidence(tmp_path):
    snapshot = tmp_path / 'campaign/source'
    for path in package_sources():
        destination = snapshot / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    env = dict(os.environ, PYTHONPATH=str(snapshot / 'src'))
    code = '''from pathlib import Path
from grokking_capacity.utils.workflow_paths import ROOT, catalog, script_source, frozen_source
from grokking_capacity.experiments.registered import ARCHIVE, read, verify_sources
assert str(ROOT) == str(Path.cwd() / 'campaign/source')
assert script_source('run_cone_ten_seeds.py').is_file()
assert frozen_source(Path.cwd() / 'campaign', 'run_cone_ten_seeds.py') == script_source('run_cone_ten_seeds.py')
assert catalog()['comparison']['dataset']['path']
manifest = read(ARCHIVE / 'manifest.json')
expected = {path: manifest['files'][path] for path in manifest['source_sets']['capacity_scaling']}
assert verify_sources(expected) == 'legacy-v1'
'''
    subprocess.run([sys.executable, '-c', code], cwd=tmp_path, env=env,
                   check=True, capture_output=True, text=True)


def test_cli_grid_hashes_package_metadata_without_frozen_controllers():
    from grokking_capacity.experiments.campaigns.cli_grid import source_hashes
    hashes = source_hashes()
    assert 'src/grokking_capacity/workflow_catalog.json' in hashes
    assert 'src/grokking_capacity/experiments/legacy_v1/manifest.json' in hashes
    assert 'src/grokking_capacity/experiments/legacy_v1/sources.zip' in hashes
    assert not any('/frozen/' in path for path in hashes)
