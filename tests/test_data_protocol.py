"""Dataset simplification must preserve seeded examples and draw order exactly."""
from pathlib import Path
import types
import zipfile

import numpy as np
import pytest
import torch

from grokking_capacity.data import modular


@pytest.fixture(scope='module')
def original():
    archive = (Path(__file__).resolve().parents[1] / 'src/grokking_capacity'
               / 'experiments/legacy_v1/sources.zip')
    module = types.ModuleType('_original_modular_data')
    with zipfile.ZipFile(archive) as source:
        path = 'src/grokking_capacity/data/modular.py'
        exec(compile(source.read(path), '<archived ' + path + '>', 'exec'), module.__dict__)
    return module


def assert_numpy_state_equal(actual, expected):
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


@pytest.mark.parametrize('operation', ['/', '*', '+', '-'])
@pytest.mark.parametrize('split', ['random', 'sequential', 'alternating'])
def test_modular_examples_order_labels_and_rng_match_archive(original, operation, split):
    for prime in [7, 97]:
        for seed, fraction in [(0, .3), (42, .5), (12, .7)]:
            kwargs = dict(p=prime, op=operation, split_type=split, train_fraction=fraction)
            np.random.seed(seed)
            expected = original.grokking_data_torch(**kwargs)
            expected_state = np.random.get_state()
            np.random.seed(seed)
            torch_state = torch.get_rng_state()
            actual = modular.grokking_data_torch(**kwargs)
            assert_numpy_state_equal(np.random.get_state(), expected_state)
            assert torch.equal(torch.get_rng_state(), torch_state)
            for old, new in zip(expected, actual):
                assert old.dtype == new.dtype == torch.long
                assert torch.equal(old, new)
            inputs = torch.cat([actual[0], actual[2]])
            assert torch.all(inputs[:, 1] == prime)
            assert torch.all(inputs[:, 3] == prime + 1)
            assert len(torch.unique(inputs, dim=0)) == len(inputs)


@pytest.mark.parametrize('fraction', [0.0, 1.0])
def test_random_split_boundaries_match_archive(original, fraction):
    for prime in [2, 7]:
        for operation in ['/', '+']:
            np.random.seed(42)
            expected = original.grokking_data_torch(prime, operation, train_fraction=fraction)
            expected_state = np.random.get_state()
            np.random.seed(42)
            actual = modular.grokking_data_torch(prime, operation, train_fraction=fraction)
            assert_numpy_state_equal(np.random.get_state(), expected_state)
            for old, new in zip(expected, actual):
                assert old.dtype == new.dtype
                assert torch.equal(old, new)


@pytest.mark.parametrize('split', ['random', 'sequential', 'alternating'])
def test_index_partition_shapes_dtypes_and_rng_are_preserved(original, split):
    for fraction in [0.0, .1, .5, .9, 1.0]:
        np.random.seed(9)
        expected = original.split_indices(42, fraction, 7, '/', split)
        expected_state = np.random.get_state()
        np.random.seed(9)
        actual = modular.split_indices(42, fraction, 7, '/', split)
        assert_numpy_state_equal(np.random.get_state(), expected_state)
        for old, new in zip(expected, actual):
            assert old.dtype == new.dtype
            np.testing.assert_array_equal(old, new)


@pytest.mark.parametrize('count,length', [(0, 4), (1, 4), (23, 7)])
def test_random_targets_keep_full_vocabulary_and_rng_order(original, count, length):
    np.random.seed(5)
    expected = original.random_target_data_torch(count, 7, seq_len=length)
    expected_state = np.random.get_state()
    np.random.seed(5)
    actual = modular.random_target_data_torch(count, 7, seq_len=length)
    assert_numpy_state_equal(np.random.get_state(), expected_state)
    for old, new in zip(expected, actual):
        assert torch.equal(old, new)
    assert actual[0].shape == (count, length)
    assert actual[1].shape == (count,)


def test_unsupported_operations_and_split_names_remain_errors():
    with pytest.raises(ValueError, match='Unsupported operation'):
        modular.grokking_data_torch(7, '^')
    with pytest.raises(ValueError, match='Invalid type'):
        modular.split_indices(42, .5, 7, '/', 'unknown')
