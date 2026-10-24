"""Protect the three-way capacity requirement, not just an A/B equality check."""
import math

import pytest

from scripts.assess_discrete_match import assess


def rows(model, capacity):
    return [dict(config=dict(model=model, dim=46 if model == 'baseline' else 96,
                             p=149, operation='/', seed=seed, n_samples=n, max_steps=60000),
                 params=80000 if model == 'baseline' else 330720,
                 source_hashes={'runner': 'same'}, dataset_sha256=f'{seed}-{n}',
                 best_bits=capacity, tail_relative_drift=.01)
            for seed in [61,62] for n in [16384,32768]]


def test_match_requires_models_and_fixed_dataset():
    target = 11026 * math.log2(151)
    result = assess(rows('baseline', target), rows('discrete', target * 1.02))
    assert result['accepted']
    # Equal capacities alone must fail when both exceed the dataset target.
    result = assess(rows('baseline', target * 1.2), rows('discrete', target * 1.2))
    assert not result['accepted']
    assert any('fixed dataset' in x for x in result['issues'])
    # Each within 10% of the dataset is insufficient if they disagree by >10%.
    assert not assess(rows('baseline', target * .91), rows('discrete', target * 1.09))['accepted']


def test_requires_paired_repeated_complete_evidence():
    a,b=rows('baseline',80000),rows('discrete',80000)
    assert not assess(a[:-1],b)['accepted']
    b[0]['dataset_sha256']='different'
    assert not assess(a,b)['accepted']
    b=rows('discrete',80000)
    b[0]['tail_relative_drift']=.2
    assert not assess(a,b)['accepted']
    a,b=rows('baseline',80000),rows('discrete',80000)
    for r in a+b: r['config']['max_steps']=20000
    assert not assess(a,b)['accepted']


def test_rejects_composite_division_modulus():
    with pytest.raises(ValueError,match='prime'):
        assess(rows('baseline',80000),rows('discrete',80000),p=147)
