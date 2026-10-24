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


def test_confirmation_uses_new_paired_modular_seeds_and_rotates_order():
    from scripts.confirm_discrete import confirmation_runs
    shared=dict(p=149,operation='/',depth=2,heads=1,dropout=0.,lr=.001,
                weight_decay=1.,beta1=.9,beta2=.98,batch_size=512)
    a={**shared,'model':'baseline','dim':44,'seed':61,'n_samples':16384,'task':'random'}
    b={**shared,'model':'discrete','dim':96,'latent_dim':14,'levels':64,'seed':61,'n_samples':16384}
    result=confirmation_runs(a,b)
    assert [name for name,_ in result]==['A_s43','B_s43','C_s43','B_s44','C_s44','A_s44','C_s45','A_s45','B_s45']
    for name,cfg in result:
        assert cfg['task']=='modular' and cfg['train_fraction']==.5
        assert cfg['seed'] in [43,44,45] and 'n_samples' not in cfg
        assert cfg['max_steps']==60000 and cfg['eval_every']==100
    for seed in [43,44,45]:
        configs=[cfg for _,cfg in result if cfg['seed']==seed]
        assert all(all(cfg[k]==shared[k] for k in shared) for cfg in configs)
        bc=[cfg for cfg in configs if cfg['model']!='baseline']
        assert {k:v for k,v in bc[0].items() if k!='model'}=={k:v for k,v in bc[1].items() if k!='model'}
