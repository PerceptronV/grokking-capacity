"""Protect the three-way capacity requirement, not just an A/B equality check."""
import math

import pytest

from grokking_capacity.experiments.discrete_workflow.capacity import assess


def rows(model, capacity):
    return [dict(config=dict(model=model, dim=46 if model == 'baseline' else 96,
                             p=149, operation='/', seed=seed, n_samples=n, max_steps=60000),
                 params=80000 if model == 'baseline' else 330720,
                 source_hashes={'runner': 'same'}, dataset_sha256=f'{seed}-{n}',
                 best_bits=capacity, tail_relative_drift=.01)
            for seed in [61,62] for n in [16384,32768]]


@pytest.mark.parametrize('family', ['discrete', 'quantized', 'categorical'])
def test_match_requires_models_and_fixed_dataset(family):
    target = 11026 * math.log2(151)
    result = assess(rows('baseline', target), rows(family, target * 1.02))
    assert result['accepted']
    # Equal capacities alone must fail when both exceed the dataset target.
    result = assess(rows('baseline', target * 1.2), rows(family, target * 1.2))
    assert not result['accepted']
    assert any('fixed dataset' in x for x in result['issues'])
    # Each within 10% of the dataset is insufficient if they disagree by >10%.
    assert not assess(rows('baseline', target * .91), rows(family, target * 1.09))['accepted']


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
    from grokking_capacity.experiments.discrete_workflow.confirmation import confirmation_runs
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


def test_distinct_full_width_alphabets_cannot_be_pooled():
    a, b = rows('baseline', 80000), rows('categorical', 80000)
    for row in b:
        row['config']['alphabet'] = 4
    b[1]['config']['alphabet'] = 2
    assert not assess(a, b)['accepted']
    assert not assess(a, rows('continuous', 80000))['accepted']


def test_sampling_noise_cannot_hide_categorical_memory(tmp_path):
    import json
    from grokking_capacity.experiments.discrete_workflow.capacity import load_run
    cfg = dict(model='categorical', task='random', p=149, max_steps=60000,
               seed=61, n_samples=16384)
    trace = [dict(step=step, train_bits=80000., train_mode_bits=100000.)
             for step in [0, 36000, 48000, 60000]]
    manifest = dict(config=cfg, param_count=400000, source_hashes={'code': 'hash'},
                    dataset_sha256='data')
    summary = dict(config=cfg, final=trace[-1], best_train_bits=80000.)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    (tmp_path/'trace.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in trace))
    row = load_run(tmp_path)
    assert row['best_bits'] == 100000.
    assert row['best_sampled_bits'] == 80000.
    assert row['best_mode_bits'] == 100000.
    assert row['capacity_evaluation'] == 'max(sampled, hard-mode)'


def test_parameter_ratio_amendment_keeps_dataset_matching():
    a, b = rows('baseline', 80000), rows('quantized', 80000)
    for row in b: row['params'] = 240000
    assert not assess(a, b)['accepted']
    assert assess(a, b, min_parameter_ratio=3.)['accepted']
    for row in b: row['best_bits'] = 100000
    assert not assess(a, b, min_parameter_ratio=3.)['accepted']


def test_confirmation_preserves_categorical_evaluation_and_continuous_control():
    from grokking_capacity.experiments.discrete_workflow.confirmation import confirmation_runs, fullwidth_control
    shared=dict(p=149,operation='/',depth=2,heads=1,dropout=0.,lr=.001,
                weight_decay=1.,beta1=.9,beta2=.98,batch_size=512)
    a={**shared,'model':'baseline','dim':44}
    b={**shared,'model':'categorical','dim':58,'alphabet':16,
       'categorical_evaluation':'hard_mode'}
    runs=confirmation_runs(a,b)
    assert all(cfg.get('categorical_evaluation')=='hard_mode' for name,cfg in runs if name[0]=='B')
    assert all('categorical_evaluation' not in cfg for name,cfg in runs if name[0]!='B')
    assert 'categorical_evaluation' not in fullwidth_control(b)


def test_source_equivalence_is_limited_to_verified_runner_pair():
    from grokking_capacity.experiments.discrete_workflow.compatibility import (
        RUNNER, VERIFIED_RANDOM_RUNNERS, baseline_random_sources_compatible)
    old, new = sorted(VERIFIED_RANDOM_RUNNERS)
    a = {RUNNER: old, 'model': 'unchanged'}
    b = {RUNNER: new, 'model': 'unchanged'}
    assert baseline_random_sources_compatible(a, b)
    assert not baseline_random_sources_compatible(a, {**b, 'model': 'changed'})
    assert not baseline_random_sources_compatible(a, {**b, RUNNER: 'unknown'})
    assert not baseline_random_sources_compatible(a, {RUNNER: new})


def test_unused_quantizer_equivalence_does_not_admit_new_model_or_data_code():
    from grokking_capacity.experiments.discrete_workflow.compatibility import (
        RUNNER, FULLWIDTH, VERIFIED_UNUSED_QUANTIZER_MODULES, VERIFIED_BASELINE_DEPENDENCIES,
        baseline_random_sources_compatible)
    old, new = sorted(VERIFIED_UNUSED_QUANTIZER_MODULES)
    a = {**VERIFIED_BASELINE_DEPENDENCIES, RUNNER: 'same', FULLWIDTH: old}
    b = {**a, FULLWIDTH: new}
    assert baseline_random_sources_compatible(a, b)
    for dependency in VERIFIED_BASELINE_DEPENDENCIES:
        assert not baseline_random_sources_compatible(a, {**b, dependency: 'changed'})
        assert not baseline_random_sources_compatible({**a, dependency: 'changed'}, {**b, dependency: 'changed'})
    assert not baseline_random_sources_compatible(a, {**b, FULLWIDTH: 'unverified'})


@pytest.mark.parametrize('p', [97, 149])
def test_prime_replication_matches_target_and_pairs_binary_control(p):
    import torch
    from grokking_capacity.experiments.discrete_workflow.confirmation import confirmation_runs
    from grokking_capacity.models.discrete import make_model
    from grokking_capacity.models.transformer import count_parameters
    target = (p * (p - 1) // 2) * math.log2(p + 2)
    a, b = rows('baseline', target), rows('quantized', target * 1.02)
    for row in a + b:
        row['config']['p'] = p
    assert assess(a, b, p=p)['accepted']
    if p == 97:
        assert not assess(a, b, p=149)['accepted']
    common = dict(p=p, operation='/', depth=2, heads=1, dropout=0., lr=.001,
                  weight_decay=1., beta1=.9, beta2=.98, batch_size=512, max_steps=120000)
    a = dict(common, model='baseline', dim=28, mlp_dim=112)
    b = dict(common, model='quantized', dim=52, mlp_dim=208, levels=2,
             quantizer_clip=1., quantizer_surrogate='rms_clipped')
    runs = confirmation_runs(a, b, seeds=(73,74,75))
    assert [name for name,_ in runs] == ['A_s73','B_s73','C_s73','B_s74','C_s74','A_s74','C_s75','A_s75','B_s75']
    assert all(cfg['p'] == p and cfg['max_steps'] == 120000 for _,cfg in runs)
    bc = [cfg for name,cfg in runs if name in ('B_s73','C_s73')]
    models = []
    for cfg in bc:
        torch.manual_seed(cfg['seed'])
        models.append(make_model(cfg))
    assert count_parameters(models[0]) == count_parameters(models[1])
    assert models[0].state_dict().keys() == models[1].state_dict().keys()
    assert all(torch.equal(v, models[1].state_dict()[k]) for k,v in models[0].state_dict().items())
    with pytest.raises(ValueError, match='prime'):
        confirmation_runs({**a,'p':147}, {**b,'p':147})
    with pytest.raises(ValueError, match='match'):
        confirmation_runs(a, {**b,'p':101})


def test_replication_requires_independent_arithmetic_audit_before_random(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import grokking_capacity.experiments.discrete_workflow.run_discrete_replication as pipeline
    gate = tmp_path / 'gate'
    gate.mkdir()
    evidence = {'path': str(gate), 'summary_sha256': 'saved'}
    monkeypatch.setattr(pipeline, 'check_gate', lambda path: evidence)
    (gate / 'audit.json').write_text(json.dumps({'arithmetic_gate_passed': False}))
    args = SimpleNamespace(output=tmp_path/'output', gate=gate, resume=False)
    monkeypatch.setattr(pipeline.subprocess, 'Popen', lambda *a, **k: pytest.fail('Training started before the gate passed'))
    with pytest.raises(ValueError, match='independent passing audit'):
        pipeline.replicate(args)
    state = json.loads((args.output/'pipeline.json').read_text())
    assert state['phase'] == 'failed' and not state['commands']
