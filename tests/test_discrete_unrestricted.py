import torch

from scripts.confirm_discrete_unrestricted import unrestricted_runs,shared_initial_tensors
from grokking_capacity.experiments.discrete import DEFAULTS
from grokking_capacity.models.discrete import make_model
from grokking_capacity.models.transformer import count_parameters


def test_unrestricted_control_has_no_channel_and_shares_base_initialisation():
    a={**DEFAULTS,'p':149,'model':'baseline','dim':44}
    b={**a,'model':'discrete','dim':96,'latent_dim':14,'levels':64}
    runs=unrestricted_runs(a,b)
    assert [name for name,cfg in runs]==['A_s43','B_s43','C_s43','B_s44','C_s44','A_s44','C_s45','A_s45','B_s45']
    for seed in (43,44,45):
        arms={name[0]:cfg for name,cfg in runs if cfg['seed']==seed}
        c=make_model(arms['C'])
        assert not hasattr(c,'channels')
        assert 'latent_dim' not in arms['C'] and 'levels' not in arms['C']
        assert count_parameters(c)==324384
        assert count_parameters(make_model(arms['B']))==329952
        assert shared_initial_tensors(arms['B'],arms['C'])
        assert arms['A']['dim']==44 and arms['B']['latent_dim']==14


def test_fullwidth_controls_are_ordinary_and_count_all_categorical_parameters():
    from scripts.confirm_discrete import fullwidth_control, confirmation_runs
    from grokking_capacity.models.fullwidth_discrete import FullWidthCategoricalTransformer
    from grokking_capacity.models.transformer import TransformerTorch
    shared = dict(p=149, operation='/', depth=2, heads=1, dropout=0., lr=.001,
                  weight_decay=1., beta1=.9, beta2=.98, batch_size=512, seed=61)
    a = dict(shared, model='baseline', dim=44)
    for family in ['quantized', 'categorical']:
        for k in [2, 4, 16]:
            b = dict(shared, model=family, dim=96,
                     **({'alphabet': k} if family == 'categorical' else
                        {'levels': k, 'quantizer_clip': 1., 'quantizer_surrogate': 'clipped'}))
            c = fullwidth_control(b)
            bm, cm = make_model(b), make_model(c)
            assert type(cm) is TransformerTorch
            assert abs(count_parameters(cm) / count_parameters(bm) - 1) <= .017
            if family == 'categorical':
                assert isinstance(bm, FullWidthCategoricalTransformer)
                assert count_parameters(bm) == 324384 + 4 * 96 * 96 * k
            else:
                assert c['dim'] == 96
                assert count_parameters(cm) == count_parameters(bm)
            runs = confirmation_runs(a, b)
            assert len(runs) == 9
            assert all(cfg['model'] == 'baseline' for name, cfg in runs if name[0] in 'AC')
            assert all(cfg['dim'] == c['dim'] for name, cfg in runs if name[0] == 'C')


def test_custom_feedforward_control_keeps_architecture_and_parameter_pairing():
    from grokking_capacity.experiments.discrete import DEFAULTS
    from scripts.confirm_discrete import confirmation_runs, fullwidth_control
    shared = {**DEFAULTS, 'p':149, 'dim':44, 'model':'baseline'}
    b = {**shared, 'model':'quantized', 'dim':72, 'mlp_dim':576, 'levels':2,
         'quantizer_clip':1., 'quantizer_surrogate':'clipped'}
    c = fullwidth_control(b)
    assert c['dim'] == 72 and c['mlp_dim'] == 576
    assert count_parameters(make_model(b)) == count_parameters(make_model(c)) == 312408
    runs = confirmation_runs(shared,b)
    for name,cfg in runs:
        if name.startswith(('B_', 'C_')):
            assert cfg['dim'] == 72 and cfg['mlp_dim'] == 576
        else:
            assert cfg['dim'] == 44 and 'mlp_dim' not in cfg


def test_registered_confirmation_seeds_are_paired_and_rotate_without_data_overrides():
    import pytest
    from scripts.confirm_discrete import confirmation_runs
    a = {**DEFAULTS, 'p':149, 'dim':44, 'model':'baseline', 'seed':61,
         'data_seed':42, 'task':'random', 'n_samples':16384}
    b = {**a, 'model':'quantized', 'dim':80, 'mlp_dim':320, 'levels':2,
         'quantizer_clip':1., 'quantizer_surrogate':'rms_clipped'}
    runs = confirmation_runs(a, b, seeds=(73,74,75))
    assert [name for name, _ in runs] == [
        'A_s73','B_s73','C_s73','B_s74','C_s74','A_s74','C_s75','A_s75','B_s75']
    assert all(cfg['task']=='modular' and 'data_seed' not in cfg and 'n_samples' not in cfg
               for _,cfg in runs)
    for seed in (73,74,75):
        assert {name[0] for name,cfg in runs if cfg['seed']==seed} == set('ABC')
    for invalid in [(73,73,75),(73,74),(-1,74,75),(73.,74,75)]:
        with pytest.raises(ValueError, match='three distinct'):
            confirmation_runs(a,b,seeds=invalid)
