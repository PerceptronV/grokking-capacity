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
