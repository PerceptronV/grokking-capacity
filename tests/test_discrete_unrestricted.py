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
