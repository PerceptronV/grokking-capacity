"""Guard capacity-constant matching against parameter-matching substitutions."""
import pytest
import torch
from grokking_capacity.experiments.capacity_scaling import model_spec,parameter_count,fit_constant,match_pairs
from grokking_capacity.experiments.multilevel_dropout import make_model,boundary_audit
from grokking_capacity.models.transformer import count_parameters


def test_count_formula_and_real_boundaries_for_all_sweep_widths():
    torch.set_num_threads(1)
    for role in ['C','B']:
        for width in range(32,86,2):
            c=dict(p=97,depth=2,heads=1,dropout=.2,**model_spec(role,width))
            m=make_model(c);assert count_parameters(m)==parameter_count(role,width)
            assert c['mlp_dim']==4*width
        a=boundary_audit(m,torch.randint(99,(3,4)),c,'cpu')
        assert a['kind']==('unrestricted_continuous' if role=='C' else 'full_width_categorical')


def test_constant_preserves_both_seeds_instead_of_selecting_best_seed():
    points=[dict(parameters=p,capacity_bits=k*p,seed=seed) for p in [100,200] for seed,k in [(61,.2),(62,.4)]]
    fit=fit_constant(points)
    assert fit['bits_per_parameter']==pytest.approx(.3)
    assert fit['per_seed_constants']==pytest.approx({'61':.2,'62':.4})
    assert len(fit['points'])==4
    with pytest.raises(ValueError):fit_constant(points[:2])


def test_ten_pairs_match_predicted_capacity_with_different_parameter_counts():
    pairs=match_pairs({'C':.8,'B':.3},range(32,69,4),range(24,161,2))
    assert len(pairs)==10 and len({p['B']['dim'] for p in pairs})==10
    for p in pairs:
        assert abs(p['B_estimated_bits']/p['C_estimated_bits']-1)<=.05
        assert p['B_parameters']>2.5*p['C_parameters']
        assert p['B']['alphabet']==8 and p['C']['model']=='baseline'
    with pytest.raises(ValueError):match_pairs({'C':.8,'B':.01},range(32,69,4),[24,26])
