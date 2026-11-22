"""Scientific regression cases: absent roots, root geometry and protocol identity."""
import numpy as np
import pandas as pd
import pytest
from grokking_capacity.analysis.aggregate import find_intersection, intersection_summary
from grokking_capacity.analysis.config_view import ArchKey, ConfigView, IntersectionFigure, StatsConfig
from grokking_capacity.analysis.stats import _ols_fit, _baseline_lr_table, compute_hypothesis_tests


def test_parallel_nonintersecting_curves_have_no_crossing():
    assert find_intersection({1: 1, 2: 1}, {1: 2, 2: 2}) is None
    assert intersection_summary({1: 1, 2: 1}, {1: 2, 2: 2})['status'] == 'mem_faster_throughout'


def test_power_law_crossing_exact_on_different_grids():
    # 100/x = x -> x=10; exact under log-log interpolation.
    assert find_intersection({1: 100, 100: 1}, {2: 2, 30: 30}) == pytest.approx((10, 10))


def test_two_roots_and_direction_are_distinguished():
    result = intersection_summary({1: 10, 10: 1, 100: 10}, {1: 2, 100: 2})
    assert [r['x'] for r in result['roots']] == pytest.approx([5, 20])
    assert intersection_summary({1: 10, 10: 1, 100: 10}, {1: 2, 100: 2}, direction='increasing')['value'] == pytest.approx(20)


def test_root_at_knot_and_endpoint_is_retained_once():
    assert len(intersection_summary({1: 4, 2: 2, 4: 1}, {1: 2, 4: 2})['roots']) == 1
    assert find_intersection({1: 2, 2: 1}, {1: 2, 2: 2}) == pytest.approx((1, 2))


def test_coincident_curves_do_not_identify_a_unique_crossing():
    assert intersection_summary({1: 2, 2: 2}, {1: 2, 2: 2})['status'] == 'coincident_interval'
    assert find_intersection({1: 2, 2: 2}, {1: 2, 2: 2}) is None


def test_grouping_distinguishes_effective_learning_rate_protocol():
    plain = ArchKey.from_dict({'lr': .001, 'mup_base': 0})
    scaled = ArchKey.from_dict({'lr': .001, 'mup_base': 128})
    assert plain != scaled and len({plain, scaled}) == 2
    assert ArchKey.from_dict({}).mup_base == 0


def test_rank_deficient_design_has_correct_degrees_of_freedom():
    x = np.arange(6.)
    _, _, df = _ols_fit(np.column_stack([np.ones(6), x, 2*x]), x + np.array([0,1,0,1,0,1]))
    assert df == 4
    result = _baseline_lr_table(x, x + .1*np.sin(x), pd.DataFrame({'dropout': [.2]*6}), axes=['dropout'])
    assert 'M3' not in result['models']
    assert 'M3_vs_M1' not in result['ftests']


def test_slice_value_is_used_for_named_prime_baseline(tmp_path):
    primes = np.array([97,101,107,113,127,139])
    x = np.array([4.8,5,5.1,5.3,5.4,5.8])
    df = pd.DataFrame({'slice_value': primes, 'predicted_onset_x': 10**x, 'empirical_onset_x': 10**(x+.01*np.arange(6)**2)})
    view = ConfigView('test', tmp_path/'unused', [], [], stats=StatsConfig(n_bootstrap=30,n_permutations=30))
    result = compute_hypothesis_tests(view, IntersectionFigure("intersection", "p", "param_count", "Parameters", "dim", "Width"), df)
    assert result['sufficiency']['models']['M2']['predictors'] == ['p']


def test_rank_permutation_matches_direct_spearman_with_ties():
    from scipy.stats import spearmanr
    from grokking_capacity.analysis.stats import _spearman_and_permutation
    x=np.array([1,1,3,4,5,6.]);y=np.array([4,2,2,3,6,5.])
    obs=spearmanr(x,y).statistic
    rng=np.random.default_rng(23);perm=y.copy();n_ge=0
    for _ in range(100):
        rng.shuffle(perm)
        n_ge += abs(spearmanr(x,perm).statistic) >= abs(obs)-1e-12
    result=_spearman_and_permutation(x,y,n_permutations=100,seed=23)
    assert result['rho']==pytest.approx(obs)
    assert result['p_permutation']==pytest.approx((n_ge+1)/101)
