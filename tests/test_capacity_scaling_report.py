import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_p97_capacity_scaling import time_summary,slopes,paired_outcomes


def test_censored_times_keep_full_sample_and_valid_median_bounds():
    a=time_summary([1000,2000,3000,4000,5000]+[None]*5)
    assert a['mean'] is None and a['median'] is None and a['censored']==5
    assert a['median_lower_bound']==27500 and a['mean_lower_bound']==26500
    b=time_summary([1000,2000,3000,4000,5000,6000]+[None]*4)
    assert b['mean'] is None and b['median']==5500
    c=time_summary([None]*10)
    assert c['mean_lower_bound']==50000 and c['min_observed'] is None


def test_paired_bootstrap_recovers_known_slopes_and_difference():
    rows=[];seeds=list(range(401,411))
    for i in range(4):
        for role,slope in [('C',-.5),('B',-1.)]:
            for seed in seeds:
                x=1000*2**i
                rows.append(dict(pair_index=i,role=role,seed=seed,estimated_capacity_bits=x,t_gen_updates=(seed-300)*1e5*x**slope))
    fit=slopes(rows,list(range(4)),seeds)
    assert fit['roles']['C']['mean_slope']==pytest.approx(-.5)
    assert fit['roles']['B']['mean_slope']==pytest.approx(-1.)
    assert fit['B_minus_C']['bootstrap_95_interval']==pytest.approx([-.5,-.5])
    assert not slopes(rows,[0,1],seeds)['available']


def test_paired_censoring_only_claims_identifiable_winners_and_ratio_bounds():
    rows=[dict(pair_index=0,seed=401,role='C',t_gen_updates=None),dict(pair_index=0,seed=401,role='B',t_gen_updates=1000)]
    result=paired_outcomes(rows,[dict(index=0)],[401])[0]
    assert result['winner']=='B' and result['B_over_C'] is None and result['B_over_C_upper_bound']==.02
    rows[1]['t_gen_updates']=None
    result=paired_outcomes(rows,[dict(index=0)],[401])[0]
    assert result['winner']=='both_censored' and result['B_over_C_upper_bound'] is None


def test_arithmetic_mean_scaling_is_not_confused_with_mean_log_scaling():
    from report_p97_capacity_scaling import mean_time_slopes
    rows=[];seeds=list(range(401,411))
    for i,x in enumerate([1,2,4,8]):
        for role in ['C','B']:
            for seed in seeds:
                time=100*x**2 if seed==401 else 1000*x**-.5
                rows.append(dict(pair_index=i,role=role,seed=seed,estimated_capacity_bits=x,t_gen_updates=time))
    typical=slopes(rows,list(range(4)),seeds)
    arithmetic=mean_time_slopes(rows,list(range(4)),seeds)
    assert typical['roles']['C']['mean_slope']==pytest.approx(-.25)
    assert arithmetic['roles']['C']['slope']>-.05
    assert arithmetic['B_minus_C']['bootstrap_95_interval']==pytest.approx([0.,0.])
