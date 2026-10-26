import importlib.util
from pathlib import Path
import pytest
p=Path(__file__).resolve().parents[1]/'scripts/report_p97_dataset_scaling.py'
spec=importlib.util.spec_from_file_location('dataset_report',p)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def test_censored_times_are_not_complete_means_or_medians():
    s=r.summarize([10,None,None,None],100)
    assert s['mean'] is None and s['median'] is None
    assert s['capped_mean']==77.5 and s['mean_lower_bound']==77.5
    s=r.summarize([10,20,30,None],100)
    assert s['mean'] is None and s['median']==25


def test_crossing_requires_adjacent_complete_points_and_reports_log_bracket():
    c=r.crossing_brackets([10,100],[2,-2],[True,True])
    assert c[0]['log_interpolated_n']==pytest.approx(10**1.5)
    assert c[0]['direction']=='C faster to B faster'
    assert not r.crossing_brackets([10,100],[2,-2],[True,False])
    assert not r.crossing_brackets([10,100],[2,1],[True,True])


def test_paired_bootstrap_preserves_exact_seed_offset_cancellation():
    rows=[]
    for seed in range(10):
        for n,diff in [(100,10),(1000,-10)]:
            for role,delta in [('C',0),('B',diff)]:
                rows.append(dict(seed=seed,n_train=n,role=role,t_gen_updates=100+seed+delta,
                    final_learning=True,joint_stop=True,paper_delay_updates=0))
    a=r.analyze(rows,[100,1000],list(range(10)),50000)
    assert a['resolved_crossing'] and len(a['observed_mean_crossing_brackets'])==1
    assert a['simultaneous_band_radius']==0
    assert a['sizes'][0]['difference_ci_low']==a['sizes'][0]['difference_ci_high']==10
    rows[0]['t_gen_updates']=None
    b=r.analyze(rows,[100,1000],list(range(10)),50000)
    assert not b['observed_mean_crossing_brackets'] and not b['resolved_crossing']
