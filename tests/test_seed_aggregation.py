"""Scientific contracts for seed aggregation, censoring and onset definitions."""
from dataclasses import replace
import numpy as np
import pytest
from grokking_capacity.analysis import aggregate as a
from grokking_capacity.analysis.config_view import _parse_intersection_figures


def test_even_median_is_not_strict_majority():
    x=[0]*5+[2]*5
    assert a.summarize_delays(x,'min')['value']==0
    assert a.summarize_delays(x,'median')['value']==1
    assert a.summarize_delays(x,'mean')['value']==1
    assert a.summarize_delays([0]*9+[10],'mean')['value']==1


def test_unresolved_seed_never_dropped_or_imputed_as_observed_delay():
    assert a.summarize_delays([0,1,None],'min')['value']==0
    for metric in ['mean','median']:
        r=a.summarize_delays([0,1,None],metric)
        assert r['value'] is None and r['n_unresolved']==1
    # Seven known zeros fix the median, even with three unknown outcomes.
    r=a.summarize_delays([0]*7+[None]*3,'median')
    assert r['value']==0 and r['n_unresolved']==3
    r=a.summarize_delays([1,2,None],'mean')
    assert r['state']=='unresolved' and r['sign']=='positive'


def test_quantile_and_custom_reducers():
    x=[0.,1.,2.,3.]
    for q in [0,.1,.5,.9,1]:
        assert a.reduce_values(x,f'quantile:{q}')==pytest.approx(np.quantile(x,q))
    assert a.reduce_values(x,lambda v:np.mean(v[1:-1]))==1.5
    assert a.summarize_delays([0,None],lambda v:np.mean(v))['value'] is None
    assert a.summarize_delays([0,None],'quantile:0')['value']==0
    assert a.summarize_delays([0,None],'quantile:1')['upper_unbounded']
    for metric in ['bogus','quantile:2']:
        with pytest.raises(ValueError):a.summarize_delays([0,None],metric)
    with pytest.raises(ValueError):a.reduce_values([1,np.nan],'mean')


def test_onset_retains_uncertain_widths_and_labels_boundaries():
    assert a.onset_summary({10:0,20:None,30:2})['status']=='unresolved'
    assert a.find_grokking_onset({10:None,20:0,30:2})==30
    assert a.onset_summary({10:1,20:2})['status']=='left_censored'
    assert a.onset_summary({10:1,20:0})['status']=='no_positive_tail'
    assert a.find_grokking_onset({10:0,20:1,30:2},threshold=1)==30


def test_censoring_default_is_unknown_and_validation_first_is_zero():
    tr=np.zeros(10);tr[2:]=100;va=np.zeros(10)
    assert a.compute_delays([dict(param_count=1,train_acc=tr,val_acc=va)])==[(1.,None)]
    assert a.compute_delays([dict(param_count=1,train_acc=va,val_acc=tr)])==[(1.,0.)]
    assert a.compute_delays([dict(param_count=1,train_acc=tr,val_acc=va)],censoring='legacy_lower_bound')==[(1.,8.)]


def test_config_parses_independent_metrics():
    f=_parse_intersection_figures({'analysis':{'intersection_figures':[dict(slice_field='p',x_field='param_count',delay_aggregation='quantile:0.25',timescale_aggregation='median',delay_positive_threshold=5)]}})[0]
    assert f.delay_aggregation=='quantile:0.25'
    assert f.timescale_aggregation=='median'
    assert f.delay_positive_threshold==5 and f.delay_censoring=='unresolved'


def test_aggregation_and_clipping_do_not_commute():
    signed=np.array([-10.,2.,2.])
    assert np.maximum(signed,0).mean()>0
    assert max(0,signed.mean())==0
    x=np.array([1,2,100]);y=np.array([0,99,100])
    assert np.median(x-y)!=np.median(x)-np.median(y)


def test_minimum_summary_is_invariant_to_seed_order_and_not_to_seed_count():
    assert a.summarize_delays([3,1,2], 'min')['value']==1
    assert a.summarize_delays([2,3,1], 'min')['value']==1
    assert a.summarize_delays([3,1,2,0], 'min')['value']==0
    with pytest.raises(ValueError):a.onset_summary({1:float('nan')})


def test_forecast_empirical_onset_uses_requested_metric():
    from grokking_capacity.analysis.forecast_onset import empirical_onset
    rows=[dict(x=x,seed=s,delay=d) for x,ds in [(10,[0,0,0]),(20,[0,1,2]),(30,[1,2,3])]
          for s,d in enumerate(ds)]
    assert empirical_onset(rows,metric='min')==30
    assert empirical_onset(rows,metric='median')==20
    assert empirical_onset(rows,metric='mean')==20
    assert empirical_onset(rows,metric='mean',threshold=1)==30


def test_missing_trace_retains_its_seed_slot(tmp_path):
    from grokking_capacity.analysis.config_view import ArchKey,ArchGroup,IntersectionFigure
    from grokking_capacity.analysis.plots import _delay_records_for_slice
    from grokking_capacity.analysis.stats import _empirical_onset_details
    path=tmp_path/'observed.npz';tr=np.array([0,100,100]);va=np.array([0,0,100])
    np.savez(path,train_acc=tr,val_acc=va)
    base=dict(p=97,dim=32,param_count=1000,experiment_type='groks')
    group=ArchGroup(key=ArchKey.from_dict({}),capacity_runs=[],speed_runs=[],groks_runs=[
        {**base,'seed':42,'uuid':'observed_test','npz_path':str(path)},
        {**base,'seed':43,'uuid':'missing_test','npz_path':str(tmp_path/'missing.npz')}])
    f=IntersectionFigure(name='x',slice_field='p',x_field='param_count',x_label='P',colour_field='dim',colour_label='d')
    for metric in ['min','median','mean']:
        ff=replace(f,delay_aggregation=metric)
        r=_delay_records_for_slice(group,ff,97)[0]
        assert r['n_seeds']==2 and r['n_unresolved']==1 and r['delay'] is None
        assert _empirical_onset_details(group,ff,97)['status']=='unresolved'
