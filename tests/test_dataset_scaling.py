"""Data leakage, update-budget, stopping and timing checks for variable N."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from grokking_capacity.experiments import dataset_scaling as ds


def test_logarithmic_registration_and_capacity_selection():
    reg = yaml.safe_load((ds.ROOT/'configs/discrete_p97_dataset_scaling.yaml').read_text())
    assert ds.logarithmic_sizes(928,7449,10) == reg['dataset']['sizes']
    assert reg['common']['max_updates'] == 50000 and reg['common']['eval_every'] == 10
    assert reg['common']['dropout'] == .2
    assert len(reg['confirmation']['seeds']) == 10
    # Freeze measured constants so this unit test does not require generated reports.
    old = ds.read(Path(__file__).parent/'fixtures/dataset_scaling_constants.json')
    assert old['source_path'] == reg['models']['source']
    pairs = ds.select_pairs(old['constants'],reg['dataset']['sizes'])
    assert len(pairs)==10
    assert [(p['C']['dim'],p['B']['dim']) for p in pairs]==[(14,16),(16,18),(18,20),(20,24),(22,26),(26,30),(30,34),(34,38),(38,42),(42,48)]
    for pair in pairs:
        assert abs(pair['predicted_relative_mismatch'])<=.1
        for role in ['C','B']:
            assert abs(pair[role+'_estimated_bits']/pair['dataset_bits']-1)<=.1
            c=dict(**reg['common'],**pair[role])
            assert ds.count_parameters(ds.model_protocols.make_model(c))==pair[role+'_parameters']
            assert pair[role]['mlp_dim']==4*pair[role]['dim']
    with pytest.raises(ValueError):
        ds.select_pairs(old['constants'],[512])
    with pytest.raises(ValueError):
        ds.select_pairs(old['constants'],[928],candidates=[1,3])


def test_train_subsets_nested_fixed_holdout_disjoint_and_arithmetic_correct():
    previous = None
    holdout = None
    for n in ds.logarithmic_sizes(928,7449,10):
        X,T,V,U = ds.data(dict(p=97,n_train=n,train_pool_size=7449,seed=501))
        assert len(T) == n and len(U) == 1863
        keys = set(map(tuple,X.tolist()))
        assert not keys.intersection(map(tuple,V.tolist()))
        assert len(keys) == n
        if previous is not None:
            assert torch.equal(X[:len(previous)], previous)
            assert torch.equal(V,holdout)
        assert all(t == (a*pow(b,-1,97))%97 for (a,op,b,eq),t in zip(X.tolist(),T.tolist()))
        previous, holdout = X,V
    assert not torch.equal(holdout,ds.data(dict(p=97,n_train=512,train_pool_size=7449,seed=502))[2])
    with pytest.raises(ValueError):
        ds.data(dict(p=97,n_train=7450,train_pool_size=7449,seed=501))


def test_data_does_not_consume_torch_or_numpy_rng():
    torch.manual_seed(1)
    np.random.seed(1)
    before_t, before_n = torch.get_rng_state(), np.random.get_state()
    ds.data(dict(p=97,n_train=512,train_pool_size=7449,seed=501))
    assert torch.equal(before_t,torch.get_rng_state())
    after_n = np.random.get_state()
    assert before_n[0] == after_n[0] and np.array_equal(before_n[1],after_n[1]) and before_n[2:] == after_n[2:]


@pytest.mark.parametrize('n,expected_examples',[(3,30),(5,25),(8,40)])
def test_training_exact_budget_across_epochs_and_partial_batches(tmp_path,monkeypatch,n,expected_examples):
    torch.set_num_threads(1)
    model = torch.nn.Linear(2,2)
    opt = torch.optim.SGD(model.parameters(),lr=.01)
    datasets = (torch.randn(n,2),torch.zeros(n,dtype=torch.long),torch.randn(3,2),torch.zeros(3,dtype=torch.long))
    # Never trigger stopping; verify the evaluation cadence independently of epoch size.
    monkeypatch.setattr(ds.arithmetic,'evaluate',lambda *args,**kw: dict(loss=1.,acc=.1))
    c = dict(batch_size=4,eval_every=2,max_updates=10)
    trace = ds.train(model,opt,datasets,c,'cpu',tmp_path/'trace.jsonl')
    assert [r['step'] for r in trace] == [2,4,6,8,10]
    assert trace[-1]['examples'] == expected_examples
    assert sum(r['online_window_examples'] for r in trace) == trace[-1]['examples']
    assert trace == [json.loads(line) for line in (tmp_path/'trace.jsonl').read_text().splitlines()]


def test_first_crossing_sustained_crossing_and_joint_stop_are_distinct():
    trace = [dict(step=10*(i+1),val_acc=v,train_online_acc=t,train_eval_acc=1.)
        for i,(v,t) in enumerate([(.99,.9),(.98,.99),(1.,.99),(1.,.9),(1.,.99)])]
    result = ds.trace_metrics(trace)
    assert result['val_acc_99_updates'] == 10
    assert result['val_99_sustained_updates'] == 30
    assert result['train_online_acc_99_updates'] == 20
    assert result['paper_delay_updates'] == 0
    assert not ds.joint_stop(trace[:4]) and ds.joint_stop(trace)
    assert ds.trace_metrics([dict(step=10,val_acc=.1,train_online_acc=.1,train_eval_acc=.1)])['paper_delay_updates'] is None


def test_training_stops_at_first_joint_success(tmp_path,monkeypatch):
    model = torch.nn.Linear(2,2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.copy_(torch.tensor([5.,-5.]))
    opt = torch.optim.SGD(model.parameters(),lr=.01)
    X,T = torch.zeros(3,2),torch.zeros(3,dtype=torch.long)
    monkeypatch.setattr(ds.arithmetic,'evaluate',lambda *args,**kw: dict(loss=0.,acc=1.))
    trace = ds.train(model,opt,(X,T,X,T),dict(batch_size=4,eval_every=2,max_updates=10),'cpu',tmp_path/'trace.jsonl')
    assert [r['step'] for r in trace] == [2,4,6]
    assert ds.joint_stop(trace) and not ds.joint_stop(trace[:2])


@pytest.mark.parametrize('role,width',[('C',14),('B',16)])
def test_smallest_real_models_train_with_finite_ste_gradients(tmp_path,role,width):
    torch.set_num_threads(1)
    c=dict(p=97,depth=2,heads=1,dropout=.2,batch_size=8,eval_every=2,max_updates=4,
        **ds.capacity_matching.model_spec(role,width))
    model=ds.model_protocols.make_model(c)
    before=[p.detach().clone() for p in model.parameters()]
    opt=ds.build_adamw(model,lr=.001,betas=(.9,.98),weight_decay=1.,mup_base=0,dim=width)
    X,T,V,U=ds.data(dict(p=97,n_train=17,train_pool_size=7449,seed=999))
    trace=ds.train(model,opt,(X,T,V[:11],U[:11]),c,'cpu',tmp_path/'trace.jsonl')
    assert trace[-1]['step']==4
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any(not torch.equal(a,b) for a,b in zip(before,model.parameters()))
    boundary=ds.model_protocols.boundary_audit(model,X,c,'cpu')
    assert boundary['kind']==('unrestricted_continuous' if role=='C' else 'full_width_categorical')
    if role=='B':
        assert all(state.to_logits.weight.grad is not None and state.to_logits.weight.grad.abs().sum()>0
            for state in model.categorical_states)
