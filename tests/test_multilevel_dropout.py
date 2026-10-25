"""Protect actual discrete boundaries, deterministic evaluation and C accounting."""
import numpy as np
import pytest
import torch
from grokking_capacity.experiments.multilevel_dropout import (
    make_model, continuous_from_b, boundary_audit, train_arithmetic, late_drift)
from grokking_capacity.models.transformer import count_parameters
from grokking_capacity.experiments.matched_dropout import train_arithmetic as original_train
from grokking_capacity.utils.optim import build_adamw


def config(kind):
    return dict(model=kind,p=7,dim=12,mlp_dim=24,depth=2,heads=1,dropout=.2,
        levels=8,quantizer_clip=1.,quantizer_surrogate='rms_clipped',alphabet=8,
        categorical_temperature=1.,categorical_normalization='slot_layernorm',
        categorical_training_sampling=False,batch_size=8,epochs=3,stop_rule='none')


@pytest.mark.parametrize('kind',['quantized','categorical'])
def test_hard_boundaries_and_gradients(kind):
    torch.set_num_threads(1);torch.manual_seed(17)
    c=config(kind);model=make_model(c);x=torch.randint(0,9,(7,4))
    rng=torch.get_rng_state().clone()
    result=boundary_audit(model,x,c,'cpu')
    assert torch.equal(rng,torch.get_rng_state())
    assert result['train_and_eval_verified'] and model.training
    model.eval();rng=torch.get_rng_state().clone()
    a=model(x);b=model(x)
    assert torch.equal(a,b) and torch.equal(rng,torch.get_rng_state())
    a.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if kind=='categorical':
        assert model.categorical_states[0].to_logits.weight.grad.abs().sum()>0
        assert model.state_embeddings[0].weight.grad.abs().sum()>0
        assert not list(model.categorical_states[0].logit_normalizer.parameters())


@pytest.mark.parametrize('kind',['quantized','categorical'])
def test_continuous_control_is_unrestricted_and_parameter_matched(kind):
    c=config(kind);torch.manual_seed(9);b=make_model(c)
    cc={**c,**continuous_from_b(c)};torch.manual_seed(9);continuous=make_model(cc)
    assert count_parameters(b)==count_parameters(continuous)
    assert not hasattr(continuous,'quantizers') and not hasattr(continuous,'categorical_states')
    assert cc['dim']==c['dim']
    if kind=='quantized':
        assert all(torch.equal(v,continuous.state_dict()[k]) for k,v in b.state_dict().items())


def test_one_hot_projection_is_sum_of_slot_embeddings():
    c=config('categorical');m=make_model(c);d=c['dim'];a=c['alphabet']
    z=torch.nn.functional.one_hot(torch.randint(a,(3,4,d)),a).float()
    weight=m.state_embeddings[0].weight.reshape(d,d,a)
    summed=sum(z[:,:,slot] @ weight[:,slot,:].T for slot in range(d))
    torch.testing.assert_close(m.state_embeddings[0](z.flatten(-2)),summed)


@pytest.mark.parametrize('kind',['baseline','quantized'])
def test_training_and_rng_match_completed_protocol(kind):
    torch.set_num_threads(1);c=config(kind);g=torch.Generator().manual_seed(15)
    x=torch.randint(0,9,(30,4),generator=g);t=torch.randint(0,9,(30,),generator=g)
    outputs=[]
    for train in [original_train,train_arithmetic]:
        torch.manual_seed(14);m=make_model(c)
        opt=build_adamw(m,lr=.001,betas=(.9,.98),weight_decay=1.,mup_base=0,dim=c['dim'])
        trace=train(m,opt,(x[:18],t[:18],x[18:],t[18:]),c,'cpu')
        outputs.append((m.state_dict(),torch.get_rng_state(),trace))
    assert all(torch.equal(v,outputs[1][0][k]) for k,v in outputs[0][0].items())
    assert torch.equal(outputs[0][1],outputs[1][1])
    for a,b in zip(outputs[0][2],outputs[1][2]):
        assert {k:v for k,v in a.items() if k!='wall_seconds'}=={k:v for k,v in b.items() if k!='wall_seconds'}


def test_late_drift_requires_sufficient_positive_history():
    assert late_drift([1.]*199) is None
    assert late_drift([0.]*200) is None
    assert late_drift([1.]*100+[1.02]*100)==pytest.approx(.02)


def test_independent_metrics_keep_censoring_and_missing_delays():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from report_p97_multilevel import independent_metrics,aggregate,bound
    from grokking_capacity.experiments.multilevel_dropout import metrics
    trace=[dict(epoch=i+1,train_online_acc=.8,train_eval_acc=1.,val_acc=v)
           for i,v in enumerate([.7,.99,.7,.99,.99])]
    a=independent_metrics(trace)
    assert a==metrics(trace) and a['val_99_sustained_epoch'] is None
    assert a['paper_delay_epochs'] is None and a['eval_delay_epochs']==1
    assert bound(trace,99)==30
    rows=[]
    for i in range(10):
        value=1000*(i+1) if i<5 else None
        rows.append(dict(paper_delay_updates=None,eval_delay_updates=0,
            val95_updates=value,val99_updates=value,val95_censor_updates=50000,
            val99_censor_updates=50000,final_val=.8))
    agg=aggregate(rows)
    assert agg['val99_updates']['median_all_ten'] is None
    assert agg['val99_updates']['median_all_ten_lower_bound']==27500
    assert agg['paper_delay_updates']['min_observed'] is None
    assert agg['paper_delay_updates']['undefined']==10
