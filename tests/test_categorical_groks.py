"""Test C-first matching, complete-seed grokking acceptance, and random-speed updates."""
import copy
import numpy as np
import torch
from grokking_capacity.experiments.categorical_groks import models,qualify,train_memorisation,independent_metrics
from grokking_capacity.experiments.multilevel_dropout import make_model,metrics
from grokking_capacity.models.transformer import count_parameters
from grokking_capacity.experiments.capacity import CapacityTrainer
from grokking_capacity.utils.optim import build_adamw


def test_equivalent_models_are_exactly_parameter_matched_and_full_width():
    for width in [60,72,96,120]:
        c,b=models(width);common=dict(p=97,depth=2,heads=1,dropout=.2)
        cm=make_model({**common,**c});bm=make_model({**common,**b})
        assert count_parameters(cm)==count_parameters(bm)
        assert not hasattr(cm,'categorical_states')
        assert len(bm.categorical_states)==2
        assert all(x.dim==width and x.alphabet==8 for x in bm.categorical_states)
        assert b['mlp_dim']==4*width


def test_minimum_requires_all_ten_defined_positive_delays_and_learning():
    r=[dict(config=dict(seed=i),metrics=dict(paper_delay_epochs=i+1),learning_gate_passed=True) for i in range(10)]
    assert qualify(r,list(range(10)))['passed']
    for value in [0,None]:
        changed=copy.deepcopy(r);changed[-1]['metrics']['paper_delay_epochs']=value
        assert not qualify(changed,list(range(10)))['passed']
    r[-1]['learning_gate_passed']=False;assert not qualify(r,list(range(10)))['passed']


def test_memorisation_updates_match_original_capacity_trainer_and_rng():
    torch.set_num_threads(1)
    c=dict(model='baseline',p=7,depth=2,heads=1,dropout=.2,dim=12,mlp_dim=24,batch_size=8,epochs=5)
    g=torch.Generator().manual_seed(59);x=torch.randint(9,(31,4),generator=g);t=torch.randint(9,(31,),generator=g)
    outputs=[]
    for method in ['original','speed']:
        torch.manual_seed(21);m=make_model(c);o=build_adamw(m,lr=.001,betas=(.9,.98),weight_decay=1.,mup_base=0,dim=12)
        if method=='original':
            trainer=CapacityTrainer(m,o,9,8,'cpu');v=trainer.train((x,t),max_epochs=5,patience=5,min_delta=0.,verbose=False)
            trace=v['train_acc_trace']
        else:trace=np.array([r['train_online_acc'] for r in train_memorisation(m,o,(x,t),c,'cpu')])
        outputs.append((m.state_dict(),torch.get_rng_state(),trace))
    assert all(torch.equal(v,outputs[1][0][k]) for k,v in outputs[0][0].items())
    assert torch.equal(outputs[0][1],outputs[1][1]);np.testing.assert_array_equal(outputs[0][2],outputs[1][2])


def test_memorisation_stops_on_online_accuracy_not_eval_loss(monkeypatch):
    class Fake:
        def __init__(self,*args):self.train_acc_trace=[];self.train_loss_trace=[]
        def train(self,*args,**kwargs):
            self.train_acc_trace.append([.8,.989,.99][len(self.train_acc_trace)])
            self.train_loss_trace.append(1.)
            return dict(final_loss=0.,total_bits_memorized=10.)
    import grokking_capacity.experiments.categorical_groks as worker
    monkeypatch.setattr(worker,'CapacityTrainer',Fake)
    t=worker.train_memorisation(None,None,(torch.zeros(9,4),torch.zeros(9)),dict(p=7,batch_size=8,epochs=10),'cpu')
    assert len(t)==3 and t[-1]['step']==6


def test_independent_metrics_do_not_substitute_eval_accuracy_for_online():
    trace=[dict(epoch=i+1,train_online_acc=a,train_eval_acc=b,val_acc=v) for i,(a,b,v) in enumerate([
        (.9,.995,.95),(.98,1.,.985),(.99,1.,.99),(.995,1.,.995),(.995,1.,.99)])]
    result=independent_metrics(trace);assert result==metrics(trace)
    assert result['paper_delay_epochs']==0 and result['eval_delay_epochs']==1
    assert result['val_99_sustained_epoch']==3
