"""Protect optimizer/RNG comparability and censored-delay interpretation."""
import numpy as np
import pytest
import torch
from grokking_capacity.experiments.groks import GrokkingTrainer
from grokking_capacity.experiments.matched_dropout import train_arithmetic, metrics, boundary_audit
from grokking_capacity.models.discrete import make_model
from grokking_capacity.utils.optim import build_adamw


@pytest.mark.parametrize('kind', ['baseline', 'quantized'])
def test_arithmetic_matches_original_updates_and_rng(kind):
    torch.set_num_threads(1)
    config=dict(model=kind, p=7, dim=8, mlp_dim=12, depth=2, heads=1,
                dropout=.2, levels=2, quantizer_clip=1., quantizer_surrogate='rms_clipped',
                batch_size=8, epochs=4, stop_rule='none')
    gen=torch.Generator().manual_seed(654)
    X=torch.randint(0,9,(31,4),generator=gen); T=torch.randint(0,9,(31,),generator=gen)
    datasets=(X[:19],T[:19],X[19:],T[19:])
    outputs=[]
    for original in (True,False):
        torch.manual_seed(123)
        model=make_model(config)
        optim=build_adamw(model,lr=.001,betas=(.9,.98),weight_decay=1.,mup_base=0,dim=8)
        if original:
            trainer=GrokkingTrainer(model,optim,9,batch_size=8)
            trace=trainer.train(datasets[:2],datasets[2:],epochs=4,early_stopping_train=None,
                                early_stopping_val=None,patience=-1,min_delta=1e-4,ignore_memorisation=True)
        else:
            trace=train_arithmetic(model,optim,datasets,config,'cpu')
        outputs.append((trace,{k:v.clone() for k,v in model.state_dict().items()},torch.get_rng_state()))
    old,new=outputs
    assert all(torch.equal(old[1][k],new[1][k]) for k in old[1])
    assert torch.equal(old[2],new[2])
    for previous,current in [('train_acc','train_online_acc'),('train_loss','train_online_loss'),
                             ('val_acc','val_acc'),('val_loss','val_loss')]:
        np.testing.assert_allclose(old[0][previous],[r[current] for r in new[0]],rtol=1e-6,atol=1e-7)
    assert [r['step'] for r in new[0]]==[3,6,9,12]
    before=torch.get_rng_state().clone()
    audit=boundary_audit(model,X,config,'cpu')
    assert torch.equal(before,torch.get_rng_state())
    assert audit['kind']==('full_width_binary' if kind=='quantized' else 'unrestricted_continuous')


def test_delay_missing_is_not_zero_and_sustained_is_not_single_crossing():
    trace=[dict(epoch=i+1,train_online_acc=a,train_eval_acc=b,val_acc=c) for i,(a,b,c) in enumerate([
        (.95,.99,.96),(.98,.995,.99),(.98,.999,.98),(.98,.999,1.),(.98,.999,1.),(.98,.999,1.)])]
    result=metrics(trace)
    assert result['paper_delay_epochs'] is None
    assert result['eval_delay_epochs']==1
    assert result['val_acc_99_epoch']==2
    assert result['val_99_sustained_epoch']==4
    trace[5]['train_online_acc']=.99
    assert metrics(trace)['paper_delay_epochs']==0
