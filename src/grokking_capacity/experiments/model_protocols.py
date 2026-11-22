"""Discrete model construction and boundary checks, independent of campaigns."""
import numpy as np
import torch
from ..models.discrete import make_model as original_make_model
from ..models.fullwidth_discrete import FullWidthCategoricalTransformer


class ModeCategoricalTransformer(FullWidthCategoricalTransformer):
    """Explicit train sampling policy; evaluation always uses hard argmax."""
    def __init__(self, *, training_sampling=False, **kwargs):
        super().__init__(**kwargs)
        self.training_sampling = training_sampling

    def forward(self, tokens, *, sample=None):
        if sample is None:
            sample = self.training and self.training_sampling
        return super().forward(tokens, sample=sample)


def make_model(config):
    if config['model'] != 'categorical':
        return original_make_model(config)
    assert 'latent_dim' not in config
    return ModeCategoricalTransformer(
        depth=config['depth'], dim=config['dim'], heads=config['heads'],
        n_tokens=config['p']+2, seq_len=4, dropout=config['dropout'],
        mlp_dim=config['mlp_dim'], alphabet=config['alphabet'],
        categorical_temperature=config['categorical_temperature'],
        categorical_normalization=config['categorical_normalization'],
        training_sampling=config['categorical_training_sampling'])


def continuous_from_b(config):
    """Ordinary continuous C: same width/depth, parameter count within 0.1%."""
    result = {k:config[k] for k in ('dim','mlp_dim')}
    result['model'] = 'baseline'
    if config['model'] == 'categorical':
        # Per block B adds two d x (d*A) matrices. Reallocate those parameters
        # to the three FF matrices in an ordinary C; no narrow adapter remains.
        result['mlp_dim'] += round(2*config['alphabet']*config['dim']/3)
    return result


def late_drift(bits):
    if len(bits)<200 or float(np.mean(bits[-200:-100]))<=0:
        return None
    return float(abs(np.mean(bits[-100:])/np.mean(bits[-200:-100])-1))


def boundary_audit(model, tokens, config, device):
    if config['model']=='baseline':
        assert not hasattr(model,'quantizers') and not hasattr(model,'channels')
        assert not hasattr(model,'categorical_states')
        return dict(kind='unrestricted_continuous')
    captured=[]
    def hook(module,args,output):
        if config['model']=='quantized':
            assert output.shape[-1]==config['dim']
            grid=torch.arange(config['levels'],device=output.device,dtype=output.dtype)
            grid=grid*(2*config['quantizer_clip']/(config['levels']-1))-config['quantizer_clip']
            assert (output[...,None]==grid).any(-1).all()
        else:
            assert output.shape[-2:]==(config['dim'],config['alphabet'])
            assert ((output==0)|(output==1)).all() and (output.sum(-1)==1).all()
        captured.append(tuple(output.shape))
    modules=model.quantizers if config['model']=='quantized' else model.categorical_states
    hooks=[m.register_forward_hook(hook) for m in modules]
    was_training=model.training
    try:
        devices=[torch.device(device).index or 0] if str(device).startswith('cuda') else []
        with torch.random.fork_rng(devices=devices),torch.no_grad():
            for training in (False,True):
                model.train(training);model(tokens[:32].to(device))
        assert len(captured)==2*config['depth']
    finally:
        for handle in hooks:handle.remove()
        model.train(was_training)
    return dict(kind='full_width_'+config['model'],train_and_eval_verified=True,shapes=captured)
