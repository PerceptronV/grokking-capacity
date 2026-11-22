"""Character diagnostics preserve training and distinguish absent crossings."""
import ast
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from grokking_capacity.analysis.characters import CharacterTrace, character_order, primitive_root, spectral_power
from grokking_capacity.analysis.mechanisms import character_alpha_width, character_lambda_sweep
from grokking_capacity.experiments import character


def test_character_basis_and_surgery_preserve_unselected_rows():
    from grokking_capacity.analysis.mechanisms.character_ablation import spectral_surgery
    rng=np.random.default_rng(8)
    for p in [7,11,97]:
        order=character_order(p)
        assert sorted(order)==list(range(1,p))
        weights=rng.normal(size=(p+2,8)).astype(np.float32)
        original=weights.copy()
        power=spectral_power(weights,p)
        changed=spectral_surgery(weights,p,list(order),drop=np.array([1]))
        assert np.array_equal(weights,original)
        assert np.array_equal(changed[[0,p,p+1]],weights[[0,p,p+1]])
        assert spectral_power(changed,p)[0] < power[0]*1e-12
        np.testing.assert_allclose(spectral_power(changed,p)[1:],power[1:],rtol=1e-6)
    with pytest.raises(ValueError):primitive_root(9)


def test_missing_grokking_and_takeoff_never_become_epoch_zero(tmp_path):
    path=tmp_path/'trace.npz'
    np.savez(path,epochs=np.arange(6),out_pow=np.ones((6,4)),train_acc=np.full(6,.8),val_acc=np.full(6,.3))
    result=character_lambda_sweep.analyse(path)
    assert result['grok_epoch'] is None and result['train_sat_epoch'] is None
    assert result['takeoff_1p5x'] is None and result['windows']=={}
    assert character_alpha_width.analyse(path)['grok'] is None


def test_empty_winner_cohort_has_no_fitted_growth_rate(tmp_path):
    path=tmp_path/'trace.npz'
    np.savez(path,epochs=np.arange(6),out_pow=np.ones((6,4)),train_acc=np.ones(6),val_acc=np.ones(6))
    result=character_alpha_width.analyse(path)
    assert result['grok']==0 and result['takeoff_1p5x'] is None
    assert result['winner_amp_rate_escape'] is None and result['winner_amp_rate_post'] is None


def test_import_never_parses_arguments_or_starts_training():
    code="import sys;sys.argv=['test','--not-a-training-option'];import grokking_capacity.analysis.mechanisms.character_growth;print('imported')"
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True)
    assert result.stdout.strip()=='imported'


def test_original_training_protocol_and_rng_match(tmp_path,monkeypatch):
    import zipfile
    from grokking_capacity.experiments.registered import ARCHIVE
    with zipfile.ZipFile(ARCHIVE/'sources.zip') as archive:
        original=archive.read('src/grokking_capacity/analysis/mechanisms/character_growth.py').decode()
    args=character.parser().parse_args(['--p','7','--dim','4','--epochs','2','--every','1',
        '--batch-size','4','--device','cpu','--wd','.3','--out',str(tmp_path/'new.npz')])
    torch.set_num_threads(1)
    ns=dict(np=np,torch=torch,F=torch.nn.functional,optim=torch.optim,
        grokking_data_torch=character.grokking_data_torch,TransformerTorch=character.TransformerTorch,
        P=7,DIM=4,SEED=42,EPOCHS=2,EVERY=1,BATCH=4,DEV='cpu',ARGS=args,OUT=str(tmp_path/'old.npz'),
        ORDER=list(character_order(7)))
    method=next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name=='mult_pow')
    exec(compile(ast.Module(body=[method],type_ignores=[]),'<original spectrum>','exec'),ns)
    exec(compile(original[original.index('np.random.seed(SEED)'):],'<original training>','exec'),ns)
    rng=torch.get_rng_state().clone()
    original_model=ns['model']
    created=[]
    factory=character.TransformerTorch
    def capture(**kwargs):
        model=factory(**kwargs);created.append(model);return model
    monkeypatch.setattr(character,'TransformerTorch',capture)
    character.run(args)
    assert torch.equal(rng,torch.get_rng_state())
    for key,value in original_model.state_dict().items():
        assert torch.equal(value,created[0].state_dict()[key])
    with np.load(tmp_path/'old.npz') as old, np.load(tmp_path/'new.npz') as new:
        for key in old.files:np.testing.assert_array_equal(old[key],new[key])
