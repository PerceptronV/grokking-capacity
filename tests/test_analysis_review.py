"""Regressions for package imports and legacy matching behavior."""
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from wallow import register

from grokking_capacity.analysis.capacity_constant import fit_capacity_line, fit_capacity_slope
from grokking_capacity.analysis import matching
from grokking_capacity.registry import build_identifying, get_store


def test_analysis_imports_do_not_run_workflows(tmp_path):
    # Preload library dependencies, then forbid experiment/analysis side effects.
    # A fresh subprocess exercises every module rather than relying on pytest's
    # already populated import cache.
    source = r'''
import argparse, importlib, os, pathlib, socket, sqlite3, subprocess, sys
import numpy, pandas, scipy.optimize, scipy.stats, seaborn, torch, wallow, yaml
import matplotlib.pyplot as plt
import grokking_capacity.analysis as analysis
import grokking_capacity.registry as registry

def forbidden(*args, **kwargs):
    raise AssertionError('analysis import attempted workflow execution')

argparse.ArgumentParser.parse_args = forbidden
sqlite3.connect = forbidden
subprocess.Popen = forbidden
socket.socket = forbidden
numpy.load = forbidden
torch.load = forbidden
torch.nn.Module.__init__ = forbidden
registry.get_store = forbidden
plt.figure = forbidden
plt.savefig = forbidden
pathlib.Path.mkdir = forbidden

# Imports may read source and the packaged workflow catalog, never result data.
root = pathlib.Path(analysis.__file__).parent
catalog = root.parent / 'utils' / 'workflow_catalog.json'
def audit(event, args):
    if event == 'open':
        path, mode, flags = args
        if isinstance(path, (str, bytes)):
            path = str(path)
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                raise AssertionError('import opened file for writing: ' + path)
            if path != str(catalog) and not path.endswith(('.py', '.pyc')):
                raise AssertionError('import opened non-source file: ' + path)
    if event in {'os.mkdir', 'os.remove', 'os.rename', 'os.system', 'shutil.copyfile'}:
        raise AssertionError('import mutated the filesystem: ' + event)
sys.addaudithook(audit)
for path in sorted(root.rglob('*.py')):
    parts = path.relative_to(root).with_suffix('').parts
    if parts[-1] == '__init__':
        parts = parts[:-1]
    importlib.import_module('.'.join(('grokking_capacity', 'analysis', *parts)))
'''
    completed = subprocess.run([sys.executable, '-c', source], cwd=tmp_path,
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout == '', 'analysis import printed workflow output'


def test_capacity_fit_shares_the_same_envelope_and_coefficients():
    rows = [dict(param_count=10,total_bits_memorized=12),
            SimpleNamespace(param_count=10,total_bits_memorized=20),
            dict(param_count=20,total_bits_memorized=35),
            dict(param_count=40,total_bits_memorized=77),
            dict(param_count=None,total_bits_memorized=999)]
    expected = np.polyfit([10,20,40],[20,35,77],1)
    line = fit_capacity_line(rows)
    assert line['param_counts'] == [10,20,40]
    assert line['slope'] == fit_capacity_slope(iter(rows)) == expected[0]
    assert line['intercept'] == expected[1]
    assert fit_capacity_slope(rows[:2]) is None


def test_parameter_count_preserves_rng_and_constructor_validation():
    from grokking_capacity.models import TransformerTorch
    matching.get_param_count.cache_clear()
    before = torch.get_rng_state().clone()
    actual = matching.get_param_count(12,2,2,13,.2)
    assert torch.equal(before, torch.get_rng_state())
    expected = sum(p.numel() for p in TransformerTorch(
        dim=12,depth=2,heads=2,n_tokens=15,seq_len=4,dropout=.2).parameters()
        if p.requires_grad)
    assert actual == expected
    with pytest.raises((RuntimeError, ValueError)):
        matching.get_param_count(-1,2,2,13)


def paired_runs(**kwargs):
    params = dict(p=13, dim=8, seed=42, dropout=0., weight_decay=0.)
    params.update(kwargs)
    for kind in ['groks','speed']:
        extra={'n_samples':matching.compute_n_equiv(13,'/',.5)[0]} if kind=='speed' else {}
        ident=build_identifying(experiment_type=kind,**params,**extra)
        register(get_store(),identifying=ident,
                 annotating={'status':'completed','param_count':1000},
                 on_duplicate='return_existing')


def test_matching_retains_zero_settings_for_capacity_calibration(isolated_repo, monkeypatch):
    from grokking_capacity.analysis import capacity_constant
    paired_runs()
    calls=[]
    def measure(**kwargs):
        calls.append(kwargs)
        return 3.
    monkeypatch.setattr(capacity_constant,'measure_capacity_constant',measure)
    matches=matching.build_match_table(capacity_constant=2.16)
    assert len(matches)==1
    assert calls[0]['dropout']==calls[0]['weight_decay']==0.
    assert matches[0].capacity_constant==3.


@pytest.mark.parametrize('name,value',[
    ('primes',[17]),('weight_decays',[1.]),('depths',[4]),
    ('heads',[2]),('dropouts',[.2]),('init_scales',[.5]),('primes',[]),
])
def test_matching_applies_requested_filters_to_both_cohorts(isolated_repo,name,value):
    paired_runs()
    assert len(matching.build_match_table(capacity_constant=2.16,measure_capacity=False))==1
    assert matching.build_match_table(capacity_constant=2.16,measure_capacity=False,**{name:value})==[]


def test_matching_does_not_join_different_primes(isolated_repo):
    for kind,prime in [('groks',13),('speed',17)]:
        ident=build_identifying(experiment_type=kind,p=prime,dim=8,seed=42,
                               n_samples=matching.compute_n_equiv(13,'/',.5)[0])
        register(get_store(),identifying=ident,
                 annotating={'status':'completed','param_count':1000},
                 on_duplicate='return_existing')
    assert matching.build_match_table(capacity_constant=2.16,measure_capacity=False)==[]
