"""The shared numerical helpers must preserve diagnostic-specific conventions."""
import ast
import subprocess
import sys
import zipfile
from typing import Optional
import numpy as np
import pytest

from grokking_capacity.analysis import numerics
from grokking_capacity.experiments.registered import ARCHIVE


def original_functions(module, names):
    with zipfile.ZipFile(ARCHIVE/'sources.zip') as z:
        text=z.read('src/grokking_capacity/analysis/'+module+'.py').decode()
    tree=ast.parse(text)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    ns=dict(np=np,Optional=Optional,N_PERMUTATIONS=101,MEM_ACC=.99)
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<original diagnostics>','exec'),ns)
    return ns


@pytest.mark.parametrize('module',['rate_vs_capacity','kappa_stability'])
def test_linear_fit_and_permutations_match_original(module):
    old=original_functions(module,['_r2','_linfit','_perm_pvalue'])
    rng=np.random.default_rng(23)
    x=rng.normal(size=19);y=2*x+rng.normal(size=19)
    expected=old['_linfit'](x,y)
    assert numerics.linear_fit(x,y)==expected
    assert numerics.permutation_slope_pvalue(x,y,expected[0],resamples=101)==old['_perm_pvalue'](x,y,expected[0])
    assert np.isnan(numerics.permutation_slope_pvalue(x[:2],y[:2],expected[0],resamples=10,min_samples=3))


def test_accuracy_units_epoch_origin_and_missing_crossings():
    from grokking_capacity.analysis import rate_vs_capacity,mem_law,threshold_invariance,contraction
    for data in [[],[.1,.9,.99],[10,90,99],[.1,.2,.3]]:
        expected=numerics.normalise_accuracy(data)
        for module in [rate_vs_capacity,mem_law,threshold_invariance,contraction]:
            np.testing.assert_array_equal(module._norm_acc(np.array(data)),expected)
    assert rate_vs_capacity._t_mem_epochs([10,90,99])==3
    assert mem_law._t_mem([10,90,99])==3
    assert threshold_invariance._t_mem(np.array([.1,.9,.99]),.99)==3
    assert contraction._first_cross(np.array([.1,.9,.99]),.99)==2
    assert contraction._first_cross(np.array([.1,.2]),.99) is None


def test_timescale_ols_preserves_exact_residual_reduction():
    old=original_functions('forecast_onset',['_ols'])
    rng=np.random.default_rng(9)
    design=rng.normal(size=(45,3));target=rng.normal(size=45)
    actual=numerics.ols(design,target);expected=old['_ols'](design,target)
    np.testing.assert_array_equal(actual[0],expected[0])
    assert actual[1:]==expected[1:]


def test_pure_analysis_import_does_not_load_plotting_or_registry():
    code='''import sys
import grokking_capacity.analysis as analysis
from grokking_capacity.analysis import numerics, characters
assert 'matplotlib' not in sys.modules
assert 'wallow' not in sys.modules
assert 'scipy' not in sys.modules
assert 'ConfigView' in dir(analysis)
from grokking_capacity.analysis import ConfigView
from grokking_capacity.analysis.config_view import ConfigView as direct
assert ConfigView is direct
'''
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)


def test_csv_serialization_keeps_key_order_and_quoting(tmp_path):
    from grokking_capacity.utils.artifacts import write_csv,read_json
    path=tmp_path/'data.csv'
    write_csv(path,[dict(name='two, words',value=3),dict(name='x',extra='line\nbreak')])
    assert path.read_text()=='name,value,extra\n"two, words",3,\nx,,"line\nbreak"\n'
    empty=tmp_path/'empty.csv';write_csv(empty,[]);assert not empty.exists()
    p=tmp_path/'data.json';p.write_text('{"ok": true}')
    assert read_json(str(p))=={'ok':True}
