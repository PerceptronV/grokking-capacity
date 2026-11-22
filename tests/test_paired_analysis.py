"""Generalized comparisons must retain pairing, censoring, and historical estimates."""
import ast
from pathlib import Path
import zipfile
import math
import numpy as np
import pytest

from grokking_capacity.analysis.paired import analyze
from grokking_capacity.experiments.registered import ARCHIVE


def rows(seeds=(1,2,3), sizes=(20,40)):
    return [dict(role=role,seed=seed,n_train=n,t_gen_updates=100+seed+(10 if n==20 else -10)*delta,
                 final_learning=True,joint_stop=True,paper_delay_updates=0)
            for seed in seeds for n in sizes for role,delta in [('C',0),('B',1)]]


def test_prime_seed_count_and_budget_are_explicit():
    result = analyze(rows(),[20,40],[1,2,3],500,label_count=13,resamples=50)
    assert result['sizes'][0]['dataset_bits'] == pytest.approx(20*math.log2(13))
    assert result['sizes'][0]['C_n'] == 3
    assert result['simultaneous_band_radius'] == 0
    assert '3 seeds' in result['confidence'] and '50 draws' in result['confidence']


def test_missing_duplicate_and_unrequested_observations_are_rejected():
    original=rows()
    for bad in [original[:-1],original+[original[0]],original+[dict(original[0],seed=99)]]:
        with pytest.raises(ValueError):
            analyze(bad,[20,40],[1,2,3],500,label_count=13)


def test_original_bootstrap_and_censoring_estimates_are_preserved():
    with zipfile.ZipFile(ARCHIVE/'sources.zip') as archive:
        text=archive.read('scripts/report_p97_dataset_scaling.py').decode()
    functions=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)
               and n.name in ['summarize','crossing_brackets','analyze']]
    ns=dict(np=np,math=math)
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<historical analysis>','exec'),ns)
    data=rows(tuple(range(10)))
    data[0]['t_gen_updates']=None
    expected=ns['analyze'](data,[20,40],list(range(10)),50000)
    actual=analyze(data,[20,40],list(range(10)),50000,label_count=99)
    expected.pop('confidence');actual.pop('confidence')
    assert actual == expected


def test_renderer_uses_report_metadata(tmp_path):
    import json
    from grokking_capacity.analysis.dataset_comparison_plot import render
    result=analyze(rows(),[20,40],[1,2,3],500,label_count=13,resamples=50)
    result.update(prime=11,dropout=.1,limit=500,seeds=[1,2,3],all_audits_passed=True)
    (tmp_path/'summary.json').write_text(json.dumps(result))
    render(tmp_path,tmp_path/'figures')
    assert (tmp_path/'figures/dataset_scaling_final.pdf').stat().st_size>1000
    assert (tmp_path/'figures/rendering_provenance.json').exists()
