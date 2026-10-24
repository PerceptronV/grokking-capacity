import json

import pytest

from scripts.report_discrete import comparison, validate


def test_report_keeps_censored_crossings_out_of_ratios_and_labels_observed_medians():
    rows=[]
    for seed, values in [(43,(100,200,50)),(44,(300,None,100))]:
        for arm, value in zip('ABC',values):
            row=dict(seed=seed,arm=arm)
            for threshold in (95,99):
                for unit in ('step','examples','epochs','wall_seconds'):
                    row[f'generalise_{threshold}_{unit}']=value
            rows.append(row)
    aggregate,pairs=comparison(rows)
    b=next(row for row in aggregate if row['arm']=='B' and row['threshold']==99)
    assert (b['reached'],b['censored'],b['total'])==(1,1,2)
    assert b['median_observed_step']==200
    second=next(row for row in pairs if row['seed']==44 and row['threshold']==99)
    assert second['B_step'] is None
    assert second['B_over_A_step'] is None
    assert second['B_over_C_wall_seconds'] is None
    first=next(row for row in pairs if row['seed']==43 and row['threshold']==99)
    assert first['B_over_A_step']==2
    assert first['B_over_C_step']==4


def test_final_report_refuses_an_unfinished_queue(tmp_path):
    (tmp_path/'orchestrator.json').write_text(json.dumps({'phase':'confirming'}))
    (tmp_path/'selection.json').write_text('{}')
    with pytest.raises(ValueError,match='All nine frozen runs must finish'):
        validate(tmp_path,[])
