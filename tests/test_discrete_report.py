import json

import pytest

from scripts.report_discrete import comparison, control_config_matches, first_passage_delay, validate


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


def test_control_metadata_label_does_not_change_scientific_pairing():
    expected = dict(model='baseline', dim=80, mlp_dim=320, seed=61,
                    n_samples=16384, max_steps=120000, lr=.001)
    assert control_config_matches(expected, expected)
    assert control_config_matches({**expected, 'purpose': 'capacity_control'}, expected)
    assert not control_config_matches({**expected, 'purpose': 'learning_validation'}, expected)
    for key, value in [('dim', 82), ('mlp_dim', 328), ('seed', 62),
                       ('n_samples', 32768), ('max_steps', 60000), ('lr', .0003)]:
        assert not control_config_matches({**expected, 'purpose': 'capacity_control', key: value}, expected)
    assert not control_config_matches({**expected, 'data_seed': 62}, expected)


def test_paper_delay_uses_first_crossings_even_when_they_are_not_sustained():
    rows = [dict(step=0, epochs=0., train_acc=.5, val_acc=.5),
            dict(step=100, epochs=4., train_acc=.99, val_acc=.97),
            dict(step=200, epochs=8., train_acc=.8, val_acc=.98),
            dict(step=300, epochs=12., train_acc=.8, val_acc=.7)]
    result = first_passage_delay(rows)
    assert result == dict(first_train_99_step=100, first_generalise_98_step=200,
                          first_passage_delay_99_98_steps=100,
                          first_passage_delay_99_98_epochs=4.)


def test_paper_delay_clamps_validation_first_to_zero_but_preserves_missing_events():
    rows = [dict(step=100, epochs=4., train_acc=.97, val_acc=.98),
            dict(step=200, epochs=8., train_acc=.99, val_acc=.99)]
    assert first_passage_delay(rows)['first_passage_delay_99_98_steps'] == 0
    for incomplete in [rows[:1], [dict(step=100, epochs=4., train_acc=.99, val_acc=.97)]]:
        result = first_passage_delay(incomplete)
        assert result['first_passage_delay_99_98_steps'] is None
        assert result['first_passage_delay_99_98_epochs'] is None
