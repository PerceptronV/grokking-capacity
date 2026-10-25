#!/usr/bin/env python3
"""Verify saved p97 central-sweep traces; never modify the original registry."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import sqlite3
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FILTERS = dict(p=97, operation='/', train_fraction=.5, split_type='random',
               depth=2, heads=1, dropout=.2, init_scale=1., lr=.001,
               weight_decay=1., beta1=.9, beta2=.98, batch_size=512,
               mup_base=0, architecture_family='transformer_gated',
               host='147-224-250-73', status='completed', experiment_type='groks',
               dataset_type='modular', max_epochs=5000)
# The central figure truncates widths at 256; keep its registered width grid.
DIMS = [*range(20,129,2), *range(130,251,10)]
SEEDS = list(range(42,52))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first(values, threshold):
    found = np.flatnonzero(values >= threshold)
    return int(found[0])+1 if len(found) else None


def csv_write(path, rows):
    with path.open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    connection = sqlite3.connect((ROOT/'runs.db').as_uri()+'?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    query = 'SELECT * FROM runs WHERE '+' AND '.join(f'{key}=?' for key in FILTERS)+' ORDER BY dim,seed'
    records = []
    for row in connection.execute(query, tuple(FILTERS.values())):
        if row['dim'] not in DIMS or row['seed'] not in SEEDS:
            continue
        path = Path(row['npz_path'])
        with np.load(path, allow_pickle=False) as saved:
            assert int(saved['n_train']) == row['n_samples'] == 4656
            assert int(saved['p']) == 97 and int(saved['dim']) == row['dim']
            assert int(saved['param_count']) == row['param_count']
            tr, va, va99 = (first(saved[key], threshold) for key, threshold in
                            [('train_acc',99), ('val_acc',98), ('val_acc',99)])
            assert va99 == row['grokking_epoch'] and row['grokking_threshold'] == .99
            records.append(dict(dim=row['dim'], params=row['param_count'], seed=row['seed'],
                                train99_epoch=tr, val98_epoch=va, val99_epoch=va99,
                                delay_epochs=max(0,va-tr) if tr is not None and va is not None else None,
                                epochs_observed=len(saved['val_acc']),
                                final_train_pct=float(saved['train_acc'][-1]),
                                final_val_pct=float(saved['val_acc'][-1]),
                                uuid=row['uuid'], git_hash=row['git_hash'], trace_path=str(path),
                                trace_sha256=digest(path)))
    connection.close()
    groups = defaultdict(list)
    for row in records:
        groups[row['dim']].append(row)
    assert sorted(groups) == DIMS
    target = 4656*math.log2(99)
    summaries = []
    for dim, rows in sorted(groups.items()):
        assert len(rows) == 10 and sorted(row['seed'] for row in rows) == SEEDS
        delays = [r['delay_epochs'] for r in rows if r['delay_epochs'] is not None]
        summaries.append(dict(dim=dim, params=rows[0]['params'], total=10,
                              published_capacity_2_16P=2.16*rows[0]['params'],
                              published_capacity_over_dataset=2.16*rows[0]['params']/target,
                              reached_train99=sum(r['train99_epoch'] is not None for r in rows),
                              reached_val99=sum(r['val99_epoch'] is not None for r in rows),
                              zero_delays=delays.count(0), positive_delays=sum(d>0 for d in delays),
                              incomplete_delays=10-len(delays),
                              min_observed_delay_epochs=min(delays,default=None),
                              max_observed_delay_epochs=max(delays,default=None)))
    onset = next(row for index,row in enumerate(summaries)
                 if all(r['positive_delays']==10 for r in summaries[index:]))
    width26 = next(r for r in summaries if r['dim']==26)
    provenance = dict(paper='https://arxiv.org/html/2605.09724v2', filters=FILTERS,
                      dimensions=DIMS, seeds=SEEDS, verified_traces=len(records),
                      target_bits=target, published_capacity_constant=2.16,
                      published_parameter_threshold=target/2.16,
                      same_width_as_A=width26, persistent_all_seed_positive_onset_in_checked_grid=onset,
                      delay='max(0, first held-out >=98% epoch minus first training >=99% epoch)',
                      censoring='If either threshold is missing, delay is null; never imputed zero. The paper reports the minimum across seeds, not a universal no-delay claim.',
                      units='Original per-epoch records; 1 epoch = 10 optimizer updates at p97.',
                      comparison_limitations=[
                          'Paper baseline dropout 0.2; new p97 comparison dropout 0.',
                          'Original train_acc is accumulated from training-mode minibatch predictions during each epoch; new train_acc is a fixed-checkpoint eval-mode full-training-set measurement.',
                          'Original evaluation each epoch, 5000-epoch maximum, 10 seeds; new evaluation every 100 updates (=10 epochs), 120000-update maximum (=12000 epochs), 3 seeds.',
                          'Published capacity uses 2.16 bits/parameter; new direct finite-budget p97 estimates use a different calibration protocol. Do not identify these capacity scales or infer a causal dropout effect.'
                      ],
                      local_capacity_provenance_caution='The existing docs/papers/2026 Grokking Capacity/submissions/logs/p01a_capacity_provenance.md records that the historical 2.16 fit could not be recovered in its audit. Values using 2.16 here reproduce the published convention, not an independently validated capacity estimate.',
                      original_source_path='src/torch_grokking/experiments/groks.py',
                      original_source_commit='f6b0754',
                      current_discrete_source_sha256=digest(ROOT/'src/grokking_capacity/experiments/discrete.py'))
    original = subprocess.check_output(['git','show','f6b0754:src/torch_grokking/experiments/groks.py'],cwd=ROOT)
    provenance['original_source_sha256'] = hashlib.sha256(original).hexdigest()
    output = ROOT/'reports/paper_p97'
    output.mkdir(parents=True,exist_ok=True)
    csv_write(output/'runs.csv',records)
    csv_write(output/'sizes.csv',summaries)
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (output/'artifact_hashes.json').write_text(json.dumps({p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!='artifact_hashes.json'},indent=2)+'\n')
    print(json.dumps(dict(verified_traces=len(records),same_width_as_A=width26,onset=onset),indent=2))


if __name__ == '__main__':
    main()
