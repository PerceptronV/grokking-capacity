#!/usr/bin/env python3
"""Independently verify all new ten-seed completion runs on CPU."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import argparse, concurrent.futures, functools, json, multiprocessing, sqlite3, time
from grokking_capacity.experiments.cone_workflow.run_cone_ten_seeds import ROOT, OUT, sha, write_json
from grokking_capacity.experiments.cone_workflow.verify_cone_h100_boundary import check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    m = json.loads((OUT / 'manifest.json').read_text())
    for rel, digest in m['source_hashes'].items():
        assert sha(OUT / 'source' / rel) == digest
    expected = {(j['p'], j['dim'], j['seed']): j for j in m['items']}
    dest = OUT / 'checkpoint_checks'
    errors = OUT / 'verification_errors.json'
    with concurrent.futures.ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context('spawn')) as pool:
        while True:
            con = sqlite3.connect((OUT / 'runs.db').as_uri() + '?mode=ro', uri=True, timeout=120)
            con.row_factory = sqlite3.Row
            rows = [dict(r) for r in con.execute("select * from runs where status='completed'")]
            con.close()
            pending = []
            for r in rows:
                spec = expected[r['p'], r['dim'], r['seed']]
                meta_path = OUT / 'jobs' / f'p{r["p"]}_d{r["dim"]}_s{r["seed"]}.json'
                if not meta_path.exists() or (dest / (r['uuid'] + '.json')).exists():
                    continue
                meta = json.loads(meta_path.read_text())
                if 'physical_gpu' not in meta:  # Worker still writing final provenance.
                    continue
                assert r['host'] == meta['host'] == m['host'] and r['gpu_type'] == 'H100'
                assert r['max_epochs'] == meta['max_epochs'] == spec['max_epochs']
                assert meta['physical_gpu'] in [0, 1]
                assert meta['training_source_manifest_sha256'] == sha(OUT / 'manifest.json')
                for k, v in m['protocol'].items():
                    assert r[k] == v, (r['uuid'], k, r[k], v)
                pending.append((r, meta))
            futures = {pool.submit(check, job, cpu_threads=1, linear_backend='numpy'): job[0]['uuid'] for job in pending}
            failures = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    value = future.result()
                    assert value['state'] == 'unresolved' or value['val98'] is not None
                    value['validation98_observed'] = value['val98'] is not None
                    write_json(dest / (value['uuid'] + '.json'), value)
                    print(json.dumps({'checked': value['uuid'], 'p': value['p'], 'dim': value['dim'], 'seed': value['seed']}), flush=True)
                except Exception as exc:
                    failures.append({'uuid': futures[future], 'error': repr(exc)})
            if failures:
                write_json(errors, failures)
                raise RuntimeError(f'Independent validation failed: {failures}')
            values = [json.loads(p.read_text()) for p in sorted(dest.glob('*.json'))]
            write_json(OUT / 'verification_status.json', {'checked': len(values), 'planned': m['jobs'], 'updated_at': time.time()})
            if len(values) == m['jobs']:
                assert {(r['p'], r['dim'], r['seed']) for r in values} == set(expected)
                write_json(OUT / 'checkpoint_verification.json', {'runs_verified': m['jobs'], 'cpu_only': True, 'checks': values})
                break
            if not args.watch:
                print(f'{len(values)}/{m["jobs"]} verified', flush=True)
                break
            if (OUT / 'completion.json').exists():
                done = json.loads((OUT / 'completion.json').read_text())
                assert not done['failed'] and done['completed'] == m['jobs']
            time.sleep(15)


if __name__ == '__main__':
    main()
