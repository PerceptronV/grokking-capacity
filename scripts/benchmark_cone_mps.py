#!/usr/bin/env python3
"""Small isolated scheduling benchmark; none of these runs enter the cone."""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, os, subprocess, sys, time
ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'results/cone_ten_seeds_h100_20261106'
OUT = CAMPAIGN / 'scheduling_benchmark'
PIPE = Path('/tmp/cone_mps_benchmark_20261106')
ITEMS = [(24, 42), (24, 43), (316, 42), (316, 43)]


def worker(args):
    sys.path.insert(0, str(CAMPAIGN / 'source/src'))
    import run_cone_h100_boundary as engine
    import grokking_capacity.experiments.groks as g
    engine.OUT = OUT / args.mode; engine.ROOT = CAMPAIGN / 'source'
    original = g.main
    def main():
        sys.argv[sys.argv.index('--epochs') + 1] = '50'
        original()
    g.main = main
    args.p = 149
    engine.worker(args)


def batch(mode):
    dest = OUT / mode
    for name in ['jobs', 'logs']:
        (dest / name).mkdir(parents=True, exist_ok=True)
    os.environ['WALLOW_JOURNAL_MODE'] = 'DELETE'
    os.environ['WALLOW_BUSY_TIMEOUT_MS'] = '120000'
    sys.path.insert(0, str(ROOT / 'src'))
    from grokking_capacity.registry.store import get_store
    get_store(str(dest / 'runs.db'))
    def job(item):
        d, s = item
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0',
               'CUDA_MPS_PIPE_DIRECTORY': str(PIPE if mode == 'mps' else PIPE / 'disabled'),
               'CUDA_MPS_LOG_DIRECTORY': str(OUT / 'mps_logs'),
               'GC_DATA_DIR': str(dest / 'data'), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
               'OPENBLAS_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1', 'PYTHONPATH': str(CAMPAIGN / 'source/src')}
        t = time.time()
        with (dest / 'logs' / f'd{d}_s{s}.log').open('w') as f:
            run = subprocess.run([sys.executable, __file__, '--worker', '--mode', mode,
                                  '--dim', str(d), '--seed', str(s)], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        return {'dim': d, 'seed': s, 'seconds': time.time() - t, 'returncode': run.returncode}
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(job, ITEMS))
    result = {'seconds': time.time() - started, 'results': results}
    (dest / 'timing.json').write_text(json.dumps(result, indent=2) + '\n')
    assert all(r['returncode'] == 0 for r in results), result
    print(mode, json.dumps(result), flush=True)
    return result


def main():
    OUT.mkdir(exist_ok=True); PIPE.mkdir(exist_ok=True)
    (OUT / 'mps_logs').mkdir(exist_ok=True)
    assert not (OUT / 'plan.json').exists(), 'Benchmark already prepared; inspect existing outcomes before reuse.'
    (OUT / 'plan.json').write_text(json.dumps({'items': ITEMS, 'prime': 149, 'epoch_cap': 50,
        'gpu': 0, 'purpose': 'Scheduling throughput and exact numerical equivalence. These runs are excluded from all cone observations.',
        'comparison': 'Same data, initialization, seed and frozen model/trainer. Compare traces and initial/final checkpoints. Primary training continues unchanged.',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + '\n')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'CUDA_MPS_PIPE_DIRECTORY': str(PIPE),
           'CUDA_MPS_LOG_DIRECTORY': str(OUT / 'mps_logs')}
    native = batch('native')
    subprocess.run(['nvidia-cuda-mps-control', '-d'], env=env, check=True)
    try:
        optimized = batch('mps')
    finally:
        subprocess.run(['nvidia-cuda-mps-control'], input='quit\n', text=True, env=env, check=True)
    import numpy as np, sqlite3, torch
    def read(mode):
        con = sqlite3.connect(OUT / mode / 'runs.db'); con.row_factory = sqlite3.Row
        rows = {(r['dim'], r['seed']): dict(r) for r in con.execute('select * from runs')}; con.close()
        return rows
    a = read('native'); b = read('mps'); checks = []
    for item in ITEMS:
        pa = Path(a[item]['npz_path']); pb = Path(b[item]['npz_path'])
        with np.load(pa) as x, np.load(pb) as y:
            traces = all(np.array_equal(x[k], y[k]) for k in ['train_acc', 'val_acc'])
        parameters = {}
        for file in ['model_init.pt', 'model.pt']:
            x = torch.load(pa.parent / file, map_location='cpu', weights_only=True)['model_state_dict']
            y = torch.load(pb.parent / file, map_location='cpu', weights_only=True)['model_state_dict']
            parameters[file] = all(torch.equal(x[k], y[k]) for k in x)
        checks.append({'dim': item[0], 'seed': item[1], 'identical_accuracy_traces': traces,
                       'identical_parameters': parameters})
    result = {'native_seconds': native['seconds'], 'mps_seconds': optimized['seconds'],
              'throughput_ratio': native['seconds'] / optimized['seconds'], 'equivalence': checks,
              'all_traces_and_checkpoints_identical': all(x['identical_accuracy_traces'] and all(x['identical_parameters'].values()) for x in checks),
              'primary_runs_not_modified': True, 'MPS_daemon_stopped': True}
    (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--worker', action='store_true')
    parser.add_argument('--mode', choices=['native', 'mps']); parser.add_argument('--dim', type=int); parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        main()
