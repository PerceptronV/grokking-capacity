#!/usr/bin/env python3
"""Complete every published cone cell to seeds42--51, using both H100s.

The manifest and source snapshots precede all outcomes. Existing H100 cells
receive missing seeds; incomplete A100 cells are replaced in full on H100.
"""
from pathlib import Path
from grokking_capacity.utils.workflow_paths import frozen_source
import argparse, collections, fcntl, hashlib, json, os, shutil, socket, sqlite3
import subprocess, sys, time

ROOT = Path(os.environ.get('GC_CONE_ROOT', Path(__file__).resolve().parents[4]))
OUT = ROOT / 'results/cone_ten_seeds_h100_20261106'
BASE = ROOT / 'results/cone_p199_h100_20261105/figure'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ['jobs', 'logs', 'source', 'base_cohort', 'checkpoint_checks']:
        (OUT / name).mkdir(exist_ok=True)
    if (OUT / 'manifest.json').exists():
        m = json.loads((OUT / 'manifest.json').read_text())
        for rel, digest in m['source_hashes'].items():
            assert sha(OUT / 'source' / rel) == digest
        return m
    cells = json.loads((BASE / 'cells.json').read_text())
    rows = json.loads((BASE / 'runs.json').read_text())
    targets = [c for c in cells if c['n_seeds'] < 10]
    assert len(cells) == 1736 and len(rows) == 16015 and len(targets) == 269
    assert all(c['seeds'] == list(range(42, 42 + c['n_seeds'])) for c in cells)
    by_cell = collections.defaultdict(list)
    for r in rows:
        by_cell[r['p'], r['dim']].append(r)
    items = []
    for c in targets:
        assert len(c['budgets']) == 1
        replace = c['gpu_type'] != 'H100'
        assert replace or c['host'] == socket.gethostname()
        reference = by_cell[c['p'], c['dim']]
        estimate = sum(r['wallclock_seconds'] or 0 for r in reference) / len(reference)
        for seed in (range(42, 52) if replace else range(47, 52)):
            items.append({'p': c['p'], 'dim': c['dim'], 'seed': seed,
                          'max_epochs': c['budgets'][0], 'param_count': c['param_count'],
                          'role': 'replace_A100_cell' if replace else 'complete_H100_cell',
                          'historical_mean_seconds': estimate})
    assert len(items) == 1495
    # Start the questioned point immediately; schedule long historical runs early.
    items.sort(key=lambda j: (0 if (j['p'], j['dim']) == (197, 316) else 1,
                             -j['historical_mean_seconds'], j['p'], j['dim'], j['seed']))
    paths = [Path(__file__).resolve(), ROOT / 'src/grokking_capacity/experiments/cone_workflow/run_cone_h100_boundary.py',
             *sorted((ROOT / 'src/grokking_capacity').rglob('*.py')), ROOT / 'wallow.toml']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    for p in paths:
        dst = OUT / 'source' / p.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    for name in ['runs.json', 'cells.json', 'summary.json']:
        shutil.copy2(BASE / name, OUT / 'base_cohort' / name)
    from grokking_capacity.experiments.cone_workflow.run_cone_h100_boundary import PROTOCOL
    protocol = {k: v for k, v in PROTOCOL.items() if k not in ['status', 'host', 'gpu_type', 'max_epochs']}
    m = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
         'host': socket.gethostname(), 'physical_gpus': [0, 1],
         'jobs': len(items), 'target_cells': len(targets), 'final_cells': 1736,
         'final_unique_runs': 17360, 'final_seed_ids': list(range(42, 52)),
         'missing_seed_observations': 1345, 'A100_cells_replaced': 30,
         'A100_observations_archived': 150, 'H100_cells_completed': 239,
         'items': items, 'targets': targets, 'protocol': protocol,
         'source_hashes': hashes,
         'base_cohort_hashes': {n: sha(OUT / 'base_cohort' / n) for n in ['runs.json', 'cells.json', 'summary.json']},
         'selection': 'All cells with fewer than ten seeds in the existing published cone, fixed before any new outcomes. Retain every existing width. Replace all five A100 observations at each incomplete A100 cell with ten fresh H100 seeds. Preserve complete ten-seed cells unchanged. No within-cell A100/H100 pooling.',
         'stopping': 'Match the existing per-cell epoch cap (1500, 2500 or5000). Stop only after online training99 and evaluation validation99 are both reached. No patience stopping. Score first training99 and validation98 with the unchanged censoring rules.',
         'analysis_plan': 'Include all planned observations. Compare old and completed cells and paired replacement seeds; retain archived controls. Assert exactly seeds42--51 at every cell and one hardware cohort percell. Recompute cone only, preserve fitted laws and main onset/crossing tests. Report every changed classification, censoring count and any inconsistency with main-text claims. No common-five figure.',
         'scheduling': 'Both H100s; independent single-GPU processes with CPU threads1. Live concurrency control changes scheduling only, never model, batch size, precision, stopping, or seed selection.'}
    write_json(OUT / 'manifest.json', m)
    write_json(OUT / 'runtime_control.json', {'workers_per_gpu': 6, 'gpu_memory_reserve_mib': 9000})
    # Initialise the NFS-safe shared registry before concurrent writers start.
    os.environ['WALLOW_JOURNAL_MODE'] = 'DELETE'
    os.environ['WALLOW_BUSY_TIMEOUT_MS'] = '120000'
    sys.path.insert(0, str(ROOT / 'src'))
    from grokking_capacity.registry.store import get_store
    get_store(str(OUT / 'runs.db'))
    return m


def worker(args):
    import grokking_capacity.experiments.cone_workflow.run_cone_h100_boundary as engine
    engine.OUT = OUT
    engine.ROOT = OUT / 'source'
    sys.path.insert(0, str(engine.ROOT / 'src'))
    import grokking_capacity.experiments.groks as g
    original_main = g.main
    def matched_main():
        sys.argv[sys.argv.index('--epochs') + 1] = str(args.epochs)
        original_main()
    g.main = matched_main
    engine.worker(args)
    path = OUT / 'jobs' / f'p{args.p}_d{args.dim}_s{args.seed}.json'
    meta = json.loads(path.read_text())
    meta.update({'physical_gpu': int(os.environ['CUDA_VISIBLE_DEVICES']),
                 'max_epochs': args.epochs, 'process_id': os.getpid(),
                 'training_source_manifest_sha256': sha(OUT / 'manifest.json')})
    write_json(path, meta)


def gpu_info():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return {int(a): {'uuid': b.strip(), 'free_mib': int(c), 'utilization': int(d)}
            for a, b, c, d in (line.split(',') for line in result.stdout.strip().splitlines())}


def run():
    m = prepare()
    lock = (OUT / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json(OUT / 'controller_identity.json', {'pid': os.getpid(), 'started_at': time.time()})
    expected = {f'p{j["p"]}_d{j["dim"]}_s{j["seed"]}': j for j in m['items']}
    completed = {p.stem for p in (OUT / 'jobs').glob('*.json')}
    assert completed <= set(expected)
    pending = [(name, j) for name, j in expected.items() if name not in completed]
    active = {}; failed = []; results = []
    initial = gpu_info()
    assert set(initial) == {0, 1}
    history = (OUT / 'execution_history.jsonl').open('a', buffering=1)
    last_status = 0
    while pending or active:
        for name, a in list(active.items()):
            code = a['process'].poll()
            if code is None:
                continue
            a['log'].close()
            ok = code == 0 and (OUT / 'jobs' / (name + '.json')).exists()
            r = {'name': name, 'returncode': code if code else (0 if ok else 99),
                 'physical_gpu': a['gpu'], 'seconds': time.time() - a['started_at']}
            results.append(r); history.write(json.dumps(r) + '\n')
            if ok:
                completed.add(name)
            else:
                failed.append(r)
            del active[name]
        control = json.loads((OUT / 'runtime_control.json').read_text())
        slots = int(control['workers_per_gpu'])
        assert 1 <= slots <= 16
        info = gpu_info()
        for gpu in [0, 1]:
            # Spread new starts across both GPUs and leave space for CUDA setup.
            active_gpu = [a for a in active.values() if a['gpu'] == gpu]
            if pending and len(active_gpu) < slots:
                name, j = pending[0]
                estimated_mib = 1800 + j['param_count'] * 36 / (1024 ** 2)
                initializing = sum(a['estimated_mib'] for a in active_gpu if time.time() - a['started_at'] < 12)
                if info[gpu]['free_mib'] - initializing - estimated_mib >= control['gpu_memory_reserve_mib']:
                    pending.pop(0)
                    env = {**os.environ, 'GC_CONE_ROOT': str(ROOT), 'CUDA_VISIBLE_DEVICES': str(gpu),
                           'GC_DATA_DIR': str(OUT / 'data'),
                           'GC_WALLOW_TOML': str(OUT / 'source/wallow.toml'),
                           'WALLOW_JOURNAL_MODE': 'DELETE', 'WALLOW_BUSY_TIMEOUT_MS': '120000',
                           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                           'PYTHONUNBUFFERED': '1', 'PYTHONPATH': str(OUT / 'source/src')}
                    log = (OUT / 'logs' / (name + '.log')).open('a')
                    cmd = [sys.executable, str(frozen_source(OUT, 'run_cone_ten_seeds.py')),
                           '--worker', '--p', str(j['p']), '--dim', str(j['dim']),
                           '--seed', str(j['seed']), '--epochs', str(j['max_epochs'])]
                    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    active[name] = {'process': proc, 'log': log, 'gpu': gpu,
                                    'started_at': time.time(), 'estimated_mib': estimated_mib}
                    history.write(json.dumps({'event': 'started', 'name': name, 'pid': proc.pid,
                                              'physical_gpu': gpu, 'time': time.time()}) + '\n')
        if time.time() - last_status >= 15:
            status = {'updated_at': time.time(), 'planned': m['jobs'], 'completed': len(completed),
                      'pending': len(pending), 'failed': failed, 'workers_per_gpu': slots,
                      'gpus': info, 'active': [{'name': name, 'pid': a['process'].pid,
                      'gpu': a['gpu'], 'elapsed_seconds': time.time() - a['started_at']} for name, a in active.items()]}
            write_json(OUT / 'status.json', status)
            print(json.dumps({k: status[k] for k in ['updated_at', 'planned', 'completed', 'pending', 'failed', 'gpus']}), flush=True)
            last_status = time.time()
        time.sleep(2)
    completion = {'planned': m['jobs'], 'completed': len(completed), 'failed': failed,
                  'results_this_session': results, 'completed_at': time.time()}
    write_json(OUT / 'completion.json', completion)
    assert not failed and completed == set(expected), completion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--p', type=int)
    parser.add_argument('--dim', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    elif args.prepare:
        m = prepare()
        print(json.dumps({k: m[k] for k in ['jobs', 'target_cells', 'final_cells', 'final_unique_runs', 'physical_gpus']}))
    else:
        run()
