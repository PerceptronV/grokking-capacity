#!/usr/bin/env python3
"""Adopt live cone workers and use validated MPS scheduling for queued jobs.

This changes process scheduling only. The immutable training worker, its source
snapshot, the manifest's seed list, and every training option remain unchanged.
"""
from pathlib import Path
from grokking_capacity.utils.workflow_paths import frozen_source
import fcntl, json, os, signal, subprocess, sys, time
from grokking_capacity.experiments.cone_workflow.run_cone_ten_seeds import ROOT, OUT, sha, write_json, gpu_info
PIPE = Path('/tmp/cone_ten_seeds_mps_20261106')
SCRIPT = frozen_source(OUT, 'run_cone_ten_seeds.py')


def live(pid, needle):
    try:
        state = Path(f'/proc/{pid}/status').read_text()
        cmd = Path(f'/proc/{pid}/cmdline').read_text()
        return '\nState:\tZ' not in state and needle in cmd
    except FileNotFoundError:
        return False


def find_workers():
    found = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().decode().split('\0')
            if str(SCRIPT) not in args or '--worker' not in args:
                continue
            env = dict(item.split('=', 1) for item in (proc / 'environ').read_bytes().decode().split('\0') if '=' in item)
            assert env['GC_DATA_DIR'] == str(OUT / 'data')
            p, d, s = [int(args[args.index('--' + key) + 1]) for key in ['p', 'dim', 'seed']]
            name = f'p{p}_d{d}_s{s}'
            assert name not in found
            found[name] = {'pid': int(proc.name), 'gpu': int(env['CUDA_VISIBLE_DEVICES']),
                           'mode': 'mps' if env.get('CUDA_MPS_PIPE_DIRECTORY') == str(PIPE) else 'native',
                           'started_at': time.time(), 'estimated_mib': 0, 'process': None, 'log': None}
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found


def run():
    m = json.loads((OUT / 'manifest.json').read_text())
    for rel, h in m['source_hashes'].items():
        assert sha(OUT / 'source' / rel) == h
    benchmark_dirs = ['scheduling_benchmark', 'scheduling_benchmark_wide']
    evidence = {}
    for name in benchmark_dirs:
        r = json.loads((OUT / name / 'result.json').read_text())
        assert r['all_traces_and_checkpoints_identical'], (name, r)
        evidence[name] = r
    assert evidence['scheduling_benchmark']['throughput_ratio'] > 1.2
    dest = OUT / 'scheduler_transition'; dest.mkdir(exist_ok=True)
    old = json.loads((OUT / 'controller_identity.json').read_text())
    if not (dest / 'manifest.json').exists():
        assert live(old['pid'], 'src/grokking_capacity/experiments/cone_workflow/run_cone_ten_seeds.py'), old
        write_json(dest / 'before_status.json', json.loads((OUT / 'status.json').read_text()))
        write_json(dest / 'manifest.json', {
            'created_at': time.time(), 'prior_controller_pid': old['pid'],
            'scheduler_sha256': sha(__file__), 'unchanged_training_manifest_sha256': sha(OUT / 'manifest.json'),
            'benchmark_results': evidence, 'private_mps_pipe_directory': str(PIPE),
            'physical_gpus': [0, 1], 'MPS_active_thread_percentage': 'default100; no SM partition or precision changes',
            'selection_and_training_unchanged': True,
            'transition': 'Terminate only the dispatch controller, then adopt all live workers without signals or restarts. Existing native CUDA jobs finish normally. New jobs run the exact immutable worker and frozen source with MPS enabled through the environment.',
            'documentation': ['https://docs.nvidia.com/deploy/mps/when-to-use-mps.html',
                              'https://docs.nvidia.com/deploy/mps/appendix-environment-variables.html']})
        # Never signal a worker or process group: all existing training is retained.
        os.kill(old['pid'], signal.SIGTERM)
        for _ in range(100):
            if not live(old['pid'], 'src/grokking_capacity/experiments/cone_workflow/run_cone_ten_seeds.py'):
                break
            time.sleep(.1)
        assert not live(old['pid'], 'src/grokking_capacity/experiments/cone_workflow/run_cone_ten_seeds.py')
    lock = (OUT / 'controller.lock').open('a')
    for attempt in range(120):
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if attempt == 119:
                raise
            time.sleep(.5)
    active = find_workers()
    write_json(dest / 'adopted_workers.json', {name: {k: v for k, v in a.items() if k not in ['process', 'log']} for name, a in active.items()})
    PIPE.mkdir(exist_ok=True); (OUT / 'mps_logs').mkdir(exist_ok=True)
    mps_env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0,1', 'CUDA_MPS_PIPE_DIRECTORY': str(PIPE),
               'CUDA_MPS_LOG_DIRECTORY': str(OUT / 'mps_logs')}
    if not (PIPE / 'nvidia-cuda-mps-control.pid').exists():
        subprocess.run(['nvidia-cuda-mps-control', '-d'], env=mps_env, check=True)
    write_json(OUT / 'controller_identity.json', {'pid': os.getpid(), 'started_at': time.time(),
               'scheduler': 'validated_MPS', 'scheduler_sha256': sha(__file__)})
    control = json.loads((OUT / 'runtime_control.json').read_text())
    control.update({'workers_per_gpu': 16, 'additional_mps_workers_while_native_running': 8})
    write_json(OUT / 'runtime_control.json', control)
    (OUT / 'launch_provenance').mkdir(exist_ok=True)
    expected = {f'p{j["p"]}_d{j["dim"]}_s{j["seed"]}': j for j in m['items']}
    completed = {p.stem for p in (OUT / 'jobs').glob('*.json')} - set(active)
    assert completed <= set(expected) and set(active) <= set(expected)
    pending = [(name, j) for name, j in expected.items() if name not in completed and name not in active]
    failed = []; results = []; last_status = 0
    history = (OUT / 'execution_history.jsonl').open('a', buffering=1)
    while pending or active:
        for name, a in list(active.items()):
            code = a['process'].poll() if a['process'] is not None else (None if live(a['pid'], str(SCRIPT)) else 0)
            if code is None:
                continue
            if a['log'] is not None:
                a['log'].close()
            path = OUT / 'jobs' / (name + '.json')
            ok = code == 0 and path.exists() and 'physical_gpu' in json.loads(path.read_text())
            r = {'name': name, 'returncode': code if code else (0 if ok else 99),
                 'physical_gpu': a['gpu'], 'mode': a['mode'], 'seconds_since_adoption_or_launch': time.time() - a['started_at']}
            results.append(r); history.write(json.dumps(r) + '\n')
            if ok:
                completed.add(name)
            else:
                failed.append(r)
            del active[name]
        control = json.loads((OUT / 'runtime_control.json').read_text())
        info = gpu_info()
        for gpu in [0, 1]:
            group = [a for a in active.values() if a['gpu'] == gpu]
            native_count = sum(a['mode'] == 'native' for a in group)
            slots = int(control['workers_per_gpu']) + min(native_count, int(control['additional_mps_workers_while_native_running']))
            if pending and len(group) < slots:
                name, j = pending[0]
                estimate = 1800 + j['param_count'] * 36 / 1024 ** 2
                initializing = sum(a['estimated_mib'] for a in group if time.time() - a['started_at'] < 12)
                if info[gpu]['free_mib'] - initializing - estimate >= control['gpu_memory_reserve_mib']:
                    pending.pop(0)
                    env = {**os.environ, 'GC_CONE_ROOT': str(ROOT), 'CUDA_VISIBLE_DEVICES': str(gpu),
                           'CUDA_MPS_PIPE_DIRECTORY': str(PIPE), 'CUDA_MPS_LOG_DIRECTORY': str(OUT / 'mps_logs'),
                           'GC_DATA_DIR': str(OUT / 'data'), 'GC_WALLOW_TOML': str(OUT / 'source/wallow.toml'),
                           'WALLOW_JOURNAL_MODE': 'DELETE', 'WALLOW_BUSY_TIMEOUT_MS': '120000',
                           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                           'PYTHONUNBUFFERED': '1', 'PYTHONPATH': str(OUT / 'source/src')}
                    write_json(OUT / 'launch_provenance' / (name + '.json'), {'mode': 'MPS', 'gpu': gpu,
                               'gpu_uuid': info[gpu]['uuid'], 'pipe': str(PIPE), 'manifest_sha256': sha(OUT / 'manifest.json'),
                               'scheduler_sha256': sha(__file__), 'launched_at': time.time()})
                    cmd = [sys.executable, str(SCRIPT), '--worker', '--p', str(j['p']), '--dim', str(j['dim']),
                           '--seed', str(j['seed']), '--epochs', str(j['max_epochs'])]
                    log = (OUT / 'logs' / (name + '.log')).open('a')
                    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    active[name] = {'process': proc, 'pid': proc.pid, 'log': log, 'gpu': gpu, 'mode': 'mps',
                                    'started_at': time.time(), 'estimated_mib': estimate}
                    history.write(json.dumps({'event': 'started', 'name': name, 'pid': proc.pid, 'physical_gpu': gpu,
                                              'mode': 'MPS', 'time': time.time()}) + '\n')
        if time.time() - last_status > 15:
            status = {'updated_at': time.time(), 'planned': m['jobs'], 'completed': len(completed), 'pending': len(pending),
                      'failed': failed, 'workers_per_gpu': control['workers_per_gpu'], 'scheduler': 'validated_MPS',
                      'gpus': info, 'active': [{'name': name, 'pid': a['pid'], 'gpu': a['gpu'], 'mode': a['mode'],
                                              'elapsed_seconds': time.time() - a['started_at']} for name, a in active.items()]}
            write_json(OUT / 'status.json', status)
            print(json.dumps({k: status[k] for k in ['updated_at', 'completed', 'pending', 'failed', 'gpus']}), flush=True)
            last_status = time.time()
        time.sleep(2)
    write_json(OUT / 'completion.json', {'planned': m['jobs'], 'completed': len(completed), 'failed': failed,
               'results_this_session': results, 'completed_at': time.time(), 'scheduler': 'validated_MPS'})
    subprocess.run(['nvidia-cuda-mps-control'], input='quit\n', text=True, env=mps_env, check=True)
    assert not failed and completed == set(expected)


if __name__ == '__main__':
    run()
