"""
Main entry point for running YAML-defined experiment suites in parallel.

Parses a YAML config file, expands parameter grids, resolves n_samples: auto
and match_by: param_count, dispatches trainer scripts in parallel across all
available GPUs, and builds match tables.

Usage:
    python main.py --config configs/weight_decay_sweep.yaml
    python main.py --config configs/central.yaml --dry-run
    python main.py --config configs/tasks.yaml --max-workers 8
    python main.py --config configs/depth_scaling.yaml --force
"""

import argparse
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event, Lock

import torch
import yaml

from matching import compute_n_equiv, find_dims_for_param_targets, build_match_table, save_match_table
from results import ResultsIndex
import consts


# ---------------------------------------------------------------------------
# Config expansion helpers
# ---------------------------------------------------------------------------

# Keys that are NOT expanded in the Cartesian product — they are iterated
# directly (primes, seeds) or resolved separately (dims/dim_ranges/targets).
_NON_GRID_KEYS = {'seeds', 'primes', 'dims', 'dim_ranges', 'param_count_targets',
                  'match_by', 'n_samples', 'type'}


def _merge(defaults: dict, spec: dict) -> dict:
    """Merge defaults into spec, with spec taking precedence."""
    merged = {**defaults}
    for k, v in spec.items():
        merged[k] = v
    return merged


def expand_experiment(exp_spec: dict, defaults: dict) -> list:
    """Merge defaults into exp_spec, then Cartesian-product all list-valued keys.

    Only scalar-typed keys (excluding _NON_GRID_KEYS) are gridded.
    Returns a list of single-valued dicts, one per grid point.
    """
    merged = _merge(defaults, exp_spec)
    grid_keys = [k for k, v in merged.items()
                 if isinstance(v, list) and k not in _NON_GRID_KEYS]
    scalar_part = {k: v for k, v in merged.items() if k not in grid_keys}

    if not grid_keys:
        return [merged]

    configs = []
    for combo in itertools.product(*[merged[k] for k in grid_keys]):
        cfg = {**scalar_part, **dict(zip(grid_keys, combo))}
        configs.append(cfg)
    return configs


def resolve_dims(cfg: dict) -> list:
    """Expand dim_ranges or match_by to a list of dims.

    For match_by: param_count, returns a list of (dim, actual_param_count) tuples.
    For dim_ranges or dims, returns a list of ints.
    """
    if cfg.get('match_by') == 'param_count':
        targets = cfg.get('param_count_targets', [])
        p = cfg['primes'] if isinstance(cfg.get('primes'), int) else cfg.get('primes', [113])[0]
        return find_dims_for_param_targets(
            targets,
            depth=cfg.get('depth', 2),
            heads=cfg.get('heads', 1),
            p=p,
        )
    elif 'dim_ranges' in cfg:
        dims = []
        for r in cfg['dim_ranges']:
            dims.extend(range(r['start'], r['end'] + 1, r.get('step', 1)))
        return sorted(set(dims))
    elif 'dims' in cfg:
        return list(cfg['dims'])
    return []


def resolve_n_samples(cfg: dict, p: int) -> int | None:
    """Resolve n_samples: auto to the matched random dataset size."""
    if cfg.get('n_samples') == 'auto':
        op = cfg.get('operation', '/')
        tf = cfg.get('train_fraction', 0.5)
        n_equiv, _ = compute_n_equiv(p, op, tf)
        return n_equiv
    return cfg.get('n_samples')


def _iter_list(cfg: dict, key: str) -> list:
    val = cfg.get(key, [])
    if isinstance(val, list):
        return val
    return [val]


# ---------------------------------------------------------------------------
# Existence checks
# ---------------------------------------------------------------------------

def _check_speed_exists(index: ResultsIndex, p: int, seed: int, dim: int, n_samples: int,
                        operation: str = '/', weight_decay: float = None,
                        depth: int = 2, heads: int = 1, init_scale: float = 1.0) -> bool:
    is_filter = (lambda x: x is None or x == init_scale) if init_scale == 1.0 else init_scale
    return index.exists(experiment_type="speed", p=p, seed=seed, dim=dim,
                        n_samples=n_samples, operation=operation,
                        weight_decay=weight_decay, depth=depth, heads=heads,
                        init_scale=is_filter)


def _check_groks_exists(index: ResultsIndex, p: int, seed: int, dim: int, depth: int,
                        heads: int, split_type: str = 'random', operation: str = '/',
                        weight_decay: float = None, train_fraction: float = 0.5,
                        init_scale: float = 1.0) -> bool:
    is_filter = (lambda x: x is None or x == init_scale) if init_scale == 1.0 else init_scale
    return index.exists(experiment_type="groks", p=p, seed=seed, dim=dim,
                        depth=depth, heads=heads, split_type=split_type,
                        operation=operation, weight_decay=weight_decay,
                        train_fraction=train_fraction, init_scale=is_filter)


def _check_capacity_exists(index: ResultsIndex, p: int, seed: int, dim: int, n_samples: int,
                            operation: str = 'random', weight_decay: float = None,
                            depth: int = 2, heads: int = 1, init_scale: float = 1.0) -> bool:
    is_filter = (lambda x: x is None or x == init_scale) if init_scale == 1.0 else init_scale
    return index.exists(experiment_type="capacity", p=p, seed=seed, dim=dim,
                        n_samples=n_samples, operation=operation,
                        weight_decay=weight_decay, depth=depth, heads=heads,
                        init_scale=is_filter)


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def build_speed_cmd(cfg: dict, p: int, seed: int, dim: int, n_samples: int,
                    force: bool = False) -> list:
    cmd = [
        sys.executable, 'speed.py',
        '--p', str(p),
        '--seed', str(seed),
        '--dim', str(dim),
        '--n-samples', str(n_samples),
        '--operation', str(cfg.get('operation', '/')),
        '--train-fraction', str(cfg.get('train_fraction', 0.5)),
        '--split-type', str(cfg.get('split_type', 'random')),
        '--weight-decay', str(cfg.get('weight_decay', 0.01)),
        '--lr', str(cfg.get('lr', 1e-3)),
        '--depth', str(cfg.get('depth', 2)),
        '--heads', str(cfg.get('heads', 1)),
        '--dropout', str(cfg.get('dropout', 0.2)),
        '--init-scale', str(cfg.get('init_scale', 1.0)),
        '--epochs', str(cfg.get('max_epochs', 5000)),
        '--batch-size', str(cfg.get('batch_size', 512)),
        '--beta1', str(cfg.get('beta1', 0.9)),
        '--beta2', str(cfg.get('beta2', 0.98)),
    ]
    if force:
        cmd.append('--force')
    return cmd


def build_groks_cmd(cfg: dict, p: int, seed: int, dim: int,
                    force: bool = False) -> list:
    cmd = [
        sys.executable, 'groks.py',
        '--p', str(p),
        '--seed', str(seed),
        '--dim', str(dim),
        '--operation', str(cfg.get('operation', '/')),
        '--train-fraction', str(cfg.get('train_fraction', 0.5)),
        '--split-type', str(cfg.get('split_type', 'random')),
        '--weight-decay', str(cfg.get('weight_decay', 1.0)),
        '--lr', str(cfg.get('lr', 1e-3)),
        '--depth', str(cfg.get('depth', 2)),
        '--heads', str(cfg.get('heads', 1)),
        '--dropout', str(cfg.get('dropout', 0.2)),
        '--init-scale', str(cfg.get('init_scale', 1.0)),
        '--epochs', str(cfg.get('max_epochs', 200)),
        '--batch-size', str(cfg.get('batch_size', 512)),
        '--beta1', str(cfg.get('beta1', 0.9)),
        '--beta2', str(cfg.get('beta2', 0.98)),
        '--ignore-memorisation',
    ]
    if force:
        cmd.append('--force')
    return cmd


def build_capacity_cmd(cfg: dict, p: int, seed: int, dim: int, n_samples: int,
                       force: bool = False) -> list:
    cmd = [
        sys.executable, 'capacity.py',
        '--p', str(p),
        '--seed', str(seed),
        '--dim', str(dim),
        '--n-samples', str(n_samples),
        '--dataset-type', str(cfg.get('operation', 'random')),
        '--weight-decay', str(cfg.get('weight_decay', 0.01)),
        '--lr', str(cfg.get('lr', 1e-3)),
        '--depth', str(cfg.get('depth', 2)),
        '--heads', str(cfg.get('heads', 1)),
        '--dropout', str(cfg.get('dropout', 0.0)),
        '--init-scale', str(cfg.get('init_scale', 1.0)),
        '--epochs', str(cfg.get('max_epochs', 5000)),
        '--batch-size', str(cfg.get('batch_size', 512)),
        '--beta1', str(cfg.get('beta1', 0.9)),
        '--beta2', str(cfg.get('beta2', 0.98)),
    ]
    if force:
        cmd.append('--force')
    return cmd


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------

_shutdown = Event()


def _kill_children() -> None:
    """Terminate all child processes spawned by this process."""
    try:
        import psutil
        current = psutil.Process()
        children = current.children(recursive=True)
        if not children:
            return
        for c in children:
            try:
                c.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(children, timeout=3)
        for c in alive:
            try:
                c.kill()
            except psutil.NoSuchProcess:
                pass
    except Exception as e:
        print(f'Warning: error killing child processes: {e}')


def _get_device_assignment(max_workers: int | None,
                           workers_per_gpu: int | None = None) -> tuple[int, list]:
    """Return (n_workers, device_id_list) based on available hardware.

    device_id_list is empty on CPU-only machines; workers are assigned to
    devices in round-robin order based on the list.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return max_workers or os.cpu_count(), []

    gpu_info = []
    total = 0
    for dev in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(dev)
        cc = major + minor / 10
        workers = workers_per_gpu if workers_per_gpu is not None else (
            4 if cc >= 9.0 else 3 if cc >= 8.0 else 2 if cc >= 7.0 else 1
        )
        total += workers
        name = torch.cuda.get_device_properties(dev).name
        gpu_info.append({'id': dev, 'name': name,
                         'capability': f'{major}.{minor}', 'workers': workers})

    # Cap --max-workers at the GPU-capacity total so per-GPU limits are respected.
    if max_workers is not None and max_workers > total:
        print(f"  Note: --max-workers {max_workers} exceeds GPU capacity ({total}); capping at {total}")
        max_workers = total
    n_workers = max_workers if max_workers is not None else total
    print('GPU Worker Allocation:')
    for g in gpu_info:
        print(f"  GPU {g['id']}: {g['name']} (Compute {g['capability']}) → {g['workers']} workers")

    assignment = []
    for g in gpu_info:
        assignment.extend([g['id']] * g['workers'])
    return n_workers, assignment


def _run_one(cmd: list, device_id: int | None, lock: Lock) -> int:
    """Run one command, injecting --device if a GPU id is provided."""
    if _shutdown.is_set():
        return -1
    full_cmd = cmd + ['--device', f'cuda:{device_id}'] if device_id is not None else cmd
    label = f'cuda:{device_id}' if device_id is not None else 'default'
    with lock:
        print(f'[{label}] {" ".join(cmd)}')
    rc = subprocess.run(full_cmd).returncode
    if rc in (2, 130) or (rc >> 8) == 130:
        _shutdown.set()
        return -1
    with lock:
        print(f'[{label}] {"✓" if rc == 0 else "✗"} ({" ".join(cmd[:3])}...)')
    return rc


def _run_batch(cmds: list, dry_run: bool, max_workers: int | None,
               workers_per_gpu: int | None = None) -> None:
    """Dispatch a list of commands in parallel. Blocks until all complete.

    Commands within a batch are independent and run concurrently. Batches
    are run sequentially by run_suite to respect capacity → speed → groks order.
    """
    if not cmds:
        return
    if dry_run:
        for cmd in cmds:
            print('DRY RUN:', ' '.join(cmd))
        return

    n_workers, assignment = _get_device_assignment(max_workers, workers_per_gpu)
    print(f'\nDispatching {len(cmds)} job(s) with {n_workers} worker(s)...\n')

    lock = Lock()
    failed = []
    cancelled = []

    # Build one executor per device so that the per-GPU worker cap is enforced
    # throughout the run, not just at the initial submission.  The single-pool
    # approach assigned devices to jobs by index, so a fast GPU could free its
    # threads and have them pick up jobs for other GPUs, breaking the cap.
    if assignment:
        unique_devs = list(dict.fromkeys(assignment))           # ordered, dedup
        wpg = assignment.count(unique_devs[0])                  # workers per GPU
        gpu_cmds: dict[int, list] = {d: [] for d in unique_devs}
        for i, cmd in enumerate(cmds):
            gpu_cmds[unique_devs[i % len(unique_devs)]].append(cmd)
    else:
        unique_devs = [None]
        wpg = n_workers
        gpu_cmds = {None: list(cmds)}

    try:
        executors = {d: ThreadPoolExecutor(max_workers=wpg) for d in unique_devs}
        futures: dict = {}
        for dev, dev_cmds in gpu_cmds.items():
            for cmd in dev_cmds:
                if _shutdown.is_set():
                    break
                futures[executors[dev].submit(_run_one, cmd, dev, lock)] = cmd

        for fut in as_completed(futures):
            if _shutdown.is_set():
                for f in futures:
                    if not f.done():
                        f.cancel()
                        cancelled.append(futures[f])
                break
            cmd = futures[fut]
            try:
                rc = fut.result()
                if rc == -1:
                    cancelled.append(cmd)
                elif rc != 0:
                    failed.append(cmd)
            except Exception as e:
                with lock:
                    print(f'Exception running {" ".join(cmd[:3])}: {e}')
                failed.append(cmd)
    except KeyboardInterrupt:
        _shutdown.set()
        _kill_children()
    finally:
        for ex in executors.values():
            ex.shutdown(wait=False)

    _kill_children()

    if cancelled:
        print(f'\nCancelled: {len(cancelled)} job(s)')
        sys.exit(130)
    if failed:
        print(f'\nFailed: {len(failed)} job(s):')
        for cmd in failed[:10]:
            print(f'  {" ".join(cmd)}')
        sys.exit(1)


# ---------------------------------------------------------------------------
# Match table builder
# ---------------------------------------------------------------------------

def _build_suite_match_table(suite_name: str, spec: dict) -> None:
    """Query ResultsIndex for this suite's results and build a match table."""
    print(f"\nBuilding match table for suite '{suite_name}'...")
    index = ResultsIndex('data')
    if len(index) == 0:
        print("  No .meta.json files found — skipping match table.")
        return

    defaults = spec.get('defaults', {})
    experiments = spec.get('experiments', {})

    # Collect groks and speed experiments defined in this suite
    groks_specs = {k: v for k, v in experiments.items() if v.get('type') == 'groks'}
    speed_specs = {k: v for k, v in experiments.items() if v.get('type') == 'speed'}

    if not groks_specs or not speed_specs:
        print("  Suite has no groks+speed pair — skipping match table.")
        return

    # Build filter sets from the suite spec to narrow the index query
    def _collect_values(specs, key, default):
        vals = set()
        for sp in specs.values():
            merged = _merge(defaults, sp)
            v = merged.get(key, default)
            if isinstance(v, list):
                vals.update(v)
            else:
                vals.add(v)
        return vals

    primes = _collect_values({**groks_specs, **speed_specs}, 'primes', [113])
    primes_flat = set()
    for pv in primes:
        if isinstance(pv, list):
            primes_flat.update(pv)
        else:
            primes_flat.add(pv)

    all_groks = []
    all_speed = []
    for p in primes_flat:
        all_groks.extend(index.query(experiment_type='groks', p=p))
        all_speed.extend(index.query(experiment_type='speed', p=p))

    if not all_groks or not all_speed:
        print(f"  No groks ({len(all_groks)}) or speed ({len(all_speed)}) entries found.")
        return

    matches = build_match_table(all_groks, all_speed, capacity_constant=consts.C,
                                capacity_index=index,
                                capacity_source=f"consts.C={consts.C}")

    os.makedirs(os.path.join('data', suite_name), exist_ok=True)
    out_path = os.path.join('data', suite_name, 'matches.json')
    save_match_table(matches, out_path)
    print(f"  Match table: {len(matches)} pairs → {out_path}")


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------

def run_suite(yaml_path: str, dry_run: bool = False, force: bool = False,
              no_match_table: bool = False, max_workers: int | None = None,
              workers_per_gpu: int | None = None,
              node_rank: int = 0, num_nodes: int = 1) -> None:
    with open(yaml_path) as f:
        spec = yaml.safe_load(f)

    suite_name = spec['name']
    defaults = spec.get('defaults', {})
    experiments = spec.get('experiments', {})

    print(f"Suite: {suite_name}")
    print(f"Description: {spec.get('description', '')}")
    print(f"Dry run: {dry_run}\n")
    if num_nodes > 1:
        print(f"Multi-node: node {node_rank} of {num_nodes}\n")

    index = ResultsIndex("data")

    # Enforce dependency order: capacity → speed → groks (then other types)
    ordered_types = ['capacity', 'speed', 'groks']
    exp_order = []
    for t in ordered_types:
        for name, esp in experiments.items():
            if esp.get('type', name) == t and name not in exp_order:
                exp_order.append(name)
    for name in experiments:
        if name not in exp_order:
            exp_order.append(name)

    for exp_name in exp_order:
        if exp_name not in experiments:
            continue
        exp_spec = experiments[exp_name]
        exp_type = exp_spec.get('type', exp_name)
        print(f"--- Experiment: {exp_name} (type={exp_type}) ---")

        # Collect all pending commands for this experiment type, then dispatch
        # them as a single parallel batch. Types are sequenced (capacity before
        # speed before groks) but jobs within a type run concurrently.
        pending_cmds = []
        configs = expand_experiment(exp_spec, defaults)

        for cfg in configs:
            primes_val = cfg.get('primes', defaults.get('primes', [113]))
            if not isinstance(primes_val, list):
                primes_val = [primes_val]

            for p in primes_val:
                n_samples = resolve_n_samples(cfg, p)
                dims_raw = resolve_dims(cfg)
                seeds = _iter_list(cfg, 'seeds')

                # Unpack (dim, param_count) tuples from match_by: param_count
                if dims_raw and isinstance(dims_raw[0], tuple):
                    dims = [d for d, _ in dims_raw]
                else:
                    dims = dims_raw

                for seed in seeds:
                    if exp_type == 'speed':
                        wd = cfg.get('weight_decay')
                        operation = cfg.get('operation', '/')
                        depth = cfg.get('depth', 2)
                        heads = cfg.get('heads', 1)
                        init_scale = cfg.get('init_scale', 1.0)
                        rate_k = cfg.get('rate_k', 0)
                        for dim in dims:
                            if not force and n_samples is not None and \
                               _check_speed_exists(index, p, seed, dim, n_samples,
                                                   operation=operation, weight_decay=wd,
                                                   depth=depth, heads=heads,
                                                   init_scale=init_scale):
                                print(f"  Skip (exists): speed p={p} seed={seed} "
                                      f"dim={dim} n={n_samples} wd={wd} depth={depth}")
                            else:
                                pending_cmds.append(
                                    build_speed_cmd(cfg, p, seed, dim, n_samples, force=force))
                            if rate_k > 0:
                                n_rate = n_samples + rate_k
                                if not force and _check_speed_exists(
                                        index, p, seed, dim, n_rate,
                                        operation=operation, weight_decay=wd,
                                        depth=depth, heads=heads, init_scale=init_scale):
                                    print(f"  Skip (exists): speed p={p} seed={seed} "
                                          f"dim={dim} n={n_rate} (rate) wd={wd}")
                                else:
                                    pending_cmds.append(
                                        build_speed_cmd(cfg, p, seed, dim, n_rate, force=force))

                    elif exp_type == 'groks':
                        depth = cfg.get('depth', 2)
                        heads = cfg.get('heads', 1)
                        split_type = cfg.get('split_type', 'random')
                        wd = cfg.get('weight_decay')
                        operation = cfg.get('operation', '/')
                        tf = cfg.get('train_fraction', 0.5)
                        init_scale = cfg.get('init_scale', 1.0)
                        for dim in dims:
                            if not force and _check_groks_exists(
                                    index, p, seed, dim, depth, heads,
                                    split_type, operation=operation,
                                    weight_decay=wd, train_fraction=tf,
                                    init_scale=init_scale):
                                print(f"  Skip (exists): groks p={p} seed={seed} "
                                      f"dim={dim} wd={wd}")
                                continue
                            pending_cmds.append(
                                build_groks_cmd(cfg, p, seed, dim, force=force))

                    elif exp_type == 'capacity':
                        n_samples_list = _iter_list(cfg, 'n_samples')
                        wd = cfg.get('weight_decay')
                        operation = cfg.get('operation', 'random')
                        depth = cfg.get('depth', 2)
                        heads = cfg.get('heads', 1)
                        init_scale = cfg.get('init_scale', 1.0)
                        for dim in dims:
                            for n in n_samples_list:
                                if not force and _check_capacity_exists(
                                        index, p, seed, dim, n,
                                        operation=operation, weight_decay=wd,
                                        depth=depth, heads=heads, init_scale=init_scale):
                                    print(f"  Skip (exists): capacity p={p} seed={seed} "
                                          f"dim={dim} n={n}")
                                    continue
                                pending_cmds.append(
                                    build_capacity_cmd(cfg, p, seed, dim, n, force=force))

        if num_nodes > 1 and pending_cmds:
            pending_cmds = [
                cmd for i, cmd in enumerate(pending_cmds)
                if i % num_nodes == node_rank
            ]
        _run_batch(pending_cmds, dry_run, max_workers, workers_per_gpu)

    if not no_match_table and not dry_run and node_rank == 0:
        _build_suite_match_table(suite_name, spec)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run a YAML-defined experiment suite in parallel'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file (e.g. configs/weight_decay_sweep.yaml)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing them')
    parser.add_argument('--force', action='store_true',
                        help='Re-run experiments even if results already exist')
    parser.add_argument('--no-match-table', action='store_true',
                        help='Skip building the match table after experiments')
    parser.add_argument('--max-workers', type=int, default=None,
                        help='Max parallel workers (default: auto-detect from GPU compute capability)')
    parser.add_argument('--workers-per-gpu', type=int, default=None,
                        help='Override workers-per-GPU (default: auto from compute capability; H100=4)')
    parser.add_argument('--node-rank', type=int, default=0,
                        help='Zero-based index of this node (default: 0)')
    parser.add_argument('--num-nodes', type=int, default=1,
                        help='Total number of nodes (default: 1, single-node mode)')
    args = parser.parse_args()
    if args.node_rank < 0 or args.node_rank >= args.num_nodes:
        print(f'Error: --node-rank {args.node_rank} out of range for --num-nodes {args.num_nodes}')
        sys.exit(1)

    if not os.path.exists(args.config):
        print(f"Error: config file not found: {args.config}")
        sys.exit(1)

    run_suite(
        yaml_path=args.config,
        dry_run=args.dry_run,
        force=args.force,
        no_match_table=args.no_match_table,
        max_workers=args.max_workers,
        workers_per_gpu=args.workers_per_gpu,
        node_rank=args.node_rank,
        num_nodes=args.num_nodes,
    )


if __name__ == '__main__':
    main()
