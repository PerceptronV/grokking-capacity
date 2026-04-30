"""YAML-driven sweep dispatcher.

For each expanded run:
  1. Outer-claim a wallow row (return_existing). Skip if status='completed' and
     no --force. Otherwise launch a worker subprocess with --run-uuid.
  2. The worker overwrites status='running' on entry and 'completed'/'failed'
     on exit (see registry.lifecycle).

Multi-node sharding partitions the expanded job list interleaved
(`i % num_nodes == node_rank`). Per-GPU concurrency is bounded by a separate
ThreadPoolExecutor per device so the per-GPU worker cap is enforced throughout
the run.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
import uuid as _uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event, Lock

import torch
import yaml
from wallow import F, register

from .. import consts
from ..analysis.matching import build_match_table, save_match_table
from ..registry import build_identifying, get_store
from .config import expand_runs


# ---------------------------------------------------------------------------
# Outer claim / dedup
# ---------------------------------------------------------------------------

def _identifying_from_run(run: dict) -> dict:
    """Strip dispatcher-internal keys (those starting with '_') before passing
    to wallow."""
    return {k: v for k, v in run.items() if not k.startswith('_')}


def _claim(run: dict, *, store, force: bool, node_rank: int) -> tuple[str | None, bool]:
    """Outer claim. Returns (run_uuid, should_dispatch).

    None uuid means we couldn't claim (e.g., row exists but force=False and
    status=completed) — caller skips the dispatch.
    """
    identifying = build_identifying(**_identifying_from_run(run))
    fresh_uuid = _uuid.uuid4().hex[:12]
    pre = register(
        store,
        identifying=identifying,
        annotating={
            'status': 'pending',
            'run_uuid': fresh_uuid,
            'node_rank': node_rank,
        },
        on_duplicate='return_existing',
    )
    row = pre.run
    if pre.was_inserted:
        return fresh_uuid, True

    # Row exists.
    if row.status == 'completed' and not force:
        return None, False
    # Reuse the existing row's uuid; the worker will overwrite annotations.
    existing_uuid = row.run_uuid or fresh_uuid
    return existing_uuid, True


# ---------------------------------------------------------------------------
# Worker subprocess command builders
# ---------------------------------------------------------------------------

_EXP_TO_MODULE = {
    'capacity': 'torch_grokking.experiments.capacity',
    'speed':    'torch_grokking.experiments.speed',
    'groks':    'torch_grokking.experiments.groks',
}


def _str_or_int(v):
    return str(v)


def build_worker_cmd(
    run: dict, *, run_uuid: str, force: bool, node_rank: int,
    db_path: str | None,
) -> list[str]:
    exp_type = run['experiment_type']
    module = _EXP_TO_MODULE[exp_type]
    cmd = [
        sys.executable, '-m', module,
        '--p', str(run['p']),
        '--seed', str(run['seed']),
        '--dim', str(run['dim']),
        '--depth', str(run['depth']),
        '--heads', str(run['heads']),
        '--dropout', str(run['dropout']),
        '--init-scale', str(run['init_scale']),
        '--lr', str(run['lr']),
        '--weight-decay', str(run['weight_decay']),
        '--beta1', str(run['beta1']),
        '--beta2', str(run['beta2']),
        '--batch-size', str(run['batch_size']),
        '--epochs', str(run['max_epochs']),
        '--run-uuid', run_uuid,
        '--node-rank', str(node_rank),
    ]
    if db_path:
        cmd += ['--db-path', db_path]

    if exp_type == 'capacity':
        cmd += ['--n-samples', str(run['n_samples'])]
        cmd += ['--dataset-type', str(run.get('_op_raw', 'random'))]
    elif exp_type == 'speed':
        cmd += ['--n-samples', str(run['n_samples'])]
        cmd += ['--operation', str(run['operation'])]
        cmd += ['--train-fraction', str(run['train_fraction'])]
        cmd += ['--split-type', str(run['split_type'])]
    elif exp_type == 'groks':
        cmd += ['--operation', str(run['operation'])]
        cmd += ['--train-fraction', str(run['train_fraction'])]
        cmd += ['--split-type', str(run['split_type'])]
        cmd += ['--ignore-memorisation']  # parity with the legacy dispatcher
    if force:
        cmd += ['--force']
    return cmd


# ---------------------------------------------------------------------------
# Parallel execution (per-GPU worker pools)
# ---------------------------------------------------------------------------

_shutdown = Event()


def _kill_children() -> None:
    try:
        import psutil
        cur = psutil.Process()
        children = cur.children(recursive=True)
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
        print(f'Warning: error killing children: {e}')


def _device_assignment(max_workers: int | None, workers_per_gpu: int | None):
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return max_workers or os.cpu_count(), []
    info = []
    total = 0
    for dev in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(dev)
        cc = major + minor / 10
        w = workers_per_gpu if workers_per_gpu is not None else (
            4 if cc >= 9.0 else 3 if cc >= 8.0 else 2 if cc >= 7.0 else 1
        )
        total += w
        info.append({'id': dev, 'name': torch.cuda.get_device_properties(dev).name,
                      'capability': f'{major}.{minor}', 'workers': w})
    if max_workers and max_workers > total:
        max_workers = total
    n = max_workers if max_workers else total
    print('GPU Worker Allocation:')
    for g in info:
        print(f"  GPU {g['id']}: {g['name']} (Compute {g['capability']}) → {g['workers']} workers")
    assignment = []
    for g in info:
        assignment.extend([g['id']] * g['workers'])
    return n, assignment


def _run_one(cmd: list[str], device_id: int | None, lock: Lock) -> int:
    if _shutdown.is_set():
        return -1
    full = cmd + ['--device', f'cuda:{device_id}'] if device_id is not None else cmd
    label = f'cuda:{device_id}' if device_id is not None else 'default'
    with lock:
        print(f'[{label}] {" ".join(cmd[:5])}...')
    rc = subprocess.run(full).returncode
    if rc in (2, 130) or (rc >> 8) == 130:
        _shutdown.set()
        return -1
    with lock:
        print(f"[{label}] {'✓' if rc == 0 else '✗'} rc={rc}")
    return rc


def _run_batch(cmds: list[list[str]], dry_run: bool, max_workers: int | None,
                workers_per_gpu: int | None) -> None:
    if not cmds:
        return
    if dry_run:
        for cmd in cmds:
            print('DRY RUN:', ' '.join(cmd))
        return
    n, assignment = _device_assignment(max_workers, workers_per_gpu)
    print(f'\nDispatching {len(cmds)} job(s) with {n} worker(s)...\n')
    lock = Lock()
    failed: list[list[str]] = []
    cancelled: list[list[str]] = []
    if assignment:
        unique = list(dict.fromkeys(assignment))
        wpg = assignment.count(unique[0])
        per_dev: dict = {d: [] for d in unique}
        for i, cmd in enumerate(cmds):
            per_dev[unique[i % len(unique)]].append(cmd)
    else:
        unique = [None]
        wpg = n
        per_dev = {None: list(cmds)}
    executors = {d: ThreadPoolExecutor(max_workers=wpg) for d in unique}
    futures: dict = {}
    try:
        for dev, dev_cmds in per_dev.items():
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
        for c in failed[:10]:
            print(f'  {" ".join(c)}')
        sys.exit(1)


# ---------------------------------------------------------------------------
# Match table
# ---------------------------------------------------------------------------

def _build_suite_match_table(suite_name: str, spec: dict, db_path: str | None) -> None:
    """Build matches.json for the suite from the wallow store."""
    print(f"\nBuilding match table for suite '{suite_name}'...")
    primes = sorted({r['p'] for r in expand_runs(spec)
                     if r['experiment_type'] in ('groks', 'speed')})
    if not primes:
        print("  No groks/speed runs in suite — skipping.")
        return
    matches = build_match_table(
        db_path=db_path,
        primes=primes,
        capacity_constant=consts.C,
        capacity_source=f"consts.C={consts.C}",
    )
    out_dir = os.path.join('data', suite_name)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'matches.json')
    save_match_table(matches, out)
    print(f"  Match table: {len(matches)} pairs → {out}")


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------

def run_suite(
    yaml_path: str, *, dry_run: bool, force: bool, no_match_table: bool,
    max_workers: int | None, workers_per_gpu: int | None,
    node_rank: int, num_nodes: int, db_path: str | None,
) -> None:
    with open(yaml_path) as f:
        spec = yaml.safe_load(f)
    suite_name = spec['name']
    print(f"Suite: {suite_name}")
    print(f"Description: {spec.get('description', '')}")
    print(f"Dry run: {dry_run}")
    if num_nodes > 1:
        print(f"Multi-node: node {node_rank} of {num_nodes}")
    print()

    store = get_store(db_path)

    # Group expanded runs by experiment_type to preserve capacity → speed → groks order.
    by_type: dict[str, list[dict]] = {'capacity': [], 'speed': [], 'groks': []}
    for run in expand_runs(spec):
        by_type[run['experiment_type']].append(run)

    for exp_type in ('capacity', 'speed', 'groks'):
        runs = by_type[exp_type]
        if not runs:
            continue
        print(f"--- {exp_type}: {len(runs)} candidate run(s) ---")

        cmds: list[list[str]] = []
        skipped = 0
        for i, run in enumerate(runs):
            if num_nodes > 1 and i % num_nodes != node_rank:
                continue
            if dry_run:
                # Don't pollute the registry on a dry-run; preview the command
                # the worker would receive (uuid is a placeholder).
                run_uuid = '<uuid>'
                cmds.append(build_worker_cmd(run, run_uuid=run_uuid, force=force,
                                              node_rank=node_rank, db_path=db_path))
                continue
            try:
                run_uuid, should_dispatch = _claim(run, store=store, force=force, node_rank=node_rank)
            except Exception as e:
                print(f"  CLAIM FAILED: {e!r} for run {run}")
                continue
            if not should_dispatch:
                skipped += 1
                continue
            cmds.append(build_worker_cmd(run, run_uuid=run_uuid, force=force,
                                          node_rank=node_rank, db_path=db_path))
        if skipped:
            print(f"  Skip (already completed): {skipped} run(s)")
        _run_batch(cmds, dry_run, max_workers, workers_per_gpu)

    if not no_match_table and not dry_run and node_rank == 0:
        _build_suite_match_table(suite_name, spec, db_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='Run a YAML-defined experiment suite.')
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--force', action='store_true',
                    help='Re-run even when a completed wallow row exists for the combo.')
    p.add_argument('--no-match-table', action='store_true')
    p.add_argument('--max-workers', type=int, default=None)
    p.add_argument('--workers-per-gpu', type=int, default=None)
    p.add_argument('--node-rank', type=int, default=0)
    p.add_argument('--num-nodes', type=int, default=1)
    p.add_argument('--db-path', type=str, default=None,
                    help='Override the wallow runs.db path (default: ./runs.db).')
    args = p.parse_args()

    if args.node_rank < 0 or args.node_rank >= args.num_nodes:
        print(f"Error: --node-rank {args.node_rank} out of range for --num-nodes {args.num_nodes}")
        sys.exit(1)
    if not os.path.exists(args.config):
        print(f"Error: config not found: {args.config}")
        sys.exit(1)

    run_suite(
        yaml_path=args.config,
        dry_run=args.dry_run, force=args.force,
        no_match_table=args.no_match_table,
        max_workers=args.max_workers, workers_per_gpu=args.workers_per_gpu,
        node_rank=args.node_rank, num_nodes=args.num_nodes,
        db_path=args.db_path,
    )


if __name__ == '__main__':
    main()
