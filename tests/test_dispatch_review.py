"""Dispatcher refactors preserve job identities while respecting execution limits."""
import ast
from collections import Counter
from concurrent.futures import Future
import itertools
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Iterator
import zipfile

import pytest
import yaml

from grokking_capacity.dispatch import config, main
from grokking_capacity.experiments.registered import ARCHIVE

ROOT = Path(__file__).resolve().parents[1]


def historical(module, *, functions=None, **namespace):
    with zipfile.ZipFile(ARCHIVE / 'sources.zip') as archive:
        source = archive.read(f'src/grokking_capacity/dispatch/{module}.py').decode()
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.Assign))
             and (functions is None or isinstance(node, ast.FunctionDef) and node.name in functions)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<original dispatcher>', 'exec'), namespace)
    return namespace


def test_repository_suites_keep_original_expansions_and_worker_commands(monkeypatch):
    # Parameter-count search belongs to analysis; use the same deterministic
    # result on both sides to isolate dispatcher expansion and argument order.
    def matched_dims(targets, *, depth, heads, p):
        return [(8 + i * heads, target) for i, target in enumerate(targets)]

    monkeypatch.setattr(config, 'find_dims_for_param_targets', matched_dims)
    old = historical('config', itertools=itertools, Iterator=Iterator,
                     compute_n_equiv=config.compute_n_equiv,
                     find_dims_for_param_targets=matched_dims)
    old_cmd = historical('main', functions={'build_worker_cmd'}, sys=sys,
                         _EXP_TO_MODULE=main._EXP_TO_MODULE)['build_worker_cmd']
    checked = 0
    for path in sorted((ROOT / 'configs').glob('*.yaml')):
        spec = yaml.safe_load(path.read_text())
        if not isinstance(spec, dict) or not isinstance(spec.get('experiments'), dict):
            continue
        actual = list(config.expand_runs(spec))
        assert actual == list(old['expand_runs'](spec)), path
        for run in actual:
            options = dict(force=True, node_rank=2, db_path='/tmp/registry path/runs.db')
            assert main.build_worker_cmd(run, **options) == old_cmd(run, **options), path
        checked += 1
    assert checked >= 20


def test_arbitrary_grid_key_order_and_named_blocks_match_original():
    old = historical('config', itertools=itertools, Iterator=Iterator,
                     compute_n_equiv=config.compute_n_equiv,
                     find_dims_for_param_targets=config.find_dims_for_param_targets)
    spec = dict(defaults=dict(beta1=[0.8, 0.9], lr=[1e-3, 2e-3],
                              max_epochs=[100, 200], seeds=[1, 2], primes=[11, 13]),
                experiments={'g1': dict(type='groks', dims=[8]),
                             's': dict(type='speed', dims=[8], n_samples=[20, 40]),
                             'c': dict(type='capacity', dims=[8], n_samples=40),
                             'g2': dict(type='groks', dims=[16])})
    assert list(config.expand_runs(spec)) == list(old['expand_runs'](spec))


def test_optional_groks_flags_preserve_command_order_and_integer_conversion():
    spec = dict(defaults=dict(primes=13, seeds=42), experiments={
        'g': dict(type='groks', dims=[8], norm_log_every=3.0, post_grok_epochs=4.0),
    })
    run, = config.expand_runs(spec)
    old_cmd = historical('main', functions={'build_worker_cmd'}, sys=sys,
                         _EXP_TO_MODULE=main._EXP_TO_MODULE)['build_worker_cmd']
    options = dict(force=False, node_rank=0, db_path=None)
    assert main.build_worker_cmd(run, **options) == old_cmd(run, **options)


def test_dry_run_does_not_create_registry_or_artifacts(tmp_path):
    suite = tmp_path / 'suite.yaml'
    suite.write_text('''name: dry_run_review
defaults: {primes: [13], seeds: [42]}
experiments:
  capacity: {dims: [8], n_samples: [20]}
  speed: {dims: [8], n_samples: auto}
  groks: {dims: [8]}
''')
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'),
               GC_WALLOW_DB=str(tmp_path / 'runs.db'),
               GC_WALLOW_TOML=str(ROOT / 'wallow.toml'),
               GC_DATA_DIR=str(tmp_path / 'data'))
    result = subprocess.run([sys.executable, '-m', 'grokking_capacity.dispatch.main',
                             '--config', str(suite), '--dry-run'],
                            cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    lines = [line for line in result.stdout.splitlines() if line.startswith('DRY RUN:')]
    assert len(lines) == 3
    assert all(f'.experiments.{name} ' in line
               for name, line in zip(('capacity', 'speed', 'groks'), lines))
    assert sorted(path.name for path in tmp_path.iterdir()) == ['suite.yaml']


def fake_gpus(monkeypatch, capabilities):
    monkeypatch.setattr(main.torch.cuda, 'is_available', lambda: bool(capabilities))
    monkeypatch.setattr(main.torch.cuda, 'device_count', lambda: len(capabilities))
    monkeypatch.setattr(main.torch.cuda, 'get_device_capability', lambda dev: capabilities[dev])
    monkeypatch.setattr(main.torch.cuda, 'get_device_properties',
                        lambda dev: SimpleNamespace(name=f'gpu-{dev}'))


@pytest.mark.parametrize('limit,expected', [(None, {0: 4, 1: 2}), (3, {0: 2, 1: 1}),
                                            (1, {0: 1}), (99, {0: 4, 1: 2})])
def test_gpu_assignment_honors_global_and_each_device_limit(monkeypatch, limit, expected):
    fake_gpus(monkeypatch, [(9, 0), (7, 5)])
    workers, slots = main._device_assignment(limit, None)
    assert workers == sum(expected.values())
    assert Counter(slots) == expected


def test_worker_override_and_cpu_fallback(monkeypatch):
    fake_gpus(monkeypatch, [(9, 0), (7, 5)])
    assert main._device_assignment(None, 1) == (2, [0, 1])
    fake_gpus(monkeypatch, [])
    monkeypatch.setattr(os, 'cpu_count', lambda: None)
    assert main._device_assignment(None, None) == (1, [])
    assert main._device_assignment(3, None) == (3, [])
    with pytest.raises(ValueError):
        main._device_assignment(0, None)
    with pytest.raises(ValueError):
        main._device_assignment(None, -1)


class ImmediateExecutor:
    """Record pool sizes and run injected fake jobs without worker processes."""
    sizes = []

    def __init__(self, max_workers):
        self.sizes.append(max_workers)

    def submit(self, function, *args):
        future = Future()
        future.set_result(function(*args))
        return future

    def shutdown(self, **kwargs):
        pass


def test_batch_uses_each_device_allocation_and_stable_job_assignment(monkeypatch):
    ImmediateExecutor.sizes = []
    calls = []
    main._shutdown.clear()
    monkeypatch.setattr(main, '_device_assignment', lambda *args: (3, [0, 1, 0]))
    monkeypatch.setattr(main, 'ThreadPoolExecutor', ImmediateExecutor)
    monkeypatch.setattr(main, '_run_one', lambda cmd, dev, lock: calls.append((cmd, dev)) or 0)
    monkeypatch.setattr(main, '_kill_children', lambda: None)
    commands = [[str(i)] for i in range(6)]
    main._run_batch(commands, False, 3, None)
    assert ImmediateExecutor.sizes == [2, 1]
    assert calls == [(commands[i], dev) for dev in [0, 1] for i in range(dev, 6, 2)]


def test_interrupted_last_worker_still_exits_as_interrupted(monkeypatch):
    def interrupted(*args):
        main._shutdown.set()
        return -1

    main._shutdown.clear()
    monkeypatch.setattr(main, '_device_assignment', lambda *args: (1, []))
    monkeypatch.setattr(main, 'ThreadPoolExecutor', ImmediateExecutor)
    monkeypatch.setattr(main, '_run_one', interrupted)
    monkeypatch.setattr(main, '_kill_children', lambda: None)
    try:
        with pytest.raises(SystemExit) as error:
            main._run_batch([['fake']], False, None, None)
        assert error.value.code == 130
    finally:
        main._shutdown.clear()
