"""Exercise the launcher with fake tmux/mamba/dispatch; never start training."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    events = tmp_path / "events.jsonl"
    program = f"#!{sys.executable}\n" + '''
import json, os, pathlib, subprocess, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
if name == 'tmux' and args[0] == 'has-session':
    sys.exit(1)
with open(os.environ['TEST_EVENTS'], 'a') as f:
    f.write(json.dumps({'name': name, 'args': args, 'db': os.environ.get('GC_WALLOW_DB')}) + '\\n')
if name == 'tmux' and args[0] == 'new-session':
    sys.exit(subprocess.call(['bash', '-c', args[-1]], cwd=args[args.index('-c')+1]))
if name == 'mamba' and args[0] == 'shell':
    print(':')
if name == 'gc-dispatch' and os.environ.get('TEST_FAIL_CONFIG') in args:
    sys.exit(7)
'''
    for name in ["tmux", "mamba", "gc-dispatch"]:
        path = binary / name
        path.write_text(program)
        path.chmod(0o755)
    env = {**os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"],
           "TEST_EVENTS": str(events), "LOG_DIR": str(tmp_path / "logs with spaces"),
           "GC_WALLOW_DB": str(tmp_path / "local registry.db"),
           "SESSION": "test-run", "WORKERS_PER_GPU": "2", "FORCE": "0"}

    def run(args, *, script="replicate.sh", **overrides):
        result = subprocess.run(["bash", str(ROOT / "scripts" / script), *args],
                                env={**env, **overrides}, cwd=tmp_path,
                                capture_output=True, text=True)
        calls = [json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []
        return result, calls
    return run


def test_subset_preserves_order_and_quotes_environment_and_paths(launcher):
    environment = "scientific env; false"
    result, calls = launcher([environment, "lr_sweep", "configs/task_mul.yaml"], FORCE="1")
    assert result.returncode == 0, result.stderr
    activation = next(call for call in calls if call["name"] == "mamba" and call["args"][0] == "activate")
    assert activation["args"] == ["activate", environment]
    jobs = [call for call in calls if call["name"] == "gc-dispatch"]
    assert [call["args"] for call in jobs] == [
        ["--config", f"configs/{name}.yaml", "--workers-per-gpu", "2", "--force"]
        for name in ["lr_sweep", "task_mul"]
    ]
    assert all(call["db"].endswith("local registry.db") for call in jobs)


def test_default_launch_retains_all_ten_suites(launcher):
    result, calls = launcher(["science"])
    assert result.returncode == 0, result.stderr
    assert [call["args"][1] for call in calls if call["name"] == "gc-dispatch"] == [
        f"configs/{name}.yaml" for name in [
            "central", "weight_decay_sweep", "alpha_sweep", "dropout_sweep", "lr_sweep",
            "init_scale_sweep", "task_add", "depth_scaling", "heads_sweep", "task_mul",
        ]
    ]


def test_failed_suite_stops_before_next_suite(launcher):
    result, calls = launcher(["science", "central", "task_mul"],
                             TEST_FAIL_CONFIG="configs/central.yaml")
    assert result.returncode == 7
    assert len([call for call in calls if call["name"] == "gc-dispatch"]) == 1


def test_unknown_suite_fails_before_starting_tmux(launcher):
    result, calls = launcher(["science", "missing-suite"])
    assert result.returncode == 1
    assert "Config not found" in result.stderr
    assert calls == []


def test_single_suite_wrapper_preserves_argument_order(launcher):
    result, calls = launcher(["central", "science"], script="run_suite.sh", SESSION="")
    assert result.returncode == 0, result.stderr
    session = next(call for call in calls if call["name"] == "tmux")
    assert session["args"][session["args"].index("-s") + 1] == "central"
    assert [call["args"][1] for call in calls if call["name"] == "gc-dispatch"] == ["configs/central.yaml"]
