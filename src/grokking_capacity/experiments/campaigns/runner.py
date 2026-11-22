"""Durable subprocess execution shared by campaign planners.

Completed jobs are reused only with unchanged specifications and artifacts.
Resume retries failed audits, but never overwrites partial training outputs.
"""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..arithmetic import atomic_json, sha
from ..registered import ROOT, read


@contextmanager
def exclusive(path):
    with Path(path).open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another controller holds {path}") from error
        yield


def freeze_json(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Frozen inputs changed: {path}")
    else:
        atomic_json(path, value)


def artifact_hashes(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob('*')) if p.is_file()}


def process_start(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def run_stage(root, name, specs, devices, module, *, resume=False,
              source_check=None, audit=None, poll_seconds=1, environment=None):
    """Execute immutable job configs; audit callbacks run after successful jobs."""
    root = Path(root)
    specs = list(specs)
    devices = list(devices)
    names = [s["name"] for s in specs]
    if not devices or len(set(names)) != len(names):
        raise ValueError("Require device slots and unique job names")
    if any(Path(n).name != n or n in (".", "..") for n in [name, *names]):
        raise ValueError("Stage and job names must be single path components")
    stage = root / "stages" / name
    if stage.exists() and not resume:
        raise FileExistsError(f"Stage exists; use --resume: {stage}")
    stage.mkdir(parents=True, exist_ok=True)
    (root / "jobs").mkdir(exist_ok=True)
    (root / "audits").mkdir(exist_ok=True)
    with exclusive(root / "controller.lock"), exclusive(stage / "controller.lock"):
        if source_check:
            source_check()
        freeze_json(stage / "plan.json", {"module": module, "jobs": specs})
        freeze_json(stage / "jobs.json", specs)
        with exclusive(root / "registry.lock"):
            registry_path = root / "job_registry.json"
            registry = read(registry_path) if registry_path.exists() else {}
            for spec in specs:
                if spec["name"] in registry and registry[spec["name"]] != spec["config"]:
                    raise ValueError(f"Job configuration changed: {spec['name']}")
                registry[spec["name"]] = spec["config"]
                freeze_json(stage / (spec["name"] + ".json"), spec["config"])
            atomic_json(registry_path, registry)
        state_path = stage / "controller.json"
        state = read(state_path) if state_path.exists() else {
            "started_at": time.time(), "completed": [], "audited": {}}
        completed_at = state.get("finished_at") if state.get("phase") == "complete" else None
        if state.get("running"):
            for job in state["running"]:
                try:
                    os.kill(job["pid"], 0)
                    if job.get("process_start") is not None and process_start(job["pid"]) != job["process_start"]:
                        continue
                except ProcessLookupError:
                    continue
                raise RuntimeError(f"Previous worker may still be live: {job['name']}")
        done = {j["name"]: j for j in state["completed"] if j["exit_code"] == 0}
        for job in done.values():
            summary = root / "jobs" / job["name"] / "summary.json"
            if not summary.is_file() or artifact_hashes(summary.parent) != job["artifact_hashes"]:
                raise ValueError(f"Completed output changed: {summary}")
        queue = [s for s in specs if s["name"] not in done]
        for spec in queue:
            if (root / "jobs" / spec["name"]).exists():
                raise RuntimeError(f"Partial output requires worker-specific recovery: {spec['name']}")
        state.update(pid=os.getpid(), phase="running", jobs=specs, errors=[], running=[], pending=names)
        state.setdefault("audited", {})
        active = {}
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"),
               "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", **(environment or {})}
        def save():
            state.update(running=[v[2] for v in active.values()], pending=[s["name"] for s in queue])
            atomic_json(state_path, state)
        try:
            save()
            while queue or active:
                for slot, (proc, handle, entry) in list(active.items()):
                    code = proc.poll()
                    if code is None:
                        continue
                    handle.close()
                    result = {**entry, "exit_code": code, "finished_at": time.time()}
                    summary = root / "jobs" / entry["name"] / "summary.json"
                    if code == 0:
                        if not summary.is_file():
                            result["exit_code"] = 99
                        else:
                            result["artifact_hashes"] = artifact_hashes(summary.parent)
                    state["completed"].append(result)
                    if result["exit_code"]:
                        state["errors"].append(entry["name"])
                    del active[slot]
                    save()
                if state["errors"]:
                    state["not_launched_after_failure"] = [s["name"] for s in queue]
                    queue.clear()
                for slot, device in enumerate(devices):
                    if slot in active or not queue or state["errors"]:
                        continue
                    if source_check:
                        source_check()
                    spec = queue.pop(0)
                    command = [sys.executable, "-u", "-m", module,
                               "--config", str(stage / (spec["name"] + ".json")),
                               "--output", str(root / "jobs" / spec["name"]), "--device", device]
                    handle = (stage / (spec["name"] + ".log")).open("a")
                    try:
                        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                                stderr=subprocess.STDOUT)
                    except BaseException:
                        handle.close()
                        raise
                    active[slot] = (proc, handle, {"name": spec["name"], "pid": proc.pid,
                                                 "device": device, "command": command, "process_start": process_start(proc.pid),
                                                 "started_at": time.time()})
                    save()
                if active:
                    time.sleep(poll_seconds)
            if state["errors"]:
                raise RuntimeError(f"Stage {name} failed: {state['errors']}")
            state["phase"] = "auditing"
            save()
            paths = [root / "jobs" / s["name"] for s in specs]
            for path in paths:
                binding = artifact_hashes(path)
                if source_check:
                    source_check()
                if audit and state["audited"].get(path.name) != binding:
                    audit(path)
                    state["audited"][path.name] = binding
                    save()
            if source_check:
                source_check()
            state.update(phase="complete", finished_at=completed_at or time.time())
            save()
            return paths
        except BaseException as error:
            # Preserve live worker identities; a subsequent resume refuses duplicates.
            state.update(phase="failed", error=str(error), finished_at=time.time())
            save()
            raise
        finally:
            for _, handle, _ in active.values():
                handle.close()
