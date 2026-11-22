"""Campaign registration and immutable input handling."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import time
import yaml

from ..registered import ROOT, read, sha, atomic_json


def options(protocol, default_root, argv=None):
    parser = argparse.ArgumentParser(description=f"Run the {protocol} campaign")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "campaigns" / ("dropout_match.yaml" if protocol == "matched_dropout" else protocol + ".yaml"))
    parser.add_argument("--root", type=Path, default=Path(default_root))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-device", default="cpu")
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    return args


def initialize(root, registration, hashes, *, resume=False):
    root, registration = Path(root), Path(registration)
    config = yaml.safe_load(registration.read_text())
    if root.exists():
        if not resume:
            raise FileExistsError(f"Campaign exists; use --resume: {root}")
        manifest = read(root / "manifest.json")
        if manifest.get("version") != 2:
            raise ValueError("This is a historical registration. Export its v1 sources or use a new --root.")
        if manifest["source_hashes"] != hashes or manifest["registration_sha256"] != sha(registration):
            raise ValueError("Registered sources or configuration changed")
        return config
    root.mkdir(parents=True)
    for name in ("jobs", "audits", "stages", "source"):
        (root / name).mkdir()
    shutil.copy2(registration, root / "registration.yaml")
    for name, digest in hashes.items():
        destination = root / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
        if sha(destination) != digest:
            raise ValueError(f"Source changed while registering: {name}")
    atomic_json(root / "manifest.json", {"version": 2, "source_hashes": hashes,
                "registration_sha256": sha(registration), "started_at": time.time(), "pid": os.getpid(),
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
    atomic_json(root / "job_registry.json", {})
    return config


def freeze_selection(path, value):
    """Reuse the first selection timestamp when replaying a completed phase."""
    if Path(path).exists():
        original = read(path)
        if {k: v for k, v in original.items() if k != "frozen_at"} != {k: v for k, v in value.items() if k != "frozen_at"}:
            raise ValueError(f"Frozen selection changed: {path}")
        return original
    atomic_json(path, value)
    return value


def check_registration(root, hashes):
    manifest = read(Path(root) / "manifest.json")
    if manifest["source_hashes"] != hashes or manifest["registration_sha256"] != sha(Path(root) / "registration.yaml"):
        raise ValueError("Registered sources or configuration changed")
