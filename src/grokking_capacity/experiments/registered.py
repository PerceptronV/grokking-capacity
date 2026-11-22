"""Versioned source evidence and campaign locations.

Current hashes always describe current bytes. Historical evidence is checked
against the archived bytes explicitly, never relabelled as current source.
"""
import json
from pathlib import Path
import zipfile

from .arithmetic import atomic_json, sha

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = Path(__file__).with_name("legacy_v1")
REGISTRATIONS = {
    "matched_dropout": "discrete_p97_dropout_match.yaml",
    "multilevel_dropout": "discrete_p97_multilevel_dropout.yaml",
    "categorical_groks": "discrete_p97_categorical_groks.yaml",
    "capacity_scaling": "discrete_p97_capacity_scaling.yaml",
    "dataset_scaling": "discrete_p97_dataset_scaling.yaml",
}
CONTROLLERS = {
    "matched_dropout": "run_p97_dropout_match",
    "multilevel_dropout": "run_p97_multilevel_stage",
    "categorical_groks": "run_p97_categorical_groks",
    "capacity_scaling": "run_p97_capacity_scaling",
    "dataset_scaling": "run_p97_dataset_scaling",
}


def read(path):
    return json.loads(Path(path).read_text())


def protocol_sources(name):
    manifest = read(ARCHIVE / "manifest.json")
    original = manifest["source_sets"][name]
    paths = [p for p in original if not p.startswith("scripts/")]
    shared = ["arithmetic.py", "model_protocols.py", "capacity_matching.py", "registered.py",
              "campaigns/runner.py", "campaigns/config.py", "campaigns/stage.py",
              "independent_audit.py", "legacy_v1/manifest.json"]
    paths += ["src/grokking_capacity/experiments/" + p for p in shared]
    paths += ["src/grokking_capacity/experiments/campaigns/" + CONTROLLERS[name] + ".py"]
    if name == "dataset_scaling":
        paths.append("src/grokking_capacity/experiments/discrete_workflow/report_p97_dataset_scaling.py")
    return list(dict.fromkeys(paths))


def verify_sources(expected, *, root=ROOT, allow_legacy=True):
    """Validate an entire source set against the live tree or an exact archive."""
    if not expected:
        raise ValueError("Missing registered source hashes")
    if all((Path(root) / p).is_file() and sha(Path(root) / p) == digest
           for p, digest in expected.items()):
        return "current"
    if allow_legacy:
        files = read(ARCHIVE / "manifest.json")["files"]
        if all(files.get(p) == digest for p, digest in expected.items()):
            import hashlib
            with zipfile.ZipFile(ARCHIVE / "sources.zip") as archive:
                for p, digest in expected.items():
                    if hashlib.sha256(archive.read(p)).hexdigest() != digest:
                        raise ValueError(f"Archived source was modified: {p}")
            return "legacy-v1"
    raise ValueError("Registered sources differ from both current and archived sources")


def export_legacy(destination):
    """Export an inspectable, exact copy of the original registered sources."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(ARCHIVE / "sources.zip") as archive:
        archive.extractall(destination)
    manifest = read(ARCHIVE / "manifest.json")
    for path, digest in manifest["files"].items():
        if sha(destination / path) != digest:
            raise ValueError(f"Archived source was modified: {path}")
    return destination
