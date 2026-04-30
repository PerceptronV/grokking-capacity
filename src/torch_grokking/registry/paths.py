"""Artefact-directory helpers.

All artefacts for a run live under `data/<experiment_type>/<run_uuid>/`.
The uuid is opaque to the path layer; the dispatcher decides it (random or,
when re-claiming an existing row, the row's pre-existing run_uuid).
"""
from __future__ import annotations

import os
from pathlib import Path


def _data_root() -> Path:
    override = os.environ.get("TG_DATA_DIR")
    if override:
        return Path(override)
    # Repo-root/data when called from anywhere within the repo.
    here = Path(__file__).resolve()
    return here.parents[3] / "data"


def artefacts_dir_for(experiment_type: str, run_uuid: str) -> Path:
    """Return data/<experiment_type>/<run_uuid>/. Does not create."""
    return _data_root() / experiment_type / run_uuid


def npz_path_for(experiment_type: str, run_uuid: str, name: str = "trace.npz") -> Path:
    return artefacts_dir_for(experiment_type, run_uuid) / name
