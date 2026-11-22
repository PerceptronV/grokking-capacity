"""Cached accessors for the project's wallow Store and Schema.

Both `wallow.toml` and `runs.db` live at the repository root by default; both
locations can be overridden via env var (`GC_WALLOW_TOML`, `GC_WALLOW_DB`).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from wallow import Store, load_schema
from wallow.schema import Schema


def _repo_root() -> Path:
    """Repo root, located by walking up from this file."""
    return Path(__file__).resolve().parents[3]


def schema_path() -> Path:
    override = os.environ.get("GC_WALLOW_TOML")
    if override:
        return Path(override)
    return _repo_root() / "wallow.toml"


def default_db_path() -> Path:
    override = os.environ.get("GC_WALLOW_DB")
    if override:
        return Path(override)
    return _repo_root() / "runs.db"


def _data_root() -> Path:
    """Artefacts root: $GC_DATA_DIR if set, else <repo-root>/data.

    Anchored to the repo root (not the cwd) so workers launched from anywhere
    write under the same tree — matching the pre-v0.2.0 `registry/paths.py`.
    """
    override = os.environ.get("GC_DATA_DIR")
    if override:
        return Path(override)
    return _repo_root() / "data"


@lru_cache(maxsize=1)
def get_schema() -> Schema:
    schema = load_schema(schema_path())
    # wallow.toml declares artefacts_root = "data" (relative). Resolve it to an
    # absolute, repo-anchored path (honouring GC_DATA_DIR) and re-validate the
    # layout so Store.artefacts_dir() returns stable paths from any cwd.
    schema.artefacts_root = str(_data_root())
    schema.validate_layout()
    return schema


@lru_cache(maxsize=4)
def get_store(db_path: str | None = None, *, check_schema: bool = True) -> Store:
    """Return a cached Store. Pass db_path=':memory:' for tests."""
    path = db_path if db_path is not None else str(default_db_path())
    return Store(path, schema=get_schema(), check_schema=check_schema)


def artefacts_dir_for_row(row: dict, *parts: str, mkdir: bool = False) -> Path:
    """Resolve a row's artefacts dir from its ``experiment_type`` + ``uuid``.

    A thin shim over ``Store.artefacts_dir`` for the analysis layer, which
    holds rows as plain dicts rather than Run objects. The layout
    (``{experiment_type}/{uuid}``) only references those two fields.
    """
    from types import SimpleNamespace

    ns = SimpleNamespace(experiment_type=row["experiment_type"], uuid=row["uuid"])
    return get_store().artefacts_dir(ns, *parts, mkdir=mkdir)


def npz_path_for_row(row: dict, name: str = "trace.npz") -> Path:
    """Canonical trace path for a row: ``<artefacts_dir>/<name>``."""
    return artefacts_dir_for_row(row, name)
