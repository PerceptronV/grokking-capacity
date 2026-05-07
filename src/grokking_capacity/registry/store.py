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


@lru_cache(maxsize=1)
def get_schema() -> Schema:
    return load_schema(schema_path())


@lru_cache(maxsize=4)
def get_store(db_path: str | None = None, *, check_schema: bool = True) -> Store:
    """Return a cached Store. Pass db_path=':memory:' for tests."""
    path = db_path if db_path is not None else str(default_db_path())
    return Store(path, schema=get_schema(), check_schema=check_schema)
