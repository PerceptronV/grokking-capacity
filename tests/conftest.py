"""Test fixtures for grokking_capacity + wallow integration."""
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def isolated_repo(tmp_path, monkeypatch):
    """Point both wallow and the artefact tree at a tmp dir for one test.

    - GC_WALLOW_TOML keeps using the real schema from the repo (it's a static spec).
    - GC_WALLOW_DB → tmp/runs.db
    - GC_DATA_DIR  → tmp/data
    """
    monkeypatch.setenv("GC_WALLOW_TOML", str(REPO_ROOT / "wallow.toml"))
    monkeypatch.setenv("GC_WALLOW_DB", str(tmp_path / "runs.db"))
    monkeypatch.setenv("GC_DATA_DIR", str(tmp_path / "data"))

    # Reset the lru_cache on Store/Schema so the new env vars take effect.
    from grokking_capacity.registry import store as _store_mod
    _store_mod.get_store.cache_clear()
    _store_mod.get_schema.cache_clear()

    yield tmp_path

    _store_mod.get_store.cache_clear()
    _store_mod.get_schema.cache_clear()
