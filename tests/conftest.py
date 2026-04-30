"""Test fixtures for torch_grokking + wallow integration."""
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def isolated_repo(tmp_path, monkeypatch):
    """Point both wallow and the artefact tree at a tmp dir for one test.

    - TG_WALLOW_TOML keeps using the real schema from the repo (it's a static spec).
    - TG_WALLOW_DB → tmp/runs.db
    - TG_DATA_DIR  → tmp/data
    """
    monkeypatch.setenv("TG_WALLOW_TOML", str(REPO_ROOT / "wallow.toml"))
    monkeypatch.setenv("TG_WALLOW_DB", str(tmp_path / "runs.db"))
    monkeypatch.setenv("TG_DATA_DIR", str(tmp_path / "data"))

    # Reset the lru_cache on Store/Schema so the new env vars take effect.
    from torch_grokking.registry import store as _store_mod
    _store_mod.get_store.cache_clear()
    _store_mod.get_schema.cache_clear()

    yield tmp_path

    _store_mod.get_store.cache_clear()
    _store_mod.get_schema.cache_clear()
