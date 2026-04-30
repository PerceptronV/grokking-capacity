"""Smoke tests for the wallow integration layer."""
import datetime as dt

import pytest
from wallow import F, register

from torch_grokking.registry import (
    build_identifying,
    claim,
    AlreadyCompleted,
    get_store,
)


def test_schema_loads(isolated_repo):
    store = get_store()
    schema = store.schema
    for required in ("experiment_type", "p", "dim", "seed",
                     "weight_decay", "dropout", "n_samples"):
        assert required in schema.identifying, f"missing identifying field: {required}"


def test_build_identifying_groks_derives_n_samples(isolated_repo):
    ident = build_identifying(
        experiment_type="groks", p=97, dim=64, seed=42,
    )
    # p=97, op='/', tf=0.5  →  n = 97*96*0.5 = 4656
    assert ident["n_samples"] == 4656
    assert ident["dataset_type"] == "modular"


def test_build_identifying_speed_random_dataset(isolated_repo):
    ident = build_identifying(
        experiment_type="speed", p=97, dim=64, seed=42, n_samples=4656,
    )
    assert ident["dataset_type"] == "random"


def test_register_then_dedup(isolated_repo):
    store = get_store()
    ident = build_identifying(
        experiment_type="capacity", p=97, dim=16, seed=42,
        n_samples=1000, dataset_type="random",
    )
    r1 = register(
        store, identifying=ident,
        annotating={"status": "running", "run_uuid": "abc123"},
        on_duplicate="return_existing",
    )
    assert r1.was_inserted is True

    r2 = register(
        store, identifying=ident,
        annotating={"status": "running", "run_uuid": "different"},
        on_duplicate="return_existing",
    )
    assert r2.was_inserted is False
    # run_uuid was NOT overwritten — return_existing returns the prior row.
    assert r2.run.run_uuid == "abc123"


def test_claim_then_already_completed(isolated_repo):
    ident = build_identifying(
        experiment_type="groks", p=13, dim=8, seed=42,
    )
    h1 = claim(ident)
    # Worker finishes successfully.
    h1.finalise(results={
        "param_count": 1234,
        "epochs_trained": 10,
        "final_train_acc": 99.5,
        "final_val_acc": 98.0,
    })

    # Re-claim without --force should raise AlreadyCompleted.
    with pytest.raises(AlreadyCompleted):
        claim(ident)

    # With --force we get a fresh handle pointing at the SAME run_uuid.
    h2 = claim(ident, force=True)
    assert h2.run_uuid == h1.run_uuid


def test_claim_creates_artefacts_dir(isolated_repo):
    import os
    ident = build_identifying(
        experiment_type="speed", p=13, dim=8, seed=42, n_samples=100,
    )
    h = claim(ident)
    assert os.path.isdir(h.artefacts_dir)
    assert h.artefacts_dir.endswith(f"speed/{h.run_uuid}")


def test_failed_run_records_status(isolated_repo):
    from torch_grokking.registry import run_lifecycle
    ident = build_identifying(
        experiment_type="speed", p=13, dim=8, seed=42, n_samples=100,
    )
    with pytest.raises(RuntimeError):
        with run_lifecycle(ident) as h:
            raise RuntimeError("simulated failure")

    store = get_store()
    rows = store.where(F("experiment_type") == "speed").all()
    assert len(rows) == 1
    assert rows[0].status == "failed"
    assert "simulated failure" in (rows[0].error_excerpt or "")
