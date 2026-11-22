"""Smoke tests for the wallow integration layer."""
import datetime as dt

import pytest
from wallow import F, register

from grokking_capacity.registry import (
    build_identifying,
    run_lifecycle,
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
        annotating={"status": "running", "host": "abc123"},
        on_duplicate="return_existing",
    )
    assert r1.was_inserted is True
    # wallow assigns a native uuid at INSERT.
    assert r1.run.uuid

    r2 = register(
        store, identifying=ident,
        annotating={"status": "running", "host": "different"},
        on_duplicate="return_existing",
    )
    assert r2.was_inserted is False
    # return_existing returns the prior row unmodified.
    assert r2.run.host == "abc123"
    assert r2.run.uuid == r1.run.uuid


def test_lifecycle_then_already_completed(isolated_repo):
    ident = build_identifying(
        experiment_type="groks", p=13, dim=8, seed=42,
    )
    with run_lifecycle(ident) as h1:
        first_uuid = h1.uuid
        h1.finalise(results={
            "param_count": 1234,
            "epochs_trained": 10,
            "final_train_acc": 99.5,
            "final_val_acc": 98.0,
        })

    # Re-claim without force should raise AlreadyCompleted (carrying the run).
    with pytest.raises(AlreadyCompleted) as exc:
        with run_lifecycle(ident):
            pass
    assert exc.value.run.uuid == first_uuid

    # With force we re-enter the SAME row (uuid is stable across reruns).
    with run_lifecycle(ident, force=True) as h2:
        assert h2.uuid == first_uuid


def test_lifecycle_creates_artefacts_dir(isolated_repo):
    import os
    ident = build_identifying(
        experiment_type="speed", p=13, dim=8, seed=42, n_samples=100,
    )
    with run_lifecycle(ident) as h:
        assert os.path.isdir(h.artefacts_dir)
        assert h.artefacts_dir.endswith(f"speed/{h.uuid}")


def test_failed_run_records_status(isolated_repo):
    from grokking_capacity.registry import run_lifecycle
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
