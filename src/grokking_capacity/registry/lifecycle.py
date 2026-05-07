"""High-level claim/finalise helpers for an experiment worker.

Encapsulates the four wallow `register()` calls that wrap a training run
(claim → start → success/failure) so the experiment scripts focus on training.
"""
from __future__ import annotations

import datetime as _dt
import os
import traceback
import uuid as _uuid
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Iterator

from wallow import register

from .paths import artefacts_dir_for, npz_path_for
from .provenance import collect_provenance
from .store import get_store


class AlreadyCompleted(Exception):
    """Raised when a worker finds its row already marked completed and --force was not set."""

    def __init__(self, run, run_uuid: str):
        super().__init__(f"run {run_uuid} already completed")
        self.run = run
        self.run_uuid = run_uuid


class WorkerHandle:
    """Mutable state for one worker run between claim and finalise."""

    __slots__ = ("identifying", "run_uuid", "artefacts_dir", "npz_path", "_t0", "_db_path", "_device")

    def __init__(
        self,
        identifying: dict[str, Any],
        run_uuid: str,
        artefacts_dir: str,
        npz_path: str,
        db_path: str | None,
        device: str | None,
    ):
        self.identifying = identifying
        self.run_uuid = run_uuid
        self.artefacts_dir = artefacts_dir
        self.npz_path = npz_path
        self._t0 = perf_counter()
        self._db_path = db_path
        self._device = device

    def store(self):
        return get_store(self._db_path)

    def elapsed(self) -> float:
        return perf_counter() - self._t0

    def finalise(self, *, results: dict[str, Any]) -> None:
        """Mark the run completed, recording results + npz_path + wallclock."""
        store = self.store()
        annotating = {
            "status": "completed",
            "completed_at": _dt.datetime.now(_dt.timezone.utc),
            "wallclock_seconds": float(self.elapsed()),
            "npz_path": self.npz_path,
            "artefacts_dir": self.artefacts_dir,
            **results,
        }
        register(
            store,
            identifying=self.identifying,
            annotating=annotating,
            on_duplicate="overwrite",
        )

    def fail(self, exc: BaseException) -> None:
        store = self.store()
        excerpt = "".join(traceback.format_exception_only(type(exc), exc)).strip()[:1000]
        register(
            store,
            identifying=self.identifying,
            annotating={
                "status": "failed",
                "completed_at": _dt.datetime.now(_dt.timezone.utc),
                "wallclock_seconds": float(self.elapsed()),
                "error_excerpt": excerpt,
            },
            on_duplicate="overwrite",
        )


def claim(
    identifying: dict[str, Any],
    *,
    run_uuid: str | None = None,
    force: bool = False,
    db_path: str | None = None,
    device: str | None = None,
    node_rank: int | None = None,
    extra_annotating: dict[str, Any] | None = None,
) -> WorkerHandle:
    """Claim a row for execution. Raises AlreadyCompleted if status=completed and not force.

    On first call (no existing row): inserts a new row with the supplied (or
    freshly generated) `run_uuid`, marks it `running`. Idempotent on retry.
    """
    store = get_store(db_path)
    chosen_uuid = run_uuid or _uuid.uuid4().hex[:12]

    # Step 1 — claim the slot or read back the existing row.
    pre = register(
        store,
        identifying=identifying,
        annotating={
            "status": "pending",
            "run_uuid": chosen_uuid,
        },
        on_duplicate="return_existing",
    )
    run = pre.run
    if not pre.was_inserted:
        # Row already existed; reuse its run_uuid (don't overwrite with our fresh one).
        if run.status == "completed" and not force:
            raise AlreadyCompleted(run, run.run_uuid)
        chosen_uuid = run.run_uuid or chosen_uuid

    # Step 2 — annotate as running with provenance + paths.
    artefacts_dir = str(artefacts_dir_for(identifying["experiment_type"], chosen_uuid))
    os.makedirs(artefacts_dir, exist_ok=True)
    npz_path = str(npz_path_for(identifying["experiment_type"], chosen_uuid))

    annotating = {
        "status": "running",
        "run_uuid": chosen_uuid,
        "started_at": _dt.datetime.now(_dt.timezone.utc),
        "artefacts_dir": artefacts_dir,
        "npz_path": npz_path,
        **collect_provenance(device=device, node_rank=node_rank),
    }
    if extra_annotating:
        annotating.update(extra_annotating)
    register(
        store,
        identifying=identifying,
        annotating=annotating,
        on_duplicate="overwrite",
    )
    return WorkerHandle(
        identifying=identifying,
        run_uuid=chosen_uuid,
        artefacts_dir=artefacts_dir,
        npz_path=npz_path,
        db_path=db_path,
        device=device,
    )


@contextmanager
def run_lifecycle(
    identifying: dict[str, Any],
    *,
    run_uuid: str | None = None,
    force: bool = False,
    db_path: str | None = None,
    device: str | None = None,
    node_rank: int | None = None,
    extra_annotating: dict[str, Any] | None = None,
) -> Iterator[WorkerHandle]:
    """Context manager: claim on entry, mark failed on exception. Caller calls
    handle.finalise(results=...) on success."""
    handle = claim(
        identifying=identifying,
        run_uuid=run_uuid,
        force=force,
        db_path=db_path,
        device=device,
        node_rank=node_rank,
        extra_annotating=extra_annotating,
    )
    try:
        yield handle
    except BaseException as exc:
        handle.fail(exc)
        raise
