"""Project-specific worker lifecycle, layered on ``wallow.contrib.lifecycle``.

wallow v0.2.0 ships a generic claim → run → finalise/fail context manager
(:func:`wallow.contrib.lifecycle.run_lifecycle`). This module wraps it with the
two things that are specific to grokking_capacity:

  * **provenance** — host / gpu / git info, collected once and written as the
    run's ``start_annotating`` (so it lands on the 'running' record);
  * **artefact paths** — resolved from the run's native ``uuid`` via
    ``Store.artefacts_dir`` (layout ``{experiment_type}/{uuid}`` in wallow.toml),
    surfaced on the handle as ``.artefacts_dir`` / ``.npz_path``.

Experiment workers keep their previous call shape: ``with run_lifecycle(...)
as h``, write to ``h.npz_path`` / ``h.artefacts_dir``, then
``h.finalise(results={...})``. The uuid is no longer plumbed in by the
dispatcher — wallow generates it at INSERT and the worker reads it back off its
claim.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from wallow.contrib.lifecycle import AlreadyCompleted, run_lifecycle as _wallow_lifecycle

from .provenance import collect_provenance
from .store import get_store

__all__ = ["AlreadyCompleted", "WorkerHandle", "run_lifecycle"]


class WorkerHandle:
    """Wraps a :class:`wallow.contrib.lifecycle.WorkerHandle` with artefact paths."""

    __slots__ = ("_inner", "artefacts_dir", "npz_path")

    def __init__(self, inner: Any, *, artefacts_dir: str, npz_path: str):
        self._inner = inner
        self.artefacts_dir = artefacts_dir
        self.npz_path = npz_path

    @property
    def run(self) -> Any:
        return self._inner.run

    @property
    def uuid(self) -> str:
        return self._inner.uuid

    def elapsed(self) -> float:
        return self._inner.elapsed()

    def finalise(self, *, results: dict[str, Any]) -> None:
        """Mark the run completed, recording results + artefact paths."""
        self._inner.finalise(
            annotating={
                "npz_path": self.npz_path,
                "artefacts_dir": self.artefacts_dir,
                **results,
            }
        )


@contextmanager
def run_lifecycle(
    identifying: dict[str, Any],
    *,
    force: bool = False,
    db_path: str | None = None,
    device: str | None = None,
    node_rank: int | None = None,
    extra_annotating: dict[str, Any] | None = None,
) -> Iterator[WorkerHandle]:
    """Claim the row, yield a :class:`WorkerHandle`, finalise/fail on exit.

    Raises :class:`wallow.contrib.lifecycle.AlreadyCompleted` (carrying the
    existing ``run``) when the row is already completed and ``force`` is False.
    On any other exception the underlying wallow lifecycle records
    ``status='failed'`` with a traceback excerpt before re-raising.
    """
    store = get_store(db_path)
    start_annotating = collect_provenance(device=device, node_rank=node_rank)
    if extra_annotating:
        start_annotating.update(extra_annotating)

    with _wallow_lifecycle(
        store,
        identifying=identifying,
        force=force,
        start_annotating=start_annotating,
    ) as inner:
        artefacts_dir = store.artefacts_dir(inner.run, mkdir=True)
        npz_path = artefacts_dir / "trace.npz"
        handle = WorkerHandle(
            inner,
            artefacts_dir=str(artefacts_dir),
            npz_path=str(npz_path),
        )
        yield handle
        # If the body returned without calling finalise, still record the
        # artefact paths on the completion row (wallow's default finalise omits
        # them). Idempotent: a prior finalise() makes this a no-op.
        if not inner._finalised:
            handle.finalise(results={})
