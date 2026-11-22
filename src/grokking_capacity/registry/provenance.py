"""Helpers for collecting run-time provenance (host, gpu, git, etc).

These are recorded as annotation fields by both the worker and the dispatcher.
"""
from __future__ import annotations

import socket
import subprocess
from typing import Any

from ..utils.device import gpu_type as _gpu_type
from .store import _repo_root


def _git_output(*arguments: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *arguments], cwd=_repo_root(), stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def _git_short_hash() -> str | None:
    return _git_output("rev-parse", "--short", "HEAD")


def _git_dirty() -> bool | None:
    output = _git_output("status", "--porcelain")
    return None if output is None else bool(output)


def collect_provenance(
    *,
    device: str | None = None,
    node_rank: int | None = None,
) -> dict[str, Any]:
    """Snapshot host/gpu/git info for the current process. Pass `device` once
    selected so we record the actual cuda:N device, not just 'cuda'."""
    out: dict[str, Any] = {
        "host": socket.gethostname(),
        "gpu_type": _gpu_type(device),
    }
    try:
        import torch
        if torch.cuda.is_available():
            out["gpu_count"] = torch.cuda.device_count()
    except Exception:
        pass
    git = _git_short_hash()
    if git is not None:
        out["git_hash"] = git
    dirty = _git_dirty()
    if dirty is not None:
        out["git_dirty"] = dirty
    if node_rank is not None:
        out["node_rank"] = node_rank
    return out
