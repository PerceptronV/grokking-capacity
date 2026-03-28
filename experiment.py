import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class ExperimentConfig:
    """Complete specification of an experiment run."""
    experiment_type: str  # "speed", "groks", "capacity"

    # Data
    p: int = 97
    operation: str = "/"
    train_fraction: float = 0.5
    split_type: str = "random"
    n_samples: Optional[int] = None
    dataset_bits: Optional[float] = None

    # Model
    dim: int = 64
    depth: int = 2
    heads: int = 1
    dropout: float = 0.2
    param_count: Optional[int] = None
    architecture_family: str = "transformer_gated"
    init_scale: float = 1.0

    # Optimizer
    lr: float = 1e-3
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98

    # Training
    batch_size: int = 512
    max_epochs: int = 5000
    seed: int = 42
    saturation_threshold: float = 99.0

    # Provenance
    matched_to: Optional[str] = None
    n_samples_derivation: Optional[str] = None
    capacity_constant: Optional[float] = None
    capacity_constant_source: Optional[str] = None

    timestamp: Optional[str] = None
    git_hash: Optional[str] = None

    @property
    def run_id(self) -> str:
        """Deterministic ID from hyperparameters (no timestamp)."""
        op_safe = (self.operation
                   .replace('/', 'div')
                   .replace('*', 'mul')
                   .replace('+', 'add')
                   .replace('-', 'sub'))
        parts = [
            self.experiment_type,
            f"p{self.p}",
            f"dim{self.dim}",
            f"depth{self.depth}",
            f"heads{self.heads}",
            f"seed{self.seed}",
            f"wd{self.weight_decay}",
            f"lr{self.lr}",
            f"do{self.dropout}",
            f"op{op_safe}",
        ]
        if self.n_samples is not None:
            parts.append(f"n{self.n_samples}")
        if self.train_fraction != 0.5:
            parts.append(f"tf{self.train_fraction}")
        if self.init_scale != 1.0:
            parts.append(f"is{self.init_scale}")
        return "_".join(parts)


def _get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def save_run(config: ExperimentConfig, results: dict, npz_path: str) -> str:
    """Write .meta.json sidecar alongside npz_path. Returns path of written JSON."""
    if config.timestamp is None:
        config.timestamp = datetime.utcnow().isoformat()
    if config.git_hash is None:
        config.git_hash = _get_git_hash()

    meta = {
        "run_id": config.run_id,
        "experiment_type": config.experiment_type,
        "timestamp": config.timestamp,
        "git_hash": config.git_hash,
        "data": {
            "p": config.p,
            "operation": config.operation,
            "train_fraction": config.train_fraction,
            "split_type": config.split_type,
            "n_samples": config.n_samples,
            "dataset_bits": config.dataset_bits,
        },
        "model": {
            "dim": config.dim,
            "depth": config.depth,
            "heads": config.heads,
            "dropout": config.dropout,
            "param_count": config.param_count,
            "architecture_family": config.architecture_family,
            "init_scale": config.init_scale,
        },
        "optimizer": {
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "beta1": config.beta1,
            "beta2": config.beta2,
        },
        "training": {
            "batch_size": config.batch_size,
            "max_epochs": config.max_epochs,
            "seed": config.seed,
            "saturation_threshold": config.saturation_threshold,
        },
        "results": results,
        "provenance": {
            "matched_to": config.matched_to,
            "n_samples_derivation": config.n_samples_derivation,
            "capacity_constant": config.capacity_constant,
            "capacity_constant_source": config.capacity_constant_source,
        },
    }

    json_path = os.path.splitext(npz_path)[0] + ".meta.json"
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)
    return json_path


def run_exists(npz_path: str) -> bool:
    """Check if the npz result file already exists."""
    return os.path.exists(npz_path)
