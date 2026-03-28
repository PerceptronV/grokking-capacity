"""Backfill .meta.json sidecars for all legacy .npz experiment files.

Scans data/capacity/, data/speed/, and data/groks/ and writes a .meta.json
sidecar next to each .npz. Then builds an initial match table pairing speed
and grokking experiments, documenting the weight-decay confound.

Usage:
    python scripts/migrate_legacy_data.py [--data-dir data] [--dry-run] [--verbose] [--force]

The script is idempotent: if .meta.json already exists it is skipped unless
--force is passed. No existing .npz files are modified or moved.
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass

import numpy as np

# Make sure the repo root is importable when running as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from consts import C as DEFAULT_CAPACITY_CONSTANT
from experiment import ExperimentConfig, save_run
from matching import (
    ExperimentMatch,
    build_match_table,
    compute_n_equiv,
    get_param_count,
    save_match_table,
)
from results import ResultsIndex

# ---------------------------------------------------------------------------
# Directory / filename regexes (match confirmed naming from each script)
# ---------------------------------------------------------------------------

CAPACITY_DIR_RE = re.compile(r'^p(\d+)_seed(\d+)_ds(\w+)$')
CAPACITY_FILE_RE = re.compile(r'^capacity_dim(\d+)_samples(\d+)\.npz$')

# Speed dir has exactly p{p}_seed{seed} — no third component.
SPEED_DIR_RE = re.compile(r'^p(\d+)_seed(\d+)$')
SPEED_FILE_RE = re.compile(r'^speed_dim(\d+)_samples(\d+)\.npz$')

GROKS_DIR_RE = re.compile(r'^p(\d+)_seed(\d+)_split(\w+)$')
GROKS_FILE_RE = re.compile(r'^grokking_dim(\d+)_depth(\d+)_heads(\d+)\.npz$')

# Reverse of capacity.py's symb_map (directory suffix → operator symbol)
DS_TO_OP = {
    'random': 'random',
    'add': '+',
    'sub': '-',
    'mul': '*',
    'div': '/',
}


# ---------------------------------------------------------------------------
# Counters for the summary report
# ---------------------------------------------------------------------------

@dataclass
class MigrationStats:
    capacity_migrated: int = 0
    capacity_skipped: int = 0
    speed_migrated: int = 0
    speed_skipped: int = 0
    groks_migrated: int = 0
    groks_skipped: int = 0
    n_samples_warnings: int = 0
    param_count_warnings: int = 0


# ---------------------------------------------------------------------------
# Per-type metadata builders
# ---------------------------------------------------------------------------

def _load_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k].item() if data[k].ndim == 0 else data[k] for k in data.files}


def _build_capacity_meta(
    npz_path: str, p: int, seed: int, dataset_type: str, stats: MigrationStats
) -> dict:
    npz = _load_npz(npz_path)
    dim = int(npz['dim'])
    depth = int(npz['depth'])
    heads = int(npz['heads'])
    param_count = int(npz['param_count'])
    n_samples = int(npz['n_samples'])

    # capacity uses random labels; compute dataset_bits = n_samples * log2(p+2)
    import math
    dataset_bits = n_samples * math.log2(p + 2)

    config = ExperimentConfig(
        experiment_type="capacity",
        p=p,
        operation=DS_TO_OP.get(dataset_type, dataset_type),
        train_fraction=0.5,  # not used by capacity but needed for run_id consistency
        split_type="random",
        n_samples=n_samples,
        dataset_bits=dataset_bits,
        dim=dim,
        depth=depth,
        heads=heads,
        dropout=0.0,           # capacity.py default
        param_count=param_count,
        architecture_family="transformer_gated",
        lr=1e-3,               # capacity.py default
        weight_decay=0.01,     # capacity.py default
        beta1=0.9,
        beta2=0.98,
        batch_size=512,
        max_epochs=5000,
        seed=seed,
        saturation_threshold=99.0,
    )

    results = {
        "final_acc": float(npz.get('final_acc', 0.0)),
        "final_loss": float(npz.get('final_loss', 0.0)),
        "total_bits_memorized": float(npz.get('total_bits_memorized', 0.0)),
        "epochs_trained": int(npz.get('epochs_trained', 0)),
    }

    config.matched_to = None
    config.n_samples_derivation = None
    config.capacity_constant = None
    config.capacity_constant_source = None

    meta = _to_meta_dict(config, results)
    meta["provenance"]["migrated_from"] = os.path.relpath(npz_path)
    meta["provenance"]["migration_note"] = (
        "Backfilled from legacy data. Optimizer/training params are inferred "
        "from capacity.py CLI defaults, not recorded in original data."
    )
    return meta


def _build_speed_meta(
    npz_path: str, p: int, seed: int, stats: MigrationStats
) -> dict:
    import math
    npz = _load_npz(npz_path)
    dim = int(npz['dim'])
    depth = int(npz['depth'])
    heads = int(npz['heads'])
    param_count = int(npz['param_count'])
    n_samples = int(npz['n_samples'])
    dataset_bits = float(npz.get('dataset_bits', n_samples * math.log2(p + 2)))

    # Verify n_samples matches expected value from main.py defaults
    expected_n, _ = compute_n_equiv(p, '/', 0.5)
    n_samples_note = ""
    if n_samples != expected_n:
        stats.n_samples_warnings += 1
        n_samples_note = (
            f" WARNING: stored n_samples={n_samples} does not match "
            f"expected n_equiv={expected_n} for p={p}, op='/', alpha=0.5."
        )

    K_mem = dataset_bits
    derivation = (
        f"K_mem(p={p}, alpha=0.5, op='/') = {K_mem:.1f} bits; "
        f"n_equiv = {expected_n} samples"
    ) + n_samples_note

    config = ExperimentConfig(
        experiment_type="speed",
        p=p,
        operation="/",             # main.py default — not stored in npz
        train_fraction=0.5,        # main.py default — not stored in npz
        split_type="random",
        n_samples=n_samples,
        dataset_bits=dataset_bits,
        dim=dim,
        depth=depth,
        heads=heads,
        dropout=0.2,               # speed.py default
        param_count=param_count,
        architecture_family="transformer_gated",
        lr=1e-3,
        weight_decay=0.01,         # speed.py default — THE CONFOUND
        beta1=0.9,
        beta2=0.98,
        batch_size=512,
        max_epochs=5000,
        seed=seed,
        saturation_threshold=99.0,
        matched_to=f"groks with p={p}, alpha=0.5, op='/'",
        n_samples_derivation=derivation,
        capacity_constant=DEFAULT_CAPACITY_CONSTANT,
        capacity_constant_source="consts.C (legacy)",
    )

    results = {
        "saturated": bool(npz.get('saturated', False)),
        "saturation_epoch": float(npz.get('saturation_epoch', 0.0)),
        "final_acc": float(npz.get('final_acc', 0.0)),
        "final_loss": float(npz.get('final_loss', 0.0)),
    }

    meta = _to_meta_dict(config, results)
    meta["provenance"]["migrated_from"] = os.path.relpath(npz_path)
    meta["provenance"]["migration_note"] = (
        "Backfilled from legacy data. Optimizer params inferred from speed.py CLI defaults. "
        "weight_decay=0.01 does NOT match groks weight_decay=1.0."
    )
    return meta


def _build_groks_meta(
    npz_path: str, p: int, seed: int, split_type: str, stats: MigrationStats
) -> dict:
    npz = _load_npz(npz_path)
    dim = int(npz['dim'])
    depth = int(npz['depth'])
    heads = int(npz['heads'])
    param_count = int(npz['param_count'])

    # 'epochs' in groks npz = max_epochs arg, not actual epochs trained
    max_epochs = int(npz.get('epochs', 5000))
    train_acc = npz.get('train_acc', [])
    epochs_trained = len(train_acc) if hasattr(train_acc, '__len__') else 0

    operation = str(npz.get('op', '/'))
    train_fraction = float(npz.get('train_fraction', 0.5))
    n_train = int(npz.get('n_train', 0))
    n_val = int(npz.get('n_val', 0))

    # Validate param_count
    expected_pc = get_param_count(dim, depth, heads, p)
    if expected_pc != param_count:
        stats.param_count_warnings += 1

    import math
    n_full = p * (p - 1) if operation == '/' else p * p
    dataset_bits = int(n_full * train_fraction) * math.log2(p + 2)

    config = ExperimentConfig(
        experiment_type="groks",
        p=p,
        operation=operation,
        train_fraction=train_fraction,
        split_type=split_type,
        n_samples=n_train,
        dataset_bits=dataset_bits,
        dim=dim,
        depth=depth,
        heads=heads,
        dropout=0.2,           # groks.py default
        param_count=param_count,
        architecture_family="transformer_gated",
        lr=1e-3,
        weight_decay=1.0,      # groks.py default
        beta1=0.9,
        beta2=0.98,
        batch_size=512,
        max_epochs=max_epochs,
        seed=seed,
        saturation_threshold=99.0,
    )

    val_acc = npz.get('val_acc', [])
    final_train_acc = float(train_acc[-1]) if epochs_trained > 0 else 0.0
    final_val_acc = float(val_acc[-1]) if epochs_trained > 0 else 0.0

    results = {
        "final_train_acc": final_train_acc,
        "final_val_acc": final_val_acc,
        "epochs_trained": epochs_trained,
        "n_train": n_train,
        "n_val": n_val,
    }

    meta = _to_meta_dict(config, results)
    meta["provenance"]["migrated_from"] = os.path.relpath(npz_path)
    meta["provenance"]["migration_note"] = (
        "Backfilled from legacy data. Optimizer params inferred from groks.py CLI defaults. "
        "weight_decay=1.0 does NOT match speed weight_decay=0.01."
    )
    return meta


def _to_meta_dict(config: ExperimentConfig, results: dict) -> dict:
    """Convert ExperimentConfig + results dict to the JSON structure."""
    from datetime import datetime
    import subprocess
    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        pass

    return {
        "run_id": config.run_id,
        "experiment_type": config.experiment_type,
        "timestamp": datetime.utcnow().isoformat(),
        "git_hash": git_hash,
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


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def scan_and_migrate(data_dir: str, dry_run: bool, verbose: bool, force: bool) -> MigrationStats:
    stats = MigrationStats()

    for subdir_name in ('capacity', 'speed', 'groks'):
        subdir_path = os.path.join(data_dir, subdir_name)
        if not os.path.isdir(subdir_path):
            print(f"  [warn] {subdir_path} not found, skipping.")
            continue

        for dir_entry in sorted(os.scandir(subdir_path), key=lambda e: e.name):
            if not dir_entry.is_dir():
                continue
            dirname = dir_entry.name

            if subdir_name == 'capacity':
                m = CAPACITY_DIR_RE.match(dirname)
                if not m:
                    continue
                p, seed, dataset_type = int(m.group(1)), int(m.group(2)), m.group(3)

                for fname in sorted(os.listdir(dir_entry.path)):
                    fm = CAPACITY_FILE_RE.match(fname)
                    if not fm:
                        continue
                    npz_path = os.path.join(dir_entry.path, fname)
                    json_path = os.path.splitext(npz_path)[0] + '.meta.json'

                    if os.path.exists(json_path) and not force:
                        stats.capacity_skipped += 1
                        if verbose:
                            print(f"  [skip] {npz_path}")
                        continue

                    if verbose:
                        print(f"  [cap ] {npz_path}")
                    meta = _build_capacity_meta(npz_path, p, seed, dataset_type, stats)
                    if not dry_run:
                        with open(json_path, 'w') as f:
                            json.dump(meta, f, indent=2)
                    stats.capacity_migrated += 1

            elif subdir_name == 'speed':
                m = SPEED_DIR_RE.match(dirname)
                if not m:
                    continue
                p, seed = int(m.group(1)), int(m.group(2))

                for fname in sorted(os.listdir(dir_entry.path)):
                    fm = SPEED_FILE_RE.match(fname)
                    if not fm:
                        continue
                    npz_path = os.path.join(dir_entry.path, fname)
                    json_path = os.path.splitext(npz_path)[0] + '.meta.json'

                    if os.path.exists(json_path) and not force:
                        stats.speed_skipped += 1
                        if verbose:
                            print(f"  [skip] {npz_path}")
                        continue

                    if verbose:
                        print(f"  [spd ] {npz_path}")
                    meta = _build_speed_meta(npz_path, p, seed, stats)
                    if not dry_run:
                        with open(json_path, 'w') as f:
                            json.dump(meta, f, indent=2)
                    stats.speed_migrated += 1

            else:  # groks
                m = GROKS_DIR_RE.match(dirname)
                if not m:
                    continue
                p, seed, split_type = int(m.group(1)), int(m.group(2)), m.group(3)

                for fname in sorted(os.listdir(dir_entry.path)):
                    fm = GROKS_FILE_RE.match(fname)
                    if not fm:
                        continue
                    npz_path = os.path.join(dir_entry.path, fname)
                    json_path = os.path.splitext(npz_path)[0] + '.meta.json'

                    if os.path.exists(json_path) and not force:
                        stats.groks_skipped += 1
                        if verbose:
                            print(f"  [skip] {npz_path}")
                        continue

                    if verbose:
                        print(f"  [grk ] {npz_path}")
                    meta = _build_groks_meta(npz_path, p, seed, split_type, stats)
                    if not dry_run:
                        with open(json_path, 'w') as f:
                            json.dump(meta, f, indent=2)
                    stats.groks_migrated += 1

    return stats


# ---------------------------------------------------------------------------
# Rename legacy files to new path format
# ---------------------------------------------------------------------------

def rename_legacy_to_new_format(data_dir: str, dry_run: bool, verbose: bool) -> dict:
    """Move legacy speed/groks files to the new path format.

    Old speed:  data/speed/p{p}_seed{seed}/speed_dim{d}_samples{n}.npz
    New speed:  data/speed/p{p}_op_div_seed{seed}/speed_dim{d}_samples{n}_wd0.01.npz

    Old groks:  data/groks/p{p}_seed{seed}_split{st}/grokking_dim{d}_depth{dep}_heads{h}.npz
    New groks:  data/groks/p{p}_op_div_seed{seed}_split{st}/grokking_dim{d}_depth{dep}_heads{h}_wd1.0.npz

    Both .npz and their .meta.json sidecars are moved together.
    Old directories are removed after all their files have been moved.
    The operation is idempotent: if the target already exists the file is skipped.
    """
    stats = {'speed_moved': 0, 'speed_skipped': 0, 'groks_moved': 0, 'groks_skipped': 0}

    def _move_pair(old_npz: str, new_npz: str) -> None:
        """Move npz and its sidecar (if present) to new paths."""
        os.makedirs(os.path.dirname(new_npz), exist_ok=True)
        os.rename(old_npz, new_npz)
        old_json = old_npz[:-4] + '.meta.json'
        new_json = new_npz[:-4] + '.meta.json'
        if os.path.exists(old_json):
            os.rename(old_json, new_json)

    # ---- Speed ----
    speed_root = os.path.join(data_dir, 'speed')
    if os.path.isdir(speed_root):
        for entry in sorted(os.scandir(speed_root), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            m = SPEED_DIR_RE.match(entry.name)
            if not m:
                continue  # already new-format dir or unrecognised
            p, seed = int(m.group(1)), int(m.group(2))
            new_dir = os.path.join(speed_root, f'p{p}_op_div_seed{seed}')

            for fname in sorted(os.listdir(entry.path)):
                fm = SPEED_FILE_RE.match(fname)
                if not fm:
                    continue
                dim, n = fm.group(1), fm.group(2)
                old_npz = os.path.join(entry.path, fname)
                new_npz = os.path.join(new_dir, f'speed_dim{dim}_samples{n}_wd0.01.npz')

                if os.path.exists(new_npz):
                    if verbose:
                        print(f'  [skip] {new_npz} already exists')
                    stats['speed_skipped'] += 1
                    continue

                if dry_run or verbose:
                    print(f'  [spd ] {old_npz}  →  {new_npz}')
                if not dry_run:
                    _move_pair(old_npz, new_npz)
                stats['speed_moved'] += 1

            if not dry_run and os.path.isdir(entry.path):
                remaining = [f for f in os.listdir(entry.path) if not f.endswith('.meta.json')]
                if not remaining:
                    # Remove any leftover empty-sidecar or truly-empty dir
                    for leftover in os.listdir(entry.path):
                        os.remove(os.path.join(entry.path, leftover))
                    os.rmdir(entry.path)
                elif verbose:
                    print(f'  [warn] Old dir not empty after rename: {entry.path} '
                          f'({len(remaining)} files remaining)')

    # ---- Groks ----
    groks_root = os.path.join(data_dir, 'groks')
    if os.path.isdir(groks_root):
        for entry in sorted(os.scandir(groks_root), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            m = GROKS_DIR_RE.match(entry.name)
            if not m:
                continue
            p, seed, split_type = int(m.group(1)), int(m.group(2)), m.group(3)
            new_dir = os.path.join(groks_root, f'p{p}_op_div_seed{seed}_split{split_type}')

            for fname in sorted(os.listdir(entry.path)):
                fm = GROKS_FILE_RE.match(fname)
                if not fm:
                    continue
                dim, depth, heads = fm.group(1), fm.group(2), fm.group(3)
                old_npz = os.path.join(entry.path, fname)
                new_npz = os.path.join(new_dir,
                                       f'grokking_dim{dim}_depth{depth}_heads{heads}_wd1.0.npz')

                if os.path.exists(new_npz):
                    if verbose:
                        print(f'  [skip] {new_npz} already exists')
                    stats['groks_skipped'] += 1
                    continue

                if dry_run or verbose:
                    print(f'  [grk ] {old_npz}  →  {new_npz}')
                if not dry_run:
                    _move_pair(old_npz, new_npz)
                stats['groks_moved'] += 1

            if not dry_run and os.path.isdir(entry.path):
                remaining = [f for f in os.listdir(entry.path) if not f.endswith('.meta.json')]
                if not remaining:
                    for leftover in os.listdir(entry.path):
                        os.remove(os.path.join(entry.path, leftover))
                    os.rmdir(entry.path)
                elif verbose:
                    print(f'  [warn] Old dir not empty after rename: {entry.path} '
                          f'({len(remaining)} files remaining)')

    return stats


# ---------------------------------------------------------------------------
# Standardise filenames
# ---------------------------------------------------------------------------

# Intermediate speed dir (post-op-rename, pre-standardise): p{p}_op_{op}_seed{seed}
_SPEED_OP_DIR_RE   = re.compile(r'^p(\d+)_op_(\w+)_seed(\d+)$')
# Intermediate speed file (has wd but no depth/heads, samples in middle)
_SPEED_INTER_RE    = re.compile(r'^speed_dim(\d+)_samples(\d+)_wd([\d.]+)\.npz$')
# Old capacity dir: p{p}_seed{seed}_ds{ds}[_depth{dep}]
_CAP_OLD_DIR_RE    = re.compile(r'^p(\d+)_seed(\d+)_ds(\w+?)(?:_depth(\d+))?$')
# Old capacity file: capacity_dim{d}_samples{n}.npz (no depth/heads/wd)
_CAP_OLD_FILE_RE   = re.compile(r'^capacity_dim(\d+)_samples(\d+)\.npz$')
_DS_TO_OP          = {'random': 'random', 'add': 'add', 'sub': 'sub',
                      'mul': 'mul', 'div': 'div'}


def standardise_filenames(data_dir: str, dry_run: bool, verbose: bool) -> dict:
    """Rename speed and capacity files/dirs to the standard naming convention.

    Speed (intermediate → standard, in-place rename within same directory):
        speed_dim{d}_samples{n}_wd{wd}.npz
        → speed_dim{d}_depth2_heads1_wd{wd}_samples{n}.npz

    Capacity directory:
        p{p}_seed{seed}_ds{ds}[_depth{dep}]  →  p{p}_op_{op}_seed{seed}

    Capacity file:
        capacity_dim{d}_samples{n}.npz
        → capacity_dim{d}_depth{dep}_heads{h}_wd{wd}_samples{n}.npz
        (depth and heads read from the npz; wd read from .meta.json or default 0.01)
    """
    stats = {'speed': 0, 'capacity_files': 0, 'capacity_dirs': 0, 'skipped': 0}

    def _move_pair(old_npz: str, new_npz: str) -> None:
        os.makedirs(os.path.dirname(new_npz), exist_ok=True)
        if dry_run or verbose:
            print(f'  {os.path.relpath(old_npz)}  →  {os.path.relpath(new_npz)}')
        if not dry_run:
            os.rename(old_npz, new_npz)
            old_json = old_npz[:-4] + '.meta.json'
            new_json = new_npz[:-4] + '.meta.json'
            if os.path.exists(old_json):
                os.rename(old_json, new_json)

    # ---- Speed ----
    speed_root = os.path.join(data_dir, 'speed')
    if os.path.isdir(speed_root):
        for entry in sorted(os.scandir(speed_root), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            if not _SPEED_OP_DIR_RE.match(entry.name):
                continue
            for fname in sorted(os.listdir(entry.path)):
                fm = _SPEED_INTER_RE.match(fname)
                if not fm:
                    continue
                dim, n, wd = fm.group(1), fm.group(2), fm.group(3)
                old_npz = os.path.join(entry.path, fname)
                new_npz = os.path.join(entry.path,
                    f'speed_dim{dim}_depth2_heads1_wd{wd}_samples{n}.npz')
                if os.path.exists(new_npz):
                    stats['skipped'] += 1
                    if verbose:
                        print(f'  [skip] {os.path.relpath(new_npz)} already exists')
                    continue
                _move_pair(old_npz, new_npz)
                stats['speed'] += 1

    # ---- Capacity ----
    cap_root = os.path.join(data_dir, 'capacity')
    if os.path.isdir(cap_root):
        for entry in sorted(os.scandir(cap_root), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            m = _CAP_OLD_DIR_RE.match(entry.name)
            if not m:
                continue
            p, seed, ds = int(m.group(1)), int(m.group(2)), m.group(3)
            op = _DS_TO_OP.get(ds, ds)
            new_dir = os.path.join(cap_root, f'p{p}_op_{op}_seed{seed}')

            for fname in sorted(os.listdir(entry.path)):
                fm = _CAP_OLD_FILE_RE.match(fname)
                if not fm:
                    continue
                dim_str, n_str = fm.group(1), fm.group(2)
                old_npz = os.path.join(entry.path, fname)

                # Read depth/heads from npz; wd from .meta.json if available
                npz = _load_npz(old_npz)
                dep = int(npz.get('depth', 2))
                h   = int(npz.get('heads', 1))
                wd_val = 0.01
                old_json = old_npz[:-4] + '.meta.json'
                if os.path.exists(old_json):
                    with open(old_json) as f:
                        meta = json.load(f)
                    wd_val = meta.get('optimizer', {}).get('weight_decay', 0.01)

                new_npz = os.path.join(new_dir,
                    f'capacity_dim{dim_str}_depth{dep}_heads{h}_wd{wd_val}_samples{n_str}.npz')
                if os.path.exists(new_npz):
                    stats['skipped'] += 1
                    if verbose:
                        print(f'  [skip] {os.path.relpath(new_npz)} already exists')
                    continue
                _move_pair(old_npz, new_npz)
                stats['capacity_files'] += 1

            if not dry_run and os.path.isdir(entry.path):
                remaining = os.listdir(entry.path)
                if not remaining:
                    os.rmdir(entry.path)
                    stats['capacity_dirs'] += 1
                elif verbose:
                    print(f'  [warn] Old capacity dir not empty after rename: {entry.path}')

    return stats


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(data_dir: str, verbose: bool) -> int:
    """Run post-migration validation checks. Returns number of errors found."""
    import glob as glob_mod
    errors = 0

    # 1. Round-trip: every .meta.json must have a corresponding .npz
    for json_path in glob_mod.glob(os.path.join(data_dir, '**', '*.meta.json'), recursive=True):
        # Strip '.meta.json' (not just '.json') to get the correct npz path
        npz_path = json_path[:-len('.meta.json')] + '.npz'
        if not os.path.exists(npz_path):
            print(f"  [error] .meta.json has no matching .npz: {json_path}")
            errors += 1

    # 2. Completeness: for primes 97–139, check speed/groks coverage
    index = ResultsIndex(data_dir)
    expected_primes = [p for p in range(97, 140) if _is_prime(p)]
    for p in expected_primes:
        speed = index.query(experiment_type="speed", p=p)
        groks = index.query(experiment_type="groks", p=p)
        if not speed and not groks:
            continue  # prime not in dataset, skip
        if verbose:
            print(f"  p={p}: {len(speed)} speed entries, {len(groks)} groks entries")

    return errors


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


# ---------------------------------------------------------------------------
# Match table builder
# ---------------------------------------------------------------------------

def build_legacy_match_table(data_dir: str, dry_run: bool) -> int:
    """Build match table from migrated sidecars. Returns number of matches."""
    index = ResultsIndex(data_dir)
    groks_entries = index.query(experiment_type="groks")
    speed_entries = index.query(experiment_type="speed")

    matches = build_match_table(
        groks_entries=groks_entries,
        speed_entries=speed_entries,
        capacity_constant=DEFAULT_CAPACITY_CONSTANT,
        capacity_source="consts.C (legacy)",
        param_tolerance=0.05,
    )

    output_path = os.path.join(data_dir, 'legacy_matches.json')
    if not dry_run:
        save_match_table(matches, output_path)
        print(f"  Match table written to {output_path}")

    # Check for wd mismatch (expected for all legacy pairs)
    mismatched = sum(
        1 for m in matches
        if m.weight_decay_speed != m.weight_decay_groks
    )
    if mismatched:
        print(f"  NOTE: {mismatched}/{len(matches)} matches have MISMATCHED "
              f"weight_decay (speed vs groks). This is expected for legacy data.")

    return len(matches)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Backfill .meta.json sidecars for legacy experiment data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data-dir', default='data', help='Root data directory to scan')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be written without writing anything')
    parser.add_argument('--verbose', action='store_true',
                        help='Print each file as it is processed')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing .meta.json files')
    parser.add_argument('--skip-match-table', action='store_true',
                        help='Skip building the match table (faster for testing)')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip post-migration validation checks')
    parser.add_argument('--rename', action='store_true',
                        help='Move legacy files to new path format (op in dir, wd in filename)')
    parser.add_argument('--rename-only', action='store_true',
                        help='Skip sidecar backfill and only perform the rename step')
    parser.add_argument('--standardise', action='store_true',
                        help='Rename speed/capacity files to standard naming convention')
    parser.add_argument('--standardise-only', action='store_true',
                        help='Skip sidecar backfill and only perform the standardise step')
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] No files will be written.\n")

    if not args.rename_only and not args.standardise_only:
        print(f"Scanning {args.data_dir} for legacy files to backfill sidecars...\n")
        stats = scan_and_migrate(
            data_dir=args.data_dir,
            dry_run=args.dry_run,
            verbose=args.verbose,
            force=args.force,
        )

        print("\nSidecar backfill complete.")
        print(f"  Capacity files: {stats.capacity_migrated} migrated, "
              f"{stats.capacity_skipped} skipped (already had .meta.json)")
        print(f"  Speed files:    {stats.speed_migrated} migrated, "
              f"{stats.speed_skipped} skipped")
        print(f"  Grokking files: {stats.groks_migrated} migrated, "
              f"{stats.groks_skipped} skipped")
        if stats.n_samples_warnings:
            print(f"  Warnings: {stats.n_samples_warnings} n_samples mismatches in speed files")
        if stats.param_count_warnings:
            print(f"  Warnings: {stats.param_count_warnings} param_count mismatches in groks files")

    if args.rename or args.rename_only:
        print(f"\nRenaming legacy files to new path format...")
        rename_stats = rename_legacy_to_new_format(
            data_dir=args.data_dir,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        print("\nRename complete.")
        print(f"  Speed files:    {rename_stats['speed_moved']} moved, "
              f"{rename_stats['speed_skipped']} skipped (target already exists)")
        print(f"  Grokking files: {rename_stats['groks_moved']} moved, "
              f"{rename_stats['groks_skipped']} skipped")

    if args.standardise or args.standardise_only:
        print(f"\nStandardising filenames...")
        std_stats = standardise_filenames(
            data_dir=args.data_dir,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        print("\nStandardise complete.")
        print(f"  Speed files:    {std_stats['speed']} renamed")
        print(f"  Capacity files: {std_stats['capacity_files']} renamed, "
              f"{std_stats['capacity_dirs']} old dirs removed")
        print(f"  Skipped:        {std_stats['skipped']} (target already exists)")

    if not args.skip_validation and not args.dry_run:
        print("\nRunning validation...")
        errors = validate(args.data_dir, verbose=args.verbose)
        if errors == 0:
            print("  Validation passed.")
        else:
            print(f"  Validation found {errors} error(s).")

    if not args.skip_match_table and not args.dry_run:
        print("\nBuilding match table...")
        n_matches = build_legacy_match_table(args.data_dir, dry_run=args.dry_run)
        print(f"  Match table: {n_matches} pairs written to {args.data_dir}/legacy_matches.json")


if __name__ == '__main__':
    main()
