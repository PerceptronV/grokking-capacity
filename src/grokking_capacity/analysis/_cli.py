"""Shared config selection and output paths for analysis commands."""
from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "results"


def analysis_parser(prog: str, description: str, *, subdir: str = "",
                    output_dir: str = "results") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=description)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="Path to a dispatcher suite YAML")
    source.add_argument("--all", action="store_true",
                        help="Analyze all dispatcher suites in configs/ (excludes discrete recipes)")
    destination = f"{output_dir}/<config_name>/" + (f"{subdir}/" if subdir else "")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"Output directory (default: {destination})")
    parser.add_argument("--db", default=None,
                        help="Path to runs.db (default: GC_WALLOW_DB env or repo runs.db)")
    return parser


def suite_configs(directory: Path) -> Iterator[Path]:
    """Discover registry-backed suites, excluding standalone campaign schemas.

    Discrete recipes use ``runs`` or campaign registrations instead of the
    dispatcher's ``experiments`` mapping. They need their own experiment runners.
    Invalid YAML still raises an error rather than silently dropping a suite.
    """
    for path in sorted(directory.glob("*.yaml")):
        spec = yaml.safe_load(path.read_text())
        if isinstance(spec, dict) and isinstance(spec.get("experiments"), dict):
            yield path


def config_jobs(args: argparse.Namespace, *, configs_dir: Path,
                output_root: Path, prog: str,
                subdir: str = "") -> Iterator[tuple[Path, Path]]:
    """Keep single-config overrides and per-suite output paths consistent."""
    if args.all:
        if args.out:
            print(f"[{prog}] --out is ignored with --all", file=sys.stderr)
        for config in suite_configs(configs_dir):
            yield config, output_root / config.stem / subdir
    else:
        yield args.config, args.out or output_root / args.config.stem / subdir
