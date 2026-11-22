"""Resolve workflow sources and preserve names used by historical manifests."""
from functools import lru_cache
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CATALOG = Path(__file__).resolve().parents[1] / 'workflow_catalog.json'
_LEGACY_SOURCES = {
    'cone_data': 'src/grokking_capacity/experiments/cone_workflow/cone_data.py',
    'discrete_source_compatibility': 'src/grokking_capacity/experiments/discrete_workflow/compatibility.py',
}


def catalog():
    return json.loads(CATALOG.read_text())


@lru_cache(maxsize=1)
def _source_paths():
    return {**{name: entry['path'] for group in catalog().values()
               for name, entry in group.items()}, **_LEGACY_SOURCES}


def __getattr__(name):
    # Keep the historical SOURCES dict API without reading data on module import.
    if name == 'SOURCES':
        return _source_paths()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def package_sources(root=ROOT):
    """Files needed when freezing the package for an independent campaign."""
    package = Path(root) / 'src/grokking_capacity'
    resources = [package / 'workflow_catalog.json',
                 *sorted((package / 'experiments/legacy_v1').glob('*.json')),
                 *sorted((package / 'experiments/legacy_v1').glob('*.zip'))]
    return [*sorted(package.rglob('*.py')), *resources]


def script_source(name, root=ROOT):
    return Path(root) / _source_paths()[Path(name).stem]


def frozen_source(campaign, name):
    """Locate an immutable worker; never substitute the live implementation."""
    source = Path(campaign) / 'source'
    historical = source / 'scripts' / name
    if historical.is_file():
        return historical
    packaged = source / _source_paths()[Path(name).stem]
    if packaged.is_file():
        return packaged
    raise FileNotFoundError(f"No frozen {name} source in {source}")
