#!/usr/bin/env bash
# Reproduce the paired H100 norm-gain comparison (15 frozen + 15 unfrozen).
# The controller uses GPU 0 and keeps the arms in separate registries.
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
exec python "$repo_dir/scripts/run_normfrozen_h100_paired.py" "$@"
