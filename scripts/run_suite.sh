#!/usr/bin/env bash
# Compatibility entry point for one suite; all launch logic lives in replicate.sh.
# Usage: scripts/run_suite.sh <suite-or-config> <mamba-env>
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "Usage: $(basename "$0") <suite-or-config> <mamba-env>" >&2
    exit 2
fi
suite_name="$(basename "$1")"
suite_name="${suite_name%.yaml}"
suite_name="${suite_name%.yml}"
export SESSION="${SESSION:-$suite_name}"
exec "$(dirname "${BASH_SOURCE[0]}")/replicate.sh" "$2" "$1"
