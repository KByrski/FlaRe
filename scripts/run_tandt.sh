#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_benchmark_common.sh"

PYTHON_BIN="${PYTHON_BIN:-/path/to/python}"
TANDT_ROOT="${TANDT_ROOT:-/path/to/tandt}"

require_executable "$PYTHON_BIN"

for scene in truck train; do
    run_scene "$PYTHON_BIN" "Tanks and Temples" "$scene" \
        "$TANDT_ROOT/$scene" 1
done
