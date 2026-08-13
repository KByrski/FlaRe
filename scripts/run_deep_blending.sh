#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_benchmark_common.sh"

PYTHON_BIN="${PYTHON_BIN:-/path/to/python}"
DEEP_BLENDING_ROOT="${DEEP_BLENDING_ROOT:-/path/to/deep_blending}"

require_executable "$PYTHON_BIN"

for scene in drjohnson playroom; do
    run_scene "$PYTHON_BIN" "Deep Blending" "$scene" \
        "$DEEP_BLENDING_ROOT/$scene" 1
done
