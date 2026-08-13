#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_benchmark_common.sh"

PYTHON_BIN="${PYTHON_BIN:-/path/to/python}"
MIPNERF360_ROOT="${MIPNERF360_ROOT:-/path/to/mipnerf360}"

require_executable "$PYTHON_BIN"

# Outdoor scenes use resolution 4. Flowers and treehill are intentionally omitted.
for scene in bicycle garden stump; do
    run_scene "$PYTHON_BIN" "Mip-NeRF360" "$scene" \
        "$MIPNERF360_ROOT/$scene" 4
done

# Indoor scenes use resolution 2.
for scene in room counter kitchen bonsai; do
    run_scene "$PYTHON_BIN" "Mip-NeRF360" "$scene" \
        "$MIPNERF360_ROOT/$scene" 2
done
