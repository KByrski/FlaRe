#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

require_executable() {
    local executable="$1"
    if ! command -v "$executable" >/dev/null 2>&1; then
        echo "Python executable not found or not executable: $executable" >&2
        exit 1
    fi
}

next_output_id() {
    local max_id=0
    local path name numeric_id

    shopt -s nullglob
    for path in "$REPO_DIR"/output/*; do
        [[ -d "$path" ]] || continue
        name="${path##*/}"
        [[ "$name" =~ ^[0-9]+$ ]] || continue
        numeric_id=$((10#$name))
        if (( numeric_id > max_id )); then
            max_id="$numeric_id"
        fi
    done
    shopt -u nullglob

    printf '%d\n' "$((max_id + 1))"
}

run_scene() {
    local python_bin="$1"
    local benchmark="$2"
    local scene_name="$3"
    local scene_path="$4"
    local resolution="$5"
    local output_id model_path

    if [[ ! -d "$scene_path" ]]; then
        echo "Scene directory not found: $scene_path" >&2
        exit 1
    fi

    output_id="$(next_output_id)"
    model_path="$REPO_DIR/output/$output_id"

    echo
    echo "[$benchmark] Training $scene_name at resolution $resolution"
    echo "Dataset: $scene_path"
    echo "Model:   $model_path"

    (
        cd "$REPO_DIR"
        "$python_bin" source/train.py \
            --source_path "$scene_path" \
            --resolution "$resolution"
    )

    if [[ ! -f "$model_path/checkpoints/best.checkpoint" ]]; then
        echo "Best checkpoint was not created: $model_path/checkpoints/best.checkpoint" >&2
        exit 1
    fi

    echo "[$benchmark] Evaluating the best FlaRe test-PSNR checkpoint for $scene_name"
    (
        cd "$REPO_DIR"
        # evaluate.py resolves best.checkpoint through checkpoint_metadata.json
        # and computes the complete PSNR/SSIM/LPIPS evaluation.
        "$python_bin" source/evaluate.py \
            --scene_path "$scene_path" \
            --model_path "$model_path" \
            --resolution "$resolution"
    )
}
