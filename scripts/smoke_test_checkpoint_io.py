#!/usr/bin/env python3
"""Smoke-test both historical FlaRe tuple checkpoint schemas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from checkpoint_io import MODEL_TENSOR_NAMES, model_from_checkpoint


def synthetic_checkpoint(include_rgb: bool) -> tuple[object, ...]:
    names = MODEL_TENSOR_NAMES if include_rgb else MODEL_TENSOR_NAMES[1:]
    tensors = tuple(torch.tensor([float(index)]) for index, _ in enumerate(names))
    return tensors + ({"state": {}}, 123.5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    compound = model_from_checkpoint(synthetic_checkpoint(True), "cpu", require_rgb=True)
    legacy = model_from_checkpoint(synthetic_checkpoint(False), "cpu")
    assert tuple(name for name in MODEL_TENSOR_NAMES if name in compound) == MODEL_TENSOR_NAMES
    assert "RGB" not in legacy
    assert tuple(name for name in MODEL_TENSOR_NAMES if name in legacy) == MODEL_TENSOR_NAMES[1:]
    assert compound["training_time_seconds"] == 123.5
    assert legacy["training_time_seconds"] == 123.5

    try:
        model_from_checkpoint(synthetic_checkpoint(False), "cpu", require_rgb=True)
    except ValueError as error:
        assert "no base RGB" in str(error)
    else:
        raise AssertionError("require_rgb=True accepted a 17-entry checkpoint")

    try:
        model_from_checkpoint((torch.zeros(1),) * 16, "cpu")
    except ValueError as error:
        assert "expected 17" in str(error)
    else:
        raise AssertionError("an unsupported tuple length was accepted")

    result = {
        "status": "passed",
        "schemas": {
            "legacy_fla_re_only": {"entries": 17, "has_rgb": False},
            "compound_rgb_fla_re": {"entries": 18, "has_rgb": True},
        },
    }
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
