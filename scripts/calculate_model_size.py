#!/usr/bin/env python3
"""Estimate the inference-model size stored in a FlaRe checkpoint."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Sequence

import torch


# Paper: model-memory accounting used for the compact-representation results.
# The formula excludes optimizer state and serialization/container overhead.
FORMULA_BYTES_PER_GAUSSIAN = 248
FORMULA_FIXED_BYTES = 78_232

TENSOR_NAMES = (
    "RGB",
    "alpha",
    "kappa",
    "W_1_uv",
    "W_1_v",
    "W_1_conditioning",
    "b_1",
    "W_2",
    "b_2",
    "W_3",
    "b_3",
    "conditioning",
    "features",
    "m",
    "s",
    "q",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate the unpadded inference-model size using "
            "248 * |G| + 78,232 bytes."
        )
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to a FlaRe .checkpoint file.",
    )
    return parser.parse_args()


def format_size(size_bytes: int) -> str:
    return (
        f"{size_bytes:,} bytes\n"
        f"  {size_bytes / 1_000_000:.6f} MB\n"
        f"  {size_bytes / 1024**2:.6f} MiB\n"
        f"  {size_bytes / 1_000_000_000:.9f} GB\n"
        f"  {size_bytes / 1024**3:.9f} GiB"
    )


def load_model_tensors(checkpoint_path: Path) -> Sequence[torch.Tensor]:
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    checkpoint = torch.load(checkpoint_path, **load_kwargs)
    if not isinstance(checkpoint, (tuple, list)) or len(checkpoint) < 16:
        raise ValueError(
            "Unsupported checkpoint format: expected a tuple/list containing "
            "at least 16 entries."
        )

    tensors = checkpoint[:16]
    for name, tensor in zip(TENSOR_NAMES, tensors):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Checkpoint entry {name!r} is not a tensor.")
    return tensors


def inference_size_from_checkpoint_shapes(tensors: Sequence[torch.Tensor]) -> int:
    (
        rgb,
        alpha,
        kappa,
        w1_uv,
        w1_v,
        w1_conditioning,
        b1,
        w2,
        b2,
        w3,
        b3,
        conditioning,
        features,
        m,
        s,
        q,
    ) = tensors

    # Paper: inference stores the shared decoder and z_i descriptors in FP16;
    # evaluate.py converts W_1, W_2, W_3, and conditioning to FP16.
    fp16_tensors = (w1_uv, w1_v, w1_conditioning, w2, w3, conditioning)
    # The remaining tensors passed to the renderer stay in FP32.
    fp32_tensors = (rgb, alpha, kappa, b1, b2, b3, features, m, s, q)
    return sum(t.numel() * 2 for t in fp16_tensors) + sum(
        t.numel() * 4 for t in fp32_tensors
    )


def shape_warnings(
    tensors: Sequence[torch.Tensor], gaussian_count: int
) -> list[str]:
    expected_shapes = {
        "RGB": (gaussian_count, 3),
        "alpha": (gaussian_count, 1),
        "kappa": (gaussian_count, 1),
        "b_1": (64,),
        "W_2": (64, 64),
        "b_2": (64,),
        "W_3": (4, 64),
        "b_3": (4,),
        "conditioning": (gaussian_count, 96),
        "features": (13_154,),
        "m": (gaussian_count, 3),
        "s": (gaussian_count, 2),
        "q": (gaussian_count, 4),
    }

    warnings = []
    named_tensors = dict(zip(TENSOR_NAMES, tensors))
    w1_elements = sum(named_tensors[name].numel() for name in TENSOR_NAMES[3:6])
    if w1_elements != 64 * 128:
        warnings.append(
            f"W_1 contains {w1_elements:,} elements; the formula expects "
            f"{64 * 128:,}."
        )

    for name, expected in expected_shapes.items():
        actual = tuple(named_tensors[name].shape)
        if actual != expected:
            warnings.append(
                f"{name} has shape {actual}; the formula expects {expected}."
            )
    return warnings


def main() -> int:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    tensors = load_model_tensors(checkpoint_path)
    gaussian_count = int(tensors[13].shape[0])
    formula_size = (
        FORMULA_BYTES_PER_GAUSSIAN * gaussian_count + FORMULA_FIXED_BYTES
    )
    shape_based_size = inference_size_from_checkpoint_shapes(tensors)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Gaussians |G|: {gaussian_count:,}")
    print(
        "\nProvided formula:\n"
        f"  248 * {gaussian_count:,} + 78,232 = {formula_size:,} bytes"
    )
    print(format_size(formula_size))

    warnings = shape_warnings(tensors, gaussian_count)
    if warnings:
        print("\nShape differences:")
        for warning in warnings:
            print(f"  - {warning}")
        print("\nSize using the checkpoint's actual inference tensor shapes:")
        print(format_size(shape_based_size))

    disk_size = checkpoint_path.stat().st_size
    print("\nCheckpoint file size (includes optimizer/training state):")
    print(format_size(disk_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())