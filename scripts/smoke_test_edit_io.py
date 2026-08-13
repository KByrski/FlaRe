#!/usr/bin/env python3
"""Smoke-test FlaRe's checkpoint-to-PLY editing interchange on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from flare_edit_io import (
    deform_vertices,
    primitives_to_vertices,
    read_edit_ply,
    vertices_to_primitives,
    write_edit_ply,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.manual_seed(7)
    count = 64
    centers = torch.randn(count, 3)
    # Keep singular values separated so the SVD tangent axes are well-defined.
    log_scales = torch.stack(
        (
            torch.empty(count).uniform_(-2.0, -1.0),
            torch.empty(count).uniform_(-3.5, -2.5),
        ),
        1,
    )
    quaternions = torch.randn(count, 4)
    quaternions /= torch.linalg.vector_norm(quaternions, dim=1, keepdim=True)
    opacity_logits = torch.randn(count, 1)
    kappa_logits = torch.randn(count, 1)

    vertices = primitives_to_vertices(
        centers, log_scales, quaternions, opacity_logits, kappa_logits
    )
    with TemporaryDirectory(prefix="flare-edit-smoke-") as directory:
        ply_path = Path(directory) / "proxy.ply"
        write_edit_ply(ply_path, vertices)
        loaded = read_edit_ply(ply_path, count)
    torch.testing.assert_close(loaded, vertices, rtol=0.0, atol=0.0)

    fitted = vertices_to_primitives(
        loaded,
        opacity_logits,
        kappa_logits,
        fallback_s=log_scales,
        fallback_q=quaternions,
    )
    reconstructed = primitives_to_vertices(
        fitted[0], fitted[1], fitted[2], opacity_logits, kappa_logits
    )
    max_roundtrip_error = float((reconstructed - vertices).abs().max())
    if max_roundtrip_error > 5e-6:
        raise AssertionError(
            f"Proxy round-trip error {max_roundtrip_error} exceeds 5e-6"
        )

    amplitude, frequency = 0.08, 8.0
    deformed = deform_vertices(loaded, "sin", amplitude=amplitude, frequency=frequency)
    expected_sin = loaded.clone()
    expected_sin[:, :, 2] += amplitude * torch.sin(frequency * loaded[:, :, 0])
    torch.testing.assert_close(deformed, expected_sin)

    deformed_sin2 = deform_vertices(loaded, "sin2", amplitude=amplitude, frequency=frequency)
    expected_sin2 = expected_sin.clone()
    expected_sin2[:, :, 2] += amplitude * torch.sin(frequency * loaded[:, :, 1])
    torch.testing.assert_close(deformed_sin2, expected_sin2)

    phase_test = torch.zeros((1, 8, 3))
    phase_result = deform_vertices(
        phase_test, "sin", amplitude=amplitude, frequency=frequency, phase_shift_degrees=90.0
    )
    torch.testing.assert_close(phase_result[:, :, 2], torch.full((1, 8), amplitude))

    rotation_test = loaded[:1]
    rotation_result = deform_vertices(
        rotation_test, "sin", amplitude=0.0, frequency=frequency, rotation_z_degrees=90.0
    )
    torch.testing.assert_close(rotation_result[:, :, 0], -rotation_test[:, :, 1], atol=1e-6, rtol=0.0)
    torch.testing.assert_close(rotation_result[:, :, 1], rotation_test[:, :, 0], atol=1e-6, rtol=0.0)

    result = {
        "status": "passed",
        "primitives": count,
        "vertices": int(vertices.numel() // 3),
        "max_roundtrip_error": max_roundtrip_error,
        "proxy_scale_convention": "legacy_sqrt_delta_s",
        "texture_correction_A": "todo",
    }
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
