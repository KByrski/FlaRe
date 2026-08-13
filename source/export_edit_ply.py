#!/usr/bin/env python3
"""Export a FlaRe checkpoint as an eight-vertex-per-primitive Blender mesh."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from checkpoint_io import load_model_checkpoint
from flare_edit_io import primitives_to_vertices, write_edit_ply


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for proxy generation; cuda matches the editing code",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu only as a fallback")
    model = load_model_checkpoint(args.checkpoint, device)
    a, k, m, s, q = (
        model[name].to(dtype=torch.float32).contiguous()
        for name in ("A", "k", "m", "s", "q")
    )
    vertices = primitives_to_vertices(m, s, q, a, k)
    write_edit_ply(args.output.expanduser().resolve(), vertices)
    print(
        f"Exported {m.shape[0]} primitives ({vertices.numel() // 3} vertices) "
        f"on {device} to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
