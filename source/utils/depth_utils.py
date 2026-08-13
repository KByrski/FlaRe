"""Depth conversion, 2DGS-style normal reconstruction, and diagnostic export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def ray_parameters_to_depth_and_normal(
    ray_t: torch.Tensor,
    valid: torch.Tensor,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    camera_forward: torch.Tensor,
    height: int,
    width: int,
    relative_depth_edge_threshold: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert per-pixel ray parameters into Z depth and world normals.

    Rays are normalized by the caller, so ``t`` is Euclidean distance along a
    ray. The camera-space Z depth used by pinhole unprojection and TSDF fusion
    is ``t * dot(ray_direction, camera_forward)``.

    Normals follow the official 2DGS ``depth_to_normal`` implementation: form
    the world-space point image, take centered vertical and horizontal point
    differences, and cross them. Invalid/background pixels are zeroed.
    """
    if ray_t.numel() != height * width or valid.numel() != height * width:
        raise ValueError(
            f"Expected {height * width} ray parameters and validity values, "
            f"got {ray_t.numel()} and {valid.numel()}"
        )

    ray_origins = ray_origins.reshape(height, width, 3)
    ray_directions = ray_directions.reshape(height, width, 3)
    ray_t = ray_t.reshape(height, width)
    valid = valid.reshape(height, width).to(torch.bool)

    ray_z_scale = torch.sum(
        ray_directions * camera_forward.reshape(1, 1, 3), dim=-1
    )
    valid = valid & torch.isfinite(ray_t) & (ray_t > 0.0) & (ray_z_scale > 0.0)

    safe_t = torch.where(valid, ray_t, torch.zeros_like(ray_t))
    depth_z = torch.where(valid, safe_t * ray_z_scale, torch.zeros_like(ray_t))
    points = ray_origins + safe_t.unsqueeze(-1) * ray_directions

    normals = torch.zeros_like(points)
    if height >= 3 and width >= 3:
        vertical = points[2:, 1:-1] - points[:-2, 1:-1]
        horizontal = points[1:-1, 2:] - points[1:-1, :-2]
        interior_normals = F.normalize(
            torch.cross(vertical, horizontal, dim=-1), dim=-1, eps=1.0e-8
        )

        interior_valid = (
            valid[1:-1, 1:-1]
            & valid[2:, 1:-1]
            & valid[:-2, 1:-1]
            & valid[1:-1, 2:]
            & valid[1:-1, :-2]
        )
        if relative_depth_edge_threshold is not None:
            center_depth = depth_z[1:-1, 1:-1]
            depth_scale = torch.clamp(center_depth.abs(), min=1.0e-6)
            max_relative_jump = torch.maximum(
                torch.maximum(
                    (depth_z[2:, 1:-1] - center_depth).abs(),
                    (depth_z[:-2, 1:-1] - center_depth).abs(),
                ),
                torch.maximum(
                    (depth_z[1:-1, 2:] - center_depth).abs(),
                    (depth_z[1:-1, :-2] - center_depth).abs(),
                ),
            ) / depth_scale
            interior_valid &= max_relative_jump <= relative_depth_edge_threshold
        normals[1:-1, 1:-1] = torch.where(
            interior_valid.unsqueeze(-1),
            interior_normals,
            torch.zeros_like(interior_normals),
        )

    return depth_z, normals, valid


def median_hits_to_depth_and_normal(
    depth_and_index: torch.Tensor,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    camera_forward: torch.Tensor,
    height: int,
    width: int,
    relative_depth_edge_threshold: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert hard median hits into Z depth and 2DGS-style world normals."""
    if depth_and_index.shape != (height * width, 2):
        raise ValueError(
            f"Expected depth/index shape {(height * width, 2)}, "
            f"got {tuple(depth_and_index.shape)}"
        )

    hit_t = depth_and_index[:, 0]
    # CUDA stores the integer index in float2 via __int_as_float; reinterpret
    # the bits instead of numerically converting the float value.
    hit_index = depth_and_index[:, 1].contiguous().view(torch.int32)
    depth_z, normals, valid = ray_parameters_to_depth_and_normal(
        hit_t,
        torch.isfinite(hit_t) & (hit_index >= 0),
        ray_origins,
        ray_directions,
        camera_forward,
        height,
        width,
        relative_depth_edge_threshold,
    )
    return depth_z, normals, valid, hit_index.reshape(height, width)


def _to_u8_rgb(image: torch.Tensor) -> np.ndarray:
    return (
        image.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


def _depth_visualization(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.zeros((*depth.shape, 3), dtype=np.uint8)
    values = depth[valid]
    if values.size == 0:
        return output

    near, far = np.percentile(values, [2.0, 98.0])
    if not np.isfinite(near) or not np.isfinite(far) or far <= near:
        near = float(np.min(values))
        far = float(np.max(values))
    scale = max(far - near, np.finfo(np.float32).eps)
    normalized = np.clip((depth - near) / scale, 0.0, 1.0)

    # A dependency-free blue/cyan/yellow/red depth visualization. Near is red.
    x = 1.0 - normalized
    output[..., 0] = (np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0) * 255).astype(np.uint8)
    output[..., 1] = (np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0) * 255).astype(np.uint8)
    output[..., 2] = (np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0) * 255).astype(np.uint8)
    output[~valid] = 0
    return output


def save_render_bundle(
    output_dir: str | Path,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    rendered_normals: torch.Tensor,
    depth_normals: torch.Tensor,
    alpha: torch.Tensor,
    valid: torch.Tensor,
    median_depth: torch.Tensor,
    median_valid: torch.Tensor,
    hit_index: torch.Tensor,
    metadata: dict,
) -> None:
    """Save RGB/depth plus both normal products used by official 2DGS.

    ``rendered_normals`` is ``rend_normal``: camera-facing Gaussian normals
    alpha-composited with the RGB weights. ``depth_normals`` is
    ``surf_normal``: finite differences of the opacity-weighted expected-depth
    point image, which is the official 2DGS default (``depth_ratio=0``).
    Hard median depth is retained under ``median_depth.*`` for diagnostics.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_np = _to_u8_rgb(rgb)
    depth_np = depth.detach().to(torch.float32).cpu().numpy()
    rend_normal_np = rendered_normals.detach().to(torch.float32).cpu().numpy()
    depth_normals_np = depth_normals.detach().to(torch.float32).cpu().numpy()
    alpha_np = alpha.detach().to(torch.float32).cpu().numpy()
    valid_np = valid.detach().cpu().numpy().astype(bool)
    median_depth_np = median_depth.detach().to(torch.float32).cpu().numpy()
    median_valid_np = median_valid.detach().cpu().numpy().astype(bool)
    hit_index_np = hit_index.detach().to(torch.int32).cpu().numpy()

    Image.fromarray(rgb_np, mode="RGB").save(output_dir / "rgb.png")
    Image.fromarray(depth_np, mode="F").save(output_dir / "depth.tiff")
    Image.fromarray(_depth_visualization(depth_np, valid_np), mode="RGB").save(
        output_dir / "depth.png"
    )
    Image.fromarray(median_depth_np, mode="F").save(output_dir / "median_depth.tiff")
    Image.fromarray(
        _depth_visualization(median_depth_np, median_valid_np), mode="RGB"
    ).save(output_dir / "median_depth.png")

    # `normal.png` is the normal image normally shown by 2DGS: normalize the
    # alpha-weighted sum to retain direction, and suppress transparent pixels.
    rend_normal_length = np.linalg.norm(rend_normal_np, axis=-1, keepdims=True)
    rendered_normals_np = np.divide(
        rend_normal_np,
        np.maximum(rend_normal_length, np.finfo(np.float32).eps),
    )
    rendered_normal_valid_np = (alpha_np > 0.01) & (rend_normal_length[..., 0] > 0.0)
    normal_visualization = np.zeros_like(rgb_np)
    normal_visualization[rendered_normal_valid_np] = np.clip(
        (rendered_normals_np[rendered_normal_valid_np] * 0.5 + 0.5) * 255.0,
        0.0,
        255.0,
    ).astype(np.uint8)
    Image.fromarray(normal_visualization, mode="RGB").save(output_dir / "normal.png")

    # Exact, unnormalized 2DGS rasterizer output. A zero vector maps to neutral
    # gray just as in the official `(rend_normal + 1) / 2` visualization.
    Image.fromarray(
        np.clip((rend_normal_np * 0.5 + 0.5) * 255.0, 0.0, 255.0).astype(np.uint8),
        mode="RGB",
    ).save(output_dir / "rend_normal.png")

    depth_normal_valid_np = valid_np & (
        np.linalg.norm(depth_normals_np, axis=-1) > 0.0
    )
    depth_normal_visualization = np.zeros_like(rgb_np)
    depth_normal_visualization[depth_normal_valid_np] = np.clip(
        (depth_normals_np[depth_normal_valid_np] * 0.5 + 0.5) * 255.0,
        0.0,
        255.0,
    ).astype(np.uint8)
    Image.fromarray(depth_normal_visualization, mode="RGB").save(
        output_dir / "depth_normal.png"
    )
    Image.fromarray(
        np.clip(alpha_np * 255.0, 0.0, 255.0).astype(np.uint8), mode="L"
    ).save(output_dir / "alpha.png")
    Image.fromarray((valid_np.astype(np.uint8) * 255), mode="L").save(
        output_dir / "valid_mask.png"
    )

    np.save(output_dir / "depth.npy", depth_np)
    np.save(output_dir / "median_depth.npy", median_depth_np)
    np.save(output_dir / "normal.npy", rendered_normals_np)
    np.save(output_dir / "rend_normal.npy", rend_normal_np)
    np.save(output_dir / "depth_normal.npy", depth_normals_np)
    np.save(output_dir / "alpha.npy", alpha_np)
    np.save(output_dir / "median_gaussian_index.npy", hit_index_np)

    valid_depth = depth_np[valid_np]
    valid_median_depth = median_depth_np[median_valid_np]
    metadata = dict(metadata)
    metadata.update(
        {
            "valid_pixel_count": int(valid_np.sum()),
            "depth_min": float(valid_depth.min()) if valid_depth.size else None,
            "depth_max": float(valid_depth.max()) if valid_depth.size else None,
            "depth_semantics": "opacity-weighted expected camera-space Z depth (2DGS depth_ratio=0); background is 0",
            "median_depth_min": float(valid_median_depth.min()) if valid_median_depth.size else None,
            "median_depth_max": float(valid_median_depth.max()) if valid_median_depth.size else None,
            "median_depth_semantics": "hard transmittance-median camera-space Z depth (2DGS depth_ratio=1); background is 0",
            "normal_space": "world",
            "normal_semantics": "unit-normalized 2DGS rend_normal, alpha > 0.01",
            "rend_normal_semantics": "raw opacity-weighted Gaussian normals (exact 2DGS rasterizer output)",
            "depth_normal_semantics": "finite differences of expected-depth world points (2DGS surf_normal, depth_ratio=0)",
        }
    )
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
