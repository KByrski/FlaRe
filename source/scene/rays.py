from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RayBundle:
    """Perspective rays and their positions in the flattened camera dataset."""

    origins: torch.Tensor
    directions: torch.Tensor
    flat_indices: torch.Tensor
    camera_indices: torch.Tensor
    pixel_indices: torch.Tensor

    def __post_init__(self) -> None:
        if self.origins.ndim != 2 or self.origins.shape[1] != 3:
            raise ValueError("ray origins must have shape (ray_count, 3)")
        if self.directions.shape != self.origins.shape:
            raise ValueError("ray directions must match ray origins")

        ray_count = self.origins.shape[0]
        for name, indices in (
            ("flat_indices", self.flat_indices),
            ("camera_indices", self.camera_indices),
            ("pixel_indices", self.pixel_indices),
        ):
            if indices.shape != (ray_count,):
                raise ValueError(f"{name} must have shape (ray_count,)")
            if indices.dtype != torch.int64:
                raise ValueError(f"{name} must use torch.int64")
            if indices.device != self.origins.device:
                raise ValueError(f"{name} must be on the ray device")

        if self.directions.device != self.origins.device:
            raise ValueError("ray origins and directions must share a device")
        if self.directions.dtype != self.origins.dtype:
            raise ValueError("ray origins and directions must share a dtype")


def generate_indexed_perspective_rays(
    flat_indices: torch.Tensor,
    origins: torch.Tensor,
    rights: torch.Tensor,
    downs: torch.Tensor,
    forwards: torch.Tensor,
    width: int,
    height: int,
    fov_x: float,
    fov_y: float,
) -> RayBundle:
    """Reproduce FlaRe's indexed, pixel-centred pinhole ray construction."""

    indices = flat_indices.to(device=origins.device, dtype=torch.int64).reshape(-1)
    pixels_per_camera = width * height
    camera_indices = torch.div(indices, pixels_per_camera, rounding_mode="floor")
    pixel_indices = torch.remainder(indices, pixels_per_camera)
    y = torch.div(pixel_indices, width, rounding_mode="floor")
    x = torch.remainder(pixel_indices, width)

    double_tan_half_fov_x = 2.0 * np.tan(0.5 * fov_x)
    double_tan_half_fov_y = 2.0 * np.tan(0.5 * fov_y)
    d_x = (-0.5 + ((x + 0.5) / width)) * double_tan_half_fov_x
    d_y = (-0.5 + ((y + 0.5) / height)) * double_tan_half_fov_y

    ray_origins = origins[camera_indices]
    directions = (
        rights[camera_indices] * d_x.unsqueeze(1)
        + downs[camera_indices] * d_y.unsqueeze(1)
        + forwards[camera_indices]
    )
    return RayBundle(
        origins=ray_origins,
        directions=directions,
        flat_indices=indices,
        camera_indices=camera_indices,
        pixel_indices=pixel_indices,
    )
