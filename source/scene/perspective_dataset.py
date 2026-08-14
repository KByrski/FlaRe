from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch

from scene.rays import RayBundle, generate_indexed_perspective_rays


class PerspectiveCamera(Protocol):
    image_name: str
    image_width: int
    image_height: int
    R: object
    T: object
    FoVx: float
    FoVy: float
    foreground_image: torch.Tensor
    gt_alpha_mask: torch.Tensor


class PerspectiveDatasetAdapter:
    """Tensor view of the existing ordered perspective ``Camera`` objects."""

    def __init__(
        self,
        cameras: Sequence[PerspectiveCamera],
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not cameras:
            raise ValueError("a perspective dataset needs at least one camera")

        self.camera_names = tuple(camera.image_name for camera in cameras)
        self.width = int(cameras[0].image_width)
        self.height = int(cameras[0].image_height)
        self.camera_count = len(cameras)
        self.rays_per_camera = self.width * self.height
        self.ray_count = self.camera_count * self.rays_per_camera
        if device is None:
            device = cameras[0].foreground_image.device
        self.device = torch.device(device)
        self.dtype = dtype

        for camera in cameras:
            if (camera.image_width, camera.image_height) != (self.width, self.height):
                raise ValueError("all perspective cameras must have the same resolution")

        self.fov_x = torch.tensor(
            [camera.FoVx for camera in cameras],
            dtype=dtype,
            device=self.device,
        )
        self.fov_y = torch.tensor(
            [camera.FoVy for camera in cameras],
            dtype=dtype,
            device=self.device,
        )
        self.origins = torch.stack(
            [
                torch.tensor(
                    -camera.R @ camera.T,
                    dtype=dtype,
                    device=self.device,
                )
                for camera in cameras
            ]
        )
        camera_bases = torch.stack(
            [
                torch.tensor(
                    camera.R.transpose(1, 0),
                    dtype=dtype,
                    device=self.device,
                )
                for camera in cameras
            ]
        )
        self.rights = camera_bases[:, 0, :].contiguous()
        self.downs = camera_bases[:, 1, :].contiguous()
        self.forwards = camera_bases[:, 2, :].contiguous()
        self.foreground = torch.cat(
            [
                camera.foreground_image.to(device=self.device, dtype=dtype)
                .reshape(3, self.rays_per_camera)
                .transpose(0, 1)
                for camera in cameras
            ],
            dim=0,
        )
        self.alpha = torch.cat(
            [
                camera.gt_alpha_mask.to(device=self.device, dtype=dtype).reshape(
                    self.rays_per_camera, 1
                )
                for camera in cameras
            ],
            dim=0,
        )

    def rays(
        self,
        flat_indices: torch.Tensor,
        *,
        fov_x: float,
        fov_y: float,
    ) -> RayBundle:
        return generate_indexed_perspective_rays(
            flat_indices,
            self.origins,
            self.rights,
            self.downs,
            self.forwards,
            self.width,
            self.height,
            fov_x,
            fov_y,
        )
