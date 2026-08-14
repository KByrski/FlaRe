from __future__ import annotations

from dataclasses import dataclass
import math
import unittest

import numpy as np
import torch

from scene.perspective_dataset import PerspectiveDatasetAdapter
from scene.rays import RayBundle, generate_indexed_perspective_rays


@dataclass
class _CameraStub:
    image_name: str
    R: np.ndarray
    T: np.ndarray
    FoVx: float
    FoVy: float
    foreground_image: torch.Tensor
    gt_alpha_mask: torch.Tensor

    @property
    def image_width(self) -> int:
        return int(self.foreground_image.shape[2])

    @property
    def image_height(self) -> int:
        return int(self.foreground_image.shape[1])


def _camera(
    name: str,
    rotation: np.ndarray,
    translation: np.ndarray,
    foreground: torch.Tensor,
    alpha: torch.Tensor,
    fov_x: float,
    fov_y: float,
) -> _CameraStub:
    return _CameraStub(
        image_name=name,
        R=np.asarray(rotation, dtype=np.float64),
        T=np.asarray(translation, dtype=np.float64),
        FoVx=fov_x,
        FoVy=fov_y,
        foreground_image=foreground,
        gt_alpha_mask=alpha,
    )


def _legacy_generate_rays(
    indices: torch.Tensor,
    origins: torch.Tensor,
    rights: torch.Tensor,
    downs: torch.Tensor,
    forwards: torch.Tensor,
    width: int,
    height: int,
    fov_x: float,
    fov_y: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    double_tan_half_fov_x = 2.0 * np.tan(0.5 * fov_x)
    double_tan_half_fov_y = 2.0 * np.tan(0.5 * fov_y)
    indices = indices.to(torch.int64).unsqueeze(1)
    camera_indices = indices // (width * height)
    pixel_indices = indices % (width * height)
    y = pixel_indices // width
    x = pixel_indices % width
    d_x = (-0.5 + ((x + 0.5) / width)) * double_tan_half_fov_x
    d_y = (-0.5 + ((y + 0.5) / height)) * double_tan_half_fov_y
    gathered_rights = torch.gather(rights, 0, camera_indices.expand(-1, 3))
    gathered_downs = torch.gather(downs, 0, camera_indices.expand(-1, 3))
    gathered_forwards = torch.gather(forwards, 0, camera_indices.expand(-1, 3))
    gathered_origins = torch.gather(origins, 0, camera_indices.expand(-1, 3))
    directions = gathered_rights * d_x + gathered_downs * d_y + gathered_forwards
    return gathered_origins, directions


class PerspectiveDatasetAdapterContractTest(unittest.TestCase):
    def setUp(self) -> None:
        foreground_a = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3)
        foreground_b = foreground_a + 100.0
        alpha_a = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3) / 10.0
        alpha_b = alpha_a + 0.25
        self.cameras = [
            _camera(
                "second_in_source",
                np.eye(3),
                np.asarray([1.0, 2.0, 3.0]),
                foreground_a,
                alpha_a,
                0.8,
                0.6,
            ),
            _camera(
                "first_in_source",
                np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
                np.asarray([-2.0, 1.0, 4.0]),
                foreground_b,
                alpha_b,
                1.1,
                0.9,
            ),
        ]

    def test_stacked_outputs_match_legacy_training_layout(self) -> None:
        dataset = PerspectiveDatasetAdapter(self.cameras, device="cpu")

        self.assertEqual(dataset.camera_names, ("second_in_source", "first_in_source"))
        self.assertEqual((dataset.width, dataset.height), (3, 2))
        self.assertEqual(dataset.camera_count, 2)
        self.assertEqual(dataset.rays_per_camera, 6)
        self.assertEqual(dataset.ray_count, 12)

        expected_origins = torch.stack(
            [
                torch.tensor(-camera.R @ camera.T, dtype=torch.float32)
                for camera in self.cameras
            ]
        )
        expected_basis = torch.stack(
            [
                torch.tensor(camera.R.transpose(1, 0), dtype=torch.float32)
                for camera in self.cameras
            ]
        )
        expected_foreground = torch.cat(
            [
                camera.foreground_image.reshape(3, 6).transpose(0, 1)
                for camera in self.cameras
            ]
        )
        expected_alpha = torch.cat(
            [camera.gt_alpha_mask.reshape(6, 1) for camera in self.cameras]
        )

        torch.testing.assert_close(dataset.origins, expected_origins)
        torch.testing.assert_close(dataset.rights, expected_basis[:, 0])
        torch.testing.assert_close(dataset.downs, expected_basis[:, 1])
        torch.testing.assert_close(dataset.forwards, expected_basis[:, 2])
        torch.testing.assert_close(dataset.foreground, expected_foreground)
        torch.testing.assert_close(dataset.alpha, expected_alpha)
        torch.testing.assert_close(dataset.fov_x, torch.tensor([0.8, 1.1]))
        torch.testing.assert_close(dataset.fov_y, torch.tensor([0.6, 0.9]))

        for tensor in (
            dataset.origins,
            dataset.rights,
            dataset.downs,
            dataset.forwards,
            dataset.foreground,
            dataset.alpha,
            dataset.fov_x,
            dataset.fov_y,
        ):
            self.assertEqual(tensor.device, torch.device("cpu"))
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertTrue(tensor.is_contiguous())

    def test_rejects_resolution_mismatch_without_reordering(self) -> None:
        mismatched = list(self.cameras)
        mismatched[1] = _camera(
            "first_in_source",
            mismatched[1].R,
            mismatched[1].T,
            torch.zeros((3, 1, 3)),
            torch.ones((1, 1, 3)),
            mismatched[1].FoVx,
            mismatched[1].FoVy,
        )

        with self.assertRaisesRegex(ValueError, "same resolution"):
            PerspectiveDatasetAdapter(mismatched, device="cpu")

    def test_ray_bundle_uses_dataset_tensors_and_flat_indices(self) -> None:
        dataset = PerspectiveDatasetAdapter(self.cameras, device="cpu")
        indices = torch.tensor([0, 2, 5, 6, 9, 11], dtype=torch.int32)

        rays = dataset.rays(indices, fov_x=0.8, fov_y=0.6)

        self.assertIsInstance(rays, RayBundle)
        self.assertEqual(rays.origins.shape, (6, 3))
        self.assertEqual(rays.directions.shape, (6, 3))
        self.assertEqual(rays.flat_indices.shape, (6,))
        self.assertEqual(rays.camera_indices.shape, (6,))
        self.assertEqual(rays.pixel_indices.shape, (6,))
        self.assertEqual(rays.origins.dtype, torch.float32)
        self.assertEqual(rays.directions.dtype, torch.float32)
        self.assertEqual(rays.flat_indices.dtype, torch.int64)
        self.assertEqual(rays.flat_indices.device, dataset.origins.device)
        torch.testing.assert_close(rays.flat_indices, indices.to(torch.int64))
        torch.testing.assert_close(rays.camera_indices, torch.tensor([0, 0, 0, 1, 1, 1]))
        torch.testing.assert_close(rays.pixel_indices, torch.tensor([0, 2, 5, 0, 3, 5]))
        torch.testing.assert_close(rays.origins[:3], dataset.origins[0].expand(3, -1))
        torch.testing.assert_close(rays.origins[3:], dataset.origins[1].expand(3, -1))


class IndexedPerspectiveRayContractTest(unittest.TestCase):
    def test_outputs_are_bitwise_equal_to_legacy_indexed_generator(self) -> None:
        generator = torch.Generator().manual_seed(147)
        origins = torch.randn((3, 3), generator=generator)
        rights = torch.randn((3, 3), generator=generator)
        downs = torch.randn((3, 3), generator=generator)
        forwards = torch.randn((3, 3), generator=generator)
        indices = torch.tensor([17, 0, 23, 11, 35, 6, 29])

        expected_origins, expected_directions = _legacy_generate_rays(
            indices, origins, rights, downs, forwards, 4, 3, 1.07, 0.63
        )
        actual = generate_indexed_perspective_rays(
            indices, origins, rights, downs, forwards, 4, 3, 1.07, 0.63
        )

        torch.testing.assert_close(actual.origins, expected_origins, rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual.directions, expected_directions, rtol=0.0, atol=0.0)

    def test_shape_order_and_pixel_centres_match_legacy_formula(self) -> None:
        origins = torch.tensor([[0.0, 0.0, 0.0], [2.0, 3.0, 4.0]])
        rights = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        downs = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        forwards = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        width, height = 4, 3
        indices = torch.tensor([0, 3, 5, 11, 12, 16, 20, 23])

        rays = generate_indexed_perspective_rays(
            indices,
            origins,
            rights,
            downs,
            forwards,
            width,
            height,
            fov_x=0.9,
            fov_y=0.7,
        )

        expected = []
        tan_x = math.tan(0.45)
        tan_y = math.tan(0.35)
        for flat_index in indices.tolist():
            camera_index = flat_index // (width * height)
            pixel_index = flat_index % (width * height)
            y, x = divmod(pixel_index, width)
            dx = (2.0 * (x + 0.5) / width - 1.0) * tan_x
            dy = (2.0 * (y + 0.5) / height - 1.0) * tan_y
            expected.append(
                rights[camera_index] * dx
                + downs[camera_index] * dy
                + forwards[camera_index]
            )

        torch.testing.assert_close(rays.directions, torch.stack(expected), rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(rays.flat_indices, indices)

    def test_odd_centre_pixel_is_exactly_forward(self) -> None:
        rays = generate_indexed_perspective_rays(
            torch.tensor([4]),
            torch.tensor([[1.0, 2.0, 3.0]]),
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 1.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 1.0]]),
            3,
            3,
            fov_x=math.pi / 2,
            fov_y=math.pi / 2,
        )

        torch.testing.assert_close(rays.origins[0], torch.tensor([1.0, 2.0, 3.0]))
        torch.testing.assert_close(rays.directions[0], torch.tensor([0.0, 0.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
