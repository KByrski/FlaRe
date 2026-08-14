from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import torch

from scene.rays import generate_indexed_perspective_rays


PROJECT_DIR = Path(__file__).resolve().parents[1]
RENDERER_DIR = PROJECT_DIR / "source" / "renderer" / "output"
RENDERER_SO = RENDERER_DIR / "PYOPTIXFLARERENDERER.so"


def reference_camera_directions(
    right: torch.Tensor,
    down: torch.Tensor,
    forward: torch.Tensor,
    width: int,
    height: int,
    fov_x: float,
    fov_y: float,
) -> torch.Tensor:
    """Independent pixel-centred pinhole reference in legacy row-major order."""
    rows = []
    tan_x = math.tan(0.5 * fov_x)
    tan_y = math.tan(0.5 * fov_y)
    for y in range(height):
        columns = []
        for x in range(width):
            d_x = (2.0 * (x + 0.5) / width - 1.0) * tan_x
            d_y = (2.0 * (y + 0.5) / height - 1.0) * tan_y
            columns.append(right * d_x + down * d_y + forward)
        rows.append(torch.stack(columns))
    return torch.stack(rows)


class IndexedPerspectiveRayRegressionTest(unittest.TestCase):
    def test_indexed_rays_match_pixel_centred_reference_for_two_poses(self) -> None:
        origins = torch.tensor([[0.0, 0.0, 0.0], [2.0, 3.0, 4.0]])
        rights = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        downs = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        forwards = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        width, height = 4, 3
        fov_x, fov_y = 0.9, 0.7
        indices = torch.tensor([0, 3, 5, 11, 12, 16, 20, 23])

        actual = generate_indexed_perspective_rays(
            indices,
            origins,
            rights,
            downs,
            forwards,
            width,
            height,
            fov_x,
            fov_y,
        )

        full_directions = torch.cat(
            [
                reference_camera_directions(
                    rights[pose],
                    downs[pose],
                    forwards[pose],
                    width,
                    height,
                    fov_x,
                    fov_y,
                ).reshape(-1, 3)
                for pose in range(2)
            ]
        )
        expected_origins = origins[indices // (width * height)]
        torch.testing.assert_close(actual.origins, expected_origins, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            actual.directions,
            full_directions[indices],
            rtol=1.0e-6,
            atol=1.0e-7,
        )

    def test_odd_centre_pixel_points_exactly_forward(self) -> None:
        origin = torch.tensor([[1.0, 2.0, 3.0]])
        right = torch.tensor([[1.0, 0.0, 0.0]])
        down = torch.tensor([[0.0, 1.0, 0.0]])
        forward = torch.tensor([[0.0, 0.0, 1.0]])

        rays = generate_indexed_perspective_rays(
            torch.tensor([4]),
            origin,
            right,
            down,
            forward,
            3,
            3,
            math.pi / 2,
            math.pi / 2,
        )

        torch.testing.assert_close(rays.origins[0], origin[0], rtol=0.0, atol=0.0)
        torch.testing.assert_close(rays.directions[0], forward[0], rtol=0.0, atol=0.0)


@unittest.skipUnless(
    torch.cuda.is_available() and RENDERER_SO.is_file(),
    "native ray regression requires CUDA and a built FlaRe renderer",
)
class NativePerspectiveRayRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if str(RENDERER_DIR) not in sys.path:
            sys.path.insert(0, str(RENDERER_DIR))
        import PYOPTIXFLARERENDERER

        cls.native = PYOPTIXFLARERENDERER

    def test_native_generator_matches_indexed_training_generator(self) -> None:
        device = torch.device("cuda:0")
        origin = torch.tensor([[1.0, 2.0, 3.0]], device=device)
        right = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        down = torch.tensor([[0.0, 1.0, 0.0]], device=device)
        forward = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        width, height = 8, 5
        fov_x, fov_y = 1.1, 0.8

        indices = torch.arange(width * height, device=device)
        expected = generate_indexed_perspective_rays(
            indices,
            origin,
            right,
            down,
            forward,
            width,
            height,
            fov_x,
            fov_y,
        ).directions
        actual = self.native.GenerateRays(
            right[0],
            down[0],
            forward[0],
            width,
            height,
            fov_x,
            fov_y,
        ).reshape(-1, 3)

        torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-7)

    def test_native_generator_matches_cpu_reference_before_normalization(self) -> None:
        device = torch.device("cuda:0")
        right = torch.tensor([0.8, 0.1, -0.2], device=device)
        down = torch.tensor([-0.1, 0.9, 0.15], device=device)
        forward = torch.tensor([0.2, -0.05, 1.1], device=device)
        width, height = 4, 3
        fov_x, fov_y = 0.7, 1.0

        actual = self.native.GenerateRays(
            right, down, forward, width, height, fov_x, fov_y
        )
        expected = reference_camera_directions(
            right.cpu(),
            down.cpu(),
            forward.cpu(),
            width,
            height,
            fov_x,
            fov_y,
        ).to(device)

        torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-7)


if __name__ == "__main__":
    unittest.main()
