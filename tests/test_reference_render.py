from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

import tests.test_native_renderer as native_regression


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from utils.image_utils import psnr


@unittest.skipUnless(
    torch.cuda.is_available() and native_regression.RENDERER_SO.is_file(),
    "reference render regression requires CUDA and a built FlaRe renderer",
)
class ReferenceRenderRegressionTest(unittest.TestCase):
    """Freeze tiny inference renders and the evaluation PSNR convention."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.native_test = native_regression.NativeFlaReRendererRegressionTest
        cls.native_test.setUpClass()
        cls.fixture = cls.native_test(methodName="runTest")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.native_test.tearDownClass()

    def setUp(self) -> None:
        self.fixture.setUp()

    def _render(self) -> tuple[torch.Tensor, torch.Tensor]:
        fixture = self.fixture
        base = torch.zeros((fixture.ray_count, 3), device=fixture.device)
        fixture.renderer.forward_inference_base(
            fixture.origins,
            fixture.directions,
            base,
            *fixture.background,
            fixture.means,
            fixture.log_scales,
            fixture.quaternions,
            fixture.rgba,
            fixture.raw_kappa,
            fixture.threshold,
        )
        full = torch.zeros_like(base)
        fixture.renderer.forward_inference(
            fixture.conditioning,
            fixture.features,
            fixture.w1,
            fixture.b1,
            fixture.w2,
            fixture.b2,
            fixture.w3,
            fixture.b3,
            fixture.origins,
            fixture.directions,
            full,
            *fixture.background,
            fixture.means,
            fixture.log_scales,
            fixture.quaternions,
            fixture.rgba,
            fixture.raw_kappa,
            fixture.threshold,
        )
        torch.cuda.synchronize()
        return base, full

    def _chw(self, image: torch.Tensor) -> torch.Tensor:
        fixture = self.fixture
        return (
            image.clamp(0.0, 1.0)
            .reshape(fixture.height, fixture.width, 3)
            .permute(2, 0, 1)
            .contiguous()
        )

    def test_base_and_full_reference_render_psnr(self) -> None:
        base, full = self._render()
        target = self._chw(self.fixture.target)
        base_psnr = psnr(self._chw(base).unsqueeze(0), target.unsqueeze(0))
        full_psnr = psnr(self._chw(full).unsqueeze(0), target.unsqueeze(0))

        torch.testing.assert_close(
            base.double().sum().cpu(),
            torch.tensor(26.130601217970252, dtype=torch.float64),
            rtol=2.0e-5,
            atol=2.0e-7,
        )
        torch.testing.assert_close(
            full.double().sum().cpu(),
            torch.tensor(4.684535137377679, dtype=torch.float64),
            rtol=2.0e-5,
            atol=2.0e-7,
        )
        torch.testing.assert_close(
            base_psnr.cpu(),
            torch.tensor([[6.28584098815918]]),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            full_psnr.cpu(),
            torch.tensor([[5.30635929107666]]),
            rtol=1.0e-6,
            atol=1.0e-6,
        )

    def test_inference_rerender_is_bitwise_repeatable(self) -> None:
        first_base, first_full = self._render()
        second_base, second_full = self._render()

        torch.testing.assert_close(
            second_base, first_base, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            second_full, first_full, rtol=0.0, atol=0.0
        )


if __name__ == "__main__":
    unittest.main()
