from __future__ import annotations

import gc
import math
from pathlib import Path
import sys
import unittest

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "source"
RENDERER_DIR = SOURCE_DIR / "renderer" / "output"
RENDERER_SO = RENDERER_DIR / "PYOPTIXFLARERENDERER.so"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from renderer_backend import NativeFlaReBackend
from renderer_facade import FlaReRenderer


def _reduce_mlp_gradients(gradients: dict[str, torch.Tensor]) -> None:
    """Apply the legacy per-SM buffer layout conversion from train.py."""
    value = gradients["w3"].sum(0).reshape((8, 2, 32))
    gradients["w3"] = value.transpose(1, 2).reshape((8, 8, -1)).transpose(0, 1).flatten(1, 2)

    value = gradients["b3"].sum(0).reshape((2, 32)).transpose(0, 1)
    gradients["b3"] = value.reshape((8, 8)).sum(1)

    value = gradients["w2"].sum(0).reshape((64, 2, 32)).transpose(1, 2)
    gradients["w2"] = value.reshape((8, 8, 8, -1)).transpose(1, 2).flatten(0, 1).flatten(1, 2)

    value = gradients["b2"].sum(0).reshape((8, 2, 32)).transpose(1, 2)
    gradients["b2"] = value.reshape((8, 8, -1)).sum(2).flatten(0, 1)

    value = gradients["w1"].sum(0).reshape((128, 2, 32)).transpose(1, 2)
    gradients["w1"] = value.reshape((8, 16, 8, -1)).transpose(1, 2).flatten(0, 1).flatten(1, 2)

    value = gradients["b1"].sum(0).reshape((8, 2, 32)).transpose(1, 2)
    gradients["b1"] = value.reshape((8, 8, -1)).sum(2).flatten(0, 1)


class _FixtureGeometry:
    def __init__(self, fixture) -> None:
        self.fixture = fixture

    def renderer_geometry(self):
        fixture = self.fixture
        return (
            fixture.means,
            torch.exp(fixture.log_scales),
            fixture.quaternions,
            torch.sigmoid(fixture.opacity_logits),
            1.0 + torch.nn.functional.softplus(fixture.raw_kappa),
        )


@unittest.skipUnless(
    torch.cuda.is_available() and RENDERER_SO.is_file(),
    "native renderer regressions require CUDA and a built FlaRe renderer",
)
class NativeFlaReRendererRegressionTest(unittest.TestCase):
    """Freeze small base/full native outputs and every gradient family."""

    @classmethod
    def setUpClass(cls) -> None:
        if str(RENDERER_DIR) not in sys.path:
            sys.path.insert(0, str(RENDERER_DIR))
        import PYOPTIXFLARERENDERER

        cls.native = PYOPTIXFLARERENDERER
        cls.backend = NativeFlaReBackend(cls.native)
        cls.device = torch.device("cuda:0")
        cls.width = cls.height = 8
        cls.ray_count = cls.width * cls.height
        cls.background = (0.01, 0.02, 0.03)
        cls.threshold = 1.0e-4
        cls.reg_a = -0.01
        cls.reg_b = 1.01

        right = torch.tensor([1.0, 0.0, 0.0], device=cls.device)
        down = torch.tensor([0.0, 1.0, 0.0], device=cls.device)
        forward = torch.tensor([0.0, 0.0, 1.0], device=cls.device)
        directions = cls.backend.generate_rays(
            right,
            down,
            forward,
            cls.width,
            cls.height,
            1.0,
            1.0,
        ).reshape(cls.ray_count, 3)
        cls.directions = torch.nn.functional.normalize(directions, dim=1).contiguous()
        cls.origins = torch.zeros_like(cls.directions)

        cls.means = torch.tensor(
            [[0.0, 0.0, 3.0], [0.35, -0.2, 3.4]], device=cls.device
        )
        cls.log_scales = torch.full(
            (2, 2), math.log(0.75), device=cls.device
        )
        cls.quaternions = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.98, 0.1, 0.03, -0.04]],
            device=cls.device,
        )
        cls.quaternions = torch.nn.functional.normalize(
            cls.quaternions, dim=1
        ).contiguous()
        cls.rgb = torch.tensor(
            [[0.8, 0.2, 0.1], [0.1, 0.7, 0.3]], device=cls.device
        )
        cls.opacity_logits = torch.tensor(
            [[math.log(9.0)], [math.log(4.0)]], device=cls.device
        )
        cls.rgba = torch.cat((cls.rgb, cls.opacity_logits), dim=1).contiguous()
        cls.raw_kappa = torch.tensor(
            [[math.log(math.expm1(1.0))], [math.log(math.expm1(0.7))]],
            device=cls.device,
        )

        generator = torch.Generator(device="cpu").manual_seed(23)

        def random_tensor(shape, scale, dtype=torch.float32):
            return (
                torch.randn(shape, generator=generator)
                .mul(scale)
                .to(device=cls.device, dtype=dtype)
                .contiguous()
            )

        cls.conditioning = random_tensor((2, 96), 0.03, torch.float16)
        cls.features = random_tensor((13154,), 1.0e-3)
        cls.w1 = random_tensor((64, 128), 0.02, torch.float16)
        cls.b1 = random_tensor((64,), 0.01)
        cls.w2 = random_tensor((64, 64), 0.02, torch.float16)
        cls.b2 = random_tensor((64,), 0.01)
        cls.w3 = random_tensor((16, 64), 0.02, torch.float16)
        cls.b3 = torch.linspace(0.1, 0.4, 16, device=cls.device)
        cls.target = torch.linspace(
            0.05, 0.95, cls.ray_count * 3, device=cls.device
        ).reshape(cls.ray_count, 3)

        cls.renderer = FlaReRenderer(
            8,
            11.3449,
            cls.ray_count,
            backend=cls.backend,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.renderer
        gc.collect()
        torch.cuda.empty_cache()

    def setUp(self) -> None:
        self.renderer.sync_geometry(_FixtureGeometry(self))

    def _auxiliary_buffers(self) -> dict[str, torch.Tensor]:
        return {
            "depth": torch.zeros((self.ray_count, 4), device=self.device),
            "depth_and_index": torch.zeros((self.ray_count, 2), device=self.device),
            "surface_normal": torch.zeros((self.ray_count, 3), device=self.device),
            "normal": torch.zeros((self.ray_count, 4), device=self.device),
        }

    def _forward_base(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image = torch.zeros((self.ray_count, 3), device=self.device)
        auxiliary = self._auxiliary_buffers()
        self.renderer.forward_training_base(
            self.origins,
            self.directions,
            image,
            *self.background,
            self.means,
            self.log_scales,
            self.quaternions,
            self.rgba,
            self.raw_kappa,
            self.threshold,
            auxiliary["depth"],
            self.reg_a,
            self.reg_b,
            auxiliary["depth_and_index"],
            auxiliary["surface_normal"],
            auxiliary["normal"],
        )
        return image, auxiliary

    def _forward_full(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image = torch.zeros((self.ray_count, 3), device=self.device)
        auxiliary = self._auxiliary_buffers()
        self.renderer.forward_training(
            self.conditioning,
            self.features,
            self.w1,
            self.b1,
            self.w2,
            self.b2,
            self.w3,
            self.b3,
            self.origins,
            self.directions,
            image,
            *self.background,
            self.means,
            self.log_scales,
            self.quaternions,
            self.rgba,
            self.raw_kappa,
            self.threshold,
            auxiliary["depth"],
            self.reg_a,
            self.reg_b,
            auxiliary["depth_and_index"],
            auxiliary["surface_normal"],
            auxiliary["normal"],
        )
        return image, auxiliary

    def assert_finite_nonzero(self, name: str, value: torch.Tensor) -> None:
        self.assertTrue(bool(torch.isfinite(value).all()), f"{name} is not finite")
        self.assertGreater(float(value.abs().sum()), 0.0, f"{name} is all zero")

    def assert_fingerprint(
        self,
        value: torch.Tensor,
        expected_sum: float,
        expected_square_sum: float,
        *,
        rtol: float = 2.0e-5,
        atol: float = 2.0e-7,
    ) -> None:
        actual = torch.stack(
            (value.double().sum(), value.double().square().sum())
        ).cpu()
        expected = torch.tensor(
            [expected_sum, expected_square_sum], dtype=torch.float64
        )
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    def test_base_and_full_training_forward_match_reference(self) -> None:
        base, base_auxiliary = self._forward_base()
        full, full_auxiliary = self._forward_full()
        torch.cuda.synchronize()

        self.assert_fingerprint(base, 26.130601217970252, 9.13155410775094)
        self.assert_fingerprint(full, 4.684535137377679, 0.12815454847124969)
        self.assert_fingerprint(
            base_auxiliary["depth"], 111.40365577954799, 184.10945946028417
        )
        self.assert_fingerprint(
            full_auxiliary["depth"], 70.07738274428993, 78.85944669789001
        )
        self.assertFalse(torch.equal(base, full))

    def test_base_backward_matches_all_reference_gradient_groups(self) -> None:
        image, auxiliary = self._forward_base()
        image_gradient = (2.0 * (image - self.target) / image.numel()).contiguous()
        gradients = {
            "RGB": torch.zeros_like(self.rgb),
            "A": torch.zeros_like(self.opacity_logits),
            "k": torch.zeros_like(self.raw_kappa),
            "m": torch.zeros_like(self.means),
            "s": torch.zeros_like(self.log_scales),
            "q": torch.zeros_like(self.quaternions),
        }
        prefix = torch.zeros((self.ray_count, 4), device=self.device)
        self.renderer.backward_base(
            self.origins,
            self.directions,
            *self.background,
            self.means,
            self.log_scales,
            self.quaternions,
            self.rgba,
            self.raw_kappa,
            image.clone(),
            image_gradient,
            gradients["RGB"],
            gradients["A"],
            gradients["k"],
            gradients["m"],
            gradients["s"],
            gradients["q"],
            self.threshold,
            auxiliary["depth"],
            prefix,
            0.0,
            self.reg_a,
            self.reg_b,
            auxiliary["depth_and_index"],
            auxiliary["surface_normal"],
            auxiliary["normal"],
            0.0,
        )
        torch.cuda.synchronize()

        references = {
            "RGB": (-0.14061586977913976, 0.007708075581112924),
            "A": (-0.001194002863485366, 4.525267840855741e-06),
            "k": (0.008808156242594123, 4.6947993266511937e-05),
            "m": (-0.02870240807533264, 0.0033179135863564326),
            "s": (-0.11031163297593594, 0.0033635509696307332),
            "q": (0.04179719484529157, 0.0019488940284928687),
        }
        for name, gradient in gradients.items():
            self.assert_finite_nonzero(name, gradient)
            self.assert_fingerprint(gradient, *references[name], rtol=5.0e-5)

    def test_full_backward_covers_every_fla_re_parameter_family(self) -> None:
        image, auxiliary = self._forward_full()
        image_gradient = (2.0 * (image - self.target) / image.numel()).contiguous()
        sm_buffers = 4 * torch.cuda.get_device_properties(self.device).multi_processor_count
        gradients = {
            "RGB": torch.zeros_like(self.rgb),
            "A": torch.zeros_like(self.opacity_logits),
            "k": torch.zeros_like(self.raw_kappa),
            "w3": torch.zeros((sm_buffers, 8 * 64), device=self.device),
            "b3": torch.zeros((sm_buffers, 8 * 8), device=self.device),
            "w2": torch.zeros((sm_buffers, 64 * 64), device=self.device),
            "b2": torch.zeros((sm_buffers, 64 * 8), device=self.device),
            "w1": torch.zeros((sm_buffers, 64 * 128), device=self.device),
            "b1": torch.zeros((sm_buffers, 64 * 8), device=self.device),
            "conditioning": torch.zeros((2, 96), device=self.device),
            "features": torch.zeros_like(self.features),
            "m": torch.zeros_like(self.means),
            "s": torch.zeros_like(self.log_scales),
            "q": torch.zeros_like(self.quaternions),
        }
        prefix = torch.zeros((self.ray_count, 4), device=self.device)
        self.renderer.backward(
            self.conditioning,
            self.features,
            self.w1,
            self.b1,
            self.w2,
            self.b2,
            self.w3,
            self.b3,
            self.origins,
            self.directions,
            *self.background,
            self.means,
            self.log_scales,
            self.quaternions,
            self.rgba,
            self.raw_kappa,
            image.clone(),
            image_gradient,
            gradients["RGB"],
            gradients["A"],
            gradients["k"],
            gradients["w3"],
            gradients["b3"],
            gradients["w2"],
            gradients["b2"],
            gradients["w1"],
            gradients["b1"],
            gradients["conditioning"],
            gradients["features"],
            gradients["m"],
            gradients["s"],
            gradients["q"],
            self.threshold,
            auxiliary["depth"],
            prefix,
            0.0,
            self.reg_a,
            self.reg_b,
            auxiliary["depth_and_index"],
            auxiliary["surface_normal"],
            auxiliary["normal"],
            0.0,
        )
        torch.cuda.synchronize()
        _reduce_mlp_gradients(gradients)

        expected_sums = {
            "RGB": -0.022374914959073067,
            "A": -0.00044601093395613134,
            "k": 0.0002537280524848029,
            "w3": 0.0020745425946202545,
            "b3": -0.06939739934989575,
            "w2": -0.0002618874549993322,
            "b2": 0.002927115781008515,
            "w1": 0.002002578828528539,
            "b1": 0.0006037098631460625,
            "conditioning": -0.00011145355192532236,
            "features": 9.879187111505277e-06,
            "m": -8.222079486586154e-05,
            "s": -0.007245935034006834,
            "q": 0.0015470862438258237,
        }
        for name, gradient in gradients.items():
            self.assert_finite_nonzero(name, gradient)
            torch.testing.assert_close(
                gradient.double().sum().cpu(),
                torch.tensor(expected_sums[name], dtype=torch.float64),
                rtol=2.0e-4,
                atol=2.0e-7,
                msg=lambda message, name=name: f"{name}: {message}",
            )

        self.assertEqual(tuple(gradients["w1"].shape), (64, 128))
        self.assertEqual(tuple(gradients["w2"].shape), (64, 64))
        self.assertEqual(tuple(gradients["w3"].shape), (8, 64))
        self.assertEqual(tuple(gradients["conditioning"].shape), (2, 96))
        self.assertEqual(tuple(gradients["features"].shape), (13154,))
        for name, value in {
            "w1_uv": gradients["w1"][:, :8],
            "w1_v": gradients["w1"][:, 8:32],
            "w1_conditioning": gradients["w1"][:, 32:128],
        }.items():
            self.assert_finite_nonzero(name, value)


if __name__ == "__main__":
    unittest.main()
