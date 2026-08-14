from __future__ import annotations

import unittest

import torch

import tests.test_native_renderer as native_regression
from scene.rays import RayBundle
from training_step import (
    FlaReTrainingStepInput,
    NativeGradientBundle,
    run_fla_re_training_step,
)



@unittest.skipUnless(
    torch.cuda.is_available() and native_regression.RENDERER_SO.is_file(),
    "training-step regression requires CUDA and a built FlaRe renderer",
)
class WarmupTrainingStepRegressionTest(unittest.TestCase):
    """Freeze the legacy base/full gradient blend and one Adam update."""

    @classmethod
    def setUpClass(cls) -> None:
        fixture_type = native_regression.NativeFlaReRendererRegressionTest
        fixture_type.setUpClass()
        cls.fixture_type = fixture_type
        cls.fixture = fixture_type(methodName="runTest")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fixture_type.tearDownClass()

    def setUp(self) -> None:
        self.fixture.setUp()

    def _base_gradients(self) -> dict[str, torch.Tensor]:
        fixture = self.fixture
        image, auxiliary = fixture._forward_base()
        image_gradient = (
            2.0 * (image - fixture.target) / image.numel()
        ).contiguous()
        gradients = {
            "RGB": torch.zeros_like(fixture.rgb),
            "A": torch.zeros_like(fixture.opacity_logits),
            "k": torch.zeros_like(fixture.raw_kappa),
            "m": torch.zeros_like(fixture.means),
            "s": torch.zeros_like(fixture.log_scales),
            "q": torch.zeros_like(fixture.quaternions),
        }
        prefix = torch.zeros(
            (fixture.ray_count, 4), device=fixture.device
        )
        fixture.renderer.backward_base(
            fixture.origins,
            fixture.directions,
            *fixture.background,
            fixture.means,
            fixture.log_scales,
            fixture.quaternions,
            fixture.rgba,
            fixture.raw_kappa,
            image.clone(),
            image_gradient,
            gradients["RGB"],
            gradients["A"],
            gradients["k"],
            gradients["m"],
            gradients["s"],
            gradients["q"],
            fixture.threshold,
            auxiliary["depth"],
            prefix,
            0.0,
            fixture.reg_a,
            fixture.reg_b,
            auxiliary["depth_and_index"],
            auxiliary["surface_normal"],
            auxiliary["normal"],
            0.0,
        )
        return gradients

    def _full_gradients(self) -> dict[str, torch.Tensor]:
        fixture = self.fixture
        image, auxiliary = fixture._forward_full()
        image_gradient = (
            2.0 * (image - fixture.target) / image.numel()
        ).contiguous()
        sm_buffers = 4 * torch.cuda.get_device_properties(
            fixture.device
        ).multi_processor_count
        gradients = {
            "RGB": torch.zeros_like(fixture.rgb),
            "A": torch.zeros_like(fixture.opacity_logits),
            "k": torch.zeros_like(fixture.raw_kappa),
            "w3": torch.zeros(
                (sm_buffers, 8 * 64), device=fixture.device
            ),
            "b3": torch.zeros(
                (sm_buffers, 8 * 8), device=fixture.device
            ),
            "w2": torch.zeros(
                (sm_buffers, 64 * 64), device=fixture.device
            ),
            "b2": torch.zeros(
                (sm_buffers, 64 * 8), device=fixture.device
            ),
            "w1": torch.zeros(
                (sm_buffers, 64 * 128), device=fixture.device
            ),
            "b1": torch.zeros(
                (sm_buffers, 64 * 8), device=fixture.device
            ),
            "conditioning": torch.zeros(
                (2, 96), device=fixture.device
            ),
            "features": torch.zeros_like(fixture.features),
            "m": torch.zeros_like(fixture.means),
            "s": torch.zeros_like(fixture.log_scales),
            "q": torch.zeros_like(fixture.quaternions),
        }
        prefix = torch.zeros(
            (fixture.ray_count, 4), device=fixture.device
        )
        fixture.renderer.backward(
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
            *fixture.background,
            fixture.means,
            fixture.log_scales,
            fixture.quaternions,
            fixture.rgba,
            fixture.raw_kappa,
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
            fixture.threshold,
            auxiliary["depth"],
            prefix,
            0.0,
            fixture.reg_a,
            fixture.reg_b,
            auxiliary["depth_and_index"],
            auxiliary["surface_normal"],
            auxiliary["normal"],
            0.0,
        )
        torch.cuda.synchronize()
        native_regression._reduce_mlp_gradients(gradients)
        return gradients

    def _parameters(self) -> dict[str, torch.nn.Parameter]:
        fixture = self.fixture

        def parameter(value: torch.Tensor) -> torch.nn.Parameter:
            return torch.nn.Parameter(value.detach().float().clone())

        return {
            "RGB": parameter(fixture.rgb),
            "A": parameter(fixture.opacity_logits),
            "k": parameter(fixture.raw_kappa),
            "w1_uv": parameter(fixture.w1[:, :8]),
            "w1_v": parameter(fixture.w1[:, 8:32]),
            "w1_conditioning": parameter(fixture.w1[:, 32:128]),
            "b1": parameter(fixture.b1),
            "w2": parameter(fixture.w2),
            "b2": parameter(fixture.b2),
            "w3": parameter(fixture.w3),
            "b3": parameter(fixture.b3),
            "conditioning": parameter(fixture.conditioning),
            "features": parameter(fixture.features),
            "m": parameter(fixture.means),
            "s": parameter(fixture.log_scales),
            "q": parameter(fixture.quaternions),
        }

    def test_blended_warmup_step_updates_all_sixteen_optimizer_groups(
        self,
    ) -> None:
        base_image, _ = self.fixture._forward_base()
        full_image, _ = self.fixture._forward_full()
        base_loss = torch.mean(
            (torch.clamp(base_image, 0.0, 1.0) - self.fixture.target) ** 2
        )
        full_loss = torch.mean(
            (torch.clamp(full_image, 0.0, 1.0) - self.fixture.target) ** 2
        )
        base = self._base_gradients()
        full = self._full_gradients()
        parameters = self._parameters()
        warmup_lambda = 0.25
        scale_regularization = 0.017

        self.assertLess(warmup_lambda, 1.0)
        self.assertGreater(warmup_lambda, 0.0)
        self.assertAlmostEqual(base_loss.item(), 0.23518840968608856)
        self.assertAlmostEqual(full_loss.item(), 0.2946890592575073)
        log_ten = torch.log(torch.tensor([10.0], device=self.fixture.device))
        self.assertAlmostEqual(
            (-10.0 * torch.log(base_loss) / log_ten).item(),
            6.28584098815918,
        )
        self.assertAlmostEqual(
            (-10.0 * torch.log(full_loss) / log_ten).item(),
            5.306360244750977,
        )

        common_names = ("RGB", "A", "k", "m", "s", "q")
        for name in common_names:
            parameters[name].grad = (
                (1.0 - warmup_lambda) * base[name]
                + warmup_lambda * full[name]
            )
        parameters["w1_uv"].grad = warmup_lambda * full["w1"][:, :8]
        parameters["w1_v"].grad = warmup_lambda * full["w1"][:, 8:32]
        parameters["w1_conditioning"].grad = (
            warmup_lambda * full["w1"][:, 32:128]
        )
        parameters["b1"].grad = warmup_lambda * full["b1"]
        parameters["w2"].grad = warmup_lambda * full["w2"]
        parameters["b2"].grad = warmup_lambda * full["b2"]
        parameters["w3"].grad = torch.nn.functional.pad(
            warmup_lambda * full["w3"], (0, 0, 0, 8)
        )
        parameters["b3"].grad = torch.nn.functional.pad(
            warmup_lambda * full["b3"], (0, 8)
        )
        parameters["conditioning"].grad = (
            warmup_lambda * full["conditioning"]
        )
        parameters["features"].grad = warmup_lambda * full["features"]
        scales_squared = torch.exp(parameters["s"].detach()) ** 2
        parameters["s"].grad += (
            scale_regularization
            / parameters["m"].shape[0]
            * scales_squared
            / torch.sqrt(scales_squared.sum(1, keepdim=True))
        )

        ordered_names = (
            "RGB",
            "A",
            "k",
            "w1_uv",
            "w1_v",
            "w1_conditioning",
            "b1",
            "w2",
            "b2",
            "w3",
            "b3",
            "conditioning",
            "features",
            "m",
            "s",
            "q",
        )
        learning_rates = {
            name: 1.0e-3 * (index + 1)
            for index, name in enumerate(ordered_names)
        }
        optimizer = torch.optim.Adam(
            [
                {
                    "params": [parameters[name]],
                    "lr": learning_rates[name],
                }
                for name in ordered_names
            ]
        )
        before = {
            name: value.detach().clone()
            for name, value in parameters.items()
        }

        torch.testing.assert_close(
            parameters["RGB"].grad,
            0.75 * base["RGB"] + 0.25 * full["RGB"],
            rtol=0.0,
            atol=0.0,
        )
        gradient_fingerprints = {
            "RGB": (-0.11105563188903034, 0.0046207730548845785),
            "A": (-0.0010070049320347607, 2.705442166490961e-06),
            "k": (0.006669549155049026, 2.6875553555578873e-05),
            "w1_uv": (-4.810164331778277e-08, 5.823106931881319e-16),
            "w1_v": (0.0005158174658683734, 3.530872062400738e-08),
            "w1_conditioning": (-1.5124661495169445e-05, 4.31951003918102e-10),
            "b1": (0.0001509274617887968, 6.311636599716227e-09),
            "w2": (-6.54718651738051e-05, 1.8431979990479594e-08),
            "b2": (0.0007317789518310747, 7.858010517069155e-07),
            "w3": (0.0005186356572224327, 2.2381524089202982e-07),
            "b3": (-0.017349349916912615, 0.00011842173594345393),
            "conditioning": (-2.786338846449965e-05, 2.2340351925773805e-10),
            "features": (2.4697967778763192e-06, 5.677158423181881e-13),
            "m": (-0.021547363605350256, 0.0019148933515244243),
            "s": (-0.06651398353278637, 0.0012909256475179716),
            "q": (0.03173466712376216, 0.0011211729327526721),
        }
        for name, fingerprint in gradient_fingerprints.items():
            self.fixture.assert_fingerprint(
                parameters[name].grad, *fingerprint, rtol=2.0e-4
            )
        self.assertEqual(len(optimizer.param_groups), 16)
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [learning_rates[name] for name in ordered_names],
        )
        for name, parameter in parameters.items():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0, name)
        torch.testing.assert_close(
            parameters["w3"].grad[8:],
            torch.zeros_like(parameters["w3"].grad[8:]),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            parameters["b3"].grad[8:],
            torch.zeros_like(parameters["b3"].grad[8:]),
            rtol=0.0,
            atol=0.0,
        )

        optimizer.step()
        torch.cuda.synchronize()

        for name in ordered_names:
            parameter = parameters[name]
            self.assertFalse(torch.equal(parameter, before[name]), name)
            state = optimizer.state[parameter]
            self.assertEqual(tuple(state["exp_avg"].shape), tuple(parameter.shape))
            self.assertEqual(
                tuple(state["exp_avg_sq"].shape), tuple(parameter.shape)
            )
            self.assertTrue(bool(torch.isfinite(state["exp_avg"]).all()), name)
            self.assertTrue(bool(torch.isfinite(state["exp_avg_sq"]).all()), name)

        torch.testing.assert_close(
            parameters["w3"][8:], before["w3"][8:], rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            parameters["b3"][8:], before["b3"][8:], rtol=0.0, atol=0.0
        )

    def test_extracted_step_matches_frozen_native_contract(self) -> None:
        base = self._base_gradients()
        full = self._full_gradients()
        parameters = self._parameters()
        parameters["conditioning_variable"] = parameters.pop("conditioning")
        model = torch.nn.Module()
        for name, parameter in parameters.items():
            setattr(model, name, parameter)

        ordered_names = (
            "RGB",
            "A",
            "k",
            "w1_uv",
            "w1_v",
            "w1_conditioning",
            "b1",
            "w2",
            "b2",
            "w3",
            "b3",
            "conditioning_variable",
            "features",
            "m",
            "s",
            "q",
        )
        learning_rates = {
            name: 1.0e-3 * (index + 1)
            for index, name in enumerate(ordered_names)
        }
        optimizer = torch.optim.Adam(
            [
                {
                    "params": [getattr(model, name)],
                    "lr": learning_rates[name],
                }
                for name in ordered_names
            ]
        )
        before = {
            name: getattr(model, name).detach().clone()
            for name in ordered_names
        }

        warmup_lambda = 0.25
        expected = {}
        for name in ("RGB", "A", "k", "m", "s", "q"):
            expected[name] = torch.zeros_like(base[name])
            expected[name] += (1.0 - warmup_lambda) * base[name]
            expected[name] += warmup_lambda * full[name]

        expected.update(
            {
                "w1_uv": warmup_lambda * full["w1"][:, :8],
                "w1_v": warmup_lambda * full["w1"][:, 8:32],
                "w1_conditioning": (
                    warmup_lambda * full["w1"][:, 32:128]
                ),
                "b1": warmup_lambda * full["b1"],
                "w2": warmup_lambda * full["w2"],
                "b2": warmup_lambda * full["b2"],
                "w3": torch.nn.functional.pad(
                    warmup_lambda * full["w3"], (0, 0, 0, 8)
                ),
                "b3": torch.nn.functional.pad(
                    warmup_lambda * full["b3"], (0, 8)
                ),
                "conditioning_variable": (
                    warmup_lambda * full["conditioning"]
                ),
                "features": warmup_lambda * full["features"],
            }
        )
        scales_squared = torch.exp(model.s.detach()) ** 2
        expected["s"] = expected["s"] + (
            0.017
            / model.m.shape[0]
            * scales_squared
            / torch.sqrt(scales_squared.sum(1, keepdim=True))
        )

        fixture = self.fixture
        flat_indices = torch.arange(
            fixture.ray_count, dtype=torch.int64, device=fixture.device
        )
        rays = RayBundle(
            origins=fixture.origins,
            directions=fixture.directions,
            flat_indices=flat_indices,
            camera_indices=torch.zeros_like(flat_indices),
            pixel_indices=flat_indices.clone(),
        )
        result = run_fla_re_training_step(
            FlaReTrainingStepInput(
                model=model,
                renderer=fixture.renderer,
                optimizer=optimizer,
                rays=rays,
                foreground=fixture.target,
                alpha=torch.ones(
                    (fixture.ray_count, 1), device=fixture.device
                ),
                background=torch.tensor(
                    fixture.background, device=fixture.device
                ).reshape(1, 3),
                background_rgb=fixture.background,
                warmup_lambda=warmup_lambda,
                ray_termination_threshold=fixture.threshold,
                depth_lambda=0.0,
                normal_lambda=0.0,
                reg_depth_a=fixture.reg_a,
                reg_depth_b=fixture.reg_b,
                scale_regularization=0.017,
                sm_count=torch.cuda.get_device_properties(
                    fixture.device
                ).multi_processor_count,
                image_height=fixture.height,
                image_width=fixture.width,
                normal_depth_edge_threshold=0.0,
            )
        )
        torch.cuda.synchronize()

        self.assertIsNotNone(result.base)
        self.assertIsNotNone(result.flare)
        self.assertAlmostEqual(result.base.loss, 0.23518840968608856)
        self.assertAlmostEqual(result.base.psnr, 6.28584098815918)
        self.assertAlmostEqual(result.flare.loss, 0.2946890592575073)
        self.assertAlmostEqual(result.flare.psnr, 5.306360244750977)
        self.assertIsInstance(result.gradients, NativeGradientBundle)
        self.assertEqual(
            tuple(result.gradients.as_dict()), ordered_names
        )
        for name in ordered_names:
            gradient = result.gradients.as_dict()[name]
            torch.testing.assert_close(
                gradient, expected[name], rtol=2.0e-4, atol=2.0e-7
            )
            self.assertIs(getattr(model, name).grad, gradient)
            self.assertFalse(torch.equal(getattr(model, name), before[name]))


if __name__ == "__main__":
    unittest.main()
