from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from arguments.config import LearningConfig
from checkpoint_io import LEGACY_MODEL_TENSOR_NAMES
from evaluation_service import (
    EvaluationOptions,
    EvaluationView,
    evaluate_training_splits,
    render_one,
)
from scene.gaussian_model import GaussianModel
from scene.rays import RayBundle
from trainer import (
    PerspectiveRayBatchSampler,
    scheduled_learning_rates,
    scheduled_warmup_lambda,
    should_evaluate,
)


class RecordingDataset:
    width = 2
    height = 1
    camera_count = 2
    fov_x = torch.tensor([0.7, 1.1])
    fov_y = torch.tensor([0.5, 0.9])
    origins = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    rights = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    downs = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    forwards = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    foreground = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 12.0
    alpha = torch.ones((4, 1), dtype=torch.float32)

    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, float, float]] = []

    def rays(self, indices, *, fov_x, fov_y):
        self.calls.append((indices.clone(), fov_x, fov_y))
        directions = torch.tensor([[3.0, 4.0, 0.0]]).repeat(indices.shape[0], 1)
        return RayBundle(
            origins=torch.zeros_like(directions),
            directions=directions,
            flat_indices=indices,
            camera_indices=torch.div(indices, 2, rounding_mode="floor"),
            pixel_indices=indices % 2,
        )


class RecordingRenderer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def generate_rays(self, *args):
        self.calls.append(("generate_rays", args))
        return torch.tensor([[0.0, 0.0, 2.0], [0.0, 3.0, 4.0]])

    def forward_inference_base(self, *args):
        self.calls.append(("forward_inference_base", args))
        args[2].copy_(torch.tensor([[1.2, -0.1, 0.5], [0.1, 0.2, 0.3]]))

    def forward_inference(self, *args):
        self.calls.append(("forward_inference", args))
        args[10].copy_(torch.tensor([[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]))


def evaluation_model() -> SimpleNamespace:
    tensors = {
        "RGB": torch.ones((1, 3)),
        "A": torch.zeros((1, 1)),
        "k": torch.zeros((1, 1)),
        "w1_uv": torch.zeros((64, 8)),
        "w1_v": torch.zeros((64, 24)),
        "w1_conditioning": torch.zeros((64, 96)),
        "b1": torch.zeros(64),
        "w2": torch.zeros((64, 64)),
        "b2": torch.zeros(64),
        "w3": torch.zeros((16, 64)),
        "b3": torch.zeros(16),
        "conditioning_variable": torch.zeros((1, 96)),
        "features": torch.zeros(13154),
        "m": torch.zeros((1, 3)),
        "s": torch.zeros((1, 2)),
        "q": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    }
    return SimpleNamespace(**tensors)


class TrainerOrchestrationContractTest(unittest.TestCase):
    def test_warmup_and_evaluation_schedule_match_the_entry_point(self) -> None:
        self.assertEqual(scheduled_warmup_lambda(-1, 0, 1000), 0.0)
        self.assertEqual(scheduled_warmup_lambda(0, 0, 1000), 0.0)
        self.assertEqual(scheduled_warmup_lambda(250, 0, 1000), 0.25)
        self.assertEqual(scheduled_warmup_lambda(1000, 0, 1000), 1.0)
        self.assertEqual(scheduled_warmup_lambda(1001, 0, 1000), 1.0)
        self.assertFalse(should_evaluate(999, 64000))
        self.assertTrue(should_evaluate(1000, 64000))
        self.assertTrue(should_evaluate(64000, 64000))

    def test_learning_rate_schedule_preserves_all_sixteen_groups(self) -> None:
        self.assertEqual(
            scheduled_learning_rates(LearningConfig(), 1234),
            {
                "RGB": 0.0044929991928149985,
                "A": 0.04492999192814999,
                "k": 0.1,
                "w1_uv": 0.011232497982037497,
                "w1_v": 0.026957995156889993,
                "w1_conditioning": 0.0013478997578444997,
                "b1": 0.004268349233174248,
                "w2": 0.0044929991928149985,
                "b2": 0.0044929991928149985,
                "w3": 0.0044929991928149985,
                "b3": 0.0044929991928149985,
                "conditioning_variable": 0.005,
                "features": 0.005,
                "m": 0.0020775529439478087,
                "s": 0.0396538025,
                "q": 0.008985998385629997,
            },
        )

    def test_model_factory_owns_legacy_tensors_without_random_initialization(self) -> None:
        values = {
            name: torch.full((2,), float(index))
            for index, name in enumerate(LEGACY_MODEL_TENSOR_NAMES)
        }

        with mock.patch.object(
            GaussianModel,
            "__init__",
            side_effect=AssertionError("random initializer must not run"),
        ):
            model = GaussianModel.from_model_tensors(
                values, requires_grad=False
            )

        self.assertIsNone(model.RGB)
        self.assertEqual(model.number_of_Gaussians, 2)
        for name, expected in values.items():
            torch.testing.assert_close(getattr(model, name), expected)
            self.assertFalse(getattr(model, name).requires_grad)

    def test_random_ray_sampler_preserves_scalar_fov_and_normalization(self) -> None:
        torch.manual_seed(7)
        expected = torch.randperm(4, dtype=torch.int64, device="cpu")[:2]
        torch.manual_seed(7)
        dataset = RecordingDataset()
        sampler = PerspectiveRayBatchSampler(dataset, full_camera_batches=False)

        batch = sampler.next()

        torch.testing.assert_close(dataset.calls[0][0], expected)
        self.assertEqual(dataset.calls[0][1], dataset.fov_x[0].item())
        self.assertEqual(dataset.calls[0][2], dataset.fov_y[0].item())
        torch.testing.assert_close(
            batch.rays.directions,
            torch.tensor([[0.6, 0.8, 0.0], [0.6, 0.8, 0.0]]),
        )
        self.assertIs(batch.foreground, dataset.foreground)
        self.assertIs(batch.alpha, dataset.alpha)
        self.assertIsNone(batch.camera_forward)

    def test_full_camera_sampler_keeps_pixels_contiguous_per_pose(self) -> None:
        torch.manual_seed(11)
        expected_pose = int(torch.randperm(2)[0].item())
        torch.manual_seed(11)
        dataset = RecordingDataset()
        sampler = PerspectiveRayBatchSampler(dataset, full_camera_batches=True)

        batch = sampler.next()

        torch.testing.assert_close(
            dataset.calls[0][0],
            torch.arange(2, dtype=torch.int64) + expected_pose * 2,
        )
        self.assertEqual(dataset.calls[0][1], dataset.fov_x[expected_pose].item())
        self.assertEqual(dataset.calls[0][2], dataset.fov_y[expected_pose].item())
        torch.testing.assert_close(batch.camera_forward, dataset.forwards[expected_pose])


class EvaluatorOrchestrationContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.view = EvaluationView(
            name="view",
            width=2,
            height=1,
            fov_x=0.7,
            fov_y=0.5,
            origin=torch.tensor([1.0, 2.0, 3.0]),
            right=torch.tensor([1.0, 0.0, 0.0]),
            down=torch.tensor([0.0, 1.0, 0.0]),
            forward=torch.tensor([0.0, 0.0, 1.0]),
            ground_truth=torch.zeros((3, 1, 2)),
        )
        self.options = EvaluationOptions(
            background=(0.1, 0.2, 0.3),
            ray_termination_threshold=0.01,
        )

    @staticmethod
    def timed_call(callback):
        callback()
        return 0.125

    def test_base_render_uses_facade_and_preserves_output_layout(self) -> None:
        renderer = RecordingRenderer()
        image, elapsed, memory, source = render_one(
            "base",
            renderer,
            self.view,
            evaluation_model(),
            self.options,
            torch.device("cpu"),
            timed_call=self.timed_call,
            memory_reader=lambda: (17.0, "test"),
        )

        self.assertEqual([call[0] for call in renderer.calls], [
            "generate_rays", "forward_inference_base"
        ])
        torch.testing.assert_close(
            renderer.calls[1][1][1],
            torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.6, 0.8]]),
        )
        torch.testing.assert_close(
            image,
            torch.tensor([[[1.0, 0.1]], [[0.0, 0.2]], [[0.5, 0.3]]]),
        )
        self.assertEqual((elapsed, memory, source), (0.125, 17.0, "test"))

    def test_periodic_evaluation_keeps_modes_split_order_and_metric_names(self) -> None:
        renderer = RecordingRenderer()
        dataset = RecordingDataset()
        result = evaluate_training_splits(
            renderer,
            evaluation_model(),
            dataset,
            dataset,
            self.options,
            warmup_lambda=0.25,
            fixed_background=torch.zeros((1, 3)),
        )

        self.assertEqual(
            tuple(result.metrics),
            (
                "PSNR_train_base",
                "FPS_train_base",
                "PSNR_test_base",
                "FPS_test_base",
                "PSNR_train_FlaRe",
                "FPS_train_FlaRe",
                "PSNR_test_FlaRe",
                "FPS_test_FlaRe",
            ),
        )
        self.assertEqual(
            result.selected_checkpoint_metric,
            ("PSNR_test_FlaRe", result.metrics["PSNR_test_FlaRe"]),
        )
        self.assertEqual(
            [name for name, _ in renderer.calls],
            [
                "generate_rays",
                "forward_inference_base",
                "generate_rays",
                "forward_inference_base",
            ]
            * 2
            + [
                "generate_rays",
                "forward_inference",
                "generate_rays",
                "forward_inference",
            ]
            * 2,
        )
        expected_flare = torch.tensor(
            [[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        )
        expected = []
        for pose_index in range(dataset.camera_count):
            target = dataset.foreground.reshape(2, 2, 3)[pose_index]
            loss = torch.mean((expected_flare - target) ** 2)
            expected.append(float((-10.0 * torch.log10(loss)).item()))
        self.assertAlmostEqual(
            result.metrics["PSNR_test_FlaRe"], sum(expected) / len(expected)
        )

    def test_full_render_prepares_the_historical_compound_abi(self) -> None:
        renderer = RecordingRenderer()
        model = evaluation_model()

        render_one(
            "FlaRe",
            renderer,
            self.view,
            model,
            self.options,
            torch.device("cpu"),
            timed_call=self.timed_call,
            memory_reader=lambda: (0.0, "test"),
        )

        arguments = renderer.calls[1][1]
        self.assertEqual(arguments[0].dtype, torch.float16)
        self.assertEqual(arguments[2].shape, (64, 128))
        self.assertEqual(arguments[2].dtype, torch.float16)
        self.assertEqual(arguments[4].dtype, torch.float16)
        self.assertEqual(arguments[6].dtype, torch.float16)
        self.assertEqual(arguments[14].shape, (1, 3))
        self.assertEqual(arguments[15].shape, (1, 2))
        self.assertEqual(arguments[16].shape, (1, 4))
        self.assertEqual(arguments[17].shape, (1, 4))
        self.assertEqual(arguments[18].shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
