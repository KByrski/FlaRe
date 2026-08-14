from __future__ import annotations

import importlib.util
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


HAS_SIMPLE_KNN = importlib.util.find_spec("simple_knn._C") is not None
INITIALIZATION_TENSORS = (
    "w1_uv",
    "w1_v",
    "w1_conditioning",
    "b1",
    "w2",
    "b2",
    "w3",
    "b3",
    "features",
    "conditioning_variable",
    "RGB",
    "A",
    "k",
    "m",
    "s",
    "q",
)

OPTIMIZER_TENSORS = (
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


@unittest.skipUnless(
    torch.cuda.is_available() and HAS_SIMPLE_KNN,
    "legacy FlaRe initialization requires CUDA and simple_knn",
)
class SeededInitializationRegressionTest(unittest.TestCase):
    """Freeze the RNG order and parameter conventions used by FlaRe today."""

    @staticmethod
    def _seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @classmethod
    def _build_model(cls, seed: int = 0):
        from scene.gaussian_model import GaussianModel
        from utils.graphics_utils import BasicPointCloud

        points = np.asarray(
            [
                [-0.8, -0.3, 2.0],
                [-0.2, 0.4, 2.2],
                [0.1, -0.5, 2.4],
                [0.6, 0.2, 2.6],
                [0.9, -0.1, 2.8],
                [-0.5, 0.7, 3.0],
            ],
            dtype=np.float32,
        )
        colors = np.asarray(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [0.2, 0.4, 0.6],
                [0.3, 0.6, 0.9],
                [0.9, 0.5, 0.1],
            ],
            dtype=np.float32,
        )
        point_cloud = BasicPointCloud(
            points=points,
            colors=colors,
            normals=np.zeros_like(points),
        )
        config = SimpleNamespace(
            initial_conditioning_std=0.01,
            initial_opacity=0.01,
            initial_k=1.01,
        )

        cls._seed(seed)
        model = GaussianModel()
        model.create_from_pcd(config, point_cloud, 1.0)
        return model, points, colors

    @staticmethod
    def _snapshot(model) -> dict[str, torch.Tensor]:
        return {
            name: getattr(model, name).detach().cpu().clone()
            for name in INITIALIZATION_TENSORS
        }

    def test_seed_zero_repeats_the_complete_initialization(self) -> None:
        first, _, _ = self._build_model(seed=0)
        first_snapshot = self._snapshot(first)
        del first

        second, _, _ = self._build_model(seed=0)
        second_snapshot = self._snapshot(second)

        for name in INITIALIZATION_TENSORS:
            torch.testing.assert_close(
                second_snapshot[name],
                first_snapshot[name],
                rtol=0.0,
                atol=0.0,
                msg=lambda message, name=name: f"{name}: {message}",
            )

    def test_seed_zero_matches_legacy_reference_values(self) -> None:
        model, points, colors = self._build_model(seed=0)

        expected_prefixes = {
            "w1_uv": [-0.0462331213, -0.0212672111, -0.1321922988, 0.0072591933],
            "w1_v": [0.0052199918, -0.0159424525, 0.0266681947, -0.0212167706],
            "w1_conditioning": [0.0367208794, -0.0103396056, -0.0071211667, 0.0018289093],
            "w2": [0.0062151547, -0.0250517484, -0.0147588626, 0.0054724845],
            "w3": [-0.0061017540, 0.0110897664, 0.0101034772, -0.0050463718],
            "features": [-9.9347590e-06, 6.8696078e-05, 8.2432198e-05, -6.0021554e-05],
            "conditioning_variable": [-0.0076866564, 0.0157541428, -0.0069803218, 0.0158762988],
            "q": [0.2050020248, 0.4730118811, -0.7099779844, 0.4797553122],
        }
        for name, expected in expected_prefixes.items():
            actual = getattr(model, name).detach().cpu().reshape(-1)[: len(expected)]
            torch.testing.assert_close(
                actual,
                torch.tensor(expected, dtype=torch.float32),
                rtol=0.0,
                atol=1.0e-8,
                msg=lambda message, name=name: f"{name}: {message}",
            )

        self.assertEqual(tuple(model.w1_uv.shape), (64, 8))
        self.assertEqual(tuple(model.w1_v.shape), (64, 24))
        self.assertEqual(tuple(model.w1_conditioning.shape), (64, 96))
        self.assertEqual(tuple(model.w3.shape), (16, 64))
        self.assertEqual(tuple(model.features.shape), (13154,))
        self.assertEqual(tuple(model.conditioning_variable.shape), (6, 96))

        torch.testing.assert_close(model.RGB.cpu(), torch.from_numpy(colors))
        torch.testing.assert_close(model.m.cpu(), torch.from_numpy(points))
        torch.testing.assert_close(
            model.A.cpu(),
            torch.full((6, 1), -4.595119953155518),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            model.k.cpu(),
            torch.full((6, 1), -4.600167274475098),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(model.q, dim=1).cpu(),
            torch.ones(6),
            rtol=0.0,
            atol=1.0e-7,
        )

    def test_complete_trainable_tensor_and_optimizer_contract(self) -> None:
        from arguments.config import LearningConfig

        model, _, _ = self._build_model(seed=0)
        self.assertIsInstance(model, torch.nn.Module)
        named_parameters = dict(model.named_parameters())
        self.assertEqual(tuple(named_parameters), OPTIMIZER_TENSORS)
        self.assertEqual(tuple(model.model_tensors()), OPTIMIZER_TENSORS)
        for name in OPTIMIZER_TENSORS:
            parameter = getattr(model, name)
            self.assertIsInstance(parameter, torch.nn.Parameter)
            self.assertIs(named_parameters[name], parameter)
            self.assertTrue(parameter.is_leaf)
            self.assertFalse(parameter.requires_grad)

        geometry = model.renderer_geometry()
        self.assertIs(geometry[0], model.m)
        self.assertIs(geometry[2], model.q)
        torch.testing.assert_close(
            geometry[1], torch.exp(model.s), rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            geometry[3], torch.sigmoid(model.A), rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            geometry[4],
            1.0 + torch.nn.functional.softplus(model.k),
            rtol=0.0,
            atol=0.0,
        )

        model.training_setup(LearningConfig())

        expected_shapes = {
            "RGB": (6, 3),
            "A": (6, 1),
            "k": (6, 1),
            "w1_uv": (64, 8),
            "w1_v": (64, 24),
            "w1_conditioning": (64, 96),
            "b1": (64,),
            "w2": (64, 64),
            "b2": (64,),
            "w3": (16, 64),
            "b3": (16,),
            "conditioning_variable": (6, 96),
            "features": (13154,),
            "m": (6, 3),
            "s": (6, 2),
            "q": (6, 4),
        }
        self.assertEqual(len(model.optimizer.param_groups), 16)
        self.assertEqual(
            tuple(group["name"] for group in model.optimizer.param_groups),
            OPTIMIZER_TENSORS,
        )
        self.assertEqual(
            tuple(
                group["name"]
                for group in model.optimizer.state_dict()["param_groups"]
            ),
            OPTIMIZER_TENSORS,
        )
        for index, name in enumerate(OPTIMIZER_TENSORS):
            tensor = getattr(model, name)
            with self.subTest(name=name):
                self.assertEqual(tuple(tensor.shape), expected_shapes[name])
                self.assertEqual(tensor.dtype, torch.float32)
                self.assertEqual(tensor.device.type, "cuda")
                self.assertTrue(tensor.is_leaf)
                self.assertTrue(tensor.requires_grad)
                self.assertIs(
                    model.optimizer.param_groups[index]["params"][0], tensor
                )


if __name__ == "__main__":
    unittest.main()
