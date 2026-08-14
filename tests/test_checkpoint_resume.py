from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import torch

from checkpoint_io import (
    MODEL_TENSOR_NAMES,
    create_checkpoint_payload,
    save_checkpoint_payload,
)


def synthetic_model() -> dict[str, torch.Tensor]:
    return {
        name: torch.full((2,), float(index))
        for index, name in enumerate(MODEL_TENSOR_NAMES)
    }


class CapturingOptimizer:
    def __init__(self):
        self.loaded_state = None

    def load_state_dict(self, state):
        self.loaded_state = state


@unittest.skipUnless(torch.cuda.is_available(), "resume smoke test requires CUDA")
class TrainingCheckpointResumeTest(unittest.TestCase):
    def test_gaussian_model_resumes_v1_and_legacy_checkpoints(self):
        from scene.gaussian_model import GaussianModel

        def fake_training_setup(instance, _learning):
            for name in MODEL_TENSOR_NAMES:
                getattr(instance, name).requires_grad_(True)
            instance.optimizer = CapturingOptimizer()

        optimizer_state = {
            "state": {0: {"step": torch.tensor(42.0)}},
            "param_groups": [],
        }
        tensors = synthetic_model()
        legacy = tuple(tensors[name] for name in MODEL_TENSOR_NAMES) + (
            optimizer_state,
            8.5,
        )
        versioned = create_checkpoint_payload(
            tensors,
            optimizer_state=optimizer_state,
            iteration=73,
            training_time_seconds=12.5,
        )

        with TemporaryDirectory(prefix="flare-checkpoint-resume-") as directory:
            root = Path(directory)
            legacy_path = root / "legacy.checkpoint"
            versioned_path = root / "versioned.checkpoint"
            torch.save(legacy, legacy_path)
            save_checkpoint_payload(versioned_path, versioned)

            for path, expected_iteration, expected_time in (
                (legacy_path, 42, 8.5),
                (versioned_path, 73, 12.5),
            ):
                with self.subTest(path=path.name):
                    model = GaussianModel.__new__(GaussianModel)
                    with mock.patch.object(
                        GaussianModel, "training_setup", fake_training_setup
                    ):
                        model.load_checkpoint(object(), path)

                    self.assertEqual(model.iteration, expected_iteration)
                    self.assertEqual(model.training_time, expected_time)
                    self.assertIsNotNone(model.optimizer.loaded_state)
                    for name in MODEL_TENSOR_NAMES:
                        value = getattr(model, name)
                        self.assertEqual(value.device.type, "cuda")
                        self.assertTrue(value.requires_grad)
                        torch.testing.assert_close(value.cpu(), tensors[name])


if __name__ == "__main__":
    unittest.main()
