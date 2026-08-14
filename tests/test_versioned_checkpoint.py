from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import torch

from checkpoint_io import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    LEGACY_MODEL_TENSOR_NAMES,
    MODEL_TENSOR_NAMES,
    checkpoint_state_from_payload,
    create_checkpoint_payload,
    load_model_checkpoint,
    load_training_checkpoint,
    save_checkpoint_payload,
)


def synthetic_model(include_rgb: bool = True) -> dict[str, torch.Tensor]:
    names = MODEL_TENSOR_NAMES if include_rgb else LEGACY_MODEL_TENSOR_NAMES
    return {
        name: torch.tensor([float(index), float(index) + 0.25])
        for index, name in enumerate(names)
    }


def legacy_payload(include_rgb: bool = True, iteration: int = 42):
    names = MODEL_TENSOR_NAMES if include_rgb else LEGACY_MODEL_TENSOR_NAMES
    optimizer = {"state": {0: {"step": torch.tensor(float(iteration))}}}
    return tuple(synthetic_model(include_rgb)[name] for name in names) + (
        optimizer,
        17.25,
    )


class VersionedCheckpointServiceTest(unittest.TestCase):
    def test_v1_payload_has_named_model_and_explicit_training_state(self):
        model = synthetic_model()
        optimizer = {"state": {0: {"step": torch.tensor(73.0)}}}
        config = {"essential": {"end_iter": 64_000}}

        payload = create_checkpoint_payload(
            model,
            optimizer_state=optimizer,
            iteration=73,
            training_time_seconds=12.5,
            config=config,
        )

        self.assertEqual(payload["format"], CHECKPOINT_FORMAT)
        self.assertEqual(payload["version"], CHECKPOINT_VERSION)
        self.assertEqual(tuple(payload["model"]), MODEL_TENSOR_NAMES)
        self.assertIs(payload["optimizer"], optimizer)
        self.assertEqual(
            payload["training"],
            {"iteration": 73, "time_seconds": 12.5},
        )
        self.assertEqual(payload["config"], config)

        state = checkpoint_state_from_payload(payload, "cpu", require_rgb=True)
        self.assertEqual(state.iteration, 73)
        self.assertEqual(state.training_time_seconds, 12.5)
        self.assertIs(state.optimizer_state, optimizer)
        self.assertEqual(state.config, config)
        for name in MODEL_TENSOR_NAMES:
            torch.testing.assert_close(state.model[name], model[name])

    def test_legacy_tuple_variants_migrate_to_the_same_state(self):
        for include_rgb in (False, True):
            with self.subTest(include_rgb=include_rgb):
                state = checkpoint_state_from_payload(
                    legacy_payload(include_rgb),
                    "cpu",
                    require_rgb=include_rgb,
                )
                expected_names = (
                    MODEL_TENSOR_NAMES
                    if include_rgb
                    else LEGACY_MODEL_TENSOR_NAMES
                )
                self.assertEqual(tuple(state.model), expected_names)
                self.assertEqual(state.iteration, 42)
                self.assertEqual(state.training_time_seconds, 17.25)
                self.assertIsNone(state.config)
                self.assertEqual(state.source_version, 0)

    def test_atomic_file_round_trip_and_compatible_model_view(self):
        payload = create_checkpoint_payload(
            synthetic_model(),
            optimizer_state={"state": {}},
            iteration=9,
            training_time_seconds=3.5,
            config=None,
        )
        with TemporaryDirectory(prefix="flare-versioned-checkpoint-") as directory:
            path = Path(directory) / "model.checkpoint"
            save_checkpoint_payload(path, payload)

            self.assertTrue(path.is_file())
            self.assertFalse(Path(str(path) + ".tmp").exists())
            state = load_training_checkpoint(path, "cpu", require_rgb=True)
            model = load_model_checkpoint(path, "cpu", require_rgb=True)

        self.assertEqual(state.iteration, 9)
        self.assertEqual(state.source_version, CHECKPOINT_VERSION)
        self.assertEqual(model["training_time_seconds"], 3.5)
        for name in MODEL_TENSOR_NAMES:
            torch.testing.assert_close(model[name], payload["model"][name])

    def test_failed_atomic_save_preserves_destination_and_removes_temporary(self):
        payload = create_checkpoint_payload(
            synthetic_model(),
            optimizer_state=None,
            iteration=0,
            training_time_seconds=0.0,
        )
        with TemporaryDirectory(prefix="flare-versioned-checkpoint-") as directory:
            path = Path(directory) / "model.checkpoint"
            path.write_bytes(b"existing-checkpoint")
            temporary = Path(str(path) + ".tmp")

            with mock.patch(
                "checkpoint_io.torch.save",
                side_effect=RuntimeError("synthetic save failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic save failure"):
                    save_checkpoint_payload(path, payload)

            self.assertEqual(path.read_bytes(), b"existing-checkpoint")
            self.assertFalse(temporary.exists())

    def test_schema_validation_rejects_future_and_incomplete_payloads(self):
        payload = create_checkpoint_payload(
            synthetic_model(),
            optimizer_state=None,
            iteration=0,
            training_time_seconds=0.0,
        )

        future = dict(payload, version=CHECKPOINT_VERSION + 1)
        with self.assertRaisesRegex(ValueError, "version"):
            checkpoint_state_from_payload(future, "cpu")

        incomplete = dict(payload)
        incomplete["model"] = dict(payload["model"])
        del incomplete["model"]["q"]
        with self.assertRaisesRegex(ValueError, "missing model tensors.*q"):
            checkpoint_state_from_payload(incomplete, "cpu")

        invalid_training = dict(payload, training={"iteration": 0})
        with self.assertRaisesRegex(ValueError, "time_seconds"):
            checkpoint_state_from_payload(invalid_training, "cpu")


if __name__ == "__main__":
    unittest.main()
