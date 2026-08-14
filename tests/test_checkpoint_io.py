from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from checkpoint_io import (
    LEGACY_MODEL_TENSOR_NAMES,
    MODEL_TENSOR_NAMES,
    checkpoint_tensor_names,
    load_model_checkpoint,
    model_from_checkpoint,
)


def synthetic_checkpoint(include_rgb: bool) -> tuple[object, ...]:
    names = MODEL_TENSOR_NAMES if include_rgb else LEGACY_MODEL_TENSOR_NAMES
    tensors = tuple(
        torch.tensor([[float(index), float(index) + 0.5]]).t()
        for index, _ in enumerate(names)
    )
    return tensors + ({"state": {}}, 123.5)


class LegacyCheckpointCompatibilityTest(unittest.TestCase):
    def test_eighteen_entry_checkpoint_maps_every_fla_re_tensor(self) -> None:
        checkpoint = synthetic_checkpoint(include_rgb=True)
        model = model_from_checkpoint(checkpoint, "cpu", require_rgb=True)

        self.assertEqual(checkpoint_tensor_names(checkpoint), MODEL_TENSOR_NAMES)
        self.assertEqual(
            tuple(name for name in MODEL_TENSOR_NAMES if name in model),
            MODEL_TENSOR_NAMES,
        )
        self.assertEqual(model["training_time_seconds"], 123.5)
        for index, name in enumerate(MODEL_TENSOR_NAMES):
            self.assertTrue(model[name].is_contiguous())
            self.assertFalse(model[name].requires_grad)
            torch.testing.assert_close(
                model[name], checkpoint[index], rtol=0.0, atol=0.0
            )

    def test_seventeen_entry_checkpoint_omits_only_base_rgb(self) -> None:
        checkpoint = synthetic_checkpoint(include_rgb=False)
        model = model_from_checkpoint(checkpoint, "cpu")

        self.assertEqual(
            checkpoint_tensor_names(checkpoint), LEGACY_MODEL_TENSOR_NAMES
        )
        self.assertNotIn("RGB", model)
        self.assertEqual(
            tuple(name for name in MODEL_TENSOR_NAMES if name in model),
            LEGACY_MODEL_TENSOR_NAMES,
        )
        with self.assertRaisesRegex(ValueError, "no base RGB"):
            model_from_checkpoint(checkpoint, "cpu", require_rgb=True)

    def test_both_tuple_variants_round_trip_through_file_loader(self) -> None:
        with TemporaryDirectory(prefix="flare-checkpoint-regression-") as directory:
            root = Path(directory)
            for include_rgb, entries in ((False, 17), (True, 18)):
                checkpoint = synthetic_checkpoint(include_rgb)
                path = root / f"schema-{entries}.checkpoint"
                torch.save(checkpoint, path)
                loaded = load_model_checkpoint(
                    path, "cpu", require_rgb=include_rgb
                )
                expected_names = (
                    MODEL_TENSOR_NAMES
                    if include_rgb
                    else LEGACY_MODEL_TENSOR_NAMES
                )
                self.assertEqual(
                    tuple(name for name in MODEL_TENSOR_NAMES if name in loaded),
                    expected_names,
                )
                self.assertEqual(loaded["training_time_seconds"], 123.5)

    def test_invalid_schema_entries_fail_with_specific_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 17.*or 18"):
            model_from_checkpoint((torch.zeros(1),) * 16, "cpu")
        with self.assertRaisesRegex(ValueError, "expected a tuple or list"):
            model_from_checkpoint({"model": {}}, "cpu")

        invalid_tensor = list(synthetic_checkpoint(include_rgb=True))
        invalid_tensor[3] = "not a tensor"
        with self.assertRaisesRegex(ValueError, r"entry 3 \(w1_uv\)"):
            model_from_checkpoint(invalid_tensor, "cpu")

        invalid_time = list(synthetic_checkpoint(include_rgb=True))
        invalid_time[-1] = object()
        with self.assertRaisesRegex(ValueError, "training time"):
            model_from_checkpoint(invalid_time, "cpu")


if __name__ == "__main__":
    unittest.main()
