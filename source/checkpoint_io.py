"""Compatibility helpers for loading FlaRe tuple checkpoints."""

from __future__ import annotations

import gc
from pathlib import Path

import torch


MODEL_TENSOR_NAMES = (
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
LEGACY_MODEL_TENSOR_NAMES = MODEL_TENSOR_NAMES[1:]


def checkpoint_tensor_names(checkpoint: tuple[object, ...] | list[object]) -> tuple[str, ...]:
    """Return tensor names for a supported historical tuple schema."""
    length = len(checkpoint)
    if length == 18:
        return MODEL_TENSOR_NAMES
    if length == 17:
        return LEGACY_MODEL_TENSOR_NAMES
    raise ValueError(
        f"Unsupported FlaRe checkpoint with {length} entries; expected "
        "17 (FlaRe-only) or 18 (RGB+FlaRe)"
    )


def model_from_checkpoint(
    checkpoint: tuple[object, ...] | list[object],
    device: torch.device | str,
    *,
    require_rgb: bool = False,
) -> dict[str, object]:
    """Move model tensors from either supported tuple schema to the requested device."""
    if not isinstance(checkpoint, (tuple, list)):
        raise ValueError(
            "Unsupported FlaRe checkpoint object; expected a tuple or list"
        )

    names = checkpoint_tensor_names(checkpoint)
    if require_rgb and names is LEGACY_MODEL_TENSOR_NAMES:
        raise ValueError(
            "This 17-entry FlaRe-only checkpoint has no base RGB tensor"
        )

    model: dict[str, object] = {}
    for index, name in enumerate(names):
        value = checkpoint[index]
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"Checkpoint entry {index} ({name}) is not a torch.Tensor"
            )
        model[name] = value.detach().to(device).contiguous()

    try:
        model["training_time_seconds"] = float(checkpoint[-1])
    except (TypeError, ValueError) as error:
        raise ValueError(
            "The final checkpoint entry must contain training time in seconds"
        ) from error
    return model


def load_model_checkpoint(
    checkpoint_path: Path | str,
    device: torch.device | str,
    *,
    require_rgb: bool = False,
) -> dict[str, object]:
    """Load a historical FlaRe tuple checkpoint and release its CPU container."""
    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = model_from_checkpoint(checkpoint, device, require_rgb=require_rgb)
    del checkpoint
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model
