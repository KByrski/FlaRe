from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys
from typing import Any


class NativeFlaReBackend:
    """Thin binding around the compiled FlaRe renderer extension."""

    def __init__(self, native_module: Any | None = None) -> None:
        if native_module is None:
            renderer_dir = Path(__file__).resolve().parent / "renderer" / "output"
            if str(renderer_dir) not in sys.path:
                sys.path.insert(0, str(renderer_dir))
            native_module = import_module("PYOPTIXFLARERENDERER")
        self.native_module = native_module

    def create_renderer(
        self,
        number_of_sides: int,
        chi_square_squared_radius: float,
        max_batch_size: int,
    ) -> Any:
        return self.native_module.CPyOptiXFLARERenderer(
            number_of_sides,
            chi_square_squared_radius,
            max_batch_size,
        )

    def set_geometry(self, renderer: Any, *geometry: object) -> None:
        renderer.SetGeometry(*geometry)

    def generate_rays(self, *args: object) -> Any:
        return self.native_module.GenerateRays(*args)

    def forward_training_base(self, renderer: Any, *args: object) -> Any:
        return renderer.Forward_training_base(*args)

    def backward_base(self, renderer: Any, *args: object) -> Any:
        return renderer.Backward_base(*args)

    def forward_training(self, renderer: Any, *args: object) -> Any:
        return renderer.Forward_training(*args)

    def backward(self, renderer: Any, *args: object) -> Any:
        return renderer.Backward(*args)

    def forward_inference_base(self, renderer: Any, *args: object) -> Any:
        return renderer.Forward_inference_base(*args)

    def forward_inference(self, renderer: Any, *args: object) -> Any:
        return renderer.Forward_inference(*args)
