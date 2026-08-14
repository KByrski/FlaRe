from __future__ import annotations

from typing import Any

from renderer_backend import NativeFlaReBackend


class FlaReRenderer:
    """Renderer facade with explicit model-to-native geometry synchronization."""

    def __init__(
        self,
        number_of_sides: int,
        chi_square_squared_radius: float,
        max_batch_size: int,
        *,
        backend: NativeFlaReBackend | None = None,
    ) -> None:
        self.backend = backend or NativeFlaReBackend()
        self._native_renderer = self.backend.create_renderer(
            number_of_sides,
            chi_square_squared_radius,
            max_batch_size,
        )
        self._geometry_is_synchronized = False

    def sync_geometry(self, model: Any) -> None:
        geometry = model.renderer_geometry()
        self.backend.set_geometry(self._native_renderer, *geometry)
        self._geometry_is_synchronized = True

    def _require_geometry(self) -> None:
        if not self._geometry_is_synchronized:
            raise RuntimeError(
                "renderer geometry is not synchronized; call sync_geometry(model)"
            )

    def generate_rays(self, *args: object) -> Any:
        return self.backend.generate_rays(*args)

    def forward_training_base(self, *args: object) -> Any:
        self._require_geometry()
        return self.backend.forward_training_base(self._native_renderer, *args)

    def backward_base(self, *args: object) -> Any:
        self._require_geometry()
        return self.backend.backward_base(self._native_renderer, *args)

    def forward_training(self, *args: object) -> Any:
        self._require_geometry()
        return self.backend.forward_training(self._native_renderer, *args)

    def backward(self, *args: object) -> Any:
        self._require_geometry()
        return self.backend.backward(self._native_renderer, *args)

    def forward_inference_base(self, *args: object) -> Any:
        self._require_geometry()
        return self.backend.forward_inference_base(self._native_renderer, *args)

    def forward_inference(self, *args: object) -> Any:
        self._require_geometry()
        return self.backend.forward_inference(self._native_renderer, *args)
