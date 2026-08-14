from __future__ import annotations

import unittest

from renderer_backend import NativeFlaReBackend
from renderer_facade import FlaReRenderer


FACADE_TO_NATIVE = (
    ("forward_training_base", "Forward_training_base"),
    ("backward_base", "Backward_base"),
    ("forward_training", "Forward_training"),
    ("backward", "Backward"),
    ("forward_inference_base", "Forward_inference_base"),
    ("forward_inference", "Forward_inference"),
)
NATIVE_RENDER_METHODS = {native for _, native in FACADE_TO_NATIVE}


class RecordingRenderer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def SetGeometry(self, *args: object) -> None:
        self.calls.append(("SetGeometry", args))

    def __getattr__(self, name: str):
        if name not in NATIVE_RENDER_METHODS:
            raise AttributeError(name)

        def record(*args: object):
            self.calls.append((name, args))
            return name, args

        return record


class RecordingNativeModule:
    def __init__(self) -> None:
        self.constructor_args: tuple[object, ...] | None = None
        self.renderer = RecordingRenderer()
        self.ray_calls: list[tuple[object, ...]] = []

    def CPyOptiXFLARERenderer(self, *args: object) -> RecordingRenderer:
        self.constructor_args = args
        return self.renderer

    def GenerateRays(self, *args: object):
        self.ray_calls.append(args)
        return "rays", args


class GeometrySource:
    def __init__(self) -> None:
        self.geometry = tuple(object() for _ in range(5))

    def renderer_geometry(self) -> tuple[object, ...]:
        return self.geometry


class RendererFacadeContractTest(unittest.TestCase):
    def _build(self):
        native = RecordingNativeModule()
        backend = NativeFlaReBackend(native)
        renderer = FlaReRenderer(8, 11.3449, 64, backend=backend)
        return native, renderer

    def test_constructor_and_explicit_geometry_upload_contract(self) -> None:
        native, renderer = self._build()
        model = GeometrySource()

        with self.assertRaisesRegex(RuntimeError, "sync_geometry"):
            renderer.forward_training_base(object())

        renderer.sync_geometry(model)

        self.assertEqual(native.constructor_args, (8, 11.3449, 64))
        self.assertEqual(len(native.renderer.calls), 1)
        method, arguments = native.renderer.calls[0]
        self.assertEqual(method, "SetGeometry")
        self.assertEqual(len(arguments), 5)
        for actual, expected in zip(arguments, model.geometry):
            self.assertIs(actual, expected)

    def test_every_training_and_inference_call_preserves_argument_order(self) -> None:
        native, renderer = self._build()
        renderer.sync_geometry(GeometrySource())
        native.renderer.calls.clear()

        for facade_name, native_name in FACADE_TO_NATIVE:
            arguments = (object(), object(), object())
            result = getattr(renderer, facade_name)(*arguments)
            self.assertEqual(result[0], native_name)
            for actual, expected in zip(result[1], arguments):
                self.assertIs(actual, expected)

        self.assertEqual(
            [name for name, _ in native.renderer.calls],
            [native for _, native in FACADE_TO_NATIVE],
        )

    def test_ray_generation_preserves_arguments_without_geometry(self) -> None:
        native, renderer = self._build()
        arguments = (object(), object(), 7, 9, 1.1, 0.9)

        result = renderer.generate_rays(*arguments)

        self.assertEqual(result[0], "rays")
        self.assertEqual(len(native.ray_calls), 1)
        for actual, expected in zip(native.ray_calls[0], arguments):
            self.assertIs(actual, expected)


if __name__ == "__main__":
    unittest.main()
