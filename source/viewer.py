#!/usr/bin/env python3
"""Optional interactive OptiX viewer for FlaRe checkpoints and triangle meshes."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
VIEWER_OUTPUT = SOURCE_DIR / "viewer_renderer" / "output"
if str(VIEWER_OUTPUT) not in sys.path:
    sys.path.insert(0, str(VIEWER_OUTPUT))
os.environ.setdefault("FLARE_VIEWER_PTX", str(VIEWER_OUTPUT / "viewer_shaders.cu.ptx"))

from checkpoint_io import load_model_checkpoint
from evaluate import load_test_cameras


def triple(values: list[float], name: str) -> np.ndarray:
    if len(values) != 3:
        raise ValueError(f"{name} expects exactly three values")
    return np.asarray(values, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ray trace a FlaRe scene with optional glass Stanford PLY objects."
    )
    parser.add_argument("--scene_path", "--source_path", "-s", required=True, type=Path)
    parser.add_argument("--checkpoint", "-c", required=True, type=Path)
    parser.add_argument("--mesh", action="append", default=[], type=Path,
                        help="Triangle PLY to insert; may be specified repeatedly.")
    parser.add_argument("--output", "-o", type=Path, default=Path("viewer_render.png"))
    parser.add_argument("--camera", type=int, default=0, help="Test-camera index.")
    parser.add_argument("--resolution", "-r", type=int, default=4)
    parser.add_argument("--images", default="images")
    parser.add_argument(
        "--preview_scale", type=float, default=1.0,
        help="Output scale after the dataset --resolution divisor (default: 1.0).",
    )
    parser.add_argument(
        "--supersample", type=float, default=1.0,
        help="Render at this multiple of output resolution, then downsample with Lanczos (default: 1.0).",
    )
    parser.add_argument("--interactive", action="store_true",
                        help="Open the Tk viewer; otherwise save one image and exit.")
    parser.add_argument("--translation", nargs=3, type=float, default=(0.0, 0.0, 0.0),
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--rotation", nargs=3, type=float, default=(0.0, 0.0, 0.0),
                        metavar=("RX", "RY", "RZ"), help="Euler rotation in degrees.")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--center_mesh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--material", choices=("glass", "diffuse"), default="glass")
    parser.add_argument("--color", nargs=3, type=float, default=(0.75, 0.88, 1.0),
                        metavar=("R", "G", "B"))
    # Garden's accepted glass renders use eta=1.12.  Keep the standalone
    # viewer aligned with that scene while allowing explicit overrides.
    parser.add_argument("--ior", type=float, default=1.12)
    parser.add_argument("--diffuse", type=float, default=None)
    parser.add_argument("--shininess", type=float, default=64.0)
    parser.add_argument("--light_color", nargs=3, type=float, default=(1.0, 1.0, 1.0))
    parser.add_argument("--light_direction", nargs=3, type=float, default=(1.0, 1.0, 1.0))
    parser.add_argument("--ambient", nargs=3, type=float, default=(0.06, 0.06, 0.07))
    parser.add_argument("--shadow_multiplier", type=float, default=0.55)
    parser.add_argument("--recursion_depth", type=int, default=8)
    parser.add_argument("--number_of_sides", type=int, default=8)
    parser.add_argument("--ray_threshold", type=float, default=0.01)
    parser.add_argument("--ray_epsilon", type=float, default=1.0e-4)
    parser.add_argument("--bg_color_R", type=float, default=0.0)
    parser.add_argument("--bg_color_G", type=float, default=0.0)
    parser.add_argument("--bg_color_B", type=float, default=0.0)
    parser.add_argument("--move_step", type=float, default=0.1)
    parser.add_argument("--turn_step", type=float, default=3.0)
    return parser.parse_args()


def rotation_matrix_xyz(degrees: np.ndarray) -> np.ndarray:
    x, y, z = np.deg2rad(degrees)
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]], dtype=np.float32)
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]], dtype=np.float32)
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


def load_mesh(path: Path, args: argparse.Namespace, device: torch.device):
    try:
        import open3d as o3d
    except ImportError as error:
        raise ImportError("The optional viewer requires open3d for loading PLY meshes") from error
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise ValueError(f"Mesh has no triangles: {path}")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices, dtype=np.float32).copy()
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32).copy()
    indices = np.asarray(mesh.triangles, dtype=np.int32).copy()
    if args.center_mesh:
        vertices -= 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    rotation = rotation_matrix_xyz(triple(list(args.rotation), "rotation"))
    vertices = (vertices @ rotation.T) * float(args.scale)
    vertices += triple(list(args.translation), "translation")
    normals = normals @ rotation.T
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-8)
    return (
        torch.from_numpy(vertices).to(device).contiguous(),
        torch.from_numpy(indices).to(device).contiguous(),
        torch.from_numpy(normals).to(device).contiguous(),
    )


def prepare_model(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    model = load_model_checkpoint(path, device)
    count = int(model["A"].shape[0])
    if "RGB" not in model:
        model["RGB"] = torch.ones((count, 3), dtype=torch.float32, device=device)
    model["RGB"] = model["RGB"].float().contiguous()
    model["w1_fp16"] = torch.cat(
        (model["w1_uv"], model["w1_v"], model["w1_conditioning"]), dim=1
    ).to(torch.float16).contiguous()
    model["w2_fp16"] = model["w2"].to(torch.float16).contiguous()
    model["w3_fp16"] = model["w3"].to(torch.float16).contiguous()
    model["conditioning_variable_fp16"] = model["conditioning_variable"].to(torch.float16).contiguous()
    return model


class Viewer:
    def __init__(self, args: argparse.Namespace):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        self.args = args
        try:
            import PYOPTIXFLAREVIEWER as viewer_renderer
        except ImportError as error:
            raise ImportError("Build the optional viewer first with ./build_viewer.sh") from error
        self.viewer_renderer = viewer_renderer
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        args.scene_path = args.scene_path.expanduser().resolve()
        args.checkpoint = args.checkpoint.expanduser().resolve()
        args.mesh = [path.expanduser().resolve() for path in args.mesh]
        views = load_test_cameras(args)
        if not 0 <= args.camera < len(views):
            raise ValueError(f"Camera index {args.camera} is outside 0..{len(views) - 1}")
        view = views[args.camera]
        if args.preview_scale <= 0.0:
            raise ValueError("--preview_scale must be positive")
        if args.supersample < 1.0:
            raise ValueError("--supersample must be at least 1.0")
        self.output_width = max(1, round(int(view["width"]) * args.preview_scale))
        self.output_height = max(1, round(int(view["height"]) * args.preview_scale))
        self.width = max(1, round(self.output_width * args.supersample))
        self.height = max(1, round(self.output_height * args.supersample))
        self.fov_x = float(view["fov_x"])
        self.fov_y = float(view["fov_y"])
        self.origin = view["origin"].to(self.device)
        self.right = view["right"].to(self.device)
        self.down = view["down"].to(self.device)
        self.forward = view["forward"].to(self.device)
        self.model = prepare_model(args.checkpoint, self.device)
        self.renderer = self.viewer_renderer.CPyOptiXFLAREVIEWERRenderer(
            args.number_of_sides, 11.3449, self.width * self.height, args.recursion_depth
        )
        self.mesh_handles = []
        instances = []
        hitgroups = []
        for index, mesh_path in enumerate(args.mesh, start=1):
            vertices, triangles, normals = load_mesh(mesh_path, args, self.device)
            handle = self.renderer.CreateMeshInstance(vertices, triangles, normals, index)
            self.mesh_handles.append((handle, vertices, triangles, normals))
            instances.append(handle.GetInstanceAsTensor().unsqueeze(0))
            hitgroups.append(handle.GetHitgroupRecordAsTensor().unsqueeze(0))
        if not instances:
            raise ValueError("At least one --mesh is required")
        self.instances = torch.cat(instances, dim=0).contiguous()
        self.hitgroups = torch.cat(hitgroups, dim=0).contiguous()
        diffuse = args.diffuse
        if diffuse is None:
            diffuse = 0.03 if args.material == "glass" else 0.95
        ior = args.ior if args.material == "glass" else 1.025
        row = torch.tensor([*args.color, diffuse, ior, args.shininess], dtype=torch.float32, device=self.device)
        self.materials = row.unsqueeze(0).repeat(len(instances), 1).contiguous()
        self.normal_matrices = torch.eye(3, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(len(instances),
                                                                                                         1, 1)
        direction = torch.tensor(args.light_direction, dtype=torch.float32, device=self.device)
        direction /= torch.linalg.vector_norm(direction).clamp_min(1.0e-8)
        self.lights = torch.tensor([[*args.light_color, 0.0, 0.0, 0.0]], dtype=torch.float32, device=self.device)
        self.lights[0, 3:] = direction
        self.renderer.SetGeometry(
            self.model["m"], torch.exp(self.model["s"]), self.model["q"],
            torch.sigmoid(self.model["A"]), 1.0 + torch.nn.functional.softplus(self.model["k"]),
            self.instances, self.hitgroups,
        )

    @torch.no_grad()
    def render(self) -> Image.Image:
        pixel_count = self.width * self.height
        origins = self.origin.repeat(pixel_count, 1)
        directions = self.viewer_renderer.GenerateRays(
            self.right, self.down, self.forward, self.width, self.height, self.fov_x, self.fov_y
        ).reshape(pixel_count, 3)
        directions /= torch.linalg.vector_norm(directions, dim=1, keepdim=True)
        image = torch.zeros((pixel_count, 3), dtype=torch.float32, device=self.device)
        self.renderer.Forward(
            self.model["conditioning_variable_fp16"], self.model["features"], self.model["RGB"],
            self.model["w1_fp16"], self.model["b1"], self.model["w2_fp16"], self.model["b2"],
            self.model["w3_fp16"], self.model["b3"], origins, directions, image,
            self.args.bg_color_R, self.args.bg_color_G, self.args.bg_color_B,
            self.model["m"], self.model["s"], self.model["q"], self.model["A"], self.model["k"],
            self.args.ray_threshold, self.materials, self.normal_matrices, self.lights,
            self.args.ray_epsilon, self.args.shadow_multiplier, *self.args.ambient,
        )
        pixels = image.clamp(0.0, 1.0).reshape(self.height, self.width, 3).mul(255).byte().cpu().numpy()
        result = Image.fromarray(pixels, mode="RGB")
        if (self.width, self.height) != (self.output_width, self.output_height):
            result = result.resize(
                (self.output_width, self.output_height), Image.Resampling.LANCZOS
            )
        return result

    def save(self) -> None:
        image = self.render()
        self.args.output.parent.mkdir(parents=True, exist_ok=True)
        image.save(self.args.output)
        print(
            f"Saved {self.args.output} ({self.output_width}x{self.output_height}; "
            f"rendered {self.width}x{self.height})",
            flush=True,
        )

    def run_interactive(self) -> None:
        import tkinter as tk
        from PIL import ImageTk
        root = tk.Tk()
        root.title("FlaRe OptiX viewer")
        canvas = tk.Label(root)
        canvas.pack()
        state = {"photo": None}

        def redraw() -> None:
            state["photo"] = ImageTk.PhotoImage(self.render())
            canvas.configure(image=state["photo"])
            root.update_idletasks()

        def rotate(axis: torch.Tensor, angle_degrees: float, vector: torch.Tensor) -> torch.Tensor:
            angle = math.radians(angle_degrees)
            axis = axis / torch.linalg.vector_norm(axis).clamp_min(1.0e-8)
            return (vector * math.cos(angle) + torch.linalg.cross(axis, vector) * math.sin(angle)
                    + axis * torch.dot(axis, vector) * (1.0 - math.cos(angle)))

        def on_key(event) -> None:
            key = event.keysym.lower()
            if key == "w":
                self.origin += self.forward * self.args.move_step
            elif key == "s":
                self.origin -= self.forward * self.args.move_step
            elif key == "a":
                self.origin -= self.right * self.args.move_step
            elif key == "d":
                self.origin += self.right * self.args.move_step
            elif key == "space":
                self.origin -= self.down * self.args.move_step
            elif key == "c":
                self.origin += self.down * self.args.move_step
            elif key in ("left", "right"):
                sign = -1.0 if key == "left" else 1.0
                self.right = rotate(-self.down, sign * self.args.turn_step, self.right)
                self.forward = rotate(-self.down, sign * self.args.turn_step, self.forward)
            elif key in ("up", "down"):
                sign = -1.0 if key == "up" else 1.0
                self.down = rotate(self.right, sign * self.args.turn_step, self.down)
                self.forward = rotate(self.right, sign * self.args.turn_step, self.forward)
            elif key == "r":
                self.save()
            elif key in ("escape", "q"):
                root.destroy(); return
            else:
                return
            redraw()

        root.bind("<KeyPress>", on_key)
        redraw()
        root.mainloop()


def main() -> int:
    args = parse_args()
    viewer = Viewer(args)
    if args.interactive:
        viewer.run_interactive()
    else:
        viewer.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
