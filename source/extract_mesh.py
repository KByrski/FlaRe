#!/usr/bin/env python3
"""Extract a colored TSDF mesh from a selected FlaRe checkpoint.

This follows the bounded-scene extraction path from the official 2DGS code:
render the training cameras, integrate RGB and surface depth into an Open3D
ScalableTSDFVolume, extract its zero level set, and remove small connected
components. Expected depth is the 2DGS default (depth_ratio=0); hard median
depth (depth_ratio=1) remains selectable for bounded-scene experiments.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import NamedTuple

import numpy as np
try:
    import open3d as o3d
except ImportError:
    o3d = None
from PIL import Image
import torch


SOURCE_DIR = Path(__file__).resolve().parent
RENDERER_DIR = SOURCE_DIR / "renderer" / "output"

if str(RENDERER_DIR) not in sys.path:
    sys.path.insert(0, str(RENDERER_DIR))

if os.name == "nt":
    os.add_dll_directory(str(Path(torch.__file__).resolve().parent / "lib"))

import PYOPTIXFLARERENDERER as flare_renderer

from checkpoint_io import load_model_checkpoint

from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.depth_utils import (
    _depth_visualization,
    median_hits_to_depth_and_normal,
    ray_parameters_to_depth_and_normal,
)


class MeshCamera(NamedTuple):
    name: str
    width: int
    height: int
    fov_x: float
    fov_y: float
    origin: torch.Tensor
    right: torch.Tensor
    down: torch.Tensor
    forward: torch.Tensor
    world_to_camera: np.ndarray
    intrinsic: o3d.camera.PinholeCameraIntrinsic
    gt_alpha: np.ndarray | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a 2DGS-style bounded TSDF mesh from a FlaRe checkpoint."
    )
    parser.add_argument(
        "--scene_path",
        "--source_path",
        "-s",
        required=True,
        type=Path,
        help="Training scene directory (COLMAP or Blender format).",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        required=True,
        type=Path,
        help="Checkpoint file, for example output/6/checkpoints/6000/iter_6000.checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        help="Output directory. Defaults to <experiment>/meshes/iteration_<N>.",
    )
    parser.add_argument(
        "--renderer",
        choices=("flare", "base", "both"),
        default="flare",
        help="Color renderer used for RGB-D fusion (default: flare).",
    )
    parser.add_argument(
        "--depth_mode",
        choices=("expected", "median"),
        default="expected",
        help="2DGS surface depth: expected=depth_ratio 0, median=depth_ratio 1.",
    )
    parser.add_argument(
        "--resolution",
        "-r",
        default=-1,
        type=int,
        help="Same resolution semantics as train.py: -1, 1, 2, 4, 8, or target width.",
    )
    parser.add_argument("--images", default="images")
    parser.add_argument("--bg_color_R", type=float, default=0.0)
    parser.add_argument("--bg_color_G", type=float, default=0.0)
    parser.add_argument("--bg_color_B", type=float, default=0.0)
    parser.add_argument("--number_of_sides", type=int, default=8)
    parser.add_argument(
        "--ray_termination_T_threshold_inference", type=float, default=0.01
    )
    parser.add_argument(
        "--alpha_threshold",
        type=float,
        default=0.01,
        help="Discard rendered depths below this accumulated alpha (default: 0.01).",
    )
    parser.add_argument(
        "--mesh_res",
        type=int,
        default=1024,
        help="Used to derive voxel size as depth_trunc/mesh_res (2DGS default: 1024).",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=-1.0,
        help="TSDF voxel size; negative derives it from depth_trunc/mesh_res.",
    )
    parser.add_argument(
        "--sdf_trunc",
        type=float,
        default=-1.0,
        help="TSDF truncation; negative uses 5*voxel_size like 2DGS.",
    )
    parser.add_argument(
        "--depth_trunc",
        type=float,
        default=-1.0,
        help="Maximum camera-space depth; negative uses twice the estimated camera radius.",
    )
    parser.add_argument(
        "--num_cluster",
        type=int,
        default=50,
        help="Keep at most this many largest connected components; 0 disables filtering.",
    )
    parser.add_argument(
        "--min_triangles",
        type=int,
        default=50,
        help="Always remove connected components smaller than this (default: 50).",
    )
    parser.add_argument(
        "--view_stride",
        type=int,
        default=1,
        help="Use every Nth training camera (default: every camera).",
    )
    parser.add_argument(
        "--max_views",
        type=int,
        default=-1,
        help="Optional maximum number of evenly sampled training views.",
    )
    parser.add_argument(
        "--no_gt_mask",
        action="store_true",
        help="Do not apply an available Blender ground-truth alpha mask.",
    )
    parser.add_argument(
        "--save_renders",
        action="store_true",
        help="Save the RGB, depth, and alpha images used for TSDF fusion.",
    )
    return parser.parse_args()


def default_output_dir(checkpoint_path: Path, iteration: int) -> Path:
    # Support rolling checkpoints and historical per-iteration subdirectories.
    if checkpoint_path.parent.name == "checkpoints":
        experiment_dir = checkpoint_path.parent.parent
    elif checkpoint_path.parent.parent.name == "checkpoints":
        experiment_dir = checkpoint_path.parent.parent.parent
    else:
        experiment_dir = checkpoint_path.parent
    suffix = f"iteration_{iteration:06d}" if iteration >= 0 else checkpoint_path.stem
    return experiment_dir / "meshes" / suffix


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[dict[str, torch.Tensor], int, float]:
    model = load_model_checkpoint(checkpoint_path, device)
    iteration_match = re.search(r"iter_(\d+)\.checkpoint$", checkpoint_path.name)
    iteration = int(iteration_match.group(1)) if iteration_match else -1
    training_time = float(model.pop("training_time_seconds"))

    model["has_base_rgb"] = "RGB" in model
    if model["has_base_rgb"]:
        rgb_for_abi = model["RGB"]
    else:
        # Legacy FlaRe predates the compound color multiplier. The current ABI
        # multiplies decoder RGB by RGBA.rgb, so ones reproduce the old renderer.
        rgb_for_abi = torch.ones((model["A"].shape[0], 3), device=device)
    model["RGBA"] = torch.cat((rgb_for_abi, model["A"]), dim=1).contiguous()
    model["w1_fp16"] = torch.cat(
        (model["w1_uv"], model["w1_v"], model["w1_conditioning"]), dim=1
    ).to(torch.float16).contiguous()
    model["w2_fp16"] = model["w2"].to(torch.float16).contiguous()
    model["w3_fp16"] = model["w3"].to(torch.float16).contiguous()
    model["conditioning_variable_fp16"] = model["conditioning_variable"].to(
        torch.float16
    ).contiguous()
    gc.collect()
    return model, iteration, training_time


def resolved_size(original_width: int, original_height: int, resolution: int) -> tuple[int, int]:
    if resolution in (1, 2, 4, 8):
        return round(original_width / resolution), round(original_height / resolution)
    if resolution == -1:
        downsample = original_width / 1600.0 if original_width > 1600 else 1.0
    elif resolution > 0:
        downsample = original_width / float(resolution)
    else:
        raise ValueError("--resolution must be -1, 1, 2, 4, 8, or a positive target width")
    return int(original_width / downsample), int(original_height / downsample)


def load_gt_alpha(image_path: str, width: int, height: int) -> np.ndarray | None:
    with Image.open(image_path) as image:
        if "A" not in image.getbands():
            return None
        alpha = image.getchannel("A").resize((width, height), Image.Resampling.BILINEAR)
        return np.asarray(alpha, dtype=np.float32) / 255.0


def load_training_cameras(args: argparse.Namespace) -> list[MeshCamera]:
    scene_path = args.scene_path
    if (scene_path / "sparse").is_dir():
        scene_info = sceneLoadTypeCallbacks["Colmap"](
            str(scene_path), args.images, True
        )
    elif (scene_path / "transforms_train.json").is_file():
        scene_info = sceneLoadTypeCallbacks["Blender"](
            str(scene_path),
            args.bg_color_R,
            args.bg_color_G,
            args.bg_color_B,
            True,
        )
    else:
        raise ValueError(
            f"Cannot recognize scene format in {scene_path}; expected sparse/ or transforms_train.json"
        )

    camera_infos = list(scene_info.train_cameras)[:: args.view_stride]
    if args.max_views > 0 and len(camera_infos) > args.max_views:
        indices = np.linspace(0, len(camera_infos) - 1, args.max_views, dtype=np.int64)
        camera_infos = [camera_infos[int(index)] for index in indices]
    if not camera_infos:
        raise ValueError(f"No training cameras found in {scene_path}")

    cameras: list[MeshCamera] = []
    for index, camera in enumerate(camera_infos):
        original_width, original_height = camera.image.size
        width, height = resolved_size(original_width, original_height, args.resolution)

        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :3] = np.asarray(camera.R, dtype=np.float64).T
        world_to_camera[:3, 3] = np.asarray(camera.T, dtype=np.float64)
        camera_to_world = np.linalg.inv(world_to_camera)

        fx = width / (2.0 * math.tan(float(camera.FovX) / 2.0))
        fy = height / (2.0 * math.tan(float(camera.FovY) / 2.0))
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, (width - 1.0) / 2.0, (height - 1.0) / 2.0
        )
        gt_alpha = None
        if not args.no_gt_mask:
            gt_alpha = load_gt_alpha(camera.image_path, width, height)

        cameras.append(
            MeshCamera(
                name=camera.image_name or f"view_{index:04d}",
                width=width,
                height=height,
                fov_x=float(camera.FovX),
                fov_y=float(camera.FovY),
                origin=torch.from_numpy(camera_to_world[:3, 3].astype(np.float32)),
                right=torch.from_numpy(camera_to_world[:3, 0].astype(np.float32)),
                down=torch.from_numpy(camera_to_world[:3, 1].astype(np.float32)),
                forward=torch.from_numpy(camera_to_world[:3, 2].astype(np.float32)),
                world_to_camera=world_to_camera,
                intrinsic=intrinsic,
                gt_alpha=gt_alpha,
            )
        )

    del scene_info, camera_infos
    gc.collect()
    return cameras


def focus_point(cameras: list[MeshCamera]) -> np.ndarray:
    camera_to_world = np.stack(
        [np.linalg.inv(camera.world_to_camera) for camera in cameras]
    )
    # Official 2DGS converts COLMAP axes to OpenGL before finding the focal axes.
    poses = camera_to_world[:, :3, :] @ np.diag([1.0, -1.0, -1.0, 1.0])
    directions = poses[:, :3, 2:3]
    origins = poses[:, :3, 3:4]
    projection = np.eye(3) - directions * np.transpose(directions, (0, 2, 1))
    normal_matrix = np.transpose(projection, (0, 2, 1)) @ projection
    try:
        return np.linalg.solve(
            normal_matrix.mean(0), (normal_matrix @ origins).mean(0)[:, 0]
        )
    except np.linalg.LinAlgError:
        return camera_to_world[:, :3, 3].mean(axis=0)


def estimate_camera_radius(cameras: list[MeshCamera]) -> tuple[np.ndarray, float]:
    center = focus_point(cameras)
    centers = np.stack(
        [np.linalg.inv(camera.world_to_camera)[:3, 3] for camera in cameras]
    )
    radius = float(np.linalg.norm(centers - center, axis=1).min())
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"Invalid camera radius estimated from training cameras: {radius}")
    return center, radius


@torch.no_grad()
def render_rgbd(
    mode: str,
    depth_mode: str,
    renderer: object,
    camera: MeshCamera,
    model: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pixel_count = camera.width * camera.height
    right = camera.right.to(device)
    down = camera.down.to(device)
    forward = camera.forward.to(device)
    origins = camera.origin.to(device).repeat(pixel_count, 1)
    directions = flare_renderer.GenerateRays(
        right,
        down,
        forward,
        camera.width,
        camera.height,
        camera.fov_x,
        camera.fov_y,
    ).reshape(pixel_count, 3)
    directions = torch.nn.functional.normalize(directions, dim=1)

    rgb = torch.zeros((pixel_count, 3), dtype=torch.float32, device=device)
    normal = torch.zeros_like(rgb)
    alpha = torch.zeros((pixel_count,), dtype=torch.float32, device=device)
    expected_depth_numerator = torch.zeros_like(alpha)
    depth_and_index = None
    if depth_mode == "median":
        depth_and_index = torch.empty((pixel_count, 2), dtype=torch.float32, device=device)

    if mode == "base":
        renderer.Forward_inference_base_with_geometry(
            origins,
            directions,
            rgb,
            normal,
            alpha,
            expected_depth_numerator,
            args.bg_color_R,
            args.bg_color_G,
            args.bg_color_B,
            model["m"],
            model["s"],
            model["q"],
            model["RGBA"],
            model["k"],
            args.ray_termination_T_threshold_inference,
        )
        if depth_and_index is not None:
            renderer.GetMedianDepth_base(
                origins,
                directions,
                model["m"],
                model["s"],
                model["q"],
                model["RGBA"],
                model["k"],
                args.ray_termination_T_threshold_inference,
                depth_and_index,
            )
    elif mode == "flare":
        # FlaRe supplies TSDF color. Geometry is rendered separately with the
        # explicit opacity only, matching the paper setting alpha_MLP = 1.
        renderer.Forward_inference(
            model["conditioning_variable_fp16"],
            model["features"],
            model["w1_fp16"],
            model["b1"],
            model["w2_fp16"],
            model["b2"],
            model["w3_fp16"],
            model["b3"],
            origins,
            directions,
            rgb,
            args.bg_color_R,
            args.bg_color_G,
            args.bg_color_B,
            model["m"],
            model["s"],
            model["q"],
            model["RGBA"],
            model["k"],
            args.ray_termination_T_threshold_inference,
        )
        geometry_rgb = torch.zeros_like(rgb)
        renderer.Forward_inference_base_with_geometry(
            origins,
            directions,
            geometry_rgb,
            normal,
            alpha,
            expected_depth_numerator,
            args.bg_color_R,
            args.bg_color_G,
            args.bg_color_B,
            model["m"],
            model["s"],
            model["q"],
            model["RGBA"],
            model["k"],
            args.ray_termination_T_threshold_inference,
        )
        if depth_and_index is not None:
            renderer.GetMedianDepth_base(
                origins,
                directions,
                model["m"],
                model["s"],
                model["q"],
                model["RGBA"],
                model["k"],
                args.ray_termination_T_threshold_inference,
                depth_and_index,
            )
    else:
        raise ValueError(f"Unknown renderer mode: {mode}")

    if depth_and_index is None:
        expected_t = expected_depth_numerator / alpha.clamp_min(1.0e-8)
        depth, depth_normal, valid = ray_parameters_to_depth_and_normal(
            expected_t,
            alpha >= args.alpha_threshold,
            origins,
            directions,
            forward,
            camera.height,
            camera.width,
        )
    else:
        depth, depth_normal, valid, _ = median_hits_to_depth_and_normal(
            depth_and_index,
            origins,
            directions,
            forward,
            camera.height,
            camera.width,
        )
        valid &= alpha.reshape(camera.height, camera.width) >= args.alpha_threshold
        depth = torch.where(valid, depth, torch.zeros_like(depth))

    rgb_np = (
        rgb.reshape(camera.height, camera.width, 3)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    depth_np = depth.reshape(camera.height, camera.width).float().cpu().numpy()
    alpha_np = alpha.reshape(camera.height, camera.width).float().cpu().numpy()
    rendered_normal_np = normal.reshape(camera.height, camera.width, 3).float().cpu().numpy()
    depth_normal_np = depth_normal.reshape(camera.height, camera.width, 3).float().cpu().numpy()
    if camera.gt_alpha is not None:
        masked = camera.gt_alpha < 0.5
        depth_np[masked] = 0.0
        rendered_normal_np[masked] = 0.0
        depth_normal_np[masked] = 0.0
    return rgb_np, depth_np, alpha_np, rendered_normal_np, depth_normal_np


def normal_visualization(normal: np.ndarray, valid: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    unit = np.divide(normal, np.maximum(length, np.finfo(np.float32).eps))
    output = np.zeros(normal.shape, dtype=np.uint8)
    mask = valid & (length[..., 0] > 0.0)
    output[mask] = np.clip((unit[mask] * 0.5 + 0.5) * 255.0, 0.0, 255.0)
    return output


def save_rgbd_renders(
    directory: Path,
    index: int,
    camera: MeshCamera,
    rgb: np.ndarray,
    depth: np.ndarray,
    alpha: np.ndarray,
    rendered_normal: np.ndarray,
    depth_normal: np.ndarray,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{index:04d}_{Path(camera.name).stem}"
    Image.fromarray(rgb, mode="RGB").save(directory / f"{stem}_rgb.png")
    Image.fromarray(depth.astype(np.float32), mode="F").save(
        directory / f"{stem}_depth.tiff"
    )
    Image.fromarray(_depth_visualization(depth, depth > 0.0), mode="RGB").save(
        directory / f"{stem}_depth.png"
    )
    Image.fromarray(
        np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8), mode="L"
    ).save(directory / f"{stem}_alpha.png")
    Image.fromarray(
        normal_visualization(rendered_normal, alpha > 0.01), mode="RGB"
    ).save(directory / f"{stem}_rend_normal.png")
    Image.fromarray(
        normal_visualization(depth_normal, depth > 0.0), mode="RGB"
    ).save(directory / f"{stem}_surf_normal.png")


def post_process_mesh(
    mesh: o3d.geometry.TriangleMesh, clusters_to_keep: int, min_triangles: int
) -> o3d.geometry.TriangleMesh:
    if clusters_to_keep <= 0:
        return mesh
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    if cluster_n_triangles.size == 0:
        return mesh

    keep_count = min(clusters_to_keep, cluster_n_triangles.size)
    size_cutoff = int(np.sort(cluster_n_triangles)[-keep_count])
    size_cutoff = max(size_cutoff, min_triangles)
    remove = cluster_n_triangles[triangle_clusters] < size_cutoff
    mesh.remove_triangles_by_mask(remove)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    return mesh


@torch.no_grad()
def extract_one_mesh(
    mode: str,
    renderer: object,
    cameras: list[MeshCamera],
    model: dict[str, torch.Tensor],
    args: argparse.Namespace,
    output_dir: Path,
    voxel_size: float,
    sdf_trunc: float,
    depth_trunc: float,
    device: torch.device,
) -> dict[str, object]:
    print(f"\n[{mode}] Creating Open3D ScalableTSDFVolume", flush=True)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    render_dir = output_dir / "renders" / mode

    integrated = 0
    nonzero_depth_fractions: list[float] = []
    for index, camera in enumerate(cameras):
        rgb, depth, alpha, rendered_normal, depth_normal = render_rgbd(
            mode, args.depth_mode, renderer, camera, model, args, device
        )
        depth[(depth <= 0.0) | (depth >= depth_trunc) | ~np.isfinite(depth)] = 0.0
        valid_fraction = float(np.count_nonzero(depth) / depth.size)
        nonzero_depth_fractions.append(valid_fraction)

        if args.save_renders:
            save_rgbd_renders(render_dir, index, camera, rgb, depth, alpha, rendered_normal, depth_normal)
        if np.any(depth > 0.0):
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(rgb)),
                o3d.geometry.Image(np.ascontiguousarray(depth.astype(np.float32))),
                depth_scale=1.0,
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
            )
            volume.integrate(rgbd, camera.intrinsic, camera.world_to_camera)
            integrated += 1

        print(
            f"[{mode}] {index + 1:4d}/{len(cameras)} {camera.name}: "
            f"valid depth {100.0 * valid_fraction:5.1f}%",
            flush=True,
        )

    if integrated == 0:
        raise RuntimeError(
            f"{mode} produced no valid depth inside depth_trunc={depth_trunc}"
        )

    print(f"[{mode}] Extracting TSDF zero level set...", flush=True)
    raw_mesh = volume.extract_triangle_mesh()
    raw_mesh.compute_vertex_normals()
    raw_path = output_dir / f"{mode}_fuse.ply"
    if not o3d.io.write_triangle_mesh(str(raw_path), raw_mesh):
        raise RuntimeError(f"Open3D failed to write {raw_path}")

    raw_vertices = len(raw_mesh.vertices)
    raw_triangles = len(raw_mesh.triangles)
    post_mesh = post_process_mesh(
        raw_mesh, args.num_cluster, args.min_triangles
    )
    post_mesh.compute_vertex_normals()
    post_path = output_dir / f"{mode}_fuse_post.ply"
    if not o3d.io.write_triangle_mesh(str(post_path), post_mesh):
        raise RuntimeError(f"Open3D failed to write {post_path}")

    result = {
        "renderer": mode,
        "raw_mesh": str(raw_path),
        "post_processed_mesh": str(post_path),
        "raw_vertices": raw_vertices,
        "raw_triangles": raw_triangles,
        "post_vertices": len(post_mesh.vertices),
        "post_triangles": len(post_mesh.triangles),
        "integrated_views": integrated,
        "mean_valid_depth_fraction": float(np.mean(nonzero_depth_fractions)),
    }
    print(
        f"[{mode}] Saved {post_path} "
        f"({result['post_vertices']} vertices, {result['post_triangles']} triangles)",
        flush=True,
    )
    return result


def main() -> int:
    args = parse_args()
    if o3d is None:
        raise RuntimeError(
            "Open3D is required for TSDF extraction; install the requirements again."
        )
    args.scene_path = args.scene_path.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.view_stride < 1:
        raise ValueError("--view_stride must be at least 1")
    if args.mesh_res <= 0:
        raise ValueError("--mesh_res must be positive")
    if not 0.0 <= args.alpha_threshold <= 1.0:
        raise ValueError("--alpha_threshold must be in [0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    cameras = load_training_cameras(args)
    model, iteration, training_time = load_model(args.checkpoint, device)
    if args.renderer in ("base", "both") and not model["has_base_rgb"]:
        raise ValueError(
            "Base mesh extraction requires an 18-entry checkpoint containing RGB."
        )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir(args.checkpoint, iteration)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    center, camera_radius = estimate_camera_radius(cameras)
    depth_trunc = args.depth_trunc if args.depth_trunc > 0.0 else 2.0 * camera_radius
    voxel_size = (
        args.voxel_size if args.voxel_size > 0.0 else depth_trunc / args.mesh_res
    )
    sdf_trunc = args.sdf_trunc if args.sdf_trunc > 0.0 else 5.0 * voxel_size
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print(f"Iteration: {iteration}", flush=True)
    print(f"Training cameras: {len(cameras)}", flush=True)
    print(f"Estimated focus point: {center.tolist()}", flush=True)
    print(f"Estimated camera radius: {camera_radius:.6f}", flush=True)
    print(f"depth_trunc={depth_trunc:.6f}", flush=True)
    print(f"voxel_size={voxel_size:.8f}", flush=True)
    print(f"sdf_trunc={sdf_trunc:.8f}", flush=True)
    print(f"depth_mode={args.depth_mode}", flush=True)

    max_batch_size = max(camera.width * camera.height for camera in cameras)
    renderer = flare_renderer.CPyOptiXFLARERenderer(
        args.number_of_sides, 11.3449, max_batch_size
    )
    renderer.SetGeometry(
        model["m"],
        torch.exp(model["s"]),
        model["q"],
        torch.sigmoid(model["A"]),
        1.0 + torch.nn.functional.softplus(model["k"]),
    )
    torch.cuda.synchronize()

    modes = ("base", "flare") if args.renderer == "both" else (args.renderer,)
    results = []
    for mode in modes:
        results.append(
            extract_one_mesh(
                mode,
                renderer,
                cameras,
                model,
                args,
                output_dir,
                voxel_size,
                sdf_trunc,
                depth_trunc,
                device,
            )
        )

    metadata = {
        "method": "2DGS bounded ScalableTSDFVolume fusion",
        "checkpoint": str(args.checkpoint),
        "iteration": iteration,
        "checkpoint_training_time_seconds": training_time,
        "scene_path": str(args.scene_path),
        "output_dir": str(output_dir),
        "depth_mode": args.depth_mode,
        "surface_depth_ratio": 0.0 if args.depth_mode == "expected" else 1.0,
        "geometry_opacity": "explicit primitive opacity only (alpha_MLP = 1)",
        "camera_count": len(cameras),
        "focus_point": center.tolist(),
        "camera_radius": camera_radius,
        "depth_trunc": depth_trunc,
        "voxel_size": voxel_size,
        "sdf_trunc": sdf_trunc,
        "alpha_threshold": args.alpha_threshold,
        "ground_truth_alpha_mask": not args.no_gt_mask,
        "num_cluster": args.num_cluster,
        "min_triangles": args.min_triangles,
        "resolution": args.resolution,
        "results": results,
        "gpu": {
            "name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    metadata_path = output_dir / "mesh_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Saved metadata: {metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
