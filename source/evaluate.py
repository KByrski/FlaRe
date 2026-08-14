#!/usr/bin/env python3
"""Evaluate the best FLARE checkpoint on a scene's test split."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import torch

if os.name == "nt":
    os.add_dll_directory(str(Path(torch.__file__).resolve().parent / "lib"))

from checkpoint_io import load_training_checkpoint
from evaluation_service import (
    EvaluationOptions,
    EvaluationView,
    calculate_metrics,
    render_dataset,
)
from lpipsPyTorch.modules.lpips import LPIPS
from renderer_facade import FlaReRenderer
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos


MIB = 1024.0 * 1024.0
PSNR_LINE = re.compile(
    r"^\s*(?P<iteration>\d+)\s*:\s*"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate base and FlaRe using the checkpoint with the best test FlaRe PSNR."
    )
    parser.add_argument(
        "--scene_path",
        "--source_path",
        "-s",
        dest="scene_path",
        required=True,
        type=Path,
        help="Training scene directory (COLMAP or Blender format).",
    )
    parser.add_argument(
        "--model_path",
        "-m",
        required=True,
        type=Path,
        help="Model directory containing stats/ and checkpoints/.",
    )
    parser.add_argument(
        "--resolution",
        "-r",
        default=-1,
        type=int,
        help="Resolution setting with the same semantics as train.py (default: -1).",
    )
    parser.add_argument(
        "--images",
        default="images",
        help="COLMAP image directory name relative to the scene (default: images).",
    )
    parser.add_argument("--bg_color_R", type=float, default=0.0)
    parser.add_argument("--bg_color_G", type=float, default=0.0)
    parser.add_argument("--bg_color_B", type=float, default=0.0)
    parser.add_argument("--number_of_sides", type=int, default=8)
    parser.add_argument(
        "--ray_termination_T_threshold_inference", type=float, default=0.01
    )
    parser.add_argument(
        "--skip_metrics",
        action="store_true",
        help="Render the complete test set without calculating PSNR/SSIM/LPIPS.",
    )
    return parser.parse_args()


def find_best_iteration(model_path: Path) -> tuple[int, float, Path, Path]:
    metadata_path = model_path / "checkpoints" / "checkpoint_metadata.json"
    rolling_checkpoint = model_path / "checkpoints" / "best.checkpoint"
    if metadata_path.is_file() and rolling_checkpoint.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        best = metadata.get("best", {})
        if "iteration" in best and "value" in best:
            metric_name = str(best.get("metric", "PSNR_test_FlaRe"))
            return (
                int(best["iteration"]),
                float(best["value"]),
                model_path / "stats" / (metric_name + ".txt"),
                rolling_checkpoint,
            )

    stats_path = model_path / "stats" / "PSNR_test_FlaRe.txt"
    if not stats_path.is_file():
        raise FileNotFoundError(f"FlaRe PSNR history not found: {stats_path}")

    values: list[tuple[int, float]] = []
    invalid_lines: list[int] = []
    for line_number, line in enumerate(stats_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        match = PSNR_LINE.match(line)
        if match is None:
            invalid_lines.append(line_number)
            continue
        values.append((int(match.group("iteration")), float(match.group("value"))))

    if invalid_lines:
        lines = ", ".join(map(str, invalid_lines))
        raise ValueError(f"Invalid rows in {stats_path} at line(s): {lines}")
    if not values:
        raise ValueError(f"No PSNR measurements found in: {stats_path}")

    # Prefer the newest checkpoint when multiple iterations have the same best PSNR.
    iteration, logged_psnr = max(values, key=lambda item: (item[1], item[0]))
    checkpoint_path = (
        model_path
        / "checkpoints"
        / str(iteration)
        / f"iter_{iteration}.checkpoint"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint selected by {stats_path.name} does not exist: {checkpoint_path}"
        )

    return iteration, logged_psnr, stats_path, checkpoint_path


def load_test_cameras(args: argparse.Namespace) -> list[EvaluationView]:
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

    camera_args = SimpleNamespace(
        resolution=args.resolution,
        data_device="cuda",
        bg_color_R=args.bg_color_R,
        bg_color_G=args.bg_color_G,
        bg_color_B=args.bg_color_B,
    )
    cameras = cameraList_from_camInfos(scene_info.test_cameras, 1.0, camera_args)
    if not cameras:
        raise ValueError(f"The scene has no test cameras: {scene_path}")

    views: list[EvaluationView] = []
    for index, camera in enumerate(cameras):
        views.append(
            EvaluationView(
                name=camera.image_name or f"view_{index:04d}",
                width=camera.image_width,
                height=camera.image_height,
                fov_x=float(camera.FoVx),
                fov_y=float(camera.FoVy),
                origin=torch.tensor(-camera.R @ camera.T, dtype=torch.float32),
                right=torch.tensor(
                    camera.R.transpose(1, 0)[0, :], dtype=torch.float32
                ),
                down=torch.tensor(
                    camera.R.transpose(1, 0)[1, :], dtype=torch.float32
                ),
                forward=torch.tensor(
                    camera.R.transpose(1, 0)[2, :], dtype=torch.float32
                ),
                ground_truth=camera.original_image.detach().cpu().contiguous(),
            )
        )

    del cameras, scene_info
    gc.collect()
    torch.cuda.empty_cache()
    return views


def load_model(checkpoint_path: Path, device: torch.device) -> GaussianModel:
    state = load_training_checkpoint(checkpoint_path, device)
    model = GaussianModel.from_model_tensors(state.model, requires_grad=False)
    model.iteration = state.iteration
    model.training_time = state.training_time_seconds
    return model


def process_gpu_memory_mb() -> tuple[float, str]:
    """Return this process' current GPU memory, including native CUDA/OptiX allocations."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        current_pid = os.getpid()
        memory = 0.0
        found = False
        for row in result.stdout.splitlines():
            fields = [field.strip() for field in row.split(",")]
            if len(fields) >= 2 and int(fields[0]) == current_pid:
                memory += float(fields[1])
                found = True
        if found:
            return memory, "nvidia-smi process used_memory"
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError):
        pass

    return torch.cuda.memory_reserved() / MIB, "torch.cuda.memory_reserved fallback"



def main() -> int:
    args = parse_args()
    args.scene_path = args.scene_path.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    iteration, logged_psnr, psnr_path, checkpoint_path = find_best_iteration(
        args.model_path
    )
    metric_name = psnr_path.stem
    print(
        f"Selected iteration {iteration} with logged {metric_name} {logged_psnr:.8f}",
        flush=True,
    )
    print(f"Checkpoint: {checkpoint_path}", flush=True)

    views = load_test_cameras(args)
    model = load_model(checkpoint_path, device)
    gaussian_count = int(model.m.shape[0])
    has_base_rgb = model.RGB is not None

    # The 17-entry historical schema contains only FlaRe appearance. Ones are
    # the neutral multiplier for the compound RGB ABI, so FlaRe remains exact;
    # the unavailable explicit base renderer is skipped rather than fabricated.
    max_batch_size = max(view.width * view.height for view in views)
    renderer = FlaReRenderer(args.number_of_sides, 11.3449, max_batch_size)
    renderer.sync_geometry(model)
    torch.cuda.synchronize()
    resident_memory_mb, resident_memory_source = process_gpu_memory_mb()

    renders_path = args.model_path / "renders" / f"evaluation_{iteration}"
    options = EvaluationOptions(
        background=(args.bg_color_R, args.bg_color_G, args.bg_color_B),
        ray_termination_threshold=args.ray_termination_T_threshold_inference,
    )
    base_images: list[torch.Tensor] = []
    if has_base_rgb:
        base_images, base_results = render_dataset(
            "base", renderer, views, model, options, device, renders_path,
            keep_images=not args.skip_metrics,
            memory_reader=process_gpu_memory_mb,
        )
        base_results["available"] = True
    else:
        print(
            "Legacy 17-entry checkpoint: base RGB is unavailable; "
            "rendering FlaRe only.",
            flush=True,
        )
        base_results = {
            "available": False,
            "reason": "17-entry FlaRe-only checkpoint has no explicit base RGB",
        }
    flare_images, flare_results = render_dataset(
        "FlaRe", renderer, views, model, options, device, renders_path,
        keep_images=not args.skip_metrics,
        memory_reader=process_gpu_memory_mb,
    )
    flare_results["available"] = True

    if not args.skip_metrics:
        print("Loading LPIPS-VGG and calculating image metrics...", flush=True)
        lpips_vgg = LPIPS("vgg").to(device).eval()
        if has_base_rgb:
            base_results.update(calculate_metrics(base_images, views, lpips_vgg, device))
        flare_results.update(calculate_metrics(flare_images, views, lpips_vgg, device))

    test_resolutions = sorted(
        {f"{view.width}x{view.height}" for view in views}
    )
    evaluation = {
        "scene_path": str(args.scene_path),
        "model_path": str(args.model_path),
        "resolution_argument": args.resolution,
        "test_resolutions": test_resolutions,
        "test_view_count": len(views),
        "selection": {
            "metric": metric_name,
            "metric_file": str(psnr_path),
            "iteration": iteration,
            "logged_psnr": logged_psnr,
            "checkpoint": str(checkpoint_path),
        },
        "number_of_gaussians": gaussian_count,
        "checkpoint_schema_entries": 18 if has_base_rgb else 17,
        "checkpoint_training_time_seconds": model.training_time,
        "resident_gpu_memory_mb": resident_memory_mb,
        "resident_gpu_memory_measurement": resident_memory_source,
        "base": base_results,
        "FlaRe": flare_results,
        "renders_path": str(renders_path),
        "gpu": {
            "name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }

    stats_path = args.model_path / "stats"
    stats_path.mkdir(parents=True, exist_ok=True)
    output_path = stats_path / "evaluation.json"
    output_path.write_text(
        json.dumps(evaluation, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    print(json.dumps(evaluation, indent=2, allow_nan=False), flush=True)
    print(f"Saved evaluation to: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
