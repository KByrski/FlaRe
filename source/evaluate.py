#!/usr/bin/env python3
"""Evaluate the best FLARE checkpoint on a scene's test split."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch


SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SOURCE_DIR.parent
RENDERER_DIR = SOURCE_DIR / "renderer" / "output"

if str(RENDERER_DIR) not in sys.path:
    sys.path.insert(0, str(RENDERER_DIR))

if os.name == "nt":
    os.add_dll_directory(str(Path(torch.__file__).resolve().parent / "lib"))

import PYOPTIXFLARERENDERER as flare_renderer

from checkpoint_io import load_model_checkpoint
from lpipsPyTorch.modules.lpips import LPIPS
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos
from utils.image_utils import psnr
from utils.loss_utils import ssim


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


def load_test_cameras(args: argparse.Namespace) -> list[dict[str, object]]:
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

    views: list[dict[str, object]] = []
    for index, camera in enumerate(cameras):
        views.append(
            {
                "name": camera.image_name or f"view_{index:04d}",
                "width": camera.image_width,
                "height": camera.image_height,
                "fov_x": float(camera.FoVx),
                "fov_y": float(camera.FoVy),
                "origin": torch.tensor(
                    -camera.R @ camera.T, dtype=torch.float32
                ),
                "right": torch.tensor(
                    camera.R.transpose(1, 0)[0, :], dtype=torch.float32
                ),
                "down": torch.tensor(
                    camera.R.transpose(1, 0)[1, :], dtype=torch.float32
                ),
                "forward": torch.tensor(
                    camera.R.transpose(1, 0)[2, :], dtype=torch.float32
                ),
                "ground_truth": camera.original_image.detach().cpu().contiguous(),
            }
        )

    del cameras, scene_info
    gc.collect()
    torch.cuda.empty_cache()
    return views


def load_model(checkpoint_path: Path, device: torch.device) -> dict[str, object]:
    return load_model_checkpoint(checkpoint_path, device)


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


def render_one(
    mode: str,
    renderer: object,
    view: dict[str, object],
    model: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, float, float, str]:
    width = int(view["width"])
    height = int(view["height"])
    pixel_count = width * height

    right = view["right"].to(device)
    down = view["down"].to(device)
    forward = view["forward"].to(device)
    origins = view["origin"].to(device).repeat(pixel_count, 1)

    directions = flare_renderer.GenerateRays(
        right,
        down,
        forward,
        width,
        height,
        float(view["fov_x"]),
        float(view["fov_y"]),
    ).reshape(pixel_count, 3)
    directions /= torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    image = torch.zeros((pixel_count, 3), dtype=torch.float32, device=device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    if mode == "base":
        renderer.Forward_inference_base(
            origins,
            directions,
            image,
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
    elif mode == "FlaRe":
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
            image,
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
    else:
        raise ValueError(f"Unknown rendering mode: {mode}")

    end.record()
    end.synchronize()
    elapsed_seconds = start.elapsed_time(end) / 1000.0
    memory_mb, memory_source = process_gpu_memory_mb()

    image_chw = (
        image.clamp(0.0, 1.0)
        .reshape(height, width, 3)
        .permute(2, 0, 1)
        .contiguous()
        .cpu()
    )
    return image_chw, elapsed_seconds, memory_mb, memory_source


def save_render(image: torch.Tensor, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    pixels = (
        image.permute(1, 2, 0).mul(255.0).round().byte().numpy()
    )
    Image.fromarray(pixels, mode="RGB").save(destination)


# Paper: novel-view synthesis renders the held-out views with the same
# ray-queryable primitive representation used during optimization.
def render_dataset(
    mode: str,
    renderer: object,
    views: list[dict[str, object]],
    model: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
    renders_path: Path,
) -> tuple[list[torch.Tensor], dict[str, object]]:
    # Warm up module loading/JIT paths without including them in FPS.
    render_one(mode, renderer, views[0], model, args, device)
    torch.cuda.synchronize()

    images: list[torch.Tensor] = []
    render_seconds: list[float] = []
    memory_mb: list[float] = []
    memory_source = ""

    print(f"Rendering test split with {mode} ({len(views)} views)...", flush=True)
    for index, view in enumerate(views):
        image, elapsed, used_memory, memory_source = render_one(
            mode, renderer, view, model, args, device
        )
        if not args.skip_metrics:
            images.append(image)
        render_seconds.append(elapsed)
        memory_mb.append(used_memory)

        image_name = Path(str(view["name"])).stem
        save_render(image, renders_path / mode / f"{index:04d}_{image_name}.png")
        print(
            f"  {index + 1:4d}/{len(views)}: "
            f"{elapsed * 1000.0:.3f} ms, {used_memory:.1f} MiB",
            flush=True,
        )

    total_render_seconds = sum(render_seconds)
    performance = {
        "fps": len(views) / total_render_seconds,
        "average_render_time_ms": 1000.0 * total_render_seconds / len(views),
        "average_gpu_memory_mb": sum(memory_mb) / len(memory_mb),
        "gpu_memory_measurement": memory_source,
    }
    return images, performance


# Paper: quantitative benchmark protocol reports PSNR, SSIM, and LPIPS.
def calculate_metrics(
    images: list[torch.Tensor],
    views: list[dict[str, object]],
    lpips_vgg: LPIPS,
    device: torch.device,
) -> dict[str, float]:
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []

    with torch.no_grad():
        for image, view in zip(images, views):
            rendered = image.unsqueeze(0).to(device)
            ground_truth = view["ground_truth"].unsqueeze(0).to(device)
            psnr_values.append(float(psnr(rendered, ground_truth).mean().item()))
            ssim_values.append(float(ssim(rendered, ground_truth).item()))
            lpips_values.append(float(lpips_vgg(rendered, ground_truth).item()))

    return {
        "psnr": sum(psnr_values) / len(psnr_values),
        "ssim": sum(ssim_values) / len(ssim_values),
        "lpips_vgg": sum(lpips_values) / len(lpips_values),
    }


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
    gaussian_count = int(model["m"].shape[0])
    has_base_rgb = "RGB" in model

    # The 17-entry historical schema contains only FlaRe appearance. Ones are
    # the neutral multiplier for the compound RGB ABI, so FlaRe remains exact;
    # the unavailable explicit base renderer is skipped rather than fabricated.
    rgb_for_abi = model["RGB"] if has_base_rgb else torch.ones(
        (gaussian_count, 3), dtype=torch.float32, device=device
    )
    model["RGBA"] = torch.cat((rgb_for_abi, model["A"]), dim=1).contiguous()
    model["w1_fp16"] = torch.cat(
        (model["w1_uv"], model["w1_v"], model["w1_conditioning"]), dim=1
    ).to(torch.float16).contiguous()
    model["w2_fp16"] = model["w2"].to(torch.float16).contiguous()
    model["w3_fp16"] = model["w3"].to(torch.float16).contiguous()
    model["conditioning_variable_fp16"] = model["conditioning_variable"].to(
        torch.float16
    ).contiguous()

    max_batch_size = max(int(view["width"]) * int(view["height"]) for view in views)
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
    resident_memory_mb, resident_memory_source = process_gpu_memory_mb()

    renders_path = args.model_path / "renders" / f"evaluation_{iteration}"
    base_images: list[torch.Tensor] = []
    if has_base_rgb:
        base_images, base_results = render_dataset(
            "base", renderer, views, model, args, device, renders_path
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
        "FlaRe", renderer, views, model, args, device, renders_path
    )
    flare_results["available"] = True

    if not args.skip_metrics:
        print("Loading LPIPS-VGG and calculating image metrics...", flush=True)
        lpips_vgg = LPIPS("vgg").to(device).eval()
        if has_base_rgb:
            base_results.update(calculate_metrics(base_images, views, lpips_vgg, device))
        flare_results.update(calculate_metrics(flare_images, views, lpips_vgg, device))

    test_resolutions = sorted(
        {f'{int(view["width"])}x{int(view["height"])}' for view in views}
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
        "checkpoint_training_time_seconds": model["training_time_seconds"],
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
