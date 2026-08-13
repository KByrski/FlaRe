#!/usr/bin/env python3
"""Refit edited Blender PLY frames and render them with a FlaRe checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import torch

from evaluate import flare_renderer, load_model, load_test_cameras, render_one, save_render
from flare_edit_io import (
    deform_vertices,
    read_edit_ply,
    vertices_to_primitives,
)


# Procedural edit defaults. These can also be overridden from the command line.
DEFAULT_DEFORMATION = "sin"
DEFAULT_DEFORM_AMPLITUDE = 0.1
DEFAULT_DEFORM_FREQUENCY = 8.0
DEFAULT_DEFORM_ROTATION_Z_DEGREES = 0.0
DEFAULT_DEFORM_PHASE_SHIFT = 0.0
DEFAULT_FADE_FRAMES = 60


def phase_shift_value(value: str) -> float | str:
    """Accept a phase in degrees or the special sweep mode."""
    special_mode = value.lower()
    if special_mode in ("sweep", "sweep_camera"):
        return special_mode
    try:
        return float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "phase shift must be degrees, 'sweep', or 'sweep_camera'"
        ) from error


def distributed_camera_index(
    frame_index: int, frame_count: int, camera_count: int
) -> int:
    """Assign consecutive frames as evenly as possible across all cameras."""
    return min(camera_count - 1, (frame_index * camera_count) // frame_count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--ply_dir", required=True, type=Path)
    parser.add_argument("--scene_path", "--source_path", "-s", required=True, type=Path)
    parser.add_argument("--output_dir", "-o", required=True, type=Path)
    parser.add_argument("--pattern", default="*.ply")
    parser.add_argument(
        "--camera",
        "--camera_index",
        dest="camera_index",
        type=int,
        default=0,
        help="Zero-based test-camera index used for every animation frame (default: 0)",
    )
    parser.add_argument("--resolution", "-r", default=-1, type=int)
    parser.add_argument("--images", default="images")
    parser.add_argument("--mode", choices=("base", "FlaRe"), default="FlaRe")
    parser.add_argument(
        "--deform",
        choices=("sin", "sin2", "none"),
        default=DEFAULT_DEFORMATION,
        help=(
            "Procedural edit: sin uses rotated X; sin2 uses rotated X and Y "
            "(default: sin)"
        ),
    )
    parser.add_argument(
        "--deform_amplitude",
        type=float,
        default=DEFAULT_DEFORM_AMPLITUDE,
    )
    parser.add_argument(
        "--deform_frequency",
        type=float,
        default=DEFAULT_DEFORM_FREQUENCY,
    )
    parser.add_argument(
        "--deform_rotation_z",
        type=float,
        default=DEFAULT_DEFORM_ROTATION_Z_DEGREES,
        metavar="DEGREES",
        help="Rigid world-Z rotation in degrees, applied before sine deformation",
    )
    parser.add_argument(
        "--phase_shift",
        type=phase_shift_value,
        default=DEFAULT_DEFORM_PHASE_SHIFT,
        metavar="DEGREES|sweep|sweep_camera",
        help=(
            "Phase in degrees; 'sweep' renders base.ply from 0 through 359 "
            "using --camera; 'sweep_camera' distributes those frames across "
            "all test cameras"
        ),
    )
    parser.add_argument(
        "--fade",
        action="store_true",
        help=(
            "For sweep modes, prepend a zero-to-full amplitude transition and "
            "append a full-to-zero transition"
        ),
    )
    parser.add_argument(
        "--fade_frames",
        type=int,
        default=DEFAULT_FADE_FRAMES,
        help="Frames in each fade transition (default: 60; requires --fade)",
    )
    parser.add_argument("--bg_color_R", type=float, default=0.0)
    parser.add_argument("--bg_color_G", type=float, default=0.0)
    parser.add_argument("--bg_color_B", type=float, default=0.0)
    parser.add_argument("--number_of_sides", type=int, default=8)
    parser.add_argument("--ray_termination_T_threshold_inference", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.number_of_sides != 8:
        raise ValueError("The edit interchange format requires --number_of_sides 8")
    if args.fade_frames <= 0:
        raise ValueError("--fade_frames must be greater than zero")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    args.scene_path, args.ply_dir = args.scene_path.resolve(), args.ply_dir.resolve()
    def natural_key(path: Path) -> list[object]:
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]

    sweep = args.phase_shift in ("sweep", "sweep_camera")
    sweep_camera = args.phase_shift == "sweep_camera"
    if args.fade and not sweep:
        raise ValueError("--fade requires --phase_shift sweep or sweep_camera")
    if sweep:
        if args.deform == "none":
            raise ValueError(
                "Phase sweep modes require --deform sin or sin2"
            )
        base_ply = args.ply_dir / "base.ply"
        if not base_ply.is_file():
            raise FileNotFoundError(
                f"Phase sweep modes require exactly this input file: {base_ply}"
            )
        sweep_jobs = [
            (base_ply, float(degrees), 1.0, "sweep")
            for degrees in range(360)
        ]
        if args.fade:
            fade_in = [
                (
                    base_ply,
                    0.0,
                    frame / args.fade_frames,
                    "fade_in",
                )
                for frame in range(args.fade_frames)
            ]
            fade_out = [
                (
                    base_ply,
                    359.0,
                    1.0 - ((frame + 1) / args.fade_frames),
                    "fade_out",
                )
                for frame in range(args.fade_frames)
            ]
            render_jobs = fade_in + sweep_jobs + fade_out
            print(
                f"Fade enabled: {args.fade_frames} frames from amplitude 0 "
                f"to {args.deform_amplitude:g}, then {args.fade_frames} "
                "frames back to 0.",
                flush=True,
            )
        else:
            render_jobs = sweep_jobs
        sweep_description = (
            " across all test cameras" if sweep_camera else ""
        )
        print(
            "Phase sweep: rendering base.ply at 1-degree resolution "
            f"(0 through 359 degrees){sweep_description}.",
            flush=True,
        )
    else:
        frames = sorted(args.ply_dir.glob(args.pattern), key=natural_key)
        if not frames:
            raise FileNotFoundError(
                f"No files matching {args.pattern!r} in {args.ply_dir}"
            )
        render_jobs = [
            (path, float(args.phase_shift), 1.0, "frames") for path in frames
        ]

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    views = load_test_cameras(args)
    if sweep_camera:
        if len(views) > 360:
            raise ValueError(
                f"Cannot give all {len(views)} test cameras at least one frame "
                "in a 360-frame sweep"
            )
        camera_render_jobs = []
        for path, phase, amplitude_scale, section in render_jobs:
            if section == "fade_in":
                camera_index = 0
            elif section == "fade_out":
                camera_index = len(views) - 1
            else:
                camera_index = distributed_camera_index(
                    int(phase), 360, len(views)
                )
            camera_render_jobs.append(
                (path, phase, amplitude_scale, section, camera_index)
            )
        render_jobs = camera_render_jobs
        camera_counts = [0] * len(views)
        for _, _, _, section, camera_index in render_jobs:
            if section == "sweep":
                camera_counts[camera_index] += 1
        print(
            f"Distributing 360 frames across {len(views)} test cameras "
            f"({min(camera_counts)}-{max(camera_counts)} frames per camera).",
            flush=True,
        )
    else:
        if not 0 <= args.camera_index < len(views):
            raise IndexError(f"camera_index must be in [0, {len(views) - 1}]")
        render_jobs = [
            (path, phase, amplitude_scale, section, args.camera_index)
            for path, phase, amplitude_scale, section in render_jobs
        ]
        view = views[args.camera_index]
        print(
            f"Using test camera {args.camera_index}/{len(views) - 1}: "
            f"{view['name']}",
            flush=True,
        )

    used_camera_indices = {job[4] for job in render_jobs}
    used_views = [views[index] for index in sorted(used_camera_indices)]
    model = load_model(args.checkpoint.resolve(), device)
    count = int(model["m"].shape[0])
    if args.mode == "base" and "RGB" not in model:
        raise ValueError(
            "This is a 17-entry FlaRe-only checkpoint without base RGB; "
            "render with --mode FlaRe (the default)"
        )
    # Legacy 17-entry checkpoints predate the separate base-RGB multiplier.
    # Current CUDA computes final_rgb = RGBA_param.rgb * MLP_rgb, so an identity
    # multiplier (ones), not zeros, reproduces the legacy MLP-only color path.
    if "RGB" in model:
        rgb = model["RGB"]
    else:
        rgb = torch.ones_like(model["m"])
        print(
            "Legacy 17-entry checkpoint: using identity base-RGB multiplier.",
            flush=True,
        )
    model["RGBA"] = torch.cat((rgb, model["A"]), 1).contiguous()
    model["w1_fp16"] = torch.cat((model["w1_uv"], model["w1_v"], model["w1_conditioning"]), 1).half().contiguous()
    model["w2_fp16"], model["w3_fp16"] = model["w2"].half().contiguous(), model["w3"].half().contiguous()
    model["conditioning_variable_fp16"] = model["conditioning_variable"].half().contiguous()

    # evaluate.py has already loaded the current renderer extension.
    max_pixel_count = max(
        int(view["width"]) * int(view["height"]) for view in used_views
    )
    renderer = flare_renderer.CPyOptiXFLARERenderer(8, 11.3449, max_pixel_count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # A sweep deforms the same base geometry 360 times, so load the large PLY
    # only once rather than parsing it again for every phase.
    sweep_vertices = (
        read_edit_ply(render_jobs[0][0], count).to(device) if sweep else None
    )
    for index, (
        ply_path,
        phase_shift_degrees,
        amplitude_scale,
        section,
        camera_index,
    ) in enumerate(render_jobs):
        view = views[camera_index]
        vertices = (
            sweep_vertices
            if sweep_vertices is not None
            else read_edit_ply(ply_path, count).to(device)
        )
        vertices = deform_vertices(
            vertices,
            None if args.deform == "none" else args.deform,
            amplitude=args.deform_amplitude * amplitude_scale,
            frequency=args.deform_frequency,
            rotation_z_degrees=args.deform_rotation_z,
            phase_shift_degrees=phase_shift_degrees,
        )
        model["m"], model["s"], model["q"] = vertices_to_primitives(
            vertices, model["A"], model["k"], model["s"], model["q"]
        )
        renderer.SetGeometry(model["m"], torch.exp(model["s"]), model["q"], torch.sigmoid(model["A"]), 1.0 + torch.nn.functional.softplus(model["k"]))
        image, elapsed, _, _ = render_one(args.mode, renderer, view, model, args, device)
        destination = args.output_dir / f"{index:06d}.png"
        save_render(image, destination)
        phase_text = (
            f", phase={phase_shift_degrees:g} deg" if sweep else ""
        )
        amplitude_text = (
            f", amplitude={args.deform_amplitude * amplitude_scale:g}"
            if args.fade
            else ""
        )
        camera_text = (
            f", camera={camera_index}:{view['name']}" if sweep_camera else ""
        )
        print(
            f"[{index + 1}/{len(render_jobs)}] {ply_path.name}"
            f"{phase_text}{amplitude_text}{camera_text} "
            f"-> {destination.name} ({elapsed * 1000:.1f} ms)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
