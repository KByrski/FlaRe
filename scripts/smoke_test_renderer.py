#!/usr/bin/env python3
"""Standalone CUDA/OptiX smoke test for the integrated FLARE renderer."""

from __future__ import annotations

import math
from pathlib import Path
import sys


def stage(message: str) -> None:
    print(f"[smoke] {message}", flush=True)


def main() -> int:
    project_dir = Path(__file__).resolve().parent.parent
    renderer_dir = project_dir / "source" / "renderer" / "output"
    module_path = renderer_dir / "PYOPTIXFLARERENDERER.so"
    ptx_path = renderer_dir / "shaders.cu.ptx"

    if not module_path.is_file():
        raise FileNotFoundError(f"Renderer module not found: {module_path}")
    if not ptx_path.is_file():
        raise FileNotFoundError(f"OptiX PTX not found: {ptx_path}")

    sys.path.insert(0, str(renderer_dir))

    import torch

    stage(f"renderer directory: {renderer_dir}")
    stage(f"PyTorch {torch.__version__}, torch CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    stage(f"GPU: {gpu_name}; compute capability: {capability}")
    if "RTX 4090" not in gpu_name:
        raise RuntimeError(f"Expected RTX 4090, got {gpu_name}")
    if capability != (8, 9):
        raise RuntimeError(f"Expected compute capability (8, 9), got {capability}")

    import PYOPTIXFLARERENDERER as flare

    stage("native module import: OK")

    width = 8
    height = 8
    ray_count = width * height
    renderer = flare.CPyOptiXFLARERenderer(8, 11.3449, ray_count)
    stage("renderer/OptiX constructor: OK")

    right = torch.tensor([1.0, 0.0, 0.0], device=device)
    down = torch.tensor([0.0, 1.0, 0.0], device=device)
    forward = torch.tensor([0.0, 0.0, 1.0], device=device)
    directions = flare.GenerateRays(
        right,
        down,
        forward,
        width,
        height,
        1.0,
        1.0,
    ).reshape(ray_count, 3)
    directions /= torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    torch.cuda.synchronize()

    if directions.shape != (ray_count, 3):
        raise AssertionError(f"Unexpected ray shape: {tuple(directions.shape)}")
    if not bool(torch.isfinite(directions).all()):
        raise AssertionError("GenerateRays returned NaN or Inf")
    stage("GenerateRays CUDA kernel: OK")

    means = torch.tensor([[0.0, 0.0, 3.0]], device=device)
    log_scales = torch.full((1, 2), math.log(0.75), device=device)
    quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    rgb = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    opacity_logits = torch.tensor([[math.log(0.9 / 0.1)]], device=device)
    k_raw = torch.tensor([[math.log(math.expm1(1.0))]], device=device)

    renderer.SetGeometry(
        means,
        torch.exp(log_scales),
        quaternions,
        torch.sigmoid(opacity_logits),
        1.0 + torch.nn.functional.softplus(k_raw),
    )
    torch.cuda.synchronize()
    stage("SetGeometry/OptiX acceleration structure: OK")

    origins = torch.zeros((ray_count, 3), device=device)
    image = torch.zeros((ray_count, 3), device=device)
    rgba = torch.cat((rgb, opacity_logits), dim=1)
    background = (0.01, 0.02, 0.03)

    renderer.Forward_inference_base(
        origins,
        directions,
        image,
        *background,
        means,
        log_scales,
        quaternions,
        rgba,
        k_raw,
        1.0e-4,
    )
    torch.cuda.synchronize()

    if not bool(torch.isfinite(image).all()):
        raise AssertionError("Renderer returned NaN or Inf")
    if image[:, 0].max().item() <= max(background):
        raise AssertionError("The test Gaussian did not contribute to the image")

    stage(
        "Forward_inference_base: OK; "
        f"range=[{image.min().item():.6f}, {image.max().item():.6f}]"
    )

    geometry_image = torch.zeros_like(image)
    rendered_normal = torch.zeros_like(image)
    alpha = torch.zeros((ray_count,), device=device)
    expected_depth_numerator = torch.zeros_like(alpha)
    renderer.Forward_inference_base_with_geometry(
        origins,
        directions,
        geometry_image,
        rendered_normal,
        alpha,
        expected_depth_numerator,
        *background,
        means,
        log_scales,
        quaternions,
        rgba,
        k_raw,
        1.0e-4,
    )
    depth_and_index = torch.empty((ray_count, 2), device=device)
    renderer.GetMedianDepth_base(
        origins,
        directions,
        means,
        log_scales,
        quaternions,
        rgba,
        k_raw,
        1.0e-4,
        depth_and_index,
    )
    torch.cuda.synchronize()

    if not bool(torch.isfinite(geometry_image).all()):
        raise AssertionError("Geometry renderer returned NaN or Inf RGB")
    if not bool(torch.isfinite(rendered_normal).all()):
        raise AssertionError("Geometry renderer returned NaN or Inf normals")
    if not bool(torch.isfinite(alpha).all()):
        raise AssertionError("Geometry renderer returned NaN or Inf alpha")
    if alpha.max().item() <= 0.0:
        raise AssertionError("Geometry renderer returned an empty alpha image")
    valid_expected = alpha > 0.0
    expected_t = expected_depth_numerator[valid_expected] / alpha[valid_expected]
    if not bool(torch.isfinite(expected_t).all()) or expected_t.min().item() <= 0.0:
        raise AssertionError("Expected depth is invalid")
    median_t = depth_and_index[:, 0]
    median_indices = depth_and_index[:, 1].contiguous().view(torch.int32)
    valid_median = median_indices >= 0
    if not bool(valid_median.any()):
        raise AssertionError("Median-depth pass did not hit the test Gaussian")
    if not bool(torch.isfinite(median_t[valid_median]).all()):
        raise AssertionError("Median-depth pass returned NaN or Inf")
    stage(
        "geometry buffers: OK; "
        f"alpha_max={alpha.max().item():.6f}, "
        f"expected_t=[{expected_t.min().item():.6f}, {expected_t.max().item():.6f}], "
        f"median_hits={valid_median.sum().item()}"
    )

    source_dir = project_dir / "source"
    sys.path.insert(0, str(source_dir))
    from utils.depth_utils import ray_parameters_to_depth_and_normal

    training_image = torch.zeros_like(image)
    depth_reg_accums = torch.zeros((ray_count, 4), device=device)
    training_depth_and_index = torch.zeros((ray_count, 2), device=device)
    surface_normal = torch.zeros_like(image)
    normal_reg_accums = torch.zeros((ray_count, 4), device=device)
    renderer.Forward_training_base(
        origins,
        directions,
        training_image,
        *background,
        means,
        log_scales,
        quaternions,
        rgba,
        k_raw,
        1.0e-4,
        depth_reg_accums,
        -0.01,
        1.01,
        training_depth_and_index,
        surface_normal,
        normal_reg_accums,
    )
    training_t = depth_reg_accums[:, 3] / depth_reg_accums[:, 1].clamp_min(1.0e-8)
    _, reconstructed_normal, training_valid = ray_parameters_to_depth_and_normal(
        training_t,
        depth_reg_accums[:, 1] > 0.01,
        origins,
        directions,
        forward,
        height,
        width,
    )
    surface_normal.copy_(reconstructed_normal.reshape(ray_count, 3))
    surface_normal[training_valid.reshape(-1), 0] += 0.2
    surface_normal.copy_(torch.nn.functional.normalize(surface_normal, dim=1))

    dL_dRGB = torch.zeros_like(rgb)
    dL_dA = torch.zeros_like(opacity_logits)
    dL_dk = torch.zeros_like(k_raw)
    dL_dm = torch.zeros_like(means)
    dL_ds = torch.zeros_like(log_scales)
    dL_dq = torch.zeros_like(quaternions)
    depth_normal_prefix_sums = torch.zeros((ray_count, 4), device=device)
    renderer.Backward_base(
        origins,
        directions,
        *background,
        means,
        log_scales,
        quaternions,
        rgba,
        k_raw,
        training_image,
        torch.zeros_like(training_image),
        dL_dRGB,
        dL_dA,
        dL_dk,
        dL_dm,
        dL_ds,
        dL_dq,
        1.0e-4,
        depth_reg_accums,
        depth_normal_prefix_sums,
        0.0,
        -0.01,
        1.01,
        training_depth_and_index,
        surface_normal,
        normal_reg_accums,
        1.0 / ray_count,
    )
    torch.cuda.synchronize()
    gradients = torch.cat((dL_dA.flatten(), dL_dq.flatten()))
    if not bool(torch.isfinite(gradients).all()):
        raise AssertionError("Normal-consistency backward returned NaN or Inf")
    if gradients.abs().max().item() <= 0.0:
        raise AssertionError("Normal-consistency backward returned only zero gradients")
    stage(
        "normal-consistency forward/backward: OK; "
        f"valid_surface_pixels={training_valid.sum().item()}, "
        f"max_gradient={gradients.abs().max().item():.6g}"
    )
    stage("PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"[smoke] FAIL: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        raise
