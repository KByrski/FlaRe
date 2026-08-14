"""Reusable FlaRe test-view rendering and metric orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Sequence

from PIL import Image
import torch

from utils.image_utils import psnr
from utils.loss_utils import ssim


@dataclass(frozen=True)
class EvaluationView:
    name: str
    width: int
    height: int
    fov_x: float
    fov_y: float
    origin: torch.Tensor
    right: torch.Tensor
    down: torch.Tensor
    forward: torch.Tensor
    ground_truth: torch.Tensor


@dataclass(frozen=True)
class EvaluationOptions:
    background: tuple[float, float, float]
    ray_termination_threshold: float


def _timed_cuda_call(callback: Callable[[], None]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    callback()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1000.0


def _inference_tensors(model) -> dict[str, torch.Tensor]:
    rgb = getattr(model, "RGB", None)
    if rgb is None:
        rgb = torch.ones(
            (model.m.shape[0], 3), dtype=torch.float32, device=model.m.device
        )
    return {
        "RGBA": torch.cat((rgb, model.A), dim=1).contiguous(),
        "w1": torch.cat(
            (model.w1_uv, model.w1_v, model.w1_conditioning), dim=1
        ).to(torch.float16).contiguous(),
        "w2": model.w2.to(torch.float16).contiguous(),
        "w3": model.w3.to(torch.float16).contiguous(),
        "conditioning": model.conditioning_variable.to(
            torch.float16
        ).contiguous(),
    }



@dataclass(frozen=True)
class TrainingEvaluationResult:
    metrics: dict[str, float]

    @property
    def selected_checkpoint_metric(self) -> tuple[str, float]:
        if "PSNR_test_FlaRe" in self.metrics:
            return "PSNR_test_FlaRe", self.metrics["PSNR_test_FlaRe"]
        return "PSNR_test_base", self.metrics["PSNR_test_base"]


def evaluate_training_splits(
    renderer,
    model,
    train_dataset,
    test_dataset,
    options: EvaluationOptions,
    warmup_lambda: float,
    fixed_background: torch.Tensor,
) -> TrainingEvaluationResult:
    """Run the historical periodic train/test base and FlaRe PSNR loops."""
    tensors = _inference_tensors(model)
    metrics: dict[str, float] = {}

    def evaluate_split(mode: str, split_name: str, dataset) -> None:
        psnr_total = 0.0
        started = time.perf_counter()
        for pose_index in range(dataset.camera_count):
            origins = dataset.origins[pose_index].repeat(
                dataset.width * dataset.height, 1
            )
            directions = renderer.generate_rays(
                dataset.rights[pose_index],
                dataset.downs[pose_index],
                dataset.forwards[pose_index],
                dataset.width,
                dataset.height,
                dataset.fov_x[pose_index].item(),
                dataset.fov_y[pose_index].item(),
            ).reshape(dataset.width * dataset.height, 3)
            directions = directions / torch.sqrt(
                torch.sum(directions * directions, 1, keepdim=True)
            )
            directions = directions.reshape(
                dataset.height, dataset.width, 3
            )
            image = torch.zeros(
                dataset.width * dataset.height,
                3,
                dtype=torch.float32,
                device=dataset.foreground.device,
            )
            if mode == "base":
                renderer.forward_inference_base(
                    origins,
                    directions,
                    image,
                    *options.background,
                    model.m,
                    model.s,
                    model.q,
                    tensors["RGBA"],
                    model.k,
                    options.ray_termination_threshold,
                )
            else:
                renderer.forward_inference(
                    tensors["conditioning"],
                    model.features,
                    tensors["w1"],
                    model.b1,
                    tensors["w2"],
                    model.b2,
                    tensors["w3"],
                    model.b3,
                    origins,
                    directions,
                    image,
                    *options.background,
                    model.m,
                    model.s,
                    model.q,
                    tensors["RGBA"],
                    model.k,
                    options.ray_termination_threshold,
                )
            foreground = dataset.foreground.reshape(
                dataset.camera_count, dataset.width * dataset.height, 3
            )[pose_index]
            alpha = dataset.alpha.reshape(
                dataset.camera_count, dataset.width * dataset.height, 1
            )[pose_index]
            ground_truth = foreground + fixed_background * (1.0 - alpha)
            loss = torch.mean(
                (torch.clamp(image, min=0.0, max=1.0) - ground_truth) ** 2
            )
            value = (
                -10.0
                * (
                    torch.log(loss)
                    / torch.log(
                        torch.tensor(
                            [10.0], device=dataset.foreground.device
                        )
                    )
                )
            ).item()
            psnr_total += value
            print(pose_index, " : ", value, sep="")

        elapsed = time.perf_counter() - started
        appearance_name = "base" if mode == "base" else "FlaRe"
        metrics[f"PSNR_{split_name}_{appearance_name}"] = (
            psnr_total / dataset.camera_count
        )
        metrics[f"FPS_{split_name}_{appearance_name}"] = (
            dataset.camera_count / elapsed
        )
        print(
            f"FPS ({split_name}, {appearance_name}): ",
            metrics[f"FPS_{split_name}_{appearance_name}"],
            sep="",
        )
        print(
            f"AVG PSNR ({split_name}, {appearance_name}): ",
            metrics[f"PSNR_{split_name}_{appearance_name}"],
            sep="",
        )

    if warmup_lambda < 1.0:
        evaluate_split("base", "train", train_dataset)
        evaluate_split("base", "test", test_dataset)
    if warmup_lambda > 0.0:
        evaluate_split("FlaRe", "train", train_dataset)
        evaluate_split("FlaRe", "test", test_dataset)
    return TrainingEvaluationResult(metrics)

def render_one(
    mode: str,
    renderer,
    view: EvaluationView,
    model,
    options: EvaluationOptions,
    device: torch.device,
    *,
    timed_call: Callable[[Callable[[], None]], float] = _timed_cuda_call,
    memory_reader: Callable[[], tuple[float, str]],
) -> tuple[torch.Tensor, float, float, str]:
    pixel_count = view.width * view.height
    origins = view.origin.to(device).repeat(pixel_count, 1)
    directions = renderer.generate_rays(
        view.right.to(device),
        view.down.to(device),
        view.forward.to(device),
        view.width,
        view.height,
        view.fov_x,
        view.fov_y,
    ).reshape(pixel_count, 3)
    directions /= torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    image = torch.zeros((pixel_count, 3), dtype=torch.float32, device=device)
    tensors = _inference_tensors(model)

    def inference() -> None:
        if mode == "base":
            if getattr(model, "RGB", None) is None:
                raise ValueError("base rendering requires explicit RGB")
            renderer.forward_inference_base(
                origins,
                directions,
                image,
                *options.background,
                model.m,
                model.s,
                model.q,
                tensors["RGBA"],
                model.k,
                options.ray_termination_threshold,
            )
        elif mode == "FlaRe":
            renderer.forward_inference(
                tensors["conditioning"],
                model.features,
                tensors["w1"],
                model.b1,
                tensors["w2"],
                model.b2,
                tensors["w3"],
                model.b3,
                origins,
                directions,
                image,
                *options.background,
                model.m,
                model.s,
                model.q,
                tensors["RGBA"],
                model.k,
                options.ray_termination_threshold,
            )
        else:
            raise ValueError(f"Unknown rendering mode: {mode}")

    elapsed_seconds = timed_call(inference)
    memory_mb, memory_source = memory_reader()
    image_chw = (
        image.clamp(0.0, 1.0)
        .reshape(view.height, view.width, 3)
        .permute(2, 0, 1)
        .contiguous()
        .cpu()
    )
    return image_chw, elapsed_seconds, memory_mb, memory_source


def save_render(image: torch.Tensor, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    pixels = image.permute(1, 2, 0).mul(255.0).round().byte().numpy()
    Image.fromarray(pixels, mode="RGB").save(destination)


def render_dataset(
    mode: str,
    renderer,
    views: Sequence[EvaluationView],
    model,
    options: EvaluationOptions,
    device: torch.device,
    renders_path: Path,
    *,
    keep_images: bool,
    memory_reader: Callable[[], tuple[float, str]],
) -> tuple[list[torch.Tensor], dict[str, object]]:
    render_one(
        mode,
        renderer,
        views[0],
        model,
        options,
        device,
        memory_reader=memory_reader,
    )
    torch.cuda.synchronize()

    images: list[torch.Tensor] = []
    render_seconds: list[float] = []
    memory_mb: list[float] = []
    memory_source = ""
    print(f"Rendering test split with {mode} ({len(views)} views)...", flush=True)
    for index, view in enumerate(views):
        image, elapsed, used_memory, memory_source = render_one(
            mode,
            renderer,
            view,
            model,
            options,
            device,
            memory_reader=memory_reader,
        )
        if keep_images:
            images.append(image)
        render_seconds.append(elapsed)
        memory_mb.append(used_memory)
        image_name = Path(view.name).stem
        save_render(image, renders_path / mode / f"{index:04d}_{image_name}.png")
        print(
            f"  {index + 1:4d}/{len(views)}: "
            f"{elapsed * 1000.0:.3f} ms, {used_memory:.1f} MiB",
            flush=True,
        )

    total_render_seconds = sum(render_seconds)
    return images, {
        "fps": len(views) / total_render_seconds,
        "average_render_time_ms": 1000.0 * total_render_seconds / len(views),
        "average_gpu_memory_mb": sum(memory_mb) / len(memory_mb),
        "gpu_memory_measurement": memory_source,
    }


def calculate_metrics(
    images: Sequence[torch.Tensor],
    views: Sequence[EvaluationView],
    lpips_vgg,
    device: torch.device,
) -> dict[str, float]:
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []
    with torch.no_grad():
        for image, view in zip(images, views):
            rendered = image.unsqueeze(0).to(device)
            ground_truth = view.ground_truth.unsqueeze(0).to(device)
            psnr_values.append(float(psnr(rendered, ground_truth).mean().item()))
            ssim_values.append(float(ssim(rendered, ground_truth).item()))
            lpips_values.append(float(lpips_vgg(rendered, ground_truth).item()))
    return {
        "psnr": sum(psnr_values) / len(psnr_values),
        "ssim": sum(ssim_values) / len(ssim_values),
        "lpips_vgg": sum(lpips_values) / len(lpips_values),
    }
