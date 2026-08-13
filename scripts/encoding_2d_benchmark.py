#!/usr/bin/env python3
"""Controlled 2D comparison of FlaRe's LUT and Instant-NGP-style hashing.

The benchmark fits continuous analytic RGB functions on [0, 1]^2.  LUT and
hash variants share interpolation, feature width, decoder, optimizer, samples,
and seeds; only the virtual grid resolutions and vertex addressing differ.

Example (paper run):
    python scripts/encoding_2d_benchmark.py \
        --device cuda --seeds 0,1,2,3,4 \
        --functions smooth,multiscale,chirp,localized,checker

Quick smoke test:
    python scripts/encoding_2d_benchmark.py \
        --device cpu --steps 10 --batch-size 256 --eval-resolution 32 \
        --log-every 5 --configs lut,hash_512 --functions smooth --seeds 0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
from torch import nn


# Paper: "LUT-Encoding" ablation (Table 3). These are the four
# collision-free grid resolutions used by the FlaRe representation.
BASE_RESOLUTIONS = (16, 25, 40, 64)
FEATURES_PER_LEVEL = 2
HASH_PRIMES_2D = (1, 2_654_435_761)
UINT32_MASK = 0xFFFF_FFFF


@dataclass(frozen=True)
class EncodingConfig:
    name: str
    resolutions: tuple[int, ...]
    max_entries_per_level: int | None
    description: str


CONFIGS = {
    "lut": EncodingConfig(
        "lut",
        BASE_RESOLUTIONS,
        None,
        "FlaRe collision-free row-major LUT.",
    ),
    "hash_control": EncodingConfig(
        "hash_control",
        BASE_RESOLUTIONS,
        4096,
        "Same grids and capacity as LUT; all levels use direct addressing.",
    ),
    "hash_2048": EncodingConfig(
        "hash_2048",
        BASE_RESOLUTIONS,
        2048,
        "Same virtual grids; finest level is hashed.",
    ),
    "hash_1024": EncodingConfig(
        "hash_1024",
        BASE_RESOLUTIONS,
        1024,
        "Same virtual grids; two finest levels are hashed.",
    ),
    "hash_512": EncodingConfig(
        "hash_512",
        BASE_RESOLUTIONS,
        512,
        "Same virtual grids; three finest levels are hashed.",
    ),
    "hash_budget": EncodingConfig(
        "hash_budget",
        (16, 40, 101, 256),
        2360,
        "Near-exact LUT parameter budget with finer virtual grids.",
    ),
}


def parse_csv(value: str, cast: Callable[[str], object] = str) -> list:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    try:
        return [cast(item) for item in items]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def hash_vertices(vertices: torch.Tensor, table_size: int) -> torch.Tensor:
    """Instant-NGP-style 2D XOR hash with uint32 multiplication semantics."""
    x = torch.bitwise_and(vertices[..., 0] * HASH_PRIMES_2D[0], UINT32_MASK)
    y = torch.bitwise_and(vertices[..., 1] * HASH_PRIMES_2D[1], UINT32_MASK)
    return torch.remainder(torch.bitwise_xor(x, y), table_size)


class GridLevel(nn.Module):
    """One bilinearly interpolated 2D feature level."""

    # Paper: direct row-major lookup avoids hash collisions; bilinear
    # interpolation keeps the encoded local radiance field continuous.

    def __init__(
        self,
        resolution: int,
        feature_dim: int,
        table_size: int,
        use_hash: bool,
    ) -> None:
        super().__init__()
        if resolution < 2:
            raise ValueError("resolution must be at least 2")
        if table_size <= 0:
            raise ValueError("table_size must be positive")
        self.resolution = resolution
        self.feature_dim = feature_dim
        self.table_size = table_size
        self.use_hash = use_hash
        self.embedding = nn.Embedding(table_size, feature_dim)
        nn.init.uniform_(self.embedding.weight, -1.0e-4, 1.0e-4)
        self.register_buffer(
            "corner_offsets",
            torch.tensor(((0, 0), (1, 0), (0, 1), (1, 1)), dtype=torch.long),
            persistent=False,
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.shape[-1] != 2:
            raise ValueError("coordinates must have shape (..., 2)")
        flat = coordinates.reshape(-1, 2).clamp(0.0, 1.0)

        # Resolution denotes the number of vertices, matching the N^2 feature
        # count inferred from FlaRe. Clamp the base cell so u/v=1 addresses
        # the last vertex with interpolation weight one.
        scaled = flat * float(self.resolution - 1)
        lower = torch.floor(scaled).to(torch.long)
        lower = lower.clamp(min=0, max=self.resolution - 2)
        fraction = scaled - lower.to(scaled.dtype)

        vertices = lower[:, None, :] + self.corner_offsets[None, :, :]
        offsets = self.corner_offsets.to(dtype=fraction.dtype)
        per_axis_weights = torch.where(
            offsets[None, :, :].bool(),
            fraction[:, None, :],
            1.0 - fraction[:, None, :],
        )
        weights = per_axis_weights.prod(dim=-1, keepdim=True)

        if self.use_hash:
            indices = hash_vertices(vertices, self.table_size)
        else:
            indices = vertices[..., 1] * self.resolution + vertices[..., 0]

        encoded = (self.embedding(indices) * weights).sum(dim=-2)
        return encoded.reshape(*coordinates.shape[:-1], self.feature_dim)


class MultiResolutionEncoding(nn.Module):
    def __init__(
        self,
        config: EncodingConfig,
        feature_dim: int = FEATURES_PER_LEVEL,
    ) -> None:
        super().__init__()
        self.config = config
        levels: list[GridLevel] = []
        for resolution in config.resolutions:
            dense_entries = resolution * resolution
            if config.max_entries_per_level is None:
                table_size = dense_entries
            else:
                table_size = min(dense_entries, config.max_entries_per_level)
            # Following Instant-NGP, avoid hashing when the dense level fits.
            use_hash = table_size < dense_entries
            levels.append(
                GridLevel(resolution, feature_dim, table_size, use_hash)
            )
        self.levels = nn.ModuleList(levels)
        self.output_dim = len(levels) * feature_dim

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return torch.cat([level(coordinates) for level in self.levels], dim=-1)

    @property
    def stored_entries(self) -> int:
        return sum(level.table_size for level in self.levels)


class Decoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.network(encoded)


class FunctionModel(nn.Module):
    def __init__(self, encoding: MultiResolutionEncoding, hidden_dim: int) -> None:
        super().__init__()
        self.encoding = encoding
        self.decoder = Decoder(encoding.output_dim, hidden_dim)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoding(coordinates))


def smooth_target(coordinates: torch.Tensor) -> torch.Tensor:
    u, v = coordinates.unbind(dim=-1)
    channels = (
        0.50 + 0.25 * torch.sin(2 * math.pi * u) * torch.cos(2 * math.pi * v),
        0.50 + 0.25 * torch.cos(2 * math.pi * u + 0.4) * torch.sin(4 * math.pi * v),
        0.50 + 0.25 * torch.sin(4 * math.pi * (u + v) + 0.8),
    )
    return torch.stack(channels, dim=-1).clamp(0.0, 1.0)


def _multiscale_channel(
    u: torch.Tensor, v: torch.Tensor, phase: float
) -> torch.Tensor:
    result = torch.full_like(u, 0.5)
    for level in range(6):
        frequency = float(2**level)
        amplitude = 0.13 / float(2**level)
        result = result + amplitude * torch.sin(
            2 * math.pi * frequency * u + phase
        ) * torch.cos(
            2 * math.pi * frequency * v + 1.7 * phase
        )
    return result


def multiscale_target(coordinates: torch.Tensor) -> torch.Tensor:
    u, v = coordinates.unbind(dim=-1)
    return torch.stack(
        (
            _multiscale_channel(u, v, 0.0),
            _multiscale_channel(u, v, 0.7),
            _multiscale_channel(u, v, 1.4),
        ),
        dim=-1,
    ).clamp(0.0, 1.0)


def chirp_target(coordinates: torch.Tensor) -> torch.Tensor:
    u, v = coordinates.unbind(dim=-1)
    phases = (
        2 * math.pi * (2 * u + 14 * u.square() + 12 * v.square()),
        2 * math.pi * (2 * v + 12 * u.square() + 14 * v.square()) + 0.7,
        2 * math.pi * (u + v + 10 * (u - v).square()) + 1.4,
    )
    return torch.stack(
        [0.5 + 0.45 * torch.sin(phase) for phase in phases], dim=-1
    ).clamp(0.0, 1.0)


def localized_target(coordinates: torch.Tensor) -> torch.Tensor:
    u, v = coordinates.unbind(dim=-1)
    window = torch.exp(-90.0 * ((u - 0.68).square() + (v - 0.34).square()))
    detail = window * torch.sin(2 * math.pi * 34 * u) * torch.cos(
        2 * math.pi * 29 * v
    )
    base = smooth_target(coordinates)
    tint = torch.stack((detail, -0.7 * detail, 0.5 * detail), dim=-1)
    return (base + 0.35 * tint).clamp(0.0, 1.0)


def checker_target(coordinates: torch.Tensor) -> torch.Tensor:
    u, v = coordinates.unbind(dim=-1)
    checker = torch.remainder(
        torch.floor(16 * u).to(torch.long)
        + torch.floor(16 * v).to(torch.long),
        2,
    ).to(coordinates.dtype)
    rings = (
        torch.sin(2 * math.pi * 12 * torch.sqrt((u - 0.5).square() + (v - 0.5).square()))
        > 0
    ).to(coordinates.dtype)
    return torch.stack(
        (0.15 + 0.7 * checker, 0.15 + 0.7 * rings, 0.2 + 0.6 * (checker != rings)),
        dim=-1,
    )


TARGETS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "smooth": smooth_target,
    "multiscale": multiscale_target,
    "chirp": chirp_target,
    "localized": localized_target,
    "checker": checker_target,
}


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def parameter_bytes(module: nn.Module) -> int:
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in module.parameters()
    )


def make_evaluation_grid(resolution: int, device: torch.device) -> torch.Tensor:
    axis = torch.linspace(0.0, 1.0, resolution, device=device)
    v, u = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((u, v), dim=-1).reshape(-1, 2)


@torch.no_grad()
def predict_in_chunks(
    model: nn.Module, coordinates: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    return torch.cat(
        [
            model(coordinates[start : start + chunk_size])
            for start in range(0, coordinates.shape[0], chunk_size)
        ],
        dim=0,
    )


def gradient_mse(
    prediction: torch.Tensor, target: torch.Tensor, resolution: int
) -> float:
    prediction = prediction.reshape(resolution, resolution, 3)
    target = target.reshape(resolution, resolution, 3)
    pred_dx = prediction[:, 1:] - prediction[:, :-1]
    true_dx = target[:, 1:] - target[:, :-1]
    pred_dy = prediction[1:] - prediction[:-1]
    true_dy = target[1:] - target[:-1]
    return float(
        0.5
        * (
            torch.mean((pred_dx - true_dx).square())
            + torch.mean((pred_dy - true_dy).square())
        ).item()
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    target_function: Callable[[torch.Tensor], torch.Tensor],
    coordinates: torch.Tensor,
    resolution: int,
    chunk_size: int,
    return_images: bool = False,
) -> tuple[dict[str, float], torch.Tensor | None, torch.Tensor | None]:
    was_training = model.training
    model.eval()
    prediction = predict_in_chunks(model, coordinates, chunk_size).clamp(0.0, 1.0)
    target = target_function(coordinates)
    mse = float(torch.mean((prediction - target).square()).item())
    psnr = -10.0 * math.log10(max(mse, 1.0e-12))
    metrics = {
        "mse": mse,
        "psnr_db": psnr,
        "gradient_mse": gradient_mse(prediction, target, resolution),
    }
    model.train(was_training)
    if return_images:
        return metrics, prediction, target
    return metrics, None, None


# Paper: report the collision rates compared in the controlled LUT/hash study.
def collision_statistics(encoding: MultiResolutionEncoding) -> list[dict]:
    statistics_per_level: list[dict] = []
    for level_index, level in enumerate(encoding.levels):
        virtual_entries = level.resolution**2
        if level.use_hash:
            axis = torch.arange(level.resolution, dtype=torch.long)
            y, x = torch.meshgrid(axis, axis, indexing="ij")
            vertices = torch.stack((x, y), dim=-1).reshape(-1, 2)
            indices = hash_vertices(vertices, level.table_size)
            loads = torch.bincount(indices, minlength=level.table_size)
            occupied = int(torch.count_nonzero(loads).item())
            maximum_load = int(loads.max().item())
        else:
            occupied = virtual_entries
            maximum_load = 1
        statistics_per_level.append(
            {
                "level": level_index,
                "resolution": level.resolution,
                "virtual_entries": virtual_entries,
                "stored_entries": level.table_size,
                "addressing": "hash" if level.use_hash else "direct",
                "occupied_buckets": occupied,
                "occupied_fraction": occupied / level.table_size,
                "collision_fraction": 1.0 - occupied / virtual_entries,
                "mean_vertices_per_occupied_bucket": virtual_entries / occupied,
                "maximum_bucket_load": maximum_load,
            }
        )
    return statistics_per_level


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_callable(
    function: Callable[[], None],
    device: torch.device,
    warmup: int,
    repetitions: int,
) -> float:
    for _ in range(warmup):
        function()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repetitions):
        function()
    synchronize(device)
    return 1000.0 * (time.perf_counter() - start) / repetitions


# Paper: measure the forward and train-step timings reported by the ablation.
def benchmark_encoding(
    encoding: MultiResolutionEncoding,
    device: torch.device,
    batch_size: int,
    warmup: int,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 91_337)
    coordinates = torch.rand((batch_size, 2), generator=generator, device=device)

    def forward() -> None:
        with torch.no_grad():
            encoding(coordinates)

    forward_ms = benchmark_callable(forward, device, warmup, repetitions)

    def forward_backward() -> None:
        encoding.zero_grad(set_to_none=True)
        encoding(coordinates).square().mean().backward()

    forward_backward_ms = benchmark_callable(
        forward_backward, device, warmup, repetitions
    )
    return {
        "batch_size": batch_size,
        "forward_ms": forward_ms,
        "forward_backward_ms": forward_backward_ms,
        "forward_samples_per_second": 1000.0 * batch_size / forward_ms,
        "forward_backward_samples_per_second": (
            1000.0 * batch_size / forward_backward_ms
        ),
    }


def save_ppm(image: torch.Tensor, resolution: int, destination: Path) -> None:
    pixels = (
        image.reshape(resolution, resolution, 3)
        .detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
        .tobytes()
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(
        f"P6\n{resolution} {resolution}\n255\n".encode("ascii") + pixels
    )


def train_one(
    config: EncodingConfig,
    function_name: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    evaluation_coordinates: torch.Tensor,
) -> dict:
    target_function = TARGETS[function_name]
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    encoding = MultiResolutionEncoding(config).to(device)
    # Reset before constructing the decoder so its identical-shape parameters
    # start identically even when encoding table sizes differ.
    torch.manual_seed(seed + 10_000)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 10_000)
    model = FunctionModel(encoding, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.99),
        eps=1.0e-8,
    )
    sample_generator = torch.Generator(device=device)
    sample_generator.manual_seed(seed + 20_000)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    training_start = time.perf_counter()
    curve: list[dict[str, float | int]] = []

    for step in range(1, args.steps + 1):
        coordinates = torch.rand(
            (args.batch_size, 2),
            generator=sample_generator,
            device=device,
        )
        target = target_function(coordinates)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(coordinates)
        loss = torch.mean((prediction - target).square())
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            synchronize(device)
            elapsed = time.perf_counter() - training_start
            metrics, _, _ = evaluate_model(
                model,
                target_function,
                evaluation_coordinates,
                args.eval_resolution,
                args.eval_chunk_size,
            )
            point = {
                "step": step,
                "training_seconds": elapsed,
                "train_batch_mse": float(loss.item()),
                **metrics,
            }
            curve.append(point)
            print(
                f"{function_name:10s} seed={seed:<2d} {config.name:12s} "
                f"step={step:5d}/{args.steps} "
                f"PSNR={metrics['psnr_db']:7.3f} dB "
                f"time={elapsed:8.2f}s",
                flush=True,
            )

    synchronize(device)
    training_seconds = time.perf_counter() - training_start
    final_metrics, prediction, target = evaluate_model(
        model,
        target_function,
        evaluation_coordinates,
        args.eval_resolution,
        args.eval_chunk_size,
        return_images=args.save_predictions,
    )
    timing = benchmark_encoding(
        model.encoding,
        device,
        args.benchmark_batch_size,
        args.benchmark_warmup,
        args.benchmark_repetitions,
        seed,
    )
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )

    if args.save_predictions and prediction is not None and target is not None:
        image_root = (
            args.output
            / "images"
            / function_name
            / f"seed_{seed}"
        )
        save_ppm(
            prediction,
            args.eval_resolution,
            image_root / f"{config.name}.ppm",
        )
        target_path = image_root / "target.ppm"
        if not target_path.exists():
            save_ppm(target, args.eval_resolution, target_path)

    result = {
        "function": function_name,
        "seed": seed,
        "config": asdict(config),
        "output_dimension": model.encoding.output_dim,
        "stored_entries": model.encoding.stored_entries,
        "encoding_parameters": count_parameters(model.encoding),
        "encoding_bytes": parameter_bytes(model.encoding),
        "decoder_parameters": count_parameters(model.decoder),
        "total_parameters": count_parameters(model),
        "total_parameter_bytes": parameter_bytes(model),
        "training_seconds": training_seconds,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "peak_cuda_memory_bytes": peak_memory,
        "final": final_metrics,
        "timing": timing,
        "collision_statistics": collision_statistics(model.encoding),
        "curve": curve,
    }
    return result


def write_summary_csv(results: Sequence[dict], destination: Path) -> None:
    fieldnames = (
        "function",
        "seed",
        "config",
        "resolutions",
        "stored_entries",
        "encoding_parameters",
        "encoding_bytes",
        "total_parameters",
        "training_seconds",
        "psnr_db",
        "mse",
        "gradient_mse",
        "peak_cuda_memory_bytes",
        "benchmark_batch_size",
        "encoding_forward_ms",
        "encoding_forward_backward_ms",
    )
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "function": result["function"],
                    "seed": result["seed"],
                    "config": result["config"]["name"],
                    "resolutions": "-".join(
                        str(value) for value in result["config"]["resolutions"]
                    ),
                    "stored_entries": result["stored_entries"],
                    "encoding_parameters": result["encoding_parameters"],
                    "encoding_bytes": result["encoding_bytes"],
                    "total_parameters": result["total_parameters"],
                    "training_seconds": result["training_seconds"],
                    "psnr_db": result["final"]["psnr_db"],
                    "mse": result["final"]["mse"],
                    "gradient_mse": result["final"]["gradient_mse"],
                    "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"],
                    "benchmark_batch_size": result["timing"]["batch_size"],
                    "encoding_forward_ms": result["timing"]["forward_ms"],
                    "encoding_forward_backward_ms": result["timing"][
                        "forward_backward_ms"
                    ],
                }
            )


def aggregate_results(results: Sequence[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for result in results:
        key = (result["function"], result["config"]["name"])
        groups.setdefault(key, []).append(result)

    aggregates: list[dict] = []
    for (function_name, config_name), group in sorted(groups.items()):
        row: dict[str, object] = {
            "function": function_name,
            "config": config_name,
            "runs": len(group),
        }
        values_by_name = {
            "psnr_db": [item["final"]["psnr_db"] for item in group],
            "mse": [item["final"]["mse"] for item in group],
            "gradient_mse": [
                item["final"]["gradient_mse"] for item in group
            ],
            "training_seconds": [item["training_seconds"] for item in group],
            "encoding_forward_ms": [
                item["timing"]["forward_ms"] for item in group
            ],
            "encoding_forward_backward_ms": [
                item["timing"]["forward_backward_ms"] for item in group
            ],
        }
        for metric_name, values in values_by_name.items():
            row[f"{metric_name}_mean"] = statistics.fmean(values)
            row[f"{metric_name}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        aggregates.append(row)
    return aggregates


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit continuous 2D functions with matched LUT/hash encoders."
    )
    parser.add_argument("--output", type=Path, default=Path("encoding_2d_results"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--configs",
        default="lut,hash_control,hash_2048,hash_1024,hash_512,hash_budget",
        help=f"Comma-separated subset of: {','.join(CONFIGS)}",
    )
    parser.add_argument(
        "--functions",
        default="smooth,multiscale,chirp,localized,checker",
        help=f"Comma-separated subset of: {','.join(TARGETS)}",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--eval-resolution", type=int, default=512)
    parser.add_argument("--eval-chunk-size", type=int, default=262144)
    parser.add_argument("--benchmark-batch-size", type=int, default=262144)
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    parser.add_argument("--benchmark-repetitions", type=int, default=100)
    parser.add_argument("--save-predictions", action="store_true")
    return parser


def validate_selection(
    requested: Iterable[str], available: dict[str, object], label: str
) -> list[str]:
    selected = list(requested)
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"unknown {label}: {', '.join(unknown)}")
    return selected


def main() -> int:
    args = build_parser().parse_args()
    config_names = validate_selection(
        parse_csv(args.configs), CONFIGS, "configurations"
    )
    function_names = validate_selection(
        parse_csv(args.functions), TARGETS, "functions"
    )
    seeds = parse_csv(args.seeds, int)
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch size must be positive")
    if args.log_every <= 0 or args.eval_resolution < 2:
        raise ValueError("log-every must be positive and eval-resolution >= 2")
    if args.benchmark_repetitions <= 0 or args.benchmark_warmup < 0:
        raise ValueError("invalid benchmark repetition counts")

    device = resolve_device(args.device)
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    evaluation_coordinates = make_evaluation_grid(
        args.eval_resolution, device
    )
    print(f"Device: {device}")
    print(f"Output: {args.output}")

    results: list[dict] = []
    for function_name in function_names:
        for seed in seeds:
            for config_name in config_names:
                result = train_one(
                    CONFIGS[config_name],
                    function_name,
                    seed,
                    args,
                    device,
                    evaluation_coordinates,
                )
                results.append(result)
                # Preserve completed runs if a long campaign is interrupted.
                (args.output / "results.json").write_text(
                    json.dumps(results, indent=2) + "\n", encoding="utf-8"
                )
                write_summary_csv(results, args.output / "summary.csv")

    aggregate = aggregate_results(results)
    (args.output / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "selected_configs": config_names,
        "selected_functions": function_names,
        "seeds": seeds,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved raw results to {args.output / 'results.json'}")
    print(f"Saved summary to {args.output / 'summary.csv'}")
    print(f"Saved aggregate statistics to {args.output / 'aggregate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())