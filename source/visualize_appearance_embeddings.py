"""Render PCA projections of FlaRe per-Gaussian appearance embeddings to PNG."""

import argparse
import csv
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from checkpoint_io import load_model_checkpoint

BACKGROUND = (14, 17, 23)
PANEL = (22, 27, 34)
GRID = (48, 54, 61)
TEXT = (230, 237, 243)
MUTED = (139, 148, 158)


def load_embedding_data(checkpoint_path: str) -> Dict[str, Optional[np.ndarray]]:
    checkpoint = load_model_checkpoint(checkpoint_path, "cpu")
    embeddings = checkpoint["conditioning_variable"].detach().float().cpu().numpy()
    positions = checkpoint["m"].detach().float().cpu().numpy()
    opacity_logits = checkpoint["A"].detach().float().cpu().numpy()
    opacities = 1.0 / (1.0 + np.exp(-np.clip(opacity_logits.reshape(-1), -80.0, 80.0)))
    return {"embeddings": embeddings, "positions": positions, "opacities": opacities}


def compute_pca(embeddings: np.ndarray, components: int, seed: int):
    values = torch.from_numpy(embeddings).float()
    mean = values.mean(dim=0, keepdim=True)
    centered = values - mean
    component_count = min(components, centered.shape[0], centered.shape[1])
    torch.manual_seed(seed)
    _, singular_values, basis = torch.pca_lowrank(centered, q=component_count, center=False)
    projection = centered @ basis
    total_variance = centered.square().sum().clamp_min(torch.finfo(centered.dtype).eps)
    variance_ratio = singular_values.square() / total_variance
    return projection.numpy(), basis.t().numpy(), variance_ratio.numpy(), mean.squeeze(0).numpy()


def sample_indices(count: int, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or count <= max_points:
        return np.arange(count)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=max_points, replace=False))


def save_csv(path, projection, embedding_norm, positions, opacities):
    header = ["gaussian_index", *[f"pc{i + 1}" for i in range(projection.shape[1])], "embedding_norm"]
    if positions is not None:
        header.extend(["x", "y", "z"])
    if opacities is not None:
        header.append("opacity")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index in range(projection.shape[0]):
            row = [index, *projection[index].tolist(), float(embedding_norm[index])]
            if positions is not None:
                row.extend(positions[index, :3].tolist())
            if opacities is not None:
                row.append(float(opacities[index]))
            writer.writerow(row)


def select_color_values(color_by, projection, embedding_norm, positions, opacities):
    if color_by == "opacity":
        if opacities is not None:
            return opacities, "Opacity"
        return embedding_norm, "Embedding norm (opacity unavailable)"
    if color_by == "norm":
        return embedding_norm, "Embedding norm"
    if color_by in {"x", "y", "z"}:
        if positions is None:
            raise ValueError(f"--color_by {color_by} requires positions in the checkpoint")
        axis = {"x": 0, "y": 1, "z": 2}[color_by]
        return positions[:, axis], f"Position {color_by.upper()}"
    if color_by in {"pc1", "pc2", "pc3"}:
        return projection[:, int(color_by[-1]) - 1], color_by.upper()
    raise ValueError(f"Unknown color mode: {color_by}")


def robust_range(values: np.ndarray) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, [1.0, 99.0])
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def viridis(value: float):
    stops = np.asarray(
        [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
        dtype=np.float32,
    )
    value = float(np.clip(value, 0.0, 1.0)) * (len(stops) - 1)
    index = min(len(stops) - 2, int(value))
    fraction = value - index
    return tuple(np.rint(stops[index] * (1.0 - fraction) + stops[index + 1] * fraction).astype(np.uint8))


def draw_projection(
    projection,
    sampled_indices,
    color_values,
    color_label,
    variance_ratio,
    x_component,
    y_component,
    width,
    height,
    point_radius,
):
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    left, top, right, bottom = 82, 58, width - 34, height - 76
    draw.rounded_rectangle((12, 12, width - 12, height - 12), radius=12, fill=PANEL, outline=GRID)

    points = projection[sampled_indices]
    xs, ys = points[:, x_component], points[:, y_component]
    x_low, x_high = robust_range(xs)
    y_low, y_high = robust_range(ys)
    x_pad, y_pad = 0.05 * (x_high - x_low), 0.05 * (y_high - y_low)
    x_low, x_high = x_low - x_pad, x_high + x_pad
    y_low, y_high = y_low - y_pad, y_high + y_pad

    def map_x(value):
        return left + (value - x_low) / (x_high - x_low) * (right - left)

    def map_y(value):
        return bottom - (value - y_low) / (y_high - y_low) * (bottom - top)

    for tick in range(6):
        fraction = tick / 5.0
        px = left + fraction * (right - left)
        py = top + fraction * (bottom - top)
        draw.line((px, top, px, bottom), fill=GRID, width=1)
        draw.line((left, py, right, py), fill=GRID, width=1)
        x_value = x_low + fraction * (x_high - x_low)
        y_value = y_high - fraction * (y_high - y_low)
        draw.text((px - 17, bottom + 8), f"{x_value:.2f}", font=font, fill=MUTED)
        draw.text((20, py - 5), f"{y_value:.2f}", font=font, fill=MUTED)
    draw.rectangle((left, top, right, bottom), outline=(75, 82, 90), width=1)

    color_low, color_high = robust_range(color_values[sampled_indices])
    normalized = np.clip(
        (color_values[sampled_indices] - color_low) / (color_high - color_low), 0.0, 1.0
    )
    order = np.argsort(normalized)
    for local_index in order:
        x, y = map_x(xs[local_index]), map_y(ys[local_index])
        color = viridis(normalized[local_index]) + (175,)
        draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), fill=color)

    x_number, y_number = x_component + 1, y_component + 1
    title = (
        f"PC{x_number} vs PC{y_number}   "
        f"({100 * variance_ratio[x_component]:.2f}% / {100 * variance_ratio[y_component]:.2f}% variance)"
    )
    draw.text((left, 29), title, font=font, fill=TEXT)
    draw.text(((left + right) // 2 - 20, height - 43), f"PC{x_number}", font=font, fill=TEXT)
    draw.text((18, 29), f"PC{y_number}", font=font, fill=TEXT)

    bar_x0, bar_x1, bar_y0, bar_y1 = right - 160, right, height - 42, height - 29
    for x in range(bar_x0, bar_x1 + 1):
        fraction = (x - bar_x0) / max(1, bar_x1 - bar_x0)
        draw.line((x, bar_y0, x, bar_y1), fill=viridis(fraction) + (255,))
    draw.text((bar_x0, bar_y1 + 3), f"{color_low:.3g}", font=font, fill=MUTED)
    draw.text((bar_x1 - 32, bar_y1 + 3), f"{color_high:.3g}", font=font, fill=MUTED)
    draw.text((bar_x0, bar_y0 - 13), color_label, font=font, fill=TEXT)
    return image


def save_pngs(output_dir, projection, variance_ratio, sampled_indices, color_values, color_label, size, point_radius):
    pairs = [(0, 1, "pc1_pc2"), (0, 2, "pc1_pc3"), (1, 2, "pc2_pc3")]
    images = []
    for x_component, y_component, name in pairs:
        image = draw_projection(
            projection, sampled_indices, color_values, color_label, variance_ratio,
            x_component, y_component, size, size, point_radius,
        )
        image.save(output_dir / f"appearance_pca_{name}.png")
        images.append(image)
    combined = Image.new("RGB", (size * 3, size), BACKGROUND)
    for index, image in enumerate(images):
        combined.paste(image, (index * size, 0))
    combined.save(output_dir / "appearance_pca.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Render PCA projections of FlaRe appearance embeddings to PNG.")
    parser.add_argument("checkpoint", help="Path to iter_*.checkpoint or a compatible .pt file")
    parser.add_argument("--output_dir", default="", help="Default: <checkpoint directory>/appearance_pca")
    parser.add_argument("--components", type=int, default=16, help="Number of PCA components to export")
    parser.add_argument("--max_plot_points", type=int, default=50000, help="Maximum points rendered per panel")
    parser.add_argument("--color_by", choices=["opacity", "norm", "x", "y", "z", "pc1", "pc2", "pc3"], default="opacity")
    parser.add_argument("--size", type=int, default=700, help="Width and height of each PCA panel")
    parser.add_argument("--point_radius", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_embedding_data(args.checkpoint)
    embeddings, positions, opacities = data["embeddings"], data["positions"], data["opacities"]
    if embeddings.shape[0] < 3:
        raise ValueError("At least three embeddings are required for PCA visualization")
    if positions is not None and positions.shape[0] != embeddings.shape[0]:
        raise ValueError("Position and embedding counts do not match")
    if opacities is not None and opacities.shape[0] != embeddings.shape[0]:
        raise ValueError("Opacity and embedding counts do not match")

    projection, basis, variance_ratio, embedding_mean = compute_pca(
        embeddings, max(3, args.components), args.seed
    )
    embedding_norm = np.linalg.norm(embeddings, axis=1)
    color_values, color_label = select_color_values(
        args.color_by, projection, embedding_norm, positions, opacities
    )
    sampled_indices = sample_indices(embeddings.shape[0], args.max_plot_points, args.seed)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent / "appearance_pca"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_pngs(
        output_dir, projection, variance_ratio, sampled_indices, color_values,
        color_label, args.size, args.point_radius,
    )
    np.savez_compressed(
        output_dir / "appearance_pca.npz", projection=projection, components=basis,
        explained_variance_ratio=variance_ratio, embedding_mean=embedding_mean,
        embedding_norm=embedding_norm, positions=positions if positions is not None else np.empty((0, 3)),
        opacities=opacities if opacities is not None else np.empty((0,)),
    )
    save_csv(output_dir / "appearance_pca.csv", projection, embedding_norm, positions, opacities)
    print(f"Embeddings: {embeddings.shape[0]} x {embeddings.shape[1]}")
    print("Explained variance:", ", ".join(f"PC{i + 1}={100 * value:.2f}%" for i, value in enumerate(variance_ratio[:3])))
    print(f"Rendered: {output_dir / 'appearance_pca.png'}")


if __name__ == "__main__":
    main()
