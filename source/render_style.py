"""Optimize and render FlaRe appearance stylization with the current OptiX ABI."""

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

source_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(source_dir, "renderer", "output")
if output_path not in sys.path:
    sys.path.insert(0, output_path)

import PYOPTIXFLARERENDERER as flare_renderer

from checkpoint_io import load_model_checkpoint
from scene.dataset_readers import sceneLoadTypeCallbacks
from style_transfer import CLIPGaussianConfig, LatentOptimizationConfig, optimize_latent_style
from utils.camera_utils import cameraList_from_camInfos
from utils.general_utils import safe_state


def ensure_shader_visible():
    """Validate renderer assets resolved relative to the extension module."""
    for name in ("PYOPTIXFLARERENDERER.so", "shaders.cu.ptx"):
        path = os.path.join(output_path, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing built renderer asset: {path}")


def make_camera_args(args):
    return SimpleNamespace(
        resolution=args.resolution,
        data_device=getattr(args, "data_device", "cuda"),
    )


def load_scene_cameras(args):
    source_path = os.path.abspath(args.source_path)
    if os.path.exists(os.path.join(source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](source_path, args.images, True)
    elif os.path.exists(os.path.join(source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        scene_info = sceneLoadTypeCallbacks["Blender"](
            source_path,
            args.bg_color_R,
            args.bg_color_G,
            args.bg_color_B,
            True,
        )
    else:
        raise RuntimeError(
            "Could not recognize scene type. Expected either sparse/ or transforms_train.json in "
            f"{source_path}"
        )
    camera_args = make_camera_args(args)
    train_cameras = cameraList_from_camInfos(scene_info.train_cameras, 1.0, camera_args)
    test_cameras = cameraList_from_camInfos(scene_info.test_cameras, 1.0, camera_args)
    return train_cameras, test_cameras


def load_test_cameras(args):
    return load_scene_cameras(args)[1]


def load_checkpoint(path):
    checkpoint = load_model_checkpoint(path, "cuda")
    rgb = checkpoint.get("RGB")
    if rgb is None:
        rgb = torch.ones_like(checkpoint["m"])
    checkpoint["RGBA"] = torch.cat((rgb, checkpoint["A"]), 1).contiguous()
    return checkpoint


def camera_frame_tensors(camera):
    O = torch.tensor(-camera.R @ camera.T, dtype=torch.float32, device="cuda")
    R = torch.tensor(camera.R.transpose(1, 0)[0, :], dtype=torch.float32, device="cuda")
    D = torch.tensor(camera.R.transpose(1, 0)[1, :], dtype=torch.float32, device="cuda")
    Fv = torch.tensor(camera.R.transpose(1, 0)[2, :], dtype=torch.float32, device="cuda")
    return O, R, D, Fv


def tensor_to_image(rgb, height, width):
    rgb = torch.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    rgb = torch.clamp(rgb, min=0.0, max=1.0).reshape(height, width, 3)
    return Image.fromarray((rgb.detach().cpu().numpy() * 255.0).astype(np.uint8), "RGB")


def save_gt(camera, path):
    gt = camera.original_image[:3].detach().cpu().permute(1, 2, 0).numpy()
    Image.fromarray((np.clip(gt, 0.0, 1.0) * 255.0).astype(np.uint8), "RGB").save(path)


def psnr(pred, gt):
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
    mse = torch.mean((torch.clamp(pred, 0.0, 1.0) - gt) ** 2)
    return (-10.0 * torch.log10(mse)).item()


def prepare_renderer(args, checkpoint, width, height):
    max_batch_size = args.max_batch_size if args.max_batch_size > 0 else width * height
    if max_batch_size < width * height:
        raise ValueError("--max_batch_size must be at least width * height for this renderer.")

    renderer = flare_renderer.CPyOptiXFLARERenderer(
        args.number_of_sides,
        args.chi_square_squared_radius,
        max_batch_size,
    )
    renderer.SetGeometry(
        checkpoint["m"],
        torch.exp(checkpoint["s"]),
        checkpoint["q"],
        torch.sigmoid(checkpoint["A"]),
        1.0 + torch.nn.functional.softplus(checkpoint["k"]),
    )
    return renderer


def update_renderer_geometry(renderer, checkpoint):
    rgb = checkpoint.get("RGB")
    if rgb is None:
        rgb = torch.ones_like(checkpoint["m"])
    checkpoint["RGBA"] = torch.cat((rgb, checkpoint["A"]), 1).contiguous()
    renderer.SetGeometry(
        checkpoint["m"],
        torch.exp(checkpoint["s"]),
        checkpoint["q"],
        torch.sigmoid(checkpoint["A"]),
        1.0 + torch.nn.functional.softplus(checkpoint["k"]),
    )


def packed_weights(checkpoint):
    return (
        torch.cat([checkpoint["w1_uv"], checkpoint["w1_v"], checkpoint["w1_conditioning"]], 1).to(torch.float16),
        checkpoint["w2"].to(torch.float16),
        checkpoint["w3"].to(torch.float16),
    )


def make_rays(camera, width, height):
    O, R, D, Fv = camera_frame_tensors(camera)
    O_chunk = O.repeat(width * height, 1)
    v_chunk = flare_renderer.GenerateRays(
        R, D, Fv, width, height, camera.FoVx, camera.FoVy
    ).reshape(width * height, 3)
    v_chunk = v_chunk / torch.sqrt(torch.sum(v_chunk * v_chunk, 1, keepdim=True))
    return O_chunk, v_chunk.reshape(height, width, 3)


def forward_inference(renderer, checkpoint, camera, conditioning, args, width, height):
    w1_fp16, w2_fp16, w3_fp16 = packed_weights(checkpoint)
    O_chunk, v_chunk = make_rays(camera, width, height)
    img_rgb = torch.zeros(width * height, 3, dtype=torch.float32, device="cuda")
    renderer.Forward_inference(
        conditioning.to(torch.float16),
        checkpoint["features"],
        w1_fp16,
        checkpoint["b1"],
        w2_fp16,
        checkpoint["b2"],
        w3_fp16,
        checkpoint["b3"],
        O_chunk,
        v_chunk,
        img_rgb,
        args.bg_color_R,
        args.bg_color_G,
        args.bg_color_B,
        checkpoint["m"],
        checkpoint["s"],
        checkpoint["q"],
        checkpoint["RGBA"],
        checkpoint["k"],
        args.ray_termination_T_threshold_inference,
    )
    return img_rgb


def pca_embedding_colors(conditioning, percentile=1.0):
    """Map the first three PCA components of per-Gaussian embeddings to RGB."""
    if not 0.0 <= percentile < 50.0:
        raise ValueError("pca_color_percentile must be in [0, 50)")
    values = conditioning.detach().float()
    centered = values - values.mean(dim=0, keepdim=True)
    torch.manual_seed(0)
    _, _, basis = torch.pca_lowrank(centered, q=3, center=False)
    projected = torch.matmul(centered, basis)
    low = torch.quantile(projected, percentile / 100.0, dim=0, keepdim=True)
    high = torch.quantile(projected, 1.0 - (percentile / 100.0), dim=0, keepdim=True)
    colors = (projected - low) / (high - low).clamp_min(1e-8)
    return colors.clamp(0.0, 1.0).contiguous()


def forward_pca(renderer, checkpoint, camera, conditioning, pca_colors, args, width, height):
    """Render PCA colors while retaining FlaRe opacity and compositing."""
    w1_fp16, w2_fp16, w3_fp16 = packed_weights(checkpoint)
    O_chunk, v_chunk = make_rays(camera, width, height)
    img_rgb = torch.zeros(width * height, 3, dtype=torch.float32, device="cuda")
    renderer.Forward_pca(
        conditioning.to(torch.float16), checkpoint["features"],
        w1_fp16, checkpoint["b1"], w2_fp16, checkpoint["b2"],
        w3_fp16, checkpoint["b3"], pca_colors, O_chunk, v_chunk, img_rgb,
        args.bg_color_R, args.bg_color_G, args.bg_color_B,
        checkpoint["m"], checkpoint["s"], checkpoint["q"], checkpoint["RGBA"],
        checkpoint["k"], args.ray_termination_T_threshold_inference,
    )
    return img_rgb


def forward_training(renderer, checkpoint, camera, conditioning_fp16, args, width, height):
    w1_fp16, w2_fp16, w3_fp16 = packed_weights(checkpoint)
    O_chunk, v_chunk = make_rays(camera, width, height)
    img_rgb = torch.zeros(width * height, 3, dtype=torch.float32, device="cuda")
    depth_accums = torch.zeros((width * height, 4), dtype=torch.float32, device="cuda")
    depth_and_index = torch.zeros((width * height, 2), dtype=torch.float32, device="cuda")
    surface_normal = torch.zeros((width * height, 3), dtype=torch.float32, device="cuda")
    normal_accums = torch.zeros((width * height, 4), dtype=torch.float32, device="cuda")
    reg_depth_a = -(args.t_near * args.t_far) / (args.t_far - args.t_near)
    reg_depth_b = args.t_far / (args.t_far - args.t_near)

    renderer.Forward_training(
        conditioning_fp16,
        checkpoint["features"],
        w1_fp16,
        checkpoint["b1"],
        w2_fp16,
        checkpoint["b2"],
        w3_fp16,
        checkpoint["b3"],
        O_chunk,
        v_chunk,
        img_rgb,
        args.bg_color_R,
        args.bg_color_G,
        args.bg_color_B,
        checkpoint["m"],
        checkpoint["s"],
        checkpoint["q"],
        checkpoint["RGBA"],
        checkpoint["k"],
        args.ray_termination_T_threshold_training,
        depth_accums,
        reg_depth_a,
        reg_depth_b,
        depth_and_index,
        surface_normal,
        normal_accums,
    )
    return (img_rgb, O_chunk, v_chunk, depth_accums, reg_depth_a, reg_depth_b,
            depth_and_index, surface_normal, normal_accums)


def reduce_backward_weight_grads(dL_dw1, dL_db1, dL_dw2, dL_db2, dL_dw3, dL_db3):
    dL_dw3 = dL_dw3.sum(0).reshape((8, 2, 32)).transpose(1, 2)
    dL_dw3 = dL_dw3.reshape((8, 8, -1)).transpose(0, 1).flatten(1, 2)

    dL_db3 = dL_db3.sum(0).reshape((2, 32)).transpose(0, 1)
    dL_db3 = dL_db3.reshape((8, 8)).sum(1)

    dL_dw2 = dL_dw2.sum(0).reshape((64, 2, 32)).transpose(1, 2)
    dL_dw2 = dL_dw2.reshape((8, 8, 8, -1)).transpose(1, 2).flatten(0, 1).flatten(1, 2)

    dL_db2 = dL_db2.sum(0).reshape((8, 2, 32)).transpose(1, 2)
    dL_db2 = dL_db2.reshape((8, 8, -1)).sum(2).flatten(0, 1)

    dL_dw1 = dL_dw1.sum(0).reshape((128, 2, 32)).transpose(1, 2)
    dL_dw1 = dL_dw1.reshape((8, 16, 8, -1)).transpose(1, 2).flatten(0, 1).flatten(1, 2)

    dL_db1 = dL_db1.sum(0).reshape((8, 2, 32)).transpose(1, 2)
    dL_db1 = dL_db1.reshape((8, 8, -1)).sum(2).flatten(0, 1)
    return dL_dw1, dL_db1, dL_dw2, dL_db2, dL_dw3, dL_db3


def backward_model(
    renderer,
    checkpoint,
    conditioning_fp16,
    camera_cache,
    img_unclamped,
    img_for_loss,
    dloss_dimage,
    args,
    width,
    height,
):
    w1_fp16, w2_fp16, w3_fp16 = packed_weights(checkpoint)
    (O_chunk, v_chunk, depth_accums, reg_depth_a, reg_depth_b,
     depth_and_index, surface_normal, normal_accums) = camera_cache
    sm_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

    dL_dRGB = torch.zeros_like(checkpoint.get("RGB", checkpoint["m"]))
    dL_dA = torch.zeros_like(checkpoint["A"])
    dL_dk = torch.zeros_like(checkpoint["k"])
    dL_dw3 = torch.zeros((4 * sm_count, 8 * 64), dtype=torch.float32, device="cuda")
    dL_db3 = torch.zeros((4 * sm_count, 8 * 8), dtype=torch.float32, device="cuda")
    dL_dw2 = torch.zeros((4 * sm_count, 64 * 64), dtype=torch.float32, device="cuda")
    dL_db2 = torch.zeros((4 * sm_count, 64 * 8), dtype=torch.float32, device="cuda")
    dL_dw1 = torch.zeros((4 * sm_count, 64 * 128), dtype=torch.float32, device="cuda")
    dL_db1 = torch.zeros((4 * sm_count, 64 * 8), dtype=torch.float32, device="cuda")
    dL_d_conditioning = torch.zeros_like(checkpoint["conditioning_variable"])
    dL_d_features = torch.zeros_like(checkpoint["features"], dtype=torch.float32, device="cuda")
    dL_dm = torch.zeros_like(checkpoint["m"])
    dL_ds = torch.zeros_like(checkpoint["s"])
    dL_dq = torch.zeros_like(checkpoint["q"])
    depth_prefix = torch.zeros((width * height, 4), dtype=torch.float32, device="cuda")

    finite = torch.isfinite(img_unclamped)
    clamped = (~finite) | (img_unclamped != img_for_loss)
    renderer.Backward(
        conditioning_fp16,
        checkpoint["features"],
        w1_fp16,
        checkpoint["b1"],
        w2_fp16,
        checkpoint["b2"],
        w3_fp16,
        checkpoint["b3"],
        O_chunk,
        v_chunk,
        args.bg_color_R,
        args.bg_color_G,
        args.bg_color_B,
        checkpoint["m"],
        checkpoint["s"],
        checkpoint["q"],
        checkpoint["RGBA"],
        checkpoint["k"],
        img_for_loss,
        dloss_dimage * (~clamped),
        dL_dRGB,
        dL_dA,
        dL_dk,
        dL_dw3,
        dL_db3,
        dL_dw2,
        dL_db2,
        dL_dw1,
        dL_db1,
        dL_d_conditioning,
        dL_d_features,
        dL_dm,
        dL_ds,
        dL_dq,
        args.ray_termination_T_threshold_training,
        depth_accums,
        depth_prefix,
        0.0,
        reg_depth_a,
        reg_depth_b,
        depth_and_index,
        surface_normal,
        normal_accums,
        0.0,
    )
    dL_dw1, dL_db1, dL_dw2, dL_db2, dL_dw3, dL_db3 = reduce_backward_weight_grads(
        dL_dw1, dL_db1, dL_dw2, dL_db2, dL_dw3, dL_db3
    )
    return {
        "w1_uv": dL_dw1[:, 0:8],
        "w1_v": dL_dw1[:, 8:32],
        "w1_conditioning": dL_dw1[:, 32:128],
        "b1": dL_db1,
        "w2": dL_dw2,
        "b2": dL_db2,
        "w3": torch.nn.functional.pad(dL_dw3, (0, 0, 0, 8), "constant", 0.0),
        "b3": torch.nn.functional.pad(dL_db3, (0, 8), "constant", 0.0),
        "features": dL_d_features,
        "conditioning_variable": dL_d_conditioning,
        "A": dL_dA,
        "k": dL_dk,
        "m": dL_dm,
        "s": dL_ds,
        "q": dL_dq,
    }


def backward_conditioning(
    renderer,
    checkpoint,
    conditioning_fp16,
    camera_cache,
    img_unclamped,
    img_for_loss,
    dloss_dimage,
    args,
    width,
    height,
):
    return backward_model(
        renderer,
        checkpoint,
        conditioning_fp16,
        camera_cache,
        img_unclamped,
        img_for_loss,
        dloss_dimage,
        args,
        width,
        height,
    )["conditioning_variable"]


def build_style_configs(args):
    style_config = CLIPGaussianConfig(
        style_prompt=args.style_prompt,
        style_image=args.style_image,
        vgg_weights=args.vgg_weights,
        object_prompt=args.object_prompt,
        clip_model=args.clip_model,
        clip_backend=args.clip_backend,
        lambda_dir=args.lambda_dir,
        lambda_patch=args.lambda_patch,
        lambda_content=args.lambda_content,
        lambda_bg=args.lambda_bg,
        object_background=args.style_object_background,
        crop_size=args.style_crop_size,
        num_crops=args.style_num_crops,
        background=(args.bg_color_R, args.bg_color_G, args.bg_color_B),
    )
    latent_config = LatentOptimizationConfig(
        steps=args.style_steps,
        lr=args.style_lr,
        save_every=args.style_save_every,
        num_views=args.style_num_views,
        strength=args.style_strength,
        latent_clip=args.style_latent_clip,
        latent_mode=args.style_latent_mode,
        latent_reg_weight=args.style_latent_reg_weight,
        grad_clip=args.style_grad_clip,
        mask_threshold=args.style_mask_threshold,
        finetune_model=not args.no_style_model_finetune,
        model_lr=args.style_model_lr,
        model_reg_weight=args.style_model_reg_weight,
        scale_reg_weight=args.style_scale_reg_weight,
        model_grad_clip=args.style_model_grad_clip,
        freeze_view_branch=not args.style_train_view_branch,
        finetune_geometry=not args.no_style_geometry_finetune,
        geometry_lr_m=args.style_geometry_lr_m,
        geometry_lr_s=args.style_geometry_lr_s,
        geometry_lr_q=args.style_geometry_lr_q,
        geometry_lr_A=args.style_geometry_lr_A,
        geometry_lr_k=args.style_geometry_lr_k,
        geometry_grad_clip=args.style_geometry_grad_clip,
        min_s_coef=args.style_min_s_coef,
        max_s_coef=args.style_max_s_coef,
    )
    return style_config, latent_config


def render_test_set(args):
    safe_state(args.quiet)
    ensure_shader_visible()

    checkpoint_path = os.path.abspath(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    train_cameras, test_cameras = load_scene_cameras(args)
    if len(test_cameras) == 0:
        raise RuntimeError("No test cameras found for this dataset.")

    first = test_cameras[0]
    width = first.image_width
    height = first.image_height
    default_output = os.path.join(os.path.dirname(checkpoint_path), "test_renders")
    render_dir = os.path.abspath(args.output_path or default_output)
    gt_dir = os.path.join(render_dir, "gt")
    os.makedirs(render_dir, exist_ok=True)
    if args.save_gt:
        os.makedirs(gt_dir, exist_ok=True)

    run_config = dict(vars(args))
    run_config.update(
        {
            "checkpoint": checkpoint_path,
            "source_path": os.path.abspath(args.source_path),
            "output_path": render_dir,
            "argv": sys.argv,
        }
    )
    config_path = os.path.join(render_dir, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(run_config, config_file, indent=2, sort_keys=True)

    renderer = prepare_renderer(args, checkpoint, width, height)

    base_conditioning = checkpoint["conditioning_variable"]
    if args.zero_conditioning_dim >= 0:
        base_conditioning = base_conditioning.clone()
        base_conditioning[:, args.zero_conditioning_dim] = 0.0

    if args.style_image or args.style_prompt:
        style_config, latent_config = build_style_configs(args)
        style_cameras = train_cameras if args.style_camera_split == "train" else test_cameras
        if len(style_cameras) == 0:
            style_cameras = test_cameras
        print(f"Optimizing style on {len(style_cameras)} {args.style_camera_split} cameras")
        conditioning, checkpoint = optimize_latent_style(
            style_config=style_config,
            latent_config=latent_config,
            renderer=renderer,
            checkpoint=checkpoint,
            base_conditioning=base_conditioning,
            cameras=style_cameras,
            width=width,
            height=height,
            render_dir=render_dir,
            forward_inference=forward_inference,
            forward_training=forward_training,
            backward_conditioning=backward_conditioning,
            render_args=args,
            backward_model=backward_model,
            update_geometry=update_renderer_geometry,
        )
    else:
        conditioning = base_conditioning

    pca_colors = None
    if args.render_pca_embeddings:
        pca_colors = pca_embedding_colors(conditioning, args.pca_color_percentile)
        torch.save(
            {"colors": pca_colors.detach().cpu(), "conditioning": conditioning.detach().cpu()},
            os.path.join(render_dir, "pca_embedding_colors.pt"),
        )

    psnr_values = []
    start = time.perf_counter()

    with torch.no_grad():
        for idx, camera in enumerate(tqdm(test_cameras, desc="Rendering test set")):
            if camera.image_width != width or camera.image_height != height:
                raise RuntimeError(
                    "All test cameras must have the same rendered resolution for this script. "
                    "Use --resolution to force a consistent resolution."
                )

            if pca_colors is None:
                img_rgb = forward_inference(renderer, checkpoint, camera, conditioning, args, width, height)
            else:
                img_rgb = forward_pca(renderer, checkpoint, camera, conditioning, pca_colors, args, width, height)
            image_name = camera.image_name or f"{idx:05d}"
            suffix = "pca" if pca_colors is not None else ("styled" if (args.style_image or args.style_prompt) else "render")
            output_file = os.path.join(
                render_dir, f"{idx:05d}_{image_name}_{suffix}.png"
            )
            tensor_to_image(img_rgb, height, width).save(output_file)

            gt = camera.original_image[:3].reshape(3, height * width).transpose(0, 1)
            if pca_colors is None:
                psnr_values.append(psnr(img_rgb, gt))
            if args.save_gt:
                save_gt(camera, os.path.join(gt_dir, f"{idx:05d}_{image_name}.png"))

    elapsed = time.perf_counter() - start
    metrics_path = os.path.join(render_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        for idx, value in enumerate(psnr_values):
            f.write(f"{idx}: {value}\n")
        if psnr_values:
            f.write(f"AVG_PSNR: {float(np.mean(psnr_values))}\n")
        else:
            f.write("MODE: PCA embedding colors\n")
        f.write(f"FPS: {len(test_cameras) / elapsed}\n")

    print(f"Rendered {len(test_cameras)} test images to {render_dir}")
    if psnr_values:
        print(f"Average PSNR: {float(np.mean(psnr_values))}")
    else:
        print("Rendered PCA embedding colors; PSNR is not applicable.")
    print(f"FPS: {len(test_cameras) / elapsed}")


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize and render a FlaRe style transfer.")
    parser.add_argument(
        "--source_path",
        required=True,
        help="Dataset path containing sparse/ or transforms_train.json.",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to iter_*.checkpoint.")
    parser.add_argument("--output_path", default="", help="Directory for rendered PNGs. Defaults next to checkpoint.")
    parser.add_argument("--resolution", default=1, type=int)
    parser.add_argument("--images", default="images")
    parser.add_argument("--bg_color_R", default=0.0, type=float)
    parser.add_argument("--bg_color_G", default=0.0, type=float)
    parser.add_argument("--bg_color_B", default=0.0, type=float)
    parser.add_argument("--number_of_sides", default=8, type=int)
    parser.add_argument("--chi_square_squared_radius", default=11.3449, type=float)
    parser.add_argument("--ray_termination_T_threshold_inference", default=0.01, type=float)
    parser.add_argument("--ray_termination_T_threshold_training", default=0.0001, type=float)
    parser.add_argument("--t_near", default=0.1, type=float)
    parser.add_argument("--t_far", default=1000.0, type=float)
    parser.add_argument("--max_batch_size", default=-1, type=int)
    parser.add_argument("--save_gt", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--zero_conditioning_dim",
        default=43,
        type=int,
        help="Descriptor dimension to zero before style optimization (default: 43; use -1 to disable).",
    )
    parser.add_argument(
        "--render_pca_embeddings",
        action="store_true",
        help="Render each Gaussian using RGB from PC1-PC3 of its appearance embedding.",
    )
    parser.add_argument(
        "--pca_color_percentile",
        default=1.0,
        type=float,
        help="Per-channel percentile clipped when mapping PCA coordinates to RGB.",
    )

    parser.add_argument("--style_image", default="", help="Reference style image for CLIPGaussian fine-tuning.")
    parser.add_argument("--style_prompt", default="", help="Text style prompt for CLIPGaussian fine-tuning.")
    parser.add_argument("--object_prompt", default="a Photo")
    parser.add_argument("--clip_model", default="ViT-B/32")
    parser.add_argument("--clip_backend", default="auto", choices=["auto", "openai", "open_clip"])
    parser.add_argument(
        "--vgg_weights",
        default="default",
        choices=["default", "none"],
        help=(
            "Use pretrained ImageNet VGG19 weights, or random weights for "
            "offline pipeline smoke tests."
        ),
    )
    parser.add_argument("--lambda_dir", "--style_weight", default=5.0, type=float)
    parser.add_argument("--lambda_patch", default=90.0, type=float)
    parser.add_argument("--lambda_content", "--content_weight", default=0.8, type=float)
    parser.add_argument("--lambda_bg", "--background_weight", default=1000.0, type=float)
    parser.add_argument(
        "--style_object_background",
        action="store_true",
        help=(
            "Enable CLIPGaussian object-mode background preservation loss; "
            "official examples leave this off unless object mode is used."
        ),
    )
    parser.add_argument("--style_crop_size", default=128, type=int)
    parser.add_argument("--style_num_crops", default=64, type=int)
    parser.add_argument("--style_steps", default=5000, type=int)
    parser.add_argument("--style_lr", default=0.005, type=float)
    parser.add_argument(
        "--style_num_views",
        default=1,
        type=int,
        help="Number of randomly sampled camera views per style optimization step.",
    )
    parser.add_argument(
        "--style_camera_split",
        default="train",
        choices=["train", "test"],
        help=(
            "Camera split used for style optimization; final rendering still "
            "uses the test split."
        ),
    )
    parser.add_argument("--style_save_every", default=100, type=int)
    parser.add_argument("--style_strength", default=1.0, type=float)
    parser.add_argument("--style_latent_clip", default=0.25, type=float)
    parser.add_argument("--style_latent_mode", default="shift", choices=["shift", "affine"])
    parser.add_argument("--style_latent_reg_weight", default=5e-2, type=float)
    parser.add_argument("--style_grad_clip", default=1.0, type=float)
    parser.add_argument("--style_mask_threshold", default=0.02, type=float)
    parser.add_argument("--no_style_model_finetune", action="store_true")
    parser.add_argument("--style_model_lr", default=5e-3, type=float)
    parser.add_argument("--style_model_reg_weight", default=1e-4, type=float)
    parser.add_argument("--style_scale_reg_weight", default=1e-3, type=float)
    parser.add_argument("--style_model_grad_clip", default=1.0, type=float)
    parser.add_argument(
        "--style_train_view_branch",
        action="store_true",
        help="Fine-tune the view-direction MLP branch w1_v during style optimization.",
    )
    parser.add_argument("--no_style_geometry_finetune", action="store_true")
    parser.add_argument("--style_geometry_lr_m", default=1e-4, type=float)
    parser.add_argument("--style_geometry_lr_s", default=5e-3, type=float)
    parser.add_argument("--style_geometry_lr_q", default=1e-3, type=float)
    parser.add_argument("--style_geometry_lr_A", default=1e-3, type=float)
    parser.add_argument("--style_geometry_lr_k", default=1e-4, type=float)
    parser.add_argument("--style_geometry_grad_clip", default=0.1, type=float)
    parser.add_argument("--style_min_s_coef", default=0.00034641, type=float)
    parser.add_argument("--style_max_s_coef", default=0.05, type=float)
    parser.add_argument("--style_color_weight", default=0.0, type=float, help=argparse.SUPPRESS)
    parser.add_argument("--edge_weight", default=0.0, type=float, help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    render_test_set(parse_args())
