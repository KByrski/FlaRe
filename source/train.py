#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the 3DGS_LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import ctypes
import tkinter as tk
import time  # !!! !!! !!!
import os
import random
import gc
import json
import shutil
import torch
import torch.nn as nn
import math
import torch.optim as optim
import numpy as np
from PIL import Image, ImageTk
from random import randint
from utils.loss_utils import l1_loss, ssim
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.depth_utils import ray_parameters_to_depth_and_normal
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import EssentialParams, PerformanceParams, LearningParams, ApplicationParams, parse_args_with_config
# ### ### ### ### ###

def set_all_seeds(seed=0):
    """Seed the random number generators used during training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_all_seeds(0)
# ### ### ### ### ###

# !!! !!! !!!
device_id = torch.cuda.current_device()
prop = torch.cuda.get_device_properties(device_id)
SM_count = prop.multi_processor_count
# !!! !!! !!!

# ### ### ### ### ###

if os.name == "nt":
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
    os.add_dll_directory(torch_lib_path)
    os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin")
renderer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "renderer", "output"))
if renderer_path not in sys.path:
    sys.path.insert(0, renderer_path)
import PYOPTIXFLARERENDERER as py_OptiX_FLARE_renderer
# ### ### ### ### ###

def GenerateRays(
        indices,
        O, R, D, F,
        width, height, number_of_poses,
        fov_X, fov_Y
):
    double_tan_half_fov_X = 2.0 * np.tan(0.5 * fov_X)
    double_tan_half_fov_Y = 2.0 * np.tan(0.5 * fov_Y)
    indices = indices.to(torch.int64).unsqueeze(1)
    poses_indices = indices // (width * height)
    pixel_indices = indices % (width * height)
    y = pixel_indices // width
    x = pixel_indices % width
    d_x = (-0.5 + ((x + 0.5) / width)) * double_tan_half_fov_X
    d_y = (-0.5 + ((y + 0.5) / height)) * double_tan_half_fov_Y
    d_z = 1.0
    R = torch.gather(R, 0, poses_indices.expand(-1, 3))
    D = torch.gather(D, 0, poses_indices.expand(-1, 3))
    F = torch.gather(F, 0, poses_indices.expand(-1, 3))
    O = torch.gather(O, 0, poses_indices.expand(-1, 3))
    v = (R * d_x) + (D * d_y) + (F * d_z)
    return (O, v)
def maybe_empty_cuda_cache():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
def _checkpoint_root():
    return os.path.join("output", str(next_available_dir_id), "checkpoints")
def _checkpoint_metadata_path():
    return os.path.join(_checkpoint_root(), "checkpoint_metadata.json")
def _write_checkpoint_metadata():
    path = _checkpoint_metadata_path()
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as metadata_file:
        json.dump(checkpoint_metadata, metadata_file, indent=2)
        metadata_file.write("\n")
    os.replace(temporary_path, path)
def _checkpoint_payload():
    return (
        RGB, A, k,
        w1_uv, w1_v, w1_conditioning, b1, w2, b2, w3, b3,
        conditioning_variable, features, m, s, q,
        optimizer.state_dict(), training_time
    )
def _prune_legacy_checkpoints():
    """Remove old per-iteration files once both rolling checkpoints exist."""
    root = _checkpoint_root()
    keep = {
        os.path.abspath(os.path.join(root, "best.checkpoint")),
        os.path.abspath(os.path.join(root, "last.checkpoint")),
    }
    for directory, _, files in os.walk(root, topdown=False):
        for filename in files:
            path = os.path.abspath(os.path.join(directory, filename))
            if filename.endswith(".checkpoint") and path not in keep:
                os.remove(path)
        if directory != root:
            try:
                os.rmdir(directory)
            except OSError:
                pass
def SaveCheckpoint(role, metric_name=None, metric_value=None):
    """Replace either the best or last rolling checkpoint."""
    if role not in ("best", "last"):
        raise ValueError("Checkpoint role must be 'best' or 'last'")
    root = _checkpoint_root()
    os.makedirs(root, exist_ok=True)
    destination = os.path.join(root, role + ".checkpoint")
    temporary_path = destination + ".tmp"
    if os.path.exists(temporary_path):
        os.remove(temporary_path)
    # Reclaim the previous latest file before serialization. The best checkpoint
    # remains available if training is interrupted during this write, and peak
    # checkpoint storage never grows to three full checkpoint files.
    if role == "last" and os.path.exists(destination):
        os.remove(destination)
    last_path = os.path.join(root, "last.checkpoint")
    last_metadata = checkpoint_metadata.get("last", {})
    can_link_last = (
            role == "best"
            and os.path.isfile(last_path)
            and last_metadata.get("iteration") == iteration
    )
    if can_link_last:
        try:
            os.link(last_path, temporary_path)
        except OSError:
            shutil.copy2(last_path, temporary_path)
    else:
        torch.save(_checkpoint_payload(), temporary_path)
    os.replace(temporary_path, destination)
    entry = {"iteration": int(iteration)}
    if metric_name is not None:
        entry["metric"] = metric_name
        entry["value"] = float(metric_value)
    checkpoint_metadata[role] = entry
    _write_checkpoint_metadata()
    if os.path.isfile(os.path.join(root, "best.checkpoint")) and os.path.isfile(last_path):
        _prune_legacy_checkpoints()
    print("Saved " + role + " checkpoint: " + destination, flush=True)
def LaunchFinalEvaluation():
    """Replace training so its GPU allocations are released before rendering."""
    evaluate_script = os.path.join(os.path.dirname(__file__), "evaluate.py")
    model_path = os.path.abspath(os.path.join("output", str(next_available_dir_id)))
    command = [
        sys.executable, evaluate_script,
        "--scene_path", os.path.abspath(ep.source_path),
        "--model_path", model_path,
        "--resolution", str(ep.resolution),
        "--images", ep.images,
        "--bg_color_R", str(ep.bg_color_R),
        "--bg_color_G", str(ep.bg_color_G),
        "--bg_color_B", str(ep.bg_color_B),
        "--number_of_sides", str(pp.number_of_sides),
        "--ray_termination_T_threshold_inference",
        str(pp.ray_termination_T_threshold_inference),
        "--skip_metrics",
    ]
    print("Training finished. Rendering the complete test set from the best checkpoint...", flush=True)
    os.execv(sys.executable, command)
# ### ### ### ### ###

parser = ArgumentParser(description="Training script parameters")

# ### ### ### ### ###

ep = EssentialParams(parser)
pp = PerformanceParams(parser)
lp = LearningParams(parser)
ap = ApplicationParams(parser)
args = parse_args_with_config(parser)
ep = ep.extract(args)
pp = pp.extract(args)
lp = lp.extract(args)
ap = ap.extract(args)
# ### ### ### ### ###

# !!! !!! !!!
if (ap.real_time_preview):
    from pynput.mouse import Controller
    mouse = Controller()
# !!! !!! !!!

# ### ### ### ### ###

reg_depth_a = -(ep.t_near * ep.t_far) / (ep.t_far - ep.t_near)
reg_depth_b = ep.t_far / (ep.t_far - ep.t_near)
# ### ### ### ### ###

# !!! !!! !!!
safe_state(False)
gaussians = GaussianModel()
if (ep.model_path == ""):
    # Create directories
    if (not os.path.exists("output")):
        os.makedirs("output")
    next_available_dir_id = 0
    for entry in os.listdir("output"):
        full_path = os.path.join("output", entry)
        if os.path.isdir(full_path):
            if (int(entry) > next_available_dir_id):
                next_available_dir_id = int(entry)
    next_available_dir_id += 1
    os.makedirs(os.path.join("output", str(next_available_dir_id)))
    os.makedirs(os.path.join("output", str(next_available_dir_id), "checkpoints"))
    os.makedirs(os.path.join("output", str(next_available_dir_id), "renders"))
    os.makedirs(os.path.join("output", str(next_available_dir_id), "screenshots"))
    os.makedirs(os.path.join("output", str(next_available_dir_id), "stats"))
    # ##############################################################################################

    # Train model from scratch
    scene = Scene(ep, pp, lp, gaussians)
    gaussians.training_setup(lp)
else:
    next_available_dir_id = int(ep.model_path)
    # Load model from iteration
    scene = Scene(ep, pp, lp, gaussians, ep.start_iter)
# !!! !!! !!!

# ### ### ### ### ###

width = scene.train_cameras[1.0][0].image_width
height = scene.train_cameras[1.0][0].image_height
bitmap_width = int(width * ap.preview_resolution_scale)
bitmap_height = int(height * ap.preview_resolution_scale)
max_batch_size = max(width * height, bitmap_width * bitmap_height)
# ### ### ### ### ###

number_of_poses = len(scene.train_cameras[1.0])
fov_x_list = []
fov_y_list = []
O_list = []
R_list = []
D_list = []
F_list = []
bitmap_list = []
alpha_list = []
for pose in range(number_of_poses):
    fov_x_pose = torch.tensor(scene.train_cameras[1.0][pose].FoVx, dtype=torch.float32, device="cuda").unsqueeze(0)
    fov_y_pose = torch.tensor(scene.train_cameras[1.0][pose].FoVy, dtype=torch.float32, device="cuda").unsqueeze(0)
    O_pose = torch.tensor(-scene.train_cameras[1.0][pose].R @ scene.train_cameras[1.0][pose].T, dtype=torch.float32,
                          device="cuda").unsqueeze(0)
    R_pose = torch.tensor(scene.train_cameras[1.0][pose].R.transpose(1, 0)[0, :], dtype=torch.float32,
                          device="cuda").unsqueeze(0)
    D_pose = torch.tensor(scene.train_cameras[1.0][pose].R.transpose(1, 0)[1, :], dtype=torch.float32,
                          device="cuda").unsqueeze(0)
    F_pose = torch.tensor(scene.train_cameras[1.0][pose].R.transpose(1, 0)[2, :], dtype=torch.float32,
                          device="cuda").unsqueeze(0)
    bitmap_pose = scene.train_cameras[1.0][pose].foreground_image.to(dtype=torch.float32).reshape(3,
                                                                                                  height * width).transpose(
        0, 1)
    alpha_pose = scene.train_cameras[1.0][pose].gt_alpha_mask.to(dtype=torch.float32).reshape(height * width, 1)
    # ### ### ### ### ###

    fov_x_list.append(fov_x_pose)
    fov_y_list.append(fov_y_pose)
    O_list.append(O_pose)
    R_list.append(R_pose)
    D_list.append(D_pose)
    F_list.append(F_pose)
    bitmap_list.append(bitmap_pose)
    alpha_list.append(alpha_pose)
fov_x = torch.cat(fov_x_list, 0)
fov_y = torch.cat(fov_y_list, 0)
O = torch.cat(O_list, 0)
R = torch.cat(R_list, 0)
D = torch.cat(D_list, 0)
F = torch.cat(F_list, 0)
bitmap = torch.cat(bitmap_list, 0)
alpha = torch.cat(alpha_list, 0)
# ### ### ### ### ###

number_of_poses_test = len(scene.test_cameras[1.0])
fov_x_list = []
fov_y_list = []
O_list = []
R_list = []
D_list = []
F_list = []
bitmap_list = []
alpha_list = []
for pose in range(number_of_poses_test):
    fov_x_pose = torch.tensor(scene.test_cameras[1.0][pose].FoVx, dtype=torch.float32, device="cuda").unsqueeze(0)
    fov_y_pose = torch.tensor(scene.test_cameras[1.0][pose].FoVy, dtype=torch.float32, device="cuda").unsqueeze(0)
    O_pose = torch.tensor(-scene.test_cameras[1.0][pose].R @ scene.test_cameras[1.0][pose].T, dtype=torch.float32,
                          device="cuda").unsqueeze(0)
    R_pose = torch.tensor(scene.test_cameras[1.0][pose].R.transpose(1, 0)[0, :], dtype=torch.float32,
                          device="cuda").unsqueeze(0)
    D_pose = torch.tensor(scene.test_cameras[1.0][pose].R.transpose(1, 0)[1, :], dtype=torch.float32,
                          device="cuda").unsqueeze(0)
    F_pose = torch.tensor(scene.test_cameras[1.0][pose].R.transpose(1, 0)[2, :], dtype=torch.float32,
                          device="cuda").unsqueeze(0)
    bitmap_pose = scene.test_cameras[1.0][pose].foreground_image.to(dtype=torch.float32).reshape(3,
                                                                                                 height * width).transpose(
        0, 1)
    alpha_pose = scene.test_cameras[1.0][pose].gt_alpha_mask.to(dtype=torch.float32).reshape(height * width, 1)
    # ### ### ### ### ###

    fov_x_list.append(fov_x_pose)
    fov_y_list.append(fov_y_pose)
    O_list.append(O_pose)
    R_list.append(R_pose)
    D_list.append(D_pose)
    F_list.append(F_pose)
    bitmap_list.append(bitmap_pose)
    alpha_list.append(alpha_pose)
fov_x_test = torch.cat(fov_x_list, 0)
fov_y_test = torch.cat(fov_y_list, 0)
O_test = torch.cat(O_list, 0)
R_test = torch.cat(R_list, 0)
D_test = torch.cat(D_list, 0)
F_test = torch.cat(F_list, 0)
bitmap_test = torch.cat(bitmap_list, 0)
alpha_test = torch.cat(alpha_list, 0)
fixed_background = torch.tensor(
    [ep.bg_color_R, ep.bg_color_G, ep.bg_color_B],
    dtype=torch.float32,
    device="cuda",
).reshape(1, 3)
random_background_enabled = pp.random_background and bool(torch.any(alpha < 1.0).item())
print(
    "Random-background transparency carving: "
    + ("enabled" if random_background_enabled else "disabled"),
    flush=True,
)
# ### ### ### ### ###

# O_cam = O[27,:].clone()
# R_cam = R[27,:].clone()
# D_cam = D[27,:].clone()
# F_cam = F[27,:].clone()
O_cam = O_test[0, :].clone()
R_cam = R_test[0, :].clone()
D_cam = D_test[0, :].clone()
F_cam = F_test[0, :].clone()
# ### ### ### ### ###

renderer = py_OptiX_FLARE_renderer.CPyOptiXFLARERenderer(
    pp.number_of_sides,
    11.3449,
    max_batch_size
)
# ### ### ### ### ###

w1_uv = gaussians.w1_uv
w1_v = gaussians.w1_v
w1_conditioning = gaussians.w1_conditioning
b1 = gaussians.b1
w2 = gaussians.w2
b2 = gaussians.b2
w3 = gaussians.w3
b3 = gaussians.b3
features = gaussians.features
conditioning_variable = gaussians.conditioning_variable
RGB = gaussians.RGB
A = gaussians.A
k = gaussians.k
m = gaussians.m
s = gaussians.s
q = gaussians.q
# ### ### ### ### ###

iteration = gaussians.iteration + 1;  # !!! !!! !!!
training_time = gaussians.training_time;  # !!! !!! !!!
optimizer = gaussians.optimizer
checkpoint_metadata = {}
metadata_path = _checkpoint_metadata_path()
if os.path.isfile(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        checkpoint_metadata = json.load(metadata_file)
# Migrate the historical best from the numbered layout when resuming an older run.
if "best" not in checkpoint_metadata:
    for metric_name in ("PSNR_test_FlaRe", "PSNR_test_base"):
        stats_path = os.path.join(
            "output", str(next_available_dir_id), "stats", metric_name + ".txt"
        )
        if not os.path.isfile(stats_path):
            continue
        measurements = []
        with open(stats_path, "r", encoding="utf-8") as stats_file:
            for line in stats_file:
                try:
                    logged_iteration, logged_value = line.split(":", 1)
                    measurements.append((int(logged_iteration), float(logged_value)))
                except ValueError:
                    continue
        if not measurements:
            continue
        historical_iteration, historical_value = max(
            measurements, key=lambda item: (item[1], item[0])
        )
        legacy_path = os.path.join(
            _checkpoint_root(), str(historical_iteration),
            "iter_" + str(historical_iteration) + ".checkpoint"
        )
        best_path = os.path.join(_checkpoint_root(), "best.checkpoint")
        if os.path.isfile(legacy_path):
            try:
                os.link(legacy_path, best_path)
            except OSError:
                shutil.copy2(legacy_path, best_path)
            checkpoint_metadata["best"] = {
                "iteration": historical_iteration,
                "metric": metric_name,
                "value": historical_value,
            }
            _write_checkpoint_metadata()
        break
best_entry = checkpoint_metadata.get("best", {})
best_checkpoint_metric = best_entry.get("metric")
best_checkpoint_value = float(best_entry.get("value", -np.inf))
# ### ### ### ### ###

del gaussians
del scene
maybe_empty_cuda_cache()
# ### ### ### ### ###

extent = torch.sqrt(((torch.max(m, 0, keepdim=True)[0] - torch.min(m, 0, keepdim=True)[0]) ** 2).sum(1)).item()
# ### ### ### ### ###

renderer.SetGeometry(m, torch.exp(s), q, torch.sigmoid(A), 1.0 + torch.nn.functional.softplus(k))
# ### ### ### ### ###

# Normal consistency needs complete rasters. The standard configuration keeps
# the historical random-ray batches and does not change baseline training.
normal_training_enabled = pp.reg_normal_lambda > 0.0
if normal_training_enabled:
    indices = torch.randperm(number_of_poses, dtype=torch.int64, device="cpu")
else:
    indices = torch.randperm(number_of_poses * height * width, dtype=torch.int64, device="cpu")
batch_start_index = 0
batch_size = width * height
# ### ### ### ### ###

PSNR_max_base = -np.inf
PSNR_max_FlaRe = -np.inf
# ### ### ### ### ###

def train():
    global extent
    global optimizer
    global w1, b1, w2, b2, w3, b3, features, conditioning_variable, A, k, m, s, q
    global RGB
    global indices
    global iteration, training_time
    global batch_start_index
    global tk_image
    global PSNR_max_base, PSNR_max_FlaRe
    t1 = time.perf_counter()
    depth_lambda = (
        pp.reg_depth_lambda if iteration > pp.reg_depth_start_iter else 0.0
    )
    normal_ramp = np.clip(
        (iteration - pp.reg_normal_start_iter)
        / max(float(pp.reg_normal_ramp_iters), 1.0),
        0.0,
        1.0,
    )
    normal_lambda = pp.reg_normal_lambda * normal_ramp
    if random_background_enabled:
        background_values = np.random.random(3).astype(np.float32)
        background = torch.as_tensor(background_values, device="cuda").reshape(1, 3)
        background_R, background_G, background_B = background_values.tolist()
    else:
        background = fixed_background
        background_R = ep.bg_color_R
        background_G = ep.bg_color_G
        background_B = ep.bg_color_B
    # ### ### ### ### ###

    if normal_training_enabled:
        training_pose = int(indices[batch_start_index].item())
        end_of_batch = batch_start_index + 1 >= number_of_poses
        indices_chunk = torch.arange(batch_size, dtype=torch.int64, device="cuda")
        indices_chunk += training_pose * batch_size
        batch_fov_x = fov_x[training_pose].item()
        batch_fov_y = fov_y[training_pose].item()
    else:
        batch_end_index = min(
            batch_start_index + batch_size, number_of_poses * height * width
        )
        end_of_batch = batch_end_index >= number_of_poses * height * width
        indices_chunk = indices[batch_start_index:batch_end_index].to(device="cuda")
        batch_fov_x = fov_x[0].item()
        batch_fov_y = fov_y[0].item()
    (O_chunk, v_chunk) = GenerateRays(
        indices_chunk,
        O, R, D, F,
        width, height, number_of_poses,
        batch_fov_x, batch_fov_y
    )
    v_chunk = v_chunk / torch.sqrt(torch.sum(v_chunk * v_chunk, 1, keepdim=True))
    # ### ### ### ### ###

    RGBA = torch.cat([RGB, A], 1)
    # ### ### ### ### ###

    dL_dRGB_final = torch.zeros_like(RGB)
    dL_dA_final = torch.zeros_like(A)
    dL_dk_final = torch.zeros_like(k)
    dL_dw3_final = torch.zeros(8, 64, dtype=torch.float32, device="cuda")
    dL_db3_final = torch.zeros(8, dtype=torch.float32, device="cuda")
    dL_dw2_final = torch.zeros(64, 64, dtype=torch.float32, device="cuda")
    dL_db2_final = torch.zeros(64, dtype=torch.float32, device="cuda")
    dL_dw1_final = torch.zeros(64, 128, dtype=torch.float32, device="cuda")
    dL_db1_final = torch.zeros(64, dtype=torch.float32, device="cuda")
    dL_d_conditioning_final = torch.zeros_like(conditioning_variable)
    dL_d_features_final = torch.zeros_like(features, dtype=torch.float32, device="cuda")
    dL_dm_final = torch.zeros_like(m)
    dL_ds_final = torch.zeros_like(s)
    dL_dq_final = torch.zeros_like(q)
    # ### ### ### ### ###

    # Paper: the base warmup path uses only view-independent primitive RGB
    # and opacity before the local neural radiance fields become expressive.
    # Base model
    if (warmup_lambda < 1.0):
        img_RGB_unclamped = torch.zeros((batch_size, 3), dtype=torch.float32, device="cuda");  # !!! !!! !!!

        # Paper: geometry-aware depth regularization accumulates ordered-hit
        # moments; this supports the geometry-oriented training variant.
        # (L_d, o_total, to_total, dL_do_total)
        depth_reg_depth_accums = torch.zeros((batch_size, 4), dtype=torch.float32, device="cuda");  # !!! !!! !!!
        depth_and_index = torch.zeros((batch_size, 2), dtype=torch.float32, device="cuda")
        surface_normal = torch.zeros((batch_size, 3), dtype=torch.float32, device="cuda")
        normal_reg_accums = torch.zeros((batch_size, 4), dtype=torch.float32, device="cuda")
        renderer.Forward_training_base(
            O_chunk, v_chunk, img_RGB_unclamped,

            background_R, background_G, background_B,

            m, s, q, RGBA, k,

            pp.ray_termination_T_threshold_training,

            depth_reg_depth_accums, reg_depth_a, reg_depth_b,
            depth_and_index, surface_normal, normal_reg_accums
        )
        if normal_lambda > 0.0:
            expected_t = depth_reg_depth_accums[:, 3] / depth_reg_depth_accums[:, 1].clamp_min(1.0e-8)
            _, surface_normal, _ = ray_parameters_to_depth_and_normal(
                expected_t,
                depth_reg_depth_accums[:, 1] > 0.01,
                O_chunk,
                v_chunk,
                F[training_pose],
                height,
                width,
                pp.reg_normal_depth_edge_threshold,
            )
            surface_normal = surface_normal.reshape(batch_size, 3).contiguous()
        # ### ### ### ### ###

        img_RGB = torch.clamp(img_RGB_unclamped, min=0.0, max=1.0)
        img_RGB = img_RGB.detach().requires_grad_(True)
        foreground_chunk = torch.gather(bitmap, 0, indices_chunk.unsqueeze(1).expand(-1, 3))
        alpha_chunk = torch.gather(alpha, 0, indices_chunk.unsqueeze(1))
        ground_truth_chunk = foreground_chunk + background * (1.0 - alpha_chunk)
        loss = torch.mean((img_RGB - ground_truth_chunk) ** 2)
        loss.backward()
        with torch.no_grad():
            dL_dI = img_RGB.grad
            clamped = (img_RGB_unclamped != img_RGB)
            # ### ### ### ### ###

            dL_dRGB = torch.zeros_like(RGB)
            dL_dA = torch.zeros_like(A)
            dL_dk = torch.zeros_like(k)
            dL_dm = torch.zeros_like(m)
            dL_ds = torch.zeros_like(s)
            dL_dq = torch.zeros_like(q)
            # ### ### ### ### ###

            # (0, o_cum, to_cum, dL_do_cum)
            depth_normal_reg_prefix_sums = torch.zeros((batch_size, 4), dtype=torch.float32,
                                                       device="cuda");  # !!! !!! !!!

            renderer.Backward_base(
                O_chunk, v_chunk,

                background_R, background_G, background_B,

                m, s, q, RGBA, k,

                img_RGB, dL_dI * (~clamped), dL_dRGB, dL_dA, dL_dk, dL_dm, dL_ds, dL_dq,

                pp.ray_termination_T_threshold_training,

                depth_reg_depth_accums, depth_normal_reg_prefix_sums,
                         depth_lambda / batch_size, reg_depth_a, reg_depth_b,
                depth_and_index, surface_normal, normal_reg_accums,
                         normal_lambda / batch_size
            )
            # ### ### ### ### ###

            dL_dRGB_final += (1.0 - warmup_lambda) * dL_dRGB
            dL_dA_final += (1.0 - warmup_lambda) * dL_dA
            dL_dk_final += (1.0 - warmup_lambda) * dL_dk
            dL_dm_final += (1.0 - warmup_lambda) * dL_dm
            dL_ds_final += (1.0 - warmup_lambda) * dL_ds
            dL_dq_final += (1.0 - warmup_lambda) * dL_dq
        # ### ### ### ### ###

        PSNR_base = (-10.0 * (torch.log(loss) / torch.log(torch.tensor([10.0], device="cuda")))).item()
        if (PSNR_base > PSNR_max_base):
            PSNR_max_base = PSNR_base
    # ### ### ### ### ###

    # Paper: "Loss function". During warmup, gradients from the constant
    # base renderer and the full local-radiance model are blended by lambda.
    # FlaRe model
    if (warmup_lambda > 0.0):
        w1_fp16 = torch.cat([w1_uv.detach(), w1_v.detach(), w1_conditioning.detach()], 1).to(torch.float16)
        w2_fp16 = w2.detach().to(torch.float16)
        w3_fp16 = w3.detach().to(torch.float16)
        conditioning_variable_fp16 = conditioning_variable.detach().to(torch.float16)
        # ### ### ### ### ###

        img_RGB_unclamped = torch.zeros((batch_size, 3), dtype=torch.float32, device="cuda");  # !!! !!! !!!

        # (L_d, o_total, to_total, dL_do_total)
        depth_reg_depth_accums = torch.zeros((batch_size, 4), dtype=torch.float32, device="cuda");  # !!! !!! !!!
        depth_and_index = torch.zeros((batch_size, 2), dtype=torch.float32, device="cuda")
        surface_normal = torch.zeros((batch_size, 3), dtype=torch.float32, device="cuda")
        normal_reg_accums = torch.zeros((batch_size, 4), dtype=torch.float32, device="cuda")
        # Paper: full Phi((u, v), d, z_i) rendering path with LUT encoding,
        # the shared auto-decoder, and generalized-Gaussian compositing.
        renderer.Forward_training(
            conditioning_variable_fp16,
            features,

            w1_fp16,
            b1,
            w2_fp16,
            b2,
            w3_fp16,
            b3,

            O_chunk, v_chunk, img_RGB_unclamped,

            background_R, background_G, background_B,

            m, s, q, RGBA, k,

            pp.ray_termination_T_threshold_training,

            depth_reg_depth_accums, reg_depth_a, reg_depth_b,
            depth_and_index, surface_normal, normal_reg_accums
        )
        if normal_lambda > 0.0:
            expected_t = depth_reg_depth_accums[:, 3] / depth_reg_depth_accums[:, 1].clamp_min(1.0e-8)
            _, surface_normal, _ = ray_parameters_to_depth_and_normal(
                expected_t,
                depth_reg_depth_accums[:, 1] > 0.01,
                O_chunk,
                v_chunk,
                F[training_pose],
                height,
                width,
                pp.reg_normal_depth_edge_threshold,
            )
            surface_normal = surface_normal.reshape(batch_size, 3).contiguous()
        # ### ### ### ### ###

        img_RGB = torch.clamp(img_RGB_unclamped, min=0.0, max=1.0)
        img_RGB = img_RGB.detach().requires_grad_(True)
        foreground_chunk = torch.gather(bitmap, 0, indices_chunk.unsqueeze(1).expand(-1, 3))
        alpha_chunk = torch.gather(alpha, 0, indices_chunk.unsqueeze(1))
        ground_truth_chunk = foreground_chunk + background * (1.0 - alpha_chunk)
        loss = torch.mean((img_RGB - ground_truth_chunk) ** 2)
        loss.backward()
        with torch.no_grad():
            dL_dI = img_RGB.grad
            clamped = (img_RGB_unclamped != img_RGB)
            # ### ### ### ### ###

            dL_dRGB = torch.zeros_like(RGB)
            dL_dA = torch.zeros_like(A)
            dL_dk = torch.zeros_like(k)
            dL_dw3 = torch.zeros((4 * SM_count, 8 * 64), dtype=torch.float32, device="cuda")
            dL_db3 = torch.zeros((4 * SM_count, 8 * 8), dtype=torch.float32, device="cuda")
            dL_dw2 = torch.zeros((4 * SM_count, 64 * 64), dtype=torch.float32, device="cuda")
            dL_db2 = torch.zeros((4 * SM_count, 64 * 8), dtype=torch.float32, device="cuda")
            dL_dw1 = torch.zeros((4 * SM_count, 64 * 128), dtype=torch.float32, device="cuda")
            dL_db1 = torch.zeros((4 * SM_count, 64 * 8), dtype=torch.float32, device="cuda")
            dL_d_conditioning = torch.zeros_like(conditioning_variable)
            dL_d_features = torch.zeros_like(features, dtype=torch.float32, device="cuda")
            dL_dm = torch.zeros_like(m)
            dL_ds = torch.zeros_like(s)
            dL_dq = torch.zeros_like(q)
            # ### ### ### ### ###

            # (0, o_cum, to_cum, dL_do_cum)
            depth_normal_reg_prefix_sums = torch.zeros((batch_size, 4), dtype=torch.float32,
                                                       device="cuda");  # !!! !!! !!!

            renderer.Backward(
                conditioning_variable_fp16,
                features,

                w1_fp16,
                b1,
                w2_fp16,
                b2,
                w3_fp16,
                b3,

                O_chunk, v_chunk,

                background_R, background_G, background_B,

                m, s, q, RGBA, k,

                img_RGB, dL_dI * (~clamped), dL_dRGB, dL_dA, dL_dk, dL_dw3, dL_db3, dL_dw2, dL_db2, dL_dw1, dL_db1,
                dL_d_conditioning, dL_d_features, dL_dm, dL_ds, dL_dq,

                pp.ray_termination_T_threshold_training,

                depth_reg_depth_accums, depth_normal_reg_prefix_sums,
                         depth_lambda / batch_size, reg_depth_a, reg_depth_b,
                depth_and_index, surface_normal, normal_reg_accums,
                         normal_lambda / batch_size
            )
            # ### ### ### ### ###

            dL_dw3 = dL_dw3.sum(0)
            dL_dw3 = dL_dw3.reshape((8, 2, 32))
            dL_dw3 = dL_dw3.transpose(1, 2)
            dL_dw3 = dL_dw3.reshape((8, 8, -1))
            dL_dw3 = dL_dw3.transpose(0, 1)
            dL_dw3 = dL_dw3.flatten(1, 2)
            dL_db3 = dL_db3.sum(0)
            dL_db3 = dL_db3.reshape((2, 32))
            dL_db3 = dL_db3.transpose(0, 1)
            dL_db3 = dL_db3.reshape((8, 8))
            dL_db3 = dL_db3.sum(1)
            dL_dw2 = dL_dw2.sum(0)
            dL_dw2 = dL_dw2.reshape((64, 2, 32))
            dL_dw2 = dL_dw2.transpose(1, 2)
            dL_dw2 = dL_dw2.reshape((8, 8, 8, -1))
            dL_dw2 = dL_dw2.transpose(1, 2)
            dL_dw2 = dL_dw2.flatten(0, 1)
            dL_dw2 = dL_dw2.flatten(1, 2)
            dL_db2 = dL_db2.sum(0)
            dL_db2 = dL_db2.reshape((8, 2, 32))
            dL_db2 = dL_db2.transpose(1, 2)
            dL_db2 = dL_db2.reshape((8, 8, -1))
            dL_db2 = dL_db2.sum(2)
            dL_db2 = dL_db2.flatten(0, 1)
            dL_dw1 = dL_dw1.sum(0)
            dL_dw1 = dL_dw1.reshape((128, 2, 32))
            dL_dw1 = dL_dw1.transpose(1, 2)
            dL_dw1 = dL_dw1.reshape((8, 16, 8, -1))
            dL_dw1 = dL_dw1.transpose(1, 2)
            dL_dw1 = dL_dw1.flatten(0, 1)
            dL_dw1 = dL_dw1.flatten(1, 2)
            dL_db1 = dL_db1.sum(0)
            dL_db1 = dL_db1.reshape((8, 2, 32))
            dL_db1 = dL_db1.transpose(1, 2)
            dL_db1 = dL_db1.reshape((8, 8, -1))
            dL_db1 = dL_db1.sum(2)
            dL_db1 = dL_db1.flatten(0, 1)
            # ### ### ### ### ###

            dL_dRGB_final += warmup_lambda * dL_dRGB
            dL_dA_final += warmup_lambda * dL_dA
            dL_dk_final += warmup_lambda * dL_dk
            dL_dw3_final += warmup_lambda * dL_dw3
            dL_db3_final += warmup_lambda * dL_db3
            dL_dw2_final += warmup_lambda * dL_dw2
            dL_db2_final += warmup_lambda * dL_db2
            dL_dw1_final += warmup_lambda * dL_dw1
            dL_db1_final += warmup_lambda * dL_db1
            dL_d_conditioning_final += warmup_lambda * dL_d_conditioning
            dL_d_features_final += warmup_lambda * dL_d_features
            dL_dm_final += warmup_lambda * dL_dm
            dL_ds_final += warmup_lambda * dL_ds
            dL_dq_final += warmup_lambda * dL_dq
        # ### ### ### ### ###

        PSNR_FlaRe = (-10.0 * (torch.log(loss) / torch.log(torch.tensor([10.0], device="cuda")))).item()
        if (PSNR_FlaRe > PSNR_max_FlaRe):
            PSNR_max_FlaRe = PSNR_FlaRe
    # ### ### ### ### ###

    # Paper: densification and pruning update the explicit primitive set
    # during optimization; the optional cap drives the primitive-budget study.
    densification_iter = (
            (iteration >= pp.densification_start_iter) and
            (iteration <= pp.densification_end_iter) and
            (iteration % pp.densification_frequency == 0)
    )
    if (densification_iter):
        m_before = m.detach().clone();  # !!! !!! !!!

        # First optimizer step
        if (iteration == 1):
            m_before_exp_avg = torch.zeros_like(m_before)
            m_before_exp_avg_sq = torch.zeros_like(m_before)
        else:
            state_dict_old = optimizer.state_dict()
            state_old = state_dict_old['state']
            m_before_exp_avg = state_old[13]['exp_avg'].clone()
            m_before_exp_avg_sq = state_old[13]['exp_avg_sq'].clone()
    # ### ### ### ### ###

    s_squared = torch.exp(s.detach()) ** 2
    # ### ### ### ### ###

    w1_uv.grad = dL_dw1_final[:, 0:8]
    w1_v.grad = dL_dw1_final[:, 8:32]
    w1_conditioning.grad = dL_dw1_final[:, 32:128]
    b1.grad = dL_db1_final
    w2.grad = dL_dw2_final
    b2.grad = dL_db2_final
    w3.grad = nn.functional.pad(dL_dw3_final, (0, 0, 0, 8), "constant", 0.0)
    b3.grad = nn.functional.pad(dL_db3_final, (0, 8), "constant", 0.0)
    features.grad = dL_d_features_final
    conditioning_variable.grad = dL_d_conditioning_final
    RGB.grad = dL_dRGB_final
    A.grad = dL_dA_final
    k.grad = dL_dk_final
    m.grad = dL_dm_final
    # Paper: L_s discourages excessively large planar primitives.
    s.grad = dL_ds_final + (
                (pp.reg_scale_lambda / m.shape[0]) * (s_squared / torch.sqrt(s_squared.sum(1, keepdim=True))))
    q.grad = dL_dq_final
    # ### ### ### ### ###

    optimizer.step()
    # ### ### ### ### ###

    if end_of_batch:
        permutation_size = (
            number_of_poses
            if normal_training_enabled
            else number_of_poses * height * width
        )
        indices = torch.randperm(permutation_size, dtype=torch.int64, device="cpu")
        batch_start_index = 0
    elif normal_training_enabled:
        batch_start_index += 1
    else:
        batch_start_index = batch_end_index
    # ###############################################################################################

    if (lp.lr_RGB_exp_decay_coef <= 0.0):
        lr_RGB_current = float(np.maximum(lp.lr_RGB * np.exp(lp.lr_RGB_exp_decay_coef * iteration), lp.lr_RGB_final))
    else:
        lr_RGB_current = float(np.minimum(lp.lr_RGB * np.exp(lp.lr_RGB_exp_decay_coef * iteration), lp.lr_RGB_final))
    if (lp.lr_A_exp_decay_coef <= 0.0):
        lr_A_current = float(np.maximum(lp.lr_A * np.exp(lp.lr_A_exp_decay_coef * iteration), lp.lr_A_final))
    else:
        lr_A_current = float(np.minimum(lp.lr_A * np.exp(lp.lr_A_exp_decay_coef * iteration), lp.lr_A_final))
    if (lp.lr_k_exp_decay_coef <= 0.0):
        lr_k_current = float(np.maximum(lp.lr_k * np.exp(lp.lr_k_exp_decay_coef * iteration), lp.lr_k_final))
    else:
        lr_k_current = float(np.minimum(lp.lr_k * np.exp(lp.lr_k_exp_decay_coef * iteration), lp.lr_k_final))
    if (lp.lr_w1_uv_exp_decay_coef <= 0.0):
        lr_w1_uv_current = float(
            np.maximum(lp.lr_w1_uv * np.exp(lp.lr_w1_uv_exp_decay_coef * iteration), lp.lr_w1_uv_final))
    else:
        lr_w1_uv_current = float(
            np.minimum(lp.lr_w1_uv * np.exp(lp.lr_w1_uv_exp_decay_coef * iteration), lp.lr_w1_uv_final))
    if (lp.lr_w1_v_exp_decay_coef <= 0.0):
        lr_w1_v_current = float(
            np.maximum(lp.lr_w1_v * np.exp(lp.lr_w1_v_exp_decay_coef * iteration), lp.lr_w1_v_final))
    else:
        lr_w1_v_current = float(
            np.minimum(lp.lr_w1_v * np.exp(lp.lr_w1_v_exp_decay_coef * iteration), lp.lr_w1_v_final))
    if (lp.lr_w1_conditioning_exp_decay_coef <= 0.0):
        lr_w1_conditioning_current = float(
            np.maximum(lp.lr_w1_conditioning * np.exp(lp.lr_w1_conditioning_exp_decay_coef * iteration),
                       lp.lr_w1_conditioning_final))
    else:
        lr_w1_conditioning_current = float(
            np.minimum(lp.lr_w1_conditioning * np.exp(lp.lr_w1_conditioning_exp_decay_coef * iteration),
                       lp.lr_w1_conditioning_final))
    if (lp.lr_b1_exp_decay_coef <= 0.0):
        lr_b1_current = float(np.maximum(lp.lr_b1 * np.exp(lp.lr_b1_exp_decay_coef * iteration), lp.lr_b1_final))
    else:
        lr_b1_current = float(np.minimum(lp.lr_b1 * np.exp(lp.lr_b1_exp_decay_coef * iteration), lp.lr_b1_final))
    if (lp.lr_w2_exp_decay_coef <= 0.0):
        lr_w2_current = float(np.maximum(lp.lr_w2 * np.exp(lp.lr_w2_exp_decay_coef * iteration), lp.lr_w2_final))
    else:
        lr_w2_current = float(np.minimum(lp.lr_w2 * np.exp(lp.lr_w2_exp_decay_coef * iteration), lp.lr_w2_final))
    if (lp.lr_b2_exp_decay_coef <= 0.0):
        lr_b2_current = float(np.maximum(lp.lr_b2 * np.exp(lp.lr_b2_exp_decay_coef * iteration), lp.lr_b2_final))
    else:
        lr_b2_current = float(np.minimum(lp.lr_b2 * np.exp(lp.lr_b2_exp_decay_coef * iteration), lp.lr_b2_final))
    if (lp.lr_w3_exp_decay_coef <= 0.0):
        lr_w3_current = float(np.maximum(lp.lr_w3 * np.exp(lp.lr_w3_exp_decay_coef * iteration), lp.lr_w3_final))
    else:
        lr_w3_current = float(np.minimum(lp.lr_w3 * np.exp(lp.lr_w3_exp_decay_coef * iteration), lp.lr_w3_final))
    if (lp.lr_b3_exp_decay_coef <= 0.0):
        lr_b3_current = float(np.maximum(lp.lr_b3 * np.exp(lp.lr_b3_exp_decay_coef * iteration), lp.lr_b3_final))
    else:
        lr_b3_current = float(np.minimum(lp.lr_b3 * np.exp(lp.lr_b3_exp_decay_coef * iteration), lp.lr_b3_final))
    if (lp.lr_conditioning_exp_decay_coef <= 0.0):
        lr_conditioning_current = float(
            np.maximum(lp.lr_conditioning * np.exp(lp.lr_conditioning_exp_decay_coef * iteration),
                       lp.lr_conditioning_final))
    else:
        lr_conditioning_current = float(
            np.minimum(lp.lr_conditioning * np.exp(lp.lr_conditioning_exp_decay_coef * iteration),
                       lp.lr_conditioning_final))
    if (lp.lr_features_exp_decay_coef <= 0.0):
        lr_features_current = float(
            np.maximum(lp.lr_features * np.exp(lp.lr_features_exp_decay_coef * iteration), lp.lr_features_final))
    else:
        lr_features_current = float(
            np.minimum(lp.lr_features * np.exp(lp.lr_features_exp_decay_coef * iteration), lp.lr_features_final))
    if (lp.lr_m_exp_decay_coef <= 0.0):
        lr_m_current = float(np.maximum(lp.lr_m * np.exp(lp.lr_m_exp_decay_coef * iteration), lp.lr_m_final))
    else:
        lr_m_current = float(np.minimum(lp.lr_m * np.exp(lp.lr_m_exp_decay_coef * iteration), lp.lr_m_final))
    if (lp.lr_s_exp_decay_coef <= 0.0):
        lr_s_current = float(np.maximum(lp.lr_s * np.exp(lp.lr_s_exp_decay_coef * iteration), lp.lr_s_final))
    else:
        lr_s_current = float(np.minimum(lp.lr_s * np.exp(lp.lr_s_exp_decay_coef * iteration), lp.lr_s_final))
    if (lp.lr_q_exp_decay_coef <= 0.0):
        lr_q_current = float(np.maximum(lp.lr_q * np.exp(lp.lr_q_exp_decay_coef * iteration), lp.lr_q_final))
    else:
        lr_q_current = float(np.minimum(lp.lr_q * np.exp(lp.lr_q_exp_decay_coef * iteration), lp.lr_q_final))
    # ### ### ### ### ###

    if (densification_iter):
        with torch.no_grad():
            m = m.detach()
            s = s.detach()
            q = q.detach()
            RGB = RGB.detach()
            A = A.detach()
            k = k.detach()
            conditioning_variable = conditioning_variable.detach()
            # ### ### ### ### ###

            state_dict_old = optimizer.state_dict()
            state_old = state_dict_old['state']
            # Paper: remove primitives whose effective generalized-Gaussian
            # support is negligible under the opacity and scale thresholds.
            # Opacity pruning
            k_final = 1.0 + torch.nn.functional.softplus(k)
            A_final = torch.sigmoid(A)
            scale = torch.clamp(k_final * (11.3449 + (2.0 * torch.log(A_final))), min=0.0)
            scale = (scale ** (1.0 / (2.0 * k_final))) / np.sqrt(11.3449)
            s_norm = torch.sqrt((torch.exp(s) ** 2).sum(1)) * scale.squeeze(1)
            mask_opacity = (
                    (A >= np.log(pp.opacity_threshold_for_Gauss_removal / (
                                1.0 - pp.opacity_threshold_for_Gauss_removal))).squeeze(1) &
                    (s_norm >= pp.min_s_norm_threshold_for_Gauss_removal)
            )
            m_before_prune = m_before[mask_opacity];  # !!! !!! !!!
            m_before_prune_exp_avg = m_before_exp_avg[mask_opacity];  # !!! !!! !!!
            m_before_prune_exp_avg_sq = m_before_exp_avg_sq[mask_opacity];  # !!! !!! !!!

            m_prune = m[mask_opacity];  # !!! !!! !!!
            m_prune_exp_avg = state_old[13]['exp_avg'][mask_opacity];  # !!! !!! !!!
            m_prune_exp_avg_sq = state_old[13]['exp_avg_sq'][mask_opacity];  # !!! !!! !!!

            s_prune = s[mask_opacity]
            s_prune_exp_avg = state_old[14]['exp_avg'][mask_opacity]
            s_prune_exp_avg_sq = state_old[14]['exp_avg_sq'][mask_opacity]
            q_prune = q[mask_opacity]
            q_prune_exp_avg = state_old[15]['exp_avg'][mask_opacity]
            q_prune_exp_avg_sq = state_old[15]['exp_avg_sq'][mask_opacity]
            RGB_prune = RGB[mask_opacity]
            RGB_prune_exp_avg = state_old[0]['exp_avg'][mask_opacity]
            RGB_prune_exp_avg_sq = state_old[0]['exp_avg_sq'][mask_opacity]
            A_prune = A[mask_opacity]
            A_prune_exp_avg = state_old[1]['exp_avg'][mask_opacity]
            A_prune_exp_avg_sq = state_old[1]['exp_avg_sq'][mask_opacity]
            k_prune = k[mask_opacity]
            k_prune_exp_avg = state_old[2]['exp_avg'][mask_opacity]
            k_prune_exp_avg_sq = state_old[2]['exp_avg_sq'][mask_opacity]
            conditioning_variable_prune = conditioning_variable[mask_opacity]
            conditioning_variable_prune_exp_avg = state_old[11]['exp_avg'][mask_opacity]
            conditioning_variable_prune_exp_avg_sq = state_old[11]['exp_avg_sq'][mask_opacity]
            # ### ### ### ### ###

            d_m = m_prune - m_before_prune
            d_m = torch.sqrt(torch.sum(d_m * d_m, 1, keepdim=True))
            # Paper: split primitives with sufficiently large center motion while
            # retaining their local appearance descriptors and optimizer state.
            mask_densify = (
                    (d_m >= pp.mu_grad_norm_threshold_for_densification) &
                    ((pp.max_Gaussians_per_model == -1) | (m_prune.shape[0] <= pp.max_Gaussians_per_model))
            ).squeeze(1)
            mask_orig = ~mask_densify
            # ### ### ### ### ###

            m_orig = m_prune[mask_orig];  # !!! !!! !!!
            m_orig_exp_avg = m_prune_exp_avg[mask_orig];  # !!! !!! !!!
            m_orig_exp_avg_sq = m_prune_exp_avg_sq[mask_orig];  # !!! !!! !!!

            m_densify_before = m_before_prune[mask_densify];  # !!! !!! !!!
            m_densify_before_exp_avg = m_before_prune_exp_avg[mask_densify];  # !!! !!! !!!
            m_densify_before_exp_avg_sq = m_before_prune_exp_avg_sq[mask_densify];  # !!! !!! !!!

            m_densify = m_prune[mask_densify];  # !!! !!! !!!
            m_densify_exp_avg = m_prune_exp_avg[mask_densify];  # !!! !!! !!!
            m_densify_exp_avg_sq = m_prune_exp_avg_sq[mask_densify];  # !!! !!! !!!

            m_densify_new = torch.cat([m_densify_before, m_densify], 0)
            m_densify_new_exp_avg = torch.cat([m_densify_before_exp_avg, m_densify_exp_avg], 0)
            m_densify_new_exp_avg_sq = torch.cat([m_densify_before_exp_avg_sq, m_densify_exp_avg_sq], 0)
            del m_densify_before, m_densify
            del m_densify_before_exp_avg, m_densify_exp_avg
            del m_densify_before_exp_avg_sq, m_densify_exp_avg_sq
            m_new = torch.cat([m_orig, m_densify_new], 0)
            m_new_exp_avg = torch.cat([m_orig_exp_avg, m_densify_new_exp_avg], 0)
            m_new_exp_avg_sq = torch.cat([m_orig_exp_avg_sq, m_densify_new_exp_avg_sq], 0)
            del m_orig, m_densify_new
            del m_orig_exp_avg, m_densify_new_exp_avg
            del m_orig_exp_avg_sq, m_densify_new_exp_avg_sq
            del m_before_prune
            del m_before_prune_exp_avg
            del m_before_prune_exp_avg_sq
            del m_prune
            del m_prune_exp_avg
            del m_prune_exp_avg_sq
            del m_before
            del m_before_exp_avg
            del m_before_exp_avg_sq
            del m
            # ### ### ### ### ###

            s_orig = s_prune[mask_orig]
            s_orig_exp_avg = s_prune_exp_avg[mask_orig]
            s_orig_exp_avg_sq = s_prune_exp_avg_sq[mask_orig]
            s_densify = s_prune[mask_densify]
            s_densify_exp_avg = s_prune_exp_avg[mask_densify]
            s_densify_exp_avg_sq = s_prune_exp_avg_sq[mask_densify]
            s_densify_new = torch.cat([s_densify, s_densify], 0)
            s_densify_new_exp_avg = torch.cat([s_densify_exp_avg, s_densify_exp_avg], 0)
            s_densify_new_exp_avg_sq = torch.cat([s_densify_exp_avg_sq, s_densify_exp_avg_sq], 0)
            del s_densify
            del s_densify_exp_avg
            del s_densify_exp_avg_sq
            s_new = torch.cat([s_orig, s_densify_new], 0)
            s_new_exp_avg = torch.cat([s_orig_exp_avg, s_densify_new_exp_avg], 0)
            s_new_exp_avg_sq = torch.cat([s_orig_exp_avg_sq, s_densify_new_exp_avg_sq], 0)
            del s_orig, s_densify_new
            del s_orig_exp_avg, s_densify_new_exp_avg
            del s_orig_exp_avg_sq, s_densify_new_exp_avg_sq
            del s_prune
            del s_prune_exp_avg
            del s_prune_exp_avg_sq
            del s
            # ### ### ### ### ###

            q_orig = q_prune[mask_orig]
            q_orig_exp_avg = q_prune_exp_avg[mask_orig]
            q_orig_exp_avg_sq = q_prune_exp_avg_sq[mask_orig]
            q_densify = q_prune[mask_densify]
            q_densify_exp_avg = q_prune_exp_avg[mask_densify]
            q_densify_exp_avg_sq = q_prune_exp_avg_sq[mask_densify]
            q_densify_new = torch.cat([q_densify, q_densify], 0)
            q_densify_new_exp_avg = torch.cat([q_densify_exp_avg, q_densify_exp_avg], 0)
            q_densify_new_exp_avg_sq = torch.cat([q_densify_exp_avg_sq, q_densify_exp_avg_sq], 0)
            del q_densify
            del q_densify_exp_avg
            del q_densify_exp_avg_sq
            q_new = torch.cat([q_orig, q_densify_new], 0)
            q_new_exp_avg = torch.cat([q_orig_exp_avg, q_densify_new_exp_avg], 0)
            q_new_exp_avg_sq = torch.cat([q_orig_exp_avg_sq, q_densify_new_exp_avg_sq], 0)
            del q_orig, q_densify_new
            del q_orig_exp_avg, q_densify_new_exp_avg
            del q_orig_exp_avg_sq, q_densify_new_exp_avg_sq
            del q_prune
            del q_prune_exp_avg
            del q_prune_exp_avg_sq
            del q
            # ### ### ### ### ###

            RGB_orig = RGB_prune[mask_orig]
            RGB_orig_exp_avg = RGB_prune_exp_avg[mask_orig]
            RGB_orig_exp_avg_sq = RGB_prune_exp_avg_sq[mask_orig]
            RGB_densify = RGB_prune[mask_densify]
            RGB_densify_exp_avg = RGB_prune_exp_avg[mask_densify]
            RGB_densify_exp_avg_sq = RGB_prune_exp_avg_sq[mask_densify]
            RGB_densify_new = torch.cat([RGB_densify, RGB_densify], 0)
            RGB_densify_new_exp_avg = torch.cat([RGB_densify_exp_avg, RGB_densify_exp_avg], 0)
            RGB_densify_new_exp_avg_sq = torch.cat([RGB_densify_exp_avg_sq, RGB_densify_exp_avg_sq], 0)
            del RGB_densify
            del RGB_densify_exp_avg
            del RGB_densify_exp_avg_sq
            RGB_new = torch.cat([RGB_orig, RGB_densify_new], 0)
            RGB_new_exp_avg = torch.cat([RGB_orig_exp_avg, RGB_densify_new_exp_avg], 0)
            RGB_new_exp_avg_sq = torch.cat([RGB_orig_exp_avg_sq, RGB_densify_new_exp_avg_sq], 0)
            del RGB_orig, RGB_densify_new
            del RGB_orig_exp_avg, RGB_densify_new_exp_avg
            del RGB_orig_exp_avg_sq, RGB_densify_new_exp_avg_sq
            del RGB_prune
            del RGB_prune_exp_avg
            del RGB_prune_exp_avg_sq
            del RGB
            # ### ### ### ### ###

            A_orig = A_prune[mask_orig]
            A_orig_exp_avg = A_prune_exp_avg[mask_orig]
            A_orig_exp_avg_sq = A_prune_exp_avg_sq[mask_orig]
            A_densify = A_prune[mask_densify]
            A_densify_exp_avg = A_prune_exp_avg[mask_densify]
            A_densify_exp_avg_sq = A_prune_exp_avg_sq[mask_densify]
            A_densify_new = torch.cat([A_densify, A_densify], 0)
            A_densify_new_exp_avg = torch.cat([A_densify_exp_avg, A_densify_exp_avg], 0)
            A_densify_new_exp_avg_sq = torch.cat([A_densify_exp_avg_sq, A_densify_exp_avg_sq], 0)
            del A_densify
            del A_densify_exp_avg
            del A_densify_exp_avg_sq
            A_new = torch.cat([A_orig, A_densify_new], 0)
            A_new_exp_avg = torch.cat([A_orig_exp_avg, A_densify_new_exp_avg], 0)
            A_new_exp_avg_sq = torch.cat([A_orig_exp_avg_sq, A_densify_new_exp_avg_sq], 0)
            del A_orig, A_densify_new
            del A_orig_exp_avg, A_densify_new_exp_avg
            del A_orig_exp_avg_sq, A_densify_new_exp_avg_sq
            del A_prune
            del A_prune_exp_avg
            del A_prune_exp_avg_sq
            del A
            # ### ### ### ### ###

            k_orig = k_prune[mask_orig]
            k_orig_exp_avg = k_prune_exp_avg[mask_orig]
            k_orig_exp_avg_sq = k_prune_exp_avg_sq[mask_orig]
            k_densify = k_prune[mask_densify]
            k_densify_exp_avg = k_prune_exp_avg[mask_densify]
            k_densify_exp_avg_sq = k_prune_exp_avg_sq[mask_densify]
            k_densify_new = torch.cat([k_densify, k_densify], 0)
            k_densify_new_exp_avg = torch.cat([k_densify_exp_avg, k_densify_exp_avg], 0)
            k_densify_new_exp_avg_sq = torch.cat([k_densify_exp_avg_sq, k_densify_exp_avg_sq], 0)
            del k_densify
            del k_densify_exp_avg
            del k_densify_exp_avg_sq
            k_new = torch.cat([k_orig, k_densify_new], 0)
            k_new_exp_avg = torch.cat([k_orig_exp_avg, k_densify_new_exp_avg], 0)
            k_new_exp_avg_sq = torch.cat([k_orig_exp_avg_sq, k_densify_new_exp_avg_sq], 0)
            del k_orig, k_densify_new
            del k_orig_exp_avg, k_densify_new_exp_avg
            del k_orig_exp_avg_sq, k_densify_new_exp_avg_sq
            del k_prune
            del k_prune_exp_avg
            del k_prune_exp_avg_sq
            del k
            # ### ### ### ### ###

            conditioning_variable_orig = conditioning_variable_prune[mask_orig]
            conditioning_variable_orig_exp_avg = conditioning_variable_prune_exp_avg[mask_orig]
            conditioning_variable_orig_exp_avg_sq = conditioning_variable_prune_exp_avg_sq[mask_orig]
            conditioning_variable_densify = conditioning_variable_prune[mask_densify]
            conditioning_variable_densify_exp_avg = conditioning_variable_prune_exp_avg[mask_densify]
            conditioning_variable_densify_exp_avg_sq = conditioning_variable_prune_exp_avg_sq[mask_densify]
            conditioning_variable_densify_new = torch.cat(
                [conditioning_variable_densify, conditioning_variable_densify], 0)
            conditioning_variable_densify_new_exp_avg = torch.cat(
                [conditioning_variable_densify_exp_avg, conditioning_variable_densify_exp_avg], 0)
            conditioning_variable_densify_new_exp_avg_sq = torch.cat(
                [conditioning_variable_densify_exp_avg_sq, conditioning_variable_densify_exp_avg_sq], 0)
            del conditioning_variable_densify
            del conditioning_variable_densify_exp_avg
            del conditioning_variable_densify_exp_avg_sq
            conditioning_variable_new = torch.cat([conditioning_variable_orig, conditioning_variable_densify_new], 0)
            conditioning_variable_new_exp_avg = torch.cat(
                [conditioning_variable_orig_exp_avg, conditioning_variable_densify_new_exp_avg], 0)
            conditioning_variable_new_exp_avg_sq = torch.cat(
                [conditioning_variable_orig_exp_avg_sq, conditioning_variable_densify_new_exp_avg_sq], 0)
            del conditioning_variable_orig, conditioning_variable_densify_new
            del conditioning_variable_orig_exp_avg, conditioning_variable_densify_new_exp_avg
            del conditioning_variable_orig_exp_avg_sq, conditioning_variable_densify_new_exp_avg_sq
            del conditioning_variable_prune
            del conditioning_variable_prune_exp_avg
            del conditioning_variable_prune_exp_avg_sq
            del conditioning_variable
            # ### ### ### ### ###

            state_old[13]['exp_avg'] = m_new_exp_avg
            state_old[13]['exp_avg_sq'] = m_new_exp_avg_sq
            state_old[14]['exp_avg'] = s_new_exp_avg
            state_old[14]['exp_avg_sq'] = s_new_exp_avg_sq
            state_old[15]['exp_avg'] = q_new_exp_avg
            state_old[15]['exp_avg_sq'] = q_new_exp_avg_sq
            state_old[0]['exp_avg'] = RGB_new_exp_avg
            state_old[0]['exp_avg_sq'] = RGB_new_exp_avg_sq
            state_old[1]['exp_avg'] = A_new_exp_avg
            state_old[1]['exp_avg_sq'] = A_new_exp_avg_sq
            state_old[2]['exp_avg'] = k_new_exp_avg
            state_old[2]['exp_avg_sq'] = k_new_exp_avg_sq
            state_old[11]['exp_avg'] = conditioning_variable_new_exp_avg
            state_old[11]['exp_avg_sq'] = conditioning_variable_new_exp_avg_sq
            # ### ### ### ### ###

            m = m_new
            s = s_new
            q = q_new
            RGB = RGB_new
            A = A_new
            k = k_new
            conditioning_variable = conditioning_variable_new
            # ### ### ### ### ###

            # !!! !!! !!!
            s.data.clamp_(min=np.log(pp.min_s_coef_clipping_threshold),
                          max=np.log(pp.max_s_coef_clipping_threshold * (extent / 2.0)))
            RGB.data.clamp_(min=0.0)
            # !!! !!! !!!

            # ### ### ### ### ###

            m.requires_grad_(True)
            s.requires_grad_(True)
            q.requires_grad_(True)
            RGB.requires_grad_(True)
            A.requires_grad_(True)
            k.requires_grad_(True)
            conditioning_variable.requires_grad_(True)
            # ### ### ### ### ###

            param_group = [
                {'params': [RGB], 'lr': lr_RGB_current},  # 0
                {'params': [A], 'lr': lr_A_current},  # 1
                {'params': [k], 'lr': lr_k_current},  # 2
                {'params': [w1_uv], 'lr': lr_w1_uv_current},  # 3
                {'params': [w1_v], 'lr': lr_w1_v_current},  # 4
                {'params': [w1_conditioning], 'lr': lr_w1_conditioning_current},  # 5
                {'params': [b1], 'lr': lr_b1_current},  # 6
                {'params': [w2], 'lr': lr_w2_current},  # 7
                {'params': [b2], 'lr': lr_b2_current},  # 8
                {'params': [w3], 'lr': lr_w3_current},  # 9
                {'params': [b3], 'lr': lr_b3_current},  # 10
                {'params': [conditioning_variable], 'lr': lr_conditioning_current},  # 11
                {'params': [features], 'lr': lr_features_current},  # 12
                {'params': [m], 'lr': lr_m_current},  # 13
                {'params': [s], 'lr': lr_s_current},  # 14
                {'params': [q], 'lr': lr_q_current},  # 15
            ]
            optimizer = optim.Adam(param_group)
            # ### ### ### ### ###

            state_dict_new = optimizer.state_dict()
            state_dict_new['state'] = state_old
            optimizer.load_state_dict(state_dict_new)
    else:
        # !!! !!! !!!
        s.data.clamp_(min=np.log(pp.min_s_coef_clipping_threshold),
                      max=np.log(pp.max_s_coef_clipping_threshold * (extent / 2.0)))
        RGB.data.clamp_(min=0.0)
        # !!! !!! !!!

        # ### ### ### ### ###

        optimizer.param_groups[0]['lr'] = lr_RGB_current
        optimizer.param_groups[1]['lr'] = lr_A_current
        optimizer.param_groups[2]['lr'] = lr_k_current
        optimizer.param_groups[3]['lr'] = lr_w1_uv_current
        optimizer.param_groups[4]['lr'] = lr_w1_v_current
        optimizer.param_groups[5]['lr'] = lr_w1_conditioning_current
        optimizer.param_groups[6]['lr'] = lr_b1_current
        optimizer.param_groups[7]['lr'] = lr_w2_current
        optimizer.param_groups[8]['lr'] = lr_b2_current
        optimizer.param_groups[9]['lr'] = lr_w3_current
        optimizer.param_groups[10]['lr'] = lr_b3_current
        optimizer.param_groups[11]['lr'] = lr_conditioning_current
        optimizer.param_groups[12]['lr'] = lr_features_current
        optimizer.param_groups[13]['lr'] = lr_m_current
        optimizer.param_groups[14]['lr'] = lr_s_current
        optimizer.param_groups[15]['lr'] = lr_q_current
    # ### ### ### ### ###

    extent = torch.sqrt(((torch.max(m, 0, keepdim=True)[0] - torch.min(m, 0, keepdim=True)[0]) ** 2).sum(1)).item()
    # ### ### ### ### ###

    t2 = time.perf_counter()
    training_time += t2 - t1;  # !!! !!! !!!

    # ### ### ### ### ###

    kappa_min = 1.0 + torch.nn.functional.softplus(torch.min(k)).item()
    kappa_avg = 1.0 + torch.nn.functional.softplus(torch.mean(k)).item()
    kappa_max = 1.0 + torch.nn.functional.softplus(torch.max(k)).item()
    progress_bar_1.update(1)
    if (warmup_lambda < 1.0):
        progress_bar_2.set_description_str(f"Batch PSNR base       : {PSNR_base} (Max: {PSNR_max_base})")
    else:
        progress_bar_2.set_description_str(f"Batch PSNR base       : - (Max: - )")
    if (warmup_lambda > 0.0):
        progress_bar_3.set_description_str(f"Batch PSNR FlaRe      : {PSNR_FlaRe} (Max: {PSNR_max_FlaRe})")
    else:
        progress_bar_3.set_description_str(f"Batch PSNR FlaRe      : - (Max: - )")
    progress_bar_4.set_description_str(f"kappa [min, avg, max] : {kappa_min, kappa_avg, kappa_max}")
    progress_bar_5.set_description_str(f"Number of Gaussians   : {m.shape[0]}")
# ### ### ### ### ###

def UpdateCamera():
    global O_cam, R_cam, D_cam, F_cam
    global camera_changed
    # ##############################################################################################

    # Get mouse position delta
    pt = mouse.position
    root.update()
    cx = root.winfo_rootx() + (root.winfo_width() // 2)
    cy = root.winfo_rooty() + (root.winfo_height() // 2)
    dx = pt[0] - cx
    dy = pt[1] - cy
    mouse.position = (cx, cy)
    # Rotate coordinate frame
    yaw = -2.0 * math.pi * (dx / 1000.0)
    (R_cam, D_cam, F_cam) = (
        (R_cam * math.cos(yaw)) + (F_cam * math.sin(yaw)),
        D_cam,
        (F_cam * math.cos(yaw)) - (R_cam * math.sin(yaw)),
    )
    pitch = -2.0 * math.pi * (dy / 1000.0)
    (R_cam, D_cam, F_cam) = (
        R_cam,
        (D_cam * math.cos(pitch)) + (F_cam * math.sin(pitch)),
        (F_cam * math.cos(pitch)) - (D_cam * math.sin(pitch))
    )
    # ##############################################################################################

    if (keys["a"]):
        O_cam -= R_cam * 0.025
    if (keys["d"]):
        O_cam += R_cam * 0.025
    if (keys["space"]):
        O_cam -= D_cam * 0.025
    if (keys["c"]):
        O_cam += D_cam * 0.025
    if (keys["s"]):
        O_cam -= F_cam * 0.025
    if (keys["w"]):
        O_cam += F_cam * 0.025
    # ##############################################################################################

    camera_changed = any(keys.values()) or (dx != 0) or (dy != 0) or t_key_released
# ### ### ### ### ###

def Preview():
    global tk_image
    # ##############################################################################################

    with torch.no_grad():
        O_chunk = O_cam.repeat(bitmap_height * bitmap_width, 1)
        v_chunk = py_OptiX_FLARE_renderer.GenerateRays(R_cam, D_cam, F_cam, bitmap_width, bitmap_height,
                                                       fov_x[0].item(), fov_y[0].item()).reshape(
            bitmap_width * bitmap_height, 3)
        v_chunk = v_chunk / torch.sqrt(torch.sum(v_chunk * v_chunk, 1, keepdim=True));  # WARNING

        v_chunk = v_chunk.reshape(bitmap_height, bitmap_width, 3)
        RGBA = torch.cat([RGB, A], 1)
        w1_fp16 = torch.cat([w1_uv.detach(), w1_v.detach(), w1_conditioning.detach()], 1).to(torch.float16)
        w2_fp16 = w2.detach().to(torch.float16)
        w3_fp16 = w3.detach().to(torch.float16)
        conditioning_variable_fp16 = conditioning_variable.detach().to(torch.float16)
        img_RGB = torch.zeros(bitmap_width * bitmap_height, 3, dtype=torch.float32, device="cuda");  # !!! !!! !!!

        if (mode == 0):
            renderer.Forward_inference_base(
                O_chunk, v_chunk, img_RGB,

                ep.bg_color_R, ep.bg_color_G, ep.bg_color_B,

                m, s, q, RGBA, k,

                pp.ray_termination_T_threshold_inference
            )
        else:
            renderer.Forward_inference(
                conditioning_variable_fp16,
                features,

                w1_fp16,
                b1,
                w2_fp16,
                b2,
                w3_fp16,
                b3,

                O_chunk, v_chunk, img_RGB,

                ep.bg_color_R, ep.bg_color_G, ep.bg_color_B,

                m, s, q, RGBA, k,

                pp.ray_termination_T_threshold_inference
            )
        img_RGB = torch.clamp(img_RGB, min=0.0, max=1.0).reshape((bitmap_height, bitmap_width, 3)) * 255.0
        img_array = img_RGB.detach().cpu().numpy().astype(np.uint8)
        pil_image = Image.fromarray(img_array, 'RGB')
        tk_image = ImageTk.PhotoImage(image=pil_image)
        canvas.itemconfig(image_on_canvas, image=tk_image)
# ### ### ### ### ###

def Evaluate():
    global best_checkpoint_metric, best_checkpoint_value
    progress_bar_1.clear()
    progress_bar_2.clear()
    progress_bar_3.clear()
    progress_bar_4.clear()
    progress_bar_5.clear()
    progress_bar_6.clear()
    progress_bar_1.disable = True
    progress_bar_2.disable = True
    progress_bar_3.disable = True
    progress_bar_4.disable = True
    progress_bar_5.disable = True
    progress_bar_6.disable = True
    progress_bar_6.refresh()
    progress_bar_5.refresh()
    progress_bar_4.refresh()
    progress_bar_3.refresh()
    progress_bar_2.refresh()
    progress_bar_1.refresh()
    dir_path = os.path.join("output", str(next_available_dir_id), "stats")
    with torch.no_grad():
        RGBA = torch.cat([RGB, A], 1)
        w1_fp16 = torch.cat([w1_uv.detach(), w1_v.detach(), w1_conditioning.detach()], 1).to(torch.float16)
        w2_fp16 = w2.detach().to(torch.float16)
        w3_fp16 = w3.detach().to(torch.float16)
        conditioning_variable_fp16 = conditioning_variable.detach().to(torch.float16)
        # ### ### ### ### ###

        ################
        # Base model   #
        ################

        if (warmup_lambda < 1.0):
            # TRAIN
            PSNR_total = 0.0
            t1 = time.perf_counter()
            for i in range(number_of_poses):
                pose_num = i
                O_chunk = O[pose_num, :].repeat(height * width, 1)
                v_chunk = py_OptiX_FLARE_renderer.GenerateRays(R[pose_num, :], D[pose_num, :], F[pose_num, :], width,
                                                               height, fov_x[pose_num].item(),
                                                               fov_y[pose_num].item()).reshape(width * height, 3)
                v_chunk = v_chunk / torch.sqrt(torch.sum(v_chunk * v_chunk, 1, keepdim=True));  # WARNING

                v_chunk = v_chunk.reshape(height, width, 3)
                img_RGB = torch.zeros(width * height, 3, dtype=torch.float32, device="cuda");  # !!! !!! !!!

                renderer.Forward_inference_base(
                    O_chunk, v_chunk, img_RGB,

                    ep.bg_color_R, ep.bg_color_G, ep.bg_color_B,

                    m, s, q, RGBA, k,

                    pp.ray_termination_T_threshold_inference
                )
                foreground_chunk = bitmap.reshape(number_of_poses, width * height, 3)[pose_num]
                alpha_chunk = alpha.reshape(number_of_poses, width * height, 1)[pose_num]
                ground_truth_chunk = foreground_chunk + fixed_background * (1.0 - alpha_chunk)
                loss_inference = torch.mean((torch.clamp(img_RGB, min=0.0, max=1.0) - ground_truth_chunk) ** 2)
                PSNR_inference = (-10.0 * (
                            torch.log(loss_inference) / torch.log(torch.tensor([10.0], device="cuda")))).item()
                PSNR_total += PSNR_inference
                print(i, ' : ', PSNR_inference, sep='')
            t2 = time.perf_counter()
            PSNR_avg_base = PSNR_total / number_of_poses
            FPS_base = number_of_poses / (t2 - t1)
            print('FPS (train, base): ', FPS_base, sep='')
            print('AVG PSNR (train, base): ', PSNR_avg_base, sep='')
            # ### ### ### ### ###

            # TEST
            PSNR_total = 0.0
            t1 = time.perf_counter()
            for i in range(number_of_poses_test):
                pose_num = i
                O_chunk = O_test[pose_num, :].repeat(height * width, 1)
                v_chunk = py_OptiX_FLARE_renderer.GenerateRays(R_test[pose_num, :], D_test[pose_num, :],
                                                               F_test[pose_num, :], width, height,
                                                               fov_x_test[pose_num].item(),
                                                               fov_y_test[pose_num].item()).reshape(width * height, 3)
                v_chunk = v_chunk / torch.sqrt(torch.sum(v_chunk * v_chunk, 1, keepdim=True));  # WARNING

                v_chunk = v_chunk.reshape(height, width, 3)
                img_RGB = torch.zeros(width * height, 3, dtype=torch.float32, device="cuda");  # !!! !!! !!!

                renderer.Forward_inference_base(
                    O_chunk, v_chunk, img_RGB,

                    ep.bg_color_R, ep.bg_color_G, ep.bg_color_B,

                    m, s, q, RGBA, k,

                    pp.ray_termination_T_threshold_inference
                )
                foreground_chunk = bitmap_test.reshape(number_of_poses_test, width * height, 3)[pose_num]
                alpha_chunk = alpha_test.reshape(number_of_poses_test, width * height, 1)[pose_num]
                ground_truth_chunk = foreground_chunk + fixed_background * (1.0 - alpha_chunk)
                loss_inference = torch.mean((torch.clamp(img_RGB, min=0.0, max=1.0) - ground_truth_chunk) ** 2)
                PSNR_inference = (-10.0 * (
                            torch.log(loss_inference) / torch.log(torch.tensor([10.0], device="cuda")))).item()
                PSNR_total += PSNR_inference
                print(i, ' : ', PSNR_inference, sep='')
            t2 = time.perf_counter()
            PSNR_avg_test_base = PSNR_total / number_of_poses_test
            FPS_test_base = number_of_poses_test / (t2 - t1)
            print('FPS (test, base): ', FPS_test_base, sep='')
            print('AVG PSNR (test, base): ', PSNR_avg_test_base, sep='')
        # ### ### ### ### ###

        ################
        # FlaRe model  #
        ################

        if (warmup_lambda > 0.0):
            # TRAIN
            PSNR_total = 0.0
            t1 = time.perf_counter()
            for i in range(number_of_poses):
                pose_num = i
                O_chunk = O[pose_num, :].repeat(height * width, 1)
                v_chunk = py_OptiX_FLARE_renderer.GenerateRays(R[pose_num, :], D[pose_num, :], F[pose_num, :], width,
                                                               height, fov_x[pose_num].item(),
                                                               fov_y[pose_num].item()).reshape(width * height, 3)
                v_chunk = v_chunk / torch.sqrt(torch.sum(v_chunk * v_chunk, 1, keepdim=True));  # WARNING

                v_chunk = v_chunk.reshape(height, width, 3)
                img_RGB = torch.zeros(width * height, 3, dtype=torch.float32, device="cuda");  # !!! !!! !!!

                renderer.Forward_inference(
                    conditioning_variable_fp16,
                    features,

                    w1_fp16,
                    b1,
                    w2_fp16,
                    b2,
                    w3_fp16,
                    b3,

                    O_chunk, v_chunk, img_RGB,

                    ep.bg_color_R, ep.bg_color_G, ep.bg_color_B,

                    m, s, q, RGBA, k,

                    pp.ray_termination_T_threshold_inference
                )
                foreground_chunk = bitmap.reshape(number_of_poses, width * height, 3)[pose_num]
                alpha_chunk = alpha.reshape(number_of_poses, width * height, 1)[pose_num]
                ground_truth_chunk = foreground_chunk + fixed_background * (1.0 - alpha_chunk)
                loss_inference = torch.mean((torch.clamp(img_RGB, min=0.0, max=1.0) - ground_truth_chunk) ** 2)
                PSNR_inference = (-10.0 * (
                            torch.log(loss_inference) / torch.log(torch.tensor([10.0], device="cuda")))).item()
                PSNR_total += PSNR_inference
                print(i, ' : ', PSNR_inference, sep='')
            t2 = time.perf_counter()
            PSNR_avg_FlaRe = PSNR_total / number_of_poses
            FPS_FlaRe = number_of_poses / (t2 - t1)
            print('FPS (train, FlaRe): ', FPS_FlaRe, sep='')
            print('AVG PSNR (train, FlaRe): ', PSNR_avg_FlaRe, sep='')
            # ### ### ### ### ###

            # TEST
            PSNR_total = 0.0
            t1 = time.perf_counter()
            for i in range(number_of_poses_test):
                pose_num = i
                O_chunk = O_test[pose_num, :].repeat(height * width, 1)
                v_chunk = py_OptiX_FLARE_renderer.GenerateRays(R_test[pose_num, :], D_test[pose_num, :],
                                                               F_test[pose_num, :], width, height,
                                                               fov_x_test[pose_num].item(),
                                                               fov_y_test[pose_num].item()).reshape(width * height, 3)
                v_chunk = v_chunk / torch.sqrt(torch.sum(v_chunk * v_chunk, 1, keepdim=True));  # WARNING

                v_chunk = v_chunk.reshape(height, width, 3)
                img_RGB = torch.zeros(width * height, 3, dtype=torch.float32, device="cuda");  # !!! !!! !!!

                renderer.Forward_inference(
                    conditioning_variable_fp16,
                    features,

                    w1_fp16,
                    b1,
                    w2_fp16,
                    b2,
                    w3_fp16,
                    b3,

                    O_chunk, v_chunk, img_RGB,

                    ep.bg_color_R, ep.bg_color_G, ep.bg_color_B,

                    m, s, q, RGBA, k,

                    pp.ray_termination_T_threshold_inference
                )
                foreground_chunk = bitmap_test.reshape(number_of_poses_test, width * height, 3)[pose_num]
                alpha_chunk = alpha_test.reshape(number_of_poses_test, width * height, 1)[pose_num]
                ground_truth_chunk = foreground_chunk + fixed_background * (1.0 - alpha_chunk)
                loss_inference = torch.mean((torch.clamp(img_RGB, min=0.0, max=1.0) - ground_truth_chunk) ** 2)
                PSNR_inference = (-10.0 * (
                            torch.log(loss_inference) / torch.log(torch.tensor([10.0], device="cuda")))).item()
                PSNR_total += PSNR_inference
                print(i, ' : ', PSNR_inference, sep='')
            t2 = time.perf_counter()
            PSNR_avg_test_FlaRe = PSNR_total / number_of_poses_test
            FPS_test_FlaRe = number_of_poses_test / (t2 - t1)
            print('FPS (test, FlaRe): ', FPS_test_FlaRe, sep='')
            print('AVG PSNR (test, FlaRe): ', PSNR_avg_test_FlaRe, sep='')
        # ### ### ### ### ###

        if (warmup_lambda < 1):
            f = open(os.path.join(dir_path, "PSNR_train_base.txt"), "a", encoding="utf-8")
            f.write(str(iteration) + ': ' + str(PSNR_avg_base) + '\n')
            f.close()
            f = open(os.path.join(dir_path, "PSNR_test_base.txt"), "a", encoding="utf-8")
            f.write(str(iteration) + ': ' + str(PSNR_avg_test_base) + '\n')
            f.close()
            f = open(os.path.join(dir_path, "FPS_train_base.txt"), "a", encoding="utf-8")
            f.write(str(iteration) + ': ' + str(FPS_base) + '\n')
            f.close()
            f = open(os.path.join(dir_path, "FPS_test_base.txt"), "a", encoding="utf-8")
            f.write(str(iteration) + ': ' + str(FPS_test_base) + '\n')
            f.close()
        if (warmup_lambda > 0):
            f = open(os.path.join(dir_path, "PSNR_train_FlaRe.txt"), "a", encoding="utf-8")
            f.write(str(iteration) + ': ' + str(PSNR_avg_FlaRe) + '\n')
            f.close()
            f = open(os.path.join(dir_path, "PSNR_test_FlaRe.txt"), "a", encoding="utf-8")
            f.write(str(iteration) + ': ' + str(PSNR_avg_test_FlaRe) + '\n')
            f.close()
            f = open(os.path.join(dir_path, "FPS_train_FlaRe.txt"), "a", encoding="utf-8")
            f.write(str(iteration) + ': ' + str(FPS_FlaRe) + '\n')
            f.close()
            f = open(os.path.join(dir_path, "FPS_test_FlaRe.txt"), "a", encoding="utf-8")
            f.write(str(iteration) + ': ' + str(FPS_test_FlaRe) + '\n')
            f.close()
        f = open(os.path.join(dir_path, "training_stats.txt"), "a", encoding="utf-8")
        f.write(
            str(iteration)
            + ': training_time_seconds='
            + str(training_time)
            + ', number_of_gaussians='
            + str(m.shape[0])
            + '\n'
        )
        f.close()
        if (warmup_lambda > 0.0):
            selected_metric = "PSNR_test_FlaRe"
            selected_value = PSNR_avg_test_FlaRe
        else:
            selected_metric = "PSNR_test_base"
            selected_value = PSNR_avg_test_base
        # Once FlaRe becomes active, it supersedes the warm-up base metric.
        replaces_warmup_metric = (
                best_checkpoint_metric == "PSNR_test_base"
                and selected_metric == "PSNR_test_FlaRe"
        )
        if (
                best_checkpoint_metric is None
                or replaces_warmup_metric
                or (
                selected_metric == best_checkpoint_metric
                and selected_value > best_checkpoint_value
        )
        ):
            best_checkpoint_metric = selected_metric
            best_checkpoint_value = selected_value
            SaveCheckpoint("best", selected_metric, selected_value)
    progress_bar_1.disable = False
    progress_bar_2.disable = False
    progress_bar_3.disable = False
    progress_bar_4.disable = False
    progress_bar_5.disable = False
    progress_bar_6.disable = False
    progress_bar_1.refresh()
    progress_bar_2.refresh()
    progress_bar_3.refresh()
    progress_bar_4.refresh()
    progress_bar_5.refresh()
    progress_bar_6.refresh()
    maybe_empty_cuda_cache()
# ### ### ### ### ###

def training():
    global warmup_lambda
    global iteration
    global mode
    global t_key_released
    # ##############################################################################################

    if (ap.real_time_preview):
        # !!! !!! !!!
        # Paper: lambda increases linearly from the base renderer to FlaRe.
        warmup_lambda = np.clip((iteration - ep.warmup_start_iter) / (ep.warmup_end_iter - ep.warmup_start_iter), 0, 1)
        # !!! !!! !!!

        if (free_roam):
            UpdateCamera()
        if ((not free_roam) or (not camera_changed)):
            train()
            renderer.SetGeometry(m, torch.exp(s), q, torch.sigmoid(A), 1.0 + torch.nn.functional.softplus(k))
        if (t_key_released):
            mode = (mode + 1) % 2
            t_key_released = False
        if (((iteration - 1) % ap.preview_frequency == 0) or (free_roam and camera_changed)):
            Preview()
        if ((not free_roam) or (not camera_changed)):
            if (iteration % 1000 == 0 or iteration >= ep.end_iter):
                SaveCheckpoint("last")
                Evaluate()
            iteration += 1
        maybe_empty_cuda_cache()
        if (iteration <= ep.end_iter):
            root.after(10, training)
        else:
            root.destroy()
            LaunchFinalEvaluation()
    else:
        while (True):
            # !!! !!! !!!
            # Paper: lambda increases linearly from the base renderer to FlaRe.
            warmup_lambda = np.clip((iteration - ep.warmup_start_iter) / (ep.warmup_end_iter - ep.warmup_start_iter), 0,
                                    1)
            # !!! !!! !!!

            train()
            renderer.SetGeometry(m, torch.exp(s), q, torch.sigmoid(A), 1.0 + torch.nn.functional.softplus(k))
            if (iteration % 1000 == 0 or iteration >= ep.end_iter):
                SaveCheckpoint("last")
                Evaluate()
            iteration += 1
            if (iteration > ep.end_iter):
                break
            maybe_empty_cuda_cache()
        LaunchFinalEvaluation()
# ### ### ### ### ###

if (ap.real_time_preview):
    root = tk.Tk()
    root.title("FLARE")
    root.resizable(0, 0)
    canvas = tk.Canvas(root, bd=0, highlightthickness=0, width=bitmap_width, height=bitmap_height, bg="black")
    canvas.pack()
    img_array = torch.zeros((bitmap_height, bitmap_width, 3)).numpy().astype(np.uint8)
    pil_image = Image.fromarray(img_array, 'RGB')
    tk_image = ImageTk.PhotoImage(image=pil_image)
    image_on_canvas = canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
    #########

    free_roam = False
    # Keys
    keys = {
        "w": False,
        "s": False,
        "a": False,
        "d": False,
        "space": False,
        "c": False,
        "t": False
    }
    #########

    def handle_keypress(event):
        global keys
        global t_key_released
        key = event.keysym.lower()
        if key in keys:
            if (key == 't'):
                t_key_released = False
            keys[key] = True
            #########


    def handle_keyrelease(event):
        global keys
        global t_key_released
        key = event.keysym.lower()
        if key in keys:
            if (key == 't'):
                t_key_released = True
            keys[key] = False
            #########


    def handle_doubleclick(event):
        global free_roam
        free_roam = not free_roam
        if (free_roam):
            root.update()
            cx = root.winfo_rootx() + (root.winfo_width() // 2)
            cy = root.winfo_rooty() + (root.winfo_height() // 2)
            mouse.position = (cx, cy)
    #########

    root.bind("<Key>", handle_keypress)
    root.bind("<KeyRelease>", handle_keyrelease)
    root.bind("<Double-Button-1>", handle_doubleclick)

#########

progress_bar_1 = tqdm(initial=ep.start_iter, total=ep.end_iter - ep.start_iter, desc="", position=5)
progress_bar_1.n = iteration
progress_bar_1.last_print_n = iteration
progress_bar_2 = tqdm(total=0, bar_format='{desc}', position=4)
progress_bar_3 = tqdm(total=0, bar_format='{desc}', position=3)
progress_bar_4 = tqdm(total=0, bar_format='{desc}', position=2)
progress_bar_5 = tqdm(total=0, bar_format='{desc}', position=1)
progress_bar_6 = tqdm(total=0, bar_format='{desc}', position=0)
#########

mode = 0
t_key_released = False
training()
if (ap.real_time_preview):
    root.mainloop()