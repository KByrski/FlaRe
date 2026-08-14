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

import tkinter as tk
import os
import random
import gc
import json
import shutil
import torch
import math
import numpy as np
from PIL import Image, ImageTk
import sys
from scene import Scene, GaussianModel
from scene.perspective_dataset import PerspectiveDatasetAdapter
from utils.general_utils import safe_state
from tqdm import tqdm
from checkpoint_io import (
    create_checkpoint_payload,
    save_checkpoint_payload,
)
from evaluation_service import EvaluationOptions, evaluate_training_splits
from renderer_facade import FlaReRenderer
from trainer import (
    FlaReTrainer,
    PerspectiveRayBatchSampler,
    scheduled_warmup_lambda,
    should_evaluate,
)
from arguments import EssentialParams, PerformanceParams, LearningParams, ApplicationParams, parse_args_with_config

def set_all_seeds(seed=0):
    """Seed the random number generators used during training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_all_seeds(0)

device_id = torch.cuda.current_device()
prop = torch.cuda.get_device_properties(device_id)
SM_count = prop.multi_processor_count


if os.name == "nt":
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
    os.add_dll_directory(torch_lib_path)
    os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin")

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
    return create_checkpoint_payload(
        gaussians.model_tensors(),
        optimizer_state=optimizer.state_dict(),
        iteration=iteration,
        training_time_seconds=training_time,
        config={
            "essential": vars(ep).copy(),
            "performance": vars(pp).copy(),
            "learning": vars(lp).copy(),
            "application": vars(ap).copy(),
        },
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
    # Keep the previous rolling file until its replacement is fully serialized.
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
        os.replace(temporary_path, destination)
    else:
        save_checkpoint_payload(destination, _checkpoint_payload())
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

parser = ArgumentParser(description="Training script parameters")


ep = EssentialParams(parser)
pp = PerformanceParams(parser)
lp = LearningParams(parser)
ap = ApplicationParams(parser)
args = parse_args_with_config(parser)
ep = ep.extract(args)
pp = pp.extract(args)
lp = lp.extract(args)
ap = ap.extract(args)

if (ap.real_time_preview):
    from pynput.mouse import Controller
    mouse = Controller()


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

    # Train model from scratch
    scene = Scene(ep, pp, lp, gaussians)
    gaussians.training_setup(lp)
else:
    next_available_dir_id = int(ep.model_path)
    # Load model from iteration
    scene = Scene(ep, pp, lp, gaussians, ep.start_iter)


train_dataset = PerspectiveDatasetAdapter(scene.train_cameras[1.0], device="cuda")
test_dataset = PerspectiveDatasetAdapter(scene.test_cameras[1.0], device="cuda")
width = train_dataset.width
height = train_dataset.height
bitmap_width = int(width * ap.preview_resolution_scale)
bitmap_height = int(height * ap.preview_resolution_scale)
max_batch_size = max(width * height, bitmap_width * bitmap_height)

number_of_poses = train_dataset.camera_count
fov_x = train_dataset.fov_x
fov_y = train_dataset.fov_y
O = train_dataset.origins
R = train_dataset.rights
D = train_dataset.downs
F = train_dataset.forwards
bitmap = train_dataset.foreground
alpha = train_dataset.alpha

number_of_poses_test = test_dataset.camera_count
fov_x_test = test_dataset.fov_x
fov_y_test = test_dataset.fov_y
O_test = test_dataset.origins
R_test = test_dataset.rights
D_test = test_dataset.downs
F_test = test_dataset.forwards
bitmap_test = test_dataset.foreground
alpha_test = test_dataset.alpha
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

# O_cam = O[27,:].clone()
# R_cam = R[27,:].clone()
# D_cam = D[27,:].clone()
# F_cam = F[27,:].clone()
O_cam = O_test[0, :].clone()
R_cam = R_test[0, :].clone()
D_cam = D_test[0, :].clone()
F_cam = F_test[0, :].clone()

renderer = FlaReRenderer(
    pp.number_of_sides,
    11.3449,
    max_batch_size,
)

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

iteration = gaussians.iteration + 1
training_time = gaussians.training_time
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

del scene
maybe_empty_cuda_cache()

renderer.sync_geometry(gaussians)

# Normal consistency needs complete rasters. The standard configuration keeps
# the historical random-ray batches and does not change baseline training.
normal_training_enabled = pp.reg_normal_lambda > 0.0
training_sampler = PerspectiveRayBatchSampler(
    train_dataset,
    full_camera_batches=normal_training_enabled,
    device="cuda",
)
trainer = FlaReTrainer(
    model=gaussians,
    renderer=renderer,
    optimizer=optimizer,
    sampler=training_sampler,
    essential_config=ep,
    performance_config=pp,
    learning_config=lp,
    sm_count=SM_count,
    training_time_seconds=training_time,
)

PSNR_max_base = -np.inf
PSNR_max_FlaRe = -np.inf

def train():
    global optimizer, training_time
    global RGB, A, k, conditioning_variable, m, s, q
    global PSNR_max_base, PSNR_max_FlaRe

    report = trainer.step(iteration, warmup_lambda)
    optimizer = trainer.optimizer
    training_time = trainer.training_time_seconds

    # Topology changes replace per-primitive Parameters; refresh the UI and
    # checkpoint aliases while the reusable trainer retains model ownership.
    RGB = gaussians.RGB
    A = gaussians.A
    k = gaussians.k
    conditioning_variable = gaussians.conditioning_variable
    m = gaussians.m
    s = gaussians.s
    q = gaussians.q

    if report.phases.base is not None:
        PSNR_base = report.phases.base.psnr
        if PSNR_base > PSNR_max_base:
            PSNR_max_base = PSNR_base
    if report.phases.flare is not None:
        PSNR_FlaRe = report.phases.flare.psnr
        if PSNR_FlaRe > PSNR_max_FlaRe:
            PSNR_max_FlaRe = PSNR_FlaRe

    progress_bar_1.update(1)
    if warmup_lambda < 1.0:
        progress_bar_2.set_description_str(
            f"Batch PSNR base       : {PSNR_base} (Max: {PSNR_max_base})"
        )
    else:
        progress_bar_2.set_description_str("Batch PSNR base       : - (Max: - )")
    if warmup_lambda > 0.0:
        progress_bar_3.set_description_str(
            f"Batch PSNR FlaRe      : {PSNR_FlaRe} (Max: {PSNR_max_FlaRe})"
        )
    else:
        progress_bar_3.set_description_str("Batch PSNR FlaRe      : - (Max: - )")
    progress_bar_4.set_description_str(
        "kappa [min, avg, max] : "
        + str((report.kappa_min, report.kappa_average, report.kappa_max))
    )
    progress_bar_5.set_description_str(
        f"Number of Gaussians   : {m.shape[0]}"
    )

def UpdateCamera():
    global O_cam, R_cam, D_cam, F_cam
    global camera_changed

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

    camera_changed = any(keys.values()) or (dx != 0) or (dy != 0) or t_key_released

def Preview():
    global tk_image

    with torch.no_grad():
        O_chunk = O_cam.repeat(bitmap_height * bitmap_width, 1)
        v_chunk = renderer.generate_rays(R_cam, D_cam, F_cam, bitmap_width, bitmap_height,
                                                       fov_x[0].item(), fov_y[0].item()).reshape(
            bitmap_width * bitmap_height, 3)
        v_chunk = v_chunk / torch.sqrt(torch.sum(v_chunk * v_chunk, 1, keepdim=True))  # WARNING

        v_chunk = v_chunk.reshape(bitmap_height, bitmap_width, 3)
        RGBA = torch.cat([RGB, A], 1)
        w1_fp16 = torch.cat([w1_uv.detach(), w1_v.detach(), w1_conditioning.detach()], 1).to(torch.float16)
        w2_fp16 = w2.detach().to(torch.float16)
        w3_fp16 = w3.detach().to(torch.float16)
        conditioning_variable_fp16 = conditioning_variable.detach().to(torch.float16)
        img_RGB = torch.zeros(bitmap_width * bitmap_height, 3, dtype=torch.float32, device="cuda")

        if (mode == 0):
            renderer.forward_inference_base(
                O_chunk, v_chunk, img_RGB,

                ep.bg_color_R, ep.bg_color_G, ep.bg_color_B,

                m, s, q, RGBA, k,

                pp.ray_termination_T_threshold_inference
            )
        else:
            renderer.forward_inference(
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

def Evaluate():
    global best_checkpoint_metric, best_checkpoint_value
    progress_bars = (
        progress_bar_1,
        progress_bar_2,
        progress_bar_3,
        progress_bar_4,
        progress_bar_5,
        progress_bar_6,
    )
    for progress_bar in progress_bars:
        progress_bar.clear()
    for progress_bar in progress_bars:
        progress_bar.disable = True
    for progress_bar in reversed(progress_bars):
        progress_bar.refresh()

    dir_path = os.path.join("output", str(next_available_dir_id), "stats")
    options = EvaluationOptions(
        background=(ep.bg_color_R, ep.bg_color_G, ep.bg_color_B),
        ray_termination_threshold=pp.ray_termination_T_threshold_inference,
    )
    with torch.no_grad():
        result = evaluate_training_splits(
            renderer,
            gaussians,
            train_dataset,
            test_dataset,
            options,
            warmup_lambda,
            fixed_background,
        )
        metric_order = (
            "PSNR_train_base",
            "PSNR_test_base",
            "FPS_train_base",
            "FPS_test_base",
            "PSNR_train_FlaRe",
            "PSNR_test_FlaRe",
            "FPS_train_FlaRe",
            "FPS_test_FlaRe",
        )
        for metric_name in metric_order:
            if metric_name not in result.metrics:
                continue
            with open(
                os.path.join(dir_path, metric_name + ".txt"),
                "a",
                encoding="utf-8",
            ) as metric_file:
                metric_file.write(
                    str(iteration) + ": " + str(result.metrics[metric_name]) + "\n"
                )
        with open(
            os.path.join(dir_path, "training_stats.txt"),
            "a",
            encoding="utf-8",
        ) as stats_file:
            stats_file.write(
                str(iteration)
                + ": training_time_seconds="
                + str(training_time)
                + ", number_of_gaussians="
                + str(m.shape[0])
                + "\n"
            )

        selected_metric, selected_value = result.selected_checkpoint_metric
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

    for progress_bar in progress_bars:
        progress_bar.disable = False
    for progress_bar in progress_bars:
        progress_bar.refresh()
    maybe_empty_cuda_cache()

def training():
    global warmup_lambda
    global iteration
    global mode
    global t_key_released

    if (ap.real_time_preview):
        # Paper: lambda increases linearly from the base renderer to FlaRe.
        warmup_lambda = scheduled_warmup_lambda(
            iteration, ep.warmup_start_iter, ep.warmup_end_iter
        )

        if (free_roam):
            UpdateCamera()
        if ((not free_roam) or (not camera_changed)):
            train()
            renderer.sync_geometry(gaussians)
        if (t_key_released):
            mode = (mode + 1) % 2
            t_key_released = False
        if (((iteration - 1) % ap.preview_frequency == 0) or (free_roam and camera_changed)):
            Preview()
        if ((not free_roam) or (not camera_changed)):
            if should_evaluate(iteration, ep.end_iter):
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
            # Paper: lambda increases linearly from the base renderer to FlaRe.
            warmup_lambda = scheduled_warmup_lambda(
                iteration, ep.warmup_start_iter, ep.warmup_end_iter
            )

            train()
            renderer.sync_geometry(gaussians)
            if should_evaluate(iteration, ep.end_iter):
                SaveCheckpoint("last")
                Evaluate()
            iteration += 1
            if (iteration > ep.end_iter):
                break
            maybe_empty_cuda_cache()
        LaunchFinalEvaluation()

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

    def handle_keypress(event):
        global keys
        global t_key_released
        key = event.keysym.lower()
        if key in keys:
            if (key == 't'):
                t_key_released = False
            keys[key] = True


    def handle_keyrelease(event):
        global keys
        global t_key_released
        key = event.keysym.lower()
        if key in keys:
            if (key == 't'):
                t_key_released = True
            keys[key] = False


    def handle_doubleclick(event):
        global free_roam
        free_roam = not free_roam
        if (free_roam):
            root.update()
            cx = root.winfo_rootx() + (root.winfo_width() // 2)
            cy = root.winfo_rooty() + (root.winfo_height() // 2)
            mouse.position = (cx, cy)

    root.bind("<Key>", handle_keypress)
    root.bind("<KeyRelease>", handle_keyrelease)
    root.bind("<Double-Button-1>", handle_doubleclick)


progress_bar_1 = tqdm(initial=ep.start_iter, total=ep.end_iter - ep.start_iter, desc="", position=5)
progress_bar_1.n = iteration
progress_bar_1.last_print_n = iteration
progress_bar_2 = tqdm(total=0, bar_format='{desc}', position=4)
progress_bar_3 = tqdm(total=0, bar_format='{desc}', position=3)
progress_bar_4 = tqdm(total=0, bar_format='{desc}', position=2)
progress_bar_5 = tqdm(total=0, bar_format='{desc}', position=1)
progress_bar_6 = tqdm(total=0, bar_format='{desc}', position=0)

mode = 0
t_key_released = False
training()
if (ap.real_time_preview):
    root.mainloop()
