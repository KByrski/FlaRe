"""Reusable orchestration for one FlaRe training iteration."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np
import torch

from optimizer_state import (
    DensificationConfig,
    apply_fla_re_densification,
    capture_densification_snapshot,
    update_fla_re_learning_rates,
)
from scene.rays import RayBundle
from training_step import (
    FlaReTrainingStepInput,
    FlaReTrainingStepResult,
    run_fla_re_training_step,
)


LEARNING_RATE_FIELDS = {
    "RGB": "RGB",
    "A": "A",
    "k": "k",
    "w1_uv": "w1_uv",
    "w1_v": "w1_v",
    "w1_conditioning": "w1_conditioning",
    "b1": "b1",
    "w2": "w2",
    "b2": "b2",
    "w3": "w3",
    "b3": "b3",
    "conditioning_variable": "conditioning",
    "features": "features",
    "m": "m",
    "s": "s",
    "q": "q",
}


def scheduled_warmup_lambda(
    iteration: int, start_iteration: int, end_iteration: int
) -> float:
    """Return the historical linear base-to-FlaRe blend coefficient."""
    return float(
        np.clip(
            (iteration - start_iteration) / (end_iteration - start_iteration),
            0,
            1,
        )
    )


def should_evaluate(iteration: int, end_iteration: int) -> bool:
    return iteration % 1000 == 0 or iteration >= end_iteration


def scheduled_learning_rates(learning_config, iteration: int) -> dict[str, float]:
    """Evaluate all historical exponential schedules without changing formulas."""
    values: dict[str, float] = {}
    for parameter_name, field_name in LEARNING_RATE_FIELDS.items():
        initial = getattr(learning_config, "lr_" + field_name)
        coefficient = getattr(
            learning_config, "lr_" + field_name + "_exp_decay_coef"
        )
        final = getattr(learning_config, "lr_" + field_name + "_final")
        decayed = initial * np.exp(coefficient * iteration)
        if coefficient <= 0.0:
            values[parameter_name] = float(np.maximum(decayed, final))
        else:
            values[parameter_name] = float(np.minimum(decayed, final))
    return values


@dataclass(frozen=True)
class TrainingBatch:
    rays: RayBundle
    foreground: torch.Tensor
    alpha: torch.Tensor
    image_height: int
    image_width: int
    camera_forward: torch.Tensor | None


class PerspectiveRayBatchSampler:
    """Preserve FlaRe's random-ray and complete-camera sampling schedules."""

    def __init__(
        self,
        dataset,
        *,
        full_camera_batches: bool,
        device: str | torch.device | None = None,
    ) -> None:
        self.dataset = dataset
        self.full_camera_batches = full_camera_batches
        self.device = torch.device(device or dataset.foreground.device)
        self.batch_size = dataset.width * dataset.height
        self._permutation_size = (
            dataset.camera_count
            if full_camera_batches
            else dataset.camera_count * self.batch_size
        )
        self.indices = torch.randperm(
            self._permutation_size, dtype=torch.int64, device="cpu"
        )
        self.batch_start_index = 0
        self._next_batch_start_index = 0
        self._end_of_permutation = False

    @property
    def foreground(self) -> torch.Tensor:
        return self.dataset.foreground

    @property
    def alpha(self) -> torch.Tensor:
        return self.dataset.alpha

    def next(self) -> TrainingBatch:
        training_pose: int | None = None
        if self.full_camera_batches:
            training_pose = int(self.indices[self.batch_start_index].item())
            self._end_of_permutation = (
                self.batch_start_index + 1 >= self.dataset.camera_count
            )
            indices = torch.arange(
                self.batch_size, dtype=torch.int64, device=self.device
            )
            indices += training_pose * self.batch_size
            fov_x = self.dataset.fov_x[training_pose].item()
            fov_y = self.dataset.fov_y[training_pose].item()
            self._next_batch_start_index = self.batch_start_index + 1
        else:
            batch_end_index = min(
                self.batch_start_index + self.batch_size,
                self.dataset.camera_count * self.batch_size,
            )
            self._end_of_permutation = (
                batch_end_index >= self.dataset.camera_count * self.batch_size
            )
            indices = self.indices[
                self.batch_start_index:batch_end_index
            ].to(device=self.device)
            # This intentionally retains the historical scalar-FOV random path.
            fov_x = self.dataset.fov_x[0].item()
            fov_y = self.dataset.fov_y[0].item()
            self._next_batch_start_index = batch_end_index

        rays = self.dataset.rays(indices, fov_x=fov_x, fov_y=fov_y)
        rays = RayBundle(
            origins=rays.origins,
            directions=(
                rays.directions
                / torch.sqrt(
                    torch.sum(
                        rays.directions * rays.directions,
                        1,
                        keepdim=True,
                    )
                )
            ),
            flat_indices=rays.flat_indices,
            camera_indices=rays.camera_indices,
            pixel_indices=rays.pixel_indices,
        )
        return TrainingBatch(
            rays=rays,
            foreground=self.dataset.foreground,
            alpha=self.dataset.alpha,
            image_height=self.dataset.height,
            image_width=self.dataset.width,
            camera_forward=(
                self.dataset.forwards[training_pose]
                if training_pose is not None
                else None
            ),
        )

    def advance(self) -> None:
        if self._end_of_permutation:
            self.indices = torch.randperm(
                self._permutation_size, dtype=torch.int64, device="cpu"
            )
            self.batch_start_index = 0
        else:
            self.batch_start_index = self._next_batch_start_index


class FlaReTopologyController:
    """Current 3D topology policy."""

    @staticmethod
    def extent(model) -> float:
        return torch.sqrt(
            (
                (
                    torch.max(model.m, 0, keepdim=True)[0]
                    - torch.min(model.m, 0, keepdim=True)[0]
                )
                ** 2
            ).sum(1)
        ).item()

    @staticmethod
    def is_due(iteration: int, performance_config) -> bool:
        return (
            iteration >= performance_config.densification_start_iter
            and iteration <= performance_config.densification_end_iter
            and iteration % performance_config.densification_frequency == 0
        )

    @staticmethod
    def snapshot(model, optimizer, iteration: int):
        return capture_densification_snapshot(
            model, optimizer, first_step=(iteration == 1)
        )

    @staticmethod
    def apply(
        model,
        optimizer,
        snapshot,
        performance_config,
        extent: float,
        learning_rates: dict[str, float],
    ):
        return apply_fla_re_densification(
            model,
            optimizer,
            snapshot,
            DensificationConfig(
                opacity_threshold=(
                    performance_config.opacity_threshold_for_Gauss_removal
                ),
                minimum_scale_norm=(
                    performance_config.min_s_norm_threshold_for_Gauss_removal
                ),
                movement_threshold=(
                    performance_config.mu_grad_norm_threshold_for_densification
                ),
                maximum_gaussians=performance_config.max_Gaussians_per_model,
                minimum_scale=(
                    performance_config.min_s_coef_clipping_threshold
                ),
                maximum_scale_fraction=(
                    performance_config.max_s_coef_clipping_threshold
                ),
            ),
            extent,
            learning_rates,
        )

    @staticmethod
    def constrain(
        model,
        optimizer,
        performance_config,
        extent: float,
        learning_rates: dict[str, float],
    ) -> None:
        model.s.data.clamp_(
            min=np.log(performance_config.min_s_coef_clipping_threshold),
            max=np.log(
                performance_config.max_s_coef_clipping_threshold
                * (extent / 2.0)
            ),
        )
        model.RGB.data.clamp_(min=0.0)
        update_fla_re_learning_rates(optimizer, learning_rates)


@dataclass(frozen=True)
class TrainerStepReport:
    phases: FlaReTrainingStepResult
    elapsed_seconds: float
    extent: float
    kappa_min: float
    kappa_average: float
    kappa_max: float


class FlaReTrainer:
    """Coordinate sampling, one isolated step, LR updates, and topology."""

    def __init__(
        self,
        *,
        model,
        renderer,
        optimizer,
        sampler: PerspectiveRayBatchSampler,
        essential_config,
        performance_config,
        learning_config,
        sm_count: int,
        training_time_seconds: float = 0.0,
        topology_controller=None,
        step_function: Callable = run_fla_re_training_step,
    ) -> None:
        self.model = model
        self.renderer = renderer
        self.optimizer = optimizer
        self.sampler = sampler
        self.essential = essential_config
        self.performance = performance_config
        self.learning = learning_config
        self.sm_count = sm_count
        self.topology = topology_controller or FlaReTopologyController()
        self.step_function = step_function
        self.extent = self.topology.extent(model)
        self.training_time_seconds = training_time_seconds
        self.fixed_background = torch.tensor(
            [
                essential_config.bg_color_R,
                essential_config.bg_color_G,
                essential_config.bg_color_B,
            ],
            dtype=torch.float32,
            device=sampler.foreground.device,
        ).reshape(1, 3)
        self.random_background_enabled = (
            performance_config.random_background
            and bool(torch.any(sampler.alpha < 1.0).item())
        )
        self.reg_depth_a = -(
            essential_config.t_near * essential_config.t_far
        ) / (essential_config.t_far - essential_config.t_near)
        self.reg_depth_b = essential_config.t_far / (
            essential_config.t_far - essential_config.t_near
        )

    def _background(self):
        if not self.random_background_enabled:
            return self.fixed_background, (
                self.essential.bg_color_R,
                self.essential.bg_color_G,
                self.essential.bg_color_B,
            )
        values = np.random.random(3).astype(np.float32)
        background = torch.as_tensor(
            values, device=self.sampler.foreground.device
        ).reshape(1, 3)
        return background, tuple(values.tolist())

    def step(self, iteration: int, warmup_lambda: float) -> TrainerStepReport:
        started = time.perf_counter()
        depth_lambda = (
            self.performance.reg_depth_lambda
            if iteration > self.performance.reg_depth_start_iter
            else 0.0
        )
        normal_ramp = np.clip(
            (iteration - self.performance.reg_normal_start_iter)
            / max(float(self.performance.reg_normal_ramp_iters), 1.0),
            0.0,
            1.0,
        )
        normal_lambda = self.performance.reg_normal_lambda * normal_ramp
        background, background_rgb = self._background()
        batch = self.sampler.next()

        densification_due = self.topology.is_due(
            iteration, self.performance
        )
        snapshot = (
            self.topology.snapshot(self.model, self.optimizer, iteration)
            if densification_due
            else None
        )
        phases = self.step_function(
            FlaReTrainingStepInput(
                model=self.model,
                renderer=self.renderer,
                optimizer=self.optimizer,
                rays=batch.rays,
                foreground=batch.foreground,
                alpha=batch.alpha,
                background=background,
                background_rgb=background_rgb,
                warmup_lambda=warmup_lambda,
                ray_termination_threshold=(
                    self.performance.ray_termination_T_threshold_training
                ),
                depth_lambda=depth_lambda,
                normal_lambda=normal_lambda,
                reg_depth_a=self.reg_depth_a,
                reg_depth_b=self.reg_depth_b,
                scale_regularization=self.performance.reg_scale_lambda,
                sm_count=self.sm_count,
                image_height=batch.image_height,
                image_width=batch.image_width,
                normal_depth_edge_threshold=(
                    self.performance.reg_normal_depth_edge_threshold
                ),
                camera_forward=(
                    batch.camera_forward if normal_lambda > 0.0 else None
                ),
            )
        )
        self.sampler.advance()

        learning_rates = scheduled_learning_rates(self.learning, iteration)
        if densification_due:
            self.optimizer = self.topology.apply(
                self.model,
                self.optimizer,
                snapshot,
                self.performance,
                self.extent,
                learning_rates,
            )
            self.model.optimizer = self.optimizer
        else:
            self.topology.constrain(
                self.model,
                self.optimizer,
                self.performance,
                self.extent,
                learning_rates,
            )
        self.extent = self.topology.extent(self.model)
        elapsed = time.perf_counter() - started
        self.training_time_seconds += elapsed

        return TrainerStepReport(
            phases=phases,
            elapsed_seconds=elapsed,
            extent=self.extent,
            kappa_min=(
                1.0 + torch.nn.functional.softplus(torch.min(self.model.k)).item()
            ),
            kappa_average=(
                1.0 + torch.nn.functional.softplus(torch.mean(self.model.k)).item()
            ),
            kappa_max=(
                1.0 + torch.nn.functional.softplus(torch.max(self.model.k)).item()
            ),
        )
