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

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from renderer_facade import FlaReRenderer
from scene.rays import RayBundle
from utils.depth_utils import ray_parameters_to_depth_and_normal

if TYPE_CHECKING:
    from scene.gaussian_model import GaussianModel


@dataclass(frozen=True)
class FlaReTrainingStepInput:
    """All data and coefficients consumed by one FlaRe optimizer step."""

    model: "GaussianModel"
    renderer: FlaReRenderer
    optimizer: torch.optim.Optimizer
    rays: RayBundle
    foreground: torch.Tensor
    alpha: torch.Tensor
    background: torch.Tensor
    background_rgb: tuple[float, float, float]
    warmup_lambda: float
    ray_termination_threshold: float
    depth_lambda: float
    normal_lambda: float
    reg_depth_a: float
    reg_depth_b: float
    scale_regularization: float
    sm_count: int
    image_height: int
    image_width: int
    normal_depth_edge_threshold: float
    camera_forward: torch.Tensor | None = None


@dataclass(frozen=True)
class PhaseMetrics:
    loss: float
    psnr: float


@dataclass(frozen=True)
class NativeGradientBundle:
    """The exact sixteen gradients assigned to the FlaRe model parameters."""

    RGB: torch.Tensor
    A: torch.Tensor
    k: torch.Tensor
    w1_uv: torch.Tensor
    w1_v: torch.Tensor
    w1_conditioning: torch.Tensor
    b1: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor
    w3: torch.Tensor
    b3: torch.Tensor
    conditioning_variable: torch.Tensor
    features: torch.Tensor
    m: torch.Tensor
    s: torch.Tensor
    q: torch.Tensor

    def assign_to(self, model: "GaussianModel") -> None:
        for name, gradient in self.as_dict().items():
            getattr(model, name).grad = gradient

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "RGB": self.RGB,
            "A": self.A,
            "k": self.k,
            "w1_uv": self.w1_uv,
            "w1_v": self.w1_v,
            "w1_conditioning": self.w1_conditioning,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "w3": self.w3,
            "b3": self.b3,
            "conditioning_variable": self.conditioning_variable,
            "features": self.features,
            "m": self.m,
            "s": self.s,
            "q": self.q,
        }


@dataclass(frozen=True)
class FlaReTrainingStepResult:
    base: PhaseMetrics | None
    flare: PhaseMetrics | None
    gradients: NativeGradientBundle


@dataclass
class _BlendedGradients:
    RGB: torch.Tensor
    A: torch.Tensor
    k: torch.Tensor
    w3: torch.Tensor
    b3: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor
    w1: torch.Tensor
    b1: torch.Tensor
    conditioning_variable: torch.Tensor
    features: torch.Tensor
    m: torch.Tensor
    s: torch.Tensor
    q: torch.Tensor

    @classmethod
    def zeros(cls, model: "GaussianModel") -> "_BlendedGradients":
        return cls(
            RGB=torch.zeros_like(model.RGB),
            A=torch.zeros_like(model.A),
            k=torch.zeros_like(model.k),
            w3=torch.zeros(8, 64, dtype=torch.float32, device="cuda"),
            b3=torch.zeros(8, dtype=torch.float32, device="cuda"),
            w2=torch.zeros(64, 64, dtype=torch.float32, device="cuda"),
            b2=torch.zeros(64, dtype=torch.float32, device="cuda"),
            w1=torch.zeros(64, 128, dtype=torch.float32, device="cuda"),
            b1=torch.zeros(64, dtype=torch.float32, device="cuda"),
            conditioning_variable=torch.zeros_like(model.conditioning_variable),
            features=torch.zeros_like(
                model.features, dtype=torch.float32, device="cuda"
            ),
            m=torch.zeros_like(model.m),
            s=torch.zeros_like(model.s),
            q=torch.zeros_like(model.q),
        )

    def parameter_bundle(
        self,
        model: "GaussianModel",
        scale_regularization: float,
    ) -> NativeGradientBundle:
        scales_squared = torch.exp(model.s.detach()) ** 2
        scale_gradient = (
            scale_regularization
            / model.m.shape[0]
            * scales_squared
            / torch.sqrt(scales_squared.sum(1, keepdim=True))
        )
        return NativeGradientBundle(
            RGB=self.RGB,
            A=self.A,
            k=self.k,
            w1_uv=self.w1[:, 0:8],
            w1_v=self.w1[:, 8:32],
            w1_conditioning=self.w1[:, 32:128],
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            w3=torch.nn.functional.pad(
                self.w3, (0, 0, 0, 8), "constant", 0.0
            ),
            b3=torch.nn.functional.pad(self.b3, (0, 8), "constant", 0.0),
            conditioning_variable=self.conditioning_variable,
            features=self.features,
            m=self.m,
            s=self.s + scale_gradient,
            q=self.q,
        )


@dataclass(frozen=True)
class _AuxiliaryBuffers:
    depth: torch.Tensor
    depth_and_index: torch.Tensor
    surface_normal: torch.Tensor
    normal: torch.Tensor

    @classmethod
    def zeros(cls, batch_size: int) -> "_AuxiliaryBuffers":
        return cls(
            depth=torch.zeros(
                (batch_size, 4), dtype=torch.float32, device="cuda"
            ),
            depth_and_index=torch.zeros(
                (batch_size, 2), dtype=torch.float32, device="cuda"
            ),
            surface_normal=torch.zeros(
                (batch_size, 3), dtype=torch.float32, device="cuda"
            ),
            normal=torch.zeros(
                (batch_size, 4), dtype=torch.float32, device="cuda"
            ),
        )


def _surface_normal(
    step: FlaReTrainingStepInput,
    auxiliary: _AuxiliaryBuffers,
) -> torch.Tensor:
    if step.normal_lambda <= 0.0:
        return auxiliary.surface_normal
    if step.camera_forward is None:
        raise ValueError(
            "camera_forward is required when normal regularization is active"
        )
    expected_t = (
        auxiliary.depth[:, 3]
        / auxiliary.depth[:, 1].clamp_min(1.0e-8)
    )
    _, surface_normal, _ = ray_parameters_to_depth_and_normal(
        expected_t,
        auxiliary.depth[:, 1] > 0.01,
        step.rays.origins,
        step.rays.directions,
        step.camera_forward,
        step.image_height,
        step.image_width,
        step.normal_depth_edge_threshold,
    )
    return surface_normal.reshape(step.rays.origins.shape[0], 3).contiguous()


def _loss_and_image_gradient(
    step: FlaReTrainingStepInput,
    image_unclamped: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = torch.clamp(image_unclamped, min=0.0, max=1.0)
    image = image.detach().requires_grad_(True)
    indices = step.rays.flat_indices
    foreground = torch.gather(
        step.foreground, 0, indices.unsqueeze(1).expand(-1, 3)
    )
    alpha = torch.gather(step.alpha, 0, indices.unsqueeze(1))
    target = foreground + step.background * (1.0 - alpha)
    loss = torch.mean((image - target) ** 2)
    loss.backward()
    image_gradient = image.grad * (~(image_unclamped != image))
    return loss, image, image_gradient


def _metrics(loss: torch.Tensor) -> PhaseMetrics:
    psnr = (
        -10.0
        * (
            torch.log(loss)
            / torch.log(torch.tensor([10.0], device="cuda"))
        )
    ).item()
    return PhaseMetrics(loss=loss.item(), psnr=psnr)


def _run_base_phase(
    step: FlaReTrainingStepInput,
    blended: _BlendedGradients,
    rgba: torch.Tensor,
) -> PhaseMetrics:
    model = step.model
    batch_size = step.rays.origins.shape[0]
    image_unclamped = torch.zeros(
        (batch_size, 3), dtype=torch.float32, device="cuda"
    )
    auxiliary = _AuxiliaryBuffers.zeros(batch_size)
    step.renderer.forward_training_base(
        step.rays.origins,
        step.rays.directions,
        image_unclamped,
        *step.background_rgb,
        model.m,
        model.s,
        model.q,
        rgba,
        model.k,
        step.ray_termination_threshold,
        auxiliary.depth,
        step.reg_depth_a,
        step.reg_depth_b,
        auxiliary.depth_and_index,
        auxiliary.surface_normal,
        auxiliary.normal,
    )
    surface_normal = _surface_normal(step, auxiliary)
    loss, image, image_gradient = _loss_and_image_gradient(
        step, image_unclamped
    )

    with torch.no_grad():
        dL_dRGB = torch.zeros_like(model.RGB)
        dL_dA = torch.zeros_like(model.A)
        dL_dk = torch.zeros_like(model.k)
        dL_dm = torch.zeros_like(model.m)
        dL_ds = torch.zeros_like(model.s)
        dL_dq = torch.zeros_like(model.q)
        prefix = torch.zeros(
            (batch_size, 4), dtype=torch.float32, device="cuda"
        )
        step.renderer.backward_base(
            step.rays.origins,
            step.rays.directions,
            *step.background_rgb,
            model.m,
            model.s,
            model.q,
            rgba,
            model.k,
            image,
            image_gradient,
            dL_dRGB,
            dL_dA,
            dL_dk,
            dL_dm,
            dL_ds,
            dL_dq,
            step.ray_termination_threshold,
            auxiliary.depth,
            prefix,
            step.depth_lambda / batch_size,
            step.reg_depth_a,
            step.reg_depth_b,
            auxiliary.depth_and_index,
            surface_normal,
            auxiliary.normal,
            step.normal_lambda / batch_size,
        )
        weight = 1.0 - step.warmup_lambda
        blended.RGB.add_(weight * dL_dRGB)
        blended.A.add_(weight * dL_dA)
        blended.k.add_(weight * dL_dk)
        blended.m.add_(weight * dL_dm)
        blended.s.add_(weight * dL_ds)
        blended.q.add_(weight * dL_dq)

    return _metrics(loss)


def _reduce_full_mlp_gradients(
    dL_dw3: torch.Tensor,
    dL_db3: torch.Tensor,
    dL_dw2: torch.Tensor,
    dL_db2: torch.Tensor,
    dL_dw1: torch.Tensor,
    dL_db1: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
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
    return dL_dw3, dL_db3, dL_dw2, dL_db2, dL_dw1, dL_db1


def _run_full_phase(
    step: FlaReTrainingStepInput,
    blended: _BlendedGradients,
    rgba: torch.Tensor,
) -> PhaseMetrics:
    model = step.model
    batch_size = step.rays.origins.shape[0]
    w1_fp16 = torch.cat(
        [
            model.w1_uv.detach(),
            model.w1_v.detach(),
            model.w1_conditioning.detach(),
        ],
        1,
    ).to(torch.float16)
    w2_fp16 = model.w2.detach().to(torch.float16)
    w3_fp16 = model.w3.detach().to(torch.float16)
    conditioning_fp16 = model.conditioning_variable.detach().to(torch.float16)
    image_unclamped = torch.zeros(
        (batch_size, 3), dtype=torch.float32, device="cuda"
    )
    auxiliary = _AuxiliaryBuffers.zeros(batch_size)
    step.renderer.forward_training(
        conditioning_fp16,
        model.features,
        w1_fp16,
        model.b1,
        w2_fp16,
        model.b2,
        w3_fp16,
        model.b3,
        step.rays.origins,
        step.rays.directions,
        image_unclamped,
        *step.background_rgb,
        model.m,
        model.s,
        model.q,
        rgba,
        model.k,
        step.ray_termination_threshold,
        auxiliary.depth,
        step.reg_depth_a,
        step.reg_depth_b,
        auxiliary.depth_and_index,
        auxiliary.surface_normal,
        auxiliary.normal,
    )
    surface_normal = _surface_normal(step, auxiliary)
    loss, image, image_gradient = _loss_and_image_gradient(
        step, image_unclamped
    )

    with torch.no_grad():
        sm_buffers = 4 * step.sm_count
        dL_dRGB = torch.zeros_like(model.RGB)
        dL_dA = torch.zeros_like(model.A)
        dL_dk = torch.zeros_like(model.k)
        dL_dw3 = torch.zeros(
            (sm_buffers, 8 * 64), dtype=torch.float32, device="cuda"
        )
        dL_db3 = torch.zeros(
            (sm_buffers, 8 * 8), dtype=torch.float32, device="cuda"
        )
        dL_dw2 = torch.zeros(
            (sm_buffers, 64 * 64), dtype=torch.float32, device="cuda"
        )
        dL_db2 = torch.zeros(
            (sm_buffers, 64 * 8), dtype=torch.float32, device="cuda"
        )
        dL_dw1 = torch.zeros(
            (sm_buffers, 64 * 128), dtype=torch.float32, device="cuda"
        )
        dL_db1 = torch.zeros(
            (sm_buffers, 64 * 8), dtype=torch.float32, device="cuda"
        )
        dL_d_conditioning = torch.zeros_like(model.conditioning_variable)
        dL_d_features = torch.zeros_like(
            model.features, dtype=torch.float32, device="cuda"
        )
        dL_dm = torch.zeros_like(model.m)
        dL_ds = torch.zeros_like(model.s)
        dL_dq = torch.zeros_like(model.q)
        prefix = torch.zeros(
            (batch_size, 4), dtype=torch.float32, device="cuda"
        )
        step.renderer.backward(
            conditioning_fp16,
            model.features,
            w1_fp16,
            model.b1,
            w2_fp16,
            model.b2,
            w3_fp16,
            model.b3,
            step.rays.origins,
            step.rays.directions,
            *step.background_rgb,
            model.m,
            model.s,
            model.q,
            rgba,
            model.k,
            image,
            image_gradient,
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
            step.ray_termination_threshold,
            auxiliary.depth,
            prefix,
            step.depth_lambda / batch_size,
            step.reg_depth_a,
            step.reg_depth_b,
            auxiliary.depth_and_index,
            surface_normal,
            auxiliary.normal,
            step.normal_lambda / batch_size,
        )
        (
            dL_dw3,
            dL_db3,
            dL_dw2,
            dL_db2,
            dL_dw1,
            dL_db1,
        ) = _reduce_full_mlp_gradients(
            dL_dw3, dL_db3, dL_dw2, dL_db2, dL_dw1, dL_db1
        )
        weight = step.warmup_lambda
        blended.RGB.add_(weight * dL_dRGB)
        blended.A.add_(weight * dL_dA)
        blended.k.add_(weight * dL_dk)
        blended.w3.add_(weight * dL_dw3)
        blended.b3.add_(weight * dL_db3)
        blended.w2.add_(weight * dL_dw2)
        blended.b2.add_(weight * dL_db2)
        blended.w1.add_(weight * dL_dw1)
        blended.b1.add_(weight * dL_db1)
        blended.conditioning_variable.add_(weight * dL_d_conditioning)
        blended.features.add_(weight * dL_d_features)
        blended.m.add_(weight * dL_dm)
        blended.s.add_(weight * dL_ds)
        blended.q.add_(weight * dL_dq)

    return _metrics(loss)


def run_fla_re_training_step(
    step: FlaReTrainingStepInput,
) -> FlaReTrainingStepResult:
    """Render, backpropagate native gradients, and perform one Adam step."""

    model = step.model
    rgba = torch.cat([model.RGB, model.A], 1)
    blended = _BlendedGradients.zeros(model)

    base_metrics = None
    if step.warmup_lambda < 1.0:
        base_metrics = _run_base_phase(step, blended, rgba)

    flare_metrics = None
    if step.warmup_lambda > 0.0:
        flare_metrics = _run_full_phase(step, blended, rgba)

    gradients = blended.parameter_bundle(model, step.scale_regularization)
    gradients.assign_to(model)
    step.optimizer.step()
    return FlaReTrainingStepResult(
        base=base_metrics,
        flare=flare_metrics,
        gradients=gradients,
    )
