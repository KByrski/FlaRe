#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np
import torch

from checkpoint_io import MODEL_TENSOR_NAMES

if TYPE_CHECKING:
    from scene.gaussian_model import GaussianModel


OPTIMIZER_GROUP_NAMES = MODEL_TENSOR_NAMES
PER_PRIMITIVE_PARAMETER_NAMES = (
    "RGB",
    "A",
    "k",
    "conditioning_variable",
    "m",
    "s",
    "q",
)


@dataclass(frozen=True)
class DensificationSnapshot:
    means: torch.Tensor
    means_exp_avg: torch.Tensor
    means_exp_avg_sq: torch.Tensor


@dataclass(frozen=True)
class DensificationConfig:
    opacity_threshold: float
    minimum_scale_norm: float
    movement_threshold: float
    maximum_gaussians: int
    minimum_scale: float
    maximum_scale_fraction: float


def named_optimizer_groups(
    optimizer: torch.optim.Optimizer,
) -> dict[str, dict]:
    if len(optimizer.param_groups) != len(OPTIMIZER_GROUP_NAMES):
        raise ValueError(
            "FlaRe optimizer must have exactly "
            f"{len(OPTIMIZER_GROUP_NAMES)} parameter groups"
        )
    groups: dict[str, dict] = {}
    for expected_name, group in zip(
        OPTIMIZER_GROUP_NAMES, optimizer.param_groups
    ):
        existing_name = group.get("name")
        if existing_name is not None and existing_name != expected_name:
            raise ValueError(
                f"optimizer group {existing_name!r} occupies the "
                f"{expected_name!r} compatibility slot"
            )
        group["name"] = expected_name
        groups[expected_name] = group
    return groups


def build_fla_re_optimizer(
    model: "GaussianModel",
    learning_rates: Mapping[str, float],
) -> torch.optim.Adam:
    missing = set(OPTIMIZER_GROUP_NAMES) - set(learning_rates)
    if missing:
        raise ValueError(
            f"missing FlaRe learning rates: {sorted(missing)}"
        )
    groups = [
        {
            "params": [getattr(model, name)],
            "lr": float(learning_rates[name]),
            "name": name,
        }
        for name in OPTIMIZER_GROUP_NAMES
    ]
    return torch.optim.Adam(groups)


def update_fla_re_learning_rates(
    optimizer: torch.optim.Optimizer,
    learning_rates: Mapping[str, float],
) -> None:
    for name, group in named_optimizer_groups(optimizer).items():
        group["lr"] = float(learning_rates[name])


def capture_densification_snapshot(
    model: "GaussianModel",
    optimizer: torch.optim.Optimizer,
    *,
    first_step: bool,
) -> DensificationSnapshot:
    means = model.m.detach().clone()
    if first_step:
        means_exp_avg = torch.zeros_like(means)
        means_exp_avg_sq = torch.zeros_like(means)
    else:
        group = named_optimizer_groups(optimizer)["m"]
        state = optimizer.state[group["params"][0]]
        means_exp_avg = state["exp_avg"].clone()
        means_exp_avg_sq = state["exp_avg_sq"].clone()
    return DensificationSnapshot(
        means=means,
        means_exp_avg=means_exp_avg,
        means_exp_avg_sq=means_exp_avg_sq,
    )


def _mapped_rows(
    value: torch.Tensor,
    original_indices: torch.Tensor,
    split_indices: torch.Tensor,
    *,
    split_before: torch.Tensor | None = None,
) -> torch.Tensor:
    middle = (
        value[split_indices]
        if split_before is None
        else split_before[split_indices]
    )
    return torch.cat(
        (
            value[original_indices],
            middle,
            value[split_indices],
        ),
        dim=0,
    )


def apply_fla_re_densification(
    model: "GaussianModel",
    optimizer: torch.optim.Optimizer,
    snapshot: DensificationSnapshot,
    config: DensificationConfig,
    extent: float,
    learning_rates: Mapping[str, float],
) -> torch.optim.Adam:
    """Apply the historical FlaRe prune/split rule and migrate Adam state."""

    named_optimizer_groups(optimizer)
    old_state_dict = optimizer.state_dict()
    old_state = old_state_dict["state"]
    group_indices = {
        group["name"]: group["params"][0]
        for group in old_state_dict["param_groups"]
    }

    with torch.no_grad():
        raw_kappa = model.k.detach()
        opacity_logits = model.A.detach()
        log_scales = model.s.detach()
        kappas = 1.0 + torch.nn.functional.softplus(raw_kappa)
        opacities = torch.sigmoid(opacity_logits)
        scale = torch.clamp(
            kappas * (11.3449 + (2.0 * torch.log(opacities))),
            min=0.0,
        )
        scale = (scale ** (1.0 / (2.0 * kappas))) / np.sqrt(11.3449)
        scale_norm = (
            torch.sqrt((torch.exp(log_scales) ** 2).sum(1))
            * scale.squeeze(1)
        )
        keep = (
            opacity_logits
            >= np.log(
                config.opacity_threshold
                / (1.0 - config.opacity_threshold)
            )
        ).squeeze(1) & (scale_norm >= config.minimum_scale_norm)

        kept_indices = torch.nonzero(keep, as_tuple=False).squeeze(1)
        kept_means = model.m.detach()[kept_indices]
        movement = torch.sqrt(
            torch.sum(
                (
                    kept_means
                    - snapshot.means[kept_indices]
                )
                ** 2,
                1,
                keepdim=True,
            )
        )
        split = (
            (movement >= config.movement_threshold)
            & (
                (config.maximum_gaussians == -1)
                | (kept_means.shape[0] <= config.maximum_gaussians)
            )
        ).squeeze(1)
        original_indices = kept_indices[~split]
        split_indices = kept_indices[split]

        new_values: dict[str, torch.Tensor] = {}
        mapped_states: dict[str, dict] = {}
        for name in PER_PRIMITIVE_PARAMETER_NAMES:
            parameter = getattr(model, name)
            value = parameter.detach()
            new_values[name] = _mapped_rows(
                value,
                original_indices,
                split_indices,
                split_before=(
                    snapshot.means if name == "m" else None
                ),
            )

            state = old_state[group_indices[name]]
            mapped_state = dict(state)
            mapped_state["exp_avg"] = _mapped_rows(
                state["exp_avg"],
                original_indices,
                split_indices,
                split_before=(
                    snapshot.means_exp_avg if name == "m" else None
                ),
            )
            mapped_state["exp_avg_sq"] = _mapped_rows(
                state["exp_avg_sq"],
                original_indices,
                split_indices,
                split_before=(
                    snapshot.means_exp_avg_sq if name == "m" else None
                ),
            )
            mapped_states[name] = mapped_state

        model.replace_parameters(new_values)
        model.s.data.clamp_(
            min=np.log(config.minimum_scale),
            max=np.log(config.maximum_scale_fraction * (extent / 2.0)),
        )
        model.RGB.data.clamp_(min=0.0)

        new_optimizer = build_fla_re_optimizer(model, learning_rates)
        new_state_dict = new_optimizer.state_dict()
        new_state = {}
        for index, name in enumerate(OPTIMIZER_GROUP_NAMES):
            if name in mapped_states:
                new_state[index] = mapped_states[name]
            else:
                new_state[index] = old_state[group_indices[name]]
        new_state_dict["state"] = new_state
        new_optimizer.load_state_dict(new_state_dict)
        named_optimizer_groups(new_optimizer)
        return new_optimizer
