"""Typed configuration schema for the current FlaRe training interface."""

from dataclasses import dataclass, fields
from typing import Any, Mapping


class ConfigGroup:
    """Shared construction helpers for one typed configuration group."""

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]):
        return cls(**{field.name: values[field.name] for field in fields(cls)})

    @classmethod
    def from_namespace(cls, namespace):
        return cls.from_mapping(vars(namespace))


@dataclass
class EssentialConfig(ConfigGroup):
    source_path: str = ""
    model_path: str = ""
    start_iter: int = 0
    end_iter: int = 64_000
    # Paper: endpoints of the linear base-to-FlaRe warmup schedule.
    warmup_start_iter: int = 0
    warmup_end_iter: int = 1_000
    resolution: int = 1_000_000
    bg_color_R: float = 0.0
    bg_color_G: float = 0.0
    bg_color_B: float = 0.0
    t_near: float = 0.1
    t_far: float = 1_000.0
    images: str = "images"
    data_device: str = "cuda"


@dataclass
class PerformanceConfig(ConfigGroup):
    # Random-background transparency carving is applied only when the
    # training images contain a non-opaque alpha channel.
    random_background: bool = False
    number_of_sides: int = 8
    border_opacity: float = 0.003439
    initial_conditioning_std: float = 0.01
    initial_opacity: float = 0.01
    initial_k: float = 1.01
    ray_termination_T_threshold_training: float = 0.0001
    ray_termination_T_threshold_inference: float = 0.01
    opacity_threshold_for_Gauss_removal: float = 1.0 / 28.0
    densification_frequency: int = 100
    densification_start_iter: int = 800
    densification_end_iter: int = 24_000
    # Paper: fixed primitive budgets used by the representation-reduction study.
    max_Gaussians_per_model: int = -1
    mu_grad_norm_threshold_for_densification: float = 0.001
    min_s_norm_threshold_for_Gauss_removal: float = 0.0001
    min_s_coef_clipping_threshold: float = 0.0001
    max_s_coef_clipping_threshold: float = 0.05
    # Paper: geometry-aware regularization setting for surface alignment.
    reg_depth_lambda: float = 0.0001
    reg_depth_start_iter: int = 0
    reg_normal_lambda: float = 0.0
    reg_normal_start_iter: int = 7_000
    reg_normal_ramp_iters: int = 3_000
    reg_normal_depth_edge_threshold: float = 0.02
    reg_scale_lambda: float = 0.001


@dataclass
class LearningConfig(ConfigGroup):
    lr_RGB: float = 0.005
    lr_RGB_exp_decay_coef: float = -0.000086643
    lr_RGB_final: float = 0.001

    lr_A: float = 0.05
    lr_A_exp_decay_coef: float = -0.000086643
    lr_A_final: float = 0.025

    lr_k: float = 0.1
    lr_k_exp_decay_coef: float = -0.000086643
    lr_k_final: float = 0.1

    lr_w1_uv: float = 0.0125
    lr_w1_uv_exp_decay_coef: float = -0.000086643
    lr_w1_uv_final: float = 0.001

    lr_w1_v: float = 0.03
    lr_w1_v_exp_decay_coef: float = -0.000086643
    lr_w1_v_final: float = 0.001

    lr_w1_conditioning: float = 0.0015
    lr_w1_conditioning_exp_decay_coef: float = -0.000086643
    lr_w1_conditioning_final: float = 0.001

    lr_b1: float = 0.00475
    lr_b1_exp_decay_coef: float = -0.000086643
    lr_b1_final: float = 0.001

    lr_w2: float = 0.005
    lr_w2_exp_decay_coef: float = -0.000086643
    lr_w2_final: float = 0.001

    lr_b2: float = 0.005
    lr_b2_exp_decay_coef: float = -0.000086643
    lr_b2_final: float = 0.001

    lr_w3: float = 0.005
    lr_w3_exp_decay_coef: float = -0.000086643
    lr_w3_final: float = 0.001

    lr_b3: float = 0.005
    lr_b3_exp_decay_coef: float = -0.000086643
    lr_b3_final: float = 0.001

    lr_conditioning: float = 0.005
    lr_conditioning_exp_decay_coef: float = -0.000086643
    lr_conditioning_final: float = 0.005

    lr_features: float = 0.005
    lr_features_exp_decay_coef: float = -0.000086643
    lr_features_final: float = 0.005

    lr_m: float = 0.0025
    lr_m_exp_decay_coef: float = -0.000150000
    lr_m_final: float = 0.000001189614075

    lr_s: float = 0.0396538025
    lr_s_exp_decay_coef: float = -0.000086643
    lr_s_final: float = 0.0396538025

    lr_q: float = 0.01
    lr_q_exp_decay_coef: float = -0.000086643
    lr_q_final: float = 0.001


@dataclass
class ApplicationConfig(ConfigGroup):
    real_time_preview: bool = False
    preview_resolution_scale: float = 1.0
    preview_frequency: int = 10


@dataclass
class FlaReConfig:
    essential: EssentialConfig
    performance: PerformanceConfig
    learning: LearningConfig
    application: ApplicationConfig

    @classmethod
    def from_namespace(cls, namespace):
        return cls(
            essential=EssentialConfig.from_namespace(namespace),
            performance=PerformanceConfig.from_namespace(namespace),
            learning=LearningConfig.from_namespace(namespace),
            application=ApplicationConfig.from_namespace(namespace),
        )
