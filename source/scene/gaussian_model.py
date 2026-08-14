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

import torch
from checkpoint_io import MODEL_TENSOR_NAMES, load_training_checkpoint
from optimizer_state import (
    build_fla_re_optimizer,
    named_optimizer_groups,
)
import numpy as np
from torch import nn
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud

class GaussianModel(nn.Module):

    PARAMETER_NAMES = MODEL_TENSOR_NAMES


    def __init__(self): #, sh_degree : int):
        super().__init__()
        for name in self.PARAMETER_NAMES:
            self.register_parameter(name, None)


        # Paper: "Global auto-decoder". The first layer consumes 8 LUT features,
        # 24 view-direction features, and the 96D per-primitive descriptor z_i.
        self.w1_uv = nn.Parameter(
            torch.empty(64, 8, dtype=torch.float32, device="cuda"), requires_grad=False)
        self.w1_v = nn.Parameter(
            torch.empty(64, 24, dtype=torch.float32, device="cuda"), requires_grad=False)
        self.w1_conditioning = nn.Parameter(
            torch.empty(64, 96, dtype=torch.float32, device="cuda"), requires_grad=False)
        self.b1 = nn.Parameter(
            torch.zeros(64, dtype=torch.float32, device="cuda"), requires_grad=False)
        self.w2 = nn.Parameter(
            torch.empty(64, 64, dtype=torch.float32, device="cuda"), requires_grad=False)
        self.b2 = nn.Parameter(
            torch.zeros(64, dtype=torch.float32, device="cuda"), requires_grad=False)
        self.w3 = nn.Parameter(
            torch.empty(16, 64, dtype=torch.float32, device="cuda"), requires_grad=False)
        self.b3 = nn.Parameter(
            torch.zeros(16, dtype=torch.float32, device="cuda"), requires_grad=False)
        # Paper: "LUT-Encoding". Four 2D grids with two features per vertex:
        # 2 * (16^2 + 25^2 + 40^2 + 64^2) = 13,154 trainable values.
        self.features = nn.Parameter(
            torch.empty(13154, dtype=torch.float32, device="cuda"), requires_grad=False)

        torch.nn.init.kaiming_normal_(self.w1_uv, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.w1_v, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.w1_conditioning, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.w2, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.w3, mode='fan_in', nonlinearity='relu')
        torch.nn.init.uniform_(self.features, a=-1e-4, b=1e-4)
        self.w1_uv *= 0.1
        self.w1_v *= 0.1
        self.w1_conditioning *= 0.1
        self.w2 *= 0.1
        self.w3 *= 0.1

        self.iteration = 0
        self.training_time = 0.0

    @classmethod
    def from_model_tensors(cls, values, *, requires_grad=False):
        """Construct model ownership without randomized CUDA initialization."""
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        for name in cls.PARAMETER_NAMES:
            model.register_parameter(name, None)
        model.replace_parameters(values, requires_grad=requires_grad)
        model.iteration = 0
        model.training_time = 0.0
        return model


    def _ensure_parameter_registry(self):
        if hasattr(self, "_parameters"):
            return
        nn.Module.__init__(self)
        for name in self.PARAMETER_NAMES:
            self.register_parameter(name, None)

    def replace_parameters(self, values, *, requires_grad=True):
        self._ensure_parameter_registry()
        unknown = set(values) - set(self.PARAMETER_NAMES)
        if unknown:
            raise ValueError("Unknown FlaRe parameters: {}".format(sorted(unknown)))
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError("{} must be a torch.Tensor".format(name))
            setattr(
                self,
                name,
                nn.Parameter(value, requires_grad=requires_grad),
            )
        if "m" in values:
            self.number_of_Gaussians = int(self.m.shape[0])

    def model_tensors(self):
        return {name: getattr(self, name) for name in self.PARAMETER_NAMES}

    @property
    def means(self):
        return self.m

    @property
    def scales(self):
        return torch.exp(self.s)

    @property
    def quaternions(self):
        return self.q

    @property
    def opacities(self):
        return torch.sigmoid(self.A)

    @property
    def kappas(self):
        return 1.0 + torch.nn.functional.softplus(self.k)

    def renderer_geometry(self):
        return (
            self.means,
            self.scales,
            self.quaternions,
            self.opacities,
            self.kappas,
        )


    def create_from_pcd(self, pp, pcd : BasicPointCloud, spatial_lr_scale : float):


        self.number_of_Gaussians = pcd.points.shape[0]
        # Paper: each planar primitive carries a compact 96D local-radiance
        # descriptor z_i, decoded by the single network shared by the scene.
        self.conditioning_variable = nn.Parameter(
            torch.empty(self.number_of_Gaussians, 96, dtype=torch.float32, device="cuda"),
            requires_grad=False,
        )
        torch.nn.init.normal_(self.conditioning_variable, mean=0.0, std=pp.initial_conditioning_std)
        # Paper: trainable primitive parameters c_const, alpha_const, kappa,
        # center mu, two tangent-plane log-scales, and rotation quaternion.
        self.RGB = nn.Parameter(
            torch.tensor(np.asarray(pcd.colors)).float().cuda(), requires_grad=False)
        self.A = nn.Parameter(
            torch.logit(torch.full((self.number_of_Gaussians, 1), pp.initial_opacity, dtype=torch.float32, device="cuda")),
            requires_grad=False,
        )
        initial_k = torch.full(
            (self.number_of_Gaussians, 1), pp.initial_k,
            dtype=torch.float32, device="cuda")
        # Store raw kappa so the renderer can enforce kappa = 1 + softplus(k).
        initial_k = (initial_k - 1.0) + torch.log(-torch.expm1(-initial_k + 1.0))  # inverse softplus
        self.k = nn.Parameter(initial_k, requires_grad=False)

        self.m = nn.Parameter(
            torch.tensor(np.asarray(pcd.points)).float().cuda(), requires_grad=False)
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        self.s = nn.Parameter(
            torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2), requires_grad=False)
        quaternions = torch.randn(
            self.number_of_Gaussians, 4, dtype=torch.float32, device="cuda")
        self.q = nn.Parameter(
            torch.nn.functional.normalize(quaternions, dim=1), requires_grad=False)

    def training_setup(self, training_args):


        self.w1_uv.requires_grad_(True)
        self.w1_v.requires_grad_(True)
        self.w1_conditioning.requires_grad_(True)
        self.b1.requires_grad_(True)
        self.w2.requires_grad_(True)
        self.b2.requires_grad_(True)
        self.w3.requires_grad_(True)
        self.b3.requires_grad_(True)
        self.features.requires_grad_(True)
        self.conditioning_variable.requires_grad_(True)
        self.RGB.requires_grad_(True)
        self.A.requires_grad_(True)
        self.k.requires_grad_(True)
        self.m.requires_grad_(True)
        self.s.requires_grad_(True)
        self.q.requires_grad_(True)

        if (training_args.lr_RGB_exp_decay_coef <= 0.0):
            lr_RGB_current = float(np.maximum(training_args.lr_RGB * np.exp(training_args.lr_RGB_exp_decay_coef * self.iteration), training_args.lr_RGB_final))
        else:
            lr_RGB_current = float(np.minimum(training_args.lr_RGB * np.exp(training_args.lr_RGB_exp_decay_coef * self.iteration), training_args.lr_RGB_final))
        if (training_args.lr_A_exp_decay_coef <= 0.0):
            lr_A_current = float(np.maximum(training_args.lr_A * np.exp(training_args.lr_A_exp_decay_coef * self.iteration), training_args.lr_A_final))
        else:
            lr_A_current = float(np.minimum(training_args.lr_A * np.exp(training_args.lr_A_exp_decay_coef * self.iteration), training_args.lr_A_final))
        if (training_args.lr_k_exp_decay_coef <= 0.0):
            lr_k_current = float(np.maximum(training_args.lr_k * np.exp(training_args.lr_k_exp_decay_coef * self.iteration), training_args.lr_k_final))
        else:
            lr_k_current = float(np.minimum(training_args.lr_k * np.exp(training_args.lr_k_exp_decay_coef * self.iteration), training_args.lr_k_final))
        if (training_args.lr_w1_uv_exp_decay_coef <= 0.0):
            lr_w1_uv_current = float(np.maximum(training_args.lr_w1_uv * np.exp(training_args.lr_w1_uv_exp_decay_coef * self.iteration), training_args.lr_w1_uv_final))
        else:
            lr_w1_uv_current = float(np.minimum(training_args.lr_w1_uv * np.exp(training_args.lr_w1_uv_exp_decay_coef * self.iteration), training_args.lr_w1_uv_final))
        if (training_args.lr_w1_v_exp_decay_coef <= 0.0):
            lr_w1_v_current = float(np.maximum(training_args.lr_w1_v * np.exp(training_args.lr_w1_v_exp_decay_coef * self.iteration), training_args.lr_w1_v_final))
        else:
            lr_w1_v_current = float(np.minimum(training_args.lr_w1_v * np.exp(training_args.lr_w1_v_exp_decay_coef * self.iteration), training_args.lr_w1_v_final))
        if (training_args.lr_w1_conditioning_exp_decay_coef <= 0.0):
            lr_w1_conditioning_current = float(np.maximum(training_args.lr_w1_conditioning * np.exp(training_args.lr_w1_conditioning_exp_decay_coef * self.iteration), training_args.lr_w1_conditioning_final))
        else:
            lr_w1_conditioning_current = float(np.minimum(training_args.lr_w1_conditioning * np.exp(training_args.lr_w1_conditioning_exp_decay_coef * self.iteration), training_args.lr_w1_conditioning_final))
        if (training_args.lr_b1_exp_decay_coef <= 0.0):
            lr_b1_current = float(np.maximum(training_args.lr_b1 * np.exp(training_args.lr_b1_exp_decay_coef * self.iteration), training_args.lr_b1_final))
        else:
            lr_b1_current = float(np.minimum(training_args.lr_b1 * np.exp(training_args.lr_b1_exp_decay_coef * self.iteration), training_args.lr_b1_final))
        if (training_args.lr_w2_exp_decay_coef <= 0.0):
            lr_w2_current = float(np.maximum(training_args.lr_w2 * np.exp(training_args.lr_w2_exp_decay_coef * self.iteration), training_args.lr_w2_final))
        else:
            lr_w2_current = float(np.minimum(training_args.lr_w2 * np.exp(training_args.lr_w2_exp_decay_coef * self.iteration), training_args.lr_w2_final))
        if (training_args.lr_b2_exp_decay_coef <= 0.0):
            lr_b2_current = float(np.maximum(training_args.lr_b2 * np.exp(training_args.lr_b2_exp_decay_coef * self.iteration), training_args.lr_b2_final))
        else:
            lr_b2_current = float(np.minimum(training_args.lr_b2 * np.exp(training_args.lr_b2_exp_decay_coef * self.iteration), training_args.lr_b2_final))
        if (training_args.lr_w3_exp_decay_coef <= 0.0):
            lr_w3_current = float(np.maximum(training_args.lr_w3 * np.exp(training_args.lr_w3_exp_decay_coef * self.iteration), training_args.lr_w3_final))
        else:
            lr_w3_current = float(np.minimum(training_args.lr_w3 * np.exp(training_args.lr_w3_exp_decay_coef * self.iteration), training_args.lr_w3_final))
        if (training_args.lr_b3_exp_decay_coef <= 0.0):
            lr_b3_current = float(np.maximum(training_args.lr_b3 * np.exp(training_args.lr_b3_exp_decay_coef * self.iteration), training_args.lr_b3_final))
        else:
            lr_b3_current = float(np.minimum(training_args.lr_b3 * np.exp(training_args.lr_b3_exp_decay_coef * self.iteration), training_args.lr_b3_final))
        if (training_args.lr_conditioning_exp_decay_coef <= 0.0):
            lr_conditioning_current = float(np.maximum(training_args.lr_conditioning * np.exp(training_args.lr_conditioning_exp_decay_coef * self.iteration), training_args.lr_conditioning_final))
        else:
            lr_conditioning_current = float(np.minimum(training_args.lr_conditioning * np.exp(training_args.lr_conditioning_exp_decay_coef * self.iteration), training_args.lr_conditioning_final))
        if (training_args.lr_features_exp_decay_coef <= 0.0):
            lr_features_current = float(np.maximum(training_args.lr_features * np.exp(training_args.lr_features_exp_decay_coef * self.iteration), training_args.lr_features_final))
        else:
            lr_features_current = float(np.minimum(training_args.lr_features * np.exp(training_args.lr_features_exp_decay_coef * self.iteration), training_args.lr_features_final))
        if (training_args.lr_m_exp_decay_coef <= 0.0):
            lr_m_current = float(np.maximum(training_args.lr_m * np.exp(training_args.lr_m_exp_decay_coef * self.iteration), training_args.lr_m_final))
        else:
            lr_m_current = float(np.minimum(training_args.lr_m * np.exp(training_args.lr_m_exp_decay_coef * self.iteration), training_args.lr_m_final))
        if (training_args.lr_s_exp_decay_coef <= 0.0):
            lr_s_current = float(np.maximum(training_args.lr_s * np.exp(training_args.lr_s_exp_decay_coef * self.iteration), training_args.lr_s_final))
        else:
            lr_s_current = float(np.minimum(training_args.lr_s * np.exp(training_args.lr_s_exp_decay_coef * self.iteration), training_args.lr_s_final))
        if (training_args.lr_q_exp_decay_coef <= 0.0):
            lr_q_current = float(np.maximum(training_args.lr_q * np.exp(training_args.lr_q_exp_decay_coef * self.iteration), training_args.lr_q_final))
        else:
            lr_q_current = float(np.minimum(training_args.lr_q * np.exp(training_args.lr_q_exp_decay_coef * self.iteration), training_args.lr_q_final))

        learning_rates = {
            "RGB": lr_RGB_current,
            "A": lr_A_current,
            "k": lr_k_current,
            "w1_uv": lr_w1_uv_current,
            "w1_v": lr_w1_v_current,
            "w1_conditioning": lr_w1_conditioning_current,
            "b1": lr_b1_current,
            "w2": lr_w2_current,
            "b2": lr_b2_current,
            "w3": lr_w3_current,
            "b3": lr_b3_current,
            "conditioning_variable": lr_conditioning_current,
            "features": lr_features_current,
            "m": lr_m_current,
            "s": lr_s_current,
            "q": lr_q_current,
        }
        self.optimizer = build_fla_re_optimizer(self, learning_rates)


    def load_checkpoint(self, lp, path):
        state = load_training_checkpoint(path, "cuda", require_rgb=True)
        if not isinstance(state.optimizer_state, dict):
            raise ValueError("Training checkpoint has no optimizer state")
        if state.iteration < 0:
            raise ValueError("Training checkpoint has no recoverable iteration")
        self.replace_parameters(state.model, requires_grad=False)

        self.iteration = state.iteration
        self.training_setup(lp)

        self.optimizer.load_state_dict(state.optimizer_state)
        if hasattr(self.optimizer, "param_groups"):
            named_optimizer_groups(self.optimizer)
        self.training_time = state.training_time_seconds
