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

from argparse import ArgumentParser, BooleanOptionalAction, Namespace
import json
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action=BooleanOptionalAction)
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action=BooleanOptionalAction)
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

"""class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        super().__init__(parser, "Optimization Parameters")"""
        
# ### ### ### ### ###
# added (start)     #
# ### ### ### ### ###

class EssentialParams(ParamGroup):
    def __init__(self, parser):
        self.source_path = "";
        self.model_path = "";
        self.start_iter = 0;
        self.end_iter = 64000;
        # Paper: endpoints of the linear base-to-FlaRe warmup schedule.
        self.warmup_start_iter = 0;
        self.warmup_end_iter = 1000;
        self.resolution = 1_000_000;
        self.bg_color_R = 0.0;
        self.bg_color_G = 0.0;
        self.bg_color_B = 0.0;
        self.t_near = 0.1;
        self.t_far = 1000.0;
        self.images = "images";
        self.data_device = "cuda";
        
        super().__init__(parser, "Essential parameters");
        
# ### ### ### ### ###

class PerformanceParams(ParamGroup):
    def __init__(self, parser):
        # Random-background transparency carving is applied only when the
        # training images contain a non-opaque alpha channel.
        self.random_background = False;
        self.number_of_sides = 8;
        self.border_opacity = 0.003439; # to be implemented
        self.initial_conditioning_std = 0.01; # 0.01
        self.initial_opacity = 0.01;
        self.initial_k = 1.01;
        self.ray_termination_T_threshold_training = 0.0001; # 0.0001
        self.ray_termination_T_threshold_inference = 0.01;
        self.opacity_threshold_for_Gauss_removal = 1.0 / 28.0; #1.0 / 25.0
        self.densification_frequency = 100;
        self.densification_start_iter = 800; #400
        self.densification_end_iter = 24000; #16000
        # Paper: fixed primitive budgets used by the representation-reduction study.
        self.max_Gaussians_per_model = -1;
        self.mu_grad_norm_threshold_for_densification = 0.001; #0.0005;
        self.min_s_norm_threshold_for_Gauss_removal = 0.0001; #0.00034641;
        self.min_s_coef_clipping_threshold = 0.0001; #0.00034641
        self.max_s_coef_clipping_threshold = 0.05; # 0.025
        # Paper: geometry-aware regularization setting for surface alignment.
        self.reg_depth_lambda = 0.0001; # !!! !!! !!!
        self.reg_depth_start_iter = 0;
        self.reg_normal_lambda = 0.0;
        self.reg_normal_start_iter = 7000;
        self.reg_normal_ramp_iters = 3000;
        self.reg_normal_depth_edge_threshold = 0.02;
        self.reg_scale_lambda = 0.001;
        
        super().__init__(parser, "Performance parameters");
        
# ### ### ### ### ###

class LearningParams(ParamGroup):
    def __init__(self, parser):
        self.lr_RGB = 0.005;
        self.lr_RGB_exp_decay_coef = -0.000086643;
        self.lr_RGB_final = 0.001;
    
        self.lr_A = 0.05;
        self.lr_A_exp_decay_coef = -0.000086643;
        self.lr_A_final = 0.025;
        
        self.lr_k = 0.1;
        self.lr_k_exp_decay_coef = -0.000086643;
        self.lr_k_final = 0.1;

        self.lr_w1_uv = 0.0125;
        self.lr_w1_uv_exp_decay_coef = -0.000086643;
        self.lr_w1_uv_final = 0.001;

        self.lr_w1_v = 0.03;
        self.lr_w1_v_exp_decay_coef = -0.000086643;
        self.lr_w1_v_final = 0.001;

        self.lr_w1_conditioning = 0.0015;
        self.lr_w1_conditioning_exp_decay_coef = -0.000086643;
        self.lr_w1_conditioning_final = 0.001;

        self.lr_b1 = 0.00475;
        self.lr_b1_exp_decay_coef = -0.000086643;
        self.lr_b1_final = 0.001;

        self.lr_w2 = 0.005;
        self.lr_w2_exp_decay_coef = -0.000086643;
        self.lr_w2_final = 0.001;

        self.lr_b2 = 0.005;
        self.lr_b2_exp_decay_coef = -0.000086643;
        self.lr_b2_final = 0.001;

        self.lr_w3 = 0.005;
        self.lr_w3_exp_decay_coef = -0.000086643;
        self.lr_w3_final = 0.001;

        self.lr_b3 = 0.005;
        self.lr_b3_exp_decay_coef = -0.000086643;
        self.lr_b3_final = 0.001;

        self.lr_conditioning = 0.005;
        self.lr_conditioning_exp_decay_coef = -0.000086643;
        self.lr_conditioning_final = 0.005;

        self.lr_features = 0.005;
        self.lr_features_exp_decay_coef = -0.000086643;
        self.lr_features_final = 0.005;

        self.lr_m = 0.0025; #0.001189614075; !!!
        self.lr_m_exp_decay_coef = -0.000150000; #-0.000086643; !!!
        self.lr_m_final = 0.000001189614075;

        self.lr_s = 0.0396538025;
        self.lr_s_exp_decay_coef = -0.000086643;
        self.lr_s_final = 0.0396538025;

        self.lr_q = 0.01;
        self.lr_q_exp_decay_coef = -0.000086643;
        self.lr_q_final = 0.001;
        
        super().__init__(parser, "Learning parameters");
        
# ### ### ### ### ###

class ApplicationParams(ParamGroup):
    def __init__(self, parser):
        self.real_time_preview = False;
        self.preview_resolution_scale = 1.0;
        self.preview_frequency = 10;
        
        super().__init__(parser, "Application parameters");
        
# ### ### ### ### ###
# added (end)       #
# ### ### ### ### ###

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs");

def _cli_provided_dests(parser: ArgumentParser, argv):
    provided = set();
    for action in parser._actions:
        for opt in action.option_strings:
            if opt in argv:
                provided.add(action.dest);
                break;
            prefix = opt + "=";
            if any(a.startswith(prefix) for a in argv):
                provided.add(action.dest);
                break;
    return provided;

def resolve_config_path(config_arg: str):
    if not config_arg:
        return None;
    if os.path.isfile(config_arg):
        return os.path.abspath(config_arg);
    name = config_arg;
    if not name.endswith(".json"):
        name = name + ".json";
    candidate = os.path.join(CONFIGS_DIR, name);
    if os.path.isfile(candidate):
        return candidate;
    raise FileNotFoundError(
        "Config not found: {!r} (looked for file path and {})".format(config_arg, candidate)
    );

def load_config_overrides(config_arg: str):
    path = resolve_config_path(config_arg);
    if path is None:
        return {}, None;
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f);
    if not isinstance(data, dict):
        raise ValueError("Config must be a JSON object: {}".format(path));
    return data, path;

def parse_args_with_config(parser: ArgumentParser, argv=None):
    """Merge priority (highest last / wins): .py defaults < --config JSON < explicit CLI flags."""
    if argv is None:
        argv = sys.argv[1:];

    parser.add_argument(
        "--config",
        default="",
        type=str,
        help="Optional JSON config name or path under arguments/configs/ (overrides .py defaults; CLI flags still win).",
    );

    args = parser.parse_args(argv);
    provided = _cli_provided_dests(parser, argv);

    if args.config:
        overrides, path = load_config_overrides(args.config);
        unknown = [k for k in overrides if not hasattr(args, k)];
        if unknown:
            raise ValueError(
                "Unknown keys in config {}: {}".format(path, ", ".join(sorted(unknown)))
            );
        applied = [];
        skipped_cli = [];
        for key, value in overrides.items():
            if key in provided:
                skipped_cli.append(key);
                continue;
            setattr(args, key, value);
            applied.append(key);
        print("Loaded config: {}".format(path));
        if applied:
            print("  applied from config: {}".format(", ".join(sorted(applied))));
        if skipped_cli:
            print("  kept CLI over config: {}".format(", ".join(sorted(skipped_cli))));

    return args;

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
