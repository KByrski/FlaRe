import contextlib
import io
import json
import os
import tempfile
import unittest
from argparse import ArgumentParser

from arguments import (
    ApplicationParams,
    EssentialParams,
    LearningParams,
    PerformanceParams,
    load_config_overrides,
    parse_args_with_config,
    resolve_config_path,
)


ESSENTIAL_DEFAULTS = {
    "source_path": "",
    "model_path": "",
    "start_iter": 0,
    "end_iter": 64_000,
    "warmup_start_iter": 0,
    "warmup_end_iter": 1_000,
    "resolution": 1_000_000,
    "bg_color_R": 0.0,
    "bg_color_G": 0.0,
    "bg_color_B": 0.0,
    "t_near": 0.1,
    "t_far": 1_000.0,
    "images": "images",
    "data_device": "cuda",
}

PERFORMANCE_DEFAULTS = {
    "random_background": False,
    "number_of_sides": 8,
    "border_opacity": 0.003439,
    "initial_conditioning_std": 0.01,
    "initial_opacity": 0.01,
    "initial_k": 1.01,
    "ray_termination_T_threshold_training": 0.0001,
    "ray_termination_T_threshold_inference": 0.01,
    "opacity_threshold_for_Gauss_removal": 1.0 / 28.0,
    "densification_frequency": 100,
    "densification_start_iter": 800,
    "densification_end_iter": 24_000,
    "max_Gaussians_per_model": -1,
    "mu_grad_norm_threshold_for_densification": 0.001,
    "min_s_norm_threshold_for_Gauss_removal": 0.0001,
    "min_s_coef_clipping_threshold": 0.0001,
    "max_s_coef_clipping_threshold": 0.05,
    "reg_depth_lambda": 0.0001,
    "reg_depth_start_iter": 0,
    "reg_normal_lambda": 0.0,
    "reg_normal_start_iter": 7_000,
    "reg_normal_ramp_iters": 3_000,
    "reg_normal_depth_edge_threshold": 0.02,
    "reg_scale_lambda": 0.001,
}

LEARNING_DEFAULTS = {
    "lr_RGB": 0.005,
    "lr_RGB_exp_decay_coef": -0.000086643,
    "lr_RGB_final": 0.001,
    "lr_A": 0.05,
    "lr_A_exp_decay_coef": -0.000086643,
    "lr_A_final": 0.025,
    "lr_k": 0.1,
    "lr_k_exp_decay_coef": -0.000086643,
    "lr_k_final": 0.1,
    "lr_w1_uv": 0.0125,
    "lr_w1_uv_exp_decay_coef": -0.000086643,
    "lr_w1_uv_final": 0.001,
    "lr_w1_v": 0.03,
    "lr_w1_v_exp_decay_coef": -0.000086643,
    "lr_w1_v_final": 0.001,
    "lr_w1_conditioning": 0.0015,
    "lr_w1_conditioning_exp_decay_coef": -0.000086643,
    "lr_w1_conditioning_final": 0.001,
    "lr_b1": 0.00475,
    "lr_b1_exp_decay_coef": -0.000086643,
    "lr_b1_final": 0.001,
    "lr_w2": 0.005,
    "lr_w2_exp_decay_coef": -0.000086643,
    "lr_w2_final": 0.001,
    "lr_b2": 0.005,
    "lr_b2_exp_decay_coef": -0.000086643,
    "lr_b2_final": 0.001,
    "lr_w3": 0.005,
    "lr_w3_exp_decay_coef": -0.000086643,
    "lr_w3_final": 0.001,
    "lr_b3": 0.005,
    "lr_b3_exp_decay_coef": -0.000086643,
    "lr_b3_final": 0.001,
    "lr_conditioning": 0.005,
    "lr_conditioning_exp_decay_coef": -0.000086643,
    "lr_conditioning_final": 0.005,
    "lr_features": 0.005,
    "lr_features_exp_decay_coef": -0.000086643,
    "lr_features_final": 0.005,
    "lr_m": 0.0025,
    "lr_m_exp_decay_coef": -0.000150000,
    "lr_m_final": 0.000001189614075,
    "lr_s": 0.0396538025,
    "lr_s_exp_decay_coef": -0.000086643,
    "lr_s_final": 0.0396538025,
    "lr_q": 0.01,
    "lr_q_exp_decay_coef": -0.000086643,
    "lr_q_final": 0.001,
}

APPLICATION_DEFAULTS = {
    "real_time_preview": False,
    "preview_resolution_scale": 1.0,
    "preview_frequency": 10,
}


def build_parser():
    parser = ArgumentParser(add_help=False)
    groups = (
        EssentialParams(parser),
        PerformanceParams(parser),
        LearningParams(parser),
        ApplicationParams(parser),
    )
    return parser, groups


class ConfigurationContractTest(unittest.TestCase):
    def test_all_historical_defaults_and_group_boundaries(self):
        parser, groups = build_parser()
        args = parse_args_with_config(parser, [])

        expected = {
            **ESSENTIAL_DEFAULTS,
            **PERFORMANCE_DEFAULTS,
            **LEARNING_DEFAULTS,
            **APPLICATION_DEFAULTS,
            "config": "",
        }
        self.assertEqual(vars(args), expected)

        extracted = [vars(group.extract(args)) for group in groups]
        self.assertEqual(
            extracted,
            [
                ESSENTIAL_DEFAULTS,
                PERFORMANCE_DEFAULTS,
                LEARNING_DEFAULTS,
                APPLICATION_DEFAULTS,
            ],
        )

    def test_boolean_optional_flags_keep_their_cli_spellings(self):
        parser, _ = build_parser()
        args = parse_args_with_config(
            parser,
            ["--random_background", "--real_time_preview"],
        )
        self.assertTrue(args.random_background)
        self.assertTrue(args.real_time_preview)

        parser, _ = build_parser()
        args = parse_args_with_config(
            parser,
            [
                "--random_background",
                "--no-random_background",
                "--real_time_preview",
                "--no-real_time_preview",
            ],
        )
        self.assertFalse(args.random_background)
        self.assertFalse(args.real_time_preview)

    def test_named_config_resolution_and_loading(self):
        path = resolve_config_path("flare_only")
        self.assertTrue(path.endswith(os.path.join("configs", "flare_only.json")))
        values, loaded_path = load_config_overrides("flare_only")
        self.assertEqual(loaded_path, path)
        self.assertEqual(values, {"warmup_start_iter": 0, "warmup_end_iter": 1})

    def test_json_overrides_defaults_but_explicit_cli_wins(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "config.json")
            with open(path, "w", encoding="utf-8") as config_file:
                json.dump(
                    {
                        "end_iter": 26_000,
                        "random_background": True,
                        "preview_frequency": 25,
                    },
                    config_file,
                )

            parser, groups = build_parser()
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                args = parse_args_with_config(
                    parser,
                    [
                        "--config",
                        path,
                        "--end_iter=123",
                        "--no-random_background",
                    ],
                )

        self.assertEqual(args.end_iter, 123)
        self.assertFalse(args.random_background)
        self.assertEqual(args.preview_frequency, 25)
        self.assertIn("applied from config: preview_frequency", stdout.getvalue())
        self.assertIn(
            "kept CLI over config: end_iter, random_background",
            stdout.getvalue(),
        )
        self.assertEqual(groups[0].extract(args).end_iter, 123)
        self.assertFalse(groups[1].extract(args).random_background)
        self.assertEqual(groups[3].extract(args).preview_frequency, 25)

    def test_json_values_keep_existing_no_coercion_behavior(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "config.json")
            with open(path, "w", encoding="utf-8") as config_file:
                json.dump({"end_iter": "26000"}, config_file)

            parser, _ = build_parser()
            with contextlib.redirect_stdout(io.StringIO()):
                args = parse_args_with_config(parser, ["--config", path])

        self.assertEqual(args.end_iter, "26000")

    def test_unknown_config_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "config.json")
            with open(path, "w", encoding="utf-8") as config_file:
                json.dump({"not_a_fla_re_option": 1}, config_file)

            parser, _ = build_parser()
            with self.assertRaisesRegex(ValueError, "not_a_fla_re_option"):
                parse_args_with_config(parser, ["--config", path])

    def test_invalid_config_sources_fail_explicitly(self):
        with self.assertRaisesRegex(FileNotFoundError, "Config not found"):
            resolve_config_path("definitely_missing_config")

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "config.json")
            with open(path, "w", encoding="utf-8") as config_file:
                json.dump(["not", "an", "object"], config_file)
            with self.assertRaisesRegex(ValueError, "Config must be a JSON object"):
                load_config_overrides(path)


if __name__ == "__main__":
    unittest.main()
