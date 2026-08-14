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

from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import fields
import json
import sys
import os

from .config import (
    ApplicationConfig,
    EssentialConfig,
    FlaReConfig,
    LearningConfig,
    PerformanceConfig,
)

class ParamGroup:
    config_type = None

    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        defaults = self.config_type()
        for field in fields(defaults):
            key = field.name
            value = getattr(defaults, key)
            setattr(self, key, value)
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
        return self.config_type.from_namespace(args)


class EssentialParams(ParamGroup):
    config_type = EssentialConfig

    def __init__(self, parser):
        super().__init__(parser, "Essential parameters")


class PerformanceParams(ParamGroup):
    config_type = PerformanceConfig

    def __init__(self, parser):
        super().__init__(parser, "Performance parameters")


class LearningParams(ParamGroup):
    config_type = LearningConfig

    def __init__(self, parser):
        super().__init__(parser, "Learning parameters")


class ApplicationParams(ParamGroup):
    config_type = ApplicationConfig

    def __init__(self, parser):
        super().__init__(parser, "Application parameters")


CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")

def _cli_provided_dests(parser: ArgumentParser, argv):
    provided = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if opt in argv:
                provided.add(action.dest)
                break
            prefix = opt + "="
            if any(a.startswith(prefix) for a in argv):
                provided.add(action.dest)
                break
    return provided

def resolve_config_path(config_arg: str):
    if not config_arg:
        return None
    if os.path.isfile(config_arg):
        return os.path.abspath(config_arg)
    name = config_arg
    if not name.endswith(".json"):
        name = name + ".json"
    candidate = os.path.join(CONFIGS_DIR, name)
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(
        "Config not found: {!r} (looked for file path and {})".format(config_arg, candidate)
    )

def load_config_overrides(config_arg: str):
    path = resolve_config_path(config_arg)
    if path is None:
        return {}, None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config must be a JSON object: {}".format(path))
    return data, path

def parse_args_with_config(parser: ArgumentParser, argv=None):
    """Merge priority (highest last / wins): .py defaults < --config JSON < explicit CLI flags."""
    if argv is None:
        argv = sys.argv[1:]

    parser.add_argument(
        "--config",
        default="",
        type=str,
        help="Optional JSON config name or path under arguments/configs/ (overrides .py defaults; CLI flags still win).",
    )

    args = parser.parse_args(argv)
    provided = _cli_provided_dests(parser, argv)

    if args.config:
        overrides, path = load_config_overrides(args.config)
        unknown = [k for k in overrides if not hasattr(args, k)]
        if unknown:
            raise ValueError(
                "Unknown keys in config {}: {}".format(path, ", ".join(sorted(unknown)))
            )
        applied = []
        skipped_cli = []
        for key, value in overrides.items():
            if key in provided:
                skipped_cli.append(key)
                continue
            setattr(args, key, value)
            applied.append(key)
        print("Loaded config: {}".format(path))
        if applied:
            print("  applied from config: {}".format(", ".join(sorted(applied))))
        if skipped_cli:
            print("  kept CLI over config: {}".format(", ".join(sorted(skipped_cli))))

    return args
