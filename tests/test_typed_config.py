import unittest
from argparse import ArgumentParser
from dataclasses import asdict, is_dataclass

from arguments import (
    ApplicationConfig,
    ApplicationParams,
    EssentialConfig,
    EssentialParams,
    FlaReConfig,
    LearningConfig,
    LearningParams,
    PerformanceConfig,
    PerformanceParams,
    parse_args_with_config,
)


class TypedConfigurationTest(unittest.TestCase):
    def test_param_adapters_extract_their_declared_dataclasses(self):
        parser = ArgumentParser(add_help=False)
        adapters_and_types = (
            (EssentialParams(parser), EssentialConfig),
            (PerformanceParams(parser), PerformanceConfig),
            (LearningParams(parser), LearningConfig),
            (ApplicationParams(parser), ApplicationConfig),
        )
        args = parse_args_with_config(parser, [])

        for adapter, config_type in adapters_and_types:
            with self.subTest(config_type=config_type.__name__):
                config = adapter.extract(args)
                self.assertIsInstance(config, config_type)
                self.assertTrue(is_dataclass(config))
                self.assertEqual(vars(adapter), asdict(config_type()))

    def test_complete_config_can_be_built_from_the_compatible_namespace(self):
        parser = ArgumentParser(add_help=False)
        EssentialParams(parser)
        PerformanceParams(parser)
        LearningParams(parser)
        ApplicationParams(parser)
        args = parse_args_with_config(
            parser,
            ["--end_iter", "321", "--random_background", "--lr_m", "0.125"],
        )

        config = FlaReConfig.from_namespace(args)

        self.assertIsInstance(config.essential, EssentialConfig)
        self.assertIsInstance(config.performance, PerformanceConfig)
        self.assertIsInstance(config.learning, LearningConfig)
        self.assertIsInstance(config.application, ApplicationConfig)
        self.assertEqual(config.essential.end_iter, 321)
        self.assertTrue(config.performance.random_background)
        self.assertEqual(config.learning.lr_m, 0.125)


if __name__ == "__main__":
    unittest.main()
