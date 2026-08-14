from __future__ import annotations

import unittest

import numpy as np
import torch

from optimizer_state import (
    DensificationConfig,
    DensificationSnapshot,
    apply_fla_re_densification,
    build_fla_re_optimizer,
    capture_densification_snapshot,
    named_optimizer_groups,
)


PER_PRIMITIVE_GROUPS = (0, 1, 2, 11, 13, 14, 15)


class LegacyDensificationRegressionTest(unittest.TestCase):
    """Freeze prune/split tensor order and Adam-state migration."""

    def _fixture(self) -> dict[str, object]:
        count = 4
        values = {
            "RGB": torch.nn.Parameter(
                torch.arange(count * 3, dtype=torch.float32).reshape(count, 3)
                / 10.0
                + 0.1
            ),
            "A": torch.nn.Parameter(
                torch.tensor([[0.0], [0.2], [-20.0], [0.4]])
            ),
            "k": torch.nn.Parameter(torch.zeros(count, 1)),
            "w1_uv": torch.nn.Parameter(torch.full((64, 8), 0.01)),
            "w1_v": torch.nn.Parameter(torch.full((64, 24), 0.02)),
            "w1_conditioning": torch.nn.Parameter(torch.full((64, 96), 0.03)),
            "b1": torch.nn.Parameter(torch.full((64,), 0.04)),
            "w2": torch.nn.Parameter(torch.full((64, 64), 0.05)),
            "b2": torch.nn.Parameter(torch.full((64,), 0.06)),
            "w3": torch.nn.Parameter(torch.full((16, 64), 0.07)),
            "b3": torch.nn.Parameter(torch.full((16,), 0.08)),
            "conditioning_variable": torch.nn.Parameter(
                torch.arange(count * 96, dtype=torch.float32).reshape(count, 96)
                / 1000.0
            ),
            "features": torch.nn.Parameter(torch.full((13154,), 0.09)),
            "m": torch.nn.Parameter(
                torch.tensor(
                    [
                        [0.0, 0.0, 1.0],
                        [1.0, 0.0, 1.0],
                        [2.0, 0.0, 1.0],
                        [3.0, 0.0, 1.0],
                    ]
                )
            ),
            "s": torch.nn.Parameter(torch.full((count, 2), np.log(0.5))),
            "q": torch.nn.Parameter(
                torch.tensor([[1.0, 0.0, 0.0, 0.0]] * count)
            ),
        }
        from scene.gaussian_model import GaussianModel

        model = GaussianModel.__new__(GaussianModel)
        torch.nn.Module.__init__(model)
        for name in GaussianModel.PARAMETER_NAMES:
            model.register_parameter(name, None)
        model.replace_parameters(values)
        values = model.model_tensors()

        ordered_names = (
            "RGB",
            "A",
            "k",
            "w1_uv",
            "w1_v",
            "w1_conditioning",
            "b1",
            "w2",
            "b2",
            "w3",
            "b3",
            "conditioning_variable",
            "features",
            "m",
            "s",
            "q",
        )
        learning_rates = {
            name: 1.0e-3 * (index + 1)
            for index, name in enumerate(ordered_names)
        }
        optimizer = build_fla_re_optimizer(model, learning_rates)
        model.optimizer = optimizer
        for parameter in values.values():
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()

        for group_index, group in enumerate(optimizer.param_groups):
            parameter = group["params"][0]
            state = optimizer.state[parameter]
            row_shape = (parameter.shape[0],) + (1,) * (parameter.ndim - 1)
            rows = torch.arange(parameter.shape[0], dtype=torch.float32).reshape(
                row_shape
            )
            state["exp_avg"].copy_((group_index + 1) * 10.0 + rows)
            state["exp_avg_sq"].copy_((group_index + 1) * 100.0 + rows)

        snapshot = capture_densification_snapshot(
            model, optimizer, first_step=False
        )
        m_before = snapshot.means
        m_state = optimizer.state[values["m"]]
        m_before_exp_avg = snapshot.means_exp_avg
        m_before_exp_avg_sq = snapshot.means_exp_avg_sq
        with torch.no_grad():
            values["m"][1, 0] += 0.2
            values["m"][3, 1] += 0.3
            m_state["exp_avg"].add_(1000.0)
            m_state["exp_avg_sq"].add_(2000.0)

        current_values = {
            name: value.detach().clone() for name, value in values.items()
        }
        current_states = {
            index: {
                key: tensor.clone() if torch.is_tensor(tensor) else tensor
                for key, tensor in optimizer.state[group["params"][0]].items()
            }
            for index, group in enumerate(optimizer.param_groups)
        }
        namespace = {
            "torch": torch,
            "np": np,
            "gaussians": model,
            "optimizer": optimizer,
            "extent": 2.0,
            "m_before": m_before,
            "m_before_exp_avg": m_before_exp_avg,
            "m_before_exp_avg_sq": m_before_exp_avg_sq,
            "learning_rates": learning_rates,
            **values,
        }
        return {
            "namespace": namespace,
            "ordered_names": ordered_names,
            "current_values": current_values,
            "current_states": current_states,
            "m_before": m_before,
            "m_before_exp_avg": m_before_exp_avg,
            "m_before_exp_avg_sq": m_before_exp_avg_sq,
            "learning_rates": learning_rates,
        }

    def _run(self) -> dict[str, object]:
        fixture = self._fixture()
        namespace = fixture["namespace"]
        model = namespace["gaussians"]
        optimizer = apply_fla_re_densification(
            model,
            namespace["optimizer"],
            DensificationSnapshot(
                means=fixture["m_before"],
                means_exp_avg=fixture["m_before_exp_avg"],
                means_exp_avg_sq=fixture["m_before_exp_avg_sq"],
            ),
            DensificationConfig(
                opacity_threshold=0.1,
                minimum_scale_norm=0.0,
                movement_threshold=0.05,
                maximum_gaussians=-1,
                minimum_scale=0.01,
                maximum_scale_fraction=10.0,
            ),
            extent=namespace["extent"],
            learning_rates=fixture["learning_rates"],
        )
        model.optimizer = optimizer
        namespace["optimizer"] = optimizer
        namespace.update(model.model_tensors())
        return fixture

    def test_prune_removes_failed_opacity_primitive(self) -> None:
        fixture = self._run()
        namespace = fixture["namespace"]

        self.assertEqual(tuple(namespace["m"].shape), (5, 3))
        self.assertFalse(
            bool(
                torch.any(
                    torch.all(namespace["m"] == torch.tensor([2.0, 0.0, 1.0]), dim=1)
                )
            )
        )

    def test_moved_primitives_split_into_before_and_after_positions(self) -> None:
        fixture = self._run()
        namespace = fixture["namespace"]
        before = fixture["m_before"]
        current = fixture["current_values"]["m"]
        expected_means = torch.stack(
            (current[0], before[1], before[3], current[1], current[3])
        )
        torch.testing.assert_close(
            namespace["m"], expected_means, rtol=0.0, atol=0.0
        )

        for name in ("RGB", "A", "k", "conditioning_variable", "s", "q"):
            value = fixture["current_values"][name]
            expected = torch.cat((value[0:1], value[[1, 3]], value[[1, 3]]), dim=0)
            torch.testing.assert_close(
                namespace[name], expected, rtol=0.0, atol=0.0
            )

    def test_adam_state_follows_prune_and_split_layout(self) -> None:
        fixture = self._run()
        namespace = fixture["namespace"]
        optimizer = namespace["optimizer"]
        current_states = fixture["current_states"]
        model = namespace["gaussians"]

        for group_index, name in enumerate(fixture["ordered_names"]):
            parameter = optimizer.param_groups[group_index]["params"][0]
            self.assertEqual(optimizer.param_groups[group_index]["name"], name)
            self.assertIs(getattr(model, name), namespace[name])
            self.assertIs(getattr(model, name), parameter)

        for group_index in PER_PRIMITIVE_GROUPS:
            parameter = optimizer.param_groups[group_index]["params"][0]
            state = optimizer.state[parameter]
            self.assertEqual(tuple(parameter.shape), tuple(state["exp_avg"].shape))
            self.assertEqual(
                tuple(parameter.shape), tuple(state["exp_avg_sq"].shape)
            )
            torch.testing.assert_close(
                state["step"], current_states[group_index]["step"],
                rtol=0.0, atol=0.0,
            )
            current = current_states[group_index]
            if group_index == 13:
                expected_avg = torch.cat(
                    (
                        current["exp_avg"][0:1],
                        fixture["m_before_exp_avg"][[1, 3]],
                        current["exp_avg"][[1, 3]],
                    ),
                    dim=0,
                )
                expected_avg_sq = torch.cat(
                    (
                        current["exp_avg_sq"][0:1],
                        fixture["m_before_exp_avg_sq"][[1, 3]],
                        current["exp_avg_sq"][[1, 3]],
                    ),
                    dim=0,
                )
            else:
                expected_avg = torch.cat(
                    (
                        current["exp_avg"][0:1],
                        current["exp_avg"][[1, 3]],
                        current["exp_avg"][[1, 3]],
                    ),
                    dim=0,
                )
                expected_avg_sq = torch.cat(
                    (
                        current["exp_avg_sq"][0:1],
                        current["exp_avg_sq"][[1, 3]],
                        current["exp_avg_sq"][[1, 3]],
                    ),
                    dim=0,
                )
            torch.testing.assert_close(
                state["exp_avg"], expected_avg, rtol=0.0, atol=0.0
            )
            torch.testing.assert_close(
                state["exp_avg_sq"], expected_avg_sq, rtol=0.0, atol=0.0
            )

        for group_index in (3, 4, 5, 6, 7, 8, 9, 10, 12):
            parameter = optimizer.param_groups[group_index]["params"][0]
            state = optimizer.state[parameter]
            torch.testing.assert_close(
                state["step"], current_states[group_index]["step"],
                rtol=0.0, atol=0.0,
            )
            torch.testing.assert_close(
                state["exp_avg"],
                current_states[group_index]["exp_avg"],
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                state["exp_avg_sq"],
                current_states[group_index]["exp_avg_sq"],
                rtol=0.0,
                atol=0.0,
            )

    def test_legacy_unnamed_optimizer_state_recovers_names_and_order(self) -> None:
        fixture = self._fixture()
        namespace = fixture["namespace"]
        model = namespace["gaussians"]
        legacy_state = namespace["optimizer"].state_dict()
        expected_rates = [
            group["lr"] for group in legacy_state["param_groups"]
        ]
        for group in legacy_state["param_groups"]:
            group.pop("name")

        optimizer = build_fla_re_optimizer(
            model, fixture["learning_rates"]
        )
        optimizer.load_state_dict(legacy_state)
        groups = named_optimizer_groups(optimizer)

        self.assertEqual(tuple(groups), fixture["ordered_names"])
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            expected_rates,
        )
        self.assertEqual(
            tuple(
                group["name"]
                for group in optimizer.state_dict()["param_groups"]
            ),
            fixture["ordered_names"],
        )
        for index, name in enumerate(fixture["ordered_names"]):
            parameter = optimizer.param_groups[index]["params"][0]
            state = optimizer.state[parameter]
            expected = fixture["current_states"][index]
            for state_name in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    state[state_name], expected[state_name],
                    rtol=0.0, atol=0.0,
                )


if __name__ == "__main__":
    unittest.main()
