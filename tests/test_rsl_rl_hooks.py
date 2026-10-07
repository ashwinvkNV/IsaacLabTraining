# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the RSL-RL resume hooks installed by the DisplayPort environment."""

from types import SimpleNamespace

from isaaclab_training.utils import rsl_rl_hooks


def _runner_class(calls: list, checkpoint_iteration: int = 100):
    """A stand-in for OnPolicyRunner that records the original save and load calls."""

    class Runner:
        def __init__(self, env):
            self.env = env
            self.current_learning_iteration = 0
            optimizer = SimpleNamespace(param_groups=[{"lr": 2.5e-5}])
            self.alg = SimpleNamespace(optimizer=optimizer, learning_rate=5e-4)

        def save(self, path, infos=None):
            calls.append(("model_save", path, infos))

        def load(self, path, load_cfg=None, strict=True, map_location=None):
            calls.append(("model_load", path))
            self.current_learning_iteration = checkpoint_iteration
            return {}

    rsl_rl_hooks.install(Runner)
    return Runner


def _env(calls: list):
    return SimpleNamespace(
        unwrapped=SimpleNamespace(
            save_training_state=lambda path: calls.append(("env_save", path)),
            load_training_state=lambda path: calls.append(("env_load", path)),
        )
    )


def test_checkpoint_save_is_followed_by_environment_state_save():
    """Every runner checkpoint is followed by its matching environment state save."""
    calls = []
    runner = _runner_class(calls)(_env(calls))
    runner.save("model_100.pt", infos={"value": 1})
    assert calls == [("model_save", "model_100.pt", {"value": 1}), ("env_save", "model_100.pt")]


def test_resume_continues_after_checkpoint_with_its_learning_rate_and_state():
    """A checkpoint from iteration N resumes at N+1, keeps its learning rate and restores env state."""
    calls = []
    runner = _runner_class(calls, checkpoint_iteration=100)(_env(calls))
    runner.load("model_100.pt")
    assert runner.current_learning_iteration == 101
    assert runner.alg.learning_rate == 2.5e-5
    assert calls == [("model_load", "model_100.pt"), ("env_load", "model_100.pt")]


def test_install_is_idempotent_and_hooks_are_optional():
    """Installing twice wraps once, and an environment without the hooks is left alone."""
    calls = []
    runner_cls = _runner_class(calls)
    rsl_rl_hooks.install(runner_cls)
    runner = runner_cls(SimpleNamespace(unwrapped=SimpleNamespace()))
    runner.save("model_0.pt")
    runner.load("model_100.pt")
    assert calls == [("model_save", "model_0.pt", None), ("model_load", "model_100.pt")]
