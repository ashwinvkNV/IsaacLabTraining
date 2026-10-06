# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for DisplayPort policy inference telemetry."""

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_task_space_logger_clips_raw_action_before_anisotropic_scale(monkeypatch: pytest.MonkeyPatch):
    """Deploy-facing CSV actions must match the RSL-RL and LEAPP transform."""
    monkeypatch.setattr(sys, "argv", ["play_policy_io"])
    play_policy_io = importlib.import_module("isaaclab_training.cli.play_policy_io")

    position_scale = (0.025, 0.020, 0.010)
    orientation_scale = (0.100, 0.200, 0.300)
    action_cfg = SimpleNamespace(position_scale=position_scale, orientation_scale=orientation_scale)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos_target=torch.zeros((1, 7)),
            joint_names=[f"joint{i}" for i in range(1, 8)],
        )
    )
    base_env = SimpleNamespace(
        cfg=SimpleNamespace(actions=SimpleNamespace(arm_action=action_cfg), num_arm_joints=7),
        scene={"robot": robot},
    )
    env = SimpleNamespace(unwrapped=base_env)
    logger = play_policy_io.InferenceLogger(
        env,
        log_dir=None,
        print_enabled=False,
        print_every=1,
        clip_actions=1.0,
    )
    raw_action = np.array([2.0, -2.0, 0.5, -0.5, 1.5, -1.5])
    row = {f"action_raw_{i}": float(value) for i, value in enumerate(raw_action)}

    logger.end_step(row, success_info=None)

    logged_action = np.array([row[f"action_{i}"] for i in range(6)])
    np.testing.assert_allclose(logged_action, [0.025, -0.020, 0.005, -0.050, 0.200, -0.300])
    np.testing.assert_allclose(
        [row[f"action_raw_{i}"] for i in range(6)],
        raw_action,
    )
