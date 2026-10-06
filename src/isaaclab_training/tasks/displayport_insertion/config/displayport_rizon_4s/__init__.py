# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flexiv Rizon 4s DisplayPort insertion task registrations."""

from __future__ import annotations

import gymnasium as gym

from . import agents

_INSERTION_ENV_ENTRY = "isaaclab_training.tasks.displayport_insertion.insertion_env:DisplayportInsertionEnv"
_AGENT = f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg"
_NEWTON_AGENT = f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionNewtonRNNPPORunnerCfg"


def _register(task_id: str, env_cfg_entry_point: str, agent_cfg_entry_point: str = _AGENT) -> None:
    """Register one Rizon4s DisplayPort task."""
    gym.register(
        id=task_id,
        entry_point=_INSERTION_ENV_ENTRY,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg_entry_point,
            "rsl_rl_cfg_entry_point": agent_cfg_entry_point,
            "default_agent": "rsl_rl",
        },
    )


_TASKS = {
    "IsaacTraining-DisplayPortInsertion-Rizon4s-Joint": (
        f"{__name__}.joint_pos_env_cfg:Rizon4sGravDisplayportInsertionEnvCfg"
    ),
    "IsaacTraining-DisplayPortInsertion-Rizon4s-Joint-Play": (
        f"{__name__}.joint_pos_env_cfg:Rizon4sGravDisplayportInsertionEnvCfg_PLAY"
    ),
    "IsaacTraining-DisplayPortInsertion-Rizon4s-Joint-NoJointVel": (
        f"{__name__}.joint_pos_env_cfg:Rizon4sGravDisplayportInsertionNoJointVelEnvCfg"
    ),
    "IsaacTraining-DisplayPortInsertion-Rizon4s-Joint-NoJointVel-Play": (
        f"{__name__}.joint_pos_env_cfg:Rizon4sGravDisplayportInsertionNoJointVelEnvCfg_PLAY"
    ),
    "IsaacTraining-DisplayPortInsertion-Rizon4s-Joint-NoJointVel-ROS-Inference": (
        f"{__name__}.ros_inference_env_cfg:Rizon4sGravDisplayportInsertionNoJointVelROSInferenceEnvCfg"
    ),
    "IsaacTraining-DisplayPortInsertion-Rizon4s-Joint-ROS-Inference": (
        f"{__name__}.ros_inference_env_cfg:Rizon4sGravDisplayportInsertionROSInferenceEnvCfg"
    ),
    "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace": (
        f"{__name__}.task_space_env_cfg:Rizon4sTaskSpaceDisplayportInsertionEnvCfg"
    ),
    "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Play": (
        f"{__name__}.task_space_env_cfg:Rizon4sTaskSpaceDisplayportInsertionEnvCfg_PLAY"
    ),
    "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-ROS-Inference": (
        f"{__name__}.task_space_ros_inference_env_cfg:Rizon4sTaskSpaceDisplayportInsertionROSInferenceEnvCfg"
    ),
}

_NEWTON_TASKS = {
    "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton": (
        f"{__name__}.task_space_newton_env_cfg:Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg"
    ),
    "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton-Play": (
        f"{__name__}.task_space_newton_env_cfg:Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg_PLAY"
    ),
    "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton-ROS-Inference": (
        f"{__name__}.task_space_newton_ros_inference_env_cfg:"
        "Rizon4sTaskSpaceNewtonDisplayportInsertionROSInferenceEnvCfg"
    ),
}

_ALIASES = {
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav": ("IsaacTraining-DisplayPortInsertion-Rizon4s-Joint"),
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Play": (
        "IsaacTraining-DisplayPortInsertion-Rizon4s-Joint-Play"
    ),
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel": (
        "IsaacTraining-DisplayPortInsertion-Rizon4s-Joint-NoJointVel"
    ),
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-Play": (
        "IsaacTraining-DisplayPortInsertion-Rizon4s-Joint-NoJointVel-Play"
    ),
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-ROS-Inference": (
        "IsaacTraining-DisplayPortInsertion-Rizon4s-Joint-NoJointVel-ROS-Inference"
    ),
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-ROS-Inference": (
        "IsaacTraining-DisplayPortInsertion-Rizon4s-Joint-ROS-Inference"
    ),
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace": (
        "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace"
    ),
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Play": (
        "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Play"
    ),
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-ROS-Inference": (
        "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-ROS-Inference"
    ),
}

_NEWTON_ALIASES = {
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton": (
        "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton"
    ),
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton-Play": (
        "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton-Play"
    ),
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton-ROS-Inference": (
        "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton-ROS-Inference"
    ),
}

for _task_id, _cfg in _TASKS.items():
    _register(_task_id, _cfg)

for _alias, _target in _ALIASES.items():
    _register(_alias, _TASKS[_target])

for _task_id, _cfg in _NEWTON_TASKS.items():
    _register(_task_id, _cfg, _NEWTON_AGENT)

for _alias, _target in _NEWTON_ALIASES.items():
    _register(_alias, _NEWTON_TASKS[_target], _NEWTON_AGENT)
