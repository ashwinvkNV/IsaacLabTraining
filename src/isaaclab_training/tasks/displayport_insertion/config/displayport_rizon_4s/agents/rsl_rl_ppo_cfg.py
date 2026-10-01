# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticRecurrentCfg, RslRlPpoAlgorithmCfg


_NUM_STEPS_PER_ENV = 512
_MAX_ITERATIONS = 1500
_SAVE_INTERVAL = 50


@configclass
class Rizon4sGravDisplayportInsertionRNNPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = _NUM_STEPS_PER_ENV
    max_iterations = _MAX_ITERATIONS
    save_interval = _SAVE_INTERVAL
    experiment_name = "displayport_insertion_rizon4s"
    clip_actions = 1.0
    resume = False
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }
    policy = RslRlPpoActorCriticRecurrentCfg(
        state_dependent_std=True,
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        noise_std_type="log",
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=2,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=8,
        num_mini_batches=16,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


# Rollout length for the task-space tasks. 128 steps fits a full insertion (~20 control
# steps for a converged policy) six times, and with 4x shorter rollouts the PPO update pads
# every trajectory fragment to 128 instead of 512 steps (LSTM update memory ~13x lower).
TASK_SPACE_NUM_STEPS_PER_ENV = 128

# Factor by which iteration-based settings are scaled so a run covers the same number of
# environment steps (and, at a fixed num_envs, the same number of samples) as before.
_TASK_SPACE_ITER_SCALE = _NUM_STEPS_PER_ENV // TASK_SPACE_NUM_STEPS_PER_ENV


@configclass
class Rizon4sGravDisplayportInsertionTaskSpaceRNNPPORunnerCfg(Rizon4sGravDisplayportInsertionRNNPPORunnerCfg):
    """Task-space runner: 128-step rollouts with the same sample budget as the 512-step base."""

    num_steps_per_env = TASK_SPACE_NUM_STEPS_PER_ENV
    max_iterations = _MAX_ITERATIONS * _TASK_SPACE_ITER_SCALE
    save_interval = _SAVE_INTERVAL * _TASK_SPACE_ITER_SCALE
