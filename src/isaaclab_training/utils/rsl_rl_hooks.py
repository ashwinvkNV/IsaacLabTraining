# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resume hooks for RSL-RL training launched through the upstream ``isaaclab train`` entry point.

Training in this repository runs through Isaac Lab's own RSL-RL entry point, which this repository
cannot modify. That entry point calls :meth:`rsl_rl.runners.OnPolicyRunner.load` on resume. This module
wraps that method once per process so a resumed run:

* keeps the learning rate stored in the checkpoint. ``load`` restores the optimizer, including a learning
  rate the adaptive schedule has decayed, but PPO's own ``learning_rate`` restarts at the config value and
  the adaptive schedule writes it back into the optimizer after the first KL check. A run resumed after its
  rate had decayed therefore took its first updates at the initial rate and collapsed.
* restores environment-owned training state that RSL-RL checkpoints do not carry, by calling
  ``env.unwrapped.load_training_state(path)`` when the environment defines it (for example the
  domain-randomization curriculum level).

Environments without the hook are unaffected apart from the learning-rate fix.
"""

from __future__ import annotations


def install() -> None:
    """Wrap :meth:`OnPolicyRunner.load` with the resume hooks. Idempotent; no-op without RSL-RL."""
    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError:
        return
    if getattr(OnPolicyRunner, "_isaaclab_training_resume_hooks", False):
        return

    runner_load = OnPolicyRunner.load

    def load(self, path: str, *args, **kwargs):
        infos = runner_load(self, path, *args, **kwargs)
        alg = getattr(self, "alg", None)
        optimizer = getattr(alg, "optimizer", None)
        if optimizer is not None and hasattr(alg, "learning_rate"):
            alg.learning_rate = optimizer.param_groups[0]["lr"]
            print(f"[INFO]: Resuming with learning rate {alg.learning_rate:.3g} from the checkpoint.")
        env = getattr(self.env, "unwrapped", self.env)
        load_training_state = getattr(env, "load_training_state", None)
        if callable(load_training_state):
            load_training_state(path)
        return infos

    OnPolicyRunner.load = load
    OnPolicyRunner._isaaclab_training_resume_hooks = True
