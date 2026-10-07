# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resume hooks for RSL-RL training launched through the upstream ``isaaclab train`` entry point.

Training in this repository runs through Isaac Lab's own RSL-RL entry point, which this repository
cannot modify. That entry point calls :meth:`rsl_rl.runners.OnPolicyRunner.load` on resume and the runner
calls its own :meth:`~rsl_rl.runners.OnPolicyRunner.save` for every checkpoint. This module wraps both
methods once per process so that:

* every checkpoint is paired with environment-owned training state that RSL-RL checkpoints do not carry,
  by calling ``env.unwrapped.save_training_state(path)`` after the checkpoint is written (for example the
  domain-randomization curriculum level).

and a resumed run:

* keeps the learning rate stored in the checkpoint. ``load`` restores the optimizer, including a learning
  rate the adaptive schedule has decayed, but PPO's own ``learning_rate`` restarts at the config value and
  the adaptive schedule writes it back into the optimizer after the first KL check. A run resumed after its
  rate had decayed therefore took its first updates at the initial rate and collapsed.
* continues at the iteration after the checkpoint. RSL-RL stores the zero-based iteration it just completed
  but restores that value as the next loop index, so the checkpoint's rollout would be repeated and later
  checkpoint numbers would lag the true iteration count.
* restores the environment-owned training state paired with that checkpoint, by calling
  ``env.unwrapped.load_training_state(path)``.

Environments without the hooks are unaffected apart from the learning-rate and iteration fixes.
"""

from __future__ import annotations


def install(runner_cls: type | None = None) -> None:
    """Wrap the runner's ``save`` and ``load`` methods. Idempotent; no-op without RSL-RL.

    Args:
        runner_cls: Runner class to patch. Defaults to :class:`rsl_rl.runners.OnPolicyRunner`.
    """
    if runner_cls is None:
        try:
            from rsl_rl.runners import OnPolicyRunner as runner_cls
        except ImportError:
            return
    OnPolicyRunner = runner_cls
    if getattr(OnPolicyRunner, "_isaaclab_training_resume_hooks", False):
        return

    runner_load = OnPolicyRunner.load
    runner_save = OnPolicyRunner.save

    def _unwrapped_env(runner):
        return getattr(runner.env, "unwrapped", runner.env)

    def save(self, path: str, *args, **kwargs):
        runner_save(self, path, *args, **kwargs)
        save_training_state = getattr(_unwrapped_env(self), "save_training_state", None)
        if callable(save_training_state):
            save_training_state(path)

    def load(self, path: str, *args, **kwargs):
        infos = runner_load(self, path, *args, **kwargs)
        completed_iteration = int(self.current_learning_iteration)
        self.current_learning_iteration = completed_iteration + 1
        print(
            f"[INFO]: Checkpoint completed learning iteration {completed_iteration}; "
            f"resuming at {self.current_learning_iteration}."
        )
        alg = getattr(self, "alg", None)
        optimizer = getattr(alg, "optimizer", None)
        if optimizer is not None and hasattr(alg, "learning_rate"):
            alg.learning_rate = optimizer.param_groups[0]["lr"]
            print(f"[INFO]: Resuming with learning rate {alg.learning_rate:.3g} from the checkpoint.")
        load_training_state = getattr(_unwrapped_env(self), "load_training_state", None)
        if callable(load_training_state):
            load_training_state(path)
        return infos

    OnPolicyRunner.save = save
    OnPolicyRunner.load = load
    OnPolicyRunner._isaaclab_training_resume_hooks = True
