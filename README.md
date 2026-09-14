# IsaacLab Training

Reusable downstream Isaac Lab training repository for custom task formulations, shared MDP terms, and policy export workflows.

This repo owns reusable MDP terms, task registrations, policy I/O tooling, and LEAPP export contracts outside the core Isaac Lab tree. DisplayPort insertion is the first concrete task packaged here.

## Setup

Python 3.12 and `uv` are expected.

```bash
uv sync
uv run pytest -q
uv run ruff check .
```

Confirm Isaac Lab discovers the downstream tasks:

```bash
uv run python -c 'import gymnasium as gym; import isaaclab_training.tasks; print([s.id for s in gym.registry.values() if s.id.startswith("IsaacTraining-")])'
```

## Concepts

`isaaclab_training.mdp` contains reusable MDP helpers for deployable manipulation tasks:

- deploy-aware joint, OSC, and DiffIK action wrappers
- rigid-object and end-effector observation terms
- two-body keypoint rewards
- single-object grasp reset and plug/curriculum reset events
- plug/object drop and orientation termination helpers
- reset-sampled noise models

`isaaclab_training.export` contains generic LEAPP export infrastructure. Task-specific output semantics live behind named contracts. The first contract is:

```text
displayport_task_space
```

## DisplayPort Task

Preferred task IDs use the repo-owned `IsaacTraining-*` namespace:

```text
IsaacTraining-DisplayPortInsertion-Rizon4s-Joint
IsaacTraining-DisplayPortInsertion-Rizon4s-Joint-Play
IsaacTraining-DisplayPortInsertion-Rizon4s-Joint-NoJointVel
IsaacTraining-DisplayPortInsertion-Rizon4s-Joint-NoJointVel-Play
IsaacTraining-DisplayPortInsertion-Rizon4s-Joint-NoJointVel-ROS-Inference
IsaacTraining-DisplayPortInsertion-Rizon4s-Joint-ROS-Inference
IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace
IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Play
IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-ROS-Inference
```

The original `IsaacContrib-Deploy-...` IDs are also registered as compatibility aliases.

Task space is the recommended DisplayPort deployment path:

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-ROS-Inference \
  --num_envs 4096 --max_iterations 1500 --seed 42 \
  --visualizer newton
```

## Export

Use the generic exporter with a named contract:

```bash
uv run --group leapp isaaclab-training-export \
  --task IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-ROS-Inference \
  --checkpoint logs/rsl_rl/displayport_insertion_rizon4s/<run>/model_<n>.pt \
  --contract displayport_task_space
```

The exporter uses packaged Isaac Lab APIs for app launch, Hydra task resolution, RSL-RL checkpoint loading, LEAPP environment patching, and graph compilation. It does not import Isaac Lab source-tree `scripts/` files.

## Layout

```text
src/isaaclab_training/
  mdp/                           reusable MDP/action/observation/reward/event terms
  export/                        generic LEAPP export contracts
  tasks/displayport_insertion/   first concrete task formulation
  cli/                           generic export and policy I/O commands
  utils/
docs/
tests/
```

The long-form DisplayPort tutorial is kept in `docs/displayport_insertion_policy.rst`.
