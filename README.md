# IsaacLab Training

Reusable downstream Isaac Lab training repository for custom task formulations, shared MDP terms, and policy export workflows.

This repo owns reusable MDP terms, task registrations, policy I/O tooling, and LEAPP export helpers outside the core Isaac Lab tree. DisplayPort insertion is the first concrete task packaged here.

Documentation: https://ashwinvknv.github.io/IsaacLabTraining/

## Setup

Python 3.12 and `uv` are expected. The Isaac Lab dependency is declared in
`pyproject.toml` and pinned in `uv.lock`; `uv sync` installs this repo in editable
mode and resolves the pinned Isaac Lab wheel-builder package. Actual PhysX / Kit
training also requires Isaac Sim, installed through the `sim` dependency group.

```bash
uv sync
```

Optional local development groups:

```bash
uv sync --group dev
uv sync --group docs
uv sync --group leapp
uv sync --group sim
```

Validate the install:

```bash
uv run pytest -q
uv run ruff check .
uv run python -c 'import gymnasium as gym; import isaaclab_training.tasks; print([s.id for s in gym.registry.values() if s.id.startswith("IsaacTraining-")])'
```

Build the docs locally:

```bash
uv run --group docs make -C docs current-docs
```

Run a small visual training smoke test:

```bash
uv sync --group sim

uv run isaaclab train --rl_library rsl_rl \
  --task IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-ROS-Inference \
  --num_envs 4 \
  --max_iterations 100 \
  --visualizer kit
```

For the full install and training workflow, see the hosted documentation:
https://ashwinvknv.github.io/IsaacLabTraining/setup_training.html

## Concepts

`isaaclab_training.mdp` contains reusable MDP helpers for deployable manipulation tasks:

- deploy-aware joint, OSC, and DiffIK action wrappers
- rigid-object and end-effector observation terms
- two-body keypoint rewards
- single-object grasp reset and plug/curriculum reset events
- plug/object drop and orientation termination helpers
- reset-sampled noise models

`isaaclab_training.export` contains generic LEAPP export infrastructure. The first task-specific export path is:

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

Use the generic exporter for task-space DisplayPort export:

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
  export/                        generic LEAPP export helpers
  tasks/displayport_insertion/   first concrete task formulation
  cli/                           generic export and policy I/O commands
  utils/
docs/
tests/
```

The long-form DisplayPort tutorial is kept in `docs/tasks/displayport_insertion.rst`.
