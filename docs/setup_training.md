# Setup and Training

This repository is a downstream Isaac Lab project. It provides task
formulations, reusable MDP terms, and export utilities, while Isaac Lab provides
the simulator integration, unified training entry points, and RSL-RL runner.

## Prerequisites

- Linux workstation or cluster node with an NVIDIA GPU.
- Python 3.12.
- `uv` for Python environment management.
- Git access to this repository and the pinned Isaac Lab dependency.
- Isaac Sim for PhysX / Kit training. In the standalone `uv` workflow below,
  install it with the `sim` dependency group.

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Install the Standalone Repo

Clone the repository and install the default dependencies:

```bash
git clone https://github.com/ashwinvkNV/IsaacLabTraining.git
cd IsaacLabTraining
uv sync
```

The `isaaclab` dependency is declared in `pyproject.toml` and pinned through
`uv.lock`. The default `uv sync` path installs the repo in editable mode and
resolves Isaac Lab from the pinned Isaac Lab wheel-builder source:

```toml
isaaclab = { git = "https://github.com/isaac-sim/IsaacLab.git", subdirectory = "tools/wheel_builder", ... }
```

This base install is enough for package imports, docs, tests, and task
registration checks. PhysX / Kit training additionally requires Isaac Sim:

```bash
uv sync --group sim
```

The `sim` group installs Isaac Sim through NVIDIA's Python package index. The
resolver also carries the small compatibility overrides needed by the pinned
Isaac Lab revision.

Use dependency groups for optional local workflows:

```bash
uv sync --group dev      # tests, ruff, pre-commit, codespell
uv sync --group docs     # Sphinx docs build
uv sync --group leapp    # LEAPP export tooling
uv sync --group sim      # Isaac Sim for PhysX / Kit training
```

## Verify the Install

Confirm the package imports and the downstream Isaac Lab tasks are registered:

```bash
uv run python -c "import isaaclab_training; print('isaaclab_training ok')"

uv run python -c 'import gymnasium as gym; import isaaclab_training.tasks; print([s.id for s in gym.registry.values() if s.id.startswith("IsaacTraining-")])'
```

The second command should list the registered `IsaacTraining-*` task IDs.

## Train a DisplayPort Policy

Install the simulation group before PhysX or Kit training:

```bash
uv sync --group sim
```

Task-space control is the recommended DisplayPort deployment path:

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-ROS-Inference \
  --num_envs 16 \
  --max_iterations 1 \
  --visualizer kit
```

Use this small visual run first to confirm that Isaac Sim launches, the task is
registered, and the robot/plug/socket scene looks correct.

The Newton task is a separate checkpoint ABI. Run a finite headless optimizer smoke with the typed Newton physics
selection before launching a full job:

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton-ROS-Inference \
  --num_envs 16 \
  --max_iterations 1 \
  --seed 123 \
  --visualizer none \
  physics=newton_sdf
```

The recurrent PPO configuration uses 16 minibatches. Keep training smokes at 16 or more environments, or explicitly
set `agent.algorithm.num_mini_batches` no higher than `--num_envs`. Use `isaaclab play` for smaller visual checks.

If this command reports `Isaac Sim is not installed or not found on PYTHONPATH`,
run `uv sync --group sim` or use the Isaac Lab source-checkout workflow below.

For a longer headless training run:

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-ROS-Inference \
  --num_envs 256 \
  --viz none \
  --video --video_length 200 --video_interval 76800
```

For the recommended Newton 1.6 task-space profile, use 256 environments per distributed rank and the calibrated
USD for the robot that will execute the policy:

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton-ROS-Inference \
  --num_envs 256 \
  --seed 123 \
  --visualizer none \
  physics=newton_sdf \
  env.scene.robot.spawn.usd_path=/absolute/path/to/calibrated_rizon4s.usd
```

Verify that the calibrated USD authors valid mass and inertia properties, including for the flange. Treat any Newton
invalid-inertia fallback as an asset error to fix and requalify before full training. Repeat the calibrated-USD
override when playing or exporting this checkpoint. Newton and PhysX task-space
checkpoints are not interchangeable even though both actor inputs contain 18 values.

Joint-space training is also packaged:

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacTraining-DisplayPortInsertion-Rizon4s-Joint-NoJointVel-ROS-Inference \
  --num_envs 256 \
  --viz none
```

Training outputs are written under `logs/rsl_rl/`. Monitor a run with
TensorBoard:

```bash
uv run python -m tensorboard.main --logdir logs/rsl_rl/displayport_insertion_rizon4s
```

## Using an Isaac Lab Source Checkout

If you already work inside an Isaac Lab source checkout, install this repository
into Isaac Lab's Python environment instead of creating a separate `uv` managed
environment:

```bash
cd /path/to/IsaacLab
./isaaclab.sh -p -m pip install -e /path/to/IsaacLabTraining

./isaaclab.sh train --rl_library rsl_rl \
  --task IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-ROS-Inference \
  --num_envs 16 \
  --max_iterations 1 \
  --visualizer kit
```

Use the source-checkout path when you need to test against a local Isaac Lab
branch. Use the standalone `uv sync` path for normal development in this repo.

## Common Overrides

- `--task`: registered Isaac Lab task ID.
- `--rl_library rsl_rl`: selects the RSL-RL backend used by these configs.
- `--num_envs`: number of parallel simulation environments.
- `--max_iterations`: short CLI override for the RSL-RL runner.
- `--viz none`: disables visualization for throughput.
- `--visualizer kit`: opens the Kit viewer for debugging.
- `--video --video_length ... --video_interval ...`: records training videos.

The task-specific [DisplayPort guide](tasks/displayport_insertion) has the full
train/play/export workflow.
