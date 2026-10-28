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
  --num_envs 16 \
  --max_iterations 1 \
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
IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton
IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton-Play
IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton-ROS-Inference
```

The original `IsaacContrib-Deploy-...` IDs are also registered as compatibility aliases.

Task space is the recommended DisplayPort deployment path. PhysX and Newton
both use Isaac Lab's stock nominal Rizon 4s with Grav USD unless
`env.scene.robot.spawn.usd_path` is explicitly overridden. This command uses
the nominal robot with the existing PhysX task:

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-ROS-Inference \
  --num_envs 4096 --max_iterations 1500 --seed 42 \
  --visualizer none
```

To train for a calibrated physical arm, pass its validated USD as an absolute
Hydra override while leaving the packaged default unchanged:

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-ROS-Inference \
  --num_envs 4096 --max_iterations 1500 --seed 42 \
  --visualizer none \
  env.scene.robot.spawn.usd_path=/absolute/path/to/calibrated_rizon4s.usd
```

### Newton task-space training

The DisplayPort USDs resolve from the versioned Isaac production asset root
selected by the installed Isaac Sim release. Set
`ISAACLAB_TRAINING_DISPLAY_ASSETS_DIR` to an approved mirror for offline or
air-gapped deployments. Newton validates every required point-SDF mesh after
composition and fails if the asset layout changes.


The Newton task is a separate checkpoint ABI and must be selected explicitly.
First run a portable optimizer smoke with the default nominal asset:

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton-ROS-Inference \
  --num_envs 16 --max_iterations 1 --seed 123 \
  --visualizer none \
  physics=newton_sdf
```

Full Newton training uses the same nominal asset by default; no robot-USD
override is required:

```bash
uv run isaaclab train_multigpu --num_gpus 4 \
  --rl_library rsl_rl \
  --task IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton-ROS-Inference \
  --num_envs 256 --seed 123 \
  --visualizer none \
  physics=newton_sdf
```

The production fixed profile uses stable Newton 1.6 and the MJWarp solver,
point-SDF contacts with a 5 mm gap, a 2 kHz solver rate, a 200 Hz collision
refresh rate, and a 33.3 Hz policy rate. It uses torque-level OSC with full
inertial-dynamics decoupling, stiffness `[300, 300, 300, 300, 300, 300]`, damping
ratio `1`, and translation scales `[0.025, 0.025, 0.010]`. The arm uses
zero-gain IdealPD actuator groups with Newton-native effort saturation and
configured solver velocity limits; OSC remains the joint-effort source. Socket-position noise
is sampled from +/-10 mm once per episode, and the at-goal reset probability is
annealed from `0.8` to `0` over the first 500 iterations. Use 256 environments
per distributed rank; a four-rank job therefore trains 1,024 environments.

#### Experimental VBD profile

`physics=newton_vbd` selects an opt-in full-scene VBD experiment. It does not
change the default MJWarp profile, policy observations or actions, PPO setup,
reward, assets, or curriculum. The initial VBD profile uses compliant ALM with
contact history and latest-contact matching, 20 iterations, a 4 kHz solver and
collision refresh, a 5 mm point-SDF gap, and 2500 N/m contact stiffness with
50 N*s/m damping. Latest-contact matching enables deterministic contact
reduction, so the shared triangle-pair buffer uses the largest capacity that
fits Newton's packed contact IDs. CUDA graph capture remains enabled.

This experiment branch pins IsaacLab at
`4db894c4088075379f6c8508073b83090c3b6ecb` and Newton main at
`aad887137c1e55c4307d083daad8d19785a0bebe`. The MJWarp preset remains the
default configuration, but the exact Newton 1.6 production control belongs on
the production branch because this branch evaluates a newer Newton dependency.

Do not start a full training job with this profile until all of these gates pass
on the exact target build and assets:

1. Run the focused configuration tests.
2. Verify one pre-inserted plug remains stable in a one-environment scene.
3. Run a scripted insertion while the robot holds the plug and inspect contact,
   grasp, and orientation health.
4. Run an 8-16 environment, 6-10 iteration diagnostic with synchronous CUDA
   error reporting.
5. Scale through 64, 128, and 256 environments while checking collision and
   per-body contact-capacity watermarks before submitting full training. The
   256-environment gate is required because triangle-pair demand grows with the
   number of replicated worlds while the packed-ID capacity is fixed.

For the first diagnostic, replace `physics=newton_sdf` in the Newton smoke
command with `physics=newton_vbd`. This profile is a solver experiment, not a
validated deployment profile; do not export or deploy its checkpoints until it
passes the stability gates and a separate simulation and real-robot policy
qualification.

The nominal USD provides the portable training baseline. For policy training
that should match a particular physical arm, provide that arm's validated
calibrated USD as an absolute Hydra override:

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton-ROS-Inference \
  --num_envs 256 --visualizer none \
  physics=newton_sdf \
  env.scene.robot.spawn.usd_path=/absolute/path/to/calibrated_rizon4s.usd
```

The task preserves valid calibrated inertia. When the flange is the stock
mass-only marker with zero inertia, it explicitly authors Newton 1.6's former
uniform-sphere fallback and a valid principal-axis quaternion before import.
Missing flange/mass data, invalid principal axes, or missing point-SDF meshes fail before training.
Requalify materially changed robot assets because fully decoupled OSC consumes
their mass matrix.

Evaluate a checkpoint with the matching Newton play task:

```bash
uv run isaaclab play --rl_library rsl_rl \
  --task IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton-Play \
  --num_envs 1 --checkpoint /absolute/path/to/model.pt \
  --visualizer kit \
  physics=newton_sdf
```

`--visualizer none` and `--visualizer kit` control rendering only. The task and
`physics=newton_sdf` selects Newton physics; a visualizer value never selects a
physics backend.

> **Checkpoint compatibility:** the Newton actor consumes socket position first
> and observes the raw flange pose. The PhysX task-space actor consumes the TCP
> pose first. Both inputs contain 18 values, so a mismatched checkpoint can load
> without a shape error while receiving the wrong semantics. Never interchange
> PhysX and Newton checkpoints, task IDs, or export task IDs.

## Export

Use the generic exporter for task-space DisplayPort export:

```bash
uv run --group leapp isaaclab-training-export \
  --task IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-ROS-Inference \
  --checkpoint logs/rsl_rl/displayport_insertion_rizon4s/<run>/model_<n>.pt \
  --contract displayport_task_space
```

For a Newton checkpoint, use its matching ROS-inference task and preset:

```bash
uv run --group leapp isaaclab-training-export \
  --task IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton-ROS-Inference \
  --checkpoint logs/rsl_rl/displayport_insertion_rizon4s_newton_osc/<run>/model_<n>.pt \
  --contract displayport_task_space \
  physics=newton_sdf
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
