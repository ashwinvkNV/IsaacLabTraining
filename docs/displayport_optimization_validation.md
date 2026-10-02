# DisplayPort Training Optimization Validation

This document is the release gate for changes intended to improve DisplayPort
training throughput or memory use. An optimization is not accepted because a
short training run starts successfully. It must preserve the task, controller,
physics, learning, export, and deployment contracts described below.

The pure-Python planner and validators live in
`isaaclab_training.cli.displayport_training_validation`. They do not import the
simulator, so matrix and log checks can run in normal CI. Physics measurements
must come from the real Isaac Lab task on a GPU; an empty evidence template is
intentionally rejected.

The profiling report names GPU probe scripts that were not committed to its
source branch. This repository therefore treats the JSON validator as an
evidence gate, not as a measurement generator: the reset, contact, rendering,
and deployment values must be produced by reviewed GPU probe jobs. A candidate
cannot be marked release-qualified until those probe artifacts exist and the
populated evidence document passes. The bounded training commands below are
executable, but they do not replace the scripted physics probes.

## Non-negotiable comparison method

Every candidate is compared with its immediate baseline using the same seed,
GPU model, environment count, rollout sample budget, robot USD, and curriculum
mode. A pure configuration sweep uses the same commit. A branch-level
optimization comparison records exact baseline and candidate commits and keeps
all other variables fixed. Change one optimization group at a time.

Use environment samples, not training iterations, as the x-axis. The legacy
curriculum budget is:

```text
1024 environments × 512 steps/iteration × 500 iterations
= 262,144,000 environment samples
```

The equivalent curriculum end is therefore iteration 2,000 for 1,024 × 128
and iteration 500 for 4,096 × 128. The planner applies the same conversion to
maximum training iterations and checkpoint intervals.

Check a conversion directly:

```bash
uv run python -m isaaclab_training.cli.displayport_training_validation \
  sample_budget \
  --reference_envs 1024 \
  --reference_steps 512 \
  --reference_iterations 500 \
  --candidate_envs 4096 \
  --candidate_steps 128
```

## Controlled experiment matrix

The committed matrix contains these profiles for both PhysX and Newton:

| Profile | Environments/GPU | Solver iterations | PPO minibatches |
|---|---:|---:|---:|
| `e4096_s64_mb16` | 4096 | 64 | 16 |
| `e4096_s128_mb16` | 4096 | 128 | 16 |
| `e1024_s64_mb16` | 1024 | 64 | 16 |
| `e4096_s64_mb64` | 4096 | 64 | 64 |

Each profile is crossed with curriculum enabled/disabled and nominal/calibrated
robot USDs. PhysX and Newton solver iteration counts are separate backend knobs:
the planner writes rigid-body position iterations for the PhysX plug and socket,
but MJWarp `solver_cfg.iterations` for Newton. It never mixes those fields.
For recurrent PPO, environments per rank must divide evenly across minibatches.
The 4,096 × 128 × 16 profile preserves the legacy samples per update and
minibatch; the 1,024 × 128 profile intentionally uses a smaller update and must
be judged as an algorithmic as well as a throughput change.

Preview the complete 32-run matrix:

```bash
uv run python -m isaaclab_training.cli.displayport_training_validation \
  matrix --backend all --asset both --curriculum both --format table \
  --calibrated_usd /absolute/path/to/calibrated_rizon4s.usd
```

The command exits with status 2 while any row exceeds its evidence-backed
per-GPU limit. This is deliberate. On the current production branch, Newton's
scene-wide candidate-pair capacity is documented for 256 environments per GPU;
4,096 is not safe merely because the process launches. The default planner
limits are 1,024 for PhysX and 256 for Newton.

Only after capacity, memory, and contact telemetry pass at the requested scale,
raise the corresponding planner limit and emit commands:

```bash
uv run python -m isaaclab_training.cli.displayport_training_validation \
  matrix --backend physx --asset both --curriculum both --format commands \
  --calibrated_usd /absolute/path/to/calibrated_rizon4s.usd \
  --validated_physx_envs_per_gpu 4096

uv run python -m isaaclab_training.cli.displayport_training_validation \
  matrix --backend newton --asset both --curriculum both --format commands \
  --calibrated_usd /absolute/path/to/calibrated_rizon4s.usd \
  --validated_newton_envs_per_gpu 4096
```

The limit flag records a validation decision; it does not change simulator
capacity. Review the evidence file before using it. Do not submit blocked lines,
silently remove the block, or infer capacity from GPU memory alone.

## Required gate suite

Run the gates in this order. A failure stops the experiment campaign until it
is explained and fixed.

### 1. Static and configuration gates

```bash
uv run pytest -q tests/test_displayport_training_validation.py
uv run pytest -q tests/test_registration.py tests/test_displayport_newton_env_cfg.py
uv run pre-commit run --all-files
```

These verify sample-equivalent schedules, matrix completeness, backend-specific
override paths, unsafe-scale blocking, task registration, and the production
configuration contracts. They do not validate GPU physics.

### 2. Scene-load and bounded training diagnostics

Run nominal and calibrated USDs separately. Start at 16 environments for scene
and asset inspection, then run a 256-environment, single-GPU 6–10 iteration
diagnostic before increasing Newton scale. Enable synchronous CUDA error
reporting for the diagnostic in the job environment.

Generate the bounded 64/100/128-iteration Newton solver sweep with:

```bash
uv run python -m isaaclab_training.cli.displayport_training_validation \
  matrix --backend newton --asset nominal --curriculum both --format commands \
  --profile e256_s64_mb16 \
  --profile e256_s100_mb16 \
  --profile e256_s128_mb16 \
  --max_iterations 10
```

Run each emitted command with `CUDA_LAUNCH_BLOCKING=1` and
`PYTHONFAULTHANDLER=1`. Repeat with `--asset calibrated` and:

```text
--calibrated_usd /absolute/path/to/calibrated_rizon4s.usd
```

Capture the full stdout/stderr log. A run fails for any traceback, CUDA error,
out-of-memory event, non-finite scalar, collision/contact capacity overflow, or
dropped-contact message.

### 3. Reset and physics gates

Run the real training environment with policy-independent scripted actions.
Terminations and timeout may be disabled only for the long scripted physics
scenarios; curriculum-transient checks keep production terminations enabled.
Record plug pose in the socket frame, contact force, plug speed, flange pose,
finger joints, in-hand slip, and lost-plug state at every physics step.

Required scenarios and criteria:

| Gate | Required measurement | Pass condition |
|---|---|---|
| Reset IK | 6 resets × 1,024 environments | p99 error ≤ 15 µm, max error ≤ 81 µm, cap-hit fraction ≤ 0.5% |
| Curriculum transient | at-goal probability 1.0; 1,024 environments; 30 zero-action steps | overlap ≤ 0.22 mm, force ≤ 10 N, tilt ≤ 2°, zero terminations |
| Seated hold | hold, zero action, release, release-and-lift; two copies each for 20 s | 8/8 seated, drift ≤ 0.25 mm, overlap ≤ 0.22 mm |
| Scripted insertion | aligned, ±offset, rim, wiggle, and hard-press cases | candidate seated count at least baseline; no increased tunneling or changed jam depth |
| Offset sweep | narrow-axis ±0.25/0.5/0.75 mm and wide-axis ±0.5 mm | candidate seated count at least baseline |
| Carry | axis/Y/Z shake, fast lateral motion, circle, twist, hold, zero action | zero lost, in-hand drift ≤ 0.25 mm |
| Mass/colliders | inspect every active collider and rigid-body mass properties | mass, center of mass, inertia, and intended active colliders match baseline |
| Capacity | highest-contact reset and 300-step random-action runs | zero overflows, reducer insertion failures, or dropped contacts; collision/reducer/contact/constraint/GPU-memory peaks ≤ 80% |
| Render | visual run from two useful views | robot, fingers, plug, and socket visible |

Overlap must be measured against the original collision geometry, not a
simplified candidate mesh. Store machine-readable measurements, full logs, and
videos. Generate the required JSON skeleton with:

```bash
uv run python -m isaaclab_training.cli.displayport_training_validation \
  evidence_template > artifacts/displayport_validation.json
```

The template contains `null` values and therefore cannot pass. Populate it only
from measured probe results, then validate it:

```bash
uv run python -m isaaclab_training.cli.displayport_training_validation \
  validate_evidence artifacts/displayport_validation.json
```

This command exits nonzero for missing data or a failed criterion. It also
requires baseline and candidate insertion/offset results, preventing a candidate
from passing on absolute metrics while regressing against the control.

### 4. Learning and throughput gates

Run real RSL-RL training for at least 20 iterations after warm-up. Compare
candidate and baseline at equal accumulated samples. Report collection time,
learning time, total FPS, GPU memory, reward, success, terminal success, episode
length, all termination causes, value loss, and non-finite/error counts.

Analyze a captured console log:

```bash
uv run python -m isaaclab_training.cli.displayport_training_validation \
  analyze_log artifacts/candidate.log \
  --baseline artifacts/baseline.log \
  --minimum_iterations 20 \
  --minimum_fps_ratio_to_baseline 1.0
```

For convergence qualification, additionally set terminal-success thresholds
appropriate to the control run, for example:

```text
--minimum_final_terminal_success 0.80
--minimum_tail_terminal_success 0.75
```

Do not use an early success spike as evidence. Report the final value and the
last-20 median. Reject runs with backend errors even when policy metrics look
good. Profile a contact-rich phase; empty-space reset throughput is not a valid
proxy for trained-policy throughput.

### 5. Export, replay, and deployment gates

Training success alone does not establish real-robot deployability. For each
candidate checkpoint:

1. Replay deterministically in the matching backend, task, robot USD, action
   scale, and observation ABI.
2. Export with the supported LEAPP toolchain.
3. Verify the flat observation order matches training.
4. Feed identical observations and zeroed recurrent state to PyTorch and LEAPP;
   maximum action error must be at most `1e-5`.
5. Inspect action direction and scale in the policy/control frame.
6. Run the scripted simulation playback and preserve its video.
7. Only then schedule a guarded real-robot evaluation with workspace limits,
   low initial speed, an operator stop, and the exact deployment configuration
   recorded.

The evidence contract requires both observation-order equality and action
parity. Hardware success remains a separate evaluation result; it is not
inferred from this suite.

## Backend-specific cautions

- A PhysX solver position iteration is not equivalent to an MJWarp Newton
  iteration. Compare each candidate against its own backend control.
- The current Newton point-SDF configuration uses a 5 mm contact gap, 20
  substeps per 10 ms environment step (2 kHz), and collision decimation 10
  (200 Hz). Do not accidentally change these while sweeping solver iterations.
- Newton candidate-pair and contact capacities must be scaled and checked before
  increasing environments. GPU memory availability is necessary but not
  sufficient.
- Newton 1.6 verifies collision-pipeline buffers and reports broad-phase, GJK,
  triangle, SDF, contact, and reducer pressure. The diagnostic must additionally
  retain MJWarp per-world contact and constraint-row peaks, structural overflow
  flags, and the CUDA memory high-water mark. Do not infer 1,024- or
  4,096-environment safety from a clean 256-environment launch.
- PhysX buffer changes are PhysX-only. Do not copy their field names or measured
  capacities into Newton.
- Collider preprocessing is accepted only after mass/inertia, grasp equilibrium,
  insertion offsets, rendering, and original-mesh overlap all pass.
- Keep allocator settings in the launcher or workflow environment, not as an
  import-time Python side effect.

## Evidence retention

For every submitted run, retain:

- Git commit, dependency lock, backend and solver version;
- exact command, seed, environment count, rollout length, curriculum schedule,
  minibatches, robot USD identity, GPU allocation, and pool;
- stdout/stderr and structured scalar logs;
- capacity and memory telemetry;
- physics evidence JSON, result tables, and videos;
- checkpoints used for comparison;
- PyTorch/LEAPP parity output and deterministic replay video.

Without these artifacts, the result is exploratory and must not become the
shipping default.
