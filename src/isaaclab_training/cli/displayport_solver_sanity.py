# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run deterministic DisplayPort dynamics and contact gates for Newton solvers."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def _to_torch(value: Any) -> Any:
    """Return a Torch view for a Torch- or Warp-backed simulation value."""
    if hasattr(value, "torch"):
        return value.torch
    try:
        import warp as wp

        return wp.to_torch(value)
    except (AttributeError, TypeError):
        return value


def _finite(*values: Any) -> bool:
    """Return whether every tensor-like value is finite."""
    import torch

    for value in values:
        if isinstance(value, dict):
            if not _finite(*value.values()):
                return False
            continue
        value = _to_torch(value)
        if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
            return False
    return True


def _maximum(value: Any) -> float:
    """Return the maximum tensor value as a host scalar."""
    import torch

    return float(torch.as_tensor(_to_torch(value)).detach().amax().cpu())


def _minimum(value: Any) -> float:
    """Return the minimum tensor value as a host scalar."""
    import torch

    return float(torch.as_tensor(_to_torch(value)).detach().amin().cpu())


def _configure_reset(raw_env: Any, *, at_goal: bool, depth_m: float, lateral_offset_m: float = 0.0) -> None:
    """Configure the task's real reset term for one deterministic scenario."""
    curriculum = raw_env.event_manager.get_term_cfg("reset_plug_curriculum").func
    curriculum.at_goal_prob = 1.0 if at_goal else 0.0
    curriculum.at_goal_prob_final = None
    curriculum.at_goal_depth_range = [depth_m, depth_m]
    curriculum.approach_depth_range = [depth_m, depth_m]
    curriculum.normal_pose_range = {
        "x": [lateral_offset_m, lateral_offset_m],
        "y": [0.0, 0.0],
        "z": [0.0, 0.0],
    }


def _step(env: Any, action: Any) -> tuple[Any, bool]:
    """Step a Gymnasium environment and report any termination or truncation."""
    import torch

    observations, _, terminated, truncated, _ = env.step(action)
    return observations, bool(torch.any(terminated | truncated))


def _reset(env: Any) -> Any:
    """Reset a Gymnasium environment and return its observations."""
    observations, _ = env.reset()
    return observations


def _plug_pose_in_flange(raw_env: Any) -> tuple[Any, Any]:
    """Return the plug root pose expressed in the controlled flange frame."""
    import isaaclab.utils.math as math_utils

    robot = raw_env.scene["robot"]
    plug = raw_env.scene["dp_plug"]
    body_name = raw_env.cfg.end_effector_body_name
    body_ids, body_names = robot.find_bodies(body_name, preserve_order=True)
    if body_names != [body_name]:
        raise RuntimeError(f"Expected one controlled body {body_name!r}, got {body_names!r}")
    return math_utils.subtract_frame_transforms(
        _to_torch(robot.data.body_pos_w)[:, body_ids[0]],
        _to_torch(robot.data.body_quat_w)[:, body_ids[0]],
        _to_torch(plug.data.root_pos_w),
        _to_torch(plug.data.root_quat_w),
    )


def _flange_pose_in_base(raw_env: Any) -> tuple[Any, Any]:
    """Return the controlled flange pose in the robot-root frame."""
    import isaaclab.utils.math as math_utils

    robot = raw_env.scene["robot"]
    body_name = raw_env.cfg.end_effector_body_name
    body_ids, _ = robot.find_bodies(body_name, preserve_order=True)
    return math_utils.subtract_frame_transforms(
        _to_torch(robot.data.root_pos_w),
        _to_torch(robot.data.root_quat_w),
        _to_torch(robot.data.body_pos_w)[:, body_ids[0]],
        _to_torch(robot.data.body_quat_w)[:, body_ids[0]],
    )


def _insertion_state(raw_env: Any) -> dict[str, Any]:
    """Return plug-to-socket state and the physical-seat error."""
    import isaaclab.utils.math as math_utils
    import torch

    from isaaclab_training.tasks.displayport_insertion.displayport_insertion_env_cfg import (
        PLUG_GOAL_ROT,
        PLUG_INSERTION_OFFSET,
        SOCKET_INSERTION_OFFSET,
    )

    socket = raw_env.scene["dp_socket"]
    plug = raw_env.scene["dp_plug"]
    socket_pos = _to_torch(socket.data.root_pos_w)
    socket_quat = _to_torch(socket.data.root_quat_w)
    plug_pos = _to_torch(plug.data.root_pos_w)
    plug_quat = _to_torch(plug.data.root_quat_w)
    socket_offset = torch.as_tensor(SOCKET_INSERTION_OFFSET, device=raw_env.device).expand(raw_env.num_envs, -1)
    plug_offset = torch.as_tensor(PLUG_INSERTION_OFFSET, device=raw_env.device).expand(raw_env.num_envs, -1)
    goal_rotation = torch.as_tensor(PLUG_GOAL_ROT, device=raw_env.device).expand(raw_env.num_envs, -1)
    local_axis = torch.tensor((1.0, 0.0, 0.0), device=raw_env.device).expand(raw_env.num_envs, -1)
    socket_keypoint = socket_pos + math_utils.quat_apply(socket_quat, socket_offset)
    plug_keypoint = plug_pos + math_utils.quat_apply(plug_quat, plug_offset)
    plug_to_socket = plug_keypoint - socket_keypoint
    axis_w = math_utils.quat_apply(socket_quat, local_axis)
    goal_quat = math_utils.quat_mul(socket_quat, goal_rotation)
    curriculum = raw_env.event_manager.get_term_cfg("reset_plug_curriculum").func
    seat_offset_s = curriculum.at_goal_seat_offset.unsqueeze(0).expand(raw_env.num_envs, -1)
    seat_offset_w = math_utils.quat_apply(socket_quat, seat_offset_s)
    seat_error_w = seat_offset_w - plug_to_socket
    seat_error_b = math_utils.quat_apply_inverse(_to_torch(raw_env.scene["robot"].data.root_quat_w), seat_error_w)
    return {
        "plug_to_socket_w": plug_to_socket,
        "axis_w": axis_w,
        "signed_depth_m": torch.sum(plug_to_socket * axis_w, dim=-1),
        "seat_signed_depth_m": torch.sum(seat_offset_w * axis_w, dim=-1),
        "mate_error_m": torch.linalg.vector_norm(plug_to_socket, dim=-1),
        "orientation_error_rad": math_utils.quat_error_magnitude(plug_quat, goal_quat),
        "seat_error_b": seat_error_b,
    }


def _scene_finite(raw_env: Any, observations: Any) -> bool:
    """Return whether robot, assets, and observations remain finite."""
    robot = raw_env.scene["robot"]
    plug = raw_env.scene["dp_plug"]
    socket = raw_env.scene["dp_socket"]
    return _finite(
        observations,
        robot.data.joint_pos,
        robot.data.joint_vel,
        robot.data.joint_acc,
        robot.data.body_pos_w,
        robot.data.body_quat_w,
        robot.data.body_link_vel_w,
        plug.data.root_pos_w,
        plug.data.root_quat_w,
        plug.data.root_com_vel_w,
        socket.data.root_pos_w,
        socket.data.root_quat_w,
        socket.data.root_com_vel_w,
    )


def _scaled_linear_action(raw_env: Any, physical_delta_b: Any, maximum_step_m: float = 0.001) -> Any:
    """Convert a bounded physical translation in the robot base to an OSC action."""
    import torch

    norm = torch.linalg.vector_norm(physical_delta_b, dim=-1, keepdim=True)
    physical_delta_b = physical_delta_b * torch.clamp(maximum_step_m / torch.clamp_min(norm, 1.0e-12), max=1.0)
    scale = torch.as_tensor(
        raw_env.cfg.actions.arm_action.position_scale,
        device=raw_env.device,
        dtype=physical_delta_b.dtype,
    )
    linear = torch.clamp(physical_delta_b / scale, -1.0, 1.0)
    return torch.cat((linear, torch.zeros_like(linear)), dim=-1)


def _run_live_checks(env: Any, report: dict[str, Any]) -> None:
    """Populate ``report`` with deterministic dynamics and contact evidence."""
    import isaaclab.utils.math as math_utils
    import torch

    raw = env.unwrapped
    zero = torch.zeros((raw.num_envs, raw.action_manager.total_action_dim), device=raw.device)
    checks: dict[str, Any] = report["checks"]
    failures: list[str] = report["failures"]

    controller = raw.cfg.actions.arm_action.controller_cfg
    checks["configuration"] = {
        "assets": {
            name: str(getattr(getattr(raw.cfg.scene, name).spawn, "usd_path", ""))
            for name in ("robot", "dp_plug", "dp_socket")
        },
        "observation_order": list(raw.cfg.task_space_obs_order),
        "position_scale": list(raw.cfg.actions.arm_action.position_scale),
        "orientation_scale": float(raw.cfg.actions.arm_action.orientation_scale),
        "stiffness": list(controller.motion_stiffness_task),
        "damping_ratio": list(controller.motion_damping_ratio_task),
        "inertial_dynamics_decoupling": bool(controller.inertial_dynamics_decoupling),
        "gravity_compensation": bool(controller.gravity_compensation),
        "physics": type(raw.cfg.sim.physics).__name__,
        "solver": type(raw.cfg.sim.physics.solver_cfg).__name__,
    }
    if tuple(controller.motion_stiffness_task) != (300.0,) * 6:
        failures.append("OSC stiffness is not 300 on all six decoupled task axes")

    # The arm and grasp must remain stable under a zero relative-pose command.
    _configure_reset(raw, at_goal=False, depth_m=0.04)
    observations = _reset(env)
    grasp_pos_0, grasp_quat_0 = _plug_pose_in_flange(raw)
    grasp_pos_0 = grasp_pos_0.clone()
    grasp_quat_0 = grasp_quat_0.clone()
    maximum_joint_speed = 0.0
    maximum_grasp_drift = 0.0
    maximum_grasp_rotation = 0.0
    unexpected_resets = 0
    finite_state = True
    for _ in range(30):
        observations, reset = _step(env, zero)
        unexpected_resets += int(reset)
        position, quaternion = _plug_pose_in_flange(raw)
        joint_speed = _maximum(torch.abs(_to_torch(raw.scene["robot"].data.joint_vel)))
        maximum_joint_speed = max(maximum_joint_speed, joint_speed)
        grasp_drift = _maximum(torch.linalg.vector_norm(position - grasp_pos_0, dim=-1))
        maximum_grasp_drift = max(maximum_grasp_drift, grasp_drift)
        maximum_grasp_rotation = max(
            maximum_grasp_rotation,
            _maximum(math_utils.quat_error_magnitude(quaternion, grasp_quat_0)),
        )
        finite_state &= _scene_finite(raw, observations)
    checks["zero_hold"] = {
        "maximum_joint_speed_rad_s": maximum_joint_speed,
        "maximum_grasp_drift_m": maximum_grasp_drift,
        "maximum_grasp_rotation_rad": maximum_grasp_rotation,
        "unexpected_resets": unexpected_resets,
        "finite": finite_state,
    }
    if not finite_state or unexpected_resets or maximum_joint_speed > 0.5:
        failures.append("zero-action robot dynamics are unstable")
    if maximum_grasp_drift > 0.003 or maximum_grasp_rotation > math.radians(6.0):
        failures.append("the gripper does not retain the plug during zero-action hold")

    # Exercise every OSC axis in free space. Translation direction is checked;
    # rotation uses a magnitude check because its rotvec basis changes with pose.
    axis_evidence = []
    for axis in range(6):
        _configure_reset(raw, at_goal=False, depth_m=0.04)
        observations = _reset(env)
        position_0, quaternion_0 = _flange_pose_in_base(raw)
        action = torch.zeros_like(zero)
        action[:, axis] = 0.4
        reset_count = 0
        for _ in range(4):
            observations, reset = _step(env, action)
            reset_count += int(reset)
        position_1, quaternion_1 = _flange_pose_in_base(raw)
        if axis < 3:
            signed_response = _minimum(position_1[:, axis] - position_0[:, axis])
            response = signed_response
            passed = 0.0001 < signed_response < 0.05
            evidence = {"axis": axis, "signed_response": response}
        else:
            rotvec_b = math_utils.quat_box_minus(quaternion_1, quaternion_0)
            commanded_axis = axis - 3
            response = _minimum(rotvec_b[:, commanded_axis])
            total_response = _maximum(torch.linalg.vector_norm(rotvec_b, dim=-1))
            cross_axis = rotvec_b.clone()
            cross_axis[:, commanded_axis] = 0.0
            cross_axis_response = _maximum(torch.linalg.vector_norm(cross_axis, dim=-1))
            passed = 0.0005 < response < 0.2 and total_response < 0.2
            evidence = {
                "axis": axis,
                "signed_response": response,
                "total_response": total_response,
                "cross_axis_response": cross_axis_response,
            }
        axis_evidence.append(
            {
                **evidence,
                "unexpected_resets": reset_count,
                "finite": _scene_finite(raw, observations),
            }
        )
        if not passed or reset_count or not _scene_finite(raw, observations):
            failures.append(f"OSC axis {axis} failed its free-space response gate")
    checks["osc_axis_response"] = axis_evidence

    # A plug reset at the physical seat must remain stable in contact.
    _configure_reset(raw, at_goal=True, depth_m=0.0)
    observations = _reset(env)
    initial = _insertion_state(raw)
    minimum_success = 1.0
    maximum_mate_drift = 0.0
    unexpected_resets = 0
    finite_state = True
    for _ in range(30):
        observations, reset = _step(env, zero)
        unexpected_resets += int(reset)
        state = _insertion_state(raw)
        maximum_mate_drift = max(
            maximum_mate_drift,
            _maximum(torch.linalg.vector_norm(state["plug_to_socket_w"] - initial["plug_to_socket_w"], dim=-1)),
        )
        success, _, _ = raw._compute_success()
        minimum_success = min(minimum_success, float(success.float().mean().cpu()))
        finite_state &= _scene_finite(raw, observations)
    checks["preinserted_hold"] = {
        "maximum_mate_drift_m": maximum_mate_drift,
        "minimum_success_rate": minimum_success,
        "unexpected_resets": unexpected_resets,
        "finite": finite_state,
    }
    if not finite_state or unexpected_resets or maximum_mate_drift > 0.002 or minimum_success < 1.0:
        failures.append("pre-inserted plug contact is not stable")

    # Drive a centered robot-held plug from 15 mm above the keypoint to the
    # physical seat, then remove the command and verify retention.
    _configure_reset(raw, at_goal=False, depth_m=0.015)
    observations = _reset(env)
    initial = _insertion_state(raw)
    consecutive_success = 0
    unexpected_resets = 0
    for _ in range(180):
        state = _insertion_state(raw)
        action = _scaled_linear_action(raw, state["seat_error_b"])
        observations, reset = _step(env, action)
        unexpected_resets += int(reset)
        success, _, _ = raw._compute_success()
        consecutive_success = consecutive_success + 1 if bool(torch.all(success)) else 0
        if reset or consecutive_success >= 5:
            break
    final = _insertion_state(raw)
    final_success, _, _ = raw._compute_success()
    insertion_progress = initial["signed_depth_m"] - final["signed_depth_m"]
    insertion = {
        "minimum_progress_m": _minimum(insertion_progress),
        "maximum_final_mate_error_m": _maximum(final["mate_error_m"]),
        "maximum_orientation_error_rad": _maximum(final["orientation_error_rad"]),
        "final_success_rate": float(final_success.float().mean().cpu()),
        "consecutive_success_steps": consecutive_success,
        "unexpected_resets": unexpected_resets,
        "finite": _scene_finite(raw, observations),
    }
    checks["centered_insertion"] = insertion
    insertion_passed = (
        insertion["finite"]
        and not unexpected_resets
        and insertion["minimum_progress_m"] >= 0.008
        and insertion["maximum_final_mate_error_m"] <= 0.004
        and insertion["maximum_orientation_error_rad"] <= math.radians(7.0)
        and insertion["final_success_rate"] == 1.0
        and consecutive_success >= 5
    )
    if not insertion_passed:
        failures.append("centered scripted insertion did not reach a stable successful seat")

    retained = _insertion_state(raw)
    minimum_success = 1.0
    maximum_mate_drift = 0.0
    unexpected_resets = 0
    for _ in range(30):
        observations, reset = _step(env, zero)
        unexpected_resets += int(reset)
        state = _insertion_state(raw)
        maximum_mate_drift = max(
            maximum_mate_drift,
            _maximum(torch.linalg.vector_norm(state["plug_to_socket_w"] - retained["plug_to_socket_w"], dim=-1)),
        )
        success, _, _ = raw._compute_success()
        minimum_success = min(minimum_success, float(success.float().mean().cpu()))
    checks["post_seat_retention"] = {
        "maximum_mate_drift_m": maximum_mate_drift,
        "minimum_success_rate": minimum_success,
        "unexpected_resets": unexpected_resets,
        "finite": _scene_finite(raw, observations),
    }
    if (
        not insertion_passed
        or maximum_mate_drift > 0.002
        or minimum_success < 1.0
        or unexpected_resets
        or not _scene_finite(raw, observations)
    ):
        failures.append("inserted plug did not remain seated after the command was removed")

    # Push straight inward with an 8 mm lateral offset. A working socket
    # collision surface must block the plug instead of allowing tunneling.
    _configure_reset(raw, at_goal=False, depth_m=0.012, lateral_offset_m=0.008)
    observations = _reset(env)
    maximum_overtravel = 0.0
    maximum_success = 0.0
    unexpected_resets = 0
    finite_state = True
    for _ in range(30):
        state = _insertion_state(raw)
        robot_quat = _to_torch(raw.scene["robot"].data.root_quat_w)
        axis_b = math_utils.quat_apply_inverse(robot_quat, state["axis_w"])
        axial_error = torch.sum(state["seat_error_b"] * axis_b, dim=-1, keepdim=True) * axis_b
        observations, reset = _step(env, _scaled_linear_action(raw, axial_error))
        unexpected_resets += int(reset)
        state = _insertion_state(raw)
        overtravel = state["seat_signed_depth_m"] - state["signed_depth_m"]
        maximum_overtravel = max(maximum_overtravel, _maximum(torch.clamp_min(overtravel, 0.0)))
        success, _, _ = raw._compute_success()
        maximum_success = max(maximum_success, float(success.float().mean().cpu()))
        finite_state &= _scene_finite(raw, observations)
        if reset:
            break
    checks["rim_collision"] = {
        "maximum_overtravel_m": maximum_overtravel,
        "maximum_success_rate": maximum_success,
        "unexpected_resets": unexpected_resets,
        "finite": finite_state,
    }
    if not finite_state or unexpected_resets or maximum_overtravel > 0.0015 or maximum_success > 0.0:
        failures.append("rim-offset plug tunneled through or falsely succeeded against the socket")


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser without importing the simulator."""
    from isaaclab.app import add_launcher_args
    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default="IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton-Play",
    )
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--report", default="artifacts/displayport_solver_sanity.json")
    add_launcher_args(parser)
    parser.set_defaults(_setup_preset_cli=setup_preset_cli)
    return parser


def main() -> int:
    """Run the live solver checks and return a process status."""
    import gymnasium as gym
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils.hydra import hydra_task_config

    import isaaclab_training.tasks  # noqa: F401

    parser = _build_parser()
    setup_preset_cli = parser.get_default("_setup_preset_cli")
    args, hydra_args = setup_preset_cli(parser)
    delattr(args, "_setup_preset_cli")
    report: dict[str, Any] = {
        "task": args.task,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "hydra_overrides": list(hydra_args),
        "checks": {},
        "failures": [],
    }
    output = Path(args.report).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0], *hydra_args]

    @hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
    def run(env_cfg: Any, _agent_cfg: Any) -> None:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        env_cfg.sim.device = args.device or env_cfg.sim.device
        env_cfg.episode_length_s = 60.0
        env_cfg.observations.policy.enable_corruption = False
        if hasattr(env_cfg.events, "randomize_arm_joint_friction"):
            env_cfg.events.randomize_arm_joint_friction = None
        if hasattr(env_cfg.events, "randomize_arm_pd_gains"):
            env_cfg.events.randomize_arm_pd_gains = None
        env_cfg.validate()
        with launch_simulation(env_cfg, args):
            env = None
            try:
                env = gym.make(args.task, cfg=env_cfg)
                _run_live_checks(env, report)
            finally:
                output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                if env is not None:
                    env.close()

    try:
        run()
    except BaseException as error:
        report["failures"].append(f"{type(error).__name__}: {error}")
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["failures"]:
        print(f"[FAIL] DisplayPort solver sanity failed: {report['failures']}", file=sys.stderr)
        return 1
    print("[PASS] DisplayPort solver sanity passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
