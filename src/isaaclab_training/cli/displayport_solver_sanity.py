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


def _accumulate_termination_counts(raw_env: Any, counts: dict[str, int]) -> None:
    """Accumulate per-environment activations for every termination term."""
    import torch

    manager = raw_env.termination_manager
    for name in manager.active_terms:
        count = int(torch.count_nonzero(manager.get_term(name)).cpu())
        counts[name] = counts.get(name, 0) + count


def _evaluate_termination(raw_env: Any, term_name: str, params: dict[str, Any]) -> int:
    """Evaluate one termination with explicit parameters without resetting environments."""
    import torch

    term = raw_env.termination_manager.get_term_cfg(term_name)
    return int(torch.count_nonzero(term.func(raw_env, **params)).cpu())


def _suspend_grasp_terminations(raw_env: Any) -> dict[str, dict[str, Any]]:
    """Widen grasp termination thresholds and return their original parameters."""
    manager = raw_env.termination_manager
    original: dict[str, dict[str, Any]] = {}
    overrides = {
        "plug_dropped": {"distance_threshold": math.inf},
        "plug_orientation_exceeded": {
            "roll_threshold_deg": 360.0,
            "pitch_threshold_deg": 360.0,
            "yaw_threshold_deg": 360.0,
        },
    }
    for term_name, params in overrides.items():
        term = manager.get_term_cfg(term_name)
        original[term_name] = dict(term.params)
        term.params.update(params)
        manager.set_term_cfg(term_name, term)
    return original


def _restore_termination_params(raw_env: Any, original: dict[str, dict[str, Any]]) -> None:
    """Restore termination parameters saved by :func:`_suspend_grasp_terminations`."""
    manager = raw_env.termination_manager
    for term_name, params in original.items():
        term = manager.get_term_cfg(term_name)
        term.params = params
        manager.set_term_cfg(term_name, term)


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


def _gravity_compensation_snapshot(robot: Any, gravity_m_s2: float = 9.81) -> dict[str, Any]:
    """Return authored body-gravity cancellation forces and their mass-based error."""
    import torch

    force_w = _to_torch(robot.permanent_wrench_composer.global_force_at_com_w).clone()
    mass = _to_torch(robot.data.body_mass)
    expected_w = torch.zeros_like(force_w)
    expected_w[..., 2] = mass * gravity_m_s2
    return {
        "force_w": force_w,
        "maximum_force_n": _maximum(torch.linalg.vector_norm(force_w, dim=-1)),
        "maximum_expected_error_n": _maximum(torch.linalg.vector_norm(force_w - expected_w, dim=-1)),
    }


def _disable_gripper_squeeze(env_cfg: Any) -> None:
    """Match the gripper target to its reset state after preset resolution."""
    env_cfg.hand_close_width = env_cfg.hand_hold_width


def _newton_dynamics_snapshot(raw_env: Any) -> dict[str, Any]:
    """Return effective robot inertias and Newton's internal joint-drive state."""
    import torch

    robot = raw_env.scene["robot"]
    body_mass = _to_torch(robot.data.body_mass)[0].detach().cpu()
    body_inertia = _to_torch(robot.data.body_inertia)[0].detach().cpu().reshape(-1, 3, 3)
    bodies = []
    for body_name, mass, inertia in zip(robot.body_names, body_mass, body_inertia, strict=True):
        symmetric_inertia = 0.5 * (inertia + inertia.T)
        bodies.append(
            {
                "name": body_name,
                "mass_kg": float(mass),
                "inertia_kg_m2": inertia.tolist(),
                "minimum_inertia_eigenvalue_kg_m2": float(torch.linalg.eigvalsh(symmetric_inertia).amin()),
            }
        )

    manager = raw_env.sim.physics_manager
    model = getattr(manager, "_model", None)
    state = getattr(manager, "_state_0", None)
    control = getattr(manager, "_control", None)
    if model is None or state is None or control is None:
        return {"bodies": bodies, "joint_drives_available": False}

    required = (
        model.joint_q_start,
        model.joint_qd_start,
        model.joint_target_ke,
        model.joint_target_kd,
        model.joint_limit_lower,
        model.joint_limit_upper,
        state.joint_q,
        state.joint_qd,
        control.joint_target_q,
        control.joint_target_qd,
    )
    if any(value is None for value in required):
        return {"bodies": bodies, "joint_drives_available": False}

    q_start = _to_torch(model.joint_q_start).detach().cpu().tolist()
    qd_start = _to_torch(model.joint_qd_start).detach().cpu().tolist()
    q = _to_torch(state.joint_q).detach().cpu()
    qd = _to_torch(state.joint_qd).detach().cpu()
    target_q = _to_torch(control.joint_target_q).detach().cpu()
    target_qd = _to_torch(control.joint_target_qd).detach().cpu()
    target_ke = _to_torch(model.joint_target_ke).detach().cpu()
    target_kd = _to_torch(model.joint_target_kd).detach().cpu()
    lower = _to_torch(model.joint_limit_lower).detach().cpu()
    upper = _to_torch(model.joint_limit_upper).detach().cpu()
    robot_joint_tokens = (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "finger_joint",
        "knuckle_joint",
        "outer_finger_joint",
    )
    joints = []
    for joint_index, joint_label in enumerate(model.joint_label):
        if not any(token in joint_label for token in robot_joint_tokens):
            continue
        coord_start, coord_end = q_start[joint_index : joint_index + 2]
        dof_start, dof_end = qd_start[joint_index : joint_index + 2]
        joints.append(
            {
                "label": joint_label,
                "position": q[coord_start:coord_end].tolist(),
                "target_position": target_q[coord_start:coord_end].tolist(),
                "velocity": qd[dof_start:dof_end].tolist(),
                "target_velocity": target_qd[dof_start:dof_end].tolist(),
                "target_ke": target_ke[dof_start:dof_end].tolist(),
                "target_kd": target_kd[dof_start:dof_end].tolist(),
                "limit_lower": lower[dof_start:dof_end].tolist(),
                "limit_upper": upper[dof_start:dof_end].tolist(),
            }
        )
    return {"bodies": bodies, "joint_drives_available": True, "joints": joints}


def _rigid_contact_snapshot(raw_env: Any) -> dict[str, Any]:
    """Return active Newton rigid-contact count, penetration, and force telemetry."""
    import newton
    import torch
    import warp as wp

    manager = raw_env.sim.physics_manager
    contacts = getattr(manager, "_contacts", None)
    model = getattr(manager, "_model", None)
    state = getattr(manager, "_state_0", None)
    if contacts is None or model is None or state is None:
        return {"available": False}
    count = int(wp.to_torch(contacts.rigid_contact_count)[0].cpu())
    snapshot: dict[str, Any] = {
        "available": True,
        "count": count,
        "maximum_penetration_m": 0.0,
        "maximum_force_n": 0.0,
    }
    if count <= 0:
        return snapshot
    distance = wp.empty_like(contacts.rigid_contact_margin0)
    newton.eval_rigid_contact_kinematics(model, state, contacts, out_distance=distance)
    distance_t = wp.to_torch(distance)[:count]
    force_t = wp.to_torch(contacts.rigid_contact_force)[:count]
    snapshot["maximum_penetration_m"] = float(torch.clamp_min(-distance_t, 0.0).amax().cpu())
    snapshot["maximum_force_n"] = float(torch.linalg.vector_norm(force_t, dim=-1).amax().cpu())
    return snapshot


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


def _run_live_checks(
    env: Any,
    report: dict[str, Any],
    *,
    probe_steps: int = 1,
    probe_only: bool = False,
    move_plug_clear: bool = False,
) -> None:
    """Populate ``report`` with deterministic dynamics and contact evidence."""
    import isaaclab.utils.math as math_utils
    import torch

    raw = env.unwrapped
    if probe_steps < 1:
        raise ValueError(f"probe_steps must be positive, got {probe_steps}.")
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
        "move_plug_clear": move_plug_clear,
        "hand_hold_width": float(raw.cfg.hand_hold_width),
        "hand_close_width": float(raw.cfg.hand_close_width),
    }
    if tuple(controller.motion_stiffness_task) != (300.0,) * 6:
        failures.append("OSC stiffness is not 300 on all six decoupled task axes")

    # Inspect one complete policy step before the task can autoreset a failed
    # grasp. This isolates arm dynamics from reset-pose artifacts in the longer
    # response gates below.
    _configure_reset(raw, at_goal=False, depth_m=0.04)
    _reset(env)
    robot = raw.scene["robot"]
    plug = raw.scene["dp_plug"]
    gravity_0 = _gravity_compensation_snapshot(robot)
    _reset(env)
    gravity_1 = _gravity_compensation_snapshot(robot)
    gravity_compensation_enabled = "robot_body_gravity_compensation" in raw.event_manager.active_terms.get("reset", [])
    checks["gravity_compensation"] = {
        "enabled": gravity_compensation_enabled,
        "maximum_force_n": gravity_1["maximum_force_n"],
        "maximum_expected_error_n": gravity_1["maximum_expected_error_n"],
        "maximum_change_after_second_reset_n": _maximum(
            torch.linalg.vector_norm(gravity_1["force_w"] - gravity_0["force_w"], dim=-1)
        ),
    }
    if gravity_compensation_enabled and (
        checks["gravity_compensation"]["maximum_expected_error_n"] > 1.0e-4
        or checks["gravity_compensation"]["maximum_change_after_second_reset_n"] > 1.0e-4
    ):
        failures.append("robot body gravity compensation is missing or accumulates across resets")
    if move_plug_clear:
        plug_pose = torch.cat((_to_torch(plug.data.root_pos_w), _to_torch(plug.data.root_quat_w)), dim=-1).clone()
        plug_pose[:, 2] += 1.0
        plug.write_root_pose_to_sim(plug_pose)
        plug.write_root_velocity_to_sim(torch.zeros((raw.num_envs, 6), device=raw.device))
        raw.sim.forward()
    checks["newton_dynamics_at_reset"] = _newton_dynamics_snapshot(raw)
    arm_joint_ids, arm_joint_names = robot.find_joints(
        ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
        preserve_order=True,
    )
    all_joint_ids, all_joint_names = robot.find_joints([".*"], preserve_order=True)
    arm_joint_id_set = set(arm_joint_ids)
    gripper_joint_ids = [joint_id for joint_id in all_joint_ids if joint_id not in arm_joint_id_set]
    gripper_joint_names = [
        joint_name
        for joint_id, joint_name in zip(all_joint_ids, all_joint_names, strict=True)
        if joint_id not in arm_joint_id_set
    ]
    flange_pos_0, flange_quat_0 = _flange_pose_in_base(raw)
    grasp_pos_0, grasp_quat_0 = _plug_pose_in_flange(raw)
    joint_pos_0 = _to_torch(robot.data.joint_pos)[:, arm_joint_ids].clone()
    plug_pos_0 = _to_torch(plug.data.root_pos_w).clone()
    plug_quat_0 = _to_torch(plug.data.root_quat_w).clone()
    maximum_joint_delta = torch.zeros(len(arm_joint_ids), device=raw.device)
    maximum_joint_speed = torch.zeros(len(arm_joint_ids), device=raw.device)
    maximum_applied_effort = torch.zeros(len(arm_joint_ids), device=raw.device)
    maximum_computed_effort = torch.zeros(len(arm_joint_ids), device=raw.device)
    maximum_gripper_speed = torch.zeros(len(gripper_joint_ids), device=raw.device)
    maximum_gripper_applied_effort = torch.zeros(len(gripper_joint_ids), device=raw.device)
    maximum_gripper_computed_effort = torch.zeros(len(gripper_joint_ids), device=raw.device)
    maximum_plug_com_velocity = torch.zeros(6, device=raw.device)
    maximum_robot_body_velocity = torch.zeros(6, device=raw.device)
    maximum_flange_translation = 0.0
    maximum_flange_rotation = 0.0
    maximum_grasp_translation = 0.0
    maximum_grasp_rotation = 0.0
    maximum_plug_translation = 0.0
    maximum_plug_rotation = 0.0
    physical_termination_counts = {
        "plug_dropped": 0,
        "plug_orientation_exceeded": 0,
    }
    unexpected_reset = False
    finite_state = True
    observations = None
    first_step: dict[str, Any] | None = None
    original_termination_params = _suspend_grasp_terminations(raw)
    try:
        for step_index in range(probe_steps):
            observations, reset = _step(env, zero)
            unexpected_reset |= reset
            for term_name, params in original_termination_params.items():
                physical_termination_counts[term_name] = max(
                    physical_termination_counts[term_name],
                    _evaluate_termination(raw, term_name, params),
                )
            flange_pos, flange_quat = _flange_pose_in_base(raw)
            grasp_pos, grasp_quat = _plug_pose_in_flange(raw)
            joint_pos = _to_torch(robot.data.joint_pos)[:, arm_joint_ids]
            joint_vel = torch.abs(_to_torch(robot.data.joint_vel)[:, arm_joint_ids])
            applied_effort = torch.abs(_to_torch(robot.data.applied_torque)[:, arm_joint_ids])
            computed_effort = torch.abs(_to_torch(robot.data.computed_torque)[:, arm_joint_ids])
            gripper_speed = torch.abs(_to_torch(robot.data.joint_vel)[:, gripper_joint_ids])
            gripper_applied_effort = torch.abs(_to_torch(robot.data.applied_torque)[:, gripper_joint_ids])
            gripper_computed_effort = torch.abs(_to_torch(robot.data.computed_torque)[:, gripper_joint_ids])
            plug_velocity = torch.abs(_to_torch(plug.data.root_com_vel_w))
            body_velocity = torch.abs(_to_torch(robot.data.body_link_vel_w))
            maximum_joint_delta = torch.maximum(
                maximum_joint_delta,
                torch.amax(torch.abs(joint_pos - joint_pos_0), dim=0),
            )
            maximum_joint_speed = torch.maximum(maximum_joint_speed, torch.amax(joint_vel, dim=0))
            maximum_applied_effort = torch.maximum(maximum_applied_effort, torch.amax(applied_effort, dim=0))
            maximum_computed_effort = torch.maximum(maximum_computed_effort, torch.amax(computed_effort, dim=0))
            maximum_gripper_speed = torch.maximum(maximum_gripper_speed, torch.amax(gripper_speed, dim=0))
            maximum_gripper_applied_effort = torch.maximum(
                maximum_gripper_applied_effort,
                torch.amax(gripper_applied_effort, dim=0),
            )
            maximum_gripper_computed_effort = torch.maximum(
                maximum_gripper_computed_effort,
                torch.amax(gripper_computed_effort, dim=0),
            )
            maximum_plug_com_velocity = torch.maximum(
                maximum_plug_com_velocity,
                torch.amax(plug_velocity, dim=0),
            )
            maximum_robot_body_velocity = torch.maximum(
                maximum_robot_body_velocity,
                torch.amax(body_velocity, dim=(0, 1)),
            )
            maximum_flange_translation = max(
                maximum_flange_translation,
                _maximum(torch.linalg.vector_norm(flange_pos - flange_pos_0, dim=-1)),
            )
            maximum_flange_rotation = max(
                maximum_flange_rotation,
                _maximum(math_utils.quat_error_magnitude(flange_quat, flange_quat_0)),
            )
            maximum_grasp_translation = max(
                maximum_grasp_translation,
                _maximum(torch.linalg.vector_norm(grasp_pos - grasp_pos_0, dim=-1)),
            )
            maximum_grasp_rotation = max(
                maximum_grasp_rotation,
                _maximum(math_utils.quat_error_magnitude(grasp_quat, grasp_quat_0)),
            )
            maximum_plug_translation = max(
                maximum_plug_translation,
                _maximum(torch.linalg.vector_norm(_to_torch(plug.data.root_pos_w) - plug_pos_0, dim=-1)),
            )
            maximum_plug_rotation = max(
                maximum_plug_rotation,
                _maximum(math_utils.quat_error_magnitude(_to_torch(plug.data.root_quat_w), plug_quat_0)),
            )
            finite_state &= _scene_finite(raw, observations)
            if step_index == 0:
                first_step = {
                    "abs_joint_delta_rad": torch.amax(torch.abs(joint_pos - joint_pos_0), dim=0).cpu().tolist(),
                    "abs_joint_speed_rad_s": torch.amax(joint_vel, dim=0).cpu().tolist(),
                    "abs_computed_effort_nm": torch.amax(computed_effort, dim=0).cpu().tolist(),
                    "abs_applied_effort_nm": torch.amax(applied_effort, dim=0).cpu().tolist(),
                    "abs_gripper_speed_rad_s": torch.amax(gripper_speed, dim=0).cpu().tolist(),
                    "abs_gripper_computed_effort_nm": torch.amax(gripper_computed_effort, dim=0).cpu().tolist(),
                    "abs_gripper_applied_effort_nm": torch.amax(gripper_applied_effort, dim=0).cpu().tolist(),
                    "flange_translation_m": _maximum(torch.linalg.vector_norm(flange_pos - flange_pos_0, dim=-1)),
                    "flange_rotation_rad": _maximum(math_utils.quat_error_magnitude(flange_quat, flange_quat_0)),
                    "grasp_translation_m": _maximum(torch.linalg.vector_norm(grasp_pos - grasp_pos_0, dim=-1)),
                    "grasp_rotation_rad": _maximum(math_utils.quat_error_magnitude(grasp_quat, grasp_quat_0)),
                    "plug_translation_m": _maximum(
                        torch.linalg.vector_norm(_to_torch(plug.data.root_pos_w) - plug_pos_0, dim=-1)
                    ),
                    "plug_rotation_rad": _maximum(
                        math_utils.quat_error_magnitude(_to_torch(plug.data.root_quat_w), plug_quat_0)
                    ),
                    "abs_plug_com_velocity": torch.amax(plug_velocity, dim=0).cpu().tolist(),
                    "abs_robot_body_velocity": torch.amax(body_velocity, dim=(0, 1)).cpu().tolist(),
                    "rigid_contacts": _rigid_contact_snapshot(raw),
                }
    finally:
        _restore_termination_params(raw, original_termination_params)
    one_step = {
        "probe_steps": probe_steps,
        "elapsed_simulation_s": probe_steps * raw.step_dt,
        "first_step": first_step,
        "arm_joint_names": list(arm_joint_names),
        "gripper_joint_names": gripper_joint_names,
        "maximum_abs_joint_delta_rad": maximum_joint_delta.cpu().tolist(),
        "maximum_abs_joint_speed_rad_s": maximum_joint_speed.cpu().tolist(),
        "maximum_abs_computed_effort_nm": maximum_computed_effort.cpu().tolist(),
        "maximum_abs_applied_effort_nm": maximum_applied_effort.cpu().tolist(),
        "maximum_abs_gripper_speed_rad_s": maximum_gripper_speed.cpu().tolist(),
        "maximum_abs_gripper_computed_effort_nm": maximum_gripper_computed_effort.cpu().tolist(),
        "maximum_abs_gripper_applied_effort_nm": maximum_gripper_applied_effort.cpu().tolist(),
        "maximum_flange_translation_m": maximum_flange_translation,
        "maximum_flange_rotation_rad": maximum_flange_rotation,
        "maximum_grasp_translation_m": maximum_grasp_translation,
        "maximum_grasp_rotation_rad": maximum_grasp_rotation,
        "maximum_plug_translation_m": maximum_plug_translation,
        "maximum_plug_rotation_rad": maximum_plug_rotation,
        "maximum_abs_plug_com_velocity": maximum_plug_com_velocity.cpu().tolist(),
        "maximum_abs_robot_body_velocity": maximum_robot_body_velocity.cpu().tolist(),
        "physical_termination_counts": physical_termination_counts,
        "unexpected_reset": unexpected_reset,
        "finite": finite_state,
        "rigid_contacts": _rigid_contact_snapshot(raw),
    }
    checks["one_step_without_autoreset"] = one_step
    if unexpected_reset:
        failures.append("one-step diagnostic reset despite suspended grasp terminations")
    if probe_only:
        return

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
    termination_counts: dict[str, int] = {}
    finite_state = True
    for _ in range(30):
        observations, reset = _step(env, zero)
        unexpected_resets += int(reset)
        _accumulate_termination_counts(raw, termination_counts)
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
        "termination_counts": termination_counts,
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
        termination_counts = {}
        for _ in range(4):
            observations, reset = _step(env, action)
            reset_count += int(reset)
            _accumulate_termination_counts(raw, termination_counts)
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
                "termination_counts": termination_counts,
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
    termination_counts = {}
    finite_state = True
    for _ in range(30):
        observations, reset = _step(env, zero)
        unexpected_resets += int(reset)
        _accumulate_termination_counts(raw, termination_counts)
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
        "termination_counts": termination_counts,
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
    termination_counts = {}
    for _ in range(180):
        state = _insertion_state(raw)
        action = _scaled_linear_action(raw, state["seat_error_b"])
        observations, reset = _step(env, action)
        unexpected_resets += int(reset)
        _accumulate_termination_counts(raw, termination_counts)
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
        "termination_counts": termination_counts,
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
    termination_counts = {}
    for _ in range(30):
        observations, reset = _step(env, zero)
        unexpected_resets += int(reset)
        _accumulate_termination_counts(raw, termination_counts)
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
        "termination_counts": termination_counts,
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
    termination_counts = {}
    finite_state = True
    for _ in range(30):
        state = _insertion_state(raw)
        robot_quat = _to_torch(raw.scene["robot"].data.root_quat_w)
        axis_b = math_utils.quat_apply_inverse(robot_quat, state["axis_w"])
        axial_error = torch.sum(state["seat_error_b"] * axis_b, dim=-1, keepdim=True) * axis_b
        observations, reset = _step(env, _scaled_linear_action(raw, axial_error))
        unexpected_resets += int(reset)
        _accumulate_termination_counts(raw, termination_counts)
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
        "termination_counts": termination_counts,
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
    parser.add_argument("--probe_steps", type=int, default=1)
    parser.add_argument("--probe_only", action="store_true")
    parser.add_argument("--move_plug_clear", action="store_true")
    parser.add_argument("--disable_gripper_squeeze", action="store_true")
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
        if args.disable_gripper_squeeze:
            _disable_gripper_squeeze(env_cfg)
        if hasattr(env_cfg.events, "randomize_arm_joint_friction"):
            env_cfg.events.randomize_arm_joint_friction = None
        if hasattr(env_cfg.events, "randomize_arm_pd_gains"):
            env_cfg.events.randomize_arm_pd_gains = None
        env_cfg.validate()
        with launch_simulation(env_cfg, args):
            env = None
            try:
                env = gym.make(args.task, cfg=env_cfg)
                _run_live_checks(
                    env,
                    report,
                    probe_steps=args.probe_steps,
                    probe_only=args.probe_only,
                    move_plug_clear=args.move_plug_clear,
                )
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
