"""LEAPP export contracts for trained Isaac Lab policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

TASK_SPACE_EXPORT_MODEL_NAME = "DisplayPortTaskSpace"

_ROT6D_ELEMENTS = ["r00", "r01", "r02", "r10", "r11", "r12"]
_TASK_SPACE_INPUT_SPEC = (
    ("eef_pos", slice(0, 3), ["x", "y", "z"], "eef_pose_pos", "state/body/position"),
    ("eef_rot_6d", slice(3, 9), _ROT6D_ELEMENTS, "eef_pose_rot6d", "state/body/rotation_6d"),
    ("socket_kp_pos", slice(9, 12), ["x", "y", "z"], "socket_kp_pose_pos", "state/body/position"),
    ("socket_kp_rot_6d", slice(12, 18), _ROT6D_ELEMENTS, "socket_kp_pose_rot6d", "state/body/rotation_6d"),
)
_ACTION_ELEMENT_NAMES = [
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_axis_angle_x",
    "delta_axis_angle_y",
    "delta_axis_angle_z",
]


class ExportContract(Protocol):
    """Task-specific policy I/O annotation contract."""

    name: str
    graph_name: str
    skip_automatic_env_patch: bool

    def prepare_observations(self, graph_name: str, obs, dtype: torch.dtype):
        """Return observations to feed the policy during LEAPP tracing."""

    def export_action(self, graph_name: str, actions, env_cfg, device, dtype: torch.dtype, export_method: str) -> None:
        """Annotate task-specific policy output tensors."""


def task_space_policy_obs(obs):
    """Return the 18D actor observation tensor from an RSL-RL TensorDict."""
    if "policy" not in obs:
        raise KeyError(f"Expected a 'policy' observation group, got keys: {list(obs.keys())}")
    policy_obs = obs["policy"]
    if policy_obs.shape[-1] != 18:
        raise ValueError(f"Expected 18D task-space actor observation, got shape {tuple(policy_obs.shape)}")
    return policy_obs


def split_and_annotate_task_space_obs(graph_name: str, policy_obs):
    """Expose the exact trained 18D observation as four Deploy-facing input tensors."""
    import leapp
    from leapp.utils.tensor_description import TensorSemantics

    parts = []
    for name, index, element_names, source, kind in _TASK_SPACE_INPUT_SPEC:
        parts.append(
            leapp.annotate.input_tensors(
                graph_name,
                TensorSemantics(
                    name=name,
                    ref=policy_obs[:, index],
                    kind=kind,
                    element_names=[element_names],
                    extra={"source": source},
                ),
            )
        )
    return torch.cat(parts, dim=-1)


def export_task_space_action(graph_name: str, tensor, export_method: str) -> None:
    """Annotate the deploy-facing clipped and scaled task-space action."""
    import leapp
    from leapp.utils.tensor_description import TensorSemantics

    leapp.annotate.output_tensors(
        graph_name,
        TensorSemantics(
            name="arm_action",
            ref=tensor,
            kind="target/body/pose_relative",
            element_names=[_ACTION_ELEMENT_NAMES],
            extra={
                "isaaclab_connection": "action:arm_action:pose_rel",
                "target_types": ["pose_rel"],
            },
        ),
        export_with=export_method,
    )


def task_space_action_scale(env_cfg, device, dtype):
    """Return ``[position_scale] * 3 + [orientation_scale] * 3`` from the task action config."""
    action_cfg = env_cfg.actions.arm_action
    for attr in ("position_scale", "orientation_scale"):
        if not hasattr(action_cfg, attr):
            raise AttributeError(
                f"Task-space export requires an operational-space action exposing '{attr}', "
                f"got {type(action_cfg).__name__}."
            )
    scale_values = [float(action_cfg.position_scale)] * 3 + [float(action_cfg.orientation_scale)] * 3
    return torch.tensor(scale_values, device=device, dtype=dtype).unsqueeze(0)


@dataclass(frozen=True)
class DisplayPortTaskSpaceContract:
    """DisplayPort task-space Deploy contract.

    The policy still receives the byte-identical 18D trained observation, but LEAPP
    exposes that vector as named EEF/socket pose tensors and emits a clipped, scaled
    Cartesian pose delta.
    """

    name: str = "displayport_task_space"
    graph_name: str = TASK_SPACE_EXPORT_MODEL_NAME
    skip_automatic_env_patch: bool = True

    def prepare_observations(self, graph_name: str, obs, dtype: torch.dtype):
        policy_obs = task_space_policy_obs(obs).to(dtype=dtype)
        obs_for_policy = obs.clone()
        obs_for_policy["policy"] = split_and_annotate_task_space_obs(graph_name, policy_obs)
        return obs_for_policy

    def export_action(self, graph_name: str, actions, env_cfg, device, dtype: torch.dtype, export_method: str) -> None:
        scale = task_space_action_scale(env_cfg, device, dtype)
        processed_action = torch.clamp(actions, -1.0, 1.0) * scale
        export_task_space_action(graph_name, processed_action, export_method)


CONTRACTS: dict[str, ExportContract] = {
    "displayport_task_space": DisplayPortTaskSpaceContract(),
}


def get_export_contract(name: str | None) -> ExportContract | None:
    """Resolve an export contract by name."""
    if name in (None, "", "none"):
        return None
    try:
        return CONTRACTS[name]
    except KeyError as exc:
        available = ", ".join(sorted(CONTRACTS))
        raise ValueError(f"Unknown export contract '{name}'. Available contracts: {available}") from exc
