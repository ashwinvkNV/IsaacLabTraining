# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton point-SDF OSC configuration for Rizon 4S DisplayPort insertion.

It is separate from :mod:`task_space_env_cfg` because its checkpoint ABI observes
the flange origin in a different tensor order than the PhysX task-space policy.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg, VBDSolverCfg
from isaaclab_newton.sim.schemas import (
    MujocoCollisionCfg,
    MujocoJointDrivePropertiesCfg,
    MujocoRigidBodyPropertiesCfg,
    NewtonCollisionCfg,
    NewtonSDFCollisionCfg,
)
from isaaclab_physx.sim.schemas import PhysxCollisionCfg

from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs import mdp as env_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.schemas import RigidBodyBaseCfg
from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.utils import clone
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg

import isaaclab_training.mdp as deploy_mdp
from isaaclab_training.tasks.displayport_insertion.displayport_insertion_env_cfg import (
    SOCKET_INSERTION_OFFSET,
    ObservationsCfg,
)
from isaaclab_training.mdp.noise_models import ResetSampledConstantNoiseModelCfg
from isaaclab_tasks.utils import PresetCfg, preset

from .task_space_env_cfg import Rizon4sTaskSpaceDisplayportInsertionEnvCfg, TaskSpaceEventCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

_ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
_OSC_POSITION_SCALE = (0.025, 0.025, 0.010)
_OSC_ORIENTATION_SCALE = 0.025
# Full inertial decoupling makes these closed-loop task-acceleration gains [s^-2].
# Equal values match the ideal translational and rotational response rates.
_OSC_STIFFNESS = (300.0,) * 6
_OSC_DAMPING_RATIO = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
_NEWTON_NUM_ENVS = 256
_NEWTON_MAX_TRIANGLE_PAIRS = 2**25
# ``latest`` contact matching enables deterministic contact reduction. Newton
# packs reducer indices into 20-bit IDs, so this is the largest valid shared
# capacity; the 256-environment gate must remain free of overflow warnings.
_VBD_MAX_TRIANGLE_PAIRS = (1 << 20) - 1
_VBD_BODY_CONTACT_BUFFER_SIZE = 1024
_FLANGE_FALLBACK_DENSITY = 1000.0

_LOGGER = logging.getLogger(__name__)


def _compensate_articulation_body_gravity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
) -> None:
    """Cancel gravity on articulation bodies for solvers without MuJoCo gravcomp."""
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(asset.num_instances, device=asset.device, dtype=torch.int32)
    else:
        env_ids = env_ids.to(device=asset.device, dtype=torch.int32)

    body_mass = asset.data.body_mass.torch[env_ids]
    if asset_cfg.body_ids is not None:
        body_mass = body_mass[:, asset_cfg.body_ids]
    gravity_w = torch.tensor(gravity, device=asset.device, dtype=body_mass.dtype)
    forces_w = -body_mass.unsqueeze(-1) * gravity_w
    asset.permanent_wrench_composer.add_forces_and_torques_index(
        forces=forces_w,
        body_ids=asset_cfg.body_ids,
        env_ids=env_ids,
        is_global=True,
    )


def _use_explicit_effort_control_arm_actuators(
    env_cfg: Rizon4sTaskSpaceDisplayportInsertionEnvCfg,
) -> None:
    """Use zero-gain explicit arm actuators to clamp OSC joint efforts."""
    for actuator_name in ("shoulder", "elbow", "wrist"):
        source_cfg = env_cfg.scene.robot.actuators[actuator_name]
        actuator_effort_limit = source_cfg.actuator_effort_limit
        if actuator_effort_limit is None:
            actuator_effort_limit = source_cfg.joint_effort_limit
        actuator_velocity_limit = source_cfg.actuator_velocity_limit
        if actuator_velocity_limit is None:
            actuator_velocity_limit = source_cfg.joint_velocity_limit
        if actuator_effort_limit is None or source_cfg.joint_effort_limit is None:
            raise ValueError(f"Newton OSC arm actuator {actuator_name!r} must define effort limits.")
        if actuator_velocity_limit is None or source_cfg.joint_velocity_limit is None:
            raise ValueError(f"Newton OSC arm actuator {actuator_name!r} must define velocity limits.")
        env_cfg.scene.robot.actuators[actuator_name] = IdealPDActuatorCfg(
            joint_names_expr=list(source_cfg.joint_names_expr),
            actuator_effort_limit=actuator_effort_limit,
            actuator_velocity_limit=actuator_velocity_limit,
            joint_effort_limit=source_cfg.joint_effort_limit,
            joint_velocity_limit=source_cfg.joint_velocity_limit,
            stiffness=0.0,
            damping=0.0,
            armature=source_cfg.armature,
            friction=source_cfg.friction,
            dynamic_friction=source_cfg.dynamic_friction,
            viscous_friction=source_cfg.viscous_friction,
        )


def _uniform_sphere_inertia(mass: float, density: float = _FLANGE_FALLBACK_DENSITY) -> tuple[float, float, float]:
    """Return the diagonal inertia for a uniform sphere with the given mass."""
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"Flange mass must be positive and finite, got {mass!r}.")
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError(f"Fallback density must be positive and finite, got {density!r}.")
    radius = (3.0 * mass / (4.0 * math.pi * density)) ** (1.0 / 3.0)
    moment = 0.4 * mass * radius**2
    return (moment, moment, moment)


def _is_valid_diagonal_inertia(diagonal: object) -> bool:
    """Return whether a diagonal inertia is finite, positive, and physically realizable."""
    if diagonal is None:
        return False
    values = tuple(float(value) for value in diagonal)
    if len(values) != 3 or not all(math.isfinite(value) and value > 0.0 for value in values):
        return False
    return all(values[index] <= values[(index + 1) % 3] + values[(index + 2) % 3] for index in range(3))


def _is_valid_principal_axes(quaternion: object) -> bool:
    """Return whether a principal-axis quaternion is finite and normalized."""
    if quaternion is None:
        return False
    if hasattr(quaternion, "GetReal") and hasattr(quaternion, "GetImaginary"):
        values = (float(quaternion.GetReal()), *(float(value) for value in quaternion.GetImaginary()))
    else:
        values = tuple(float(value) for value in quaternion)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        return False
    return math.isclose(sum(value * value for value in values), 1.0, rel_tol=1.0e-5, abs_tol=1.0e-5)


def _flatten_rizon_gripper_mimic_chains(root_prim) -> tuple[str, ...]:
    """Retarget the Rizon gripper's chained mimics to their equivalent direct driver."""
    from pxr import Sdf

    root_path = root_prim.GetPath()
    driver_path = root_path.AppendPath(Sdf.Path("Grav_gripper/finger_joint"))
    intermediate_path = root_path.AppendPath(Sdf.Path("Grav_gripper/gripper_base/right_outer_knuckle_joint"))
    follower_specs = (
        ("Grav_gripper/gripper_base/right_inner_knuckle_joint", -1.0),
        ("Grav_gripper/right_outer_knuckle/right_outer_finger_joint", 1.0),
    )
    stage = root_prim.GetStage()

    def _mimic_state(joint_path: Sdf.Path, default_gearing: float) -> tuple[object, Sdf.Path, float, float]:
        joint = stage.GetPrimAtPath(joint_path)
        if not joint.IsValid():
            raise ValueError(f"Rizon USD is missing required gripper joint {joint_path}.")
        relationship = joint.GetRelationship("physxMimicJoint:rotX:referenceJoint")
        targets = relationship.GetTargets() if relationship else []
        if len(targets) != 1:
            raise ValueError(
                f"Rizon gripper joint {joint_path} must have exactly one mimic reference, got {targets!r}."
            )
        gearing_attr = joint.GetAttribute("physxMimicJoint:rotX:gearing")
        offset_attr = joint.GetAttribute("physxMimicJoint:rotX:offset")
        gearing = float(gearing_attr.Get()) if gearing_attr and gearing_attr.HasValue() else default_gearing
        offset = float(offset_attr.Get()) if offset_attr and offset_attr.HasValue() else 0.0
        return relationship, targets[0], gearing, offset

    _, intermediate_target, intermediate_gearing, intermediate_offset = _mimic_state(intermediate_path, -1.0)
    if (
        intermediate_target != driver_path
        or not math.isclose(intermediate_gearing, -1.0)
        or not math.isclose(intermediate_offset, 0.0)
    ):
        raise ValueError(
            "Rizon right-outer-knuckle mimic no longer matches the validated gripper chain: "
            f"target={intermediate_target}, gearing={intermediate_gearing}, offset={intermediate_offset}."
        )

    rewired = []
    for relative_path, expected_gearing in follower_specs:
        follower_path = root_path.AppendPath(Sdf.Path(relative_path))
        relationship, target, gearing, offset = _mimic_state(follower_path, expected_gearing)
        if not math.isclose(gearing, expected_gearing) or not math.isclose(offset, 0.0):
            raise ValueError(
                f"Rizon gripper mimic {follower_path} no longer matches the validated coefficients: "
                f"gearing={gearing}, offset={offset}."
            )
        if target == driver_path:
            continue
        if target != intermediate_path:
            raise ValueError(
                f"Rizon gripper mimic {follower_path} must reference {intermediate_path} or {driver_path}, got {target}."
            )
        relationship.SetTargets([driver_path])
        rewired.append(str(follower_path))

    flattened = tuple(rewired)
    if flattened:
        _LOGGER.info("Flattened Rizon gripper mimic chains for Newton: %s", flattened)
    return flattened


@clone
def _spawn_rizon_with_validated_flange_inertia(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn Rizon and author the deterministic Newton fallback inertia when needed."""
    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)

    from pxr import Gf, UsdPhysics

    flange = prim.GetStage().GetPrimAtPath(prim.GetPath().AppendChild("flange"))
    if not flange.IsValid():
        raise ValueError(f"Rizon USD {cfg.usd_path!r} does not contain the required flange prim.")
    mass_api = UsdPhysics.MassAPI(flange)
    if not mass_api:
        raise ValueError(f"Rizon flange {flange.GetPath()} must have UsdPhysics.MassAPI.")

    diagonal = mass_api.GetDiagonalInertiaAttr().Get()
    principal_axes = mass_api.GetPrincipalAxesAttr().Get()
    if _is_valid_diagonal_inertia(diagonal):
        if not _is_valid_principal_axes(principal_axes):
            raise ValueError(f"Rizon flange {flange.GetPath()} must have normalized principal axes.")
    else:
        mass = mass_api.GetMassAttr().Get()
        diagonal = _uniform_sphere_inertia(float(mass) if mass is not None else float("nan"))
        mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*diagonal))
        mass_api.GetPrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        _LOGGER.info("Authored deterministic flange inertia %s kg*m^2 for Newton asset %s.", diagonal, cfg.usd_path)

    _flatten_rizon_gripper_mimic_chains(prim)
    return prim


def _validate_sdf_meshes(root_prim, relative_paths: tuple[str, ...]) -> None:
    """Fail when a release asset no longer contains an expected colliding SDF mesh."""
    from pxr import UsdGeom, UsdPhysics

    missing = []
    for relative_path in relative_paths:
        mesh_path = f"{root_prim.GetPath()}{relative_path}"
        mesh_prim = root_prim.GetStage().GetPrimAtPath(mesh_path)
        if not mesh_prim.IsValid() or not mesh_prim.IsA(UsdGeom.Mesh) or not UsdPhysics.CollisionAPI(mesh_prim):
            missing.append(mesh_path)
    if missing:
        raise ValueError(f"DisplayPort asset is missing required colliding SDF meshes: {missing!r}.")


def _block_invalid_mesh_normals(root_prim) -> tuple[str, ...]:
    """Block malformed mesh normals so Newton can reconstruct them from topology."""
    from pxr import Usd, UsdGeom

    blocked_paths = []
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        expected_counts = {
            UsdGeom.Tokens.constant: 1,
            UsdGeom.Tokens.uniform: len(mesh.GetFaceVertexCountsAttr().Get() or ()),
            UsdGeom.Tokens.vertex: len(mesh.GetPointsAttr().Get() or ()),
            UsdGeom.Tokens.varying: len(mesh.GetPointsAttr().Get() or ()),
            UsdGeom.Tokens.faceVarying: len(mesh.GetFaceVertexIndicesAttr().Get() or ()),
        }

        primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar("normals")
        primvar_values = primvar.Get() if primvar else None
        if primvar_values is not None:
            interpolation = primvar.GetInterpolation()
            indices = primvar.GetIndices() if primvar.IsIndexed() else None
            expanded_count = len(indices) if indices else len(primvar_values)
            indices_valid = not indices or all(0 <= index < len(primvar_values) for index in indices)
            if expanded_count != expected_counts.get(interpolation) or not indices_valid:
                primvar.GetAttr().Block()
                indices_attr = primvar.GetIndicesAttr()
                if indices_attr:
                    indices_attr.Block()
                blocked_paths.append(str(prim.GetPath()))
                continue

        normals_attr = mesh.GetNormalsAttr()
        normals = normals_attr.Get() if normals_attr else None
        interpolation = mesh.GetNormalsInterpolation()
        if normals is not None and len(normals) != expected_counts.get(interpolation):
            normals_attr.Block()
            blocked_paths.append(str(prim.GetPath()))

    blocked = tuple(blocked_paths)
    if blocked:
        _LOGGER.warning("Blocked malformed normals on DisplayPort meshes so Newton can reconstruct them: %s", blocked)
    return blocked


@clone
def _spawn_plug_with_validated_sdf_meshes(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the plug, sanitize visual normals, and validate the point-SDF mesh contract."""
    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    _block_invalid_mesh_normals(prim)
    _validate_sdf_meshes(prim, ("/collision_mesh",))
    return prim


@clone
def _spawn_socket_with_validated_sdf_meshes(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the socket, sanitize visual normals, and validate the point-SDF mesh contract."""
    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    _block_invalid_mesh_normals(prim)
    _validate_sdf_meshes(
        prim,
        tuple(f"/tn__2584N111_DisplayportCord_jP/Body{body_id}/Mesh" for body_id in (5, 6, 8, 12, 13)),
    )
    return prim


def _newton_sdf_properties(
    contact_offset: float,
    rest_offset: float,
    sdf_prim_paths: tuple[str, ...],
    contact_priority: int | None = None,
) -> dict[str, list[PhysxCollisionCfg | MujocoCollisionCfg | NewtonCollisionCfg | NewtonSDFCollisionCfg]]:
    """Create point-SDF properties only for meshes authored as SDF colliders."""
    properties: dict[str, list[PhysxCollisionCfg | MujocoCollisionCfg | NewtonCollisionCfg | NewtonSDFCollisionCfg]] = {
        "/.*": [PhysxCollisionCfg(contact_offset=contact_offset, rest_offset=rest_offset)]
    }
    for prim_path in sdf_prim_paths:
        properties[prim_path] = [
            *([MujocoCollisionCfg(priority=contact_priority)] if contact_priority is not None else []),
            NewtonCollisionCfg(contact_margin=0.0, contact_gap=0.005),
            NewtonSDFCollisionCfg(
                sdf_max_resolution=256,
                sdf_narrow_band_inner=-0.005,
                sdf_narrow_band_outer=0.005,
                sdf_texture_format="uint16",
                sdf_padding=0.005,
                hydroelastic_enabled=False,
                hydroelastic_stiffness=1.0e8,
            ),
        ]
    return properties


@configclass
class DisplayportNewtonPhysicsCfg(PresetCfg):
    """Newton point-SDF physics profile for DisplayPort insertion."""

    newton_sdf: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=8192,
            nconmax=8192,
            iterations=100,
            ls_iterations=50,
            update_data_interval=10,
            impratio=10.0,
            cone="elliptic",
            ccd_iterations=35,
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(
            reduce_contacts=True,
            # Scene-wide candidate-pair capacity for the 256-environment default.
            # If this overflows, Newton warns and may omit candidate contacts;
            # increase it when increasing environment count or mesh complexity.
            max_triangle_pairs=_NEWTON_MAX_TRIANGLE_PAIRS,
        ),
        num_substeps=20,
        collision_decimation=10,
        default_shape_cfg=NewtonShapeCfg(gap=0.005),
        debug_mode=False,
        use_cuda_graph=True,
    )
    newton_vbd: NewtonCfg = NewtonCfg(
        solver_cfg=VBDSolverCfg(
            iterations=20,
            rigid_compliant_alm=True,
            rigid_contact_history=True,
            rigid_body_contact_buffer_size=_VBD_BODY_CONTACT_BUFFER_SIZE,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(
            reduce_contacts=True,
            contact_matching="latest",
            max_triangle_pairs=_VBD_MAX_TRIANGLE_PAIRS,
        ),
        num_substeps=40,
        collision_decimation=1,
        default_shape_cfg=NewtonShapeCfg(gap=0.005, ke=2500.0, kd=50.0),
        debug_mode=False,
        use_cuda_graph=True,
    )
    default: NewtonCfg = newton_sdf


@configclass
class NewtonTaskSpaceObservationsCfg:
    """Checkpoint-compatible actor and critic observations for Newton OSC."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations in the 18-dimensional checkpoint ABI order."""

        socket_pos = ObsTerm(
            func=deploy_mdp.rigid_object_pos_w,
            params={"asset_cfg": SceneEntityCfg("dp_socket"), "offset": SOCKET_INSERTION_OFFSET},
            noise=ResetSampledConstantNoiseModelCfg(
                noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01, operation="add")
            ),
        )
        tool_pos = ObsTerm(
            func=deploy_mdp.eef_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "flange", "offset": [0.0, 0.0, 0.0]},
        )
        tool_rot_6d = ObsTerm(
            func=deploy_mdp.eef_rot_6d_w,
            params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "flange"},
        )
        socket_rot_6d = ObsTerm(
            func=deploy_mdp.rigid_object_rot_6d_w,
            params={"asset_cfg": SceneEntityCfg("dp_socket")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    # Keep every robot joint in the privileged critic to preserve the
    # checkpoint's 40-dimensional critic input.
    critic: ObservationsCfg.CriticCfg = ObservationsCfg.CriticCfg()


@configclass
class NewtonTaskSpaceEventCfg(TaskSpaceEventCfg):
    """Newton-specific material and arm-friction randomization events."""

    randomize_arm_joint_friction = EventTerm(
        func=env_mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=_ARM_JOINTS),
            "friction_distribution_params": (0.0, 0.15),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    # OSC is torque controlled, so position-controller gain randomization is
    # intentionally disabled while joint-friction randomization remains active.
    randomize_arm_pd_gains: EventTerm | None = None

    robot_body_gravity_compensation = preset(
        default=None,
        newton_vbd=EventTerm(
            func=_compensate_articulation_body_gravity,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
    )


@configclass
class Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg(Rizon4sTaskSpaceDisplayportInsertionEnvCfg):
    """Newton point-SDF OSC training configuration.

    The actor contract is ``socket_pos, flange_pos, flange_rot_6d,
    socket_rot_6d``. This differs from the TCP-first PhysX task-space contract,
    so the configurations must not share checkpoints despite both being 18-D.

    Override ``scene.robot.spawn.usd_path`` with a USD calibrated for the
    robot that will execute the policy.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # 100 Hz outer simulation ticks, 20 solver substeps, collision every 10
        # substeps, and one policy action every three outer ticks give 2 kHz,
        # 200 Hz, and 33.3 Hz respectively.
        self.sim.dt = 0.01
        self.sim.physics = DisplayportNewtonPhysicsCfg()
        # Use Newton-native actuator execution. Zero joint gains preserve direct
        # OSC effort control while effort saturation and solver limits remain active.
        self.sim.use_newton_actuators = True
        # The collision candidate-pair capacity is scene-wide and supports this
        # per-rank default. Scale the capacity when increasing this value.
        self.scene.num_envs = _NEWTON_NUM_ENVS
        self.decimation = 3
        self.sim.render_interval = self.decimation

        # Preserve source PhysX offsets on every collider, but apply Newton's
        # point-SDF schema only to meshes authored for SDF collision.
        self.scene.robot.spawn.func = _spawn_rizon_with_validated_flange_inertia
        self.scene.dp_plug.spawn.func = _spawn_plug_with_validated_sdf_meshes
        self.scene.dp_socket.spawn.func = _spawn_socket_with_validated_sdf_meshes
        self.scene.dp_plug.spawn.collision_props = _newton_sdf_properties(
            0.00001,
            -0.00005,
            ("/collision_mesh",),
        )
        self.scene.dp_socket.spawn.collision_props = _newton_sdf_properties(
            0.0001,
            -0.0001,
            tuple(f"/tn__2584N111_DisplayportCord_jP/Body{body_id}/Mesh" for body_id in (5, 6, 8, 12, 13)),
            # The socket parameters must govern contacts with lower-priority
            # colliders instead of inheriting the plug's grasp friction.
            contact_priority=1,
        )

        self.observations = NewtonTaskSpaceObservationsCfg()
        self.task_space_obs_order = ["socket_pos", "tool_pos", "tool_rot_6d", "socket_rot_6d"]

        # RSL-RL clips raw actor outputs to +/-1 before OSC applies these scales.
        # Keep the action-term clip unset so the checkpoint contract has one
        # effective clipping stage and cannot acquire a latent +/-0.5 limit.
        self.actions.arm_action = deploy_mdp.DeployOperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=_ARM_JOINTS,
            body_name="flange",
            body_offset=deploy_mdp.DeployOperationalSpaceControllerActionCfg.OffsetCfg(),
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_rel"],
                impedance_mode="fixed",
                inertial_dynamics_decoupling=True,
                partial_inertial_dynamics_decoupling=False,
                gravity_compensation=False,
                motion_stiffness_task=_OSC_STIFFNESS,
                motion_damping_ratio_task=_OSC_DAMPING_RATIO,
                nullspace_control="none",
            ),
            nullspace_joint_pos_target="none",
            clip=None,
            position_scale=_OSC_POSITION_SCALE,
            orientation_scale=_OSC_ORIENTATION_SCALE,
        )

        # Retain the fully wired grasp reset while adding Newton-only events.
        # The subclass owns the same base event fields, so replacing the group
        # does not change reward, curriculum, or domain-randomization behavior.
        grasp_reset = self.events.set_robot_to_grasp_pose
        newton_events = NewtonTaskSpaceEventCfg()
        newton_events.set_robot_to_grasp_pose = grasp_reset
        self.events = newton_events
        self.events.plug_physics_material.params["static_friction_range"] = (3.0, 3.0)
        self.events.plug_physics_material.params["dynamic_friction_range"] = (3.0, 3.0)
        self.events.robot_physics_material.params["static_friction_range"] = (1.0, 1.0)
        self.events.robot_physics_material.params["dynamic_friction_range"] = (1.0, 1.0)

        # MJWarp consumes body gravcomp directly. Full-scene VBD keeps world
        # gravity active and receives an equal-and-opposite body force from its
        # reset event because it does not consume MuJoCo gravcomp metadata.
        self.scene.robot.spawn.rigid_props = preset(
            default=MujocoRigidBodyPropertiesCfg(gravcomp=1.0),
            newton_vbd=RigidBodyBaseCfg(disable_gravity=False),
        )
        self.scene.robot.spawn.joint_drive_props = preset(
            default=MujocoJointDrivePropertiesCfg(actuatorgravcomp=False),
            newton_vbd=None,
        )
        _use_explicit_effort_control_arm_actuators(self)

        self.scene.robot.actuators["gripper_drive"] = ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=200.0,
            velocity_limit_sim=2.0,
            stiffness=2000.0,
            damping=10.0,
            friction=0.0,
            armature=0.1,
        )
        self.scene.robot.actuators["gripper_passive"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_knuckle_joint", ".*_outer_finger_joint"],
            effort_limit_sim=20.0,
            velocity_limit_sim=1.0,
            stiffness=2000.0,
            damping=10.0,
            friction=0.0,
            armature=0.05,
        )
        self.hand_hold_width = -0.1
        self.hand_close_width = -0.1


@configclass
class Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg_PLAY(Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg):
    """Deterministic play configuration for a trained Newton OSC policy."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.reset_plug_curriculum.params["at_goal_prob"] = 0.0
        self.events.reset_plug_curriculum.params["at_goal_prob_final"] = 0.0
