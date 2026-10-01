# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawner configurations used by the training tasks."""

from __future__ import annotations

from collections.abc import Callable

from isaaclab.sim import UsdFileCfg
from isaaclab.utils.configclass import configclass


@configclass
class UsdFileWithMassOverrideCfg(UsdFileCfg):
    """:class:`UsdFileCfg` with optional explicit mass properties and removal of disabled colliders.

    With the defaults this spawns exactly like :class:`UsdFileCfg`.
    """

    func: Callable | str = "isaaclab_training.utils.spawners:spawn_usd_with_mass_override"

    deactivate_disabled_colliders: bool = False
    """Deactivate collider prims with ``physics:collisionEnabled = False`` before cloning.

    PhysX includes such shapes when it derives the centre of mass and inertia, so on a dynamic body
    set the explicit mass properties below as well to keep its dynamics unchanged.
    """

    center_of_mass: tuple[float, float, float] | None = None
    """Centre of mass in the rigid body's frame [m]. Set together with the two fields below."""

    principal_axes_wxyz: tuple[float, float, float, float] | None = None
    """Orientation (w, x, y, z) of the principal axes of inertia relative to the rigid body frame."""

    diagonal_inertia: tuple[float, float, float] | None = None
    """Principal moments of inertia [kg m^2] about :attr:`principal_axes_wxyz`."""
