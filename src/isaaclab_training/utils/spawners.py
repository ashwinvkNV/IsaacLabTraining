# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD spawner with optional explicit mass properties and removal of disabled colliders.

Imported lazily through the ``func`` string of
:class:`~isaaclab_training.utils.spawners_cfg.UsdFileWithMassOverrideCfg`, i.e. only once the simulation app is running.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.utils import clone
from pxr import Gf, Usd, UsdPhysics

if TYPE_CHECKING:
    from .spawners_cfg import UsdFileWithMassOverrideCfg


@clone
def spawn_usd_with_mass_override(
    prim_path: str,
    cfg: UsdFileWithMassOverrideCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a USD asset, optionally author explicit mass properties and drop disabled colliders.

    PhysX derives a rigid body's centre of mass and inertia from all of its collision shapes, including
    shapes with ``physics:collisionEnabled = False``. Authoring the centre of mass, principal axes and
    principal moments explicitly makes PhysX use them as given, after which the disabled shapes can be
    deactivated without changing the body's dynamics. Both steps run on the prototype prim before
    :func:`clone` copies it to the other environments.
    """
    prim = spawn_from_usd.__wrapped__(prim_path, cfg, translation, orientation, **kwargs)

    if cfg.center_of_mass is not None:
        if cfg.principal_axes_wxyz is None or cfg.diagonal_inertia is None:
            raise ValueError("center_of_mass, principal_axes_wxyz and diagonal_inertia must be set together.")
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*cfg.center_of_mass))
        mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(*cfg.principal_axes_wxyz))
        mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*cfg.diagonal_inertia))

    if cfg.deactivate_disabled_colliders:
        disabled = []
        for child in Usd.PrimRange(prim):
            attr = child.GetAttribute("physics:collisionEnabled")
            if attr and attr.HasAuthoredValue() and attr.Get() is False:
                disabled.append(child)
        for child in disabled:
            child.SetActive(False)
    return prim
