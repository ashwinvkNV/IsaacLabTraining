# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral compatibility checks for the locked Newton and MuJoCo-Warp stack."""

import warp as wp
from newton._src.solvers.mujoco.kernels import create_convert_mjw_contacts_to_newton_kernel


def test_newton_mjwarp_contact_conversion_kernel_compiles_with_locked_dependencies(monkeypatch, tmp_path):
    """The Newton contact bridge must compile against MuJoCo-Warp's contact-force ABI."""
    monkeypatch.setattr(wp.config, "kernel_cache_dir", str(tmp_path))
    kernel = create_convert_mjw_contacts_to_newton_kernel()

    module_exec = kernel.module.load(wp.get_device("cpu"))

    assert module_exec is not None
