# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for import-safe DisplayPort solver diagnostic helpers."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab_training.cli.displayport_solver_sanity import (
    _newton_fk_consistency_snapshot,
    _pose_residual_snapshot,
    _vbd_structural_snapshot,
)


def test_pose_residual_snapshot_reports_largest_body_errors():
    """The pose helper must preserve the labels of the largest translation and rotation errors."""
    reference = torch.tensor(
        (
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        )
    )
    candidate = torch.tensor(
        (
            (0.25, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
            (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        )
    )

    snapshot = _pose_residual_snapshot(reference, candidate, ["translated", "rotated"])

    assert snapshot["maximum_translation_m"] == pytest.approx(0.25)
    assert snapshot["maximum_translation_body"] == "translated"
    assert snapshot["maximum_rotation_rad"] == pytest.approx(torch.pi)
    assert snapshot["maximum_rotation_body"] == "rotated"


@pytest.mark.parametrize(
    ("candidate", "labels", "match"),
    (
        (torch.zeros((2, 6)), ["a", "b"], "matching.*shapes"),
        (torch.zeros((2, 7)), ["a"], "Expected 2 labels"),
    ),
)
def test_pose_residual_snapshot_rejects_invalid_inputs(candidate, labels, match):
    """Malformed pose arrays and label lists must fail with actionable errors."""
    with pytest.raises(ValueError, match=match):
        _pose_residual_snapshot(torch.zeros((2, 7)), candidate, labels)


def test_newton_snapshots_report_unavailable_without_runtime_state():
    """Non-Newton or uninitialized environments must produce explicit unavailable telemetry."""
    raw_env = SimpleNamespace(sim=SimpleNamespace(physics_manager=SimpleNamespace()))

    assert _newton_fk_consistency_snapshot(raw_env) == {"available": False}
    assert _vbd_structural_snapshot(raw_env) == {"available": False}
