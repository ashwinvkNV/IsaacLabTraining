IsaacLab Training
=================

Reusable downstream Isaac Lab training repository for custom task formulations,
shared MDP terms, and policy export workflows.

Quick Start
-----------

Install the standalone training repo and its pinned Isaac Lab dependency:

.. code-block:: bash

   git clone https://github.com/ashwinvkNV/IsaacLabTraining.git
   cd IsaacLabTraining
   uv sync

Verify that Isaac Lab discovers the downstream tasks:

.. code-block:: bash

   uv run python -c 'import gymnasium as gym; import isaaclab_training.tasks; print([s.id for s in gym.registry.values() if s.id.startswith("IsaacTraining-")])'

Run a small visual training smoke test:

.. code-block:: bash

   uv sync --group sim

   uv run isaaclab train --rl_library rsl_rl \
     --task IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-ROS-Inference \
     --num_envs 16 \
     --max_iterations 1 \
     --visualizer kit

See :doc:`setup_training` for the complete install, source-checkout, and
headless-training workflow.

Trainable Policies
------------------

This repository currently packages DisplayPort cable insertion policies for the
Flexiv Rizon 4s arm with a Grav parallel gripper.

.. list-table::
   :widths: 35 45 20
   :header-rows: 1

   * - Policy
     - Task ID
     - Notes
   * - DisplayPort task-space insertion
     - ``IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-ROS-Inference``
     - Recommended for real-robot deployment.
   * - DisplayPort task-space insertion with Newton
     - ``IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton-ROS-Inference``
     - Recommended Newton 1.6 MJWarp profile for calibrated-robot training.
   * - DisplayPort joint-space insertion
     - ``IsaacTraining-DisplayPortInsertion-Rizon4s-Joint-NoJointVel-ROS-Inference``
     - Deployable joint-position policy without joint velocity in actor observations.
   * - DisplayPort joint-space insertion with velocity
     - ``IsaacTraining-DisplayPortInsertion-Rizon4s-Joint-ROS-Inference``
     - Useful for experiments when joint velocity is available.

Use the matching ``-Play`` tasks for visualization and evaluation, and the plain
non-ROS-inference tasks for ablations you do not intend to deploy directly.

Repository Architecture
-----------------------

The repository separates reusable Isaac Lab infrastructure from concrete task
formulations.

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Path
     - Purpose
   * - ``src/isaaclab_training/mdp/``
     - Shared actions, observations, rewards, events, terminations, and noise models.
   * - ``src/isaaclab_training/tasks/``
     - Registered task formulations. DisplayPort insertion is the first task.
   * - ``src/isaaclab_training/cli/``
     - Policy export and policy I/O validation commands.
   * - ``src/isaaclab_training/export/``
     - LEAPP export helpers used by the generic exporter.
   * - ``docs/tasks/``
     - Long-form task tutorials and deployment notes.
   * - ``tests/``
     - Lightweight checks for task registration and export-specific tensor semantics.

.. toctree::
   :maxdepth: 2
   :caption: Guides
   :hidden:

   setup_training
   repo_structure

.. toctree::
   :maxdepth: 2
   :caption: Tasks
   :hidden:

   tasks/displayport_insertion
