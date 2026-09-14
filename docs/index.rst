IsaacLab Training
=================

Reusable downstream Isaac Lab training repository for custom task formulations,
shared MDP terms, and policy export workflows.

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

   repo_structure

.. toctree::
   :maxdepth: 2
   :caption: Tasks
   :hidden:

   tasks/displayport_insertion
