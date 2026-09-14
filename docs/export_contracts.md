# Export Contracts

The generic exporter handles Isaac Lab app launch, Hydra task config resolution, RSL-RL checkpoint loading, recurrent-state tracing, and LEAPP graph compilation.

Task-specific I/O semantics are isolated behind named export contracts in `isaaclab_training.export.contracts`.

Current contract:

```text
displayport_task_space
```

Use it with:

```bash
uv run --group leapp isaaclab-training-export \
  --task IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-ROS-Inference \
  --checkpoint logs/rsl_rl/displayport_insertion_rizon4s/<run>/model_<n>.pt \
  --contract displayport_task_space
```

To add a contract:

1. Implement the `ExportContract` protocol.
2. Define how observations are annotated before the policy call.
3. Define how actions are post-processed and annotated after the policy call.
4. Add the contract to `CONTRACTS`.
5. Add unit tests for tensor slicing, scaling, and registry lookup.
