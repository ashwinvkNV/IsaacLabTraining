"""Task registrations exposed to the Isaac Lab CLI."""

import os

# Let the PyTorch CUDA caching allocator grow segments in place instead of keeping one cached block per
# size. The recurrent PPO update allocates differently sized buffers every minibatch, which otherwise
# leaves several GiB reserved but unused. Allocator behaviour only; computation is unchanged. Must be set
# before the first CUDA allocation, which is why it lives in the module the CLI imports to register tasks.
# An explicit PYTORCH_CUDA_ALLOC_CONF set by the user takes precedence.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from isaaclab_tasks.utils import import_packages  # noqa: E402

import_packages(__name__, ["isaaclab_training.mdp"])
