"""B1 native key handoff and raw suffix metadata; not a store implementation.

No suffix-to-vector map exists. The caller must pass the actual native stream
key computed at the query position, before adding consultation content. This
module neither changes the dense Cell nor installs a production B1 caller.
"""
from dataclasses import dataclass
import torch
from .c14_b_suffix import SuffixOwner


@dataclass(frozen=True)
class ArrayAppend:
    key: torch.Tensor
    owner: SuffixOwner


def array_append(native_key: torch.Tensor, owner: SuffixOwner):
    """Reuse native addressing; full suffix bytes serve only slot identity.

    One invocation describes one slot. Keep native shape/dtype/device and
    storage without copying, normalizing, padding or recomputing its key.
    Native keys are detached at construction, so reject a trainable input
    rather than silently hiding a wrongly wired consultation key path.
    """
    if not isinstance(owner, SuffixOwner) or not isinstance(owner.suffix, bytes):
        raise TypeError('slot owner must contain raw suffix bytes')
    if not owner.suffix:
        return None
    if not isinstance(native_key, torch.Tensor):
        raise TypeError('native key must be a tensor')
    if native_key.requires_grad:
        raise ValueError('native APPEND key must already be detached')
    if native_key.shape != (64,) or not native_key.is_floating_point():
        raise ValueError('one native 64-dimensional key required')
    if not bool(torch.isfinite(native_key).all()):
        raise ValueError('nonfinite native key')
    return ArrayAppend(native_key, owner)
