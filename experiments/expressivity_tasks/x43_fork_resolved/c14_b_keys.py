"""Parameter-free tail coordinates for B1 (not a shared-store implementation).

Both stream and suffix callers must use this constructor. Full suffix identity
is retained separately for ownership; truncated coordinates cannot own slots.
"""
import torch


def tail_coordinates(observed: bytes, *, device=None):
    """Raw byte magnitudes in right-aligned float32 coordinates, reset locally."""
    if not isinstance(observed, bytes):
        raise TypeError('key context must be raw bytes')
    tail = observed[-64:]
    coordinates = torch.zeros(64, dtype=torch.float32, device=device)
    if tail:
        coordinates[-len(tail):] = torch.tensor(list(tail), dtype=torch.float32,
                                              device=device)
    return coordinates


def frozen_tail_key(observed: bytes, geometry):
    """Apply the standing frozen map, without trainable contextual input."""
    if any(parameter.requires_grad for parameter in geometry.parameters()):
        raise ValueError('key geometry must be frozen')
    parameter = next(geometry.parameters())
    with torch.no_grad(), torch.autocast(device_type=parameter.device.type,
                                        enabled=False):
        return geometry(tail_coordinates(observed, device=parameter.device))


def array_key(suffix: bytes, geometry):
    """None means skip APPEND, including for a frozen map with nonzero bias."""
    if not isinstance(suffix, bytes):
        raise TypeError('key suffix must be raw bytes')
    return frozen_tail_key(suffix, geometry) if suffix else None
