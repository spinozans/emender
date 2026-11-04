"""Causal suffix identities for B1 store ownership (no store implementation).

Only observed input bytes are accepted. Callers must retain separate owner
identities per independent window; neither targets nor another window belong
in this interface.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SuffixOwner:
    window: tuple[int, int]
    suffix: bytes

    def is_current(self, window: tuple[int, int], observed: bytes) -> bool:
        """Whether this answer still belongs to the running causal suffix."""
        return self.window == window and observed.endswith(self.suffix)


def queried_suffix(window: tuple[int, int], observed: bytes,
                   context_length: int) -> SuffixOwner:
    """Reconstruct the cached query without reading a shifted target.

    Zero-length queries own the empty suffix, which remains current until
    reset. Malformed lengths fail closed instead of silently clipping a cache
    identity and thereby consulting a different suffix.
    """
    if not isinstance(observed, bytes):
        raise TypeError('observed context must be raw bytes')
    if isinstance(context_length, bool) or not isinstance(context_length, int):
        raise TypeError('context length must be an integer')
    if not 0 <= context_length <= len(observed):
        raise ValueError('cached query crosses the independent window reset')
    if len(window) != 2 or any(isinstance(v, bool) or not isinstance(v, int)
                               or v < 0 for v in window):
        raise ValueError('window identity must be (document, offset)')
    suffix = observed[-context_length:] if context_length else b''
    return SuffixOwner(window, suffix)
