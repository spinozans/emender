"""Dense single-head algebra oracle, not an index, trained layer or serving kernel.

Controls come from the passage alone. ``write`` is the already gated value w*v.
Only the input memory varies when applying a compiled span to a live seed.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Effect:
    a: Tensor  # [key, key]
    b: Tensor  # [key, value]; the zero-seeded terminal state

    def apply(self, seed: Tensor) -> Tensor:
        return self.a @ seed + self.b

    def after(self, earlier: "Effect") -> "Effect":
        """Sequence composition: tokens in earlier, THEN tokens in self."""
        return Effect(self.a @ earlier.a, self.apply(earlier.b))


def compile_gdn2(key: Tensor, decay: Tensor, erase: Tensor, write: Tensor) -> Effect:
    """Fixed-control GDN-2 core: decay first, then gated delta update."""
    if key.ndim != 2 or not key.shape == decay.shape == erase.shape:
        raise ValueError("key, decay and erase must have shape [time, key]")
    if write.ndim != 2 or len(write) != len(key):
        raise ValueError("write must have shape [time, value]")
    eye = torch.eye(key.shape[1], dtype=key.dtype, device=key.device)
    result = Effect(eye, key.new_zeros((key.shape[1], write.shape[1])))
    for k, d, e, u in zip(key, decay, erase, write, strict=True):
        step = Effect((eye - torch.outer(k, e * k)) * d[None, :], torch.outer(k, u))
        result = step.after(result)
    return result


def merged_read(
    effects: list[Effect], mass: Tensor, seed: Tensor, query: Tensor
) -> Tensor:
    """Parallel alternatives: normalize support, mix maps, then read once."""
    if (
        mass.shape != (len(effects),)
        or not torch.isfinite(mass).all()
        or (mass < 0).any()
    ):
        raise ValueError("expected one finite nonnegative mass per hit")
    if not effects or mass.max() == 0:
        return seed.new_zeros(seed.shape[-1])  # no hint, NOT an identity passage
    # Mass is a relative weight. A large common factor need not overflow the sum.
    scaled = mass / mass.max().detach()
    p = scaled / scaled.sum()
    a = torch.einsum("i,ijk->jk", p, torch.stack([h.a for h in effects]))
    b = torch.einsum("i,ijk->jk", p, torch.stack([h.b for h in effects]))
    return query @ (a @ seed + b)
