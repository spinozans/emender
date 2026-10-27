"""Device-resident finite-gradient diagnostics for the steady training step."""
import torch


class FiniteGradients:
    """Accumulate failure on device; the caller checks at a loop boundary.

    Clipping arithmetic is the native PyTorch implementation. A nonfinite norm
    poisons this run and is reported at check(), never silently certified.
    """
    def __init__(self, device):
        self.ok = torch.ones((), device=device, dtype=torch.bool)

    def clip(self, parameters):
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=False)
        self.ok.logical_and_(torch.isfinite(norm))
        return norm

    def check(self):
        if not self.ok.item():
            raise RuntimeError('nonfinite gradient norm in steady training loop')
