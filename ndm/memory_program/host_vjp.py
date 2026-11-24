"""First-order CPU workbench / differentiable byte-host seam for a single exclusive gradient batch.

`encoder` is trusted pure causal code: one registered module owning all its parameters, no
external mutable state, custom random generators, caches, optimizer, DDP or concurrent calls.
It returns one FP32 [B,T,width+256] packet (causal features + base logits). This adapter checks
ordinary PyTorch module/tensor versions, RNG and exact replay; it cannot sandbox arbitrary code.
Caller must discard *controller* gradients if the transaction fails. No higher-order derivatives.
"""

from __future__ import annotations
from collections import Counter
import torch
from torch import nn


def _stamp(module):
    return (
        tuple((n, id(m), m.training) for n, m in module.named_modules()),
        tuple(
            (n, id(p), p._version, p.shape, p.dtype, p.device, p.data_ptr(), p.requires_grad)
            for n, p in (*module.named_parameters(), *module.named_buffers())
        ),
    )


class HostVJP:
    def __init__(self, encoder, *, width, microbatch, remat):
        params = list(encoder.parameters())
        devices = {p.device for p in params}
        if (
            not params
            or len(devices) != 1
            or next(iter(devices)).type not in ("cpu", "cuda")
            or not any(p.requires_grad for p in params)
            or any(p.grad is not None for p in params)
            or width < 1
            or microbatch < 1
            or not torch.is_grad_enabled()
            or torch.is_inference_mode_enabled()
        ):
            raise ValueError("fresh differentiable single-device host required")
        for m in encoder.modules():
            if m.training and (
                isinstance(m, nn.modules.dropout._DropoutNd)
                and m.p > 0
                or isinstance(m, nn.modules.batchnorm._BatchNorm)
                and m.track_running_stats
            ):
                raise ValueError("stochastic or stateful encoder")
        self.model = encoder
        self.device = next(iter(devices))
        self.width = width
        self.microbatch = microbatch
        self.remat = remat
        self.version = _stamp(encoder)
        self.rows = []
        self.state = "open"
        self.work = Counter()

    def _check(self):
        if _stamp(self.model) != self.version:
            raise ValueError("changed host version/mode")

    def _run(self, x, under_grad, kind):
        self._check()
        dev = self.device
        cpu_rng = torch.get_rng_state()
        gpu_rng = torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None
        with torch.set_grad_enabled(under_grad):
            packet = self.model(x.to(dev))
        if (
            not torch.equal(cpu_rng, torch.get_rng_state())
            or gpu_rng is not None
            and not torch.equal(gpu_rng, torch.cuda.get_rng_state(dev))
        ):
            raise ValueError("stochastic encoder changed RNG")
        self._check()
        if (
            packet.shape != (*x.shape, self.width + 256)
            or packet.dtype != torch.float32
            or packet.device != dev
            or packet.requires_grad != under_grad
        ):
            raise ValueError("invalid differentiable host packet")
        self.work[kind + "_tokens"] += x.numel()
        self.work[kind + "_batches"] += 1
        return packet

    def stage(self, ids):
        if self.state != "open":
            raise ValueError("closed host transaction")
        try:
            if (
                ids.ndim != 2
                or not ids.numel()
                or ids.dtype != torch.long
                or ids.device.type != "cpu"
                or (ids < 0).any()
                or (ids > 255).any()
            ):
                raise ValueError("expected observed CPU byte indices")
            parts = []
            for batch in ids.split(self.microbatch):
                x = batch.contiguous().clone()
                p = self._run(x, not self.remat, "forward").to("cpu", copy=True)
                if not torch.isfinite(p).all():
                    raise ValueError("nonfinite host output")
                if self.remat:
                    p = p.detach().requires_grad_()
                    self.rows.append((x, p, p._version))
                parts.append(p)
            return torch.cat(parts).split((self.width, 256), dim=-1)
        except BaseException:
            self.abort()
            raise

    def backward(self, objective):
        if self.state != "open" or not self.work["forward_tokens"]:
            raise ValueError("missing or closed host transaction")
        self.state = "consuming"
        try:
            self._check()
            if any(p.grad is not None for p in self.model.parameters()):
                raise ValueError("contaminated host gradients")
            if objective.ndim or not torch.isfinite(objective.detach()).item():
                raise ValueError("invalid objective")
            objective.backward()
            self._check()
            while self.rows:
                x, p, ver = self.rows.pop()
                if p._version != ver or p.grad is None or not p.grad.isfinite().all():
                    raise ValueError("missing/invalid staged cotangent")
                q = self._run(x, True, "recompute")
                if not torch.equal(q.detach().cpu(), p.detach()):
                    raise ValueError("host rematerialization changed packet")
                torch.autograd.backward(q, p.grad.to(self.device))
                del q, x, p
            gs = [p.grad for p in self.model.parameters() if p.grad is not None]
            if not gs or not torch.stack([g.isfinite().all() for g in gs]).all().item():
                raise ValueError("missing/nonfinite host gradients")
            self._check()
            self.state = "done"
        except BaseException:
            self.abort()
            raise

    def abort(self):
        self.rows.clear()
        self.model.zero_grad(set_to_none=True)
        self.state = "aborted"
