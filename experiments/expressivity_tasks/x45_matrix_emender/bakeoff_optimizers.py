"""OPTIMIZER/SCHEDULE BAKE-OFF -- the training-surface arms for the winner.

PROVENANCE (the supervisor's rule, stated first): nothing here reimplements a
paper.  The Muon is the SAME implementation the prior invertible line
receipted -- ``experiments/expressivity_tasks/hybrid_muon.py``, the pinned
KellerJordan/Muon quintic convention (Frobenius normalization + 1e-7, tall
matrices transposed, five (3.4445, -4.7750, 2.0315) iterations, FP32
momentum/NS arithmetic, upstream lerp momentum .05 + Nesterov .95, Moonshot
``.2 * sqrt(max(shape))`` update scaling) -- imported VERBATIM via
``newton_schulz`` / ``scaled_update``.  The schedule-free arm is the SAME
package that line used (``schedulefree.AdamWScheduleFree``, Meta's reference
implementation; the prior line's ``run_invertible_stage10_optimizers_8m.py``
imported exactly this class).  The only new code is (a) the parameter routing
for the x45 winner's names (the prior line's ``HIDDEN`` regex matched the
invertible model's names, not this model's) and (b) parameterizing the hybrid's
aux betas/weight-decay so the bake-off can hold them at the probe protocol's
own transferred values.

ROUTING (approved convention, "embeddings + projections"): Muon owns every 2D
embedding/linear-projection weight -- ``embed.weight``, ``head.weight``,
``blocks.N.mixer.{value,key,query_key,out}.weight``,
``blocks.N.mlp.{w1,w2,w3}.weight``; AdamW owns the rest (the RMSNorm weights
and the DECAY gate's ``mixer.decay.weight``/``.bias`` -- a gate, not a hidden
map).  NOTE, stated: this differs from the prior line's HybridMuon routing,
which EXCLUDED embeddings/head; the bake-off's routing follows the approved
directive text.

Symmetry rule (approved): every arm sees the same grad-clip (1.0,
error_if_nonfinite), the same stream/data order, the same matched-exposure
stopping rule and honest clock, the same calibration eval; the ONLY
differences are the optimizer family and (where the family demands it) the
schedule -- receipted per arm in the run receipts and the RESULT doc.
"""
from __future__ import annotations

import re

import torch

from experiments.expressivity_tasks.hybrid_muon import scaled_update

# The x45 winner's Muon-owned parameter names (the converted K10 model:
# blocks.N.mixer.{value,key,query_key,out} are nn.Linear weights, embed/head
# are the embedding and the output projection).
_MUON_EMBED_HEAD = frozenset(("embed.weight", "head.weight"))
_MUON_MIXER = re.compile(
    r"blocks\.\d+\.mixer\.(value|key|query_key|out)\.weight$")
_MUON_MLP = re.compile(r"blocks\.\d+\.mlp\.w[123]\.weight$")


def split_parameters(model):
    """Route the converted winner's parameters to (muon, adamw) by name.

    Fails loudly on a Muon-owned name that is not a full 2D matrix (the
    Newton-Schulz convention requires it) and on distinct parameters sharing
    storage (the prior line's own guard -- a shared-storage parameter would be
    updated twice).
    """
    muon, adamw = [], []
    storage = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.device.type != "meta":
            key = (param.device, param.untyped_storage().data_ptr())
            if key in storage:
                raise ValueError(f"distinct parameters share storage: {name}")
            storage.add(key)
        is_muon = (name in _MUON_EMBED_HEAD
                   or _MUON_MIXER.fullmatch(name) is not None
                   or _MUON_MLP.fullmatch(name) is not None)
        if is_muon and param.ndim != 2:
            raise ValueError(f"Muon-owned parameter is not a full 2D matrix: {name}")
        (muon if is_muon else adamw).append(name)
    if not muon or not adamw:
        raise ValueError("the hybrid requires both semantic groups non-empty")
    params = dict(model.named_parameters())
    return [params[n] for n in muon], [params[n] for n in adamw], muon, adamw


class HybridMuonX45(torch.optim.Optimizer):
    """``hybrid_muon.HybridMuon``'s step VERBATIM (the pinned quintic, the
    lerp/Nesterov momentum, the decoupled decay on the matrix group, the aux
    AdamW kept in step with the facade's lr/wd), with the two bake-off
    parameterizations: the x45 routing (explicit parameter lists) and the aux
    optimizer's betas/weight-decay (the probe's own transferred beta2 and the
    tuned incumbent's decay, so the hybrid holds every non-Muon knob at the
    incumbent's value -- symmetry)."""

    def __init__(self, muon_params, adamw_params, lr: float, betas, weight_decay: float):
        super().__init__(
            [{"params": list(muon_params), "family": "muon"},
             {"params": list(adamw_params), "family": "adamw"}],
            {"lr": lr, "weight_decay": weight_decay})
        self.aux = torch.optim.AdamW(self.param_groups[1]["params"], lr=lr,
                                     betas=tuple(betas), eps=1e-8,
                                     weight_decay=weight_decay)
        self.last_metrics = {}

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("closures unsupported in the registered experiment")
        squared, count = 0.0, 0
        group = self.param_groups[0]                    # the muon group
        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad.float()
            if grad.is_sparse:
                raise ValueError("sparse Muon gradients unsupported")
            state = self.state[param]
            # "momentum_buffer" not in state -- NOT "if not state": a
            # schedule-free wrapper pre-populates this state dict with its
            # own "z" buffer BEFORE the base step runs (schedulefree's
            # ScheduleFreeWrapper.step), so an empty-dict test would skip
            # creating the momentum buffer and crash on the read.
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(param,
                                                           dtype=torch.float32)
            buf = state["momentum_buffer"]
            # Upstream lerp momentum convention; Nesterov enabled (verbatim).
            buf.lerp_(grad, .05)
            direction = grad.lerp(buf, .95)
            update = scaled_update(direction)
            if not torch.isfinite(update).all():
                raise FloatingPointError("nonfinite Muon update")
            param.mul_(1 - group["lr"] * group["weight_decay"])
            param.add_(update.to(param.dtype), alpha=-group["lr"])
            ss = update.square().sum().item()
            squared += ss
            count += update.numel()
        for key in ("lr", "weight_decay"):
            self.aux.param_groups[0][key] = self.param_groups[1][key]
        self.aux.step()
        self.last_metrics = {"muon_update_rms":
                             (squared / count) ** .5 if count else 0.}

    def state_dict(self):
        return {"family": "x45-bakeoff-hybrid-muon-v1",
                "muon": super().state_dict(), "adamw": self.aux.state_dict()}

    def load_state_dict(self, state_dict):
        if state_dict.get("family") != "x45-bakeoff-hybrid-muon-v1":
            raise ValueError("incompatible hybrid optimizer family")
        super().load_state_dict(state_dict["muon"])
        self.aux.load_state_dict(state_dict["adamw"])


def sf_x_snapshot(model, optimizer, swap_beta: float):
    """The schedule-free primal-averaged (eval) weights as a CPU state_dict
    clone, WITHOUT touching the live parameters -- the training path stays
    unperturbed by construction (the side-by-side snapshot rule), which the
    package's own in-place ``eval()``/``train()`` swap cannot promise.

    ``x = y + (z - y) * (1 - 1/beta)`` is the package's ``eval()`` line
    verbatim (``p.lerp_(end=z, weight=1 - 1/beta)``); ``swap_beta`` is the
    first momentum coefficient -- ``betas[0]`` for ``AdamWScheduleFree``, the
    wrapper's ``momentum`` for ``ScheduleFreeWrapper``.
    """
    weight = 1.0 - 1.0 / swap_beta
    snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    alias = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        alias.setdefault(id(param), []).append(name)
    for group in optimizer.param_groups:
        for param in group["params"]:
            state = optimizer.state.get(param)
            if not state or "z" not in state:
                continue
            x = param + (state["z"].to(param.device) - param) * weight
            for name in alias.get(id(param), []):
                snapshot[name] = x.detach().cpu().clone()
    return snapshot
