"""X10-only full-matrix Muon reference v1; not a production/DiLoCo optimizer.

KellerJordan/Muon quintic convention, pinned here: Frobenius normalization
(norm + 1e-7), transpose tall matrices, five (3.4445,-4.7750,2.0315)
iterations. Momentum and NS arithmetic are FP32; no BF16 master weights.
Moonshot scaling matches AdamW's approximate .2 update RMS, not exact polar
orthogonality (the deliberately short quintic is not an SVD).
"""
from __future__ import annotations

import copy
import math
import re

import torch

VERSION = 'x10-hybrid-muon-fp32-v1'
COEFFICIENTS = (3.4445, -4.7750, 2.0315)
EPS = 1e-7
ITERATIONS = 5
# Semantic hidden linear-map allowlist, deliberately not ndim-only.
HIDDEN = re.compile(r'(layers\.\d+\.(qkv_proj|q_proj|k_proj|v_proj|a_proj|g_proj|erase_gate_proj|value_write_gate_proj|o_proj)|mlps\.\d+\.w[123])\.weight$')


def newton_schulz(matrix):
    if matrix.ndim != 2:
        raise ValueError('NS requires a full 2D matrix')
    x = matrix.float()
    if not torch.isfinite(x).all():
        raise FloatingPointError('nonfinite Muon input')
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    x = x / (x.norm() + EPS)
    a, b, c = COEFFICIENTS
    for _ in range(ITERATIONS):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.T if tall else x


def scaled_update(momentum):
    return newton_schulz(momentum) * (.2 * math.sqrt(max(momentum.shape)))


def parameter_routes(model):
    aliases = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        if param.requires_grad:
            aliases.setdefault(id(param), []).append(name)
    rows = []
    storage = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Named-parameter aliases are deduplicated by identity; reject distinct
        # Parameter objects sharing storage rather than updating them twice.
        if param.device.type != 'meta':
            key = (param.device, param.untyped_storage().data_ptr())
            if key in storage:
                raise ValueError('distinct parameters share storage')
            storage.add(key)
        muon = bool(HIDDEN.fullmatch(name))
        if muon and param.ndim != 2:
            raise ValueError('allowlisted linear map is not 2D')
        rows.append({'name': name, 'aliases': aliases[id(param)], 'shape': list(param.shape),
                     'numel': param.numel(), 'owner': 'muon' if muon else 'adamw',
                     'weight_decay': .01, 'lr_multiplier': 1.,
                     'no_weight_decay_tag_ignored': bool(getattr(param, '_no_weight_decay', False))})
    return rows


class HybridMuon(torch.optim.Optimizer):
    """One facade/scheduler clock; AdamW-owned state stays inside aux optimizer."""
    def __init__(self, model, lr=0., weight_decay=.01):
        self.routes = parameter_routes(model)
        params = dict(model.named_parameters())
        muon = [params[r['name']] for r in self.routes if r['owner'] == 'muon']
        aux = [params[r['name']] for r in self.routes if r['owner'] == 'adamw']
        if not muon or not aux:
            raise ValueError('hybrid requires both semantic groups')
        super().__init__([{'params': muon, 'family': 'muon'}, {'params': aux, 'family': 'adamw'}],
                         {'lr': lr, 'base_lr': 1e-3, 'weight_decay': weight_decay})
        self.aux = torch.optim.AdamW(aux, lr=lr, betas=(.9, .999), eps=1e-8, weight_decay=weight_decay)
        self.last_metrics = {}

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('closures unsupported in registered experiment')
        squared, count, rms_rows = 0., 0, {}
        group = self.param_groups[0]
        for param, route in zip(group['params'], (r for r in self.routes if r['owner'] == 'muon')):
            if param.grad is None:
                continue
            grad = param.grad.float()
            if grad.is_sparse:
                raise ValueError('sparse Muon gradients unsupported')
            state = self.state[param]
            if not state:
                state['momentum_buffer'] = torch.zeros_like(param, dtype=torch.float32)
            buf = state['momentum_buffer']
            # Upstream lerp momentum convention; Nesterov enabled.
            buf.lerp_(grad, .05)
            direction = grad.lerp(buf, .95)
            update = scaled_update(direction)
            if not torch.isfinite(update).all():
                raise FloatingPointError('nonfinite Muon update')
            param.mul_(1 - group['lr'] * group['weight_decay'])
            param.add_(update.to(param.dtype), alpha=-group['lr'])
            ss = update.square().sum().item()
            rms_rows[route['name']] = math.sqrt(ss / update.numel())
            squared += ss
            count += update.numel()
        for key in ('lr', 'weight_decay'):
            self.aux.param_groups[0][key] = self.param_groups[1][key]
        self.aux.step()
        self.last_metrics = {'muon_update_rms': math.sqrt(squared / count) if count else 0.,
                             'matrix_update_rms': rms_rows}

    def state_dict(self):
        return {'family': VERSION, 'routes': copy.deepcopy(self.routes),
                'muon': super().state_dict(), 'adamw': self.aux.state_dict()}

    def load_state_dict(self, state_dict):
        if state_dict.get('family') != VERSION or state_dict.get('routes') != self.routes:
            raise ValueError('incompatible hybrid optimizer family or parameter routing')
        super().load_state_dict(state_dict['muon'])
        self.aux.load_state_dict(state_dict['adamw'])
