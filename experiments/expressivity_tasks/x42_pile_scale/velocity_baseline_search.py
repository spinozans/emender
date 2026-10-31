"""Registered baseline search space; anchor adapters remain byte-identical."""
from contextlib import nullcontext
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import subprocess
from unittest.mock import patch
import numpy as np
import torch
from . import velocity_baselines as anchors
from .velocity_train import sha, verify_registration

ROOT = Path(__file__).resolve().parent
COMMON = [
    ('lr', 1e-4, 1e-3, .00025180085693680385, True),
    ('warmup_steps', 50., 2000., 122., True),
    ('weight_decay', .001, .3, .1, True),
    ('microbatch_log2', 0., 6., 4., False),
    ('init_scale', .5, 1.5, 1., True),
    ('beta2', .9, .999, .95, False),
]
GDN = [('decay_rate_scale', .25, 4., 1., True),
       ('dt_bias_shift', -2., 2., 0., False),
       ('expansion_index', 0., 1., 1., False)]

@dataclass(frozen=True)
class SearchPolicy:
    lr: float = .00025180085693680385
    warmup_steps: int = 122
    weight_decay: float = .1
    microbatch: int = 16
    init_scale: float = 1.
    beta2: float = .95
    decay_rate_scale: float = 1.
    dt_bias_shift: float = 0.
    expansion: float = 2.


def dimensions(arm):
    if arm not in ('transformer', 'gdn2'): raise ValueError(arm)
    return COMMON + (GDN if arm == 'gdn2' else [])


def center(arm):
    return np.array([(math.log(c/lo)/math.log(hi/lo) if log else (c-lo)/(hi-lo))
                     for _, lo, hi, c, log in dimensions(arm)])


def decode(arm, vector):
    if len(vector) != len(dimensions(arm)) or not np.isfinite(vector).all():
        raise ValueError('invalid search vector')
    unit = 1-np.abs(np.mod(vector, 2)-1)
    values = {name: float(min(hi,max(lo,math.exp(math.log(lo)+u*math.log(hi/lo)) if log else lo+u*(hi-lo))))
              for u, (name, lo, hi, _, log) in zip(unit, dimensions(arm))}
    values['warmup_steps'] = round(values['warmup_steps'])
    values['microbatch'] = 2**round(values.pop('microbatch_log2'))
    if arm == 'gdn2': values['expansion'] = (1., 2.)[round(values.pop('expansion_index'))]
    return SearchPolicy(**values)


def validate_policy(policy, arm):
    # Explicit policies must be representable by this registered search space.
    if policy.microbatch not in (1, 2, 4, 8, 16, 32, 64): raise ValueError('B64 microbatch divisor required')
    values = asdict(policy)
    for name, lo, hi, _, _ in dimensions(arm):
        if name in ('microbatch_log2', 'expansion_index'): continue
        if not math.isfinite(values[name]) or not lo <= values[name] <= hi: raise ValueError(name)
    if type(policy.warmup_steps) is not int: raise ValueError('integer warmup required')
    if policy.expansion not in (1., 2.): raise ValueError('expansion')
    if arm == 'transformer' and (policy.expansion != 2. or policy.decay_rate_scale != 1. or policy.dt_bias_shift != 0.):
        raise ValueError('GDN-only parameters')


class SearchBaseline(anchors.Baseline):
    def __init__(self, arm, policy):
        validate_policy(policy, arm)
        # Reuse the tested capacity solver for each expansion. The patch is local
        # to this candidate process; anchor and pure sources are untouched.
        from ndm.models import external_gdn2
        original = external_gdn2.GDN2ExternalMLPLayer
        def expanded(*args, **kwargs):
            return original(*args, **{**kwargs, 'expansion': policy.expansion})
        with (patch.object(external_gdn2, 'GDN2ExternalMLPLayer', expanded)
              if arm == 'gdn2' else nullcontext()):
            super().__init__(arm)
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                    module.weight.mul_(policy.init_scale)
            if arm == 'gdn2':
                for block in self.model.blocks:
                    core = block.layer.gdn2.gdn2
                    core.A_log.add_(math.log(policy.decay_rate_scale))
                    core.dt_bias.add_(policy.dt_bias_shift)


def verify_search(path, arm):
    from scripts.registration_guard import diagnose
    report = diagnose(path)
    if not report['matches']:
        raise ValueError(json.dumps(report, indent=2))
    path = Path(path)
    registration = json.loads(path.read_text())
    if subprocess.check_output(['git', 'show', 'HEAD:'+str(path.relative_to(Path.cwd()) if path.is_absolute() else path)]) != path.read_bytes():
        raise ValueError('search registration must be committed')
    if registration['arm'] != arm or registration['dimensions'] != [list(d) for d in dimensions(arm)]:
        raise ValueError('family/search space mismatch')
    if registration['score'] != 'held-out NLL only' or registration['train_seconds'] != 2700:
        raise ValueError('protocol mismatch')
    verify_registration((ROOT/'velocity_registration.json').relative_to(Path.cwd()))
    for name, digest in registration['sources'].items():
        if sha(name) != digest: raise ValueError('search source changed: '+name)
    import importlib.metadata
    if importlib.metadata.version('flash-linear-attention') != registration['fla_version']:
        raise ValueError('FLA version mismatch')
    return registration
