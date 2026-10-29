"""Exact X42 velocity objective, independent of the legacy search evaluator.

This module only scores already validated metrics. It does not attest training
duration, corpus identity, telemetry provenance, or pure-stack parity.
"""
import math
from numbers import Real

TRAIN_SECONDS = 2700
EVAL_CONTEXT = 2048
W_OCC = 0.01


def _finite(name, value, lower=None, upper=None):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite real number')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if lower is not None and value < lower:
        raise ValueError(f'{name} must be >= {lower}')
    if upper is not None and value > upper:
        raise ValueError(f'{name} must be <= {upper}')
    return value


def score_metrics(*, nll_heldout, occ_frac, ever_touched_fractions,
                  r_cap_fire_rate, occupancy_slope):
    """Return additive score components and quantized degeneracy flags.

    ever_touched_fractions must contain every evaluated layer/head. Aggregation
    of per-token live occupancy belongs to the evaluator, before this function.
    R-CAP rate is deliberately not clipped: multiple heads can fire per token.
    """
    nll = _finite('nll_heldout', nll_heldout, 0)
    occ = _finite('occ_frac', occ_frac, 0, 1)
    touched = [_finite('ever_touched_fraction', x, 0, 1)
               for x in ever_touched_fractions]
    if not touched:
        raise ValueError('ever_touched_fractions must include every head')
    rate = _finite('r_cap_fire_rate', r_cap_fire_rate, 0)
    slope = _finite('occupancy_slope', occupancy_slope)
    flags = dict(dead_store=any(x < .01 for x in touched),
                 pressure_runaway=rate > .05, slope_one=slope >= .9)
    penalties = {name: amount if flags[name] else 0.
                 for name, amount in [('dead_store', .05),
                                      ('pressure_runaway', .05),
                                      ('slope_one', .02)]}
    total_penalty = sum(penalties.values())
    return dict(score=nll + W_OCC * occ + total_penalty,
                nll_heldout=nll, occ_frac=occ, w_occ=W_OCC,
                occupancy_penalty=W_OCC * occ,
                degeneracy_flags=flags, degeneracy_penalties=penalties,
                degeneracy_penalty=total_penalty,
                ever_touched_fractions=touched, r_cap_fire_rate=rate,
                occupancy_slope=slope, minimized=True)
