"""X10b: unchanged X10 hybrid-Muon instrument, with only the LR scale varied.

The adapter patches X10's instrument only within this process/call; historical
X10 source and registrations remain untouched. Both hybrid groups scale equally.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

from experiments.expressivity_tasks import run_invertible_stage10_optimizers_8m as x10

ROOT = Path('experiments/expressivity_tasks/results_invertible_stage10b_muonlr_8m')
MULTIPLIERS = {'muonlr3x': 3, 'muonlr10x': 10}
ARMS = tuple(MULTIPLIERS)
CARDS = {'muonlr3x': 3, 'muonlr10x': 4}
ROOTS = dict.fromkeys(ARMS, ROOT)
DEPTH, CEILING, EVAL_EVERY = x10.DEPTH, x10.CEILING, x10.EVAL_EVERY
x7, np = x10.x7, x10.np
InvertibleQuadraticTask = x10.InvertibleQuadraticTask
verify_registration = x10.verify_registration
_build_model, _make_optimizer, _point = x10.build_model, x10.make_optimizer, x10.point


def build_model(arm):
    if arm not in ARMS:
        raise ValueError('unregistered arm')
    return _build_model('muon')


def make_optimizer(model, arm):
    if arm not in ARMS:
        raise ValueError('unregistered arm')
    return _make_optimizer(model, 'muon')


def point(arm):
    if arm not in ARMS:
        raise ValueError('unregistered arm')
    return _point('muon')


def learning_rate(arm, step):
    return MULTIPLIERS[arm] * x10.learning_rate(step)


def rates(opt, arm, step):
    for group in opt.param_groups:
        group['lr'] = learning_rate(arm, step)


def train_update(model, opt, arm, batch, step):
    with patch.object(x10, 'rates', rates):
        return x10.train_update(model, opt, arm, batch, step)


def run(args):
    # Reuse the exact training loop, frozen-pool checks, per-position validation,
    # checkpoint/retention, finite checks and safety-only stopping from X10.
    with patch.multiple(x10, ROOTS=ROOTS, build_model=build_model,
                        make_optimizer=make_optimizer, point=point, rates=rates):
        return x10.run(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seed', type=int, choices=[0], default=0)
    run(parser.parse_args())
