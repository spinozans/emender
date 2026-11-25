"""TRAIN-only diagnostic: use the already computed literal predictor as an action-independent control.

This is post-hoc analysis code, not the learner used to produce any X49 endpoint. It does not alter
the hard sampler, look up unchosen destinations, remove misses from NLL, or learn a key teacher.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def relative_objective(roll, literal_lp, targets):
    """Input comes from one actual same-weight rollout. Targets are used AFTER scoring only.

    The host sees only observed bytes; the literal B answer is computed BEFORE file/query and
    is independent of all private symbols at these fixed weights. Subtracting its realized NLL
    from the score-function return is unbiased even when the baseline shares host parameters,
    because targets are exogenous and the entire control is detached. No extra memory read.
    """
    from experiments.expressivity_tasks.x49_live_trace_memory.literal import (
        MIN_AGE,
        HORIZONS,
    )

    n = len(roll.lp)
    if (
        literal_lp.shape != roll.lp.shape
        or roll.lp.shape != (n, 256)
        or targets.shape != (n,)
        or targets.dtype != torch.long
        or roll.horizon not in HORIZONS
        or roll.file_lp.shape != (n - 1,)
        or roll.query_lp.shape != (n,)
    ):
        raise ValueError("misaligned full-window rollout or literal control")
    each = F.nll_loss(roll.lp, targets, reduction="none")
    nll = each.mean()
    with torch.no_grad():
        base = F.nll_loss(literal_lp, targets, reduction="none")
        advantage = each.detach().double() - base.double()
        js = torch.arange(1, n, device=advantage.device)
        cs = F.pad(advantage.cumsum(0), (1, 0))
        returns = (
            cs[(js + roll.horizon).clamp_max(n)] - cs[(js + MIN_AGE).clamp_max(n)]
        ).float()
    score = (
        (roll.query_lp * advantage.float()).sum() + (roll.file_lp * returns).sum()
    ) / n
    return nll + (score - score.detach()), nll


def enumeration(*, file_literal, query_literal, nll_miss, nll_hit):
    """Exact 256 x 256 one-eligible-record oracle. NOT a method for sampling a training read.

    For this synthetic final frontier only j=1 is eligible. Its contribution depends only on
    equality of the two actual private symbols. Different actions that miss give the same base
    loss. The two action probabilities and trace reward are fully specified; no sampled labels.
    """
    from math import log

    init = torch.zeros(2, 256, dtype=torch.float64)
    init[0, file_literal] = init[1, query_literal] = log(1 + 9 * 256)
    par = init.requires_grad_()
    lp = par.log_softmax(-1)
    p, q = lp.exp().unbind(0)
    weights = p[:, None] * q[None, :]
    costs = torch.full_like(weights, nll_miss)
    costs.fill_diagonal_(nll_hit)
    (true,) = torch.autograd.grad((costs * weights).sum(), par, retain_graph=True)
    out = {}
    for name, base in [("registered_constant", 2.0), ("literal_control", nll_miss)]:
        adv = costs - base
        surrogate = (weights.detach() * adv * (lp[0, :, None] + lp[1, None, :])).sum()
        (g,) = torch.autograd.grad(surrogate, par, retain_graph=True)
        err = float((g - true).abs().max())
        assert err < 1.0e-13
        # Per-sample joint squared score norm for two disjoint logit vectors.
        with torch.no_grad():
            ss = (1 - 2 * p + p.square().sum())[:, None] + (
                1 - 2 * q + q.square().sum()
            )[None, :]
            var = float((weights * adv.square() * ss).sum() - g.square().sum())
        out[name] = dict(expected_gradient_max_abs_error=err, total_variance=var)
    return dict(
        hit_probability=float(weights.detach().diagonal().sum()),
        true_gradient_norm=float(true.norm()),
        estimators=out,
    )


def synthetic():
    """Compare actual hard-sampled gradients with the production objective on the same rollout."""
    import math
    from ndm.memory_program.direct import DirectTrust
    from experiments.expressivity_tasks.x49_live_trace_memory.values import (
        LiveValueHead,
    )
    from experiments.expressivity_tasks.x49_live_trace_memory.trace import (
        TraceHead,
        objective,
    )

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4901)
        word = b"a" + b"z" * 16 + b"a"
        n = len(word)
        ys = torch.tensor(list(word[1:]) + [54])
        c = TraceHead(LiveValueHead(DirectTrust(8)), learned=True)
        h = torch.randn(n, 8, requires_grad=True)
        z = torch.randn(n, 256, requires_grad=True)
        out = {}
        for stage in ("birth", "nonzero"):
            if stage == "nonzero":
                with torch.no_grad():
                    c.delta.bias[54] += 1.0
            captured = []

            def output_only(module, args, output):
                captured.append(output)

            hook = c.literal.register_forward_hook(output_only)
            try:
                r = c(word, h, z, seed=4901, document=7, visit=1, horizon=128)
            finally:
                hook.remove()
            assert len(captured) == 1 and r.source == (None,) * (n - 1) + (1,)
            raw, nll = objective(r, ys, learned=True)
            rel, other = relative_objective(r, captured[0], ys)
            assert torch.equal(raw, rel) and torch.equal(nll, other)
            pol = [*c.filing.parameters(), *c.ask.parameters()]
            pg = torch.autograd.grad(raw, pol, retain_graph=True)
            cg = torch.autograd.grad(rel, pol, retain_graph=True)

            def norm(gs):
                return float(sum(g.double().square().sum() for g in gs).sqrt())

            shared = [
                h,
                z,
                *c.literal.parameters(),
                *c.value.parameters(),
                *c.mix.parameters(),
                *c.delta.parameters(),
            ]
            # At zero policy weights score gradients have no path to h. All ordinary pathwise
            # derivatives must be identical; both the reward and the control must be detached.
            a = torch.autograd.grad(rel, shared, retain_graph=True, allow_unused=True)
            b = torch.autograd.grad(nll, shared, allow_unused=True)
            assert all(
                (u is None and v is None)
                or (u is not None and v is not None and torch.equal(u, v))
                for u, v in zip(a, b, strict=True)
            )
            out[stage] = dict(
                raw_policy_grad=norm(pg),
                relative_policy_grad=norm(cg),
                same_pathwise_gradients=True,
            )
            assert out[stage]["raw_policy_grad"] > 0 and (
                (out[stage]["relative_policy_grad"] == 0) == (stage == "birth")
            )
    exact = {
        str(q): enumeration(
            file_literal=97,
            query_literal=q,
            nll_miss=-math.log(0.25),
            nll_hit=-math.log(0.4),
        )
        for q in (97, 98)
    }
    return dict(
        scope="CPU synthetic hard-read and exact expectation only; not an X49 natural learner",
        hard_read=out,
        oracle=exact,
    )


if __name__ == "__main__":
    import json
    import os

    if os.getenv("CUDA_VISIBLE_DEVICES") != "" or os.getenv("GPU_LEASE_HELD"):
        raise ValueError("CPU only")
    print(json.dumps(synthetic(), allow_nan=False, sort_keys=True))
