"""Factorial comparisons on NATIVE logits. A separate parameter-level identity predicts the odds change."""

from itertools import product

import torch

from ..semantic_tail.editions import K, VALUE, VOCAB
from ..semantic_tail.measure import Means


def polarity(m):
    # Rows = output answer ID, cols = observed archive successor ID. No bias term.
    with torch.no_grad():
        e = m.embed.weight[list(VALUE)].double()
        M = (
            m.lm.weight[list(VALUE)].double()
            @ m.out.weight.double()
            @ m.value.weight.double()
            @ e.T
        )
        diag = M.diag()
        return diag[:, None] + diag[None, :] - M - M.T


class Stats(Means):
    def __init__(self):
        super().__init__()
        self.max_error = 0.0


def comparisons(t, name, old, new, a, b, coeff, kappa):
    """Each aligned record has old label a != new label b on the same live query."""
    ix = torch.arange(len(a))
    va, vb = a + VALUE[0], b + VALUE[0]
    both = (old.argmax(1) == va) & (new.argmax(1) == vb)
    t.put(name + ".both_correct", both)
    t.put(name + ".gain_vs_stale", new[ix, vb] - old[ix, vb])
    odds = (new[ix, vb] - new[ix, va]) - (old[ix, vb] - old[ix, va])
    kap = kappa[a, b]
    predicted = coeff * kap
    err = (odds - predicted).abs()
    t.max_error = max(t.max_error, err.max().item())
    torch.testing.assert_close(odds, predicted, atol=2e-4, rtol=2e-5)
    t.put(name + ".delta_odds", odds)
    t.put(name + ".both_with_negative_decoder_contrast", both & (kap < 0))


def accumulate(total, keys, logp, g, kappa, corpus):
    """logp[a,b,s,z] for the two queried keys in one 64-edition panel."""
    if logp.shape != (K, K, 2, VOCAB) or g.shape != (K, K):
        raise ValueError("wrong panel axes")
    xy = torch.tensor(list(product(range(K), repeat=2)))
    trip = torch.tensor(
        [(a, c, b) for a, c, b in product(range(K), repeat=3) if a != b]
    )
    swap = xy[xy[:, 0] != xy[:, 1]]
    for s, i in enumerate(keys):
        j = keys[1 - s]
        # Now the first TWO indices are own and peer VALUE in that order.
        ll = logp[:, :, s] if s == 0 else logp[:, :, s].transpose(0, 1)
        a, c = xy.T
        z = ll[a, c]
        clamp = ll[0, c]
        ix = torch.arange(len(a))
        y = a + VALUE[0]
        total.put("answer_nll", -z[ix, y])
        total.put("answer_acc", z.argmax(1) == y)
        total.put("clamped_nll", -clamp[ix, y])
        total.put("clamped_acc", clamp.argmax(1) == y)
        total.put("target_joint_mass", g[i, i].view(1))
        total.put("corpus_share", g[i].sum().view(1))
        if corpus:
            total.put("target_conditional_mass", (g[i, i] / g[i].sum()).view(1))
            total.put("target_top_row", (g[i].argmax() == i).view(1))
        a, c, b = trip.T
        comparisons(total, "single", ll[a, c], ll[b, c], a, b, g[i, i], kappa)
        a, b = swap.T
        comparisons(total, "swap", ll[a, b], ll[b, a], a, b, g[i, i] - g[i, j], kappa)


def finish(t, kappa):
    if (
        t.counts["answer_acc"],
        t.counts["single.both_correct"],
        t.counts["swap.both_correct"],
    ) != (3584, 25088, 3136):
        raise ValueError("foreign edit denominators")
    result = t.result()
    i, j = torch.triu_indices(K, K, 1)
    cap = kappa[i, j]
    result.update(
        decoder_contrast_mean=cap.mean().item(),
        negative_decoder_contrasts=(cap < 0).sum().item(),
        max_odds_identity_error=t.max_error,
    )
    return result
