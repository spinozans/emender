"""Read the frozen model before consulting labels. Counterfactuals share ALL prefixes."""

from dataclasses import dataclass

import torch

from ..corpus_kernel.attention import log_features
from ..corpus_windows import training
from .editions import K


@dataclass(frozen=True)
class View:
    logp: torch.Tensor
    shares: torch.Tensor
    rows: torch.Tensor | None  # conditional on the corpus, answer frontiers only


@torch.no_grad()
def view(w, m, law):
    """Same vector logits used by the trainer; score stably even for stale answers."""
    s = training.encode(w, m, law, "torch")
    logits, shares, log_z = [], [], []
    for b in w.blocks:
        sl = slice(b.start, b.stop)
        x = s.x[sl]
        a = training.read(w, m, s, law, b, x, s.q[sl])
        logits.append(m.lm(x + m.out(a.read)))
        shares.append(a.shares)
        log_z.append(a.source_log_z[:, 0])
    rows = None
    if w.layout.rows:
        # Independent literal decomposition for this *diagnostic*, not the scored
        # tree read. Sorted rows correspond to key 0..7 even when successors swap.
        if tuple(w.archive.index.row_doc) != tuple(range(K)):
            raise ValueError("foreign row order")
        q, k = s.q[1::2].double(), s.rows[1].double()
        if law == "dense":
            score = (q @ k.T) * m.dim**-0.5
        else:
            qf, kf = (log_features(z, m.omega.double()) for z in (q, k))
            score = (qf[:, None] + kf[None, :]).logsumexp(-1)
        score = score + s.rows[3].double()
        torch.testing.assert_close(
            score.logsumexp(1), torch.cat(log_z)[1::2].double(), rtol=2e-5, atol=2e-5
        )
        rows = score.softmax(1)
    return View(
        torch.cat(logits).double().log_softmax(-1), torch.cat(shares).double(), rows
    )


class Means:
    def __init__(self):
        self.sums = {}
        self.counts = {}

    def put(self, name, values):
        x = values.double().sum().item()
        self.sums[name] = self.sums.get(name, 0.0) + x
        self.counts[name] = self.counts.get(name, 0) + values.numel()

    def result(self):
        return {k: s / self.counts[k] for k, s in self.sums.items()}


def nll(logp, y):
    return -logp.gather(1, y[:, None]).flatten()


def compare(total, old, y0, new, y1, changed, *, cross=None):
    """Pair by LIVE key, not by answer token or child row rank."""
    old_y, new_y = y0[1::2], y1[1::2]
    edits = old_y != new_y
    if tuple(edits.nonzero().flatten().tolist()) != tuple(changed) or not torch.equal(
        y0[::2], y1[::2]
    ):
        raise ValueError("wrong key pairing or changed cue")
    # Archive keys/row scores, local rows, and zero-hint mass are fixed in the
    # two-fact edit. Only the old-to-new assignment of successor vectors changes.
    torch.testing.assert_close(old.shares, new.shares, atol=0, rtol=0)
    if old.rows is not None:
        torch.testing.assert_close(old.rows, new.rows, atol=0, rtol=0)
    for label, p, y in (("old", old, y0), ("edited", new, y1)):
        z = nll(p.logp, y)
        total.put(f"{label}.all_nll", z)
        total.put(f"{label}.cue_nll", z[::2])
        total.put(f"{label}.answer_nll", z[1::2])
        total.put(f"{label}.answer_acc", p.logp[1::2].argmax(1) == y[1::2])
    a, b = old.logp[1::2], new.logp[1::2]
    p0, p1 = a.argmax(1), b.argmax(1)
    stale, intact = nll(a, new_y), nll(b, new_y)
    for name, mask in (("changed", edits), ("unchanged", ~edits)):
        total.put(name + ".old_and_new_correct", ((p0 == old_y) & (p1 == new_y))[mask])
        total.put(name + ".top1_changed", (p0 != p1)[mask])
        total.put(name + ".stale_nll", stale[mask])
        total.put(name + ".edited_nll", intact[mask])
        total.put(name + ".gain_vs_stale", (stale - intact)[mask])
    for i, src in enumerate(("corpus", "local", "no_read")):
        total.put("share." + src, new.shares[1::2, i])
    if new.rows is not None:
        total.put("address.intended_mass", new.rows.diag())
        total.put("address.intended_top1", new.rows.argmax(1) == torch.arange(K))
    if cross is not None:
        cp = cross.logp[1::2]
        total.put("other_law.edited_answer_nll", nll(cp, new_y))
        total.put("other_law.edited_answer_acc", cp.argmax(1) == new_y)
        total.put("other_law.token_tv", 0.5 * (cp.exp() - b.exp()).abs().sum(1))
        total.put("other_law.row_tv", 0.5 * (cross.rows - new.rows).abs().sum(1))
        total.put(
            "other_law.abs_share_error",
            (cross.shares[1::2, 0] - new.shares[1::2, 0]).abs(),
        )
