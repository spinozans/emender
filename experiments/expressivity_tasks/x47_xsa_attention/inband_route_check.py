"""Diagnostic-only tensors. Never passed into a predictor or retained in git."""

from collections import Counter
from dataclasses import dataclass

import torch
from .frozen_packet_bench import tensor_sha
from .packed_program_profile import BindingNames, digest_rows


@dataclass
class Witness:
    tape: object
    scores: dict  # Detached byte distributions, joint scores, scalar all-byte NLL.
    requests: list[str]  # Ordered nonempty logical submissions for each decision (before coalescing).
    receipt: dict


def submitted(batch, sessions):
    names = BindingNames(sessions)
    return [
        digest_rows(
            [
                (names.origin(e.reader), tuple(map(names.key, e.keys)))
                for e in d.trace
                if e.inst.op == "LOOKUP" and e.keys
            ]
        )
        for doc in batch.decisions
        for d in doc
    ]


def scores(batch):
    ds = [d for doc in batch.decisions for d in doc]
    return dict(
        byte_vectors=torch.stack([d.log_probs.detach() for d in ds]),
        joint_scores=torch.stack([d.log_policy.detach() for d in ds]),
        nll=batch.mean_nll.detach(),
    )


def delta(ref, actual):
    if ref.dtype != actual.dtype or ref.dtype != torch.float32 or ref.shape != actual.shape:
        raise ValueError("different comparison schema")
    if not ref.isfinite().all().item() or not actual.isfinite().all().item():
        raise ValueError("nonfinite comparison")
    a = ref.reshape(-1)
    b = actual.reshape(-1)
    bits = (a.view(torch.int32) != b.view(torch.int32)).sum().item()
    return dict(
        elements=a.numel(),
        unequal_bits=bits,
        max_abs=(a.double() - b.double()).abs().max().item() if bits else 0.0,
        ref_sha=tensor_sha(a),
        actual_sha=tensor_sha(b),
    )


def sensitivity(initial, post, fixed, counter):
    """Same full weights in post/counter; differing earlier instructions may affect later read values."""
    a = post.scores["byte_vectors"]
    b = counter.scores["byte_vectors"]

    def different(x, y):
        return (x.view(torch.int32) != y.view(torch.int32)).any(-1).tolist()

    byte_changes = different(a, b)
    top_changes = (a.argmax(1) != b.argmax(1)).tolist()
    fixed_changes = different(initial.scores["byte_vectors"], fixed.scores["byte_vectors"])
    err = (a.double() - b.double()).abs()
    frac = err / (5e-5 + 2e-5 * a.double().abs())
    bydoc = []
    i = 0
    for left, right in zip(initial.tape.programs, post.tape.programs, strict=True):
        same_history = True
        row = Counter()
        peak = 0.0
        absolute = 0.0
        for old_byte, new_byte in zip(left, right, strict=True):
            ix = [c.instruction for c in old_byte]
            iy = [c.instruction for c in new_byte]
            same_history &= old_byte == new_byte
            row.update(
                positions=1,
                opcode=int([v.op for v in ix] != [v.op for v in iy]),
                instruction=int(ix != iy),
                structure=int(old_byte != new_byte),
                request=int(initial.requests[i] != post.requests[i]),
                full_live_vs_original_argmax=int(top_changes[i]),
                full_live_vs_original_vector=int(byte_changes[i]),
                fixed_route_value_vector=int(fixed_changes[i]),
                prefix_identical=int(same_history),
            )
            if same_history:
                peak = max(peak, frac[i].max().item())
                absolute = max(absolute, err[i].max().item())
                row["prefix_tolerance_violations"] += int((frac[i] > 1).sum().item())
            i += 1
        bydoc.append(dict(counts=dict(row), prefix_max_abs=absolute, prefix_max_fraction=peak))
    if i != len(a):
        raise ValueError("incomplete post-update contrast")
    return dict(
        per_document=bydoc,
        full_live_vs_original_max_abs=err.max().item(),
        fixed_value_max_abs=(initial.scores["byte_vectors"] - fixed.scores["byte_vectors"])
        .abs()
        .max()
        .item(),
    )
