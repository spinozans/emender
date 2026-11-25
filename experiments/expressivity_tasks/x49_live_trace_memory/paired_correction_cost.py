"""Pre-reservation pricing on invented full-size byte/host packets with each strictly loaded frozen head."""

from time import perf_counter
import numpy as np
import torch
from ..x48_muon_continuation.data import check
from . import (
    paired_correction as pc,
    paired_correction_fit as fit,
    paired_correction_panel as panel,
    joint_reader as joint,
)
from .panels import T


def cases():
    rng = np.random.default_rng(4917)
    mixed = b"".join(
        b"abcd\xff\0\x1ek" + bytes([c]) + b"~more;!"
        for c in (10, 32, 0, 32, 10, 32, 80, 81, 81, 65, 10, 32, 255, 80, 10, 32)
    ) * (T // 256)
    q = np.array(
        [80 + (t // 16) % 2 if t % 16 == 7 else t % 7 for t in range(T)], np.uint8
    )
    # Many conditional alternatives; four forecasts share the repeated literal prefix.
    wide = (
        b"".join(
            b"\0" * 8 + bytes([v]) + b"x" for v in rng.integers(0, 256, T // 10 + 1)
        )
    )[:T]
    return dict(
        uniform=(rng.bytes(T), rng.bytes(T)),
        recurring_pairs=(mixed, q.tobytes()),
        one_key_wrong_fanout=(bytes(T), bytes(range(256)) * 16),
        many_alternatives=(
            wide,
            bytes((160 + (t // 10) % 4) if t % 10 == 7 else 0 for t in range(T)),
        ),
    )


@torch.inference_mode()
def audit_head(s, e, head):
    ck, b = panel.fixed(s)
    check(e["sha256"] == ck, "wrong PC price checkpoint")
    writer = joint._writer()
    gen = torch.Generator().manual_seed(4917)
    h = torch.randn(T, head.direct.host_dim, generator=gen)
    noise = torch.rand(T, 256, generator=gen)
    at = torch.arange(T)
    beta = {k: [0.125 * (i + 1) for i in range(d)] for k, d in pc.DIMS.items()}
    out = {}
    for name, (w, q) in cases().items():
        check(len(w) == len(q) == T, "short PC price case")
        z = noise.clone()
        z[at, list(q)] = 4.0
        tic = perf_counter()
        p = pc.fields(
            head,
            w,
            h,
            z,
            b["J"],
            b["ERROR"],
            writer,
            mode="panel",
            anchors=(b["U"], b["HISTORY"]),
        )
        made = perf_counter() - tic
        m = fit.Moments()
        y = w[1:] + b"\xff"
        tic = perf_counter()
        a, _ = m.add(p, y)
        fc = perf_counter() - tic
        m.freeze()
        tic = perf_counter()
        right, loss, times = fit.score(p, beta, y)
        sc = perf_counter() - tic
        sh = p.seconds["shared_posthost_ERROR_s"]
        base_extra = made - sh - p.seconds["other_anchors_s"]
        check(
            a == right and m.targets == T and set(loss) == set(pc.ARMS),
            "incomplete PC price case",
        )
        out[name] = dict(
            positions=T,
            matched_query_rows=p.work["matched_query_rows"],
            changed_conditional_rows=p.work["changed_conditional_rows"],
            true_peak_pair_cells=p.work["pair_peak_pair_cells"],
            true_pair_index_s=p.seconds["true_pair_index_s"],
            sham_index_s=p.seconds["shuf_index_s"],
            shared_posthost_ERROR_s=sh,
            MARG_chosen_extra_s=p.seconds["marginal_s"] + times["MARG_response_s"],
            PAIR_chosen_extra_s=p.seconds["marginal_s"]
            + p.seconds["true_pair_index_s"]
            + p.seconds["true_pair_feature_s"]
            + times["PAIR_response_s"],
            extra_fit_s=base_extra + fc,
            extra_probe_s=made - sh + sc,
            posthost_fit_s=made - p.seconds["other_anchors_s"] + fc,
            posthost_probe_s=made + sc,
        )
    check(
        out["recurring_pairs"]["changed_conditional_rows"] > 0
        and out["many_alternatives"]["changed_conditional_rows"] > 0
        and out["one_key_wrong_fanout"]["changed_conditional_rows"] == 0,
        "PC price fixtures missed their intended support",
    )
    return dict(
        checkpoint_sha=ck,
        invented_cpu_packets=out,
        natural_windows=0,
        host_executions=0,
    )
