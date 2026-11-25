"""Prospective CPU admission check: actual head on invented full-size forecast packets. No corpus or GPU."""

from time import perf_counter
import numpy as np
import torch
from ..x48_muon_continuation.data import check
from . import (
    error_history as eh,
    error_history_fit as fit,
    joint_reader as joint,
    local_u,
)
from .panels import T


def invented():
    g = np.random.default_rng(4915)
    mixed = b"".join(
        b"\0color: " + bytes([c]) + b", key: " + g.bytes(2) + b"\n"
        for c in g.integers(0, 256, size=T // 12 + 1)
    )[:T]
    return dict(
        recurring_mixed=mixed,
        uniform_bytes=g.bytes(T),
        one_key_many_wrong_bytes=bytes(T),
    )


@torch.inference_mode()
def audit_head(s, e, head):
    c, ub = local_u.contract(s)
    j, jb = joint.contract(s)
    check(c == j == e["sha256"], "wrong EH price head/coefficients")
    writer = joint._writer()
    g = torch.Generator().manual_seed(4915)
    h = torch.randn(T, head.direct.host_dim, generator=g)
    z = torch.randn(T, 256, generator=g)
    out = {}
    beta = {
        k: [0.0625 * (i + 1) for i in range(d)] for k, d in eh.DIMS.items()
    }  # invented; never installed
    for key, w in invented().items():
        if key == "one_key_many_wrong_bytes":
            z = torch.zeros(T, 256)
            z[torch.arange(T), torch.arange(T) % 256] = 3.0
        check(len(w) == T, "incomplete invented price packet")
        p = eh.fields(head, w, h, z, jb, writer, ub=ub)
        y = w[1:] + b"\xff"
        m = fit.Moments()
        tic = perf_counter()
        right, _ = m.add(p, y)
        fit_wall = perf_counter() - tic
        m.freeze()
        tic = perf_counter()
        right2, sums, _ = fit.score(p, beta, y)
        probe_wall = perf_counter() - tic
        check(
            right2 == right
            and m.targets == len(w)
            and m.documents == 1
            and set(sums) == set(eh.ARMS),
            "synthetic price exposure",
        )
        extra = {
            k: p.seconds[k]
            for k in (
                "guess_and_tape_prep_s",
                "true_pool_s",
                "shift_pool_s",
                "feature_s",
            )
        }
        out[key] = dict(
            positions=T,
            active_rows=p.work["pool_active_rows"],
            packed_feature_entries=p.work["packed_feature_entries"],
            fp64_feature_bytes=p.work["fp64_feature_bytes"],
            changed_histogram_rows=p.work["changed_histogram_rows"],
            extra_shared_components_s=extra,
            extra_fit_s=sum(extra.values()) + fit_wall,
            extra_probe_s=sum(extra.values()) + probe_wall,
        )
    return dict(
        checkpoint_sha=c, invented_cpu_packets=out, natural_windows=0, host_executions=0
    )
