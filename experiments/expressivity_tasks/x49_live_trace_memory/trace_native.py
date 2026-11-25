"""Compare the F/C private-key engine with actual XSA sessions on sampled synthetic traces."""

from collections import Counter
import json
import os
import torch
from ndm.memory_program.direct import DirectTrust
from ..x47_xsa_attention.workbench_native import Live, EXE_SHA
from ..x48_muon_continuation.data import check
from .literal import MIN_AGE
from .trace import TraceHead
from .values import LiveValueHead


@torch.no_grad()
def run():
    check(
        os.getenv("CUDA_VISIBLE_DEVICES") == "" and not os.getenv("GPU_LEASE_HELD"),
        "CPU only",
    )
    stats = Counter()
    torch.manual_seed(4901)
    word = (b"\0\xffctx\x1e.private-key.?" * 15)[:271]
    h = torch.randn(len(word), 8)
    z = torch.randn(len(word), 256)
    kwargs = dict(seed=4901, document=17, visit=1, horizon=128)
    for learned in (False, True):
        head = TraceHead(LiveValueHead(DirectTrust(8)), learned=learned)
        if (
            learned
        ):  # deliberately nonzero synthetic policy, not a natural training result
            head.ask.weight.normal_(0, 0.55)
            head.filing.weight.normal_(0, 0.55)
        r = head(word, h, z, **kwargs)
        with Live() as obs, Live() as tags:
            for t, q in enumerate(r.queried):
                obs.append(word[t : t + 1])
                stats["observed_bytes"] += 1
                if t:
                    tags.append(r.filed[t - 1 : t])
                    stats["filed_symbols"] += 1
                check(
                    obs.visible == t + 1 and tags.visible == t,
                    "private and observed clocks mixed",
                )
                if t == 103:
                    for src in (obs, tags):
                        check(
                            src._rpc("seal") == {"sealed": src.visible}, "seal failed"
                        )
                lo = max(1, t + 1 - 128) - 1
                hi = t - MIN_AGE
                j = None
                if hi > lo:
                    ans = tags._rpc(f"locate {q:02x} {lo} {hi} 4096")
                    loc = ans.get("positions")
                    check(
                        isinstance(loc, list)
                        and loc == sorted(set(loc))
                        and all(type(k) is int and lo <= k < hi for k in loc),
                        "bad trace locate",
                    )
                    if loc:
                        j = loc[-1] + 1
                    stats["native_queries"] += 1
                check(j == r.source[t], "private lookup mismatch")
                stats["frontiers"] += 1
                if j is not None:
                    check(
                        obs._rpc(f"byte {j}").get("byte") == word[j],
                        "trace key substituted for observed follower",
                    )
                    stats["bound_byte_reads"] += 1
        check(obs.proc.returncode == tags.proc.returncode == 0, "unclean exit")
        stats["closed_sessions"] += 2
    return dict(
        scope="synthetic trace namespace/lookup parity only",
        binary_sha256=EXE_SHA,
        counts=dict(stats),
    )


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
