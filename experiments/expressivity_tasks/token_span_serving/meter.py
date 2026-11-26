"""Measure serial prefix-to-next-logits latency on invented IDs. Not a quality test.

No GPU/data/weights from the language panel. Cold and repeated-page passes each
include index querying, fetch/encoding as needed, both host updates and vocab head.
"""

import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

from experiments.expressivity_tasks.retrieved_span_effects.natural_model import (
    Shape,
    TokenLM,
)
from .archive import Archive, digest
from .session import FrozenHost


CELLS = (
    ("gdn2", "local"),
    ("gdn2", "g-state"),
    ("gdn2", "g-effect"),
    ("e97", "local"),
    ("e97", "e-state"),
    ("e97", "e-effect"),
    ("gdn2", "mix-effect"),
)
BUDGET = 2 * 1024**2
BLOCKS = 5


def fixture(sh):
    rng = np.random.default_rng(5512)
    docs = []
    live = []
    for topic in range(128):
        anchors = np.arange(
            topic * sh.prefix + 1, (topic + 1) * sh.prefix + 1, dtype=np.int32
        )
        for row in range(sh.hits):
            n = 12 - 2 * row
            ids = np.r_[anchors[:n], rng.integers(3000, sh.vocab, size=sh.page - n)]
            docs.append(ids.astype("<u2").tobytes())
        if topic < 5:
            live.append(
                np.r_[
                    anchors, rng.integers(3000, sh.vocab, size=sh.time - sh.prefix)
                ].tolist()
            )
    codec = digest(b"invented little-endian uint16 IDs, v1")
    a = Archive(
        docs,
        lambda x: np.frombuffer(x, dtype="<u2"),
        lambda i: i.to_bytes(2, "little"),
        tokenizer_sha256=codec,
        vocab=sh.vocab,
        page=sh.page,
    )
    return a, live


def drive(host, live):
    sh = host._net.sh
    before = (host.encodes, host.reuses)
    hits = 0
    front_s = 0
    other_s = 0
    obs = 0
    start = time.perf_counter()
    for i, ids in enumerate(live):
        s = host.session(digest(f"invented live {i}".encode()))
        for j, t in enumerate(ids):
            a = time.perf_counter_ns()
            s.observe(t)
            b = time.perf_counter_ns()
            if j == sh.prefix - 1:
                front_s += (b - a) * 1e-9
                hits += len(s.last_hits)
            else:
                other_s += (b - a) * 1e-9
        obs += s.observations
    elapsed = time.perf_counter() - start
    if hits != len(live) * sh.hits or obs != len(live) * sh.time:
        raise RuntimeError("unexpected fixture work")
    return dict(
        ms_per_token=1000 * elapsed / obs,
        refresh_ms=1000 * front_s / len(live),
        ordinary_ms=1000 * other_s / (obs - len(live)),
        encodes=host.encodes - before[0],
        reuses=host.reuses - before[1],
        lru_tensor_bytes=host.cache_payload_bytes,
    )


def main():
    if (
        os.getenv("CUDA_VISIBLE_DEVICES") != ""
        or os.getenv("GPU_LEASE_HELD")
        or not __debug__
        or any(
            os.getenv(n) != "1"
            for n in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        )
    ):
        raise RuntimeError("single-thread CPU with no visible/leased device required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sh = Shape()
    a, live = fixture(sh)
    clocks = {}
    largest = 0.0
    for family, arm in CELLS:
        torch.manual_seed(5513)
        m = TokenLM(sh, family, arm)
        impl = [
            FrozenHost(m, a, tokenizer_sha256=a.tokenizer, cache_bytes=n)
            for n in (0, BUDGET)
        ]
        sessions = [p.session(digest(b"untimed fixture")) for p in impl]
        for t in live[-1]:
            x, y = [s.observe(t) for s in sessions]
            torch.testing.assert_close(x, y, rtol=4e-5, atol=3e-6)
            largest = max(largest, (x - y).abs().max().item())
        times = {}
        for j in range(BLOCKS):
            for i in (0, 1) if not j % 2 else (1, 0):
                server = impl[i]
                server.clear_cache()
                mode = ("replay", "cache")[i]
                for label in ("cold", "repeat"):
                    out = drive(server, live[:4])
                    times.setdefault(mode + "/" + label, []).append(out)
                    if (
                        i
                        and label == "repeat"
                        and arm != "local"
                        and (out["encodes"] != 0 or out["reuses"] != 4 * sh.hits)
                    ):
                        raise RuntimeError("warm workload missed expected reuse")
        c = {}
        for mode, rows in times.items():
            c[mode] = {k: statistics.median(z[k] for z in rows) for k in rows[0]}
        clocks[family + "/" + arm] = c
    return dict(
        schema="fixed-cpu-stream-clock-v1",
        geometry=vars(sh),
        torch=str(torch.__version__),
        threads=1,
        device="cpu",
        dtype="float32",
        mode="serial next logits, no sampling, invented IDs and random weights only",
        cells=clocks,
        parameter_count=sum(p.numel() for p in m.parameters()),
        pre_lookup_index_array_bytes=a.index_payload_bytes,
        max_lru_bytes=BUDGET,
        archive_pages=128 * sh.hits,
        windows_per_pass=4,
        pass_observations=4 * sh.time,
        blocks=BLOCKS,
        compiled_replay_logit_max_abs=largest,
        archive_natural_docs=0,
        card_seconds=0,
    )


if __name__ == "__main__":
    p = Path(sys.argv[1])
    if p.exists():
        raise FileExistsError(p)
    ans = main()
    with p.open("x") as f:
        json.dump(ans, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print("serial timing completed, location:", p)
