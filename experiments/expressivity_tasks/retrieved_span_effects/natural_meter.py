"""Select natural panel geometry/dose from invented inputs only, never probe loss."""

import gc
from functools import partial
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .cuda_step import Step, eval_losses, invented
from .meter import await_ack, sync_clock
from .natural_model import ARMS, HOSTS, Shape
from .natural_oracle import dense_reference
from .token_postings import fetch, index, locate
from .window import need, put


def clean():
    gc.collect()
    torch.cuda.empty_cache()


def main(d):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    sh = Shape()
    work = 0
    begin = time.monotonic()

    def tick(label):
        nonlocal work
        work += 1
        put(d / "progress.json", dict(units=work, label=label, pid=os.getpid()))

    with (d / "device-tests.log").open("xb") as f:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                str(Path(__file__).with_name("test_natural.py")),
            ],
            env=dict(os.environ, SPAN_TEST_DEVICE="cuda"),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    tick("natural implementation device parity")
    # Check that captured warmup and dummy data do not train the real model; fresh packets, not stale inputs.
    small = Shape(vocab=251, width=32, key=8, heads=2, time=16, prefix=4, page=8)
    max_diff = 0.0
    for host, arm in [("gdn2", "g-effect"), ("e97", "mix-effect")]:
        eager = Step(small, 8, host, arm, 7740, graph=False)
        graph = Step(small, 8, host, arm, 7740)
        for a, b in zip(
            eager.model.parameters(), graph.model.parameters(), strict=True
        ):
            need(torch.equal(a, b), "warmup changed starting weights")
        for j in range(3):
            packet = invented(small, 8, seed=7760 + j)
            eager.train(packet, 0.0003 * (1 + j))
            graph.train(packet, 0.0003 * (1 + j))
            torch.testing.assert_close(
                eager.stat[:2], graph.stat[:2], atol=0.005, rtol=3e-5
            )
            for a, b in zip(
                eager.model.parameters(), graph.model.parameters(), strict=True
            ):
                delta = (a - b).abs().max().item()
                max_diff = max(max_diff, delta)
                torch.testing.assert_close(a, b, atol=5e-5, rtol=4e-4)
        del eager, graph
        clean()
    tick("three distinct graph optimizer updates")
    sizes = {}
    for batch in (16, 32, 64):
        graph = Step(sh, batch, "gdn2", "g-effect", 7751)
        packet = invented(sh, batch)
        sizes[str(batch)] = sync_clock(partial(graph.train, packet, 0.0005), 8, 5)
        del graph, packet
        clean()
        tick("batch " + str(batch))
    total = torch.cuda.get_device_properties(0).total_memory
    batch = max(
        (
            b
            for b in (16, 32, 64)
            if sizes[str(b)]["base_alloc_bytes"] + sizes[str(b)]["extra_peak_bytes"]
            < 0.65 * total
        ),
        key=lambda b: b / sizes[str(b)]["ms"],
    )
    packet = invented(sh, batch)
    clocks = {}
    for host in HOSTS:
        for arm in ARMS:
            graph = Step(sh, batch, host, arm, 7752)
            tr = sync_clock(partial(graph.train, packet, 0.0005), 8, 5)
            trial = "wrong_effect" if arm.endswith("effect") else "intact"
            ev = sync_clock(partial(eval_losses, graph.model, packet, trial), 8, 5)
            clocks[host + "/" + arm] = dict(train=tr, eval=ev)
            del graph
            clean()
            tick(host + "/" + arm)
    # Same precision, batch and optimizer. Graph dense too: don't credit launch avoidance to recurrence math.
    with dense_reference():
        ref = Step(sh, batch, "gdn2", "g-effect", 7752)
        dense = sync_clock(partial(ref.train, packet, 0.0005), 6, 5)
    del ref
    clean()
    tick("graph dense baseline")
    rng = np.random.default_rng(7716)
    docs = rng.integers(1, sh.vocab, (2048, 256), dtype=np.int32)
    im = index(docs, sh.page).tensors("cuda")
    prefix = torch.from_numpy(docs[:128, : sh.prefix].copy()).cuda()
    on = torch.ones(len(prefix), device="cuda", dtype=torch.bool)
    lookup = sync_clock(lambda: fetch(im, locate(im, prefix, on)[0]), 16, 5)
    tick("real postings + fetch on invented tokens")
    ok = clocks["gdn2/g-effect"]["train"]["ms"] < dense["ms"]
    put(
        d / "complete.json",
        dict(
            units=work,
            admitted=ok,
            batch=batch,
            sweep=sizes,
            clocks=clocks,
            dense_graph=dense,
            search_fetch_128=lookup,
            graph_parameter_max_abs=max_diff,
            wall_s=time.monotonic() - begin,
            device=torch.cuda.get_device_name(),
            natural_documents=0,
        ),
    )
    await_ack(d)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
