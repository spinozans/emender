"""S5 same retained host math and fresh programs, change the single-token dispatch mechanism."""

import argparse
from collections import Counter
import os
from pathlib import Path
from time import perf_counter_ns as clock
import torch

from ndm.memory_program.host_graph import PreparedArena, StepGraph
from .cached_host_bench import HostDriver
from .large_host_bench import adapt, prepare_large, bare, PREFIX, LENGTH
from .large_host_check import stock, make
from .serving_bench import measure
from .workbench_run import dump


def prepared(net, capacity, b, capture):
    backend = StepGraph(net)
    a = PreparedArena(backend, capacity, 64, 256 << 20, reserve=b)
    backend.build(a, b, capture=capture)
    return a


class GraphDriver(HostDriver):
    capture = True

    def __init__(self, net, ss, lives):
        a = prepared(net, len(ss), len(ss), self.capture)
        super().__init__(a, ss, lives, adapt)

    def clear(self):
        self.arena.clear()
        self.arena.backend.close()

    def report(self):
        t = clock()
        out = super().report()
        g = self.arena.backend
        torch.cuda.synchronize()
        return dict(out, setup=g.setup, traffic=g.stats, dispatch=g.work, retire_verify_ns=clock() - t)


class EagerDriver(GraphDriver):
    capture = False


@torch.no_grad()
def qualify(net):
    a = prepared(net, 3, 2, True)
    e = make(net, 3)
    h = [a.admit(str(i)) for i in range(3)]
    rh = [e.admit(str(i)) for i in range(3)]
    ws = [b"\x00", b"\x1e\xffabcde", b"different-prefix"]
    a.prefill(h, ws)
    e.prefill(rh, ws)
    mx = Counter()
    comp = Counter()

    def close(k, x, y):
        torch.testing.assert_close(x, y, atol=5e-4, rtol=2e-4, msg=lambda msg: k + ": " + msg)
        mx[k] = max(mx[k], (x - y).abs().max().item())
        comp[k] += x.numel()

    saved = None
    for n, (ix, bs) in enumerate([([1, 0], [13, 0]), ([0, 1], [255, 61]), ([2], [30]), ([2, 1], [33, 255])]):
        old = [len(ws[i]) for i in ix]
        out = a.advance([h[i] for i in ix], old, bs)
        oracle = e.advance([rh[i] for i in ix], old, bs)
        for k, x, y in zip(("hidden", "byte_logp"), out, oracle, strict=True):
            close(k, x, y)
        for j, (i, b) in enumerate(zip(ix, bs, strict=True)):
            ws[i] += bytes([b])
            oh, op = stock(net, [ws[i]])
            close("stock_hidden", out[0][j], oh[0])
            close("stock_base", out[1][j], op[0])
            assert op[0].argmax() == out[1][j].argmax()
        for k, x in e.slots.items():
            close("arena_" + k, a.slots[k][:3], x)
        if saved is not None:
            assert torch.equal(saved[0], saved[1])
        if n == 0:
            saved = (out[0], out[0].clone())
        if n == 2:
            a.release([h[1]])
            e.release([rh[1]])
            assert all(
                t[h[1].slot].count_nonzero().item() == 0 and t[3:].count_nonzero().item() == 0
                for t in a.slots.values()
            )
            assert all(x.count_nonzero().item() == 0 for x in (*a.backend.static, *a.backend.output))
            for k, x in e.slots.items():
                assert torch.equal(a.slots[k][[0, 2]], x[[0, 2]])
            h[1] = a.admit("fresh")
            rh[1] = e.admit("fresh")
            ws[1] = b"\x00\\\x1e"
            a.prefill([h[1]], [ws[1]])
            e.prefill([rh[1]], [ws[1]])
    try:
        with torch.cuda.stream(torch.cuda.Stream()):
            a.backend._binding(a.slots)
        raise AssertionError("foreign stream accepted")
    except ValueError:
        pass
    a.clear()
    e.clear()
    a.backend.close()
    assert all(t.count_nonzero().item() == 0 for t in a.slots.values())
    assert (
        a.backend.work["real_replays"] == 3
        and a.backend.work["width_miss"] == 1
        and a.backend.work["neutral_retire_replays"] == 2
    )
    return dict(max_error=mx, compared=comp, work=a.backend.work, setup=a.backend.setup, scrubbed=True)


def main(run):
    net, eager, fused, selected, windows, manifest = prepare_large(run)
    start = clock()
    q = qualify(net)
    torch.cuda.synchronize()
    q["qualification_s"] = (clock() - start) / 1e9
    dump(run / "device-parity.json", q)
    print("host-step graph qualified", flush=True)
    total = 0
    for block in range(2):
        for b in (1, 16):
            ix = list(range(16 * block, 16 * block + b))
            ws = [windows[i] for i in ix]
            arms = [
                (m, e) for m in ("bare", "direct", "fused-greedy", "fused-sample") for e in ("eager", "graph")
            ]
            for m, e in arms[::-1] if block else arms:
                label = f"{e}-{m}"
                engine = GraphDriver if e == "graph" else EagerDriver
                if m == "bare":
                    total += bare(run, net, engine, block, ix, ws, total, label)
                else:
                    total += measure(
                        run,
                        net,
                        eager if m == "direct" else fused,
                        m,
                        block,
                        [selected[i] for i in ix],
                        ws,
                        ix,
                        manifest,
                        total,
                        label=label,
                        host_factory=engine,
                        length=LENGTH,
                        prefix=PREFIX,
                    )
    dump(
        run / "progress.json", dict(cell="all", phase="complete", total_emitted=total, server_pid=os.getpid())
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    main(p.parse_args().run)
