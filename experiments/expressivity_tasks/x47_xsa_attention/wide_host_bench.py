"""Actual dense wide-shape service test. Not a 2B trained checkpoint or a quality comparison."""

from collections import Counter
import argparse, json, os
from pathlib import Path
from time import perf_counter_ns as clock
import torch
from ndm.memory_program.resident_host import ResidentArena
from .wide_host import build
from .resident_host_check import memory
from .cached_host_bench import HostDriver
from .large_host_check import stock
from .large_host_bench import prepare_large, adapt, bare, PREFIX, LENGTH
from .serving_bench import measure
from .workbench_run import dump


class Bound(HostDriver):
    def __init__(self, pool, net, ss, lives):
        if pool.net is not net or not pool.empty:
            raise ValueError("wrong resident model or occupied comparison")
        self.before = pool.stats.copy()
        self.before_dispatch = pool.backend.work.copy()
        self.buckets = {b: g.work.copy() for b, g in pool.backend.buckets.items()}
        super().__init__(pool, ss, lives, adapt)

    def clear(self):
        if self.bound:
            self.arena.release(self.bound)
            self.bound = []

    def report(self):
        self.clear()
        p = self.arena
        assert p.empty and all(t.count_nonzero().item() == 0 for t in p.slots.values())
        s = p.stats - self.before
        return dict(
            tensor_bytes=p.tensor_bytes,
            encoded_tokens=s["step_tokens"] + s["prefill_tokens"],
            stats=s,
            dispatch=p.backend.work - self.before_dispatch,
            buckets={b: g.work - self.buckets[b] for b, g in p.backend.buckets.items()},
            all_released_zero=True,
        )


@torch.no_grad()
def qualify(small, wide, windows):
    p = ResidentArena(wide, 3, 64, [2], byte_budget=1 << 30, private_budget=256 << 20)
    hs = [p.admit(str(i)) for i in range(3)]
    ws = [windows[i][:n] for i, n in enumerate([1, 17, 32])]
    mx = Counter()
    comp = Counter()

    def compare(out, ix):
        for j, i in enumerate(ix):
            h, lp = stock(small, [ws[i]])
            for k, x, y in [("hidden_all_four", out[0][j], h[0].repeat(4)), ("logp", out[1][j], lp[0])]:
                torch.testing.assert_close(
                    x, y, atol=5e-4, rtol=2e-4, msg=lambda msg: f"{k} visible={len(ws[i])}: {msg}"
                )
                mx[k] = max(mx[k], (x - y).abs().max().item())
                comp[k] += x.numel()
            assert (
                out[1][j].argmax() == lp[0].argmax()
                and torch.isfinite(out[0][j]).all()
                and torch.isfinite(out[1][j]).all()
            )

    try:
        compare(p.prefill(hs, ws), [0, 1, 2])
        ix = [2, 0]
        bs = [0, 255]
        out = p.advance([hs[i] for i in ix], [len(ws[i]) for i in ix], bs)
        for i, b in zip(ix, bs, strict=True):
            ws[i] += bytes([b])
        compare(out, ix)
        old = hs[0]
        p.release([old])
        assert all(t[old.slot].count_nonzero().item() == 0 for t in p.slots.values())
        hs[0] = p.admit("new")
        ws[0] = windows[16][:11]
        compare(p.prefill([hs[0]], [ws[0]]), [0])
        compare(p.read([hs[2]], [33]), [2])
        assert p.backend.buckets[2].work["real_replays"] == 1
        p.clear()
        assert all(t.count_nonzero().item() == 0 for t in p.slots.values())
        return dict(compared=comp, max_error=mx, scrubbed=True, distinct_argmax_agree=True)
    finally:
        p.clear()
        p.close()


@torch.no_grad()
def panel(run, net, size, eager, fused, selected, windows, manifest, total):
    base = memory()
    t0 = clock()
    pool = ResidentArena(
        net,
        16,
        64,
        (1, 16),
        byte_budget=(1 << 30 if size == "wide" else 256 << 20),
        private_budget=(256 if size == "wide" else 64) << 20,
    )
    torch.cuda.synchronize()
    setup = dict(
        prep_ns=clock() - t0,
        baseline=base,
        after_prepare=memory(),
        tensor_bytes=pool.tensor_bytes,
        private_bytes=pool.backend.pool_bytes,
        buckets={b: g.setup for b, g in pool.backend.buckets.items()},
        parameters=sum(p.numel() for p in net.parameters()),
    )
    dump(run / f"{size}-setup.json", setup)

    def host(model, ss, lives):
        return Bound(pool, model, ss, lives)

    try:
        for block in range(2):
            for b in (1, 16):
                ix = list(range(16 * block, 16 * block + b))
                ws = [windows[i] for i in ix]
                arms = ["bare", "direct", "fused-greedy", "fused-sample"]
                for m in arms[::-1] if block else arms:
                    label = f"{size}-{m}"
                    if m == "bare":
                        total += bare(run, net, host, block, ix, ws, total, label)
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
                            host_factory=host,
                            length=LENGTH,
                            prefix=PREFIX,
                        )
    finally:
        pool.clear()
        pool.close()
    dump(run / f"{size}-closed.json", dict(memory=memory(), slots_freed=not pool.slots))
    return total


@torch.no_grad()
def main(run):
    small, eager, fused, selected, windows, manifest = prepare_large(run)
    torch.cuda.reset_peak_memory_stats()
    t0 = clock()
    wide, rule = build(small)
    torch.cuda.synchronize()
    rule.update(
        build_verify_ns=clock() - t0,
        parameter_bytes=sum(p.numel() * p.element_size() for p in wide.parameters()),
        peak_build_allocated=torch.cuda.max_memory_allocated(),
        width=wide.width,
        layers=len(wide.blocks),
    )
    dump(run / "construction.json", rule)
    t0 = clock()
    q = qualify(small, wide, windows)
    torch.cuda.synchronize()
    q["wall_ns"] = clock() - t0
    dump(run / "device-parity.json", q)
    print("actual dense four-copy host qualified", flush=True)
    meta = json.loads((run / "identity.json").read_text())
    meta.update(
        profile="known-function dense shape stress",
        wide_parameters=rule["parameters_verified"],
        size_order=["wide", "small"],
    )
    dump(run / "identity.json", meta)
    t0 = clock()
    small.cpu()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    xfer = clock() - t0
    assert all(p.device.type == "cpu" for p in small.parameters())
    total = panel(run, wide, "wide", eager, fused, selected, windows, manifest, 0)
    del wide
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    pre = memory()
    t0 = clock()
    small.cuda()
    torch.cuda.synchronize()
    back = clock() - t0
    dump(run / "transfers.json", dict(small_to_cpu_ns=xfer, after_wide_removed=pre, small_to_gpu_ns=back))
    total = panel(run, small, "small", eager, fused, selected, windows, manifest, total)
    dump(
        run / "progress.json", dict(cell="all", phase="complete", total_emitted=total, server_pid=os.getpid())
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    main(p.parse_args().run)
