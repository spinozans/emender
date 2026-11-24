"""S10: split the observed prompt, never a speculative future, at the known FP32 recurrent width."""

import argparse, os
from pathlib import Path
from time import perf_counter_ns as clock
import torch
from ndm.memory_program.resident_host import ResidentArena
from .wide_host_bench import Bound
from .long_prefill_bench import SingleCache, independent, prepare_long
from .serving_bench import measure
from .workbench_run import dump


def arena(net, b, widths):
    return ResidentArena(
        net, b, 2048, widths, byte_budget=256 << 20, private_budget=64 << 20, segmented_prefill=True
    )


class Driver(Bound):
    def __init__(self, pool, n, s, l, serial):
        self.serial = serial
        self.prior = pool.backend.prefill_work.copy()
        super().__init__(pool, n, s, l)

    def prefill(self, hs, words):
        if not self.serial:
            return self.arena.prefill(hs, words)
        if len(set(map(len, words))) != 1:
            raise ValueError("fixed service comparator expects equal prompt lengths")
        out = self.arena.prefill(hs, [w[:1] for w in words])
        for j in range(1, len(words[0])):
            out = self.arena.advance(hs, [j] * len(hs), [w[j] for w in words])
        return out

    def report(self):
        return dict(super().report(), native_prompt_pieces=self.arena.backend.prefill_work - self.prior)


@torch.no_grad()
def check(run, n, ws):
    measured = []
    bad = []
    t0 = clock()

    def mark(tag, k):
        dump(
            run / "progress.json",
            dict(
                cell="qualification",
                phase=tag,
                observed_prefix_bytes=k,
                total_emitted=0,
                server_pid=os.getpid(),
            ),
        )

    def compare(tag, xs, ys):
        row = {}
        assert xs.keys() == ys.keys()
        for k, x in xs.items():
            y = ys[k]
            assert x.shape == y.shape
            finite = (x.isfinite() & y.isfinite()).all().item()
            ok = finite and torch.allclose(x, y, atol=5e-4, rtol=2e-4)
            row[k] = dict(elements=x.numel(), max_abs=(x - y).abs().max().item() if finite else None, ok=ok)
            if not ok:
                bad.append((tag, k))
        argmax = bool(xs["base"].argmax() == ys["base"].argmax())
        measured.append(dict(tag=tag, fields=row, argmax=argmax))
        if not argmax:
            bad.append((tag, "argmax"))

    with arena(n, 2, (1, 2)) as a:

        def actual(h):
            return {k: t[h.slot] for k, t in a.slots.items()}

        def oracle(c, out):
            return dict(
                hidden=out[0],
                base=out[1],
                recurrent=torch.stack([s["recurrent_state"][0] for s in c.values()]),
                **{
                    k: torch.stack([s["conv_state"][i][0] for s in c.values()])
                    for i, k in enumerate(("q", "k", "v"))
                },
            )

        word = ws[0][:2048]
        s = a.admit("segmented")
        r = a.admit("graph-serial")
        a.prefill([s], [word])
        a.prefill([r], [word[:1]])
        for j in range(1, len(word)):
            a.advance([r], [j], [word[j]])
            if not (j + 1) % 128:
                mark("serial-graph", j + 1)
        compare("2048-token-stream", actual(s), actual(r))
        a.clear()
        assert all(not t.count_nonzero() for t in a.slots.values())
        s = a.admit("native-oracle")
        word = ws[1][:130]
        c = SingleCache()
        a.prefill([s], [word[:129]])
        for j in range(129):
            out = independent(n, c, word[j : j + 1])
        compare("129-native-singletons", actual(s), oracle(c, out))
        a.advance([s], [129], [word[129]])
        compare("130-native-singletons", actual(s), oracle(c, independent(n, c, word[129:130])))
        a.clear()
        mark("independent-native", 130)
        hs = [a.admit(str(i)) for i in range(2)]
        words = [ws[i][:t] for i, t in [(2, 65), (3, 513)]]
        a.prefill(hs, words)
        cs = [SingleCache(), SingleCache()]
        for i, w in enumerate(words):
            for pos in range(0, len(w), 31):
                out = independent(n, cs[i], w[pos : pos + 31])
            compare(f"ragged-{len(w)}", actual(hs[i]), oracle(cs[i], out))
        a.advance(hs[::-1], [513, 65], [255, 0])
        for i, z in enumerate([0, 255]):
            compare(
                f"reordered-{len(words[i]) + 1}",
                actual(hs[i]),
                oracle(cs[i], independent(n, cs[i], bytes([z]))),
            )
        peer = {k: x.clone() for k, x in actual(hs[1]).items()}
        old = hs[0]
        a.release([old])
        assert all(not t.count_nonzero() for t in actual(old).values())
        x = a.admit("new")
        assert x.slot == old.slot
        a.prefill([x], [ws[16][:130]])
        try:
            a.read([old], [130])
            raise AssertionError("stale ticket accepted")
        except ValueError:
            pass
        assert all(torch.equal(v, actual(hs[1])[k]) for k, v in peer.items())
        a.clear()
        assert all(not t.count_nonzero() for t in a.slots.values())
        work = dict(a.backend.prefill_work)
    dump(
        run / "device-parity.json",
        dict(
            checks=measured,
            failed=bad,
            qualified=not bad,
            old_slot_scrubbed=True,
            peer_unchanged=True,
            native_prompt_pieces=work,
            wall_ns=clock() - t0,
        ),
    )
    if bad:
        raise ValueError(f"segmented prefill failed: {bad}; no service timing")


@torch.no_grad()
def main(run):
    n, eager, cpu, rows, ws, man = prepare_long(run)
    check(run, n, ws)
    t = clock()
    total = 0
    with arena(n, 16, (1, 16)) as pool:
        torch.cuda.synchronize()
        dump(
            run / "setup.json",
            dict(wall_ns=clock() - t, tensor_bytes=pool.tensor_bytes, private_bytes=pool.backend.pool_bytes),
        )
        for block in (0, 1):
            for b in (1, 16):
                ix = list(range(16 * block, 16 * block + b))
                arms = [
                    (m, h) for m in ("direct", "fused-greedy", "fused-sample") for h in ("serial", "segment")
                ]
                for m, h in arms[::-1] if block else arms:

                    def host(n, s, l):
                        return Driver(pool, n, s, l, h == "serial")

                    total += measure(
                        run,
                        n,
                        eager if m == "direct" else cpu,
                        m,
                        block,
                        [rows[i] for i in ix],
                        [ws[i] for i in ix],
                        ix,
                        man,
                        total,
                        label=f"{h}-{m}",
                        host_factory=host,
                        prefix=512,
                        length=8,
                    )
    dump(
        run / "progress.json", dict(cell="all", phase="complete", total_emitted=total, server_pid=os.getpid())
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    main(ap.parse_args().run)
