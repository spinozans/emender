"""S11: measured odd-width clients, exact real-slot covers, no placeholder work."""

from collections import Counter
import argparse, copy, os, json
from pathlib import Path
from time import perf_counter_ns as clock
import torch
from ndm.memory_program.resident_host import ResidentArena
from .large_host_bench import prepare_large
from .large_host_check import make
from .serving_bench import measure
from .wide_host_bench import Bound
from .workbench_run import dump

WIDTHS = (1, 4, 8, 16)
SIZES = (1, 3, 7, 13, 15, 16)


@torch.no_grad()
def qualify(pool, net, windows):
    a = pool
    b = make(net, 16)
    a.set_cover(True)
    aa = [a.admit(str(i)) for i in range(16)]
    bb = [b.admit(str(i)) for i in range(16)]
    ws = [w[: (i % 4) * 3 + 1] for i, w in enumerate(windows[:16])]
    at = list(map(len, ws))
    err = Counter()
    checked = Counter()
    seen = Counter()
    a.prefill(aa, ws)
    b.prefill(bb, ws)

    def compare(k, x, y):
        torch.testing.assert_close(x, y, atol=5e-4, rtol=2e-4, msg=lambda m: k + ": " + m)
        err[k] = max(err[k], (x - y).abs().max().item())
        checked[k] += x.numel()

    def store():
        for k, t in b.slots.items():
            compare(k, a.slots[k][:16], t)
            assert a.slots[k][16:].count_nonzero().item() == 0

    store()
    before = {w: g.work.copy() for w, g in a.backend.buckets.items()}
    for t, n in enumerate((3, 7, 13, 15, 1, 16)):
        ix = [(5 * i + 3 * t) % 16 for i in range(n)]
        old = [at[i] for i in ix]
        bs = [windows[i][at[i]] for i in ix]
        out = a.advance([aa[i] for i in ix], old, bs)
        ref = b.advance([bb[i] for i in ix], old, bs)
        for j, k in enumerate(("output_hidden", "output_base")):
            compare(k, out[j], ref[j])
        assert torch.equal(out[1].argmax(1), ref[1].argmax(1))
        store()
        seen.update(a.backend.plans[n])
        for i in ix:
            at[i] += 1
    assert (
        all(
            g.work["real_replays"] - before[w]["real_replays"] == seen[w]
            for w, g in a.backend.buckets.items()
        )
        and sum(w * c for w, c in seen.items()) == 55
    )
    old = aa[0]
    a.release([old])
    b.release([bb[0]])
    assert all(t[old.slot].count_nonzero().item() == 0 for t in a.slots.values())
    aa[0] = a.admit("new")
    bb[0] = b.admit("new")
    a.prefill(aa[:1], [windows[16][:32]])
    b.prefill(bb[:1], [windows[16][:32]])
    store()
    try:
        a.read([old], [32])
        raise AssertionError("stale ticket")
    except ValueError:
        pass
    a.clear()
    b.clear()
    assert all(not t.count_nonzero().item() for t in a.slots.values())
    return dict(
        max_errors=err, elements=checked, replays=seen, all_argmax=True, slots_reused_and_scrubbed=True
    )


@torch.no_grad()
def main(run):
    net, eager, fused, selected, ws, man = prepare_large(run)
    cpu = copy.deepcopy(eager).cpu()
    del fused
    idn = json.loads((run / "identity.json").read_text())
    idn.update(generated_length=12, step_widths=WIDTHS, cover_launch_limit=6)
    dump(run / "identity.json", idn)
    t = clock()
    total = 0
    with ResidentArena(net, 16, 64, WIDTHS, byte_budget=256 << 20, private_budget=64 << 20) as pool:
        torch.cuda.synchronize()
        dump(
            run / "setup.json",
            dict(
                wall_ns=clock() - t,
                tensor_bytes=pool.tensor_bytes,
                private_bytes=pool.backend.pool_bytes,
                buckets={w: g.setup for w, g in pool.backend.buckets.items()},
                plans=pool.backend.plans,
            ),
        )
        dump(run / "device-parity.json", qualify(pool, net, ws))

        def host(n, s, l):
            return Bound(pool, n, s, l)

        for block in (0, 1):
            for b in SIZES:
                ix = list(range(16 * block, 16 * block + b))
                arms = [(m, h) for m in ("direct", "fused-sample") for h in ("exact", "cover")]
                for m, h in arms[::-1] if block else arms:
                    pool.set_cover(h == "cover")
                    total += measure(
                        run,
                        net,
                        eager if m == "direct" else cpu,
                        m,
                        block,
                        [selected[i] for i in ix],
                        [ws[i] for i in ix],
                        ix,
                        man,
                        total,
                        label=f"{h}-{m}",
                        host_factory=host,
                        prefix=32,
                        length=12,
                    )
    dump(
        run / "progress.json", dict(cell="all", phase="complete", total_emitted=total, server_pid=os.getpid())
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    main(ap.parse_args().run)
