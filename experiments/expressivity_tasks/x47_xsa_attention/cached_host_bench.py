"""S3: separate warm host-only clocks from complete paired direct/program decode."""

import argparse
from collections import defaultdict
from pathlib import Path
from time import perf_counter_ns as clock

import torch
from .cached_host_check import check_cache, make_cache
from .serving_bench import causal_last, prepare, measure, PREFIX, LENGTH
from .workbench_run import dump
from .workbench_witness import require


class HostDriver:
    """Bind owned arenas to current observations. Only read a full tape at admission."""

    def __init__(self, arena, sessions, lives, adapt=lambda h, p: (h, p)):
        self.arena, self.ss, self.lives, self.adapt = arena, sessions, lives, adapt
        self.bound = [arena.admit(s.id) for s in sessions]
        self.at = [0] * len(sessions)

    def prefill(self, handles, words):
        return self.arena.prefill(handles, words)

    def inputs(self, ids):
        if len(set(ids)) != len(ids) or not ids:
            raise ValueError("invalid host batch")
        groups = defaultdict(list)
        for j, i in enumerate(ids):
            s = self.ss[i]
            old = self.at[i]
            n = s.visible
            if (
                self.bound[i].session != s.id
                or len(self.lives[i].data) != n
                or n < 1
                or s.pending
                or (old and n not in (old, old + 1))
            ):
                raise ValueError("causal host binding/frontier mismatch")
            groups["prefill" if not old else "read" if old == n else "advance"].append((j, i))
        ordered = [None] * len(ids)
        for tag, items in groups.items():
            js, ii = zip(*items)
            handles = [self.bound[i] for i in ii]
            old = [self.at[i] for i in ii]
            if tag == "prefill":
                out = self.prefill(handles, [bytes(self.lives[i].data) for i in ii])
            elif tag == "read":
                out = self.arena.read(handles, old)
            else:
                out = self.arena.advance(handles, old, [self.lives[i].data[-1] for i in ii])
            for i in ii:
                self.at[i] = self.ss[i].visible
            if len(groups) == 1:
                return self.adapt(*out)
            for j, h, p in zip(js, *out, strict=True):
                ordered[j] = (h, p)
        return self.adapt(*(torch.stack(v) for v in zip(*ordered)))

    def clear(self):
        self.arena.clear()

    def report(self):
        self.clear()
        zero = all(t.count_nonzero().item() == 0 for t in self.arena.slots.values())
        require(zero, "released host state not zero")
        return dict(
            tensor_bytes=self.arena.tensor_bytes,
            encoded_tokens=self.arena.stats["prefill_tokens"] + self.arena.stats["step_tokens"],
            stats=self.arena.stats,
            all_released_zero=zero,
        )


def cached(net, ss, lives):
    return HostDriver(make_cache(net, len(ss)), ss, lives)


@torch.no_grad()
def host_only(net, windows):
    rows = []
    for b in (1, 16):
        ws = windows[:b]
        for name in ("full", "cache"):
            cache = make_cache(net, b) if name == "cache" else None
            ss = [cache.admit(str(i)) for i in range(b)] if cache else None
            # Warm both exact measurement methods first, then create fresh admissions.
            if cache:
                cache.prefill(ss, [w[:PREFIX] for w in ws])
                cache.advance(ss, [PREFIX] * b, [w[PREFIX] for w in ws])
                cache.clear()
                ss = [cache.admit(str(i)) for i in range(b)]
                cache.stats.clear()
            else:
                for n in (PREFIX, PREFIX + 1):
                    causal_last(net, [w[:n] for w in ws], "cuda")
            torch.cuda.synchronize()
            start = clock()
            tokens = 0
            pf = 0
            for i in range(LENGTH):
                t = PREFIX + i
                if cache:
                    out = (
                        cache.advance(ss, [t - 1] * b, [w[t - 1] for w in ws])
                        if i
                        else cache.prefill(ss, [w[:t] for w in ws])
                    )
                    tokens += b * (1 if i else t)
                else:
                    out = causal_last(net, [w[:t] for w in ws], "cuda")
                    tokens += b * t
                if i == 0:
                    torch.cuda.synchronize()
                    pf = clock() - start
            torch.cuda.synchronize()
            wall = clock() - start
            require(torch.isfinite(out[1]).all().item(), "bad host-only output")
            if cache:
                cache.clear()
            rows.append(
                dict(
                    b=b,
                    kind=name,
                    positions=b * LENGTH,
                    encoded_tokens=tokens,
                    prefill_ns=pf,
                    wall_ns=wall,
                    tensor_bytes=0 if cache is None else cache.tensor_bytes,
                )
            )
    return rows


def main(run):
    net, eager, fused, selected, windows, manifest = prepare(run)
    checks = check_cache(net)
    dump(run / "device-parity.json", checks)
    print("incremental host parity passed", flush=True)
    dump(run / "host-only.json", host_only(net, windows))
    total = 0
    for block in range(2):
        for b in (1, 16):
            ix = list(range(16 * block, 16 * block + b))
            arms = [(m, e) for m in ("direct", "fused-greedy", "fused-sample") for e in ("full", "cache")]
            for m, e in arms[::-1] if block else arms:
                total += measure(
                    run,
                    net,
                    eager if m == "direct" else fused,
                    m,
                    block,
                    [selected[i] for i in ix],
                    [windows[i] for i in ix],
                    ix,
                    manifest,
                    total,
                    label=f"{e}-{m}",
                    host_factory=cached if e == "cache" else None,
                )
    import os

    dump(
        run / "progress.json", dict(cell="all", phase="complete", total_emitted=total, server_pid=os.getpid())
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    main(p.parse_args().run)
