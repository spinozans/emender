"""S6 native mixed-client admission/reuse stress; no native index or held-out target consumed."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter_ns as clock
import torch
from ndm.memory_program.resident_host import ResidentArena
from .large_host_bench import prepare_large
from .large_host_check import make
from .serving_bench import process_usage
from .workbench_run import dump


def memory():
    torch.cuda.synchronize()
    return dict(
        allocated=torch.cuda.memory_allocated(),
        reserved=torch.cuda.memory_reserved(),
        rss_kib=process_usage([])[2][0],
    )


def wave(a, w, windows):
    hs = []
    words = []
    rows = []
    generation = [0] * 16
    events = []
    for i in range(16):
        row = (w % 2) * 16 + i
        rows.append(row)
        words.append(bytearray(windows[row][:32]))
        hs.append(a.admit(f"{w}:{i}"))

    def respond(ids, prompt=False):
        handles = [hs[i] for i in ids]
        out = (
            a.prefill(handles, [bytes(words[i]) for i in ids])
            if prompt
            else a.advance(handles, [len(words[i]) - 1 for i in ids], [words[i][-1] for i in ids])
        )
        h, p = out
        valid = p.isfinite().all(1) & (p.logsumexp(1).abs() < 5e-5) & h.isfinite().all(1)
        for i, (b, ok) in zip(ids, torch.stack([p.argmax(1), valid.long()], 1).tolist(), strict=True):
            if not ok:
                raise ValueError("invalid byte distribution")
            events.append((i, rows[i], generation[i], len(words[i]), b))
            words[i].append(b)

    ids = list(range(16))
    order = ids[w % 16 :] + ids[: w % 16]
    i = order[0]
    respond(ids, True)
    respond(ids)
    respond([i])
    respond(order[1:4])
    prior = hs[i]
    peer = hs[order[1]].slot
    kept = {k: v[peer].clone() for k, v in a.slots.items()}
    a.release([prior])
    assert all(
        v[prior.slot].count_nonzero().item() == 0 and torch.equal(v[peer], kept[k])
        for k, v in a.slots.items()
    )
    del kept
    try:
        a.read([prior], [len(words[i]) - 1])
        raise AssertionError("stale host re-admission")
    except ValueError:
        pass
    rows[i] = (rows[i] + 16) % 32
    generation[i] = 1
    hs[i] = a.admit(f"{w}:{i}:new")
    assert hs[i].slot == prior.slot and hs[i].generation != prior.generation
    words[i] = bytearray(windows[rows[i]][:32])
    respond([i], True)
    respond(order)
    a.release(hs)
    assert a.empty and all(t.count_nonzero().item() == 0 for t in a.slots.values())
    return dict(
        wave=w,
        signature=w % 16,
        sha=hashlib.sha256(json.dumps(events, separators=(",", ":")).encode()).hexdigest(),
        emitted=len(events),
        admissions=17,
        erased=True,
    )


@torch.no_grad()
def main(run):
    net, e, f, selected, windows, manifest = prepare_large(run)
    del e, f
    ident = json.loads((run / "identity.json").read_text())
    ident.pop("generated_length")
    ident.update(
        profile="resident-churn",
        oracle_waves=16,
        resident_waves=40,
        bytes_per_wave=53,
        memory_readers_opened=0,
    )
    dump(run / "identity.json", ident)
    expected = []
    total = 0

    def progress(cell, phase):
        dump(run / "progress.json", dict(cell=cell, phase=phase, total_emitted=total, server_pid=os.getpid()))

    for w in range(16):
        a = make(net, 16)
        r = wave(a, w, windows)
        expected.append(r)
        del a
        total += r["emitted"]
        progress("eager-reference", "generating")
    dump(run / "oracle.json", expected)
    baseline = memory()
    t0 = clock()
    a = ResidentArena(net, 16, 64, (1, 16), byte_budget=256 << 20, private_budget=64 << 20)
    torch.cuda.synchronize()
    build_ns = clock() - t0
    setup = dict(
        baseline=baseline,
        after_prepare=memory(),
        build_ns=build_ns,
        preparation_streams=1,
        tensor_bytes=a.tensor_bytes,
        pool_bytes=a.backend.pool_bytes,
        buckets={k: v.setup for k, v in a.backend.buckets.items()},
    )
    dump(run / "setup.json", setup)
    rows = []
    for w in range(40):
        t0 = clock()
        r = wave(a, w, windows)
        assert r["emitted"] == 53 and r["sha"] == expected[w % 16]["sha"]
        r.update(post_release=memory(), wave_ns=clock() - t0)
        rows.append(r)
        total += r["emitted"]
        progress("resident-churn", "generating")
        if w % 8 == 7:
            print("resident waves", w + 1, "post-release bytes", r["post_release"]["allocated"], flush=True)
    counts = dict(
        arena=a.stats, dispatch=a.backend.work, buckets={k: g.work for k, g in a.backend.buckets.items()}
    )
    a.close()
    try:
        a.admit("late")
        raise AssertionError("closed pool accepted admission")
    except ValueError:
        pass
    del a
    torch.cuda.synchronize()
    after = memory()
    torch.cuda.empty_cache()
    purged = memory()
    result = dict(waves=rows, counts=counts, after_close=after, after_purge=purged)
    dump(run / "churn.json", result)
    progress("all", "complete")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    main(p.parse_args().run)
