"""S1 closed-load generation. No held-out data, supplied program, or future target in a timing cell."""

from __future__ import annotations
import argparse
import asyncio
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
from threading import Lock
from time import perf_counter_ns as clock

import numpy as np
import torch
from ndm.memory_program import Limits, MaterializedBranches, PackedMemoryPolicy, Unavailable
from ndm.memory_program.packed import PackedRoundPlanner
from ndm.memory_program.serving import serve_ready, StreamUniforms, place_inputs
from .memory_program_profile import mount_batch, training_data
from .packed_program_profile import BindingNames, digest_rows
from .serving_check import check_device
from .w3_data import HERE
from .w3_model import ANCHOR_SHA, base_model
from .workbench_native import EXE_SHA
from .workbench_run import check_gpu_owner, dump, sha
from .workbench_witness import digest_state, require

PREFIX = 32
LENGTH = 64
CK = Path("output/memory-program-p2/attempt1/packed-b32.pt")
CK_SHA = "c323daffc42bba10edbd50adbcedcb47a3feca7902a687d71b5af3b0772e67be"
ROUTE_SEED = 7530


def cells(block, b, *, cohort=False):
    if cohort:
        arms = [f"{e}-{m}" for m in ("greedy", "sample") for e in ("fused", "ready", "cohort")]
        return arms[::-1] if block else arms
    arms = ["direct"] + [
        f"{engine}-{mode}" for mode in ("greedy", "sample") for engine in ("packed", "fused", "ready")
    ]
    if b == 16:
        arms.append("inline-sample")
    return arms[::-1] if block else arms


class LockedBranches(MaterializedBranches):
    """Protect this shared short page traversal, never hold this lock for a positional LF walk."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.guard = Lock()

    def histograms(self, keys):
        with self.guard:
            return super().histograms(keys)


def causal_last(net, streams, device, pad_byte=0):
    longest = max(map(len, streams))
    require(0 < longest <= 128 and all(streams), "outside C14 block")
    ids = torch.tensor(
        [list(w) + [pad_byte] * (longest - len(w)) for w in streams], dtype=torch.long, device=device
    )
    h = net.embed(ids) + net.position(torch.arange(longest, device=device))
    mask = torch.ones((longest, longest), dtype=torch.bool, device=device).triu(1)
    for layer in net.blocks:
        h = layer(h, src_mask=mask, is_causal=True)
    h = net.norm(h[torch.arange(len(ids), device=device), [len(w) - 1 for w in streams]])
    return h, net.head(h).float().log_softmax(1)


def direct(bank, lives, base, work):
    """Fixed longest-positive branch backoff; root permitted only on the archive, END never votes."""
    b = len(lives)
    arch = [None] * b
    tape = [None] * b
    total = np.zeros((b, 256), np.float32)
    weights = np.ones(b, np.float32)
    for k in (8, 6, 4, 2, 1, 0):
        pending = {
            j: bytes(l.data[-k:]) if k else b""
            for j, l in enumerate(lives)
            if arch[j] is None and k <= len(l.data)
        }
        keys = tuple(dict.fromkeys(pending.values()))
        if not keys:
            continue
        hs = dict(zip(keys, bank.histograms(keys), strict=True))
        work["archive_keys"] += len(keys)
        work["archive_batches"] += 1
        for j, key in pending.items():
            h = hs[key]
            if not isinstance(h, Unavailable) and sum(h[:256]):
                arch[j] = h[:256]
    for j, live in enumerate(lives):
        for k in (8, 4, 2, 1):
            if k > len(live.data):
                continue
            h = live.histogram(bytes(live.data[-k:]))
            work["live_keys"] += 1
            if sum(h[:256]):
                tape[j] = h[:256]
                break
        for found in (arch[j], tape[j]):
            if found is not None:
                v = np.asarray(found, np.float64)
                total[j] += 2 * v / v.sum()
                weights[j] += 2
    return (
        (base.exp() + torch.from_numpy(total).to(base.device))
        / torch.from_numpy(weights).to(base.device)[:, None]
    ).log()


def quantiles(ns):
    return {
        k: float(np.percentile(ns, q, method="higher") / 1e6)
        for k, q in (("p50_ms", 50), ("p95_ms", 95), ("p99_ms", 99), ("max_ms", 100))
    }


def process_usage(children):
    ticks = os.sysconf("SC_CLK_TCK")
    c = 0.0
    rss = []
    for pid in (os.getpid(), *children):
        if pid != os.getpid():
            stat = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
            c += (int(stat[11]) + int(stat[12])) / ticks
        status = Path(f"/proc/{pid}/status").read_text().splitlines()
        rss.append(int(next(s.split()[1] for s in status if s.startswith("VmRSS:"))))
    me = resource.getrusage(resource.RUSAGE_SELF)
    return me.ru_utime + me.ru_stime, c, rss


class CountOps:
    def operand_scores(self, q, prior, layout, width, second):
        self.p.calls["operand_tiles" if self.p.fused else "operand_eager_calls"] += 1
        self.p.calls["candidates"] += len(layout[0])
        return super().operand_scores(q, prior, layout, width, second)

    def choose_many(self, frames, *args):
        self.p.calls["choose_batches"] += 1
        self.p.calls["instructions"] += len(frames)
        return super().choose_many(frames, *args)


def models(state):
    from ndm.memory_program.triton_operands import TritonRoundPlanner, TritonMemoryPolicy

    class E(CountOps, PackedRoundPlanner):
        pass

    class T(CountOps, TritonRoundPlanner):
        pass

    out = []
    for cls, planner, flag in ((PackedMemoryPolicy, E, False), (TritonMemoryPolicy, T, True)):
        m = cls(256)
        m.load_state_dict(state)
        m.planner_type = planner
        m.calls = Counter()
        m.fused = flag
        m.eval().cuda().requires_grad_(False)
        out.append(m)
    return out


@torch.no_grad()
def measure(
    run,
    net,
    model,
    name,
    block,
    selected,
    windows,
    row_ids,
    manifest,
    prior_bytes,
    *,
    label=None,
    host_factory=None,
    length=LENGTH,
    prefix=PREFIX,
    oracle_lengths=None,
):
    begin_setup = clock()
    b = len(selected)
    cell = f"{block}-b{b}-{label or name}"
    is_direct = name == "direct"
    mode = name.split("-")[-1]
    provider = defaultdict(Counter)
    append = Counter()
    model.calls.clear()
    work = Counter()
    logs = [[] for _ in selected]
    stamped = [[] for _ in selected]
    native_intervals = []
    reader_intervals = []
    total = 0
    host_ns = 0
    host_batches = 0
    policy_dev = model.embed.weight.device
    transfer = Counter(copies=0, bytes=0, blocking_ns=0)

    def progress(phase):
        dump(
            run / "progress.json",
            dict(
                cell=cell,
                phase=phase,
                total_emitted=prior_bytes + total,
                cell_emitted=total,
                server_pid=os.getpid(),
            ),
        )

    progress("setup")
    with ExitStack() as stack:
        pages, bank, record, sessions, lives = mount_batch(
            stack,
            model,
            selected,
            manifest,
            provider,
            append,
            Limits(pattern=16),
            lambda *_: None,
            bank_type=LockedBranches,
        )
        pids = [pages.proc.pid, record.worker.proc.pid, *[l.native.proc.pid for l in lives]]
        request = record.worker.request

        def timed(keys):
            a = clock()
            out = request(keys)
            native_intervals.append((a, clock()))
            return out

        record.worker.request = timed
        for s, w in zip(sessions, windows, strict=True):
            s.observe(w[:prefix])
            require(s.visible == prefix, "prompt mismatch")
        host = host_factory(net, sessions, lives) if host_factory is not None else None
        if host is not None:
            stack.callback(host.clear)
        host_tokens = 0
        names = BindingNames(sessions)
        rng = StreamUniforms(sessions, row_ids, length, ROUTE_SEED, policy_dev) if mode == "sample" else None

        def inputs(ids):
            nonlocal host_ns, host_batches, host_tokens
            a = clock()
            if host is None:
                host_tokens += len(ids) * max(len(lives[i].data) for i in ids)
                out = causal_last(net, [lives[i].data for i in ids], "cuda")
            else:
                out = host.inputs(ids)
            host_ns += clock() - a
            host_batches += 1
            if not is_direct and out[0].device != policy_dev:
                t0 = clock()
                size = sum(t.numel() * t.element_size() for t in out)
                out = place_inputs(*out, width=model.host.in_features, device=policy_dev)
                transfer["blocking_ns"] += clock() - t0
                transfer["copies"] += 1
                transfer["bytes"] += size
            return out

        def acknowledge(ids, lp, ds=None):
            nonlocal total
            require(lp.shape == (len(ids), 256), "bad serving prediction")
            valid = lp.isfinite().all(1) & (lp.logsumexp(1).abs() < 5e-5)
            replies = torch.stack([lp.argmax(1), valid.long()], 1).tolist()  # actual D2H response fence
            before = total
            for j, (i, (symbol, ok)) in enumerate(zip(ids, replies, strict=True)):
                require(ok and 0 <= symbol < 256, "invalid byte distribution")
                s = sessions[i]
                if ds is not None:
                    d = ds[j]
                    require(
                        s.pending and d.session_id == s.id and d.visible == s.visible and d.mode == mode,
                        "stale/forced service decision",
                    )
                    logs[i].append((d.visible, d.requests, d.trace))
                s.observe(bytes([symbol]))
                stamped[i].append(clock())
                total += 1
            if total // 256 > before // 256:
                progress("generating")

        def deliver(ids, ds):
            acknowledge(ids, torch.stack([d.log_probs for d in ds]), ds)

        slow = sessions[0].readers[1]
        require(
            all(s.readers[1] is slow for s in sessions)
            and (slow.source, slow.version, slow.operation) == ("P100", bank.version, "occurrences"),
            "wrong async reader",
        )

        def on_io(reader, a, z, ids):
            if reader is slow:
                reader_intervals.append((a, z, ids))

        ids = list(range(b))
        runner = None if is_direct else model.batch_runner()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        cpu0, child0, _ = process_usage(pids)
        start = clock()
        if name.startswith(("ready-", "inline-", "cohort-")):
            stats = asyncio.run(
                serve_ready(
                    model,
                    sessions,
                    length,
                    inputs,
                    deliver,
                    mode=mode,
                    generator=rng,
                    offload=() if name.startswith("inline") else (slow,),
                    on_io=on_io,
                    cohort=name.startswith("cohort-"),
                )
            )
        else:
            stats = {}
            for t in range(length):
                h, base = inputs(ids)
                if is_direct:
                    acknowledge(ids, direct(bank, lives, base, work))
                else:
                    deliver(ids, runner(sessions, h, base, mode=mode, generator=rng))
        torch.cuda.synchronize()
        stop = clock()
        cpu1, child1, rss = process_usage(pids)
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        progress("cell-done")
        require(
            total == b * length and all(s.visible == prefix + length and not s.pending for s in sessions),
            "dropped stream",
        )
        latency = [[x - y for x, y in zip(ts, [start] + ts[:-1], strict=True)] for ts in stamped]
        first = [ts[0] - start for ts in stamped]
        action = []
        prefix_digests = []
        if oracle_lengths is not None and (
            len(oracle_lengths) != len(lives)
            or any(type(k) is not int or not 0 < k <= length for k in oracle_lengths)
        ):
            raise ValueError("invalid prefix audit lengths")
        ops = Counter()
        statuses = Counter()
        keys = 0
        for i, seq in enumerate(logs):
            ev = []
            for visible, nkeys, trace in seq:
                keys += nkeys
                parts = []
                for e in trace:
                    ops[e.inst.op] += 1
                    statuses[e.status] += 1
                    parts.append(
                        (
                            e.inst,
                            e.status,
                            e.register,
                            e.rows,
                            names.origin(e.reader),
                            tuple(map(names.key, e.keys)),
                        )
                    )
                ev.append((visible, nkeys, tuple(parts)))
            action.append(digest_rows(ev) if not is_direct else None)
            if oracle_lengths is not None:
                k = oracle_lengths[i]
                prefix_digests.append(
                    dict(
                        length=k,
                        event_sha=digest_rows(ev[:k]) if not is_direct else None,
                        byte_sha=hashlib.sha256(lives[i].data[prefix : prefix + k]).hexdigest(),
                    )
                )
        work.update(model.calls)
        work["logical_keys"] = keys
        overlap = sum(
            any(a <= x < z and i not in peers for a, z, peers in reader_intervals)
            for i, row in enumerate(stamped)
            for x in row
        )
        native_overlap = sum(any(a <= x < z for a, z in native_intervals) for row in stamped for x in row)
        result = dict(
            cell=cell,
            block=block,
            arm=label or name,
            b=b,
            train_rows=row_ids,
            emitted=total,
            wall_ns=stop - start,
            setup_prompt_ns=start - begin_setup,
            rate=total * 1e9 / (stop - start),
            interbyte_ns=latency,
            interbyte=quantiles(sum(latency, [])),
            first_ns=first,
            first=quantiles(first),
            host_cpu_seconds=cpu1 - cpu0,
            native_cpu_seconds=child1 - child0,
            native_processes=len(pids),
            host_rss_kib=rss[0],
            sum_child_rss_kib=sum(rss[1:]),
            peak_allocated=peak,
            peak_reserved=reserved,
            backbone_submit_ns=host_ns,
            backbone_batches=host_batches,
            policy_device=str(policy_dev),
            host_device=str(net.embed.weight.device),
            transfer=transfer,
            host=host.report() if host is not None else dict(encoded_tokens=host_tokens, tensor_bytes=0),
            generated_sha=[hashlib.sha256(l.data[prefix:]).hexdigest() for l in lives],
            event_sha=action,
            work=work,
            opcodes=ops,
            statuses=statuses,
            ready_stats=stats,
            provider=provider,
            live_append_including_prompt=append,
            archive=record.stats,
            live=sum([l.stats for l in lives], Counter()),
            m20_pages=bank.pages_fetched,
            m20_batches=bank.page_batches,
            acknowledged_during_occurrence=overlap,
            acknowledged_during_positional_worker=native_overlap,
            position_worker_ns=sum(z - a for a, z in native_intervals),
        )
    if oracle_lengths is not None:
        result["prefix_digests"] = prefix_digests
    result["lifetime_ns"] = clock() - begin_setup
    dump(run / (cell + ".json"), result)
    print(
        f"{cell}: {total} bytes {result['rate']:.1f}/s p99={result['interbyte']['p99_ms']:.1f}ms", flush=True
    )
    return total


def prepare(run):
    lease = check_gpu_owner()
    require(platform.node() == "lambda01" and os.environ.get("OMP_NUM_THREADS") == "4", "wrong host")
    torch.cuda.set_per_process_memory_fraction(0.55)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    require(sha(CK) == CK_SHA, "wrong program checkpoint")
    ck = torch.load(CK, map_location="cpu", weights_only=True)
    selected, windows, manifest, data = training_data(32)
    require(
        ck["step"] == 1
        and ck["cell"] == "packed-b32"
        and ck["base_sha"] == ANCHOR_SHA
        and ck["data"] == dict(**data, window129_sha=[hashlib.sha256(w).hexdigest() for w in windows]),
        "changed train population/checkpoint",
    )
    p2id = json.loads((HERE / "receipts/memory-program-p2/attempt1/packed-b32-identity.json").read_text())
    require(
        (manifest["mount"], manifest["exe_sha"], EXE_SHA)
        == tuple(p2id[k] for k in ("mount", "branch_exe_sha", "position_and_live_exe_sha")),
        "changed P2 mounts",
    )
    eager, fused = models(ck["model"])
    require(digest_state(eager) == digest_state(fused) == ck["policy_sha"], "different weights")
    net = base_model().eval().cuda()
    meta = dict(
        ref=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        owner=lease,
        server_pid=os.getpid(),
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        checkpoint=CK_SHA,
        policy_sha=ck["policy_sha"],
        base_sha=ANCHOR_SHA,
        data=data,
        prompt_bytes=PREFIX,
        prompt_sha=[hashlib.sha256(w[:PREFIX]).hexdigest() for w in windows],
        generated_length=LENGTH,
        route_seed=ROUTE_SEED,
        mount=manifest["mount"],
        branch_exe_sha=manifest["exe_sha"],
        position_exe_sha=EXE_SHA,
        limits=asdict(Limits(pattern=16)),
    )
    dump(run / "identity.json", meta)
    dump(
        run / "progress.json",
        dict(cell="qualification", phase="checking", total_emitted=0, server_pid=os.getpid()),
    )
    return net, eager, fused, selected, windows, manifest


def main(run, *, cohort=False):
    net, eager, fused, selected, windows, manifest = prepare(run)
    checks = check_device()
    if cohort:
        from ndm.memory_program.triton_operands import TritonMemoryPolicy
        from .serving_check import check_ready

        checks["cohort"] = check_ready("cuda", cls=TritonMemoryPolicy, cohort=True)
    with torch.no_grad():
        prompts = [w[:s] for w, s in zip(windows[:3], (1, 17, 32), strict=True)]
        _, p = causal_last(net, prompts, "cuda", 255)
        single = torch.cat(
            [net(torch.tensor(list(w), device="cuda")[None])[:, -1].log_softmax(1) for w in prompts]
        )
        torch.testing.assert_close(p, single, rtol=2e-5, atol=5e-5)
        checks["causal_host_max_logp_error"] = (p - single).abs().max().item()
    dump(run / "device-parity.json", checks)
    print("device conformance passed", flush=True)
    total = 0
    for block in range(2):
        for b in (1, 16):
            ix = list(range(16 * block, 16 * block + b))
            for name in cells(block, b, cohort=cohort):
                total += measure(
                    run,
                    net,
                    eager if name == "direct" or name.startswith("packed") else fused,
                    name,
                    block,
                    [selected[i] for i in ix],
                    [windows[i] for i in ix],
                    ix,
                    manifest,
                    total,
                )
    dump(
        run / "progress.json", dict(cell="all", phase="complete", total_emitted=total, server_pid=os.getpid())
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--cohort", action="store_true")
    args = ap.parse_args()
    main(args.run, cohort=args.cohort)
