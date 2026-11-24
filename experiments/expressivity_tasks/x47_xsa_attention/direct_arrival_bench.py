"""S14: actual generated-byte service with a fixed T9 head. No trainer or policy replay on card."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing, contextmanager
from dataclasses import asdict
from hashlib import sha256
import os
from pathlib import Path
import resource
from time import perf_counter_ns as clock

import torch
from ndm.memory_program.byte_service import ByteService
from ndm.memory_program.direct_service import DirectReadout, DirectStream
from ndm.memory_program.m20 import MaterializedBranches, TerminalBranches
from ndm.memory_program.resident_host import ResidentArena
from ndm.memory_program.runtime import Limits
from .arrival_service_bench import children_cpu, target
from .offer_clock import serve_offers
from .direct_arrival_check import Transcript, compare
from .direct_control_bench import bind
from .direct_service_head import load_head
from .large_host_bench import prepare_large
from .resident_host_check import memory
from .serving_bench import process_usage
from .serving_verify import load, require
from .terminal_branch_bench import WirePages
from .wide_host import build
from .workbench_run import dump
from .workbench_witness import digest_state


ORDER = {
    "replicated": (
        ("spaced", "ref"),
        ("spaced", "terminal"),
        ("burst", "terminal"),
        ("burst", "ref"),
    ),
    "retained": (
        ("spaced", "terminal"),
        ("spaced", "ref"),
        ("burst", "ref"),
        ("burst", "terminal"),
    ),
}


def case(run, size, offer, provider, pool, head, rows, prompts, man, prior):
    name = f"{size}-{offer}-{provider}"
    closed = offer == "closed"
    expected = 384 if closed else 348
    begin = clock()
    proc0 = process_usage([])[0]
    child0 = children_cpu()
    sent = 0

    def progress(phase):
        dump(
            run / "progress.json",
            dict(
                cell=name,
                phase=phase,
                total_emitted=prior + sent,
                server_pid=os.getpid(),
            ),
        )

    progress("setup")
    transcript = Transcript(rows)
    limits = Limits()
    host0 = pool.stats.copy()
    graphs0 = {w: g.work.copy() for w, g in pool.backend.buckets.items()}
    dispatch0 = pool.backend.work.copy()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    local = Counter()

    @contextmanager
    def binding(i):
        r = rows[i]
        a = DirectStream(r["doc"], r["start"], limits=limits, byte_limit=pool.context)
        local["opened"] += 1
        try:
            yield a.session, a
        finally:
            local.update(
                observed_bytes=a.counts.visible,
                distinct_rows_sum=len(a.counts.rows),
                edges_sum=a.counts.edges,
                increments=a.counts.increments,
            )
            a.close()
            require(
                a.counts.closed
                and not a.counts.rows
                and not a.counts.tail
                and not a.session.recent
                and not a.session.readers
                and a.session.advance is None
                and not any(a.session.cells),
                "retained direct client payload",
            )
            local["closed"] += 1

    with closing(WirePages()) as p:
        bank = bind(p, man, TerminalBranches if provider == "terminal" else MaterializedBranches)
        readout = DirectReadout(head, bank, observe=transcript.observe)
        with ByteService(
            pool,
            feature_direct=readout,
            features=lambda h: h[:, :256].contiguous(),
            limits=limits,
        ) as service:
            ids = [p.proc.pid]
            cpu0, nat0, _ = process_usage(ids)

            def delivered(i, r):
                require(r.decision is None, "unexpected program trace on direct")
                transcript.commit(i, r.byte)

            def advancing(n):
                nonlocal sent
                sent = n
                progress("serving")

            origin, timing = serve_offers(
                service,
                prompts,
                binding,
                offer,
                delivered,
                advancing,
                trace=False,
                synchronize=torch.cuda.synchronize,
            )
            work = service.work.copy()
            cpu1, nat1, rss = process_usage(ids)
        require(
            pool.empty and all(t.count_nonzero().item() == 0 for t in pool.slots.values()),
            "uncleared host",
        )
        cache_size = len(bank.cache)
        bank.cache.clear()
        require(not bank.cache, "dirty native domain")
    life = clock() - begin
    proc1 = process_usage([])[0]
    child1 = children_cpu()
    h = pool.stats - host0
    full = [transcript.prefix(i, len(bs)) for i, bs in enumerate(transcript.raw)]
    at_target = [transcript.prefix(i, target(i)) for i in range(32)]
    rep = dict(
        name=name,
        offer=offer,
        provider=provider,
        size=size,
        **timing,
        full=full,
        arrival_prefix=at_target,
        lifecycle_ns=life,
        domain_setup_ns=origin - begin,
        begin_clock=begin,
        work=work,
        host=h,
        direct_work=readout.work,
        local=local,
        cache_pages_before_close=cache_size,
        cache_empty=not bank.cache,
        slots_zero=True,
        wire=p.receipt(),
        buckets={w: g.work - graphs0[w] for w, g in pool.backend.buckets.items()},
        dispatch=pool.backend.work - dispatch0,
        peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),
        server_rss_kib=rss[0],
        server_hwm_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        measured_main_core_s=cpu1 - cpu0,
        measured_native_core_s=nat1 - nat0,
        lifecycle_main_core_s=proc1 - proc0,
        lifecycle_children_core_s=child1 - child0,
    )
    dump(run / (name + ".json"), rep)
    require(
        sent == expected
        and local["observed_bytes"] == 1024 + sent
        and local["opened"] == local["closed"] == 32
        and work["acknowledged"] == readout.work["predictions"] == sent
        and h["prefill_tokens"] == 1024
        and h["step_tokens"] == sent - 32
        and work["batches"] == readout.work["transfer_batches"]
        and readout.work["transfer_bytes"] == sent * 512 * 4,
        "wrong observation/placement workload",
    )
    print(
        name,
        sent,
        "bytes,",
        round(sent * 1e9 / rep["wall_ns"], 1),
        "bytes/s",
        flush=True,
    )
    return sent, transcript


@contextmanager
def prepared_arena(run, size, net):
    """Charge this size's first construction and final scrub separately from its service domains."""
    before = memory()
    t = clock()
    torch.cuda.reset_peak_memory_stats()
    with ResidentArena(
        net,
        16,
        64,
        (1, 4, 8, 16),
        byte_budget=(1 << 30 if size == "replicated" else 256 << 20),
        private_budget=(256 if size == "replicated" else 64) << 20,
    ) as pool:
        pool.set_cover(True)
        after = memory()
        dump(
            run / (size + "-setup.json"),
            dict(
                prepare_ns=clock() - t,
                before=before,
                after=after,
                tensor_bytes=pool.tensor_bytes,
                private_bytes=pool.backend.pool_bytes,
                buckets={w: g.setup for w, g in pool.backend.buckets.items()},
                peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved(),
                parameters=sum(p.numel() for p in net.parameters()),
            ),
        )
        yield pool
    require(
        not pool.slots and not pool.backend.buckets and pool.empty,
        "resident arena not closed",
    )
    dump(
        run / (size + "-closed.json"),
        dict(
            slots_freed=True,
            graphs_freed=True,
            after_close=memory(),
            lifecycle_ns=clock() - t,
            peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(),
        ),
    )


def panel(
    run,
    size,
    net,
    head,
    rows,
    prompts,
    man,
    total,
    wide_oracle=None,
    *,
    trial=case,
    orders=ORDER,
    reference="ref",
    parity=compare,
):
    with prepared_arena(run, size, net) as pool:
        n, oracle = trial(run, size, "closed", reference, pool, head, rows, prompts, man, total)
        total += n
        if wide_oracle is not None:
            cross = parity(wide_oracle, oracle)  # retained geometry is the numerical reference
            dump(run / "cross-size.json", cross)
            require(
                cross["passed"],
                "cross-size closed generated byte/evidence/score mismatch",
            )
        for offer, provider in orders[size]:
            n, actual = trial(run, size, offer, provider, pool, head, rows, prompts, man, total)
            total += n
            audit = parity(actual, oracle)
            dump(run / f"{size}-{offer}-{provider}-parity.json", audit)
            require(audit["passed"], "closed vs. arrival semantic mismatch")
    return total, oracle


@torch.no_grad()
def main(run):
    beg = clock()
    small, eager, fused, rows, ws, man = prepare_large(run)
    del eager, fused
    prompts = [w[:32] for w in ws]
    del ws
    head, meta = load_head(run.parent)
    idn = load(run / "identity.json")
    require(
        [sha256(p).hexdigest() for p in prompts] == idn["prompt_sha"] and meta["ref"] == idn["ref"],
        "unbound prompts/head",
    )
    idn["setup_only"] = {
        k: idn.pop(k)
        for k in (
            "checkpoint",
            "policy_sha",
            "controller_training_base_sha",
            "limits",
            "route_seed",
            "position_exe_sha",
        )
    }
    idn.update(
        profile="trained direct byte arrivals",
        direct_head=meta,
        limits=asdict(Limits()),
        active_cap=16,
        waiting_cap=32,
        generated_length=12,
        step_widths=[1, 4, 8, 16],
        order=ORDER,
        threads=torch.get_num_threads(),
        size_order=list(ORDER),
        note="P2 and C14 were validated by host setup, not predictors in S14",
    )
    dump(run / "identity.json", idn)
    total = host_pair(run, beg, small, head, rows, prompts, man)
    require(
        total == 3552 and digest_state(head) == meta["after"],
        "wrong work or updated head",
    )
    complete(run, total, beg, head=digest_state(head))


def host_pair(run, beg, small, head, rows, prompts, man, *, exercise=panel):
    """Charge construction and transfer; neither size receives a preconstructed arena."""
    load_mem = memory()
    load_peak = torch.cuda.max_memory_allocated()
    load_reserved = torch.cuda.max_memory_reserved()
    t = clock()
    wide, rule = build(small)
    torch.cuda.synchronize()
    build_ns = clock() - t
    t = clock()
    small.cpu()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    xfer = clock() - t
    dump(
        run / "construction.json",
        dict(
            **rule,
            parameter_bytes=sum(p.numel() * p.element_size() for p in wide.parameters()),
            load_peak_allocated=load_peak,
            load_peak_reserved=load_reserved,
            loaded=load_mem,
            prepare_head_host_ns=t - beg - build_ns,
            build_verify_ns=build_ns,
            small_to_cpu_ns=xfer,
            peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(),
        ),
    )
    total, wide_oracle = exercise(run, "replicated", wide, head, rows, prompts, man, 0)
    t = clock()
    del wide
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = memory()
    small.cuda()
    torch.cuda.synchronize()
    dump(
        run / "transfer.json",
        dict(
            wide_drop_and_small_to_gpu_ns=clock() - t,
            before_small=before,
            peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(),
        ),
    )
    total, _ = exercise(run, "retained", small, head, rows, prompts, man, total, wide_oracle=wide_oracle)
    return total


def complete(run, total, beg, **weights):
    dump(
        run / "complete.json",
        dict(
            total_emitted=total,
            server_pid=os.getpid(),
            lifecycle_ns=clock() - beg,
            steps=0,
            **weights,
            memory=memory(),
        ),
    )
    dump(
        run / "progress.json",
        dict(cell="all", phase="complete", total_emitted=total, server_pid=os.getpid()),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    main(ap.parse_args().run)
