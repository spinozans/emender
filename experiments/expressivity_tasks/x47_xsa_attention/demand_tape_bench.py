"""S17: eager or demand-indexed private tapes under the same generated DIRECT/ENTER traffic."""

import argparse
from dataclasses import asdict
from functools import partial
from hashlib import sha256
import os
from pathlib import Path
import resource
from time import perf_counter_ns as clock

import torch
from ndm.memory_program.byte_service import ByteService
from ndm.memory_program.direct_entry import DirectEntryProgram
from ndm.memory_program.entry_service import EntryReadout
from ndm.memory_program.runtime import Limits
from .arrival_domain import ServingDomain
from .arrival_service_bench import children_cpu, target
from .branch_domain_bench import high_water
from .demand_tape_check import EntryTrace, compare
from .direct_arrival_bench import complete, host_pair, panel
from .direct_control_bench import bind
from .direct_service_head import load_head
from .large_host_bench import prepare_large
from .offer_clock import serve_offers
from .entry_admissions import EntryAdmissions
from .serving_bench import process_usage
from .serving_verify import HERE, load, require
from .terminal_branch_bench import WirePages
from .workbench_run import dump
from .workbench_witness import digest_state

ORDER = {
    "replicated": (
        ("spaced", "eager"),
        ("spaced", "demand"),
        ("burst", "demand"),
        ("burst", "eager"),
    ),
    "retained": (
        ("spaced", "demand"),
        ("spaced", "eager"),
        ("burst", "eager"),
        ("burst", "demand"),
    ),
}
GATE_SEED = 7823
ROUTE_SEED = 7530


def case(run, size, offer, provider, pool, model, rows, prompts, man, prior):
    require(
        provider in ("eager", "demand") and len(rows) == len(prompts) == 32,
        "unknown fixture",
    )
    name = f"{size}-{offer}-{provider}"
    begin = clock()
    cpu0 = process_usage([])[0]
    child0 = children_cpu()

    def progress(n, phase="serving"):
        dump(
            run / "progress.json",
            dict(cell=name, phase=phase, total_emitted=prior + n, server_pid=os.getpid()),
        )

    progress(0, "setup")
    limits = Limits(pattern=16)
    trace = EntryTrace(rows, prompts)
    host0 = pool.stats.copy()
    graphs0 = {w: g.work.copy() for w, g in pool.backend.buckets.items()}
    dispatch0 = pool.backend.work.copy()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with ServingDomain(man, model.program, limits, pages_factory=WirePages) as domain:
        direct = bind(domain.pages, man)  # same fresh verified process; separate, initially cold page LRUs
        require(not direct.cache and not domain.bank.cache, "nonempty domain at open")
        fn = EntryReadout(model, direct, gate_seed=GATE_SEED, on_direct=trace.direct)
        position = domain.record.worker.proc
        actors = [domain.pages.proc.pid, position.pid]

        admissions = EntryAdmissions(
            rows,
            trace,
            admitted_rows=range(32),
            byte_limit=pool.context,
            demand=provider == "demand",
            encode=domain.encode,
            session=domain._session,
            check=domain.record.verify,
            active=domain.active,
        )

        model.program.calls.clear()
        model.program.record_page = trace.page
        try:
            with ByteService(
                pool,
                entry=fn,
                mode="sample",
                seed=ROUTE_SEED,
                features=lambda h: h[:, :256].contiguous(),
                limits=limits,
            ) as svc:
                ms0, ns0, _ = process_usage(actors)
                reaped0 = children_cpu()
                origin, timing = serve_offers(
                    svc,
                    prompts,
                    admissions.binding,
                    offer,
                    admissions.reply,
                    progress,
                    trace=True,
                    synchronize=torch.cuda.synchronize,
                )
                ms1, ns1, rss = process_usage(actors)
                reaped1 = children_cpu()
                work = svc.work.copy()
        finally:
            del model.program.record_page
        closed = admissions.finish()
        peak_workers = admissions.peak
        require(
            pool.empty and not any(t.count_nonzero().item() for t in pool.slots.values()),
            "unscrubbed host",
        )
        pos_peak = high_water(position.pid)
        cache = {
            k: dict(pages=len(b.cache), rows=b.pages_fetched, batches=b.page_batches)
            for k, b in [("program", domain.bank), ("direct", direct)]
        }
        direct.cache.clear()
        domain.bank.cache.clear()
    life = clock() - begin
    sent = timing["emitted"]
    wire = domain.pages.receipt()
    require(
        domain.closed
        and domain.record.f.closed
        and position.returncode == 0
        and not direct.cache
        and not domain.bank.cache,
        "incomplete domain close",
    )
    full = [trace.prefix(i, len(bs)) for i, bs in enumerate(trace.raw)]
    rep = dict(
        name=name,
        size=size,
        offer=offer,
        provider=provider,
        **timing,
        full=full,
        arrival_prefix=[trace.prefix(i, target(i)) for i in range(32)],
        begin_clock=begin,
        lifecycle_ns=life,
        domain_setup_ns=origin - begin,
        slots_zero=True,
        cache_empty=True,
        cache=cache,
        work=work,
        host=pool.stats - host0,
        routing=fn.work,
        planner=model.program.calls,
        private=closed,
        peak_private_workers=peak_workers,
        position=dict(
            pid=position.pid,
            rc=position.returncode,
            peak_kib=pos_peak,
            open_ns=domain.record.worker.start_ns,
            work=domain.record.stats,
        ),
        readers=domain.provider,
        wire=wire,
        buckets={w: g.work - graphs0[w] for w, g in pool.backend.buckets.items()},
        dispatch=pool.backend.work - dispatch0,
        server_rss_kib=rss[0],
        server_hwm_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        measured_main_core_s=ms1 - ms0,
        measured_native_core_s=ns1 - ns0 + reaped1 - reaped0,
        lifecycle_main_core_s=process_usage([])[0] - cpu0,
        lifecycle_children_core_s=children_cpu() - child0,
        peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),
    )
    dump(run / (name + ".json"), rep)
    require(
        sent == fn.work["predictions"] == work["acknowledged"] == (384 if offer == "closed" else 348)
        and sum(x["enter"] for x in full) == fn.work["enter"]
        and fn.work["direct"] + fn.work["enter"] == sent
        and fn.work["transfer_batches"] == work["batches"]
        and fn.work["transfer_bytes"] == 2048 * sent,
        "wrong work/placement",
    )
    print(
        name,
        sent,
        "emitted;",
        peak_workers,
        "peak private workers;",
        round(sent * 1e9 / timing["wall_ns"], 1),
        "bytes/s",
        flush=True,
    )
    return sent, trace


@torch.no_grad()
def prepare_predictor(run, profile, order):
    """Rebuild only the frozen S17 inference stack; no archive domain is opened here."""
    beg = clock()
    small, policy, fused, rows, windows, man = prepare_large(run)
    prompts = [w[:32] for w in windows]
    del windows, fused
    head, meta = load_head(Path("output/memory-program-s14"))
    with torch.device("cpu"):
        model = DirectEntryProgram(policy.cpu(), head).eval().requires_grad_(False)
    idn = load(run / "identity.json")
    gate = digest_state(model.entry)
    require(
        meta == load(HERE / "receipts/memory-program-s14/head.json")
        and idn["policy_sha"] == digest_state(model.program)
        and gate == load(HERE / "receipts/memory-program-t2/attempt1/identity.json")["entry_initial"]
        and [sha256(w).hexdigest() for w in prompts] == idn["prompt_sha"]
        and idn["route_seed"] == ROUTE_SEED
        and idn["limits"] == asdict(Limits(pattern=16)),
        "unbound weights, gate, limits or prompts",
    )
    initial = digest_state(model)
    idn.update(
        profile=profile,
        direct_head=meta,
        gate=gate,
        gate_seed=GATE_SEED,
        model=initial,
        active_cap=16,
        waiting_cap=32,
        generated_length=12,
        step_widths=[1, 4, 8, 16],
        order=order,
        threads=torch.get_num_threads(),
        size_order=list(order),
    )
    dump(run / "identity.json", idn)
    return beg, small, model, rows, prompts, man, initial


@torch.no_grad()
def main(run):
    beg, small, model, rows, prompts, man, initial = prepare_predictor(
        run, "generated direct or enter / private tape materialization", ORDER
    )
    exercise = partial(panel, trial=case, orders=ORDER, reference="eager", parity=compare)
    total = host_pair(run, beg, small, model, rows, prompts, man, exercise=exercise)
    require(
        total == 3552 and digest_state(model) == initial,
        "changed model or incomplete service",
    )
    complete(run, total, beg, model=initial)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    main(p.parse_args().run)
