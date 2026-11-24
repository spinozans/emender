"""S19: an actual offered descriptor waits for its grant. No generated-service oracle in the readout."""

import argparse
from collections import Counter, defaultdict
import gc
import os
from pathlib import Path
from threading import get_native_id
from time import perf_counter_ns as clock

import torch
from ndm.memory_program import Limits, Session
from ndm.memory_program.byte_service import ByteService
from ndm.memory_program.entry_service import EntryReadout
from ndm.memory_program.spans import mean_byte_values
from .arrival_service_bench import children_cpu, target
from .branch_domain_bench import high_water
from .branch_image_bench import DescriptorMeter
from .demand_tape_bench import GATE_SEED, ROUTE_SEED, prepare_predictor
from .demand_tape_check import EntryTrace, compare
from .direct_arrival_bench import complete, host_pair, prepared_arena
from .entry_admissions import EntryAdmissions
from .memory_program_profile import measured
from .offered_waves import DescriptorOffers, serve_group
from .onclock_mount_verify import ORDER, prior_prefix, signature
from .pinned_branch import ROOT as NVME
from .record_retirement_bench import sample
from .scoped_records import ScopedRecords
from .serving_bench import process_usage
from .serving_verify import HERE, load, require
from .w3_data import EXE as M20_EXE
from .workbench_native import EXE, EXE_SHA
from .workbench_run import dump, sha
from .workbench_witness import digest_state


def refuse(handles):
    refused = []
    for r in handles:
        try:
            r.read_many((), 8)
        except ValueError as e:
            msg = "expired branch grant" if r.operation == "byte_edges" else "retired record reader"
            require(str(e) == msg, "different stale-reader failure")
            refused.append(r.operation)
        else:
            raise ValueError("previous grant remained usable")
    return refused


def wave(
    meter,
    offered,
    model,
    pool,
    rows,
    prompts,
    receipt,
    trace,
    old,
    progress,
    *,
    defer_position=False,
    native_log_type=None,
    program_only=False,
):
    start = clock()
    require(all(r["offered"] <= start for r in offered), "mount before offer")
    host0 = pool.stats.copy()
    graphs0 = {w: g.work.copy() for w, g in pool.backend.buckets.items()}
    dispatch0 = pool.backend.work.copy()
    stat = defaultdict(Counter)
    limits = Limits(pattern=16)

    def encode(xs):
        return mean_byte_values(xs, model.program.embed.weight)

    with meter.open(1) as (domain, desc):
        meter.pages.begin()
        with domain.borrow("P100", meter.man["mount"], scope="whole") as grant:
            log = native_log_type(trace) if native_log_type is not None else None
            with ScopedRecords(
                desc,
                receipt,
                scope="whole",
                branches=grant,
                encode=encode,
                defer_position=defer_position,
                on_native=log.raw if log else None,
            ) as records:
                if log is None:  # S19 independently hashed its already-open native process too
                    require(
                        sha(Path(f"/proc/{records.pid}/exe")) == EXE_SHA, "changed running positional binary"
                    )
                handles = grant.reader(), *records.readers()
                shared = tuple(measured(r, stat) for r in handles)

                def session(row, live):
                    return Session(
                        shared + tuple(measured(r, stat) for r in live.readers()),
                        limits,
                        advance=live.advance,
                        session_id=live.id,
                        document=row["doc"],
                        start=row["start"],
                    )

                clients = EntryAdmissions(
                    rows,
                    trace,
                    admitted_rows=[r["row"] for r in offered],
                    byte_limit=pool.context,
                    demand=True,
                    check=records.check,
                    session=session,
                    encode=encode,
                    program_only=program_only,
                )
                readout = (
                    None
                    if program_only
                    else EntryReadout(model, grant, gate_seed=GATE_SEED, on_direct=trace.direct)
                )
                model.program.calls.clear()
                with ByteService(
                    pool,
                    entry=readout,
                    policy=model.program if program_only else None,
                    mode="sample",
                    seed=ROUTE_SEED,
                    limits=limits,
                    features=lambda h: h[:, :256].contiguous(),
                ) as svc:
                    ready = clock()
                    rejected = refuse(old)
                    checked = clock()
                    ticks, stale = serve_group(
                        svc,
                        offered,
                        prompts,
                        clients,
                        progress=progress,
                        on_lookup=log.lookup if log and program_only else None,
                    )
                private = clients.finish()
                require(
                    pool.empty and not any(t.count_nonzero().item() for t in pool.slots.values()),
                    "unscrubbed host arena",
                )
                pos_peak = high_water(records.pid) if records.pid is not None else None
            require(
                records.closed and records.final["rc"] == (None if records.pid is None else 0),
                "scoped positional worker still live",
            )
        tap = meter.pages.finish()
        require(grant.final and domain.idle, "branch grant not returned")
    reaped = clock()
    require(domain.closed and meter.pages is None, "archive worker still live")
    return dict(
        start=start,
        ready=ready,
        stale_checked=checked,
        stale_readers=rejected,
        retired_at=reaped,
        requests=offered,
        ticks=ticks,
        stale_cancels=stale,
        work=dict(svc.work),
        routing=dict(readout.work) if readout else {},
        planner=dict(model.program.calls),
        provider=dict(stat),
        private=private,
        peak_private=clients.peak,
        position=dict(**records.final, peak_kib=pos_peak),
        **(dict(native_reads=log.finish(ticks)) if log else {}),
        cache=grant.final,
        tap=tap,
        native=meter.starts[-1],
        mount=meter.opens[-1],
        host=pool.stats - host0,
        buckets={w: g.work - graphs0[w] for w, g in pool.backend.buckets.items()},
        dispatch=pool.backend.work - dispatch0,
        slots_zero=True,
    ), handles


def trial(
    run,
    size,
    label,
    pool,
    model,
    rows,
    prompts,
    man,
    prior,
    *,
    run_wave=wave,
    schedule=None,
    trace_type=EntryTrace,
    check_prefix=None,
    summarize=signature,
):
    name = f"{size}-{label}"
    require(gc.isenabled(), "GC must remain enabled")
    trace = trace_type(rows, prompts)
    meter = DescriptorMeter(man)  # no I/O/verification or source until schedule/open below
    conf = load(HERE / "b1_mounts.json")["P100"]
    paths = {str(NVME / p["path"]) for p in conf["files"]} | {str(EXE), str(M20_EXE)}
    waves, old = [], ()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    cpu0, child0 = process_usage([])[0], children_cpu()
    initial = sample([], paths)
    require(initial["registered_fds"] == 0, "prior trial left source files open")

    def progress(n, phase="serve"):
        dump(
            run / "progress.json",
            dict(cell=name, phase=phase, total_emitted=prior + n, server_pid=os.getpid()),
        )

    progress(0, "intake")
    model.program.record_page = trace.page
    try:
        with DescriptorOffers() as queue:
            first = queue.take(0)
            schedule_start = clock()
            with meter.schedule(label if schedule is None else schedule) as groups:
                schedule_ready = clock()
                require(groups == [(i,) for i in range(4)], "changed grant schedule")
                for (j,) in groups:
                    offered = first if j == 0 else queue.take(j)
                    w, old = run_wave(
                        meter,
                        offered,
                        model,
                        pool,
                        rows,
                        prompts,
                        conf["files"][1],
                        trace,
                        old,
                        lambda n: progress(87 * j + n),
                    )
                    w["after_release"] = sample([], paths)
                    waves.append(w)
                    dump(run / f"{name}-w{j}.json", w)
                close_start = clock()
            schedule_closed = clock()
            final_refusals = refuse(old)
            torch.cuda.synchronize()
            end = clock()
    finally:
        del model.program.record_page
    final = sample([], paths)
    require(
        final["registered_fds"] == 0
        and meter.pages is meter.image is None
        and len(waves) == len(meter.opens) == len(meter.starts) == 4,
        "incomplete trial teardown",
    )
    full = [trace.prefix(i, target(i)) for i in range(32)]
    rep = dict(
        name=name,
        origin=queue.origin,
        owner=get_native_id(),
        sender=queue.sender,
        peak_waiting=queue.q.high,
        schedule_start=schedule_start,
        schedule_ready=schedule_ready,
        close_start=close_start,
        schedule_closed=schedule_closed,
        end=end,
        image=meter.image_summary,
        full=full,
        final_refusals=final_refusals,
        initial=initial,
        final=final,
        lifecycle_main_core_s=process_usage([])[0] - cpu0,
        lifecycle_child_core_s=children_cpu() - child0,
        peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),
    )
    dump(run / f"{name}.json", rep)
    # Independent historical prefix hashes are read only after all predictions of this trial.
    if check_prefix is None:
        require(full == prior_prefix(size), "S17 generated bytes, decisions, evidence or work differ")
    else:
        check_prefix(full)
    print(
        name,
        "348 generated bytes,",
        round((end - first[0]["offered"]) / 1e9, 3),
        "s from first offer",
        flush=True,
    )
    return trace, [summarize(w) for w in waves]


def panel(run, size, net, model, rows, prompts, man, total, wide_oracle=None):
    first = None
    with prepared_arena(run, size, net) as pool:
        for label in ORDER[size]:
            tr, sig = trial(run, size, label, pool, model, rows, prompts, man, total)
            total += 348
            if first is None:
                first, first_sig = tr, sig
                if wide_oracle is not None:
                    cross = compare(wide_oracle, first)  # retained is the numerical reference
                    dump(run / "cross-size.json", cross)
                    require(cross["passed"], "size-specific scores differ outside tolerance")
            else:
                audit = compare(tr, first)
                dump(run / f"{size}-{label}-parity.json", audit)
                require(
                    audit["passed"] and sig == first_sig,
                    "different generated scores, wire or chosen work",
                )
    return total, first


@torch.no_grad()
def main(run):
    beg, small, model, rows, prompts, man, initial = prepare_predictor(
        run, "on-clock whole-archive grants", ORDER
    )
    total = host_pair(run, beg, small, model, rows, prompts, man, exercise=panel)
    require(total == 2784 and digest_state(model) == initial, "changed weights or workload")
    complete(run, total, beg, model=initial)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    main(ap.parse_args().run)
