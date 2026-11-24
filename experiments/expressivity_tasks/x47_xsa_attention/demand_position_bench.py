"""S20: two already-registered generated predictors, each with eager or first-use immutable locate."""

import argparse
from functools import partial
from pathlib import Path
from time import perf_counter_ns as clock

import torch
from .demand_tape_bench import prepare_predictor
from .demand_tape_check import EntryTrace, compare
from .demand_position_trace import NativeLog, ProgramTrace, compare_program
from .demand_position_verify import ORDER, check_prefix, same_reads
from .direct_arrival_bench import complete, prepared_arena
from .onclock_mount_bench import trial, wave
from .resident_host_check import memory
from .serving_verify import load, require
from .workbench_run import dump
from .workbench_witness import digest_state


@torch.no_grad()
def main(run):
    beg, host, model, rows, prompts, man, initial = prepare_predictor(
        run, "synchronous first-use archive position", ORDER
    )
    idn = load(run / "identity.json")
    idn.update(size_order=["retained"], router_order=list(ORDER))
    dump(run / "identity.json", idn)
    dump(
        run / "loaded.json",
        dict(
            prepare_host_predictor_ns=clock() - beg,
            parameters=sum(p.numel() for p in host.parameters()),
            peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(),
            memory=memory(),
        ),
    )
    total = 0
    with prepared_arena(run, "retained", host) as pool:
        for router, order in ORDER.items():
            program = router == "program"
            first = None
            for label in order:
                serve = partial(
                    wave,
                    defer_position=label.startswith("demand"),
                    native_log_type=NativeLog,
                    program_only=program,
                )
                trace, sig = trial(
                    run,
                    router,
                    label,
                    pool,
                    model,
                    rows,
                    prompts,
                    man,
                    total,
                    run_wave=serve,
                    schedule="fresh-" + label[-1],
                    trace_type=ProgramTrace if program else EntryTrace,
                    check_prefix=partial(check_prefix, router),
                    summarize=same_reads,
                )
                total += 348
                if first is None:
                    first, base_sig = trace, sig
                else:
                    proof = (compare_program if program else compare)(trace, first)
                    dump(run / f"{router}-{label}-parity.json", proof)
                    require(
                        proof["passed"] and sig == base_sig, "unmatched actual scores or request/result work"
                    )
    require(total == 2784 and digest_state(model) == initial, "changed model or incomplete service")
    complete(run, total, beg, model=initial)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    main(ap.parse_args().run)
