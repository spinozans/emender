"""T2: real sparse entry + dynamic program gradients, two exact schedules, no optimizer."""

import argparse, gc, json, os, hashlib
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter_ns as clock
import torch
from ndm.memory_program import Limits
from ndm.memory_program.host_vjp import HostVJP
from ndm.memory_program.sparse_clock import SparseProgram, sparse_teacher_force
from .host_vjp_bench import prepare, compare
from .memory_program_profile import mount_batch, training_data
from .packed_program_profile import BindingNames, events, digest_rows
from .serving_bench import process_usage
from .workbench_run import dump
from .workbench_witness import require, digest_state

GATE = 7823
PROGRAM = 7809


def cell(run, enc, model, n, schedule, prior):
    name = f"b{n}-{schedule}"
    t0 = clock()
    selected, windows, man, data = training_data(n)
    append = Counter()
    provider = defaultdict(Counter)
    limits = Limits(pattern=16)
    dispatch = []

    def progress(phase, k):
        dump(
            run / "progress.json",
            dict(cell=name, phase=phase, total_targets=prior + k, trainer_pid=os.getpid()),
        )

    progress("setup", 0)
    gc.collect()
    torch.cuda.empty_cache()
    model.zero_grad(set_to_none=True)
    enc.zero_grad(set_to_none=True)
    tx = HostVJP(enc, width=256, microbatch=4, remat=False)
    enc.saved.clear()
    with ExitStack() as stack:
        pages, _, record, sessions, lives = mount_batch(
            stack, model.program, selected, man, provider, append, limits, lambda *_: None, dispatch
        )
        pids = [pages.proc.pid, record.worker.proc.pid, *[l.native.proc.pid for l in lives]]
        cpu0, kid0, _ = process_usage(pids)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        resident = torch.cuda.memory_allocated()
        ts = clock()
        tm = dict(setup_and_purge_ns=ts - t0)

        def encode(ids):
            t = clock()
            h, b = tx.stage(ids)
            torch.cuda.synchronize()
            tm["host_forward_ns"] = clock() - t
            return h, b

        observed = 0

        def update(k):
            nonlocal observed
            if k - observed >= 128 or k == n * 128:
                progress("scored-and-observed", k)
                observed = k

        batch = sparse_teacher_force(
            model,
            encode,
            sessions,
            windows,
            row_ids=data["train_rows"],
            gate_seed=GATE,
            program_seed=PROGRAM,
            clock=schedule,
            key_price=0.001,
            instruction_price=0.0001,
            baseline_per_remaining=2.0,
            on_progress=update,
        )
        torch.cuda.synchronize()
        fwd = clock()
        post = torch.cuda.memory_allocated()
        require(
            batch.targets == n * 128 == observed and all(bytes(l.data) == w for w, l in zip(windows, lives)),
            "incomplete raw observations",
        )
        tx.backward(batch.objective)
        torch.cuda.synchronize()
        end = clock()
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        cpu1, kid1, rss = process_usage(pids)
        tm.update(forward_ns=fwd - ts, backward_ns=end - fwd, gradient_wall_ns=end - ts)
        progress("gradient-complete", batch.targets)
        params = {}
        stats = {}
        for label, module in [("host", enc), ("gate", model.entry), ("program", model.program)]:
            st = Counter()
            for k, p in module.named_parameters():
                g = p.grad.detach().to("cpu", copy=True) if p.grad is not None else None
                params[label + "/" + k] = g
                if g is not None:
                    require(torch.isfinite(g).all().item() and g.shape == p.shape, "bad " + label + "/" + k)
                    nz = g.count_nonzero().item()
                    st.update(
                        used_elements=g.numel(), tensors=1, nonzero_tensors=int(nz > 0), nonzero_elements=nz
                    )
            require(st["nonzero_elements"] > 0, "empty " + label + " gradient")
            stats[label] = st
        tm["snapshot_ns"] = clock() - end
        ds = batch.decisions
        flat = sum(ds, [])
        names = BindingNames(sessions)
        logical = dict(
            events=digest_rows(events(ds, names)),
            entry_positions=digest_rows(batch.entries),
            entries_per_doc=list(map(len, batch.entries)),
            window_sha=[hashlib.sha256(w).hexdigest() for w in windows],
            train_rows=data["train_rows"],
        )
        ops = Counter(e.inst.op for d in flat for e in d.trace)
        keys = sum(d.requests for d in flat)
        numeric = {
            k: torch.cat([t.detach() for t in getattr(batch, k)])
            for k in ("gate_logp", "joint_logp", "byte_logp")
        }
        numeric.update(
            conditional_logp=torch.stack([d.log_policy.detach() for d in flat]),
            nll=batch.nll.detach(),
            objective=batch.objective.detach(),
        )
        report = dict(
            cell=name,
            targets=batch.targets,
            mode="fresh_sample",
            optimizer_steps=0,
            logical=logical,
            work=batch.work,
            host=tx.work,
            ops=ops,
            logical_keys=keys,
            provider=provider,
            physical_dispatches=len(dispatch),
            nll=batch.nll.item(),
            objective=batch.objective.item(),
            gradients=stats,
            times=tm,
            targets_s=batch.targets * 1e9 / tm["gradient_wall_ns"],
            initial=resident,
            post_rollout=post,
            peak_allocated=peak,
            peak_reserved=reserved,
            rss_kib=rss[0],
            children_rss_kib=sum(rss[1:]),
            cpu_core_s=cpu1 - cpu0 + kid1 - kid0,
        )
        dump(run / (name + ".json"), report)
        print(
            f"{name}: {len(flat)}/{batch.targets} entered, {report['targets_s']:.1f} gradient targets/s",
            flush=True,
        )
    model.zero_grad(set_to_none=True)
    enc.zero_grad(set_to_none=True)
    return dict(logical=logical, numeric=numeric, grads=params)


def main(run):
    enc, p = prepare(run)
    model = SparseProgram(p)
    idn = json.loads((run / "identity.json").read_text())
    idn.update(
        gate_seed=GATE,
        program_seed=PROGRAM,
        entry_initial=digest_state(model.entry),
        entry_parameters=sum(t.numel() for t in model.entry.parameters()),
        entry_initial_bias=model.entry.linear.bias.item(),
    )
    dump(run / "identity.json", idn)
    total = 0
    for n, order in [(4, ("index", "compact")), (32, ("compact", "index"))]:
        pair = {}
        for schedule in order:
            pair[schedule] = cell(run, enc, model, n, schedule, total)
            total += n * 128
        compare(run, n, pair["index"], pair["compact"])
        del pair
        gc.collect()
    dump(
        run / "progress.json",
        dict(cell="all", phase="complete", total_targets=total, trainer_pid=os.getpid()),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    main(ap.parse_args().run)
