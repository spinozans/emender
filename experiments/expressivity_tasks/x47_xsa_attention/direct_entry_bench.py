"""T11 CPU-only: two clocks for the same sampled DIRECT/ENTER model on T4's retained packet."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, closing
from hashlib import sha256
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter_ns as clock, time, time_ns

import numpy as np
import torch
from ndm.memory_program import Limits, MaterializedBranches
from ndm.memory_program.direct_entry import DirectEntryProgram, direct_entry_teacher_force
from ndm.memory_program.m20 import TerminalBranches
from .arrival_service_bench import children_cpu
from .branch_domain_bench import high_water, prepare
from .direct_control_bench import add_hash, note, setup
from .frozen_packet_bench import no_card, tensor_sha
from .inband_route_check import delta
from .memory_program_profile import bind_program_readers
from .packed_program_profile import BindingNames, digest_rows, events
from .serving_bench import process_usage
from .serving_verify import HERE, load, require
from .terminal_branch_bench import WirePages
from .workbench_run import dump
from .workbench_witness import digest_state

GATES = 156
ORDER = ("index-A", "queue-A", "queue-B", "index-B")
FD = "t11-private"
LIMIT = 8 << 20
T2 = HERE / "receipts/memory-program-t2/attempt1"


def worker(root, name, fd):
    ref = setup(root, gate_count=GATES)
    run = root / name
    require(
        load(root / "driver.json")["pid"] == os.getppid()
        and not (run / "cell.json").exists()
        and os.readlink(f"/proc/self/fd/{fd}") == "/memfd:" + FD + " (deleted)"
        and os.fstat(fd).st_size == 0,
        "unbound cell or diagnostic channel",
    )
    life = clock()
    start = time_ns()
    total0 = process_usage([])[0] + children_cpu()
    (rows, ws, packet, head, policy), man, fixture = prepare(shifted_targets=True)
    model = DirectEntryProgram(policy.requires_grad_(True), head.requires_grad_(True)).train()
    init = {
        k: digest_state(m)
        for k, m in [("direct", model.direct), ("program", model.program), ("gate", model.entry)]
    }
    require(
        init
        == dict(
            direct=fixture["head"],
            program=fixture["policy"],
            gate=load(T2 / "identity.json")["entry_initial"],
        ),
        "wrong initial direct/program/gate",
    )
    identity = {k: v for k, v in fixture.items() if k not in ("seed", "observed32_sha")}
    identity.update(
        gate=init["gate"],
        gate_seed=7823,
        program_seed=7809,
        window129_sha=[sha256(w).hexdigest() for w in ws],
        gate_floor=8,
        key_price=0.001,
        instruction_price=0.0001,
        remaining_baseline=2.0,
    )
    stat = defaultdict(Counter)
    append = Counter()
    disp = []
    latest = 0

    def progress(k):
        nonlocal latest
        if k - latest >= 256 or k == 4096:
            dump(run / "progress.json", dict(pid=os.getpid(), phase="scored", targets=k))
            latest = k

    with ExitStack() as stack:
        pages = stack.enter_context(closing(WirePages()))
        require(
            (pages.exe_sha, pages.desc.version) == (man["exe_sha"], man["mount"]),
            "unverified shared immutable workspace",
        )
        args = ("P100", man["mount"], pages)
        bank = MaterializedBranches(*args, scope="whole", cache_pages=256)
        direct = TerminalBranches(*args, scope="whole", cache_pages=256)
        record, ss, lives = bind_program_readers(
            stack, policy, rows, pages.desc, bank, stat, append, Limits(pattern=16), lambda *_: None, disp
        )
        actors = [record.worker.proc, *[l.native.proc for l in lives]]
        pids = [pages.proc.pid, *(p.pid for p in actors)]
        src = packet.encoder(ss)
        calls = []

        def encode(ids):
            t = clock()
            h, b = src(ids)
            require(
                not calls and ids.shape == (32, 128) and not h.requires_grad and not b.requires_grad,
                "expected one frozen bytewise feature batch",
            )
            require(tensor_sha(torch.cat((h, b), -1)) == fixture["features"], "wrong host features")
            calls.append(clock() - t)
            return h, b

        names = BindingNames(ss)
        index = {s.id: i for i, s in enumerate(ss)}
        direct_hash = [sha256() for s in ss]
        reg_hash = [sha256() for s in ss]
        direct_counts = [0] * len(ss)
        reg_counts = [0] * len(ss)

        def witness(s, v, a):
            j = index[s.id]
            require(
                s.pending and v.visible == s.visible and v.tail == s.recent[-8:], "unbound direct witness"
            )
            note(direct_hash[j], v, a)
            direct_counts[j] += 1

        def page(s, step, reg, p):
            j = index[s.id]
            add_hash(reg_hash[j], (s.visible, step, reg, names.layout(p)))
            reg_counts[j] += 1

        policy.record_page = page
        cpu0, kid0, _ = process_usage(pids)
        begin = clock()
        v = direct_entry_teacher_force(
            model,
            encode,
            ss,
            ws,
            bank=direct,
            row_ids=list(range(32)),
            gate_seed=7823,
            program_seed=7809,
            clock=name.split("-")[0],
            key_price=0.001,
            instruction_price=0.0001,
            baseline_per_remaining=2.0,
            on_direct=witness,
            on_progress=progress,
        )
        fwd = clock()
        v.objective.backward()
        done = clock()
        cpu1, kid1, rss = process_usage(pids)
        del policy.record_page
        require(
            latest == v.targets == 4096
            and len(calls) == 1
            and all(
                bytes(l.data) == w
                and l.native.visible == s.visible == 129
                and s.pending
                and not any(s.cells)
                and not s.readers
                and s.advance is None
                for l, s, w in zip(lives, ss, ws, strict=True)
            ),
            "unfinished or crossed transcript",
        )
        act_hwm = [high_water(p.pid) for p in actors]
        for live in lives:
            live.data.clear()
            live.native.tail = b""
        docs = []
        es = []
        flat = [d for ds in v.decisions for d in ds]
        for i, ds in enumerate(v.decisions):
            entered = [t for t, d in enumerate(ds) if d.program is not None]
            es.append(entered)
            progs = [d.program for d in ds if d.program is not None]
            require(
                len(ds) == 128 and direct_counts[i] + len(entered) == 128,
                "wrong per-client decision coverage",
            )
            docs.append(
                dict(
                    row=i,
                    entries=len(entered),
                    direct=direct_counts[i],
                    registers=reg_counts[i],
                    entry_sha=digest_rows(entered),
                    direct_evidence=direct_hash[i].hexdigest(),
                    program_events=digest_rows(events([progs], names)),
                    register_bindings=reg_hash[i].hexdigest(),
                    price_sha=digest_rows([(d.program is not None, d.keys, d.nonemit) for d in ds]),
                    argmax=sha256(bytes(d.log_probs.detach().argmax().item() for d in ds)).hexdigest(),
                )
            )
        snap = {
            "byte_logp": torch.stack([d.log_probs.detach() for d in flat]),
            "gate_logp": torch.stack([d.gate_logp.detach() for d in flat]),
            "conditional_logp": torch.stack(
                [d.program.log_policy.detach() for d in flat if d.program is not None]
            ),
            "joint_logp": torch.stack([d.log_policy.detach() for d in flat]),
            "trust": torch.stack([d.trust.detach() for d in flat if d.program is None]),
            "losses": torch.cat(v.losses).detach(),
            "future": torch.cat(v.future).detach(),
            "mean_nll": v.mean_nll.detach(),
            "objective": v.objective.detach(),
        }
        require(
            snap["byte_logp"].logsumexp(1).abs().max().item() < 5e-5
            and (snap["trust"].sum(1) - 1).abs().max().item() < 5e-5,
            "unnormalized outputs",
        )
        schema = {}
        nonzero = Counter()
        for k, p in model.named_parameters():
            g = p.grad
            schema[k] = dict(shape=list(p.shape), elements=p.numel(), used=g is not None)
            if g is not None:
                require(g.shape == p.shape, "wrong adjoint shape")
                snap["grad:" + k] = g.detach()
                nonzero[k.partition(".")[0]] += g.count_nonzero().item()
        require(all(nonzero[k] > 0 for k in ("program", "entry", "direct")), "missing module adjoints")
        meta = {k: dict(shape=list(x.shape), sha=tensor_sha(x)) for k, x in snap.items()}
        require(
            all(x.dtype == torch.float32 and x.isfinite().all().item() for x in snap.values()),
            "nonfinite diagnostic",
        )
        sig = digest_rows(es)
        require(
            sig == load(T2 / "b32-index.json")["logical"]["entry_positions"],
            "changed initial gate decisions",
        )
        work = dict(v.work)
        logical = dict(
            docs=docs,
            entry_positions=sig,
            opcodes=Counter(e.inst.op for d in flat if d.program for e in d.program.trace),
            **{
                k: work[k]
                for k in (
                    "program",
                    "direct",
                    "direct_keys",
                    "program_keys",
                    "nonemit",
                    "append_bytes",
                    "append_calls",
                    "local_increments",
                )
            },
        )
        demand = dict(
            branch_rows=bank.pages_fetched,
            direct_rows=direct.pages_fetched,
            branch_batches=bank.page_batches,
            direct_batches=direct.page_batches,
            trust=v.direct_work,
            widths=v.widths,
            clock=v.work,
            provider=stat,
            archive=record.stats,
            live=sum((l.stats for l in lives), Counter()),
            append=append,
            dispatches=len(disp),
            dispatch_sha=digest_rows(
                [(names.origin(o), c, tuple(map(names.key, ks))) for o, c, ks in disp]
            ),
        )
    require(
        all(p.returncode == 0 for p in actors) and record.f.closed and pages.proc.returncode == 0,
        "native exit",
    )
    require(
        init
        == dict(
            direct=digest_state(model.direct),
            program=digest_state(model.program),
            gate=digest_state(model.entry),
        ),
        "weights changed in gradient-only cell",
    )
    raw = sum(x.numel() * x.element_size() for x in snap.values())
    require(0 < raw < LIMIT, "diagnostic capacity")
    with os.fdopen(os.dup(fd), "wb") as f:
        np.savez(f, **{k: x.numpy() for k, x in snap.items()})
    size = os.fstat(fd).st_size
    require(raw <= size <= LIMIT, "private transfer too large")
    no_card()
    dump(run / "progress.json", dict(pid=os.getpid(), phase="released", targets=v.targets))
    dump(
        run / "cell.json",
        dict(
            ref=ref,
            name=name,
            pid=os.getpid(),
            parent=os.getppid(),
            start_epoch_ns=start,
            end_epoch_ns=time_ns(),
            identity=identity,
            logical=logical,
            demand=demand,
            wire=pages.receipt(),
            snapshots=meta,
            parameters=schema,
            nonzero=nonzero,
            worker_exits=[[p.pid, p.returncode] for p in actors],
            worker_peak_kib=act_hwm,
            setup_ns=begin - life,
            feature_ns=calls[0],
            forward_ns=fwd - begin,
            backward_ns=done - fwd,
            wall_ns=done - begin,
            lifecycle_ns=clock() - life,
            core_s=cpu1 - cpu0 + kid1 - kid0,
            lifecycle_core_s=process_usage([])[0] + children_cpu() - total0,
            main_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            end_rss_kib=rss,
            torch=str(torch.__version__),
            strict=torch.are_deterministic_algorithms_enabled(),
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            threads=torch.get_num_threads(),
            visible=os.environ["CUDA_VISIBLE_DEVICES"],
            cuda_available=torch.cuda.is_available(),
            cuda_initialized=torch.cuda.is_initialized(),
            ipc_bytes=size,
            targets=v.targets,
            updates=0,
            nll=v.mean_nll.item(),
            objective=v.objective.item(),
        ),
    )


def contrast(ref, actual):
    require(ref.keys() == actual.keys(), "different snapshots/used gradients")
    out = {}
    for k, x in ref.items():
        y = actual[k]
        e = delta(x, y)
        diff = (x.double() - y.double()).abs()
        bound = (
            2e-4 + 2e-4 * x.double().abs().max()
            if k.startswith("grad:")
            else 5e-5 + 2e-5 * x.double().abs()
        )
        out[k] = dict(**e, violations=(diff > bound).sum().item(), max_fraction=(diff / bound).max().item())
    return out


def main(root):
    t = clock()
    ref = setup(root, gate_count=GATES)
    require(time() + 365 < 1795531800 and not (root / "driver.json").exists(), "too late or used run")
    dump(
        root / "driver.json",
        dict(ref=ref, pid=os.getpid(), order=ORDER, start_epoch_ns=time_ns(), card_s=0),
    )
    first = {}
    logical = None
    for name in ORDER:
        run = root / name
        run.mkdir()
        begin = clock()
        with (
            os.fdopen(os.memfd_create(FD, os.MFD_CLOEXEC), "w+b") as f,
            (run / "console").open("xb") as console,
        ):
            p = subprocess.run(
                [sys.executable, "-m", __spec__.name, str(root), "--cell", name, "--fd", str(f.fileno())],
                pass_fds=(f.fileno(),),
                stdin=subprocess.DEVNULL,
                stdout=console,
                stderr=subprocess.STDOUT,
                timeout=max(1, 355 - (clock() - t) / 1e9),
            )
            dump(run / "exit.json", dict(rc=p.returncode, process_wall_ns=clock() - begin))
            require(p.returncode == 0, "failed cell: " + name)
            s = load(run / "cell.json")
            require(0 < os.fstat(f.fileno()).st_size == s["ipc_bytes"] <= LIMIT, "private transfer")
            f.seek(0)
            with np.load(f, allow_pickle=False) as z:
                require(set(z.files) == set(s["snapshots"]), "missing diagnostic")
                xx = {k: torch.from_numpy(z[k]) for k in z.files}
        require(
            all(
                x.dtype == torch.float32
                and list(x.shape) == s["snapshots"][k]["shape"]
                and tensor_sha(x) == s["snapshots"][k]["sha"]
                for k, x in xx.items()
            ),
            "wrong diagnostic transfer",
        )
        if logical is None:
            logical = s["logical"]
        same = logical == s["logical"]
        family = name.split("-")[0]
        checks = {
            l: contrast(arrays, xx)
            for l, arrays in first.items()
            if l == "index-A" or l.split("-")[0] == family
        }
        dump(run / "comparison.json", dict(same_logical=same, checks=checks))
        require(
            same
            and all(
                e["violations"] == 0 and (l.split("-")[0] != family or e["unequal_bits"] == 0)
                for l, c in checks.items()
                for e in c.values()
            ),
            "different routes, scores or gradient: " + name,
        )
        if name.endswith("-A"):
            first[name] = xx
        print(
            name,
            round(s["wall_ns"] / 1e9, 3),
            "s,",
            sum(s["demand"]["widths"]["program"].values()),
            "program batches",
            flush=True,
        )
    no_card()
    dump(
        root / "complete.json",
        dict(
            pid=os.getpid(),
            end_epoch_ns=time_ns(),
            elapsed_ns=clock() - t,
            targets=16384,
            distinct=4096,
            updates=0,
            card_s=0,
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--cell", choices=ORDER)
    ap.add_argument("--fd", type=int)
    a = ap.parse_args()
    worker(a.root, a.cell, a.fd) if a.cell else main(a.root)
