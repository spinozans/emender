"""P2: one native reference trajectory, fixed-program parity, then fresh measured packed updates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter_ns as clock

import torch

from ndm.memory_program import Limits, PackedMemoryPolicy, Span, teacher_force
from ndm.memory_program.machine import _objective
from ndm.memory_program.packed import PackedRoundPlanner
from .memory_program_profile import TimedPolicy, activity, mount_batch, training_data
from .packed_program_check import Errors, exercise
from .w3_data import HERE
from .w3_model import ANCHOR_SHA, backbone, base_model
from .workbench_native import EXE_SHA
from .workbench_run import check_gpu_owner, dump, sha
from .workbench_witness import digest_state, require

CELLS = ("reference-b4", "replay-b4", "packed-b4", "packed-b32")


class TimedPlanner(PackedRoundPlanner):
    def encode(self, pages):
        self.p.calls["encoded_pages"] += len(pages)
        self.p.calls["encoded_rows"] += sum(len(pg.keys) for pg in pages)
        return self.p.timed("encode_pages", super().encode, pages)

    def rebuild(self, *args):
        return self.p.timed("rebuild", super().rebuild, *args)

    def choose_many(self, *args):
        return self.p.timed("choose_batch", super().choose_many, *args)

    def operand(self, f, role, choices, args):
        self.p.calls["operand_candidates"] += len(choices)
        return self.p.timed("operand_layout", super().operand, f, role, choices, args)

    def emit_many(self, *args):
        return self.p.timed("emit", super().emit_many, *args)


class PackedTimed(PackedMemoryPolicy):
    planner_type = TimedPlanner
    timed = TimedPolicy.timed

    def __init__(self):
        super().__init__(256)
        self.ns = Counter()
        self.calls = Counter()


class BindingNames:
    """Fresh native sessions get fresh versions. Rename ONLY actual live nonces bijectively."""

    def __init__(self, sessions):
        self.live = {s.id: (s.document, s.start) for s in sessions}

    def origin(self, r):
        if r is None:
            return None
        source, ver, op, wm = r
        if source in ("tape", "computed") and ver in self.live:
            ver = ("live", self.live[ver])
        return (source, ver, op, wm)

    def key(self, k):
        if not isinstance(k, Span):
            return k
        version = (
            ("live", self.live[k.version]) if k.source == "tape" and k.version in self.live else k.version
        )
        return ("span", k.source, version, k.document, k.start, k.stop)

    def layout(self, p):
        return (
            tuple((f.name, f.kind, self.origin(f.origin)) for f in p.fields),
            tuple(tuple(self.key(k) for k in row) for row in p.keys),
        )


def events(decisions, names):
    return [
        (
            names.live[d.session_id],
            d.visible,
            d.requests,
            tuple(
                (e.inst, e.status, e.rows, e.register, names.origin(e.reader), tuple(map(names.key, e.keys)))
                for e in d.trace
            ),
        )
        for ds in decisions
        for d in ds
    ]


def digest_rows(rows):
    h = hashlib.sha256()
    for row in rows:
        data = repr(row).encode()
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
    return h.hexdigest()


@dataclass
class Witness:
    plans: list
    logical: dict
    numeric: dict
    grads: dict


def witness(batch, sessions, dispatch, pages, grads, obj):
    names = BindingNames(sessions)
    ds = batch.decisions
    flat = sum(ds, [])
    regs = [(names.live[sid], t, step, reg, p) for sid, t, step, reg, p in pages]
    end = [
        (names.live[s.id], s.visible, -1, i, p)
        for s in sessions
        for i, p in enumerate(s.cells)
        if p is not None
    ]
    # All intermediates as well as final committed cells, including the complete duplicate rows.
    rows = regs + end
    logical = dict(
        events=events(ds, names),
        dispatch=[(names.origin(m), c, tuple(map(names.key, k))) for m, c, k in dispatch],
        pages=[(sid, t, step, r, names.layout(p)) for sid, t, step, r, p in rows],
    )

    def cpu(t):
        return t.detach().cpu()

    numeric = dict(
        log_mass=cpu(torch.cat([p.log_mass for *_, p in rows])),
        value=cpu(torch.cat([p.value for *_, p in rows])),
        byte_logp=cpu(torch.stack([d.log_probs for d in flat])),
        joint_logp=cpu(torch.stack([d.log_policy for d in flat])),
        nll=cpu(batch.mean_nll),
        surrogate=cpu(obj),
    )
    return Witness(
        [[[e.inst for e in d.trace] for d in seq] for seq in ds],
        logical,
        numeric,
        {k: (None if v is None else cpu(v)) for k, v in grads.items()},
    )


def compare(old, new):
    c = Errors()
    for k, v in old.logical.items():
        require(v == new.logical[k], f"changed {k}")
    for k, x in old.numeric.items():
        c(k, x, new.numeric[k])
    require(old.grads.keys() == new.grads.keys(), "parameter schema drift")
    unused = []
    near = {}
    for k, x in old.grads.items():
        y = new.grads[k]
        require((x is None) == (y is None), f"unused parameter mask: {k}")
        if x is None:
            unused.append(k)
            continue
        c("parameter_grad", x, y, grad=True)
        near[k] = dict(reference_max=x.abs().max().item(), error=(x - y).abs().max().item())
    return dict(
        max_errors=c.maxima,
        gradient_errors=near,
        unused=unused,
        checks=dict(c.counts),
        scalar_counts={k: v.numel() for k, v in old.numeric.items()},
        exact={k: dict(count=len(v), sha256=digest_rows(v)) for k, v in old.logical.items()},
    )


def cell(run, name, meta, old):
    start = clock()
    n = 32 if name == "packed-b32" else 4
    forced = name == "replay-b4"
    expect = n * 128
    selected, windows, manifest, data = training_data(n)
    p1 = json.loads((HERE / "receipts/memory-program-p1/attempt1/identity.json").read_text())
    for k in ("manifest", "panel", "shard"):
        require(data[k] == p1["data"][k], "wrong P1 source")
    if n == 4:
        require(data == p1["data"], "changed B4 rows")
    require(
        len({r["sha256"] for r in selected}) == len({r["doc"] for r in selected}) == n, "nonindependent batch"
    )
    net = base_model().cuda()
    torch.manual_seed(7411)
    model = (TimedPolicy if name == "reference-b4" else PackedTimed)().cuda().train()
    rng = torch.Generator(device="cuda").manual_seed(7412)
    init = digest_state(model)
    require(init == p1["initial_policy"], "initial weights differ")
    opt = None if forced else torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    limits = Limits(pattern=16)
    provider = defaultdict(Counter)
    append = Counter()
    page_log = []
    dispatch = []
    tm = {}

    def record_page(s, step, reg, p):
        page_log.append((s.id, s.visible, step, reg, p))

    model.record_page = record_page  # never consumed by the controller; same observer cost in every cell

    def progress(phase, scored):
        dump(
            run / "progress.json",
            dict(
                cell=name,
                phase="replay" if forced and phase == "sampling" else phase,
                targets_scored=scored,
                optimizer_steps=int(phase == "updated"),
                trainer_pid=os.getpid(),
            ),
        )

    progress("setup", 0)
    with ExitStack() as stack:
        pages, bank, record, sessions, lives = mount_batch(
            stack, model, selected, manifest, provider, append, limits, progress, dispatch
        )
        require(
            (pages.desc.version, pages.exe_sha, EXE_SHA, ANCHOR_SHA)
            == tuple(p1[k] for k in ("mount", "branch_exe_sha", "position_and_live_exe_sha", "base_sha")),
            "P1 mount/base changed",
        )
        data = dict(**data, window129_sha=[hashlib.sha256(w).hexdigest() for w in windows])
        identity = dict(
            **meta,
            data=data,
            base_sha=ANCHOR_SHA,
            branch_exe_sha=pages.exe_sha,
            position_and_live_exe_sha=EXE_SHA,
            mount=pages.desc.version,
            limits=asdict(limits),
            init_seed=7411,
            sampling_seed=None if forced else 7412,
            initial_policy=init,
        )
        dump(run / (name + "-identity.json"), identity)
        baseline = 2 * torch.arange(128, 0, -1, device="cuda").float()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        resident = torch.cuda.memory_allocated()
        enter = clock()
        tm["setup_ns"] = enter - start

        def encode(x):
            require(x.shape == (n, 128), "encoder geometry")
            ts = clock()
            h, pr = backbone(net, x)
            logits = pr.log()
            torch.cuda.synchronize()
            tm["backbone_ns"] = clock() - ts
            return h, logits

        batch = teacher_force(
            model,
            encode,
            sessions,
            windows,
            mode="replay" if forced else "sample",
            generator=None if forced else rng,
            replay=old.plans if forced else None,
            key_price=0.001,
            instruction_price=0.0001,
            baseline=baseline,
        )
        require(
            batch.targets == expect and all(s.visible == 129 for s in sessions), "wrong training exposure"
        )
        if forced:
            require(batch.objective is None, "replay is not a sample")
            # Compare the gradient of the *reference sampled path*; don't optimize the forced replay.
            obj = torch.stack(
                [
                    _objective(
                        ds,
                        torch.stack([-d.log_probs[y] for d, y in zip(ds, w[1:], strict=True)]),
                        0.001,
                        0.0001,
                        baseline,
                    ).loss
                    for ds, w in zip(batch.decisions, windows, strict=True)
                ]
            ).mean()
        else:
            obj = batch.objective
        torch.cuda.synchronize()
        fwd = clock()
        require(torch.isfinite(obj).item(), "nonfinite objective")
        obj.backward()
        torch.cuda.synchronize()
        bwd = clock()
        # Capture pre-clip adjoints before an in-place clip, with explicit (inclusive) timing in every cell.
        grads = {
            k: (None if p.grad is None else p.grad.detach().clone()) for k, p in model.named_parameters()
        }
        torch.cuda.synchronize()
        snap = clock()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        if opt is not None:
            opt.step()
        torch.cuda.synchronize()
        end = clock()
        tm.update(
            forward_ns=fwd - enter,
            backward_ns=bwd - fwd,
            gradient_snapshot_ns=snap - bwd,
            clip_update_ns=end - snap,
            batch_ns=end - enter,
        )
        peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        progress("replayed" if forced else "updated", expect)
        require(all(torch.isfinite(p).all().item() for p in model.parameters()), "nonfinite parameter")
        final = digest_state(model)
        require((final == init) == forced, "update invariant")
        norms = {k: None if v is None else v.norm().item() for k, v in grads.items()}
        require(norm.item() > 0, "zero gradient for complete batch")
        ck_sha = None
        if not forced:
            ck = run / (name + ".pt")
            tmp = ck.with_suffix(".tmp")
            torch.save(
                dict(
                    model=model.state_dict(),
                    optimizer=opt.state_dict(),
                    rng=rng.get_state(),
                    step=1,
                    cell=name,
                    targets=expect,
                    data=data,
                    base_sha=ANCHOR_SHA,
                    ref=meta["ref"],
                    policy_sha=final,
                ),
                tmp,
            )
            tmp.replace(ck)
            ck_sha = sha(ck)
        snap = witness(batch, sessions, dispatch, page_log, grads, obj) if name in CELLS[:2] else None
        names = BindingNames(sessions)
        ev = snap.logical["events"] if snap is not None else events(batch.decisions, names)
        report = dict(
            cell=name,
            mode="fixed_trace" if forced else "fresh_sample",
            ref=meta["ref"],
            targets=batch.targets,
            optimizer_steps=int(not forced),
            times=tm,
            emitted_event_digest=digest_rows(ev),
            registers_observed=len(page_log),
            provider_dispatches=len(dispatch),
            policy_host_ns=model.ns,
            policy_calls=model.calls,
            provider=provider,
            archive=record.stats,
            live=sum([l.stats for l in lives], Counter()),
            live_append=append,
            m20_pages=bank.pages_fetched,
            m20_batches=bank.page_batches,
            resident_bytes=resident,
            peak_allocated=peak,
            peak_reserved=reserved,
            rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            gradient_norm_before_clip=norm.item(),
            preclip_gradient_norms=norms,
            final_policy=final,
            checkpoint_sha=ck_sha,
            train_nll=batch.mean_nll.item(),
            docs_nll=[
                torch.stack([-d.log_probs[y].detach() for d, y in zip(ds, w[1:], strict=True)]).mean().item()
                for ds, w in zip(batch.decisions, windows, strict=True)
            ],
            activity=activity(batch.decisions),
        )
        dump(run / (name + ".json"), report)
        if forced:
            dump(run / "natural-parity.json", compare(old, snap))
        print(f"{name}: {expect} targets, {tm['batch_ns'] / 1e9:.3f} s, update={int(not forced)}", flush=True)
    return snap if name == "reference-b4" else None


def main(run):
    require(platform.node() == "lambda01" and os.environ.get("OMP_NUM_THREADS") == "4", "wrong host/threads")
    lease = check_gpu_owner()
    torch.cuda.set_per_process_memory_fraction(0.55)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    meta = dict(
        ref=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        owner=lease,
        trainer_pid=os.getpid(),
        host=platform.node(),
        torch=str(torch.__version__),
        device=torch.cuda.get_device_name(),
        python=sys.version.split()[0],
    )
    parity = exercise("cuda")
    dump(run / "device-parity.json", dict(**meta, **parity))
    print("CUDA fixed-program conformance passed", flush=True)
    old = None
    for name in CELLS:
        gc.collect()
        torch.cuda.empty_cache()
        result = cell(run, name, meta, old)
        old = result if name == "reference-b4" else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    main(ap.parse_args().run)
