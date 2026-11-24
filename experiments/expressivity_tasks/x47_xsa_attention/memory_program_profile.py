"""P1: one TRAIN-only batch through actual sampled addresses, native I/O, BPTT and optimizer."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter_ns as clock

import numpy as np
import torch

from ndm.memory_program import Limits, MaterializedBranches, MemoryPolicy, Session, Unavailable, teacher_force
from ndm.memory_program.spans import mean_byte_values
from .memory_program_mounts import ArchiveRecords, LiveRecords
from .w3_data import HERE, OUT, MANIFEST, MAN_SHA, Pages, getwindow
from .w3_model import ANCHOR_SHA, backbone, base_model
from .workbench_native import EXE_SHA
from .workbench_run import check_gpu_owner, dump, sha
from .workbench_witness import digest_state, require


def training_data(n=4):
    require(n in (4, 32), "only registered TRAIN batches")
    ix = slice(4, 8) if n == 4 else slice(0, 32)
    receipts = HERE / "receipts/w3"
    manifest = json.loads((receipts / "train-pages.json").read_text())
    p = receipts / "panels.json"
    panel = json.loads(p.read_text())
    source = panel["source"]
    require(
        manifest["stage"] == "train"
        and sha(p) == manifest["panel_sha"]
        and sha(MANIFEST) == panel["manifest_sha"] == MAN_SHA,
        "panel integrity",
    )
    st = Path(source["path"]).stat()
    require(
        (st.st_ino, st.st_size, st.st_mtime_ns) == tuple(source[k] for k in ("inode", "size", "mtime_ns")),
        "source identity",
    )
    shard = manifest["shards"][0]
    path = OUT / "train" / shard["file"]
    require(sha(path) == shard["sha256"], "shard integrity")
    selected = panel["panels"]["train"][ix]
    windows = []
    with np.load(path, allow_pickle=False) as z, Path(source["path"]).open("rb") as f:
        raw = z["raw"]
        require(raw.shape == (32, 2049) and raw.dtype == np.uint8, "raw geometry")
        for row, x in zip(selected, raw[ix], strict=True):
            w = getwindow(f, row)
            require(w == bytes(x) and row["offset"] >= 100_600_000_000, "window/source mismatch")
            windows.append(w[:129])
    return (
        selected,
        windows,
        manifest,
        dict(
            documents=selected,
            train_rows=list(range(ix.start, ix.stop)),
            panel=sha(p),
            manifest=MAN_SHA,
            shard=shard["sha256"],
        ),
    )


class TimedPolicy(MemoryPolicy):
    def __init__(self):
        super().__init__(256)
        self.ns = Counter()
        self.calls = Counter()

    def timed(self, name, fn, *args):
        t = clock()
        out = fn(*args)
        self.ns[name] += clock() - t
        self.calls[name] += 1
        return out

    def rebuild(self, *a):
        return self.timed("rebuild", super().rebuild, *a)

    def choose_many(self, *a):
        return self.timed("choose_batch", super().choose_many, *a)

    def emit(self, *a):
        return self.timed("emit", super().emit, *a)


def measured(reader, stats, dispatch=None):
    tag = f"{reader.source}/{reader.operation}"

    def fetch(keys, cap):
        if dispatch is not None:
            dispatch.append(
                (
                    (
                        reader.source,
                        reader.version,
                        reader.operation,
                        reader.watermark() if reader.watermark else None,
                    ),
                    cap,
                    keys,
                )
            )
        t = clock()
        rows = reader.read_many(keys, cap)
        s = stats[tag]
        s["ns"] += clock() - t
        s["batches"] += 1
        s["keys"] += len(keys)
        for hits in rows:
            if isinstance(hits, Unavailable):
                s["refused: " + hits.reason] += 1
            elif not hits:
                s["empty"] += 1
            else:
                s["positive"] += 1
                s["rows"] += len(hits)
        return rows

    return replace(reader, read_many=fetch)


def activity(results):
    """Count explicit chosen operands, not assumed effects of changing them; never retain key bytes."""
    ops = Counter()
    outcomes = defaultdict(Counter)
    usage = Counter()
    lags = []
    lookups = defaultdict(Counter)
    programs = Counter()
    for ds in results:
        cells = {}
        for t, d in enumerate(ds):
            regs = []
            usage["logical_keys"] += d.requests
            programs[tuple(e.inst.op for e in d.trace)] += 1
            for e in d.trace:
                op, a = e.inst.op, e.inst.args
                ops[op] += 1
                outcomes[op][e.status] += 1
                if op == "LOOKUP":
                    name, _, cap, _ = e.reader
                    tag = f"{name}/{cap}"
                    lookups[tag][e.status] += 1
                if e.status.startswith("refused:"):
                    continue
                if op == "EMIT":
                    if a[0] >= 0:
                        for flag in regs[a[0]][1]:
                            usage["emit lineage: " + flag] += 1
                    continue
                if op == "SLICE":
                    fields = ("text", "slice_span")
                    flags = set()
                elif op == "RECALL":
                    old, (fields, flags) = cells[a[0]]
                    if t > old and e.rows:
                        lags.append(t - old)
                        flags = flags | {"old_recall"}
                else:
                    fields, flags = regs[a[0]]
                    if op == "LOOKUP":
                        if cap == "byte_edges":
                            added = ("byte", "extension")
                        elif cap == "occurrences":
                            added = (tag,)
                        elif cap == "contents":
                            added = ("text",)
                        elif cap == "following":
                            added = ("byte", "text", "following_span")
                        else:
                            raise ValueError(cap)
                        if e.rows and cap in ("contents", "following"):
                            usage["value address: " + fields[a[1]] + " -> " + tag] += 1
                            flags = flags | {"span_value"}
                        fields = fields + added
                    elif op in ("JOIN", "RELATE"):
                        other, more = regs[a[2] if op == "JOIN" else a[1]]
                        fields = fields + other
                        flags = flags | more | {op}
                    elif op == "PROJECT":
                        fields = (fields[a[1]],)
                    elif op == "CONCAT":
                        fields = fields + ("text",)
                    elif op == "SAVE":
                        cells[a[1]] = (t, (fields, flags))
                    else:
                        require(op == "FOCUS", "bad instruction trace")
                require(e.register == len(regs), "trace/register mismatch")
                regs.append((fields, flags))
    return dict(
        opcodes=dict(ops),
        outcomes=outcomes,
        lookups=lookups,
        usage=usage,
        distinct_opcode_sequences=len(programs),
        nonempty_prior_byte_recalls=len(lags),
        recall_lags=lags,
    )


def mount_batch(
    stack,
    model,
    selected,
    manifest,
    stat,
    live_wall,
    limits,
    progress,
    dispatch=None,
    *,
    bank_type=MaterializedBranches,
    pages_factory=Pages,
):
    pages = pages_factory()
    stack.callback(pages.close)
    require((pages.exe_sha, pages.desc.version) == (manifest["exe_sha"], manifest["mount"]), "mount identity")
    bank = bank_type("P100", pages.desc.version, pages, scope="whole")
    record, sessions, lives = bind_program_readers(
        stack, model, selected, pages.desc, bank, stat, live_wall, limits, progress, dispatch
    )
    return pages, bank, record, sessions, lives


def bind_program_readers(
    stack, model, selected, desc, bank, stat, live_wall, limits, progress, dispatch=None
):
    """Fresh value encoder, positional worker and private live clients; no prior session or payload reuse."""
    conf = json.loads((HERE / "b1_mounts.json").read_text())["P100"]

    def encode_text(pieces):
        return mean_byte_values(pieces, model.embed.weight)

    record = ArchiveRecords(desc, conf["files"][1], scope="whole", encode=encode_text, branches=bank)
    stack.callback(record.close)

    def wrap(r):
        return measured(r, stat, dispatch)

    shared = [wrap(r) for r in (bank.reader(), *record.readers())]
    sessions = []
    lives = []
    for j, row in enumerate(selected):
        live = LiveRecords(row["doc"], row["start"], encode=encode_text)
        stack.callback(live.close)
        lives.append(live)

        def advance(k, live=live, j=j):
            t = clock()
            live.advance(k)
            live_wall["append_ns"] += clock() - t
            scored = len(live.data) - 1
            if j == len(selected) - 1 and scored and not scored % 32:
                progress("sampling", len(selected) * scored)

        sessions.append(
            Session(
                shared + [wrap(r) for r in live.readers()],
                limits,
                advance,
                document=row["doc"],
                start=row["start"],
                session_id=live.id,
            )
        )
    return record, sessions, lives


def profile(run):
    require(
        os.environ.get("OMP_NUM_THREADS") == "4" and platform.node() == "lambda01", "registered host/threads"
    )
    lease = check_gpu_owner()
    torch.cuda.set_per_process_memory_fraction(0.55)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    start = clock()
    selected, windows, manifest, data = training_data()
    net = base_model().cuda()
    torch.manual_seed(7411)
    model = TimedPolicy().cuda().train()
    init = digest_state(model)
    rng = torch.Generator(device="cuda").manual_seed(7412)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    stat = defaultdict(Counter)
    live_wall = Counter()
    times = {}
    limits = Limits(pattern=16)

    def progress(phase, scored):
        dump(
            run / "progress.json",
            dict(
                phase=phase,
                targets_scored=scored,
                optimizer_steps=int(phase == "updated"),
                trainer_pid=os.getpid(),
            ),
        )

    progress("setup", 0)
    with ExitStack() as stack:
        pages, bank, record, sessions, lives = mount_batch(
            stack, model, selected, manifest, stat, live_wall, limits, progress
        )
        identity = dict(
            host=platform.node(),
            torch=str(torch.__version__),
            device=torch.cuda.get_device_name(),
            python=sys.version.split()[0],
            owner=lease,
            trainer_pid=os.getpid(),
            ref=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            data=data,
            base_sha=ANCHOR_SHA,
            initial_policy=init,
            init_seed=7411,
            sampling_seed=7412,
            limits=asdict(limits),
            mount=pages.desc.version,
            branch_exe_sha=pages.exe_sha,
            position_and_live_exe_sha=EXE_SHA,
        )
        dump(run / "identity.json", identity)
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.arange(128, 0, -1, device="cuda").float() * 2
        resident = torch.cuda.memory_allocated()
        start_batch = clock()
        times["setup_ns"] = start_batch - start

        def encode(x):
            t = clock()
            require(x.shape == (4, 128), "backbone geometry")
            h, p = backbone(net, x)
            logits = p.log()
            torch.cuda.synchronize()
            times["backbone_ns"] = clock() - t
            return h, logits

        batch = teacher_force(
            model,
            encode,
            sessions,
            windows,
            mode="sample",
            generator=rng,
            baseline=baseline,
            key_price=0.001,
            instruction_price=0.0001,
        )
        torch.cuda.synchronize()
        end_fwd = clock()
        require(batch.targets == 512 and all(s.visible == 129 for s in sessions), "training exposure")
        require(torch.isfinite(batch.objective).item(), "nonfinite objective")
        batch.objective.backward()
        torch.cuda.synchronize()
        end_bwd = clock()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        torch.cuda.synchronize()
        end = clock()
        times.update(
            batch_ns=end - start_batch,
            forward_ns=end_fwd - start_batch,
            backward_ns=end_bwd - end_fwd,
            clip_update_ns=end - end_bwd,
        )
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        progress("updated", 512)
        require(all(torch.isfinite(p).all().item() for p in model.parameters()), "nonfinite updated model")
        groups = {
            name: None if p.grad is None else p.grad.norm().item() for name, p in model.named_parameters()
        }
        final = digest_state(model)
        require(
            final != init and all(groups[k] and groups[k] > 0 for k in ("op.weight", "pointer.weight")),
            "inert update",
        )
        ck = run / "latest.pt"
        tmp = ck.with_suffix(".tmp")
        torch.save(
            dict(
                model=model.state_dict(),
                optimizer=opt.state_dict(),
                rng=rng.get_state(),
                step=1,
                targets=512,
                ref=identity["ref"],
                data=data,
                base_sha=ANCHOR_SHA,
                policy_sha=final,
            ),
            tmp,
        )
        tmp.replace(ck)
        live_stats = sum([l.stats for l in lives], Counter())
        used = activity(batch.decisions)
        report = dict(
            ref=identity["ref"],
            targets=batch.targets,
            optimizer_steps=1,
            sampled_train_nll=batch.mean_nll.item(),
            docs_nll=[
                float(torch.stack([-d.log_probs[y].detach() for d, y in zip(ds, w[1:], strict=True)]).mean())
                for ds, w in zip(batch.decisions, windows, strict=True)
            ],
            gradient_norm_before_clip=norm.item(),
            clipped_norms=groups,
            final_policy=final,
            checkpoint_sha=sha(ck),
            times=times,
            policy_host_ns=model.ns,
            policy_calls=model.calls,
            provider=stat,
            live_append=live_wall,
            archive=record.stats,
            live=live_stats,
            m20_pages=bank.pages_fetched,
            m20_batches=bank.page_batches,
            resident_bytes=resident,
            peak_allocated=peak,
            peak_reserved=reserved,
            trainer_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            activity=used,
            projected_seconds_one_arm_1048576=times["batch_ns"] * 2048 / 1e9,
        )
    dump(run / "profile.json", report)
    print(json.dumps(dict(times=times, targets=512, peak=peak, checkpoint=sha(ck))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    profile(parser.parse_args().out)
