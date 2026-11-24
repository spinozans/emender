"""T6 extraction-only producer. Fresh complete frontier tables, not endpoint prefill or free decode."""

from __future__ import annotations
import argparse, gc, hashlib, json, os
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter_ns as clock
import torch
from ndm.memory_program.frozen_packet import Window, write_packet
from ndm.memory_program.resident_host import ResidentArena
from ndm.memory_program.stream_packet import record_frontiers
from ndm.memory_program.segment_packet import SegmentRecorder
from .frozen_packet_bench import CAP, SIZES, canonical, tensor_sha
from .host_vjp_bench import prepare
from .memory_program_profile import training_data
from .serving_verify import HERE, load
from .stream_packet_bench import binding as serial_binding
from .wide_host import build
from .workbench_run import dump
from .workbench_witness import require

ORDER = {
    "retained": ("serial", "segment", "segment", "serial"),
    "replicated": ("segment", "serial", "serial", "segment"),
}
T4 = HERE / "receipts/memory-program-t4/attempt1"


def binding(size, method, rows, ws, data, idn):
    b, spec = serial_binding(size, rows, ws, data, idn)
    if method == "segment":
        del spec["graph_width"]
        spec.update(
            mode="frozen FP32 eval recurrence: every output of <=64 observed byte pieces, blocks of 16",
            piece=64,
        )
    elif method != "serial":
        raise ValueError("unknown producer")
    return replace(b, encoder_sha=canonical(spec)), spec


def produce(net, size, method, b, ws, *, path=None, expected=None):
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    empty = torch.cuda.memory_allocated()
    beg = clock()
    small = size == "retained"
    ctx = (
        ResidentArena(
            net,
            16,
            128,
            (16,),
            byte_budget=(256 << 20 if small else 1 << 30),
            private_budget=256 << 20,
            segmented_prefill=True,
        )
        if method == "serial"
        else SegmentRecorder(net, 16, 128, byte_budget=(128 if small else 512) << 20)
    )
    torch.cuda.synchronize()
    ready = clock()
    with ctx as owner:
        base = torch.cuda.memory_allocated()
        out = (
            record_frontiers(owner, b, ws, byte_limit=CAP)
            if method == "serial"
            else owner.record(b, ws, byte_limit=CAP)
        )
        torch.cuda.synchronize()
        mid = clock()
        hp = tensor_sha(out)
        if expected is not None:
            require(hp == expected, "repeated packet changed")
        h, n = write_packet(path, b, out, byte_limit=CAP) if path is not None else (None, 0)
        end = clock()
        if method == "serial":
            work = dict(owner.stats)
            states = owner.slots
            traffic = dict(owner.backend.buckets[16].work)
            private = owner.backend.pool_bytes
            scrub = owner.empty
        else:
            work = dict(owner.work)
            states = owner.state
            traffic = dict(owner.host.stats)
            private = 0
            scrub = not owner.busy
        require(scrub and all(not v.count_nonzero().item() for v in states.values()), "residual state")
        storage = owner.tensor_bytes
    torch.cuda.synchronize()
    last = clock()
    return out, dict(
        payload_sha=hp,
        sha256=h,
        bytes=n,
        shape=list(out.shape),
        work=work,
        traffic=traffic,
        scrubbed=True,
        scratch_bytes=storage,
        graph_private=private,
        starting_allocated=empty,
        prepared_allocated=base,
        peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),
        prepare_ns=ready - beg,
        features_ns=mid - ready,
        publish_hash_ns=end - mid,
        extract_ns=end - ready,
        check_and_close_ns=last - end,
        lifecycle_ns=last - beg,
    )


def compare(x, y):
    require(x.shape == y.shape and x.shape[-1] == 512, "missing features")
    delta = (x - y).abs()
    scaled = delta.double() / (5e-5 + 2e-5 * x.double().abs())
    return dict(
        elements=x.numel(),
        positions=x.shape[0] * x.shape[1],
        unequal_bits=int((x.contiguous().view(torch.int32) != y.contiguous().view(torch.int32)).sum()),
        violations=int((scaled > 1).sum()),
        max_abs=delta.max().item(),
        max_fraction=scaled.max().item(),
        argmax_changes=int((x[:, :, 256:].argmax(-1) != y[:, :, 256:].argmax(-1)).sum()),
    )


@torch.no_grad()
def main(run):
    enc, p = prepare(run)
    net = enc.net.eval().requires_grad_(False)
    del enc, p
    idn = json.loads((run / "identity.json").read_text())
    idn["producer_pid"] = idn.pop("trainer_pid")
    idn.update(
        frozen_host=True,
        microbatch=16,
        precision="FP32 eval recurrence / normalized byte log probabilities",
        artifact_cap=CAP,
    )
    rows, windows, _, data = training_data(32)
    words = [w[:-1] for w in windows]
    idn["input128_sha"] = [hashlib.sha256(w).hexdigest() for w in words]
    dump(run / "identity.json", idn)
    total = 0

    def progress(cell, phase):
        dump(
            run / "progress.json",
            dict(cell=cell, phase=phase, total_frontiers=total, producer_pid=os.getpid()),
        )

    for size in SIZES:
        progress(size, "qualify")
        t = clock()
        proof = None
        if size == "replicated":
            net, proof = build(net)
        torch.cuda.synchronize()
        dump(
            run / (size + "-build.json"),
            dict(count=sum(p.numel() for p in net.parameters()), proof=proof, wall_ns=clock() - t),
        )
        require(sum(p.numel() for p in net.parameters()) == SIZES[size], "wrong host size")
        bs = {}
        info = {}
        expect = {}
        qual = {}
        for method in ("serial", "segment"):
            bs[method], info[method] = binding(size, method, rows, windows, data, idn)
        x, xm = produce(net, size, "serial", bs["serial"], words)
        require(
            xm["payload_sha"] == load(T4 / (size + "-packet.json"))["payload_sha"], "serial differs from T4"
        )
        expect["serial"] = xm["payload_sha"]
        y, ym = produce(net, size, "segment", bs["segment"], words)
        expect["segment"] = ym["payload_sha"]
        qual["full"] = compare(x, y)
        for ln in (31, 65):
            short = [w[:ln] for w in words[:16]]
            b = replace(
                bs["segment"],
                rows=tuple(
                    Window.observe(r.document, r.start, w)
                    for r, w in zip(bs["segment"].rows[:16], short, strict=True)
                ),
            )
            z, _ = produce(net, size, "segment", b, short)
            qual[str(ln)] = compare(y[:16, :ln], z)
        dump(run / (size + "-oracle.json"), dict(comparisons=qual, payloads=expect, serial=xm, segment=ym))
        require(
            all(v["violations"] == v["argmax_changes"] == 0 for v in qual.values()),
            "segmented frontier qualification",
        )
        del x, y, z
        seen = dict(serial=0, segment=0)
        for method in ORDER[size]:
            seen[method] += 1
            cell = f"{size}-{method}-{seen[method]}"
            progress(cell, "extract")
            b = bs[method]
            out, m = produce(net, size, method, b, words, path=run / (cell + ".f32"), expected=expect[method])
            del out
            m.update(cell=cell, method=method, model_spec=info[method], binding=asdict(b))
            dump(run / (cell + ".json"), m)
            total += 4096
            progress(cell, "published")
            print(cell, f"{4096e9 / m['extract_ns']:.1f} verified observed frontiers/s", flush=True)
    progress("all", "complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    main(ap.parse_args().run)
