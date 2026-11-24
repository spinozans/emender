"""T4: train from exact observed FP32 serving frontiers, separate from the vetoed BF16 chunk."""

import argparse, copy, gc, hashlib, json, os
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter_ns as clock
import torch
from ndm.memory_program.stream_packet import record_frontiers
from ndm.memory_program.frozen_packet import Window, write_packet
from ndm.memory_program.resident_host import ResidentArena
from .frozen_packet_bench import (
    binding as chunk_binding,
    canonical,
    learner,
    step,
    SIZES,
    CAP,
    OPT,
    tensor_sha,
)
from .host_vjp_bench import prepare
from .large_host_check import stock
from .memory_program_profile import training_data
from .wide_host import build
from .workbench_run import dump
from .workbench_witness import require


def binding(name, rows, ws, data, idn):
    b, d = chunk_binding(name, rows, ws, data, idn)
    del d["chunk"]
    d.update(
        mode="frozen FP32 eval recurrence: blocks of 16, one real observation per host call",
        bridge="post-norm hidden[:256], 256 serving byte log probabilities (FP32)",
        graph_width=16,
        context=128,
    )
    return replace(b, encoder_sha=canonical(d)), d


def check_work(pool, before, bs, t):
    delta = pool.stats - before
    require(
        delta["prefill_tokens"] == bs
        and delta["step_tokens"] == bs * (t - 1)
        and delta["released"] == bs
        and pool.empty
        and all(not v.count_nonzero().item() for v in pool.slots.values()),
        "wrong real host work or scrub",
    )
    return dict(delta)


def qualify(run, name, pool, b, ws):
    t = clock()
    b16 = replace(b, rows=b.rows[:16])
    full = record_frontiers(pool, b16, ws[:16], byte_limit=CAP)
    short = [w[:63] for w in ws[:16]]
    bs = replace(
        b16, rows=tuple(Window.observe(r.document, r.start, w) for w, r in zip(short, b16.rows, strict=True))
    )
    a = record_frontiers(pool, bs, short, byte_limit=CAP)
    require(torch.equal(a, full[:, :63]), "short genuine stream differs")
    errors = {}
    counts = {}
    scaled = {}

    def cmp(key, z, h, lp):
        ref = torch.cat((h[:, :256], lp), 1).cpu()
        errors[key] = (z - ref).abs().max().item()
        counts[key] = z.numel()
        scaled[key] = ((z - ref).abs() / (5e-5 + 2e-5 * ref.abs())).max().item()
        torch.testing.assert_close(z, ref, atol=5e-5, rtol=2e-5)
        require(torch.equal(z[:, 256:].argmax(1), lp.argmax(1).cpu()), "different stock byte")

    for j in (1, 31, 63):
        cmp("stock-" + str(j), full[:4, j - 1], *stock(pool.net, [w[:j] for w in ws[:4]]))
    hs = [pool.admit(f"segment:{r.document}") for r in b16.rows]
    cmp("streaming-segment-128", full[:, -1], *pool.prefill(hs, ws[:16]))
    pool.release(hs)
    require(
        pool.empty and all(not v.count_nonzero().item() for v in pool.slots.values()), "qualification residue"
    )
    proof = dict(
        short_exact=a.numel(),
        elements=counts,
        max_errors=errors,
        max_tolerance_fraction=scaled,
        argmax=True,
        scrubbed=True,
        wall_ns=clock() - t,
    )
    dump(run / (name + "-prefix.json"), proof)
    return proof


@torch.no_grad()
def producer(run):
    t = clock()
    enc, base = prepare(run)
    net = enc.net.eval().requires_grad_(False)
    del enc
    idn = json.loads((run / "identity.json").read_text())
    idn.update(
        base_prepare_ns=clock() - t,
        frozen_host=True,
        opt=OPT,
        clip=1.0,
        artifact_cap=CAP,
        precision="FP32 eval recurrence / normalized byte log probabilities",
        microbatch=16,
    )
    ds = training_data(32)
    rows, windows, _, data = ds
    words = [w[:-1] for w in windows]
    idn["input128_sha"] = [hashlib.sha256(w).hexdigest() for w in words]
    dump(run / "identity.json", idn)
    total = 0
    for name in SIZES:

        def prog(phase, k):
            dump(
                run / "progress.json",
                dict(cell=name, phase=phase, total_targets=total + k, trainer_pid=os.getpid()),
            )

        prog("empty serving host", 0)
        torch.cuda.synchronize()
        t = clock()
        audit = None
        if name == "replicated":
            net, audit = build(net)
        gc.collect()
        torch.cuda.empty_cache()
        require(
            sum(p.numel() for p in net.parameters()) == SIZES[name]
            and not net.training
            and not any(p.requires_grad for p in net.parameters()),
            "unbound frozen host",
        )
        torch.cuda.synchronize()
        made = clock() - t
        dump(run / (name + "-build.json"), dict(parameters=SIZES[name], build_ns=made, shape_check=audit))
        b, spec = binding(name, rows, windows, data, idn)
        t = clock()
        with ResidentArena(
            net,
            16,
            128,
            (16,),
            byte_budget=(256 << 20 if name == "retained" else 1 << 30),
            private_budget=256 << 20,
            segmented_prefill=True,
        ) as pool:
            torch.cuda.synchronize()
            prep = clock() - t
            dump(
                run / (name + "-arena.json"),
                dict(setup_ns=prep, private_bytes=pool.backend.pool_bytes, tensor_bytes=pool.tensor_bytes),
            )
            proof = qualify(run, name, pool, b, words)
            before = pool.stats.copy()

            def source(_, arena=pool):
                def encode(ids):
                    require(
                        ids.device.type == "cpu"
                        and ids.dtype == torch.long
                        and ids.shape == (32, 128)
                        and bool(((ids >= 0) & (ids < 256)).all()),
                        "real unpadded inputs required",
                    )
                    out = record_frontiers(
                        arena, b, [z.tobytes() for z in ids.to(torch.uint8).numpy()], byte_limit=CAP
                    )
                    return out.split(256, -1)

                return encode

            with torch.enable_grad():
                rep = step(run, name + "-coupled", copy.deepcopy(base), source, ds, prog, coupled=True)
            del source
            work1 = check_work(pool, before, 32, 128)
            before = pool.stats.copy()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            ts = clock()
            feat = record_frontiers(pool, b, words, byte_limit=CAP)
            torch.cuda.synchronize()
            mid = clock()
            hp = tensor_sha(feat)
            require(hp == rep["checks"]["input_sha"], "unequal full-length serving frontiers")
            h, n = write_packet(run / (name + ".f32"), b, feat, byte_limit=CAP)
            end = clock()
            del feat
            work2 = check_work(pool, before, 32, 128)
            require(
                work1 == work2 and all(p.grad is None for p in net.parameters()),
                "unexpected work or gradients",
            )
            dump(
                run / (name + "-packet.json"),
                dict(
                    model_spec=spec,
                    binding=asdict(b),
                    sha256=h,
                    payload_sha=hp,
                    bytes=n,
                    payload_bytes=32 * 128 * 512 * 4,
                    build_ns=made,
                    arena_setup_ns=prep,
                    prefix_elements=sum(proof["elements"].values()),
                    prefix_max_error=max(proof["max_errors"].values()),
                    prefix_exact=proof["short_exact"],
                    encode_copy_ns=mid - ts,
                    hash_publish_ns=end - mid,
                    extract_ns=end - ts,
                    peak_allocated=torch.cuda.max_memory_allocated(),
                    peak_reserved=torch.cuda.max_memory_reserved(),
                    shape_check=audit,
                    host_gradients=0,
                    work=work2,
                ),
            )
            prog("published", 4096)
        del pool
        total += 4096
    dump(
        run / "progress.json",
        dict(cell="all", phase="complete", total_targets=total, trainer_pid=os.getpid()),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--learner", action="store_true")
    args = ap.parse_args()
    if args.learner:
        learner(args.path, bind=binding, lease_cap=160)
    else:
        producer(args.path)
