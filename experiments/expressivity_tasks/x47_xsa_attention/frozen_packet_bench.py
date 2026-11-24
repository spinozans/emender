"""T3: charge the frozen host producer and actually run the standalone learner after release."""

from __future__ import annotations
import argparse, copy, gc, hashlib, json, os, subprocess
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from time import perf_counter_ns as clock, time_ns

import torch
from ndm.memory_program import Limits, PackedMemoryPolicy, teacher_force
from ndm.memory_program.frozen_packet import FrozenPacket, PacketBinding, Window, write_packet
from ndm.memory_program.serving import StreamUniforms
from .host_vjp_bench import prepare, Packet, SEED, CHUNK_SHA
from .large_host_bench import STATE_SHA
from .memory_program_profile import activity, mount_batch, training_data
from .packed_program_profile import BindingNames, digest_rows, events
from .serving_bench import CK, CK_SHA, process_usage
from .serving_verify import ledger, fields
from .wide_host import build
from .workbench_witness import digest_state, require
from .workbench_run import dump, sha

CAP = 9 << 20
SIZES = {"retained": 149855670, "replicated": 2221786584}
OPT = dict(lr=0.002, weight_decay=0.01, foreach=False)


def canonical(d):
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def tensor_sha(v):
    assert v.device.type == "cpu"
    return hashlib.sha256(v.detach().contiguous().numpy().tobytes()).hexdigest()


def binding(name, selected, windows, data, idn):
    spec = dict(
        weights=STATE_SHA,
        host=name,
        parameters=SIZES[name],
        native=idn["host"],
        chunk=CHUNK_SHA,
        recipe=sha(Path(__file__).with_name("wide_host.py")) if name == "replicated" else "original",
        code=idn["ref"],
        torch=idn["torch"],
        device=idn["device"],
        mode="frozen train-mode chunk BF16 autocast / FP32 weights, batch of four",
        bridge="post-norm hidden[:256], 256 raw next-byte logits (FP32)",
    )
    source = canonical({k: data[k] for k in ("manifest", "panel", "shard")})
    return PacketBinding(
        canonical(spec),
        source,
        "train",
        256,
        tuple(Window.observe(r["doc"], r["start"], w[:-1]) for r, w in zip(selected, windows, strict=True)),
    ), spec


@torch.no_grad()
def forward(enc, x):
    require(
        x.dtype == torch.long
        and x.shape[0] % 4 == 0
        and enc.training
        and not any(p.requires_grad for p in enc.parameters()),
        "invalid frozen host",
    )
    return torch.cat([enc(chunk.cuda()).cpu() for chunk in x.split(4)])


def step(
    run, name, policy, make_encode, data_set, progress, *, coupled, capture=None, capture_scores=False,
    rollout=teacher_force, mount=mount_batch,
):
    """Identical learner in both processes. Host invocation / packet verification are not policy inputs."""
    start = clock()
    selected, ws, man, data = data_set
    stat = defaultdict(Counter)
    disp = []
    append = Counter()
    limits = Limits(pattern=16)
    init = digest_state(policy)
    holder = {}
    opt = torch.optim.AdamW(policy.parameters(), **OPT)
    opt.zero_grad(set_to_none=True)
    with ExitStack() as stack:
        pages, _, record, ss, lives = mount(
            stack, policy, selected, man, stat, append, limits, progress, disp
        )
        rng = StreamUniforms(ss, data["train_rows"], 129, SEED, "cpu")
        baseline = torch.arange(128, 0, -1).float() * 2
        pids = [pages.proc.pid, record.worker.proc.pid, *[l.native.proc.pid for l in lives]]
        src = make_encode(ss)
        cpu0, kid0, _ = process_usage(pids)

        def encode(ids):
            require(ids.shape == (32, 128) and not holder, "more than one host call/geometry")
            tm = clock()
            h, p = src(ids)
            holder["ns"] = clock() - tm
            require(
                not h.requires_grad
                and not p.requires_grad
                and h.device.type == p.device.type == "cpu"
                and h.dtype == p.dtype == torch.float32,
                "unfrozen or missing CPU features",
            )
            holder["sha"] = tensor_sha(torch.cat((h, p), -1))
            return h, p

        gc.collect()
        if coupled:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        ts = clock()
        batch = rollout(
            policy,
            encode,
            ss,
            ws,
            mode="sample",
            generator=rng,
            baseline=baseline,
            key_price=0.001,
            instruction_price=0.0001,
        )
        fwd = clock()
        require(
            batch.targets == 4096
            and all(
                s.visible == 129 and not s.pending and bytes(l.data) == w
                for s, l, w in zip(ss, lives, ws, strict=True)
            ),
            "incomplete observations",
        )
        batch.objective.backward()
        back = clock()
        # Hash before clip; timings distinguish this conformance work from the actual optimizer.
        gradients = {}
        ge = 0
        for k, p in policy.named_parameters():
            g = p.grad
            if g is not None:
                require(g.shape == p.shape and g.isfinite().all().item(), "invalid program adjoint")
                ge += g.numel()
                if capture is not None:
                    capture("grad", k, g)
            gradients[k] = None if g is None else tensor_sha(g)
        clipped_start = clock()
        norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True))
        require(norm > 0, "no update")
        opt.step()
        end = clock()
        progress("updated", 4096)
        cpu1, kid1, rss = process_usage(pids)
        after = digest_state(policy)
        require(after != init and ge > 0, "unchanged policy")
        ds = sum(batch.decisions, [])
        names = BindingNames(ss)
        act = activity(batch.decisions)
        del act["outcomes"], act["lookups"], act["recall_lags"]
        mom = {}
        for k, p in policy.named_parameters():
            if capture is not None:
                capture("param", k, p)
            if p not in opt.state:
                continue
            st = opt.state[p]
            require(
                st["step"].item() == 1 and all(t.isfinite().all().item() for t in st.values()),
                "invalid optimizer state",
            )
            mom[k] = {s: tensor_sha(t) for s, t in st.items()}
            if capture is not None:
                for s, t in st.items():
                    capture("adam." + s, k, t)
        vectors = torch.stack([d.log_probs.detach() for d in ds])
        joint = torch.stack([d.log_policy.detach() for d in ds])
        if capture_scores:
            for k, z in [
                ("byte_vectors", vectors),
                ("joint_scores", joint),
                ("nll", batch.mean_nll),
                ("objective", batch.objective),
            ]:
                capture("score", k, z)
        checks = dict(
            input_sha=holder["sha"],
            byte_vectors=tensor_sha(vectors),
            joint_scores=tensor_sha(joint),
            events=digest_rows(events(batch.decisions, names)),
            per_document_events=[digest_rows(events([row], names)) for row in batch.decisions],
            dispatch=digest_rows([(names.origin(o), c, tuple(map(names.key, ks))) for o, c, ks in disp]),
            nll=tensor_sha(batch.mean_nll),
            objective=tensor_sha(batch.objective),
            before=init,
            after=after,
            gradients=gradients,
            optimizer=canonical(mom),
            gradient_elements=ge,
            grad_norm=norm,
        )
        times = dict(
            setup_ns=ts - start,
            encode_or_copy_ns=holder["ns"],
            forward_ns=fwd - ts,
            backward_ns=back - fwd,
            gradient_attestation_ns=clipped_start - back,
            clip_update_ns=end - clipped_start,
            wall_to_update_ns=end - ts,
        )
        rep = dict(
            cell=name,
            targets=4096,
            mode="fresh_sample",
            optimizer_steps=1,
            coupled=coupled,
            checks=checks,
            activity=act,
            nll=batch.mean_nll.item(),
            objective=batch.objective.item(),
            provider=stat,
            archive=record.stats,
            append=append,
            dispatches=len(disp),
            times=times,
            targets_s=4096e9 / times["wall_to_update_ns"],
            cpu_core_s=cpu1 - cpu0 + kid1 - kid0,
            rss_kib=rss[0],
            child_rss_kib=sum(rss[1:]),
            peak_allocated=torch.cuda.max_memory_allocated() if coupled else 0,
            peak_reserved=torch.cuda.max_memory_reserved() if coupled else 0,
        )
    rep["lifetime_ns"] = clock() - start
    dump(run / (name + ".json"), rep)
    print(
        f"{name}: {rep['targets_s']:.1f} targets/s to fresh AdamW step, train NLL {rep['nll']:.5f}",
        flush=True,
    )
    return rep


def producer(run):
    begin = clock()
    enc, base = prepare(run)
    enc.requires_grad_(False)
    idn = json.loads((run / "identity.json").read_text())
    idn["base_prepare_ns"] = clock() - begin
    idn.update(frozen_host=True, opt=OPT, clip=1.0, artifact_cap=CAP)
    dump(run / "identity.json", idn)
    data_set = training_data(32)
    selected, ws, _, data = data_set
    x = torch.tensor([list(w[:-1]) for w in ws])
    idn["input128_sha"] = [hashlib.sha256(w[:-1]).hexdigest() for w in ws]
    dump(run / "identity.json", idn)
    total = 0
    for name in SIZES:

        def prog(phase, k):
            dump(
                run / "progress.json",
                dict(cell=name, phase=phase, total_targets=total + k, trainer_pid=os.getpid()),
            )

        prog("host setup", 0)
        torch.cuda.synchronize()
        tm = clock()
        audit = None
        if name == "replicated":
            wide, audit = build(enc.net)
            enc = Packet(wide).train().requires_grad_(False)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        made = clock() - tm
        require(
            sum(p.numel() for p in enc.parameters()) == SIZES[name]
            and not any(isinstance(m, torch.nn.Dropout) and m.p for m in enc.modules()),
            "wrong or stochastic frozen host",
        )
        dump(
            run / (name + "-build.json"),
            dict(parameters=sum(p.numel() for p in enc.parameters()), build_ns=made, shape_check=audit),
        )
        t = clock()
        long = forward(enc, x[:4])
        short = forward(enc, x[:4, :63].contiguous())
        actual = long[:, :63]
        err = (actual - short).abs().max().item()
        qual_ns = clock() - t
        dump(
            run / (name + "-prefix.json"),
            dict(
                elements=actual.numel(),
                violations=int((~torch.isclose(actual, short, rtol=0.01, atol=0.04)).sum()),
                max_error=err,
                seconds=qual_ns / 1e9,
            ),
        )
        torch.testing.assert_close(actual, short, rtol=0.01, atol=0.04)
        del actual, short, long

        def source(_):
            def encode(ids):
                return forward(enc, ids).split(256, dim=-1)

            return encode

        rep = step(run, name + "-coupled", copy.deepcopy(base), source, data_set, prog, coupled=True)
        b, model_spec = binding(name, selected, ws, data, idn)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = clock()
        feat = forward(enc, x)
        mid = clock()
        hp = tensor_sha(feat)
        require(hp == rep["checks"]["input_sha"], "nondeterministic host packet")
        h, n = write_packet(run / (name + ".f32"), b, feat, byte_limit=CAP)
        end = clock()
        del feat
        require(
            not enc.saved and all(p.grad is None for p in enc.parameters()), "unexpected host tape/gradient"
        )
        dump(
            run / (name + "-packet.json"),
            dict(
                model_spec=model_spec,
                binding=asdict(b),
                sha256=h,
                payload_sha=hp,
                bytes=n,
                payload_bytes=32 * 128 * 512 * 4,
                build_ns=made,
                causal_prefix_ns=qual_ns,
                prefix_elements=4 * 63 * 512,
                prefix_max_error=err,
                encode_copy_ns=mid - t,
                hash_publish_ns=end - mid,
                extract_ns=end - t,
                peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved(),
                shape_check=audit,
                host_gradients=0,
            ),
        )
        prog("published", 4096)
        total += 4096
    dump(
        run / "progress.json",
        dict(cell="all", phase="complete", total_targets=total, trainer_pid=os.getpid()),
    )


def no_card():
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not os.environ.get("GPU_LEASE_HELD")
        and not torch.cuda.is_available()
        and not torch.cuda.is_initialized()
        and os.environ.get("OMP_NUM_THREADS") == "4",
        "consumer requires CUDA-invisible CPU-only process",
    )


def learner(root, *, bind=binding, lease_cap=145):
    no_card()
    aa, good = ledger(root, 8192, work_key="total_targets", actor_key="trainer_pid")
    require(len(good) == 1 and sum(a["upper"] for a in aa) <= lease_cap, "no released, qualified producer")
    run = good[0]
    release = fields(run / "released")
    stamp = time_ns()
    idn = json.loads((run / "identity.json").read_text())
    require(
        stamp >= int(release["released_epoch"]) * 10**9
        and idn["ref"] == subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        and CK_SHA == sha(CK),
        "early learner or changed code/weights",
    )
    cpu = root / "cpu-learner"
    cpu.mkdir(exist_ok=False, mode=0o700)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    ck = torch.load(CK, map_location="cpu", weights_only=True)
    base = PackedMemoryPolicy(256).train()
    base.load_state_dict(ck["model"])
    require(digest_state(base) == ck["policy_sha"] == idn["policy_sha"], "wrong learner policy")
    data_set = training_data(32)
    selected, ws, man, data = data_set
    require(
        idn["data"] == dict(data, window129_sha=[hashlib.sha256(w).hexdigest() for w in ws])
        and man["mount"] == idn["mount"],
        "consumer source drift",
    )
    total = 0
    for name in SIZES:

        def prog(phase, k):
            dump(
                cpu / "progress.json",
                dict(cell=name, phase=phase, total_targets=total + k, learner_pid=os.getpid()),
            )

        prog("open verified packet", 0)
        b, spec = bind(name, selected, ws, data, idn)
        meta = json.loads((run / (name + "-packet.json")).read_text())
        require(meta["model_spec"] == spec, "wrong producer recipe")
        tm = clock()
        packet = FrozenPacket.load(run / (name + ".f32"), b, meta["sha256"], byte_limit=CAP)
        load_ns = clock() - tm
        rep = step(cpu, name + "-offline", copy.deepcopy(base), packet.encoder, data_set, prog, coupled=False)
        ref = json.loads((run / (name + "-coupled.json")).read_text())
        require(
            rep["checks"] == ref["checks"]
            and rep["activity"] == ref["activity"]
            and rep["checks"]["input_sha"] == meta["payload_sha"],
            "post-release update mismatch",
        )
        dump(
            cpu / (name + "-read.json"),
            dict(expected_sha=meta["sha256"], load_ns=load_ns, bytes=meta["bytes"], equal_update=True),
        )
        del packet
        total += 4096
    no_card()
    dump(
        cpu / "complete.json",
        dict(
            total_targets=total,
            optimizer_steps=2,
            learner_pid=os.getpid(),
            start_epoch_ns=stamp,
            end_epoch_ns=time_ns(),
            visible=os.environ["CUDA_VISIBLE_DEVICES"],
            cuda_available=torch.cuda.is_available(),
            cuda_initialized=torch.cuda.is_initialized(),
            producer_release_epoch=int(release["released_epoch"]),
            threads=torch.get_num_threads(),
            ref=idn["ref"],
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--learner", action="store_true")
    args = ap.parse_args()
    (learner if args.learner else producer)(args.path)
