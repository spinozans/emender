"""T7 TRAIN-only functionality/price of a frozen-execution control. No quality panel or device."""

import argparse
import copy
import gc
import hashlib
import os
import re
import subprocess
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from time import perf_counter_ns as clock
import torch
from ndm.memory_program import PackedMemoryPolicy, Limits
from ndm.memory_program.frozen_routes import capture_routes, replay_routes
from ndm.memory_program.frozen_packet import FrozenPacket
from ndm.memory_program.serving import StreamUniforms
from .frozen_packet_bench import CAP, OPT, SEED, canonical, no_card, step, tensor_sha
from .memory_program_profile import activity, mount_batch, training_data
from .packed_program_profile import BindingNames, digest_rows, events
from .serving_bench import CK, CK_SHA, process_usage
from .serving_verify import HERE, load, require
from .stream_packet_bench import binding
from .workbench_run import dump, sha
from .workbench_witness import digest_state

MAX = 32 * 128 * 6
PURE = ("op.weight", "prior", "transitions.weight", "pointer.weight", "pointer.bias")


def describe(tape):
    steps = [v for d in tape.programs for b in d for v in b]
    return dict(
        steps=len(steps),
        reads=sum(v.request is not None for v in steps),
        digest_bytes=sum(len(v.result) + (len(v.request) if v.request else 0) for v in steps),
        tape_sha=digest_rows(
            [(tape.authority, tape.origins)] + [(v.instruction, v.request, v.result) for v in steps]
        ),
    )


def play(root, label, model, ds, packet, authority, *, tape=None, update=False):
    beg = clock()
    limits = Limits(pattern=16)
    rows, ws, man, data = ds
    before = digest_state(model)
    pure0 = {k: tensor_sha(v) for k, v in model.named_parameters() if k in PURE}
    params = dict(model.named_parameters())
    writer_before = tensor_sha(params["writer.weight"])
    provider = defaultdict(Counter)
    append = Counter()
    dispatch = []
    opt = torch.optim.AdamW(model.parameters(), **OPT) if update else None
    generated = tape is None

    def progress(phase, k):
        dump(
            root / "progress.json",
            dict(pass_name=label, phase=phase, targets=k, pid=os.getpid()),
        )

    gc.collect()
    progress("binding", 0)
    with ExitStack() as stack:
        pages, _, record, ss, lives = mount_batch(
            stack, model, rows, man, provider, append, limits, progress, dispatch
        )
        rng = StreamUniforms(ss, data["train_rows"], 129, SEED, "cpu")
        pids = [
            pages.proc.pid,
            record.worker.proc.pid,
            *[live.native.proc.pid for live in lives],
        ]
        cpu0, kid0, _ = process_usage(pids)
        ts = clock()
        enc = packet.encoder(ss)
        with torch.set_grad_enabled(update):
            if generated:
                tape, batch = capture_routes(
                    model,
                    enc,
                    ss,
                    ws,
                    authority=authority,
                    max_steps=MAX,
                    generator=rng,
                )
            else:
                batch = replay_routes(model, enc, ss, ws, tape, authority=authority, max_steps=MAX)
        mid = clock()
        require(
            all(
                bytes(live.data) == w and s.visible == len(w) and not s.pending
                for live, s, w in zip(lives, ss, ws, strict=True)
            )
            and batch.targets == 4096,
            "lost supervised observation",
        )
        norm = writer = 0.0
        used = []
        if update:
            require(
                not generated and batch.objective is None,
                "no trajectory surrogate under fixed choices",
            )
            opt.zero_grad(set_to_none=True)
            batch.mean_nll.backward()
            back = clock()
            used = [k for k, v in params.items() if v.grad is not None]
            require(not set(used) & set(PURE), "forced route received score gradients")
            writer = params["writer.weight"].grad.norm().item()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
            opt.step()
        else:
            back = mid
        end = clock()
        cpu1, kid1, rss = process_usage(pids)
        after = digest_state(model)
        writer_after = tensor_sha(params["writer.weight"])
        require(
            (writer_before != writer_after) == update
            and (before != after) == update
            and (not update or writer > 0)
            and pure0 == {k: tensor_sha(params[k]) for k in PURE},
            "unexpected policy mutation or no writer path",
        )
        flat = sum(batch.decisions, [])
        names = BindingNames(ss)
        vectors = torch.stack([d.log_probs.detach() for d in flat])
        joint = torch.stack([d.log_policy.detach() for d in flat])
        a = activity(batch.decisions)
        del a["lookups"], a["outcomes"], a["recall_lags"]
        native = []
        for d in flat:
            native.append(
                digest_rows(
                    [
                        (names.origin(e.reader), tuple(map(names.key, e.keys)))
                        for e in d.trace
                        if e.inst.op == "LOOKUP" and e.keys
                    ]
                )
            )
        desc = describe(tape)
        report = dict(
            pass_name=label,
            mode="frozen-selector sample" if generated else "checked forced choices",
            targets=4096,
            optimizer_steps=int(update),
            nll=batch.mean_nll.item(),
            nll_sha=tensor_sha(batch.mean_nll),
            vectors=tensor_sha(vectors),
            joint=tensor_sha(joint),
            before=before,
            after=after,
            per_document_events=[digest_rows(events([doc], names)) for doc in batch.decisions],
            dispatch_sha=digest_rows(
                [(names.origin(o), cap, tuple(map(names.key, ks))) for o, cap, ks in dispatch]
            ),
            dispatches=len(dispatch),
            activity=a,
            tape=desc,
            issuer_sha=digest_rows(native),
            pathwise_writer_grad=writer,
            writer_before=writer_before,
            writer_after=writer_after,
            used=used,
            clip_norm=norm,
            pure_before_after=pure0,
            provider=provider,
            archive=record.stats,
            append=append,
            setup_ns=ts - beg,
            forward_ns=mid - ts,
            backward_checks_ns=back - mid,
            update_ns=end - back,
            wall_ns=end - ts,
            cpu_core_s=cpu1 - cpu0 + kid1 - kid0,
            rss_kib=rss[0],
            child_rss_kib=sum(rss[1:]),
        )
    report["lifetime_ns"] = clock() - beg
    dump(root / (label + ".json"), report)
    print(
        label,
        desc["steps"],
        "instructions",
        round(report["wall_ns"] / 1e9, 3),
        "s",
        flush=True,
    )
    return tape, vectors, native, report


def main(root):
    no_card()
    os.umask(0o077)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    require(
        head == (root / "gated-ref").read_text().strip()
        and re.fullmatch(
            r"=+ 108 passed in [\d.]+s =+", (root / "qualification.txt").read_text().splitlines()[-1]
        )
        and not (root / "identity.json").exists(),
        "unfrozen/reused workspace",
    )
    subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "ndm/memory_program",
            str(HERE),
            "tests/test_memory_program*.py",
        ],
        check=True,
    )
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.manual_seed(SEED)
    beg = clock()
    src = HERE / "receipts/memory-program-t4/attempt1"
    prod = load(src / "identity.json")
    meta = load(src / "retained-packet.json")
    require(sha(CK) == CK_SHA, "wrong policy")
    ck = torch.load(CK, map_location="cpu", weights_only=True)
    p = PackedMemoryPolicy(256).train()
    p.load_state_dict(ck["model"], strict=True)
    init = digest_state(p)
    require(init == prod["policy_sha"] == ck["policy_sha"], "wrong weights")
    ds = training_data(32)
    rows, ws, man, data = ds
    b, spec = binding("retained", rows, ws, data, prod)
    require(
        meta["model_spec"] == spec
        and prod["data"] == dict(data, window129_sha=[hashlib.sha256(w).hexdigest() for w in ws]),
        "wrong TRAIN/host",
    )
    packet = FrozenPacket.load(
        Path("output/memory-program-t4/attempt1/retained.f32"),
        b,
        meta["sha256"],
        byte_limit=CAP,
    )
    require(
        man["mount"] == prod["mount"] and man["exe_sha"] == prod["branch_exe_sha"],
        "wrong authorized branch mount",
    )
    context = dict(
        encoder=b.encoder_sha,
        source=b.source_sha,
        features=meta["payload_sha"],
        mount=man["mount"],
        scope="whole",
        live_exe=prod["position_exe_sha"],
        branch_exe=prod["branch_exe_sha"],
        limits=asdict(Limits(pattern=16)),
        seed=SEED,
        code=head,
    )
    auth = canonical(dict(context, selector=init))
    dump(
        root / "identity.json",
        dict(
            ref=head,
            context=context,
            authority=auth,
            initial=init,
            threads=torch.get_num_threads(),
            strict=torch.are_deterministic_algorithms_enabled(),
            torch=str(torch.__version__),
            cuda_available=torch.cuda.is_available(),
            visible=os.environ["CUDA_VISIBLE_DEVICES"],
            pid=os.getpid(),
            prior_packet=meta["sha256"],
            max_steps=MAX,
        ),
    )
    tape, start, before, cap = play(root, "capture", copy.deepcopy(p).requires_grad_(False), ds, packet, auth)
    full = copy.deepcopy(p)
    fixed = copy.deepcopy(p)

    def prog(phase, k):
        dump(
            root / "progress.json",
            dict(pass_name="full-update", phase=phase, targets=k, pid=os.getpid()),
        )

    live = step(root, "full-update", full, packet.encoder, ds, prog, coupled=False)
    old = load(HERE / "receipts/memory-program-t5/retained-strict-A/step.json")
    require(live["checks"] == old["checks"], "on-policy strict control differs from T5")
    require(
        cap["vectors"] == live["checks"]["byte_vectors"]
        and cap["joint"] == live["checks"]["joint_scores"]
        and cap["per_document_events"] == live["checks"]["per_document_events"]
        and cap["dispatch_sha"] == live["checks"]["dispatch"],
        "frozen selector did not reproduce the initial live computation",
    )
    _, initial, fixreq, fx = play(root, "fixed-update", fixed, ds, packet, auth, tape=tape, update=True)
    c = live["checks"]
    require(
        torch.equal(start, initial)
        and fixreq == before
        and cap["vectors"] == c["byte_vectors"]
        and cap["joint"] == c["joint_scores"]
        and fx["joint"] == c["joint_scores"]
        and fx["nll_sha"] == cap["nll_sha"] == c["nll"]
        and fx["per_document_events"] == cap["per_document_events"] == c["per_document_events"]
        and cap["dispatch_sha"] == fx["dispatch_sha"] == c["dispatch"],
        "unmatched initial computation",
    )
    del initial
    newauth = canonical(dict(context, selector=live["checks"]["after"]))
    changed, new, sampled, post = play(
        root, "updated-full-live", full.requires_grad_(False), ds, packet, newauth
    )
    _, fixed_out, fxreq, afterfix = play(root, "updated-fixed", fixed, ds, packet, auth, tape=tape)
    _, counter, ctreq, afterfull = play(root, "updated-full-on-old", full, ds, packet, auth, tape=tape)
    require(
        fxreq == ctreq == before
        and all(
            u["per_document_events"] == cap["per_document_events"]
            and u["dispatch_sha"] == c["dispatch"]
            and u["tape"] == cap["tape"]
            for u in (afterfix, afterfull)
        ),
        "frozen route drift",
    )
    per = []
    j = 0
    for aa, bb in zip(tape.programs, changed.programs, strict=True):
        row = Counter()
        for a, z in zip(aa, bb, strict=True):
            ia = [s.instruction for s in a]
            iz = [s.instruction for s in z]
            row["positions"] += 1
            row["opcode_sequence_changed"] += int([v.op for v in ia] != [v.op for v in iz])
            row["instructions_or_operands_changed"] += int(ia != iz)
            row["request_sequence_changed"] += int(before[j] != sampled[j])
            row["structural_evidence_changed"] += int(a != z)
            row["full_live_vs_old_route_argmax_changed"] += int(new[j].argmax() != counter[j].argmax())
            row["full_live_vs_old_route_distribution_changed"] += int(not torch.equal(new[j], counter[j]))
            row["fixed_route_value_distribution_changed"] += int(not torch.equal(start[j], fixed_out[j]))
            j += 1
        per.append(row)
    require(j == 4096, "missing positions")
    no_card()
    dump(
        root / "complete.json",
        dict(
            target_computations=6 * j,
            distinct_target_positions=j,
            optimizer_steps=2,
            per_doc=per,
            full_live_vs_old_route_max_logp_error=(new - counter).abs().max().item(),
            fixed_value_max_logp_change=(fixed_out - start).abs().max().item(),
            wall_ns=clock() - beg,
            cuda_initialized=torch.cuda.is_initialized(),
            card_s=0,
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    main(ap.parse_args().root)
