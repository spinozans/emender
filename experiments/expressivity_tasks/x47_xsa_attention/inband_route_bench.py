"""T8: sample once in the training graph, then price checked value-learning and interventions."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import os
from pathlib import Path
import re
import subprocess
from time import perf_counter_ns as clock

import numpy as np
import torch
from ndm.memory_program import Limits, PackedMemoryPolicy
from ndm.memory_program.frozen_packet import FrozenPacket
from ndm.memory_program.frozen_routes import record_training_routes, replay_routes
from ndm.memory_program.serving import StreamUniforms
from .frozen_packet_bench import CAP, OPT, SEED, canonical, no_card, step, tensor_sha
from .frozen_route_bench import MAX, PURE, describe
from .frozen_route_verify import work
from .inband_route_check import Witness, delta, scores, sensitivity, submitted
from .memory_program_profile import activity, mount_batch, training_data
from .packed_program_profile import BindingNames, digest_rows, events
from .serving_bench import CK, CK_SHA, process_usage
from .serving_verify import HERE, load, require
from .stream_packet_bench import binding
from .workbench_run import dump, sha
from .workbench_witness import digest_state

PRIOR = HERE / "receipts/memory-program-t7"
SCORES = ("byte_vectors", "joint_scores", "nll")
LIVE = "source-step"


class Journal:
    def __init__(self, root):
        self.root = root
        self.passes = {}
        self.label = None
        self.start = clock()

    def begin(self, label):
        self.label = label
        self.passes[label] = dict(targets=0, updates=0)
        self.progress("binding", 0)

    def progress(self, phase, k):
        state = self.passes[self.label]
        state["targets"] = k
        if phase == "updated":
            state["updates"] = 1
        dump(
            self.root / "progress.json",
            dict(pid=os.getpid(), label=self.label, phase=phase, passes=self.passes),
        )

    def finish(self, good, error=None):
        dump(
            self.root / ("complete.json" if good else "stop.json"),
            dict(
                qualified=good,
                pid=os.getpid(),
                passes=self.passes,
                scored=sum(v["targets"] for v in self.passes.values()),
                updates=sum(v["updates"] for v in self.passes.values()),
                elapsed_ns=clock() - self.start,
                error=error,
                cuda_initialized=torch.cuda.is_initialized(),
                card_s=0,
            ),
        )


def inputs(root):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    require(
        head == (root / "gated-ref").read_text().strip()
        and re.fullmatch(
            r"=+ 110 passed in [\d.]+s =+", (root / "qualification.txt").read_text().splitlines()[-1]
        )
        and not (root / "identity.json").exists(),
        "wrong code/gate or reused run",
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
    prod = load(HERE / "receipts/memory-program-t4/attempt1/identity.json")
    meta = load(HERE / "receipts/memory-program-t4/attempt1/retained-packet.json")
    old = load(PRIOR / "identity.json")
    require(
        sha(CK) == CK_SHA and old["strict"] and old["prior_packet"] == meta["sha256"],
        "unbound retained inputs",
    )
    ck = torch.load(CK, map_location="cpu", weights_only=True)
    p = PackedMemoryPolicy(256).train()
    p.load_state_dict(ck["model"])
    init = digest_state(p)
    require(init == ck["policy_sha"] == prod["policy_sha"] == old["initial"], "wrong initial selector")
    ds = training_data(32)
    rows, ws, man, data = ds
    b, spec = binding("retained", rows, ws, data, prod)
    require(
        meta["model_spec"] == spec
        and prod["data"] == dict(data, window129_sha=[sha256(w).hexdigest() for w in ws])
        and man["mount"] == prod["mount"]
        and man["exe_sha"] == prod["branch_exe_sha"],
        "different source",
    )
    packet = FrozenPacket.load(
        Path("output/memory-program-t4/attempt1/retained.f32"), b, meta["sha256"], byte_limit=CAP
    )
    ctx = dict(
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
    require(dict(ctx, code=old["ref"]) == old["context"], "different scope/RNG/encoder")
    dump(
        root / "identity.json",
        dict(
            ref=head,
            context=ctx,
            initial=init,
            policy_file=CK_SHA,
            packet_file=meta["sha256"],
            threads=torch.get_num_threads(),
            strict=torch.are_deterministic_algorithms_enabled(),
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            torch=str(torch.__version__),
            visible=os.environ["CUDA_VISIBLE_DEVICES"],
            cuda_available=torch.cuda.is_available(),
            pid=os.getpid(),
        ),
    )
    return p, packet, ds, ctx


def independent_oracle():
    """Historical scores/gradients for *external diagnosis only*, never exposed to any learning actor."""
    ref = load(PRIOR / "full-update.json")
    root = HERE / "receipts/memory-program-t6/cpu/retained-serial"
    require(
        ref["checks"]
        == load(HERE / "receipts/memory-program-t5/retained-strict-A/step.json")["checks"]
        == load(root / "step.json")["checks"],
        "wrong prior strict reference",
    )
    meta = load(root / "identity.json")["tensors"]
    tensors = {}
    with np.load("output/memory-program-t6/cpu/retained-serial/private.npz", allow_pickle=False) as z:
        require(set(z.files) == set(meta), "partial diagnostic reference")
        for name, m in meta.items():
            v = torch.from_numpy(z[name])
            require(
                v.dtype == torch.float32 and list(v.shape) == m["shape"] and tensor_sha(v) == m["sha"],
                "changed private oracle",
            )
            tensors[name] = v
    require(
        all(tensor_sha(tensors["score:" + k]) == ref["checks"][k] for k in SCORES),
        "unmatched reference scores",
    )
    return ref, tensors


def source_step(j, model, ds, packet, authority, ref, oracle):
    j.begin(LIVE)
    recorded = {}
    snapshot = {}

    def record(p, enc, ss, ws, *, mode, **kwargs):
        require(mode == "sample" and not recorded, "not a single sampled rollout")
        try:
            tape, batch = record_training_routes(p, enc, ss, ws, authority=authority, max_steps=MAX, **kwargs)
        except BaseException:
            j.progress("rollout-fault", sum(max(0, s.visible - 1) for s in ss))
            raise
        recorded.update(tape=tape, requests=submitted(batch, ss))
        return batch

    def snap(kind, name, value):
        snapshot[kind + ":" + name] = value.detach().clone()

    s = step(
        j.root,
        LIVE,
        model,
        packet.encoder,
        ds,
        j.progress,
        coupled=False,
        rollout=record,
        capture=snap,
        capture_scores=True,
    )
    require(snapshot.keys() == oracle.keys(), "partial sampled score/gradient/state")
    diff = {k: delta(oracle[k], v) for k, v in snapshot.items()}
    c = s["checks"]
    t = recorded["tape"]
    att = dict(
        differences=diff,
        all_t5_t7_checks_match=c == ref["checks"],
        same_work=work(s) == work(ref),
        tape=describe(t),
        authority=t.authority,
        issuer_sha=digest_rows(recorded["requests"]),
    )
    dump(j.root / "source-vs-strict.json", att)
    require(
        not any(v["unequal_bits"] for v in diff.values())
        and att["all_t5_t7_checks_match"]
        and att["same_work"],
        "in-band training step diverged from strict live reference",
    )
    model.zero_grad(set_to_none=True)
    return Witness(
        t,
        {k: snapshot["score:" + k] for k in SCORES},
        recorded["requests"],
        dict(
            s, per_document_events=c["per_document_events"], dispatch_sha=c["dispatch"], authority=t.authority
        ),
    )


def pass_once(j, label, model, ds, packet, ctx, *, force=None, exact=False, update=False):
    j.begin(label)
    beg = clock()
    rows, ws, man, data = ds
    initial = digest_state(model)
    require(
        model.training
        and torch.is_grad_enabled()
        and all(p.requires_grad and p.grad is None for p in model.parameters()),
        "non-training forward or previous gradients",
    )
    sampled = force is None
    require(not update or not sampled, "only the value-learning arm updates here")
    tape = None if sampled else force.tape
    authority = canonical(dict(ctx, selector=initial)) if sampled else tape.authority
    opt = torch.optim.AdamW(model.parameters(), **OPT) if update else None
    params = dict(model.named_parameters())
    protected = {k: tensor_sha(params[k]) for k in (*PURE, "writer.weight")}
    provider = defaultdict(Counter)
    app = Counter()
    physical = []
    with ExitStack() as stack:
        pages, _, archive, ss, lives = mount_batch(
            stack, model, rows, man, provider, app, Limits(pattern=16), j.progress, physical
        )
        enc = packet.encoder(ss)
        pids = [pages.proc.pid, archive.worker.proc.pid, *[client.native.proc.pid for client in lives]]
        cpu0, kids0, _ = process_usage(pids)
        ts = clock()
        try:
            if sampled:
                rng = StreamUniforms(ss, data["train_rows"], 129, SEED, "cpu")
                tape, batch = record_training_routes(
                    model,
                    enc,
                    ss,
                    ws,
                    authority=authority,
                    max_steps=MAX,
                    generator=rng,
                    key_price=0.001,
                    instruction_price=0.0001,
                    baseline=torch.arange(128, 0, -1).float() * 2,
                )
            else:
                batch = replay_routes(model, enc, ss, ws, tape, authority=authority, max_steps=MAX)
        except BaseException:
            j.progress("rollout-fault", sum(max(0, s.visible - 1) for s in ss))
            raise
        fwd = clock()
        require(
            batch.targets == 4096
            and (batch.objective is not None) == sampled
            and batch.mean_nll.requires_grad
            and all(
                bytes(client.data) == w and s.visible == 129 and not s.pending
                for client, s, w in zip(lives, ss, ws, strict=True)
            ),
            "incorrect scoring/reset",
        )
        j.progress("scored", 4096)
        values = scores(batch)
        keys = submitted(batch, ss)
        names = BindingNames(ss)
        act = activity(batch.decisions)
        for k in ("lookups", "outcomes", "recall_lags"):
            del act[k]
        rep = dict(
            cell=label,
            mode="fresh_sample" if sampled else "checked_replay",
            authority=authority,
            tape=describe(tape),
            before=initial,
            targets=batch.targets,
            nll=batch.mean_nll.item(),
            score_sha={k: tensor_sha(v) for k, v in values.items()},
            issuer_sha=digest_rows(keys),
            per_document_events=[digest_rows(events([d], names)) for d in batch.decisions],
            dispatch_sha=digest_rows(
                [(names.origin(o), c, tuple(map(names.key, ks))) for o, c, ks in physical]
            ),
            dispatches=len(physical),
            activity=act,
            provider=provider,
            archive=archive.stats,
            append=app,
        )
        if not sampled:
            prior = force.receipt
            match = dict(
                work=work(rep) == work(prior),
                events=rep["per_document_events"] == prior["per_document_events"],
                dispatch=rep["dispatch_sha"] == prior["dispatch_sha"],
                requests=keys == force.requests,
            )
            diffs = {k: delta(force.scores[k], v) for k, v in values.items()}
            dump(
                j.root / (label + "-gate.json"),
                dict(exact_scores=exact, logical_match=match, scores=diffs, checked_before_step=True),
            )
            require(
                all(match.values()) and (not exact or not any(d["unequal_bits"] for d in diffs.values())),
                "changed frozen work or failed exact score gate before update",
            )
        checked = clock()
        used = []
        writer = norm = 0.0
        if update:
            opt.zero_grad(set_to_none=True)
            batch.mean_nll.backward()
            back = clock()
            used = [k for k, p in params.items() if p.grad is not None]
            require(not set(used).intersection(PURE), "score adjoint in value-only arm")
            writer = params["writer.weight"].grad.norm().item()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
            opt.step()
            j.progress("updated", batch.targets)
        else:
            back = checked
        end = clock()
        cpu1, kids1, rss = process_usage(pids)
        after = digest_state(model)
        new = {k: tensor_sha(params[k]) for k in protected}
        require(
            all(protected[k] == new[k] for k in PURE)
            and (initial != after) == (protected["writer.weight"] != new["writer.weight"]) == update
            and (not update or writer > 0),
            "unexpected weight change/unchanged writer",
        )
        rep.update(
            after=after,
            optimizer_steps=int(update),
            used=used,
            writer_grad=writer,
            clip_norm=norm,
            protected_before=protected,
            protected_after=new,
            setup_ns=ts - beg,
            forward_ns=fwd - ts,
            check_ns=checked - fwd,
            backward_ns=back - checked,
            update_ns=end - back,
            wall_ns=end - ts,
            cpu_core_s=cpu1 - cpu0 + kids1 - kids0,
            rss_kib=rss[0],
            child_rss_kib=sum(rss[1:]),
        )
    model.zero_grad(set_to_none=True)
    rep["lifetime_ns"] = clock() - beg
    dump(j.root / (label + ".json"), rep)
    print(
        label,
        rep["targets"],
        "targets;",
        rep["tape"]["steps"],
        "instructions;",
        round(rep["wall_ns"] / 1e9, 3),
        "s",
        flush=True,
    )
    return Witness(tape, values, keys, rep)


def main(root):
    no_card()
    os.umask(0o077)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.manual_seed(SEED)
    require(
        not any(
            (root / n).exists() for n in ("stop.json", "complete.json", "progress.json", "identity.json")
        ),
        "reused workspace",
    )
    journal = Journal(root)
    try:
        p, packet, ds, ctx = inputs(root)
        full = deepcopy(p)
        fixed = deepcopy(p)
        ref, oracle = independent_oracle()
        authority = canonical(dict(ctx, selector=digest_state(p)))
        initial = source_step(journal, full, ds, packet, authority, ref, oracle)
        del oracle

        def run(label, model, **kwargs):
            return pass_once(journal, label, model, ds, packet, ctx, **kwargs)

        value = run("value-step", fixed, force=initial, update=True, exact=True)
        post = run("updated-full-live", full)
        positive = run("updated-full-positive", full, force=post, exact=True)
        require(
            positive.receipt["before"] == post.receipt["before"] == initial.receipt["checks"]["after"],
            "post-replay weights",
        )
        fixed_read = run("updated-fixed-on-old", fixed, force=initial)
        counter = run("updated-full-on-old", full, force=initial)
        require(
            fixed_read.receipt["before"] == value.receipt["after"]
            and counter.receipt["before"] == post.receipt["before"],
            "changed counterfactual weights",
        )
        d = sensitivity(initial, post, fixed_read, counter)
        dump(root / "sensitivity.json", d)
        require(
            all(x["prefix_max_fraction"] <= 1 for x in d["per_document"]),
            "same-route-history numerical floor failed",
        )
        no_card()
    except BaseException as exc:
        journal.finish(False, f"{type(exc).__name__}: {exc}")
        raise
    journal.finish(True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    main(ap.parse_args().root)
