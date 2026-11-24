"""Leased, single-arm continuation. A candidate fork owns its host, controller and both optimizers."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
import resource
import shutil
from statistics import fmean
import subprocess
from time import monotonic, time

import torch
from ndm.memory_program import PackedMemoryPolicy
from ndm.memory_program.direct_entry import DirectEntryProgram
from ..x45_matrix_emender.matched_arms import build_gdn2
from ..x47_xsa_attention.direct_service_head import load_head
from ..x47_xsa_attention.host_vjp_bench import CHUNK_SHA, Packet
from ..x47_xsa_attention.large_host_bench import DIR, EXTERNAL, identity as host_identity
from ..x47_xsa_attention.serving_bench import CK, CK_SHA
from ..x47_xsa_attention.workbench_run import check_gpu_owner
from ..x47_xsa_attention.workbench_witness import digest_state
from .data import B, T, STEPS, OUT, ROOT, X47, Stream, check, load, sha, write_json
from .learning import BRACKET, OnlyDirect, conditional, direct, optimizers, resources, updated

P2 = "64e9d9341019baeabe639331640f331bd2dfc52232f048f81aac0396033b33e0"
TRUST = "67d2bac4af62f535aa2adaca7a025677440e2fdd115f1c5289d4155b31e21813"
ENTRY = "fb3a6b531e1f8a97105bb9c6fe5d8aa7066f2f1dd5e7d7263768ca9ce07efe31"
STREAM_ID = "c40b0eacaa2af2f87ac14217854bc5399dc1dfd82b31b1cacb514af8e8fe1596"
MAN = X47 / "receipts/w3/train-pages.json"
MAN_SHA = "41d3b8e514fa9d2889bfff304f9a731c23b053f3090ed38d2a7fb77324c89f7a"


def controller(arm):
    h, meta = load_head(ROOT / "output/memory-program-s14")
    h.train().requires_grad_(True)
    check(
        meta == load(X47 / "receipts/memory-program-s14/head.json") and digest_state(h) == TRUST,
        "different warm direct parameters",
    )
    if arm == "direct":
        return OnlyDirect(h)
    check(arm == "program" and sha(CK) == CK_SHA, "different warm program")
    ck = torch.load(CK, map_location="cpu", weights_only=True)
    p = PackedMemoryPolicy(256)
    p.load_state_dict(ck["model"], strict=True)
    q = DirectEntryProgram(p, h).train()
    check(
        ck["step"] == 1 and digest_state(p) == ck["policy_sha"] == P2 and digest_state(q.entry) == ENTRY,
        "wrong actor origin",
    )
    return q


def cpu_seconds():
    return [
        u.ru_utime + u.ru_stime
        for u in (resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN))
    ]


def flush_dir(path):
    fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save(folder, binding, step, host, ctrl, opts):
    """Write the blob first, then publish one durable cursor; retain its predecessor until publication."""
    old = load(folder / "latest.json") if (folder / "latest.json").exists() else None
    check(
        old is None or old["binding"] == binding and step > old["step"], "checkpoint cursor did not advance"
    )
    path = folder / f"step-{step:04}.pt"
    temp = path.with_suffix(".pt.tmp")
    with temp.open("xb") as f:
        os.fchmod(f.fileno(), 0o600)
        torch.save(
            dict(
                binding=binding,
                step=step,
                host=host.state_dict(),
                controller=ctrl.state_dict(),
                host_opt=opts[0].state_dict(),
                ctrl_opt=opts[1].state_dict(),
            ),
            f,
        )
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)
    flush_dir(folder)
    note = dict(file=path.name, step=step, sha256=sha(path), binding=binding, bytes=path.stat().st_size)
    write_json(folder / "latest.json", note)
    flush_dir(folder)
    if old is not None:
        (folder / old["file"]).unlink()
    return note


def restore(folder, binding, expected, host, ctrl, opts):
    desc = load(folder / "latest.json")
    check(
        desc["binding"] == binding and desc["step"] == expected and desc["file"] == f"step-{expected:04}.pt",
        "foreign checkpoint or cursor",
    )
    p = folder / desc["file"]
    check(sha(p) == desc["sha256"] and p.stat().st_size == desc["bytes"], "incomplete checkpoint")
    ck = torch.load(p, map_location="cpu", mmap=True, weights_only=True)
    check(ck["binding"] == binding and ck["step"] == expected, "unbound checkpoint contents")
    host.load_state_dict(ck["host"], strict=True)
    ctrl.load_state_dict(ck["controller"], strict=True)
    opts[0].load_state_dict(ck["host_opt"])
    opts[1].load_state_dict(ck["ctrl_opt"])
    return desc["sha256"]


def metrics(folder, first, last):
    m = list(map(json.loads, (folder / "steps.jsonl").read_text().splitlines()))
    check(
        [r["step"] for r in m] == list(range(first, last + 1)) and all(r["targets"] == B * T for r in m),
        "incomplete step trajectory",
    )
    return m


def elect(candidates):
    check(set(candidates) == {"L", "H"}, "incomplete LR bracket")
    scores = {k: fmean(m["preupdate_nll"] for m in rows[BRACKET // 2 :]) for k, rows in candidates.items()}
    return min(scores, key=lambda k: (scores[k], k != "L")), scores


def segment(root, arm, candidate, main, stream, manifest, common):
    t0 = monotonic()
    c0 = cpu_seconds()
    parent = root / arm
    folder = parent / ("main" if main else candidate)
    folder.mkdir(mode=0o700)
    check(shutil.disk_usage(root).free > 16 << 30, "need 16 GiB free for atomic checkpoints")
    outer, _, _ = build_gdn2(3401)
    # The historical builder forces a single CPU thread during construction.
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    check(
        torch.are_deterministic_algorithms_enabled()
        and not torch.is_deterministic_algorithms_warn_only_enabled(),
        "strict learner required",
    )
    outer.load_state_dict(
        torch.load(DIR / "final.pt", map_location="cpu", mmap=True, weights_only=True), strict=True
    )
    host = Packet(outer.model).train()
    ctrl = controller(arm)
    opts, routing = optimizers(host, ctrl, candidate)
    check(
        len(list(host.parameters())) == 597
        and sum(p.numel() for p in host.parameters()) == 149855670
        and sum(1 for _ in ctrl.parameters()) == (4 if arm == "direct" else 41),
        "different architecture",
    )
    binding = dict(**common, arm=arm, candidate=candidate, routing=routing)
    first, last = (BRACKET + 1, STEPS) if main else (1, BRACKET)
    prior = restore(parent / candidate, binding, BRACKET, host, ctrl, opts) if main else None
    write_json(folder / "origin.json", dict(binding=binding, first=first, last=last, from_checkpoint=prior))
    torch.cuda.reset_peak_memory_stats()
    with (folder / "steps.jsonl").open("x", buffering=1) as log:
        for step in range(first, last + 1):
            start = monotonic()
            rows, words = stream.batch(step)
            meter = {}

            def forward(encode):
                with resources(manifest, arm, ctrl, rows, words, meter) as (bank, ss):
                    if arm == "direct":
                        return direct(ctrl, encode, bank, rows, words)
                    return conditional(ctrl, encode, bank, rows, words, ss)

            result = updated(host, ctrl, opts, candidate, step, forward)
            check(
                result["targets"] == B * T and all(v["muon"] > 0 for v in result["update_rms"].values()),
                "missing coupled update",
            )
            result.update(
                candidate=candidate,
                arm=arm,
                native=meter,
                step_wall_s=monotonic() - start,
                batch_sha=sha256("".join(r["window_sha"] for r in rows).encode()).hexdigest(),
            )
            log.write(json.dumps(result, allow_nan=False) + "\n")
            write_json(
                parent / "progress.json",
                dict(
                    pid=os.getpid(),
                    phase=folder.name,
                    step=step,
                    physical_updates=step + (BRACKET if main or candidate == "H" else 0),
                    time=time(),
                ),
            )
            if step in (1, last) or step % 16 == 0:
                os.fsync(log.fileno())
                note = save(folder, binding, step, host, ctrl, opts)
                print(
                    f"{arm}/{folder.name} step {step} nll={result['preupdate_nll']:.4f} checkpoint={note['sha256']}",
                    flush=True,
                )
    m = metrics(folder, first, last)
    c1 = cpu_seconds()
    write_json(
        folder / "complete.json",
        dict(
            optimizer_steps=len(m),
            targets=len(m) * B * T,
            raw_mean_nll=fmean(r["preupdate_nll"] for r in m),
            main_core_s=c1[0] - c0[0],
            children_core_s=c1[1] - c0[1],
            wall_s=monotonic() - t0,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            final_checkpoint=note["sha256"],
        ),
    )
    return m


def run(arm):
    from .lease import authorize

    root = OUT
    auth = authorize(root)
    owner = check_gpu_owner()
    check(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        and os.environ["OMP_NUM_THREADS"] == "4"
        and all(
            os.environ[k] == "expandable_segments:True"
            for k in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
        ),
        "unregistered execution environment",
    )
    check(
        int(owner["pid"]) == int((root / arm / "owner-pid").read_text()), "trainer not owned by arm launcher"
    )
    os.umask(0o077)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    check(
        torch.cuda.memory._snapshot()["allocator_settings"]["expandable_segments"] is True,
        "allocator not active",
    )
    torch.cuda.set_per_process_memory_fraction(0.80)
    host = host_identity()
    check(
        sha(EXTERNAL / "lit_gpt/gdn2_ops/chunk_gdn2.py") == CHUNK_SHA and sha(MAN) == MAN_SHA,
        "unbound live host kernel or archive",
    )
    manifest = load(MAN)
    stream = Stream(root, STREAM_ID)
    ref = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    common = dict(
        schema=1,
        ref=ref,
        stream=STREAM_ID,
        host=host["checkpoint"],
        chunk=CHUNK_SHA,
        trust=TRUST,
        policy=P2 if arm == "program" else None,
        entry=ENTRY if arm == "program" else None,
    )
    write_json(
        root / arm / "identity.json",
        dict(binding=common, owner=owner, auth=auth, pid=os.getpid(), torch=str(torch.__version__)),
    )
    candidates = {}
    for candidate in ("L", "H"):
        candidates[candidate] = segment(root, arm, candidate, False, stream, manifest, common)
        torch.cuda.empty_cache()
    pick, scores = elect(candidates)
    write_json(root / arm / "election.json", dict(winner=pick, train_scores=scores))
    segment(root, arm, pick, True, stream, manifest, common)
    write_json(
        root / arm / "complete.json",
        dict(
            winner=pick,
            selected_targets=STEPS * B * T,
            repeated_targets=BRACKET * B * T,
            physical_updates=STEPS + BRACKET,
            final_checkpoint=load(root / arm / "main/latest.json")["sha256"],
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("arm", choices=("direct", "program"))
    run(p.parse_args().arm)
