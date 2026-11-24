"""T1: first-order gradient equivalence on scored TRAIN bytes, no optimizer or model update."""

from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import asdict
import gc, hashlib, json, os, subprocess
from pathlib import Path
from time import perf_counter_ns as clock
import torch
from torch import nn
from ndm.memory_program import Limits, PackedMemoryPolicy, teacher_force
from ndm.memory_program.serving import StreamUniforms
from ndm.memory_program.host_vjp import HostVJP
from .large_host_bench import DIR, EXTERNAL, identity, digest
from .memory_program_profile import mount_batch, training_data, activity
from .packed_program_profile import BindingNames, events, digest_rows
from .serving_bench import CK, CK_SHA, process_usage
from .workbench_native import EXE_SHA
from .workbench_run import check_gpu_owner, dump, sha
from .workbench_witness import require, digest_state

SEED = 7809
CHUNK_SHA = "51e2037b204dbd2cc9e2b9c9bf301adcb770bb23d4a825dd23032b222a68a1b7"


class Packet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.saved = Counter()

    def forward(self, ids):
        # Only the retained host scope is instrumented. Logical saved elements may alias weights.
        mode = "tape" if torch.is_grad_enabled() else "stage"

        def pack(t):
            self.saved[mode + "_references"] += 1
            self.saved[mode + "_logical_bytes"] += t.numel() * t.element_size()
            return t.detach()

        with (
            torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            n = self.net
            x = n.embed(ids)
            for block in n.blocks:
                x = block(x, ids)
            h = n.norm(x)
            logits = n.head(h)
            return torch.cat((h[:, :, :256].float(), logits.float()), dim=-1)


def prepare(run):
    lease = check_gpu_owner()
    require(
        os.environ["OMP_NUM_THREADS"] == "4"
        and all(
            os.environ[k] == "expandable_segments:True"
            for k in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
        ),
        "environment",
    )
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    require(torch.cuda.memory._snapshot()["allocator_settings"]["expandable_segments"] is True, "allocator")
    torch.cuda.set_per_process_memory_fraction(0.55)
    host = identity()
    require(
        digest(EXTERNAL / "lit_gpt/gdn2_ops/chunk_gdn2.py") == CHUNK_SHA and sha(CK) == CK_SHA,
        "unbound kernels or policy",
    )
    from ..x45_matrix_emender.matched_arms import build_gdn2

    outer, _, _ = build_gdn2(3401)
    outer.load_state_dict(
        torch.load(DIR / "final.pt", map_location="cpu", mmap=True, weights_only=True), strict=True
    )
    enc = Packet(outer.model).train().requires_grad_(True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    ck = torch.load(CK, map_location="cpu", weights_only=True)
    p = PackedMemoryPolicy(256).train()
    p.load_state_dict(ck["model"], strict=True)
    require(
        sum(t.numel() for t in enc.parameters()) == 149855670
        and all(t.dtype == torch.float32 for t in enc.parameters())
        and ck["step"] == 1
        and ck["cell"] == "packed-b32"
        and digest_state(p) == ck["policy_sha"],
        "unbound weights",
    )
    _, windows, man, data = training_data(32)
    require(
        ck["data"] == dict(**data, window129_sha=[hashlib.sha256(w).hexdigest() for w in windows]),
        "wrong TRAIN windows",
    )
    dump(
        run / "identity.json",
        dict(
            ref=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            owner=lease,
            trainer_pid=os.getpid(),
            host=host,
            chunk_sha=CHUNK_SHA,
            parameters=149855670,
            precision="BF16 autocast, FP32 master",
            checkpoint=CK_SHA,
            policy_sha=ck["policy_sha"],
            torch=str(torch.__version__),
            device=torch.cuda.get_device_name(),
            data=ck["data"],
            mount=man["mount"],
            branch_exe_sha=man["exe_sha"],
            position_exe_sha=EXE_SHA,
            seed=SEED,
            microbatch=4,
            limits=asdict(Limits(pattern=16)),
            policy_device="cpu",
            host_device=str(next(enc.parameters()).device),
        ),
    )
    return enc, p


def cell(run, enc, policy, n, remat, prior):
    start = clock()
    arm = "remat" if remat else "direct"
    name = f"b{n}-{arm}"

    def progress(phase, count):
        dump(
            run / "progress.json",
            dict(cell=name, phase=phase, total_targets=prior + count, trainer_pid=os.getpid()),
        )

    progress("setup", 0)
    selected, windows, manifest, data = training_data(n)
    limits = Limits(pattern=16)
    provider = defaultdict(Counter)
    append = Counter()
    disp = []
    torch.cuda.synchronize()
    purge = clock()
    gc.collect()
    torch.cuda.empty_cache()
    purge = clock() - purge
    enc.zero_grad(set_to_none=True)
    policy.zero_grad(set_to_none=True)
    enc.saved.clear()
    tx = HostVJP(enc, width=256, microbatch=4, remat=remat)
    with ExitStack() as stack:
        pages, _, record, ss, lives = mount_batch(
            stack, policy, selected, manifest, provider, append, limits, progress, disp
        )
        rng = StreamUniforms(ss, data["train_rows"], 129, SEED, "cpu")
        baseline = 2 * torch.arange(128, 0, -1).float()
        pids = [pages.proc.pid, record.worker.proc.pid, *[l.native.proc.pid for l in lives]]
        m = {}
        tm = {}
        cpu0, kid0, _ = process_usage(pids)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        m["initial"] = torch.cuda.memory_allocated()
        ts = clock()
        tm["setup_including_purge_ns"] = ts - start
        tm["purge_ns"] = purge

        def encode(ids):
            t = clock()
            out = tx.stage(ids)
            torch.cuda.synchronize()
            tm["host_first_pass_ns"] = clock() - t
            m["after_host_forward"] = torch.cuda.memory_allocated()
            return out

        batch = teacher_force(
            policy,
            encode,
            ss,
            windows,
            mode="sample",
            generator=rng,
            baseline=baseline,
            key_price=0.001,
            instruction_price=0.0001,
        )
        torch.cuda.synchronize()
        fwd = clock()
        tm["forward_ns"] = fwd - ts
        m["post_rollout"] = torch.cuda.memory_allocated()
        saved_before = dict(enc.saved)
        require(
            batch.targets == n * 128 and all(s.visible == 129 for s in ss) and batch.objective is not None,
            "missing fresh targets",
        )
        tx.backward(batch.objective)
        torch.cuda.synchronize()
        end = clock()
        tm.update(backward_ns=end - fwd, gradient_wall_ns=end - ts)
        m["peak_allocated"] = torch.cuda.max_memory_allocated()
        m["peak_reserved"] = torch.cuda.max_memory_reserved()
        m["post_backward"] = torch.cuda.memory_allocated()
        cpu1, kid1, rss = process_usage(pids)
        progress("backpropagated", batch.targets)
        params = {}
        stats = {}
        for label, model in [("host", enc), ("policy", policy)]:
            nnz = nelts = tensors = nonzero_tensors = 0
            for k, p in model.named_parameters():
                g = p.grad.detach().to("cpu", copy=True) if p.grad is not None else None
                params[label + "/" + k] = g
                if g is None:
                    continue
                require(g.shape == p.shape and g.isfinite().all().item(), f"bad gradient {label}/{k}")
                nelts += g.numel()
                tensors += 1
                nz = g.count_nonzero().item()
                nnz += nz
                nonzero_tensors += int(nz > 0)
            stats[label] = dict(
                used_elements=nelts, nonzero_elements=nnz, tensors=tensors, nonzero_tensors=nonzero_tensors
            )
            require(nnz > 0, "no gradient for " + label)
        tm["snapshot_ns"] = clock() - end
        flat = sum(batch.decisions, [])
        names = BindingNames(ss)
        num = dict(
            byte_logp=torch.stack([d.log_probs.detach() for d in flat]),
            joint_logp=torch.stack([d.log_policy.detach() for d in flat]),
            objective=batch.objective.detach(),
            nll=batch.mean_nll.detach(),
        )
        act = activity(batch.decisions)
        del act["outcomes"], act["lookups"], act["recall_lags"]
        logical = dict(
            event_sha=digest_rows(events(batch.decisions, names)),
            dispatch_sha=digest_rows([(names.origin(o), c, tuple(map(names.key, ks))) for o, c, ks in disp]),
            window_sha=[hashlib.sha256(w).hexdigest() for w in windows],
            train_rows=data["train_rows"],
        )
        rep = dict(
            cell=name,
            targets=batch.targets,
            mode="fresh_sample",
            optimizer_steps=0,
            logical=logical,
            gradients=stats,
            nll=batch.mean_nll.item(),
            objective=batch.objective.item(),
            provider=provider,
            dispatches=len(disp),
            activity=act,
            host_work=tx.work,
            saves_at_rollout=saved_before,
            saves_total=dict(enc.saved),
            memory=m,
            times=tm,
            core_s=cpu1 - cpu0 + kid1 - kid0,
            rss_kib=rss[0],
            children_rss_kib=sum(rss[1:]),
            targets_s=batch.targets * 1e9 / tm["gradient_wall_ns"],
        )
        dump(run / (name + ".json"), rep)
        print(f"{name}: {batch.targets} real gradient targets, {rep['targets_s']:.1f}/s", flush=True)
    policy.zero_grad(set_to_none=True)
    enc.zero_grad(set_to_none=True)
    return dict(logical=logical, numeric=num, grads=params)


def compare(run, n, direct, remat):
    require(direct["logical"] == remat["logical"], "changed logical program/targets")
    err = {}
    for k, x in direct["numeric"].items():
        y = remat["numeric"][k]
        torch.testing.assert_close(x, y, atol=5e-5, rtol=2e-5, msg=lambda m: f"{k}: {m}")
        err[k] = dict(count=x.numel(), max_abs=(x - y).abs().max().item())
    grads = {}
    unused = []
    require(direct["grads"].keys() == remat["grads"].keys(), "parameter schema")
    for k, x in direct["grads"].items():
        y = remat["grads"][k]
        require((x is None) == (y is None), "unused mismatch: " + k)
        if x is None:
            unused.append(k)
            continue
        m = x.abs().max().item()
        g = (x - y).abs().max().item()
        require(g <= 2e-4 + 2e-4 * m, f"{k}: gradient error {g}, reference max {m}")
        grads[k] = dict(n=x.numel(), ref_max=m, error=g)
    dump(
        run / f"b{n}-equivalence.json",
        dict(numeric=err, logical=direct["logical"], unused=unused, gradients=grads),
    )


def main(run):
    enc, p = prepare(run)
    prior = 0
    for n, order in [(4, (False, True)), (32, (True, False))]:
        pair = {}
        for remat in order:
            pair[remat] = cell(run, enc, p, n, remat, prior)
            prior += n * 128
        compare(run, n, pair[False], pair[True])
        del pair
        gc.collect()
    dump(
        run / "progress.json",
        dict(cell="all", phase="complete", total_targets=prior, trainer_pid=os.getpid()),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    main(ap.parse_args().run)
