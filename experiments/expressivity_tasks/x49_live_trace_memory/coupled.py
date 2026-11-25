"""X49 host/head coupling and CPU qualification; no corpus loader, lease or launch entry point."""

from __future__ import annotations

import json
import os

import torch
from torch.nn import functional as F
from ndm.memory_program.direct import DirectTrust
from ndm.memory_program.host_vjp import HostVJP
from ..x45_matrix_emender.muon_control import split_parameters_gdn2
from ..x48_muon_continuation.data import ROOT, check, sha
from ..x48_muon_continuation.learning import hybrid, factor
from .panels import SEEDS, T
from .values import LiveValueHead
from .trace import TraceHead, objective as trace_objective

ORIGIN = "cf7337b520958056a648e4f78a939ded3ed5a4d1bc8f871d8a900d64464b1357"
PEAK = (1e-4, 6e-4)


def head_groups(model):
    # Muon for hidden/value/logit maps; byte embedding, expert votes and biases on AdamW.
    mu = {"direct.hidden.weight", "writer.weight", "mix.weight", "delta.weight"}
    aux = {
        "byte.weight",
        "direct.vote.weight",
        "direct.hidden.bias",
        "direct.vote.bias",
        "writer.bias",
        "mix.bias",
        "delta.bias",
    }
    frozen = set()
    if isinstance(model, TraceHead):
        mu = {"literal." + n for n in mu} | {
            "value.weight",
            "mix.weight",
            "delta.weight",
        }
        aux = {"literal." + n for n in aux} | {"value.bias", "mix.bias", "delta.bias"}
        policy = {"ask.weight", "ask.bias", "filing.weight", "filing.bias"}
        if model.learned:
            aux |= policy
        else:
            frozen = policy
    params = dict(model.named_parameters())
    check(
        set(params) == mu | aux | frozen and not mu & aux,
        "unclassified X49 head tensor",
    )
    check(
        len({p.untyped_storage().data_ptr() for p in params.values()}) == len(params)
        and all(
            p.requires_grad == (n not in frozen)
            and p.dtype == torch.float32
            and (n not in mu or p.ndim == 2)
            for n, p in params.items()
        ),
        "aliased/frozen or non-FP32 head",
    )
    names = [[n for n in params if n in group] for group in (mu, aux)]
    return ([params[n] for n in names[0]], [params[n] for n in names[1]], *names)


def cpu_origin(seed, arm="B"):
    """Instantiate the real host on CPU and load the measured endpoint, not the older pre-X48 anchor."""
    from ..x42_pile_scale import lattice
    from ..x42_pile_scale.byte_velocity import model_for
    from ..x47_xsa_attention.host_vjp_bench import Packet, CHUNK_SHA
    from ..x47_xsa_attention.large_host_bench import identity, EXTERNAL

    check(seed in SEEDS and arm in ("A", "B", "F", "C"), "unregistered seed or arm")
    ident = identity()
    check(
        sha(EXTERNAL / "lit_gpt/gdn2_ops/chunk_gdn2.py") == CHUNK_SHA,
        "different chunk kernel",
    )
    folder = ROOT / "output/x48-muon/direct/main"
    latest = json.loads((folder / "latest.json").read_bytes())
    blob = folder / "step-1024.pt"
    check(
        latest["step"] == 1024
        and latest["file"] == blob.name
        and latest["sha256"] == ORIGIN
        and blob.stat().st_size == latest["bytes"]
        and sha(blob) == ORIGIN,
        "wrong or incomplete X48 endpoint",
    )
    ck = torch.load(blob, map_location="cpu", weights_only=True, mmap=True)
    b = ck["binding"]
    check(
        ck["step"] == 1024
        and b == latest["binding"]
        and b["arm"] == "direct"
        and b["candidate"] == "H"
        and b["host"] == ident["checkpoint"]
        and b["chunk"] == CHUNK_SHA,
        "foreign source lineage",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3401)
        base, policy, _ = lattice.build_policy("lattice_gdn2")
        host = Packet(model_for(base, policy).model)
        host.load_state_dict(ck["host"], strict=True)
        head = DirectTrust(256)
        old = ck["controller"]
        check(all(n.startswith("direct.") for n in old), "foreign controller")
        head.load_state_dict(
            {n.removeprefix("direct."): p for n, p in old.items()}, strict=True
        )
        torch.manual_seed(seed)
        head = LiveValueHead(head)
        if arm in ("F", "C"):
            head = TraceHead(head, learned=arm == "C")
    check(
        sum(p.numel() for p in host.parameters()) == 149855670
        and all(p.device.type == "cpu" for p in host.parameters()),
        "wrong CPU host",
    )
    return host.train(), head.train()


def optimizers(host, head):
    a, ar = hybrid(split_parameters_gdn2(host.net, gates_on_adamw=True), PEAK[0])
    b, br = hybrid(head_groups(head), PEAK[1])
    return (a, b), dict(host=ar, head=br)


def step(host, head, opts, word, *, mode, index, width=256, seed=None, document=None):
    """One external shifted window; no chunk detaches, no host step per instruction, no X48 work price."""
    check(
        type(word) is bytes
        and 2 <= len(word) <= T + 1
        and mode in ("A", "B", "F", "C")
        and isinstance(head, TraceHead) == (mode in ("F", "C"))
        and (mode not in ("F", "C") or head.learned == (mode == "C"))
        and type(index) is int
        and 1 <= index <= 1024,
        "invalid X49 update",
    )
    check(
        torch.are_deterministic_algorithms_enabled()
        and not torch.is_deterministic_algorithms_warn_only_enabled(),
        "strict deterministic learner required",
    )
    check(
        all(not p.is_cuda for p in head.parameters()), "reference head is CPU resident"
    )
    if next(host.parameters()).is_cuda:
        from ..x47_xsa_attention.workbench_run import check_gpu_owner

        check_gpu_owner()
    for opt, peak in zip(opts, PEAK, strict=True):
        opt.zero_grad(set_to_none=True)
        for g in opt.param_groups:
            g["lr"] = peak * factor(index)
    tx = HostVJP(host, width=width, microbatch=1, remat=False)
    inputs = word[:-1]
    h, logits = tx.stage(torch.tensor([list(inputs)], dtype=torch.long))
    work = dict(
        private_queries=0, private_writes=0, private_hits=0, private_collisions=0
    )
    if mode in ("F", "C"):
        roll = head(
            inputs,
            h[0],
            logits[0],
            seed=seed,
            document=document,
            visit=index,
            horizon=4096,
            work=work,
        )
        loss, nll = trace_objective(
            roll, torch.tensor(list(word[1:]), dtype=torch.long), learned=head.learned
        )
    else:
        lp = head(inputs, h[0], logits[0], mode=mode, horizon=4096, work=work)
        # Target-specific gather happens only after the head has returned every byte probability.
        loss = nll = F.nll_loss(lp, torch.tensor(list(word[1:]), dtype=torch.long))
    tx.backward(loss)
    norm = torch.nn.utils.clip_grad_norm_(
        [*host.parameters(), *head.parameters()], 1.0, error_if_nonfinite=True
    )
    for opt in opts:
        opt.step()
    check(
        tx.state == "done"
        and tx.work["forward_batches"] == 1
        and tx.work["forward_tokens"] == len(inputs),
        "wrong host exposure",
    )
    return dict(
        step=index,
        targets=len(inputs),
        preupdate_nll=float(nll.detach()),
        preclip=float(norm),
        work=work,
        host_lr=opts[0].param_groups[0]["lr"],
        head_lr=opts[1].param_groups[0]["lr"],
    )


def audit_origin():
    check(
        os.getenv("CUDA_VISIBLE_DEVICES") == "" and not os.getenv("GPU_LEASE_HELD"),
        "CPU origin audit only",
    )
    host, head = cpu_origin(SEEDS[0])
    opts, r = optimizers(host, head)
    check(
        len(r["host"]["muon"]) == 191
        and len(r["host"]["adamw"]) == 406
        and all(not o.state for o in opts),
        "wrong routing or non-fresh moments",
    )
    return dict(
        origin=ORIGIN,
        host_parameters=sum(p.numel() for p in host.parameters()),
        head_parameters=sum(p.numel() for p in head.parameters()),
        routing={k: {f: len(v) for f, v in g.items()} for k, g in r.items()},
        optimizer_state_entries=sum(len(o.state) for o in opts),
        cuda_initialized=torch.cuda.is_initialized(),
        natural_targets=0,
    )


if __name__ == "__main__":
    print(json.dumps(audit_origin(), sort_keys=True))
