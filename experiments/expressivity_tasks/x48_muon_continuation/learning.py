"""One fresh coupled step. No routing teacher, detached host packet or feature cache."""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass
from math import sqrt
from time import perf_counter

import torch
from torch import nn
from ndm.memory_program import Limits, Session
from ndm.memory_program.direct import DocumentCounts, direct_branches, direct_features
from ndm.memory_program.direct_entry import direct_entry_teacher_force
from ndm.memory_program.host_vjp import HostVJP
from ndm.memory_program.m20 import TerminalBranches
from ndm.memory_program.spans import mean_byte_values
from ..x45_matrix_emender.bakeoff_optimizers import HybridMuonX45
from ..x45_matrix_emender.muon_control import split_parameters_gdn2
from ..x47_xsa_attention.pinned_branch import p100_domain, whole_p100
from ..x47_xsa_attention.private_tape import PrivateTape
from ..x47_xsa_attention.scoped_records import ScopedRecords
from ..x47_xsa_attention.terminal_branch_bench import WirePages
from .data import STEPS, check

RATES = {"L": (5e-5, 3e-4), "H": (1e-4, 6e-4)}
BRACKET = 128
KEY_PRICE = 0.001
OP_PRICE = 0.0001


class OnlyDirect(nn.Module):
    def __init__(self, direct):
        super().__init__()
        self.direct = direct


# Semantic allowlist: not all 2D parameters are hidden maps.
MATRIX = {
    "direct.hidden.weight",
    *(
        "program." + s
        for s in (
            "host.weight",
            "meta.weight",
            "budget.weight",
            "encoder.self_attn.in_proj_weight",
            "encoder.self_attn.out_proj.weight",
            "encoder.linear1.weight",
            "encoder.linear2.weight",
            "pointer.weight",
            "query.weight",
            "writer.weight",
            "fuse.weight",
            "pair_query.weight",
            "pair_key.weight",
            "payload_out.weight",
            "binding_out.weight",
        )
    ),
}
AUX2 = {
    "direct.vote.weight",
    "entry.linear.weight",
    "program.op.weight",
    "program.mix.weight",
    "program.transitions.weight",
    "program.prior",
    "program.embed.weight",
}


def controller_groups(model):
    by = dict(model.named_parameters())
    matrix = []
    aux = []
    for n, p in by.items():
        check(p.requires_grad and p.dtype == torch.float32, "inert or non-FP32 controller parameter")
        if n in MATRIX and p.ndim == 2:
            matrix.append(n)
        elif (
            n in AUX2
            and p.ndim == 2
            or p.ndim == 1
            and (n.endswith("bias") or ".norm" in n and n.endswith("weight"))
        ):
            aux.append(n)
        else:
            raise ValueError("unclassified controller parameter: " + n)
    check(
        len(set(p.untyped_storage().data_ptr() for p in by.values())) == len(by), "shared controller storage"
    )
    return [by[n] for n in matrix], [by[n] for n in aux], matrix, aux


def hybrid(split, lr):
    mu, aux, mnames, anames = split
    opt = HybridMuonX45(mu, aux, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    return opt, dict(muon=mnames, adamw=anames)


def optimizers(host, ctrl, candidate):
    h, c = RATES[candidate]
    a, ha = hybrid(split_parameters_gdn2(host.net, gates_on_adamw=True), h)
    b, ca = hybrid(controller_groups(ctrl), c)
    return (a, b), dict(host=ha, controller=ca)


def factor(step):
    check(type(step) is int and 1 <= step <= STEPS, "invalid update index")
    return min(1.0, step / 16, (STEPS + 1 - step) / 128)


@dataclass
class Scored:
    objective: torch.Tensor
    nll: torch.Tensor
    targets: int
    work: dict


def direct(model, encode, bank, rows, words):
    check(
        len(rows) == len(words) > 0 and len({len(w) for w in words}) == 1 and all(len(w) >= 2 for w in words),
        "invalid direct batch",
    )
    h, b = encode(torch.tensor([list(w[:-1]) for w in words], dtype=torch.long))
    costs = Counter()
    losses = []
    with ExitStack() as stack:
        locals_ = [
            stack.enter_context(closing(DocumentCounts(r["doc"], r["start"], byte_limit=len(w))))
            for r, w in zip(rows, words, strict=True)
        ]
        for s, w in zip(locals_, words, strict=True):
            s.observe(w[0], at=0)
        for j in range(len(words[0]) - 1):
            views = [s.view(at=j + 1) for s in locals_]
            branches = direct_branches(bank, views, grouped=True, work=costs)
            logp, _ = model.direct(h[:, j], b[:, j], direct_features(views, branches))
            ys = [w[j + 1] for w in words]  # score first, append second (including last observation)
            losses.append(-logp.gather(1, torch.tensor(ys)[:, None]))
            for s, y in zip(locals_, ys, strict=True):
                s.observe(y, at=j + 1)
    loss = torch.cat(losses)
    nll = loss.mean()
    return Scored(
        nll,
        nll,
        loss.numel(),
        dict(direct=loss.numel(), program=0, direct_keys=costs["logical_probes"], program_keys=0, nonemit=0),
    )


def conditional(model, encode, bank, rows, words, sessions):
    batch = direct_entry_teacher_force(
        model,
        encode,
        sessions,
        words,
        bank=bank,
        row_ids=[r["doc"] for r in rows],
        gate_seed=7823,
        program_seed=7809,
        clock="queue",
        key_price=KEY_PRICE,
        instruction_price=OP_PRICE,
        baseline_per_remaining=2.0,
    )
    ops = Counter(e.inst.op for ds in batch.decisions for d in ds if d.program for e in d.program.trace)
    return Scored(batch.objective, batch.mean_nll, batch.targets, dict(**batch.work, opcodes=dict(ops)))


@contextmanager
def resources(manifest, arm, model, rows, words, meter):
    opened = []

    def pages(**kwargs):
        p = WirePages(**kwargs)
        opened.append(p)
        return p

    lives = []
    rec = None
    conf, version = whole_p100(manifest, "whole")
    with (
        p100_domain(manifest, scope="whole", max_grants=1, pages_type=pages) as (domain, desc),
        domain.borrow("P100", version, scope="whole") as grant,
        ExitStack() as stack,
    ):
        # Direct transport sees only its own literal suffix probes and has its own cold LRU.
        terminal = TerminalBranches("P100", version, opened[0], scope="whole", cache_pages=256)
        ss = []
        if arm == "program":

            def encode(pieces):
                return mean_byte_values(pieces, model.program.embed.weight)

            rec = stack.enter_context(
                ScopedRecords(
                    desc, conf["files"][1], scope="whole", branches=grant, encode=encode, defer_position=True
                )
            )
            shared = [grant.reader(), *rec.readers()]
            for r, w in zip(rows, words, strict=True):
                live = stack.enter_context(
                    closing(PrivateTape(r["doc"], r["start"], encode=encode, byte_limit=len(w), demand=True))
                )
                ss.append(
                    Session(
                        [*shared, *live.readers()],
                        Limits(pattern=16),
                        live.advance,
                        document=r["doc"],
                        start=r["start"],
                        session_id=live.id,
                    )
                )
                lives.append(live)
        else:
            check(arm == "direct", "unknown arm")

        class AdmittedDirect:
            source = terminal.source
            version = terminal.version

            @staticmethod
            def histograms(keys):
                grant.check_descriptor(desc)
                answers = terminal.histograms(keys)
                grant.check_descriptor(desc)
                return answers

        yield AdmittedDirect(), ss
        grant.check_descriptor(desc)
        check(
            all(bytes(live.data) == w for live, w in zip(lives, words, strict=True))
            if arm == "program"
            else not lives,
            "different observed bytes",
        )
    (p,) = opened
    wire = p.receipt()
    priv = Counter()
    for live in lives:
        priv.update(live.stats)
    meter.update(
        wire_bytes=wire["bytes"],
        m20=wire["rpc"],
        positional_birth=int(rec is not None and rec.pid is not None),
        position={} if rec is None else rec.final["work"],
        private=dict(priv),
    )


@torch.no_grad()
def snapshot_groups(opt):
    return [[p.detach().clone() for p in g["params"]] for g in opt.param_groups]


@torch.no_grad()
def delta_rms(opt, before):
    out = {}
    for g, old in zip(opt.param_groups, before, strict=True):
        # One synchronization per parameter group, not per tensor. Includes LR and decay.
        vals = [(p - x).float().square().sum() for p, x in zip(g["params"], old, strict=True)]
        out[g["family"]] = sqrt(torch.stack(vals).sum().item() / sum(x.numel() for x in old))
    return out


def updated(host, ctrl, opts, candidate, step, forward, *, width=256, microbatch=4):
    for opt, peak in zip(opts, RATES[candidate], strict=True):
        opt.zero_grad(set_to_none=True)
        for g in opt.param_groups:
            g["lr"] = peak * factor(step)
    tx = HostVJP(host, width=width, microbatch=microbatch, remat=False)
    ts = perf_counter()
    a = forward(tx.stage)
    scored = perf_counter()
    count = tx.work["forward_tokens"]
    check(
        a.targets == count and a.work.get("direct", 0) + a.work.get("program", 0) == count,
        "unmatched causal positions",
    )
    tx.backward(a.objective)
    grad = perf_counter()
    norm = torch.nn.utils.clip_grad_norm_(
        [*host.parameters(), *ctrl.parameters()], 1.0, error_if_nonfinite=True
    )
    # Measure actual parameter displacement including weight decay on steps 1 and every 16th.
    rms = {}
    for k, opt in zip(("host", "controller"), opts, strict=True):
        before = snapshot_groups(opt) if step == 1 or step % 16 == 0 else None
        opt.step()
        if before is not None:
            rms[k] = delta_rms(opt, before)
        del before
    if next(host.parameters()).is_cuda:
        torch.cuda.synchronize()
    nll = a.nll.detach().item()
    objective = a.objective.detach().item()
    work = a.work
    price = KEY_PRICE * (work.get("direct_keys", 0) + work.get("program_keys", 0)) + OP_PRICE * work.get(
        "nonemit", 0
    )
    return dict(
        step=step,
        targets=count,
        preupdate_nll=nll,
        train_objective=objective,
        priced_nll=nll + price / count,
        forward_s=scored - ts,
        backward_s=grad - scored,
        clip_update_s=perf_counter() - grad,
        preclip=float(norm),
        update_rms=rms,
        host_lr=opts[0].param_groups[0]["lr"],
        controller_lr=opts[1].param_groups[0]["lr"],
        work=work,
    )
