"""Synchronous online DIRECT/ENTER. No teacher-forcing queue, route mask or target argument."""

from collections import Counter
import os
from threading import get_ident
from time import perf_counter_ns

import torch

from .direct_entry import DirectEntryProgram, EntryStream
from .serving import place_inputs


class EntryReadout:
    """Fixed CPU readout for ByteService(entry=...). Caller owns the bank and private bindings.

    on_direct is an optional trusted post-score diagnostic on the *chosen* direct evidence.
    Program draws use the service's per-admission horizon, not a count of past program entries.
    """

    def __init__(self, model, bank, *, gate_seed, on_direct=None):
        if (
            not isinstance(model, DirectEntryProgram)
            or any(m.training for m in model.modules())
            or any(
                p.requires_grad or p.device.type != "cpu" or p.dtype != torch.float32
                for p in model.parameters()
            )
            or type(gate_seed) is not int
            or not 0 <= gate_seed < 2**64
        ):
            raise ValueError("frozen FP32 CPU direct-entry predictor and bounded seed required")
        self.model, self.bank, self.gate_seed = model, bank, gate_seed
        self.on_direct = on_direct
        self._owner = os.getpid(), get_ident()
        self._version = self._stamp()
        self.work = Counter()

    def _stamp(self):
        m, b = self.model, self.bank
        return (
            self.gate_seed,
            id(b),
            b.source,
            b.version,
            id(getattr(b, "pages", b)),
            tuple((id(x), x.training) for x in m.modules()),
            tuple((id(p), p._version, p.device, p.dtype, p.requires_grad) for p in m.parameters()),
        )

    def check(self):
        if (
            (os.getpid(), get_ident()) != self._owner
            or torch.is_grad_enabled()
            or self._stamp() != self._version
        ):
            raise ValueError("changed inference owner, direct-entry weights or bank")

    def validate_binding(self, s, q):
        self.check()
        if not isinstance(q, EntryStream) or q.session is not s:
            raise ValueError("foreign direct-entry attachment")
        q.check()

    def __call__(self, sessions, streams, h, base, *, row_ids, generator, mode):
        self.check()
        if len(streams) != len(sessions) or any(
            q.session is not s for s, q in zip(sessions, streams, strict=True)
        ):
            raise ValueError("changed direct-entry attachments")
        other = h.device.type != "cpu"
        start = perf_counter_ns()
        h, base = place_inputs(h, base, width=self.model.direct.host_dim, device="cpu")
        if other:
            self.work.update(
                transfer_batches=1,
                transfer_bytes=(h.numel() + base.numel()) * 4,
                blocking_transfer_ns=perf_counter_ns() - start,
            )
        out = self.model(
            streams,
            h,
            base,
            row_ids=row_ids,
            gate_seed=self.gate_seed,
            program_rng=generator,
            mode=mode,
            bank=self.bank,
            on_direct=self.on_direct,
            direct_work=self.work,
        )
        progs = [d for d in out if d.program is not None]
        self.work.update(
            predictions=len(out),
            gate_batches=1,
            enter=len(progs),
            direct=len(out) - len(progs),
            program_batches=int(bool(progs)),
            direct_batches=int(len(progs) != len(out)),
            program_keys=sum(d.keys for d in progs),
            nonemit=sum(d.nonemit for d in progs),
        )
        return out
