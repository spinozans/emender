"""T12: model-free physical sources and post-prediction, non-raw evidence. No alternate dataset."""

from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import torch
from ndm.memory_program import Limits, PackedMemoryPolicy, Session, Unavailable
from ndm.memory_program.frozen_packet import FrozenPacket
from ndm.memory_program.frontier_reads import for_model, schema
from .direct_control_bench import T4, add_hash
from .frozen_packet_bench import CAP, tensor_sha
from .memory_program_profile import training_data
from .packed_program_profile import BindingNames, events
from .private_tape import PrivateTape
from .serving_bench import CK, CK_SHA
from .serving_verify import load, require
from .stream_packet_bench import binding
from .workbench_native import EXE_SHA
from .workbench_run import sha
from .workbench_witness import digest_state

ORDER = ("separate-A", "pair-A", "pair-B", "separate-B")
GATES = 196
SEED = 7809


def inputs():
    rows, ws, man, data = training_data(32)
    prod = load(T4 / "identity.json")
    meta = load(T4 / "retained-packet.json")
    b, spec = binding("retained", rows, ws, data, prod)
    require(
        spec == meta["model_spec"]
        and prod["data"] == dict(data, window129_sha=[sha256(w).hexdigest() for w in ws])
        and (man["exe_sha"], man["mount"], EXE_SHA)
        == (prod["branch_exe_sha"], prod["mount"], prod["position_exe_sha"])
        and sha(CK) == CK_SHA,
        "changed TRAIN/encoder/authority",
    )
    packet = FrozenPacket.load(
        Path("output/memory-program-t4/attempt1/retained.f32"), b, meta["sha256"], byte_limit=CAP
    )
    ck = torch.load(CK, map_location="cpu", weights_only=True)
    m = PackedMemoryPolicy(256).train()
    m.load_state_dict(ck["model"])
    require(digest_state(m) == ck["policy_sha"] == prod["policy_sha"], "changed original selector")
    identity = dict(
        initial=prod["policy_sha"],
        checkpoint=CK_SHA,
        packet=meta["sha256"],
        features=meta["payload_sha"],
        encoder=b.encoder_sha,
        source=b.source_sha,
        manifests={k: data[k] for k in ("manifest", "shard", "panel")},
        inputs=[r.input_sha for r in b.rows],
        observed129_sha=prod["data"]["window129_sha"],
        origins=[[r["doc"], r["start"]] for r in rows],
        mount=man["mount"],
        branch_exe_sha=man["exe_sha"],
        position_exe_sha=EXE_SHA,
        split=b.split,
        seed=SEED,
    )
    return m, packet, rows, ws, man, identity


class Physical:
    """A separate rollout or a single paired source. Its callback counts only actual physical dispatches."""

    def __init__(self, records, shared, rows, *, model=None):
        self.records = records
        self.stack = ExitStack()
        self.live = []
        self.sessions = []
        self._guards = {}
        self.work = Counter()
        self.encoded = Counter()
        self.dispatch = sha256()
        self.final = None
        try:
            for r in rows:
                l = PrivateTape(r["doc"], r["start"], byte_limit=129, demand=False)
                self.stack.callback(l.close)
                self.live.append(l)
            raw = [l.readers() for l in self.live]
            limits = Limits(pattern=16)
            self.names = BindingNames(self.live)

            def bind(r, check):
                schema(r)

                def fetch(ks, cap):
                    out = r.read_many(ks, cap)
                    tag = (r.source, r.version, r.operation, r.watermark() if r.watermark else None)
                    add_hash(self.dispatch, (self.names.origin(tag), cap, tuple(map(self.names.key, ks))))
                    self.work.update(
                        frames=1,
                        keys=len(ks),
                        rows=sum(len(v) for v in out if not isinstance(v, Unavailable)),
                        refusals=sum(isinstance(v, Unavailable) for v in out),
                        empty=sum(not v for v in out if not isinstance(v, Unavailable)),
                    )
                    return (
                        out
                        if model is None
                        else for_model(out, r.operation, model.embed.weight, self.encoded)
                    )

                op = replace(r, read_many=fetch)
                self._guards[op] = check
                return op

            a = [bind(r, records.check) for r in shared]
            for l, ops in zip(self.live, raw, strict=True):
                self.sessions.append(
                    Session(
                        [*a, *[bind(r, l.watermark) for r in ops]],
                        limits,
                        advance=l.advance,
                        document=l.document,
                        start=l.start,
                        session_id=l.id,
                    )
                )
        except BaseException:
            self.close()
            raise

    def guard(self, r):
        if r is None:
            self.records.check()
        elif r not in self._guards:
            raise ValueError("unlicensed raw provider")
        else:
            self._guards[r]()

    def observed(self, words):
        require(
            len(words) == len(self.live)
            and all(
                s.visible == 129 and not s.pending and l.watermark() == 129 and bytes(l.data) == w
                for s, l, w in zip(self.sessions, self.live, words, strict=True)
            ),
            "incomplete actual observations",
        )
        return [sha256(l.data).hexdigest() for l in self.live]

    def close(self):
        try:
            self.stack.close()
        finally:
            for s in self.sessions:
                s.cells = [None] * s.limits.cells
                s.readers = ()
                s.advance = None
                s.recent = b""
                s.pending = True
            self._guards.clear()
            self.final = dict(
                physical=dict(self.work),
                encoded=dict(self.encoded),
                requests=self.dispatch.hexdigest(),
                live_work=dict(sum((l.stats for l in self.live), Counter())),
                live_exits=[l.exit for l in self.live],
            )
            self.records = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class Report:
    """Stream discrete and numeric fingerprints. No logp or native answer is fed back to either model."""

    keys = ("bytes", "argmax", "joint", "events", "choices")

    def __init__(self, ss):
        self.names = BindingNames(ss)
        self.ids = [s.id for s in ss]
        self.hashes = [{k: sha256() for k in self.keys} for s in ss]
        self.saves = [{} for _ in ss]
        self.work = Counter()
        self.t = 0

    def note(self, ds, proof):
        self.t += 1
        require(len(ds) == len(self.ids) == len(proof), "incomplete role output")
        with torch.no_grad():
            lp = torch.stack([d.log_probs for d in ds])
            joint = torch.stack([d.log_policy for d in ds])
            tops = lp.argmax(1).to(torch.uint8).tolist()
            require(
                lp.shape == (len(ds), 256)
                and lp.dtype == joint.dtype == torch.float32
                and lp.device.type == "cpu"
                and torch.isfinite(joint).all()
                and lp.isfinite().all()
                and lp.logsumexp(1).abs().max() < 5e-5,
                "invalid fixed-role scores",
            )
            l = lp.numpy()
            j = joint.numpy()
        for i, (sig, d, sid, c, top) in enumerate(zip(self.hashes, ds, self.ids, proof, tops, strict=True)):
            require(
                (sid, self.t) == (d.session_id, d.visible) and len(c) == 64, "changed role owner or frontier"
            )
            sig["bytes"].update(l[i].tobytes())
            sig["argmax"].update(bytes([top]))
            sig["joint"].update(j[i].tobytes())
            sig["choices"].update(bytes.fromhex(c))
            add_hash(sig["events"], events([[d]], self.names)[0])
            self.work.update(targets=1, instructions=len(d.trace), logical_keys=d.requests)
            for e in d.trace:
                self.work[e.inst.op] += 1
                if e.status.startswith("refused"):
                    continue
                if e.inst.op == "SAVE":
                    self.saves[i][e.inst.args[1]] = self.t
                if e.inst.op == "RECALL" and e.rows and self.saves[i][e.inst.args[0]] < self.t:
                    self.work["old_recalls"] += 1

    def finish(self, loss):
        require(self.t == 128 and loss.isfinite() and loss.ndim == 0, "incomplete fixed-role rollout")
        return dict(
            rows=[{k: h.hexdigest() for k, h in sig.items()} for sig in self.hashes],
            work=dict(self.work),
            nll=loss.detach().item(),
            nll_sha=tensor_sha(loss),
        )
