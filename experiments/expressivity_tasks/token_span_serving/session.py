"""Serial FP32 CPU reference with optional bounded cross-session passage compilation.

No training, text generation, future target, window padding, fake observations or
cross-document state. `observe` returns real-vocabulary logits for ONE next token.
"""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json

import torch
from torch.nn import functional as F

from .archive import token_id
from experiments.expressivity_tasks.retrieved_span_effects.natural_oracle import (
    pack,
    step,
)


def model_ref(m):
    h = hashlib.sha256(
        json.dumps(
            [
                "cpu-fp32-token-span-v1",
                str(torch.__version__),
                m.host,
                m.arm,
                vars(m.sh),
            ],
            sort_keys=True,
        ).encode()
    )
    for name, v in m.state_dict().items():
        h.update(json.dumps([name, str(v.dtype), list(v.shape)]).encode())
        h.update(v.detach().contiguous().numpy().view("u1"))
    return h.hexdigest()


def replay(state, controls, family):
    for ctl in controls.unbind(0):
        state = step(state, ctl, family)
    return state


def build_affine(controls, k):
    # For A, run a zero-write recurrence from I. For B, use the true writes from 0.
    # The time loop is CPU reference work done only on a cache miss.
    h = controls.shape[1]
    a = torch.eye(k, device=controls.device, dtype=controls.dtype).expand(h, -1, -1)
    b = controls.new_zeros((h, k, k))
    for ctl in controls.unbind(0):
        key, d, r, _ = ctl.split(k, -1)
        dec = d[..., None] * a
        a = dec - key[..., None] * (r[..., None] * dec).sum(-2).unsqueeze(-2)
        b = step(b, ctl, "gdn2")
    return a, b


@dataclass(frozen=True)
class Value:
    kind: str
    payload: tuple

    def apply(self, seed, family, seeded):
        if self.kind == "blank":
            return self.payload[0]
        if self.kind == "affine":
            return self.payload[0] @ seed + self.payload[1]
        return replay(
            seed if seeded else torch.zeros_like(seed), self.payload[0], family
        )


@dataclass(frozen=True)
class Packet:
    key: torch.Tensor
    parts: tuple

    @property
    def nbytes(self):
        return sum(
            t.numel() * t.element_size()
            for t in (self.key, *(t for v in self.parts for t in v.payload))
        )


class FrozenHost:
    """One mounted, private weight snapshot and optional LRU. Do not mutate internals.

    A new optimizer generation, tokenizer or archive requires a new mount. The cache
    contains no live seed, query, route weight, mixture, or cached next-token logits.
    """

    def __init__(self, model, archive, *, tokenizer_sha256, cache_bytes=0):
        if (
            cache_bytes < 0
            or archive.vocab != model.sh.vocab
            or archive.page != model.sh.page
            or archive.tokenizer != tokenizer_sha256
        ):
            raise ValueError("invalid cache limit or archive/host binding")
        if any(
            p.device.type != "cpu" or p.dtype != torch.float32
            for p in model.parameters()
        ):
            raise ValueError("this serial reference requires FP32 CPU parameters")
        self._net = m = deepcopy(model).eval().requires_grad_(False)
        self.weights = model_ref(m)
        self.archive = archive
        self.binding = (self.weights, archive.version, "cpu-fp32-token-span-v1")
        self.budget = cache_bytes
        self._cache = OrderedDict()
        self.cache_payload_bytes = 0
        self.encodes = 0
        self.reuses = 0
        h = m.sh.heads
        self.parts = (
            ((0, h // 2, "gdn2"), (h // 2, h, "e97"))
            if m.arm == "mix-effect"
            else ((0, h, "gdn2" if m.arm.startswith("g-") else "e97"),)
        )

    def clear_cache(self):
        """Drop page summaries, not sessions, weights, the index or lifetime counters."""
        self._cache.clear()
        self.cache_payload_bytes = 0

    def session(self, document_sha256):
        return ReadSession(self, document_sha256)

    def _packet(self, hit):
        # Handles are internal products of this archive. Cached ordinals never leave this mount.
        prior = self._cache.get(hit.ordinal)
        if prior is not None:
            self.reuses += 1
            self._cache.move_to_end(hit.ordinal)
            return prior
        self.encodes += 1
        m = self._net
        sh = m.sh
        k = sh.key
        tok = m.embedding(torch.from_numpy(self.archive.tokens(hit)).long())
        key = m.page_key(tok.mean(0))
        z = tok + m.left(F.pad(tok[:-1], (0, 0, 1, 0)))
        raw = m.control(z).view(len(tok), sh.heads, 5 * k)
        vs = []
        for lo, hi, family in self.parts:
            ctl = pack(raw[:, lo:hi], k, k, family=family)
            if self.budget and not m.arm.endswith("effect"):
                v = Value(
                    "blank", (replay(ctl.new_zeros((hi - lo, k, k)), ctl, family),)
                )
            elif self.budget and family == "gdn2":
                v = Value("affine", build_affine(ctl, k))
            else:
                # No exact seed-independent terminal map for nonlinear E97 EFFECT.
                v = Value("controls", (ctl,))
            vs.append(v)
        pkt = Packet(key, tuple(vs))
        size = pkt.nbytes
        if size <= self.budget:
            while self.cache_payload_bytes + size > self.budget:
                _, old = self._cache.popitem(last=False)
                self.cache_payload_bytes -= old.nbytes
            self._cache[hit.ordinal] = pkt
            self.cache_payload_bytes += size
        return pkt

    def _read(self, live, x, hits):
        if not hits:
            return None, None
        m = self._net
        seed = m.align_l @ live @ m.align_r
        votes = seed.new_tensor([h.mass for h in hits])
        logmass = votes.sum().log1p().reshape(1)
        if m.arm == "local":
            return seed, logmass
        packets = [self._packet(h) for h in hits]
        q = m.address(x)
        keys = torch.stack([p.key for p in packets], 0)
        prob = ((keys * q).sum(-1) * q.numel() ** -0.5 + votes.log()).softmax(0)
        values = []
        for j, (lo, hi, family) in enumerate(self.parts):
            states = torch.stack(
                [
                    p.parts[j].apply(seed[lo:hi], family, m.arm.endswith("effect"))
                    for p in packets
                ]
            )
            avg = (states * prob[:, None, None, None]).sum(0)
            values.append(avg + seed[lo:hi] if m.arm.endswith("neutral") else avg)
        return torch.cat(values), logmass


def advance(block, x, state):
    k = block.shape.key
    h = block.shape.heads
    z = block.norm(x)
    ctl = pack(block.controls(z).view(h, 5 * k), k, k, family=block.family)
    state = step(state, ctl, block.family)
    y = (block.query(z).view(h, k, 1) * state).sum(-2)
    v = x + block.out(y.flatten())
    return v + block.mlp(v), state


class ReadSession:
    def __init__(self, host, document_sha256):
        self.host = host
        self.reset(document_sha256)

    def _clear_window(self):
        m = self.host._net
        sh = m.sh
        self._lower = m.align_l.new_zeros((sh.heads, sh.key, sh.key))
        self._upper = torch.zeros_like(self._lower)
        self._prefix = []
        self.position = 0
        self._memory = None
        self._logmass = None
        self.last_hits = ()

    def reset(self, document_sha256):
        # Identity metadata is only an admission check, not a model/search input. The
        # caller need not reveal any following live text, including to compute its ID.
        self.host.archive.check_live(document_sha256)
        self.observations = 0
        self.lookups = 0
        self._clear_window()

    @torch.inference_mode()
    @torch.autocast("cpu", enabled=False)
    def observe(self, token):
        m = self.host._net
        sh = m.sh
        i = token_id(token, sh.vocab)
        if self.position == sh.time:
            self._clear_window()
        x, self._lower = advance(m.lower, m.embedding.weight[i], self._lower)
        if self.position < sh.prefix:
            self._prefix.append(i)
            if len(self._prefix) == sh.prefix:
                self.last_hits = self.host.archive.locate(self._prefix, sh.hits)
                self.lookups += 1
                self._memory, self._logmass = self.host._read(
                    self._lower, x, self.last_hits
                )
                self._prefix.clear()
        if self._memory is not None:
            q = m.read_query(x).view(sh.heads, sh.key, 1)
            y = (q * self._memory).sum(-2)
            gate = m.gate(torch.cat((x, self._logmass))).sigmoid()
            x = x + m.out((y * gate[:, None]).flatten())
        y, self._upper = advance(m.upper, x, self._upper)
        self.position += 1
        self.observations += 1
        return F.linear(m.norm(y), m.embedding.weight[: sh.vocab])
