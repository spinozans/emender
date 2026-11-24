"""Inference-only byte-host seam. This cache contains causal host state, never memory-program scratch.

Exclusive trusted owner, one device stream. Opaque in-process admissions, not tenant credentials.
The caller binds one handle per document/window and releases it rather than rolling positions over.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import prod

import torch
from torch.nn import functional as F


@dataclass(frozen=True, eq=False, slots=True)
class Admission:
    slot: int
    session: str
    generation: int


class CachedByteHost:
    """Manage lifetime and frontier. A backend supplies bounded per-slot tensors and `encode`.

    `prefill` requires a fresh admission. `advance` takes (handle, expected old length, new byte).
    `read` never advances the model. Backend writes and lengths are transactional *per call*: on a
    backend exception all admissions touched by the failed call are invalidated, never continued.
    CUDA device faults need process teardown, not reuse. No gradients or cross-session persistence.
    """

    def __init__(self, backend, capacity, context, byte_budget, *, reserve=0):
        if (
            any(type(n) is not int or n < 1 for n in (capacity, context, byte_budget))
            or type(reserve) is not int
            or reserve < 0
            or context > backend.max_context
        ):
            raise ValueError("invalid host capacity/context")
        self.net, self.backend = backend.net, backend
        self.device = backend.device
        self.context = context
        self.capacity = capacity
        self.reserve = reserve  # charged private rows; never admissible handles
        spec = dict(
            backend.spec(context), hidden=((backend.width,), torch.float32), base=((256,), torch.float32)
        )
        needed = (capacity + reserve) * sum(prod(shape) * dtype.itemsize for shape, dtype in spec.values())
        if needed > byte_budget:
            raise ValueError(f"host tensor byte budget exceeded ({needed} > {byte_budget})")
        self.tensor_bytes = needed
        self.slots = {
            k: torch.zeros((capacity + reserve, *shape), device=self.device, dtype=dtype)
            for k, (shape, dtype) in spec.items()
        }
        self._bound = [None] * capacity
        self._positions = [0] * capacity
        self._generation = 0
        self._stamp = self.stamp()
        self.stats = Counter()

    @property
    def empty(self):
        return all(h is None for h in self._bound)

    def stamp(self):
        return tuple((id(p), p._version, p.dtype, p.device) for p in self.net.parameters())

    def _check_model(self):
        if torch.is_grad_enabled() or self.net.training or self.stamp() != self._stamp:
            raise ValueError("changed weights or not an inference context")

    def admit(self, session):
        if (
            not isinstance(session, str)
            or not session
            or any(h is not None and h.session == session for h in self._bound)
        ):
            raise ValueError("invalid or duplicate host session")
        try:
            i = self._bound.index(None)
        except ValueError:
            raise ValueError("host admissions full") from None
        self._generation += 1
        self._bound[i] = h = Admission(i, session, self._generation)
        return h

    def _indices(self, handles):
        if not handles or len(handles) > self.capacity:
            raise ValueError("empty or oversized host batch")
        if any(
            not isinstance(h, Admission) or not 0 <= h.slot < self.capacity or self._bound[h.slot] is not h
            for h in handles
        ):
            raise ValueError("stale or foreign host admission")
        ix = [h.slot for h in handles]
        if len(set(ix)) != len(ix):
            raise ValueError("duplicate host admission")
        return ix

    def _tensor(self, x):
        return torch.as_tensor(x, device=self.device, dtype=torch.long)

    def read(self, handles, visible):
        self._check_model()
        ix = self._indices(handles)
        if len(ix) != len(visible) or any(
            type(v) is not int or not v or v != self._positions[i] for i, v in zip(ix, visible)
        ):
            raise ValueError("stale host frontier")
        select = self._tensor(ix)
        return self.slots["hidden"].index_select(0, select), self.slots["base"].index_select(0, select)

    def release(self, handles):
        ix = self._indices(handles)
        select = self._tensor(ix)
        for t in self.slots.values():
            t.index_fill_(0, select, 0)
        for i in ix:
            self._positions[i] = 0
            self._bound[i] = None
        self.stats["released"] += len(ix)

    def clear(self):
        active = [h for h in self._bound if h is not None]
        if active:
            self.release(active)

    def prefill(self, handles, prompts):
        ix = self._indices(handles)
        if len(prompts) != len(ix) or any(
            self._positions[i] or not isinstance(w, bytes) or not 0 < len(w) <= self.context
            for i, w in zip(ix, prompts)
        ):
            raise ValueError("prefill needs fresh slot and bounded nonempty raw bytes")
        return self._run(handles, ix, prompts, True)

    def advance(self, handles, visible, symbols):
        ix = self._indices(handles)
        if (
            len(ix) != len(visible)
            or len(ix) != len(symbols)
            or any(
                type(v) is not int or not 0 < v < self.context or v != self._positions[i]
                for i, v in zip(ix, visible)
            )
            or any(type(b) is not int or not 0 <= b < 256 for b in symbols)
        ):
            raise ValueError("stale host frontier, capacity exceeded, or invalid byte")
        return self._run(handles, ix, [bytes([b]) for b in symbols], False)

    def _run(self, handles, ix, words, prompt):
        self._check_model()
        groups = defaultdict(list)
        for j, w in enumerate(words):
            groups[len(w)].append(j)
        try:
            for length, js in groups.items():
                slots = self._tensor([ix[j] for j in js])
                ids = self._tensor([list(words[j]) for j in js])
                pos = [self._positions[ix[j]] for j in js]
                h, lp = self.backend.encode(self.slots, slots, pos, ids, prompt)
                if (
                    h.shape != (len(js), self.backend.width)
                    or lp.shape != (len(js), 256)
                    or h.dtype != torch.float32
                    or lp.dtype != torch.float32
                ):
                    raise ValueError("wrong host feature contract")
                self.slots["hidden"].index_copy_(0, slots, h)
                self.slots["base"].index_copy_(0, slots, lp)
                self.stats["prefill_batches" if prompt else "step_batches"] += 1
                self.stats["prefill_tokens" if prompt else "step_tokens"] += len(js) * length
            for i, w in zip(ix, words, strict=True):
                self._positions[i] += len(w)
            # Backend output must not borrow mutable cache storage. Preserve admission order for ragged prefills.
            if len(groups) == 1:
                return h, lp
            return self.read(handles, [self._positions[i] for i in ix])
        except Exception:
            self.release(handles)
            raise


class AbsoluteKV:
    """Dense pre-norm TransformerEncoderLayer with learned absolute positions (retained C14).

    Prefill uses square causal SDPA. A decoded query has length one; its rectangular mask marks
    the session-specific true prefix explicitly. No training, ALiBi/RoPE, GQA, padding or wrap.
    """

    def __init__(self, net):
        self.net = net
        self.device = net.embed.weight.device
        self.width = net.embed.embedding_dim
        self.max_context = net.position.num_embeddings
        self.layers = len(net.blocks)
        self.heads = net.blocks[0].self_attn.num_heads
        if (
            net.training
            or self.width % self.heads
            or net.embed.num_embeddings != 256
            or net.head.out_features != 256
            or net.position.embedding_dim != self.width
            or any(p.dtype != torch.float32 or p.device != self.device for p in net.parameters())
            or any(
                not layer.norm_first
                or layer.activation_relu_or_gelu not in (1, 2)
                or layer.linear1.bias is None
                or not layer.self_attn.batch_first
                or not layer.self_attn._qkv_same_embed_dim
                or layer.self_attn.num_heads != self.heads
                or layer.self_attn.add_zero_attn
                or layer.self_attn.bias_k is not None
                or layer.self_attn.bias_v is not None
                for layer in net.blocks
            )
        ):
            raise ValueError("unsupported byte Transformer")

    def spec(self, context):
        return {"kv": ((self.layers, context, 2, self.heads, self.width // self.heads), torch.float32)}

    def encode(self, arena, slots, old, ids, prompt):
        if torch.is_autocast_enabled(self.device.type):
            raise ValueError("FP32 KV required")
        b, t = ids.shape
        net = self.net
        pos = torch.tensor(old, device=self.device)[:, None] + torch.arange(t, device=self.device)
        x = net.embed(ids) + net.position(pos)
        end = max(old) + t
        prefix_mask = (
            None
            if prompt or min(old) == max(old)
            else torch.arange(end, device=self.device)[None, None, None, :] <= pos[:, None, None, -1:]
        )
        for j, layer in enumerate(net.blocks):
            att = layer.self_attn
            proj = F.linear(layer.norm1(x), att.in_proj_weight, att.in_proj_bias).reshape(
                b, t, 3, self.heads, -1
            )
            q, k, v = proj.unbind(2)
            cache = arena["kv"][:, j]
            if prompt:
                cache[slots, :t] = proj[:, :, 1:]
            else:
                cache[slots, pos[:, 0]] = proj[:, 0, 1:]
                k, v = cache[slots, :end].unbind(2)
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=prefix_mask,
                is_causal=prompt,
            )
            x = x + att.out_proj(y.transpose(1, 2).reshape(b, t, self.width))
            # Match TransformerEncoderLayer's fused eval FFN primitive (including
            # the device-specific GELU rounding). Splitting bias/GELU into a
            # separate eager op is not the stock CUDA eval path.
            ff = layer.linear1
            hidden = torch.ops.aten._addmm_activation(
                ff.bias,
                layer.norm2(x).reshape(b * t, self.width),
                ff.weight.T,
                use_gelu=layer.activation_relu_or_gelu == 2,
            ).view(b, t, -1)
            x = x + layer.linear2(hidden)
        h = net.norm(x[:, -1])
        return h, net.head(h).float().log_softmax(-1)
