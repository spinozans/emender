"""KEY SEPARATION PROBE (SEP-KV) -- is key-value separation NECESSARY?

The operator's hypothesis, sharpened by a compression-theoretic argument (the
Wheeler-graph frame): a store whose keys ARE its contents has compression
ratio 1 and degenerates to identity caching -- the receipted diagnosis at 150 M
(``CMAES_TUNING.md``: CORRECT 60-71 % of decisions, firing mostly as exact-match
*identity rewriting* over repeated byte-level payload directions).  The
hypothesis under test: **key-value separation is necessary** -- with key=value
(the registered design, matching by raw content cosine) two memories with
identical content are mechanically indistinguishable regardless of parameters;
with a small separate key, context-dependent addressing solves it.

The collision lemma (exact, parameter-free; machine-checked in
``test_key_separation_probe.py``)
---------------------------------------------------------------------------
At layer 1 the registered payload is a fixed function of the token:
``payload_t = W_v RMSNorm(embed(token_t))`` -- no cross-token term.  Therefore:

* two occurrences of the same token produce BIT-IDENTICAL payloads, for every
  parameter setting;
* while the first copy's row is live with content ``c * f(X)`` (``c`` the
  accumulated decay scalar), the second copy's routing similarity to that row is
  ``cos(f(X), c f(X)) = 1 >= tau_match`` -- the second write is CORRECT or MERGE
  onto that row, NEVER an APPEND into a fresh slot.  The two copies mechanically
  cannot occupy two rows of the store;
* if two live rows ever share content, every read score against them is
  identical (cosine is a function of content only), hence their softmax masses
  are exactly equal -- the read channel is mechanically role-blind between
  content-identical rows.

Task families (the answer REQUIRES filing identical content under
context-dependent addresses; the binding is per-window random as in SEP-AR, so
no fixed token association can be memorised)
---------------------------------------------------------------------------
**M1 (context-keyed retrieval).**  One window, one shared token X drawn from
4 ambiguous identities, presented twice under explicit ROLE MARKERS:
``BOS PAD PAD R0 X c0 R1 X c1 probe(0,x) c0 probe(1,x) c1``.  The continuations
``c0, c1`` are drawn without replacement from 32 value tokens per window and
revealed teacher-forced right after each copy.  The probes are combined
(role, x) tokens; the target is the role-dependent continuation.  The binding
write is the continuation's own write: under separated keys its ADDRESS is the
write context (``R_r, X``); under key=value it is the content ``f(c_r)``, which
carries no role at all.

**M2 (composition).**  The role is a contextual feature ABSENT from every
token's raw content: the ORDER of two shared frame tokens.  Frame-0 presents
``P Q`` before its X copy, frame-1 presents ``Q P``:
``BOS PAD PAD P Q X c0 Q P X c1 P Q X(->c0) c0 Q P X(->c1) c1``.  Content
addressing is order-blind (``P`` and ``Q`` write the same rows in either order;
the cosine channel cannot see order); the separated key channel sees the
context sequence mechanically.  The probes re-present the frame (``P Q X`` vs
``Q P X``) so the retrieval must key on the compositional order feature.

Arms (the necessity claim is PER-LAYER; both arms run at DEPTH 1)
---------------------------------------------------------------------
**(a) FUSED -- the CURRENT registered per-token arm, verbatim.**  The
granularity separator's blocks at ``chunk_tokens=1`` (the K6 semantic): the
registered five-op ``RoutingPolicy``, the registered eager scan
(``scan.routed_chunk_scan``, ``schedule.route_chunk``), matching by raw
content cosine, at ONE block -- the first store layer, where the payload is a
fixed function of the token and the collision lemma binds exactly.
**(b) SEPARATED -- the minimal separated-key variant, ORACLE LEVEL.**  The
smallest change that enables context addressing: the writer emits a small key
``k_t in R^key_dim`` alongside the payload -- a learned projection of the WRITE
CONTEXT (the last ``CONTEXT`` positions of the block input, per-position
RMSNormed) -- and matching switches from ``cos(q, S_j)`` to ``cos(q_key, K_j)``
for BOTH the read score and the routing similarity, with the values (payloads,
DECAY, MERGE blend, POP) unchanged.  The read keeps the certified form
(``softmax_j(scale * cos) * strength_j`` with ``strength_j = ||S_j||``).
Implemented as an eager per-token torch oracle (``separated_oracle_scan``), the
K6 doctrine verbatim: decisions from the detached state, read weights attached.
Keys are NOT decayed (an address is categorical here) -- flagged as the first
open question for the registration.  No kernel anywhere: this probe must not
touch k6/k7/k8/k9.

**The depth-2 corroboration arm** (part 3): the registered FUSED arm at two
blocks, receipting the DEPTH-SUBSTITUTION finding -- at layer 2 the binding
row's payload is a composite of (its value, its write context injected by the
layer-1 read), so content matching synthesizes context-dependent addressing
one layer deep (``composite_receipt``).  The necessity claim is per-layer; the
cost contrast (one full extra layer vs a small key channel) is the finding,
not a failure.

Pre-stated predictions live in ``RESULT_KEY_SEPARATION.md`` (committed before
any training run), with the refutation clause: if the fused arm learns M1/M2
anywhere in its sweep, the necessity claim is REFUTED and reported as such.

    eval "$(GPU_LEASE_VISIBLE=2,3 scripts/gpu_lease.sh acquire 1 --wait)"
    /home/erikg/emender/.venv/bin/python -m \
        experiments.expressivity_tasks.x45_matrix_emender.key_separation_probe \
        --part 1 --task both --arm both
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .algebra import (APPEND, CORRECT, MERGE, NONE, POP, op_histogram)
from .granularity_separator import SeparatorBlock
from .model import DECAY_FLOOR, DECAY_HORIZON_TOKENS
from .schedule import RoutingPolicy, route_chunk
from .scan import routed_chunk_scan

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / "receipts"

EPS = 1e-6
CHANCE = 1.0 / 32          # the value-restricted accuracy floor (32 values)
CONTEXT = 4                 # the separated arm's write-context window
DEFAULT_KEY_DIM = 16

# ------------------------------------------------------------------ vocabulary
BOS, PAD = 0, 1
# M1: explicit role markers R0/R1; the ambiguous token X; 32 values; the
# combined (role, x) probe tokens.
M1_R0, M1_R1 = 2, 3
X_BASE, N_X = 4, 4
VAL_BASE, N_VALUES = 8, 32
M1_PROBE_BASE = VAL_BASE + N_VALUES                       # 40
M1_PROBE = lambda role, x: M1_PROBE_BASE + role * N_X + x  # 8 combined probes
M1_VOCAB = M1_PROBE_BASE + 2 * N_X                        # 48
M1_WINDOW = 13
M1_COPY_POSITIONS = (4, 7)          # the two identical-content X writes
M1_REVEAL_POSITIONS = (5, 8)       # the continuation writes (the binding rows)
M1_PROBE_POSITIONS = (9, 11)       # targets: probe(0,x)->c0, probe(1,x)->c1

# M2: the role is the ORDER of the shared frame tokens P, Q.
M2_P, M2_Q = 2, 3
M2_VOCAB = VAL_BASE + N_VALUES                           # 40
M2_WINDOW = 19
M2_COPY_POSITIONS = (5, 9)
M2_REVEAL_POSITIONS = (6, 10)
M2_PROBE_POSITIONS = (13, 17)      # targets: (P,Q,X)->c0, (Q,P,X)->c1


class TaskSpec:
    """One task family's layout: everything the arms and receipts share."""

    def __init__(self, name, vocab, window, copy_positions, reveal_positions,
                 probe_positions, make_batch):
        self.name = name
        self.vocab = vocab
        self.window = window
        self.copy_positions = tuple(copy_positions)
        self.reveal_positions = tuple(reveal_positions)
        self.probe_positions = tuple(probe_positions)
        self.make_batch = make_batch

    @property
    def n_probes(self) -> int:
        return len(self.probe_positions)


def make_m1_batch(rng: torch.Generator, count: int):
    """Sample ``count`` M1 windows.  The ambiguous identity and the two
    continuations are drawn per window (continuations without replacement, so
    the answer is unambiguous and the binding is window-specific)."""
    x_id = torch.randint(0, N_X, (count,), generator=rng)
    values = torch.argsort(torch.rand(count, N_VALUES, generator=rng),
                           dim=1)[:, :2]
    ids = torch.empty(count, M1_WINDOW, dtype=torch.long)
    ids[:, 0], ids[:, 1], ids[:, 2] = BOS, PAD, PAD
    ids[:, 3], ids[:, 6] = M1_R0, M1_R1
    ids[:, 4] = ids[:, 7] = X_BASE + x_id
    ids[:, 5] = VAL_BASE + values[:, 0]
    ids[:, 8] = VAL_BASE + values[:, 1]
    ids[:, 9] = M1_PROBE(0, x_id)
    ids[:, 10] = VAL_BASE + values[:, 0]
    ids[:, 11] = M1_PROBE(1, x_id)
    ids[:, 12] = VAL_BASE + values[:, 1]
    targets = torch.stack((VAL_BASE + values[:, 0], VAL_BASE + values[:, 1]),
                          dim=1)
    return ids, targets, x_id


def make_m2_batch(rng: torch.Generator, count: int):
    """Sample ``count`` M2 windows.  Frame-0 is the order (P, Q), frame-1 the
    order (Q, P); the probes re-present the frame before the bare X whose
    logits must name the frame-dependent continuation."""
    x_id = torch.randint(0, N_X, (count,), generator=rng)
    values = torch.argsort(torch.rand(count, N_VALUES, generator=rng),
                           dim=1)[:, :2]
    ids = torch.empty(count, M2_WINDOW, dtype=torch.long)
    ids[:, 0], ids[:, 1], ids[:, 2] = BOS, PAD, PAD
    ids[:, 3], ids[:, 4] = M2_P, M2_Q          # frame 0: P Q
    ids[:, 7], ids[:, 8] = M2_Q, M2_P          # frame 1: Q P
    ids[:, 11], ids[:, 12] = M2_P, M2_Q        # probe frame 0
    ids[:, 15], ids[:, 16] = M2_Q, M2_P        # probe frame 1
    ids[:, 5] = ids[:, 9] = ids[:, 13] = ids[:, 17] = X_BASE + x_id
    ids[:, 6] = VAL_BASE + values[:, 0]
    ids[:, 10] = VAL_BASE + values[:, 1]
    ids[:, 14] = VAL_BASE + values[:, 0]
    ids[:, 18] = VAL_BASE + values[:, 1]
    targets = torch.stack((VAL_BASE + values[:, 0], VAL_BASE + values[:, 1]),
                          dim=1)
    return ids, targets, x_id


TASKS = {
    "m1": TaskSpec("m1", M1_VOCAB, M1_WINDOW, M1_COPY_POSITIONS,
                   M1_REVEAL_POSITIONS, M1_PROBE_POSITIONS, make_m1_batch),
    "m2": TaskSpec("m2", M2_VOCAB, M2_WINDOW, M2_COPY_POSITIONS,
                   M2_REVEAL_POSITIONS, M2_PROBE_POSITIONS, make_m2_batch),
}


# ------------------------------------------------------------- arm (a): fused
class FusedModel(nn.Module):
    """The registered per-token arm: the granularity separator's blocks at
    ``chunk_tokens=1`` (the K6 semantic), on this probe's task layout.  The
    mixer, policy, scan and shapes are the separator's modules verbatim; only
    the vocabulary and probe positions differ."""

    def __init__(self, task: TaskSpec, width: int = 64, hidden: int = 128,
                 depth: int = 2, heads: int = 1, value_dim: int = 32,
                 rank: int = 16, policy: RoutingPolicy | None = None):
        super().__init__()
        policy = RoutingPolicy() if policy is None else policy
        self.task, self.policy = task, policy
        self.heads, self.value_dim, self.rank = heads, value_dim, rank
        self.embed = nn.Embedding(task.vocab, width)
        self.blocks = nn.ModuleList(
            SeparatorBlock(width, hidden, value_dim, heads, rank, policy, 1)
            for _ in range(depth))
        self.norm = nn.RMSNorm(width, eps=1e-5)
        self.head = nn.Linear(width, task.vocab, bias=False)

    def forward(self, ids, collect: bool = False):
        x = self.embed(ids)
        results = [] if collect else None
        for block in self.blocks:
            x, result = block(x, collect=collect)
            if collect:
                results.append(result)
        return self.head(self.norm(x)), results

    def loss(self, ids, targets):
        logits, _ = self.forward(ids)
        probe = logits[:, self.task.probe_positions]
        return F.cross_entropy(probe.reshape(-1, probe.shape[-1]),
                               targets.reshape(-1))


# --------------------------------------------- arm (b): separated keys (oracle)
def causal_context(x: torch.Tensor, ctx: int) -> torch.Tensor:
    """The last ``ctx`` positions' block input, position-clamped at 0.
    ``(b, T, width) -> (b, T, ctx*width)``.  This is the WRITE CONTEXT the key
    projections see -- the minimal change that makes a write's address a
    function of the local context rather than of the current token alone (at
    layer 1 any current-position-only projection is a fixed function of the
    token, which is exactly the fused arm's blindness)."""
    batch, tokens, width = x.shape
    offsets = torch.arange(tokens).unsqueeze(1) \
        - torch.arange(ctx - 1, -1, -1).unsqueeze(0)      # (T, ctx): t-ctx+1..t
    return x[:, offsets.clamp(min=0).to(x.device)]


def _five_op_selection(sim, live, policy: RoutingPolicy, rank: int):
    """The registered selection chain (``route_chunk`` at chunk granularity 1)
    on one token's similarity row ``(b, h, rank)``.  ``sim`` must already be the
    DETACHED routing similarity with dead rows at -2.  Everything downstream
    of the similarity is a comparison, so nothing here is differentiable."""
    penalty = torch.arange(rank, dtype=torch.float32,
                           device=sim.device) * policy.tie_break
    score = sim - penalty
    best_index = score.argmax(-1)
    best_value = sim.gather(-1, best_index.unsqueeze(-1)).squeeze(-1)
    second_score = score.scatter(-1, best_index.unsqueeze(-1), -3.0)
    second_index = second_score.argmax(-1)
    second_value = sim.gather(-1, second_index.unsqueeze(-1)).squeeze(-1)
    worst_index = score.argmin(-1)
    worst_value = sim.gather(-1, worst_index.unsqueeze(-1)).squeeze(-1)
    correct = best_value >= policy.tau_match
    merge = correct & (second_value >= policy.tau_merge)
    free = ~live
    free_count = free.sum(-1)
    need = ~correct
    appendable = need & (free_count > 0)
    free_slot = free.to(torch.int64).argmax(-1)   # first free index (chunk-1)
    full = free_count <= 0
    any_live = live.any(-1)
    pop = (~correct) & (~appendable) & full & (worst_value <= policy.tau_pop) \
        & any_live
    op0 = torch.where(correct,
                      torch.where(merge, torch.full_like(best_index, MERGE),
                                  torch.full_like(best_index, CORRECT)),
                      torch.where(appendable,
                                  torch.full_like(best_index, APPEND),
                                  torch.where(pop, torch.full_like(best_index, POP),
                                              torch.full_like(best_index, NONE))))
    slot0 = torch.where(correct, best_index,
                        torch.where(appendable, free_slot, worst_index))
    erase0 = (op0 == CORRECT) | (op0 == MERGE) | (op0 == POP)
    live0 = op0 != POP
    return dict(best_index=best_index, second_index=second_index,
                correct=correct, merge=merge, op0=op0, slot0=slot0,
                erase0=erase0, live0=live0,
                carries=(op0 == APPEND) | (op0 == CORRECT))


def separated_oracle_scan(payload, write_keys, query_keys, decay, *,
                          policy: RoutingPolicy, rank: int, collect=True):
    """The separated-key arm's eager per-token oracle, differentiable.

    Walks the window token by token, vectorised over (batch, head): per token
    the READ (the certified weights on the KEY channel, from the attached
    current state and keys), the DECISIONS (the registered five-op chain on the
    DETACHED key similarity), then DECAY (values only -- keys are addresses and
    are not decayed in this probe) and the lanes in the registered order.  The
    stored key is re-filed on every APPEND/CORRECT/MERGE (the row's address
    becomes the new write context) and cleared on POP; a MERGE's source row
    loses value, key and liveness.
    """
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    device = payload.device
    state = torch.zeros(batch, heads, rank, value_dim, dtype=torch.float32,
                        device=device)
    keys = torch.zeros(batch, heads, rank, key_dim, dtype=torch.float32,
                       device=device)
    live = torch.zeros(batch, heads, rank, dtype=torch.bool, device=device)
    reads = torch.empty(batch, tokens, heads, value_dim, dtype=torch.float32,
                        device=device)
    channels = dict(
        slot=torch.zeros(batch, tokens, heads, dtype=torch.int64,
                         device=device),
        second=torch.zeros(batch, tokens, heads, dtype=torch.int64,
                           device=device),
        op=torch.zeros(batch, tokens, heads, dtype=torch.int64,
                       device=device),
        emitted_keys=torch.zeros(batch, tokens, heads, key_dim,
                                 dtype=torch.float32, device=device),
        weights=torch.zeros(batch, tokens, heads, rank, dtype=torch.float32,
                            device=device),
        live_pre=torch.zeros(batch, tokens, heads, rank, dtype=torch.bool,
                             device=device))
    b_index = torch.arange(batch, device=device)[:, None]
    h_index = torch.arange(heads, device=device)[None, :]

    for t in range(tokens):
        k_t = write_keys[:, t]                                   # (b, h, K)
        qk_t = query_keys[:, t]
        p_t = payload[:, t]
        lam_t = decay[:, t]                                      # (b, h)
        # ---- the read: the certified weights on the key channel (attached).
        strength = state.norm(dim=-1) * live.to(torch.float32)   # (b, h, rank)
        uq = qk_t / qk_t.norm(dim=-1, keepdim=True).clamp_min(EPS)
        uk = keys / keys.norm(dim=-1, keepdim=True).clamp_min(EPS)
        scores = policy.read_scale * torch.einsum("bhk,bhrk->bhr", uq, uk)
        any_live = live.any(-1, keepdim=True)
        masked = scores.masked_fill(~live, float("-inf"))
        safe = torch.where(any_live, masked, torch.zeros_like(masked))
        probs = torch.where(any_live, torch.softmax(safe, -1),
                            torch.zeros_like(safe))
        w = probs * strength
        reads[:, t] = torch.einsum("bhr,bhrv->bhv", w, state)
        if collect:
            channels["live_pre"][:, t] = live
            channels["weights"][:, t] = w
            channels["emitted_keys"][:, t] = k_t.detach()
        # ---- the decisions: the registered chain on the DETACHED key channel.
        keys_det = keys.detach()
        sim = torch.einsum(
            "bhk,bhrk->bhr",
            k_t.detach() / k_t.detach().norm(dim=-1, keepdim=True).clamp_min(EPS),
            keys_det / keys_det.norm(dim=-1, keepdim=True).clamp_min(EPS))
        sim = sim.masked_fill(~live, -2.0)
        sel = _five_op_selection(sim, live, policy, rank)
        # ---- the payload lane (attached; the MERGE blend's content attached,
        # its strength weights detached -- the registered construction).
        rows = state[b_index.expand(batch, heads),
                     h_index.expand(batch, heads), sel["best_index"]]
        rows2 = state[b_index.expand(batch, heads),
                      h_index.expand(batch, heads), sel["second_index"]]
        fw = strength.gather(-1, sel["best_index"].unsqueeze(-1)
                             ).clamp_min(EPS).squeeze(-1)
        sw = strength.gather(-1, sel["second_index"].unsqueeze(-1)
                             ).clamp_min(EPS).squeeze(-1)
        total = fw + sw
        blend = ((fw / total).detach().unsqueeze(-1) * rows
                 + (sw / total).detach().unsqueeze(-1) * rows2)
        carries = sel["carries"].unsqueeze(-1)
        lane0 = torch.where(carries, p_t, torch.zeros_like(p_t))
        lane0 = torch.where(sel["merge"].unsqueeze(-1), blend, lane0)
        # ---- the key lane: APPEND/CORRECT/MERGE re-file the row under the new
        # write context; POP clears the address; NONE leaves it untouched.
        key_lane = torch.where(sel["op0"].eq(POP).unsqueeze(-1),
                               torch.zeros_like(k_t), k_t)
        if collect:
            channels["slot"][:, t] = sel["slot0"]
            channels["second"][:, t] = sel["second_index"]
            channels["op"][:, t] = sel["op0"]
        # ---- apply: DECAY (values only), then the lanes in the registered order.
        state = state * lam_t[:, :, None, None]
        rows_at = state[b_index.expand(batch, heads),
                        h_index.expand(batch, heads), sel["slot0"]]
        written = torch.where(sel["erase0"].unsqueeze(-1), lane0, rows_at + lane0)
        state = state.index_put((b_index.expand(batch, heads),
                                 h_index.expand(batch, heads), sel["slot0"]),
                                written)
        keys_at = keys[b_index.expand(batch, heads),
                       h_index.expand(batch, heads), sel["slot0"]]
        keys = keys.index_put(
            (b_index.expand(batch, heads), h_index.expand(batch, heads),
             sel["slot0"]),
            torch.where(sel["op0"].ne(NONE).unsqueeze(-1), key_lane, keys_at))
        live = live.index_put((b_index.expand(batch, heads),
                               h_index.expand(batch, heads), sel["slot0"]),
                              torch.where(sel["op0"].ne(NONE), sel["live0"],
                                          live.gather(-1,
                                                      sel["slot0"].unsqueeze(-1)
                                                      ).squeeze(-1)))
        if bool(sel["merge"].any()):
            state = state.index_put(
                (b_index.expand(batch, heads), h_index.expand(batch, heads),
                 sel["second_index"]),
                torch.zeros(batch, heads, value_dim, dtype=torch.float32,
                            device=device))
            keys = keys.index_put(
                (b_index.expand(batch, heads), h_index.expand(batch, heads),
                 sel["second_index"]),
                torch.zeros(batch, heads, key_dim, dtype=torch.float32,
                            device=device))
            live = live.index_put(
                (b_index.expand(batch, heads), h_index.expand(batch, heads),
                 sel["second_index"]),
                torch.zeros(batch, heads, dtype=torch.bool, device=device))
    return dict(reads=reads, state=state, live=live, keys=keys, **channels)


class SeparatedMixer(nn.Module):
    """The minimal separated-key mixer: payloads/DECAY/output unchanged from
    the registered mixer; the content query is REPLACED by a key-space query,
    and the writer gains a small key projection of the write context."""

    def __init__(self, width: int, value_dim: int, heads: int, rank: int,
                 key_dim: int, policy: RoutingPolicy, ctx: int = CONTEXT):
        super().__init__()
        self.heads, self.value_dim, self.rank = heads, value_dim, rank
        self.policy, self.key_dim, self.ctx = policy, key_dim, ctx
        self.value = nn.Linear(width, heads * value_dim, bias=False)
        self.key = nn.Linear(ctx * width, heads * key_dim, bias=False)
        self.query_key = nn.Linear(ctx * width, heads * key_dim, bias=False)
        self.out = nn.Linear(heads * value_dim, width, bias=False)
        self.decay = nn.Linear(width, heads, bias=True)
        self.ctx_norm = nn.RMSNorm(width, eps=1e-5)
        with torch.no_grad():
            lam0 = math.exp(-1.0 / DECAY_HORIZON_TOKENS)
            self.decay.bias.fill_(math.log(lam0 / (1.0 - lam0)))

    def forward(self, x, collect: bool = False):
        batch, tokens, _ = x.shape
        ctx = causal_context(x, self.ctx)
        ctx_flat = self.ctx_norm(ctx.reshape(batch * tokens * self.ctx, -1)) \
            .reshape(batch, tokens, self.ctx * x.shape[-1])
        payload = self.value(x).view(batch, tokens, self.heads, self.value_dim)
        write_keys = self.key(ctx_flat).view(batch, tokens, self.heads,
                                             self.key_dim)
        query_keys = self.query_key(ctx_flat).view(batch, tokens, self.heads,
                                                   self.key_dim)
        decay = torch.sigmoid(self.decay(x)).clamp(min=DECAY_FLOOR,
                                                   max=1.0 - 1e-6)
        result = separated_oracle_scan(payload, write_keys, query_keys, decay,
                                      policy=self.policy, rank=self.rank,
                                      collect=collect)
        reads = result["reads"].reshape(batch, tokens,
                                        self.heads * self.value_dim)
        return self.out(reads), result


class SeparatedBlock(nn.Module):
    def __init__(self, width: int, hidden: int, value_dim: int, heads: int,
                 rank: int, key_dim: int, policy: RoutingPolicy,
                 ctx: int = CONTEXT):
        super().__init__()
        self.norm1 = nn.RMSNorm(width, eps=1e-5)
        self.mixer = SeparatedMixer(width, value_dim, heads, rank, key_dim,
                                    policy, ctx)
        self.norm2 = nn.RMSNorm(width, eps=1e-5)
        self.w1 = nn.Linear(width, hidden, bias=False)
        self.w2 = nn.Linear(width, hidden, bias=False)
        self.w3 = nn.Linear(hidden, width, bias=False)

    def forward(self, x, collect: bool = False):
        mixed, result = self.mixer(self.norm1(x), collect=collect)
        h = x + mixed
        gate = F.silu(self.w1(self.norm2(h))) * self.w2(self.norm2(h))
        return h + self.w3(gate), result


class SeparatedModel(nn.Module):
    """The separated-key arm: the registered block shape with the mixer's
    addressing moved to a small learned key channel."""

    def __init__(self, task: TaskSpec, key_dim: int = DEFAULT_KEY_DIM,
                 width: int = 64, hidden: int = 128, depth: int = 2,
                 heads: int = 1, value_dim: int = 32, rank: int = 16,
                 policy: RoutingPolicy | None = None):
        super().__init__()
        policy = RoutingPolicy() if policy is None else policy
        self.task, self.policy = task, policy
        self.embed = nn.Embedding(task.vocab, width)
        self.blocks = nn.ModuleList(
            SeparatedBlock(width, hidden, value_dim, heads, rank, key_dim,
                           policy) for _ in range(depth))
        self.norm = nn.RMSNorm(width, eps=1e-5)
        self.head = nn.Linear(width, task.vocab, bias=False)

    def forward(self, ids, collect: bool = False):
        x = self.embed(ids)
        results = [] if collect else None
        for block in self.blocks:
            x, result = block(x, collect=collect)
            if collect:
                results.append(result)
        return self.head(self.norm(x)), results

    def loss(self, ids, targets):
        logits, _ = self.forward(ids)
        probe = logits[:, self.task.probe_positions]
        return F.cross_entropy(probe.reshape(-1, probe.shape[-1]),
                               targets.reshape(-1))


def make_model(task_name: str, arm: str, key_dim: int = DEFAULT_KEY_DIM,
               depth: int = 1):
    """The necessity probe's arms run at DEPTH 1 (the airtight layer-1 claim:
    at the first store layer every payload is a fixed function of the token,
    so the collision lemma binds exactly).  The depth-2 corroboration arm
    passes ``depth=2`` and receipts the depth-substitution escape."""
    task = TASKS[task_name]
    if arm == "fused":
        return FusedModel(task, depth=depth)
    if arm == "separated":
        return SeparatedModel(task, key_dim=key_dim, depth=depth)
    raise ValueError(f"unknown arm {arm!r}")


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def probe_predictions(model, ids, task: TaskSpec, batch: int = 1024):
    """Value-restricted argmax at the probe positions: (N, n_probes) predicted
    value-token ids.  The task is "which value"; chance = 1/32."""
    model.eval()
    picks = []
    for start in range(0, ids.shape[0], batch):
        logits, _ = model.forward(ids[start:start + batch])
        probe = logits[:, task.probe_positions, VAL_BASE:VAL_BASE + N_VALUES]
        picks.append(probe.argmax(-1) + VAL_BASE)
    return torch.cat(picks)


@torch.no_grad()
def evaluate(model, ids, targets, task: TaskSpec) -> dict:
    picks = probe_predictions(model, ids, task)
    correct = picks == targets
    return dict(value_acc=float(correct.float().mean()),
                probe_acc=[float(correct[:, j].float().mean())
                           for j in range(task.n_probes)],
                n_probes=int(correct.numel()), chance=CHANCE)


# ------------------------------------------------------------------- receipts
def _pairwise_identical_rows(state, live):
    """Count live row pairs whose CONTENT is identical (cos > 0.999).  Under
    the collision lemma's write half this count is 0: the routing never files
    two identical payloads into two live rows."""
    unit = state / state.norm(dim=-1, keepdim=True).clamp_min(EPS)
    cos = torch.einsum("bhrv,bhsv->bhrs", unit, unit)
    live2 = live.unsqueeze(-1) & live.unsqueeze(-2)
    triu = torch.triu(torch.ones_like(cos), diagonal=1).to(torch.bool)
    return int(((cos > 0.999) & live2 & triu).sum())


@torch.no_grad()
def fused_prefix_state(model: FusedModel, ids, layer: int, upto: int):
    """The state token ``upto``'s read sees, at ``layer`` -- the prefix scan of
    the SAME causal mixer (the full run's recorded weights at ``upto`` were
    computed against exactly this state)."""
    x = model.embed(ids[:, :upto])
    for index, block in enumerate(model.blocks):
        h = block.norm1(x)
        mixer = block.mixer
        batch, tokens = h.shape[0], h.shape[1]
        query = mixer.query(h).view(batch, tokens, mixer.heads, mixer.value_dim)
        payload = mixer.value(h).view(batch, tokens, mixer.heads,
                                      mixer.value_dim)
        decay = torch.sigmoid(mixer.decay(h)).clamp(min=DECAY_FLOOR,
                                                     max=1.0 - 1e-6)
        result = routed_chunk_scan(query, payload, decay, router=route_chunk,
                                  policy=mixer.policy, rank=mixer.rank,
                                  rank_max=mixer.rank, chunk_tokens=1,
                                  collect=False, fused=False)
        if index == layer:
            return result.state, result.live
        x = x + mixer.out(result.reads.reshape(batch, tokens,
                                                mixer.heads * mixer.value_dim))
        h2 = block.norm2(x)
        x = x + block.w3(F.silu(block.w1(h2)) * block.w2(h2))
    raise ValueError("layer out of range")


@torch.no_grad()
def composite_receipt(model: FusedModel, ids, task: TaskSpec):
    """The depth-substitution MECHANISM receipt (supervisor requirement): the
    layer-2 binding payload decomposes into (its value + window-invariant
    context components injected by the layer-1 read).  For each reveal write
    we receipt the cosine of the layer-2 payload against the value's own
    layer-1 payload, the three preceding tokens' layer-1 payloads (the marker/
    frame/X directions), and the layer-1 read output at the write -- the
    actual injection channel.  The contrast arm (separated, depth 1) keeps
    the value payload a pure fixed function of the token (cosine exactly 1)
    and carries the context in the KEY channel instead (receipted by
    ``collect_separated_channels``)."""
    assert len(model.blocks) >= 2, "the composite receipt needs layer 2"
    x = model.embed(ids)
    block1, block2 = model.blocks[0], model.blocks[1]
    h1 = block1.norm1(x)
    m1 = block1.mixer
    batch, tokens = h1.shape[0], h1.shape[1]
    query = m1.query(h1).view(batch, tokens, m1.heads, m1.value_dim)
    payload1 = m1.value(h1).view(batch, tokens, m1.heads, m1.value_dim)
    decay = torch.sigmoid(m1.decay(h1)).clamp(min=DECAY_FLOOR, max=1.0 - 1e-6)
    scan1 = routed_chunk_scan(query, payload1, decay, router=route_chunk,
                              policy=m1.policy, rank=m1.rank,
                              rank_max=m1.rank, chunk_tokens=1, collect=False,
                              fused=False)
    x_mid = x + m1.out(scan1.reads.reshape(batch, tokens,
                                           m1.heads * m1.value_dim))
    h2 = block2.norm1(x_mid)
    payload2 = block2.mixer.value(h2).view(batch, tokens, block2.mixer.heads,
                                          block2.mixer.value_dim)
    reads1 = scan1.reads[:, :, 0]                     # (N, T, V), head 0
    rows = []
    for reveal in task.reveal_positions:
        p2 = payload2[:, reveal, 0]                    # (N, V)
        entry = {"position": reveal,
                 "cos_to_value": float(_cos_rows(p2, payload1[:, reveal, 0]))}
        for back in (1, 2, 3):
            entry[f"cos_to_prev_{back}"] = float(
                _cos_rows(p2, payload1[:, reveal - back, 0]))
        entry["cos_to_l1_read"] = float(_cos_rows(p2, reads1[:, reveal]))
        rows.append(entry)
    return dict(
        note=("layer-2 payload cosines at the reveal (binding) writes: the "
              "value component plus the window-invariant context components "
              "(prev_1..3 = the marker/frame/X directions) and the layer-1 "
              "read output (the injection channel). cos_to_prev_* are signed; "
              "|cos| is the component share."),
        reveals=rows)


def _cos_rows(a, b):
    """Mean cosine between two batches of vectors ``(N, V)``."""
    return ((a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(EPS)
            ).mean()


@torch.no_grad()
def collect_fused_channels(model: FusedModel, ids, targets, task: TaskSpec):
    """The fused arm's mechanistic receipts: the collision (the second copy's
    identity rewrite of the first copy's row), the read-weight channel at the
    probes on the two candidate answer rows (role-blind means their aggregate
    masses are equal), the mass on the merged X row, and the count of live
    identical-content row pairs (the lemma's write half says it is 0)."""
    model.eval()
    logits, results = model.forward(ids, collect=True)
    picks = probe_predictions(model, ids, task)
    correct = picks == targets
    copy1, copy2 = task.copy_positions
    reveal1, reveal2 = task.reveal_positions
    per_layer = []
    for layer, result in enumerate(results):
        schedule, weights = result.schedule, result.weights
        slot = schedule.slot[:, :, 0, 0]                       # (N, T)
        op = schedule.op[:, :, 0, 0]
        s1, s2 = slot[:, copy1], slot[:, copy2]
        rewrite = ((op[:, copy2] == CORRECT) | (op[:, copy2] == MERGE)) \
            & (s2 == s1)
        a1, a2 = slot[:, reveal1], slot[:, reveal2]
        answer_slots = torch.stack((a1, a2), dim=1)             # (N, 2)
        # weights is (N, T, heads, rank); head 0 throughout this probe
        probe_weights = weights[:, list(task.probe_positions), 0, :]  # (N, J, rank)
        true_index = torch.arange(task.n_probes, device=ids.device)
        mass_answer = probe_weights.gather(
            2, answer_slots.unsqueeze(2)).squeeze(2)           # (N, J)
        other_slots = answer_slots[:, [1, 0]]
        mass_other = probe_weights.gather(
            2, other_slots.unsqueeze(2)).squeeze(2)
        mass_merged_x = probe_weights.gather(2, s1.unsqueeze(1).unsqueeze(2)
                                             ).squeeze(2)      # (N, J)
        state_pre, live_pre = fused_prefix_state(model, ids, layer,
                                                 task.probe_positions[0])
        per_layer.append(dict(
            layer=layer,
            op_histogram=op_histogram(schedule),
            # the collision receipt
            identity_rewrite_frac=float(rewrite.float().mean()),
            copy2_op_counts={name: int((op[:, copy2] == code).sum())
                             for code, name in ((APPEND, "append"),
                                                (CORRECT, "correct"),
                                                (MERGE, "merge"),
                                                (POP, "pop"), (NONE, "none"))},
            copy_slots_equal_frac=float((s1 == s2).float().mean()),
            identical_live_row_pairs=_pairwise_identical_rows(state_pre,
                                                               live_pre),
            # the read-channel receipts at the probes
            answer_row_mass=dict(
                mean_true=float(mass_answer.mean()),
                mean_other=float(mass_other.mean()),
                mean_signed_gap=float((mass_answer - mass_other).mean()),
                frac_true_heavier=float((mass_answer > mass_other)
                                        .float().mean())),
            merged_x_row_mass=dict(mean=float(mass_merged_x.mean()),
                                    max=float(mass_merged_x.max())),
            accuracy_per_probe=[float(correct[:, j].float().mean())
                                for j in range(task.n_probes)]))
    return dict(arm="fused", per_layer=per_layer,
                value_acc=float(correct.float().mean()))


@torch.no_grad()
def collect_separated_channels(model: SeparatedModel, ids, targets,
                               task: TaskSpec):
    """The separated arm's receipts: the two copies' emitted keys differ
    MECHANICALLY (contexts differ), the copy rows' fate (distinct slots, live
    at the probe), the probe read mass on the role-matching copy vs the other
    (the context-addressing witness -- it must flip with the probe's role),
    and the probe mass on the true answer row vs the other candidate."""
    model.eval()
    logits, results = model.forward(ids, collect=True)
    picks = probe_predictions(model, ids, task)
    correct = picks == targets
    copy1, copy2 = task.copy_positions
    reveal1, reveal2 = task.reveal_positions
    per_layer = []
    for layer, channel in enumerate(results):
        # head 0 throughout this probe: squeeze the head axis off the channels
        slot, op = channel["slot"][:, :, 0], channel["op"][:, :, 0]   # (N, T)
        emitted = channel["emitted_keys"][:, :, 0, :]                  # (N, T, K)
        live_pre = channel["live_pre"][:, :, 0, :]                    # (N, T, rank)
        weights = channel["weights"][:, :, 0, :]                       # (N, T, rank)
        k1 = emitted[:, copy1]
        k2 = emitted[:, copy2]
        key_cos = (k1 * k2).sum(-1) / (k1.norm(dim=-1).clamp_min(EPS)
                                       * k2.norm(dim=-1).clamp_min(EPS))
        s1, s2 = slot[:, copy1], slot[:, copy2]
        probe_live = live_pre[:, task.probe_positions[0]]       # (N, rank)
        live1 = probe_live.gather(1, s1.unsqueeze(1)).squeeze(1)
        live2 = probe_live.gather(1, s2.unsqueeze(1)).squeeze(1)
        a1, a2 = slot[:, reveal1], slot[:, reveal2]
        answer_slots = torch.stack((a1, a2), dim=1)
        probe_weights = weights[:, list(task.probe_positions), :]
        mass_answer = probe_weights.gather(
            2, answer_slots.unsqueeze(2)).squeeze(2)
        mass_other = probe_weights.gather(
            2, answer_slots[:, [1, 0]].unsqueeze(2)).squeeze(2)
        # copy-1 carries role 0 (probe 0's), copy-2 role 1: the role-addressing
        # witness is mass(copy-1)-mass(copy-2) at probe 0 and its flip at probe 1.
        copy_slots = torch.stack((s1, s2), dim=1)
        mass_copy = probe_weights.gather(2, copy_slots.unsqueeze(2)).squeeze(2)
        flip0 = mass_copy[:, 0] - mass_copy[:, 1]
        flip1 = mass_copy[:, 1] - mass_copy[:, 0]
        per_layer.append(dict(
            layer=layer,
            op_histogram={name: int((op == code).sum())
                          for code, name in ((APPEND, "append"),
                                             (CORRECT, "correct"),
                                             (MERGE, "merge"), (POP, "pop"),
                                             (NONE, "none"))},
            # the key-channel receipts
            copy_key_cos=dict(mean=float(key_cos.mean()),
                              min=float(key_cos.min()),
                              max=float(key_cos.max()),
                              frac_below_tau_match=float(
                                  (key_cos < RoutingPolicy().tau_match)
                                  .float().mean())),
            copy_rows=dict(distinct_slot_frac=float((s1 != s2).float().mean()),
                           both_live_frac=float((live1 & live2
                                                 & (s1 != s2)).float().mean()),
                           copy2_append_frac=float(
                               (op[:, copy2] == APPEND).float().mean())),
            role_addressing=dict(flip_probe0_mean=float(flip0.mean()),
                                 flip_probe1_mean=float(flip1.mean()),
                                 frac_correctly_flipped=float(
                                     ((flip0 > 0) & (flip1 > 0)).float()
                                     .mean())),
            answer_row_mass=dict(
                mean_true=float(mass_answer.mean()),
                mean_other=float(mass_other.mean()),
                frac_true_heavier=float((mass_answer > mass_other)
                                        .float().mean())),
            accuracy_per_probe=[float(correct[:, j].float().mean())
                                for j in range(task.n_probes)]))
    return dict(arm="separated", per_layer=per_layer,
                value_acc=float(correct.float().mean()))


def collect_channels(model, ids, targets, task: TaskSpec):
    if isinstance(model, FusedModel):
        return collect_fused_channels(model, ids, targets, task)
    return collect_separated_channels(model, ids, targets, task)


# ------------------------------------------------------------------- one run
def run_one(task_name: str, arm: str, lr: float, seed: int, steps: int,
            batch: int, key_dim: int, device: str, eval_sets, depth: int = 1):
    """Train one (task, arm, lr, seed, depth) cell of the best-effort sweep and
    return its receipt and state dict.  Every cell shares the fixed eval set
    (seed 4999) so arms are compared on identical probes."""
    task = TASKS[task_name]
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = make_model(task_name, arm, key_dim, depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                                  weight_decay=0.01)
    rng = torch.Generator().manual_seed(seed)
    eval_ids, eval_targets = eval_sets
    curve = []
    started = time.monotonic()
    model.train()
    for step in range(1, steps + 1):
        ids, targets, _ = task.make_batch(rng, batch)
        loss = model.loss(ids.to(device), targets.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # the early curve is the depth-substitution timing receipt: every 50
        # steps for the first 500 (when the composite channel forms), then the
        # protocol's every 500
        if step % 500 == 0 or step == steps or (step <= 500 and step % 50 == 0):
            picks = probe_predictions(model, eval_ids[:1024], task)
            curve.append(dict(step=step, loss=float(loss.detach()),
                              value_acc=float(
                                  (picks == eval_targets[:1024])
                                  .float().mean())))
    walls = time.monotonic() - started
    metrics = evaluate(model, eval_ids, eval_targets, task)
    receipt_ids, receipt_targets = eval_ids[:512], eval_targets[:512]
    channels = collect_channels(model, receipt_ids, receipt_targets, task)
    receipt = dict(
        task=task_name, arm=arm, lr=lr, seed=seed, key_dim=key_dim, depth=depth,
        steps=steps, batch=batch, device=device, window=task.window,
        probe_positions=list(task.probe_positions), chance=CHANCE,
        policy=RoutingPolicy().record(), metrics=metrics,
        curve_tail=curve[-14:], walls_s=round(walls, 1), channels=channels,
        params=sum(p.numel() for p in model.parameters()),
        torch=torch.__version__)
    if arm == "fused" and depth >= 2:
        receipt["composite"] = composite_receipt(model, receipt_ids, task)
    state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    return receipt, state


def parse_floats(text):
    return [float(x) for x in text.split(",")]


def parse_ints(text):
    return [int(x) for x in text.split(",")]


def sweep(task_name: str, arm: str, lrs, seeds, steps: int, batch: int,
          key_dim: int, device: str, eval_sets, quick: bool = False,
          depth: int = 1):
    """The (lr, seed) grid for one (task, arm, depth); per-cell receipts, the
    best model per (task, arm, depth) kept, and the sweep summary receipt."""
    runs, best, best_state = [], None, None
    for lr in lrs:
        for seed in seeds:
            receipt, state = run_one(task_name, arm, lr, seed, steps, batch,
                                    key_dim, device, eval_sets, depth)
            if not quick:
                name = f"keysep_{task_name}_{arm}_d{depth}_lr{lr:.0e}_seed{seed}.json"
                (RECEIPTS / name).write_text(json.dumps(receipt, indent=1))
            runs.append({k: receipt[k] for k in
                         ("task", "arm", "lr", "seed", "key_dim", "depth",
                          "metrics")})
            if best is None or receipt["metrics"]["value_acc"] \
                    > best["metrics"]["value_acc"]:
                best, best_state = receipt, state
            print(f"{task_name} {arm} d{depth} lr={lr:.0e} seed={seed}: "
                  f"value_acc={receipt['metrics']['value_acc']:.4f} "
                  f"probes={receipt['metrics']['probe_acc']} "
                  f"[{receipt['walls_s']}s]", flush=True)
    summary = dict(task=task_name, arm=arm, key_dim=key_dim, depth=depth,
                   steps=steps, batch=batch, seeds=list(seeds), chance=CHANCE,
                   runs=runs,
                   best=dict(lr=best["lr"], seed=best["seed"],
                             metrics=best["metrics"]))
    if not quick:
        torch.save(best_state,
                   RECEIPTS / f"keysep_{task_name}_{arm}_d{depth}_best.pt")
        (RECEIPTS / f"keysep_{task_name}_{arm}_d{depth}_sweep.json").write_text(
            json.dumps(summary, indent=1))
    return summary, best, best_state


def keydim_sweep(task_name: str, lr: float, seeds, dims, steps: int,
                 batch: int, device: str, eval_sets, depth: int = 1):
    """Part 2: the address-size question -- accuracy vs key dimension on the
    separated arm (the necessity arm, depth 1), at part 1's best lr."""
    task = TASKS[task_name]
    runs = []
    for key_dim in dims:
        cell_accs = []
        for seed in seeds:
            receipt, _ = run_one(task_name, "separated", lr, seed, steps,
                                 batch, key_dim, device, eval_sets, depth)
            runs.append(dict(key_dim=key_dim, seed=seed,
                             metrics=receipt["metrics"]))
            cell_accs.append(receipt["metrics"]["value_acc"])
            print(f"{task_name} keydim={key_dim} seed={seed}: "
                  f"value_acc={receipt['metrics']['value_acc']:.4f}",
                  flush=True)
        print(f"{task_name} keydim={key_dim}: mean={sum(cell_accs) / len(cell_accs):.4f}",
              flush=True)
    curve = []
    for key_dim in dims:
        cells = [r["metrics"]["value_acc"] for r in runs
                  if r["key_dim"] == key_dim]
        curve.append(dict(key_dim=key_dim, mean_value_acc=sum(cells) / len(cells),
                          per_seed=cells))
    summary = dict(task=task_name, arm="separated", depth=depth, lr=lr,
                   steps=steps, batch=batch, seeds=list(seeds), chance=CHANCE,
                   runs=runs, curve=curve)
    (RECEIPTS / f"keysep_{task_name}_separated_d{depth}_keydim.json").write_text(
        json.dumps(summary, indent=1))
    return summary


def best_lr(task_name: str, arm: str, depth: int = 1):
    """The sweep's best lr for a (task, arm, depth), from its receipt."""
    path = RECEIPTS / f"keysep_{task_name}_{arm}_d{depth}_sweep.json"
    if not path.exists():
        raise FileNotFoundError(f"run the sweep first: {path} is missing")
    return float(json.loads(path.read_text())["best"]["lr"])


def build_eval_sets(task_name: str, windows: int, device: str):
    task = TASKS[task_name]
    eval_rng = torch.Generator().manual_seed(4999)
    ids, targets, _ = task.make_batch(eval_rng, windows)
    return ids.to(device), targets.to(device)


def main():
    parser = argparse.ArgumentParser(
        description="SEP-KV: the key-separation necessity probe")
    parser.add_argument("--part", choices=["1", "2", "3", "all"],
                        default="1",
                        help="1: the depth-1 necessity sweep; 2: the key-dim "
                             "sweep (separated, depth 1); 3: the depth-2 "
                             "depth-substitution corroboration (fused)")
    parser.add_argument("--task", choices=["m1", "m2", "both"], default="both")
    parser.add_argument("--arm", choices=["fused", "separated", "both"],
                        default="both")
    parser.add_argument("--lrs", default="3e-3,1e-3,3e-4,1e-4")
    parser.add_argument("--seeds", default="3401,3402,3403")
    parser.add_argument("--keydims", default="4,8,16,32,64")
    parser.add_argument("--keydim", type=int, default=DEFAULT_KEY_DIM)
    parser.add_argument("--keydim-lr", type=float, default=None,
                        help="part 2 lr; default: part 1's best per task")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--eval-windows", type=int, default=4096)
    parser.add_argument("--quick", action="store_true",
                        help="smoke: one lr, one seed, 200 steps, no receipts")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.quick:
        args.lrs, args.seeds, args.steps, args.eval_windows = \
            "1e-3", "3401", 200, 512
        args.task = "m1" if args.task == "both" else args.task
        args.arm = "separated" if args.arm == "both" else args.arm

    tasks = ["m1", "m2"] if args.task == "both" else [args.task]
    arms = ["fused", "separated"] if args.arm == "both" else [args.arm]
    parts = {"1": ("1",), "2": ("2",), "3": ("3",), "all": ("1", "2", "3")}[args.part]
    for task_name in tasks:
        eval_sets = build_eval_sets(task_name, args.eval_windows, args.device)
        if "1" in parts:
            for arm in arms:
                summary, _, _ = sweep(task_name, arm, parse_floats(args.lrs),
                                      parse_ints(args.seeds), args.steps,
                                      args.batch, args.keydim, args.device,
                                      eval_sets, args.quick, depth=1)
                if not args.quick:
                    print(json.dumps(summary["best"], indent=1))
        if "2" in parts and "separated" in arms and not args.quick:
            lr = args.keydim_lr or best_lr(task_name, "separated")
            summary = keydim_sweep(task_name, lr, parse_ints(args.seeds),
                                   parse_ints(args.keydims), args.steps,
                                   args.batch, args.device, eval_sets, depth=1)
            print(json.dumps(summary["curve"], indent=1))
        if "3" in parts and "fused" in arms and not args.quick:
            # the depth-substitution corroboration: the registered arm at
            # depth 2, at the fused arm's own best depth-1 lr, with the
            # composite-payload mechanism receipt per cell
            lr = args.keydim_lr or best_lr(task_name, "fused")
            summary, _, _ = sweep(task_name, "fused", [lr],
                                  parse_ints(args.seeds), args.steps,
                                  args.batch, args.keydim, args.device,
                                  eval_sets, depth=2)
            print(json.dumps(summary["best"], indent=1))


if __name__ == "__main__":
    main()
