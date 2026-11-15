"""K5B -- the routing region at WHOLE-SCAN granularity: one op per (layer, window).

Why this file exists
--------------------
K5A finished the launch-count program and *lost* wall time doing it: the fused
routing pair removed 83k compiled launches per step (208k -> 125k kernels) but
the step got slower (8.53 -> 9.08 s), because every custom-op boundary the
fused arms cross runs its Python body once per chunk -- K4's pair and K5A's
pair, four op bodies per chunk, ~24.6k op dispatches per step -- and the op
boundary itself now costs more host time than the launches it saves
(``K5A_ROUTING_DIFF.md`` gate D: 202+179 us/chunk for the routing pair, of
which ~230 us is the Triton launch binding path).  The K5A doc named the
direction and it is *not* more kernel fusion: remove the per-chunk Python.

The sequential constraint that forces the design
------------------------------------------------
Routing for chunk ``c`` consumes the chunk-*entry* state, which is chunk
``c-1``'s exit state -- the state carry is the only sequential axis (the same
constraint ``hostloop_free.py`` states, the runtime shadow of
``state_dependent_death_not_affine``).  So a whole-scan op cannot parallelise
over chunks; it must *walk* them, and the walk needs the decision-input
preprocessing (the norms, the stored division, the similarity bmm) at every
chunk.  The settled parity ruling (K5A's probe receipts: a Triton reproduction
of the bf16 norm/bmm chain matched only ~48% of similarity entries, 0/1024
payload norms) forbids moving that preprocessing into a persistent Triton
kernel -- decisions are argmaxes against thresholds, and the torch reduction
orders ARE the decision semantics.  So the "persistent whole-window kernel"
variant of the K5A doc's direction is blocked by the parity gate, and what
remains is the other variant the doc named: **capture the whole per-chunk
sequence into a CUDA graph inside one op per scan** -- the proven
``BlockGraph`` discipline (``hostloop_free.py`` Option B), extended from the
forward-only block call to the whole routing+scan region with its adjoint.

What this module builds
-----------------------
``x45::whole_scan_fwd`` / ``x45::whole_scan_bwd`` -- ONE opaque
``torch.library`` op pair per (layer, window) scan, the K4/K5A hosting pattern.
The forward impl walks the scan's chunks exactly as the per-chunk K5A arm does
-- the eager preprocessing (from detached inputs, verbatim in values), the
K5A prep/emit pair, the K4 persistent scan kernel -- and on the second call
for a slot the whole walk is captured into a CUDA graph with static input
buffers; every later call is three device copies plus one replay.  The
backward impl is the same story in reverse: the K4 recompute-adjoint and the
K5A routing adjoint per chunk, chained exactly as autograd chains them in the
per-chunk arm (``dstate_out`` of chunk ``c-1`` = ``dstate`` of chunk ``c`` +
``drows`` of chunk ``c``), captured once and replayed.  The kernels, their
order and their inputs are IDENTICAL to the per-chunk K5A arm's -- the op
boundary moved, nothing else -- so ``reads``, ``state``, ``dquery``,
``dpayload`` and ``ddecay`` are **bit-identical to that arm by construction**
(probe-verified before this module was written, and asserted by the gate
tests), and the discrete channel is bit-identical to the registered eager
``route_chunk`` by the same construction K5A receipted (the fused kernels
consume the preprocessing's own ``similarity``/``strength`` tensors; the
mandatory flip test runs at the registered shape, 8.65M lanes).

One consequence stated up front: the preprocessing inside the op runs EAGER
(its kernels are captured; the values are the eager router's own), so the
whole-scan arm's decisions are bit-equal to the registered *eager* router's
-- the flip-test bar -- while the registered *compiled* arm carries its own
inductor-vs-eager similarity delta (K5A's probe row: 130979/131072 entries
bit-equal), which predates this stage and is re-measured in the K5B receipt
so the two deltas cannot be confused.

Memory and lifetime (the x45 graph discipline, honoured)
--------------------------------------------------------
Every graph instance ("slot") owns its static input/output buffers and its
forward capture pool; a slot id is issued per ``MatrixMixer``
(``next_slot_id()``), so a slot's buffers are rewritten only by its own next
microbatch -- always after the previous microbatch's forward AND backward
have consumed them (the ``one_step`` protocol runs fwd(m), bwd(m), fwd(m+1),
... strictly interleaved).  That is the output-lifetime rule that
``torch.compile(mode="reduce-overhead")`` broke at 24 shared modules
(``RESULT_MATRIX_EMENDER.md``); here it is enforced by construction.  The
forward saves the chunk-entry checkpoints in its capture pool (the same bytes
the per-chunk arm's autograd graph keeps alive, ~86 MB per scan at the
registered shape); the backward graphs share ONE pool handle (they replay
strictly sequentially and their intermediates -- the recompute trajectory,
the adjoint partials -- die inside each own replay).

Refusals (by name, to the per-chunk K5A+K4 stack)
--------------------------------------------------
``policy.ops=False`` (the append-only control), ``policy.grow`` (the rank
moves at a chunk boundary), ``rank > K5A_MAX_RANK`` (K5A's register-tile
budget), a store that does not fit the persistent kernel
(``persistent_available``), non-CUDA inputs, explicit ``initial_state``/
``initial_live``, or the slot registry overflow all take
``routed_chunk_scan(router=routing_diff_route_chunk, trainable_fused=True)``
-- the per-chunk stack this module subsumes, whose own refusals then apply
unchanged.  The eager body (first call per slot, and the overflow path) is
the same walk without the graph: still ONE op dispatch per scan.

Determinism: no atomics anywhere in the walk (the K5A pair and the K4 scan
are both atomics-free by receipt), cuBLAS is run-to-run deterministic at
fixed shapes, and a replay is the same kernels on the same buffers -- the
gate tests assert two replays are bit-identical.
"""
from __future__ import annotations

import contextlib
import threading
from typing import Tuple

import torch
import triton

from .algebra import MatrixSchedule
from .hostloop_free import persistent_available, value_block, warp_count
from .k4_fused_backward import (_persistent_scan_forward,
                                persistent_scan_recompute_backward)
from .k5a_routing_diff import (K5A_MAX_RANK, _routing_diff_bwd_op,
                               _routing_diff_fwd_op,
                               routing_diff_route_chunk)
from .scan import (DEFAULT_CHUNK_TOKENS, RoutedResult, ScanStats,
                   empty_schedule, routed_chunk_scan, _write_chunk)
from .schedule import EPS, RoutingPolicy

# One graph instance per MatrixMixer keeps a slot's buffers unwritten from a
# scan's forward to its own backward; more slots than this would mean more
# distinct (slot, shape) keys than the model can justify, so new keys run the
# eager body instead of capturing graphs forever (correct, one dispatch).
MAX_SLOT_ENTRIES = 192
# The backward graphs share one pool: they replay strictly sequentially (the
# autograd engine unwinds the mixer stack in order, one thread per device)
# and their intermediates die inside each own replay before the next begins.
_BWD_POOL = [None]
_BWD_POOL_LOCK = threading.Lock()

# The autocast regime, encoded explicitly: ``torch.compile`` runs a custom-op
# impl with ambient autocast DISABLED (inductor bakes the casts into the
# traced graph; an opaque impl never sees them), so the wrapper captures the
# caller's regime -- dynamo guards the autocast state, so the traced value is
# the caller's -- and the op runs its body under exactly that context, eager
# or compiled, and the slot key carries it instead of an ambient read.
_CAST_CODES = {None: 0, torch.bfloat16: 1, torch.float16: 2, torch.float32: 3}
_CAST_DTYPES = {code: dtype for dtype, code in _CAST_CODES.items() if code}


def _ambient_cast_code() -> int:
    """The caller's CUDA autocast regime (traced as a constant under compile)."""
    if not torch.is_autocast_enabled("cuda"):
        return 0
    return _CAST_CODES.get(torch.get_autocast_dtype("cuda"), 0)


def _autocast_context(cast_code: int):
    if not cast_code:
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=_CAST_DTYPES[int(cast_code)])

# The op-dispatch ledger the step receipt reads: per-scan host cost is the
# thing this stage moves, so the counts are first-class evidence.  Reset only
# together with the slot registry (:func:`reset_slots`).
DISPATCH = {
    "fwd_calls": 0,           # x45::whole_scan_fwd impls entered
    "bwd_calls": 0,          # x45::whole_scan_bwd impls entered
    "fwd_graph_replays": 0,  # of which served by a CUDA-graph replay
    "bwd_graph_replays": 0,
    "fwd_eager": 0,          # of which ran the body without a graph
    "bwd_eager": 0,
    "fwd_captures": 0,       # whole-scan graphs captured (one-time cost)
    "bwd_captures": 0,
    "fallbacks": 0,          # whole_scan() refusals to the per-chunk stack
}

_SLOTS: dict = {}
_SLOTS_BY_TOKEN: dict = {}
_SLOT_LOCK = threading.Lock()
_SLOT_SEQUENCE = [0]
_SLOT_TOKEN_SEQUENCE = [0]


def next_slot_id() -> int:
    """A stable graph-slot id for one ``MatrixMixer`` (its buffers' owner)."""
    with _SLOT_LOCK:
        _SLOT_SEQUENCE[0] += 1
        return _SLOT_SEQUENCE[0]


def reset_slots() -> dict:
    """Drop every graph, static buffer and ledger entry (tests only)."""
    global _BWD_POOL
    with _SLOT_LOCK:
        _SLOTS.clear()
        _SLOTS_BY_TOKEN.clear()
    with _BWD_POOL_LOCK:
        _BWD_POOL = [None]
    ledger = dict(DISPATCH)
    for key in DISPATCH:
        DISPATCH[key] = 0
    return ledger


def _bwd_pool():
    with _BWD_POOL_LOCK:
        if _BWD_POOL[0] is None:
            _BWD_POOL[0] = torch.cuda.graph_pool_handle()
        return _BWD_POOL[0]


# ------------------------------------------------------------------ the bodies


def _fwd_body(query, payload, decay, policy: RoutingPolicy, rank: int,
              rank_max: int, chunk_tokens: int):
    """The whole scan's forward walk: routing + scan per chunk, no autograd.

    The chunk sequence is the per-chunk K5A arm's, verbatim in kernels and in
    order: the eager preprocessing from detached inputs (its reduction orders
    are the decision semantics), the K5A prep/emit pair consuming the
    preprocessing's own similarity/strength, and the K4 persistent scan kernel
    over the emitted schedule.  Returns ``(reads, state, live, saves)`` with
    ``reads`` ``(batch, tokens, heads, d_v)`` fp32 and ``saves`` the per-chunk
    checkpoint set the adjoint walk needs (the same tensors the per-chunk
    arm's autograd graph keeps alive).
    """
    batch, tokens, heads, value_dim = query.shape
    device = query.device
    state = torch.zeros(batch, heads, rank_max, value_dim, dtype=torch.float32,
                        device=device)
    live = torch.zeros(batch, heads, rank_max, dtype=torch.bool, device=device)
    reads = torch.empty(batch, tokens, heads, value_dim, dtype=torch.float32,
                        device=device)
    block = value_block(rank_max, value_dim)
    warps = warp_count(triton.next_power_of_2(rank_max) * block)
    saves = []
    for start in range(0, tokens, chunk_tokens):
        stop = min(start + chunk_tokens, tokens)
        qc = query[:, start:stop].permute(0, 2, 1, 3)
        pc = payload[:, start:stop].permute(0, 2, 1, 3)
        dc = decay[:, start:stop].permute(0, 2, 1).to(torch.float32)
        entry_state, entry_live = state, live
        # The eager router's own preprocessing, verbatim in values and built
        # from detached inputs -- the K5A ruling: the reduction orders are the
        # decision semantics, and the branch is grad-dead exactly as in the
        # registered path.
        rows = entry_state[:, :, :rank]
        active = entry_live[:, :, :rank]
        frozen_rows = rows.detach()
        strength = frozen_rows.norm(dim=-1) * active.to(torch.float32)
        stored = frozen_rows / strength.clamp_min(EPS).unsqueeze(-1)
        payload_value = pc.detach()
        value = payload_value / payload_value.norm(dim=-1,
                                                    keepdim=True).clamp_min(EPS)
        similarity = value @ stored.transpose(-1, -2)
        slot, op, erase, live_after, payload_field, weights, scratch = \
            _routing_diff_fwd_op(
                qc, pc, rows, active, similarity, strength, rank_max,
                policy.tau_match, policy.tau_merge, policy.tau_pop,
                policy.tie_break, policy.read_scale)
        schedule = MatrixSchedule(
            slot=slot.permute(0, 2, 1, 3),
            erase=erase.permute(0, 2, 1, 3),
            payload=payload_field.permute(0, 2, 1, 3, 4),
            op=op.permute(0, 2, 1, 3),
            live_after=live_after.permute(0, 2, 1, 3))
        chunk_reads, state, live = _persistent_scan_forward(
            entry_state, entry_live, weights, dc, schedule, block, warps, 3)
        reads[:, start:stop] = chunk_reads.permute(0, 2, 1, 3)
        saves.append(dict(state=entry_state, live=entry_live, weights=weights,
                          dc=dc, schedule=schedule, strength=strength,
                          scratch=scratch))
    return reads, state, live, saves


def _bwd_body(query, payload, decay, saves, dreads, policy: RoutingPolicy,
              rank: int, chunk_tokens: int):
    """The whole scan's adjoint in one reverse walk.

    Per chunk, in reverse: the K4 recompute-adjoint (chunk-entry checkpoint
    in, ``dstate``/``dweights``/``ddecay``/``dpayload`` out) then the K5A
    routing adjoint (consuming the scan's ``dweights`` and schedule-payload
    gradients), with the state chain carried exactly as autograd carries it
    in the per-chunk arm: ``dstate_out(c-1) = dstate(c) + drows(c)``.
    ``dreads`` is head-major ``(batch, heads, tokens, d_v)`` fp32.  Returns
    the gradients of the op's three differentiable inputs in their layouts.
    """
    batch, tokens, heads, value_dim = query.shape
    device = query.device
    dquery = torch.empty_like(query)
    dpayload = torch.empty_like(payload)
    ddecay = torch.zeros(batch, tokens, heads, dtype=torch.float32,
                         device=device)
    carry = torch.zeros(batch, heads, saves[0]["state"].shape[2], value_dim,
                        dtype=torch.float32, device=device)
    for index in reversed(range(len(saves))):
        start = index * chunk_tokens
        stop = min(start + chunk_tokens, tokens)
        piece = saves[index]
        dstate, dweights, ddecay_chunk, dpay_field = \
            persistent_scan_recompute_backward(
                piece["state"], piece["live"], piece["weights"], piece["dc"],
                piece["schedule"], dreads[:, :, start:stop], carry)
        dquery_chunk, dpayload_chunk, drows = _routing_diff_bwd_op(
            query[:, start:stop].permute(0, 2, 1, 3),
            payload[:, start:stop].permute(0, 2, 1, 3),
            piece["state"][:, :, :rank], piece["live"][:, :, :rank],
            piece["strength"], piece["scratch"], dweights,
            dpay_field.permute(0, 2, 1, 3, 4), policy.read_scale)
        if dstate.shape[2] == drows.shape[2]:
            carry = dstate + drows
        else:
            # a rank prefix: the routing adjoint addresses the active rows
            # only, and the state gradient above the active rank passes
            # through untouched -- autograd's view-backward plus accumulate,
            # written as the same add on the prefix.
            carry = dstate
            carry[:, :, :rank] += drows
        dquery[:, start:stop] = dquery_chunk.permute(0, 2, 1, 3)
        dpayload[:, start:stop] = dpayload_chunk.permute(0, 2, 1, 3)
        ddecay[:, start:stop, :] = ddecay_chunk.permute(0, 2, 1)
    return dquery, dpayload, ddecay


# ------------------------------------------------------------- the graph slots


class _ScanSlot:
    """One (mixer, shape) graph instance: static buffers, two graphs, saves.

    Lifetime contract (enforced by the interleaved training protocol): the
    statics and the forward pool's saves are rewritten only by this slot's
    next forward, which the protocol schedules after the backward that
    consumes them.
    """

    def __init__(self, key, token, query, payload, decay, rank_max,
                 graph_enabled):
        self.key = key
        self.token = token
        self.graph_enabled = graph_enabled
        self.fwd_graph = None
        self.bwd_graph = None
        self.fwd_calls = 0
        self.query_static = torch.zeros_like(query)
        self.payload_static = torch.zeros_like(payload)
        self.decay_static = torch.zeros_like(decay)
        self.dreads_static = torch.zeros(
            query.shape[0], query.shape[2], query.shape[1], query.shape[3],
            dtype=torch.float32, device=query.device)
        self.last_inputs = None    # (query, payload, decay) of the last forward
        self.last_saves = None     # the checkpoints the last forward produced
        self.fwd_outputs = None    # (reads, state, live) statics, post-capture
        self.bwd_outputs = None    # (dquery, dpayload, ddecay) statics


def _slot_key(query, payload, decay, policy, rank, rank_max, chunk_tokens,
              slot_id, cast_code):
    # The effective autocast regime (an explicit argument, not an ambient
    # read) is part of the key: the captured preprocessing's dtypes are baked
    # into the graph.
    return (int(slot_id), tuple(query.shape), query.dtype, payload.dtype,
            decay.dtype, query.device.index, int(rank), int(rank_max),
            int(chunk_tokens), policy, int(cast_code))


def _slot_for(key, query, payload, decay, rank_max):
    with _SLOT_LOCK:
        slot = _SLOTS.get(key)
        if slot is None:
            _SLOT_TOKEN_SEQUENCE[0] += 1
            slot = _ScanSlot(key, _SLOT_TOKEN_SEQUENCE[0], query, payload,
                             decay, rank_max,
                             graph_enabled=len(_SLOTS) < MAX_SLOT_ENTRIES)
            _SLOTS[key] = slot
            _SLOTS_BY_TOKEN[slot.token] = slot
        return slot


def _warm_capture(body, pool=None):
    """The BlockGraph discipline: eager warmups, side-stream warmups, capture.

    ``body`` is a zero-argument callable returning its outputs; the graph, once
    replayed, reproduces the body's kernels on the very buffers the captured
    run allocated.  Everything is warmed outside the capture (the Triton
    kernels compile, cuBLAS fills its workspaces) and the device is
    synchronised before and after the side-stream warmups, so the capture
    never races pending work.
    """
    for _ in range(2):
        body()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            body()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    if pool is None:
        context = torch.cuda.graph(graph)
    else:
        context = torch.cuda.graph(graph, pool=pool)
    with context:
        outputs = body()
    return graph, outputs


# ----------------------------------------------------------------- the op pair


@torch.library.custom_op("x45::whole_scan_fwd", mutates_args=())
def _whole_scan_fwd_op(
    query: torch.Tensor, payload: torch.Tensor, decay: torch.Tensor,
    rank: int, rank_max: int, chunk_tokens: int, tau_match: float,
    tau_merge: float, tau_pop: float, tie_break: float, scale: float,
    cast_code: int, slot_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """The whole (layer, window) routing+scan region as ONE opaque op.

    First call per slot: the eager walk (still one dispatch).  Second call:
    the walk is captured into a CUDA graph and replayed.  Later calls: three
    input copies and one replay.  The kernels are the per-chunk K5A+K4
    stack's, so the outputs are bit-identical to that arm by construction.
    ``cast_code`` is the caller's autocast regime passed EXPLICITLY
    (``torch.compile`` runs this impl with ambient autocast disabled, and the
    preprocessing's dtypes are the decision semantics).  The fourth output
    is the slot TOKEN the backward must use -- the slot key contains the
    regime, which the backward cannot rebuild, so the token travels through
    the graph instead of a key.
    """
    policy = RoutingPolicy(ops=True, tau_match=tau_match, tau_merge=tau_merge,
                           tau_pop=tau_pop, tie_break=tie_break,
                           read_scale=scale)
    key = _slot_key(query, payload, decay, policy, rank, rank_max,
                    chunk_tokens, slot_id, cast_code)
    slot = _slot_for(key, query, payload, decay, rank_max)
    slot.fwd_calls += 1
    DISPATCH["fwd_calls"] += 1
    if slot.fwd_graph is not None:
        slot.query_static.copy_(query)
        slot.payload_static.copy_(payload)
        slot.decay_static.copy_(decay)
        slot.fwd_graph.replay()
        DISPATCH["fwd_graph_replays"] += 1
        return (*slot.fwd_outputs, slot.token)

    def body():
        with torch.no_grad(), _autocast_context(cast_code):
            return _fwd_body(slot.query_static, slot.payload_static,
                             slot.decay_static, policy, rank, rank_max,
                             chunk_tokens)

    if slot.fwd_calls >= 2 and slot.graph_enabled:
        slot.query_static.copy_(query)
        slot.payload_static.copy_(payload)
        slot.decay_static.copy_(decay)
        slot.fwd_graph, outputs = _warm_capture(body)
        reads, state, live, saves = outputs
        slot.fwd_outputs = (reads, state, live)
        slot.last_inputs = (slot.query_static, slot.payload_static,
                            slot.decay_static)
        slot.last_saves = saves
        slot.fwd_graph.replay()
        DISPATCH["fwd_captures"] += 1
        DISPATCH["fwd_graph_replays"] += 1
        return (*slot.fwd_outputs, slot.token)

    DISPATCH["fwd_eager"] += 1
    with torch.no_grad(), _autocast_context(cast_code):
        reads, state, live, saves = _fwd_body(query, payload, decay, policy,
                                             rank, rank_max, chunk_tokens)
    slot.last_inputs = (query, payload, decay)
    slot.last_saves = saves
    return reads, state, live, slot.token


@_whole_scan_fwd_op.register_fake
def _(query, payload, decay, rank, rank_max, chunk_tokens, tau_match,
      tau_merge, tau_pop, tie_break, scale, cast_code, slot_id):
    batch, tokens, heads, value_dim = query.shape
    device = query.device
    return (torch.empty((batch, tokens, heads, value_dim), dtype=torch.float32,
                        device=device),
            torch.empty((batch, heads, rank_max, value_dim),
                        dtype=torch.float32, device=device),
            torch.empty((batch, heads, rank_max), dtype=torch.bool,
                        device=device),
            int(slot_id))


@torch.library.custom_op("x45::whole_scan_bwd", mutates_args=())
def _whole_scan_bwd_op(
    query: torch.Tensor, payload: torch.Tensor, decay: torch.Tensor,
    dreads: torch.Tensor, rank: int, chunk_tokens: int, scale: float,
    slot_token: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The whole scan's adjoint as its own opaque op (the K4/K5A hosting
    pattern): the K4 recompute-adjoint and the K5A routing adjoint per chunk,
    reverse-walked in one captured graph.  ``query``/``payload``/``decay``
    arrive only to pin shapes/dtypes and keep the forward's inputs alive; the
    walk reads the slot's statics (or, before the forward was captured, the
    eager forward's own tensors).  The slot arrives as the token the
    forward's ``setup_context`` resolved -- the backward cannot rebuild the
    forward's key, because the ambient autocast state is part of that key and
    autograd runs the backward outside it."""
    policy = RoutingPolicy(ops=True, read_scale=scale)
    with _SLOT_LOCK:
        slot = _SLOTS_BY_TOKEN.get(int(slot_token))
    if slot is None or slot.last_saves is None:
        raise RuntimeError("x45::whole_scan_bwd ran before its forward")
    DISPATCH["bwd_calls"] += 1
    # the incoming gradient arrives in the reads' (batch, tokens, heads, d_v)
    # layout; the walk wants head-major.
    if slot.bwd_graph is not None:
        slot.dreads_static.copy_(dreads.permute(0, 2, 1, 3))
        slot.bwd_graph.replay()
        DISPATCH["bwd_graph_replays"] += 1
        return slot.bwd_outputs

    if slot.fwd_graph is None or not slot.graph_enabled:
        # the forward ran eagerly (first call, or registry overflow): the
        # adjoint runs eagerly against that forward's own tensors.
        DISPATCH["bwd_eager"] += 1
        with torch.no_grad():
            return _bwd_body(*slot.last_inputs, slot.last_saves,
                             dreads.permute(0, 2, 1, 3), policy, rank,
                             chunk_tokens)

    slot.dreads_static.copy_(dreads.permute(0, 2, 1, 3))

    def body():
        with torch.no_grad():
            return _bwd_body(slot.query_static, slot.payload_static,
                             slot.decay_static, slot.last_saves,
                             slot.dreads_static, policy, rank, chunk_tokens)

    slot.bwd_graph, slot.bwd_outputs = _warm_capture(body, pool=_bwd_pool())
    slot.bwd_graph.replay()
    DISPATCH["bwd_captures"] += 1
    DISPATCH["bwd_graph_replays"] += 1
    return slot.bwd_outputs


@_whole_scan_bwd_op.register_fake
def _(query, payload, decay, dreads, rank, chunk_tokens, scale, slot_token):
    return (torch.empty_like(query), torch.empty_like(payload),
            torch.empty_like(decay))


def _whole_scan_setup(ctx, inputs, output):
    query, payload, decay, rank, rank_max, chunk_tokens, tau_match, \
        tau_merge, tau_pop, tie_break, scale, cast_code, slot_id = inputs
    # The slot token is the forward's own fourth output: the backward cannot
    # rebuild the slot key (it contains the forward's ambient autocast state,
    # and autograd runs the backward outside it), so the token travels through
    # the graph instead.  Under fake propagation the token is the placeholder
    # the register_fake returns, and the traced backward never executes.
    ctx.save_for_backward(query, payload, decay)
    ctx.rank, ctx.chunk_tokens = int(rank), int(chunk_tokens)
    ctx.scale = float(scale)
    ctx.slot_token = int(output[-1])


def _whole_scan_backward(ctx, *grads):
    """The adjoint of the whole-scan op: dquery, dpayload and ddecay.

    ``state``'s gradient is ignored (the mixer consumes only ``reads``; the
    window-entry state is the op's own zeros), ``live`` is a bool channel
    with no gradient by construction, and the token output carries none.
    """
    dreads = grads[0]
    query, payload, decay = ctx.saved_tensors
    batch, tokens, heads, value_dim = query.shape
    if dreads is None:
        dreads = torch.zeros(batch, tokens, heads, value_dim,
                             dtype=torch.float32, device=query.device)
    dquery, dpayload, ddecay = _whole_scan_bwd_op(
        query, payload, decay, dreads, ctx.rank, ctx.chunk_tokens, ctx.scale,
        ctx.slot_token)
    return dquery, dpayload, ddecay, None, None, None, None, None, None, \
        None, None, None, None


torch.library.register_autograd("x45::whole_scan_fwd", _whole_scan_backward,
                                setup_context=_whole_scan_setup)


# ------------------------------------------------------------------ the wrapper


def whole_scan(query, payload, decay, *, policy: RoutingPolicy, rank=None,
               rank_max=None, chunk_tokens=None, slot_id: int = 0,
               initial_state=None, initial_live=None) -> RoutedResult:
    """One whole (layer, window) scan: routing + scan + adjoint, ONE op call.

    The drop-in for ``scan.routed_chunk_scan`` at the mixer's boundary, with
    the whole per-chunk K5A+K4 stack walked (and then graph-captured) inside a
    single opaque op per scan -- the per-chunk op dispatch removed.  Refusals
    (append-only control, growth, ``rank > K5A_MAX_RANK``, a store the
    persistent kernel cannot hold, non-CUDA inputs, explicit initial state)
    take the per-chunk stack verbatim, so the flag is a host-cost knob and
    not a semantics knob.
    """
    batch, tokens, heads, value_dim = payload.shape
    chunk_tokens = DEFAULT_CHUNK_TOKENS if chunk_tokens is None \
        else int(chunk_tokens)
    if rank_max is None:
        rank_max = int(value_dim if rank is None else rank)
    rank_max = int(rank_max)
    rank_now = rank_max if rank is None else int(rank)
    # The caller's autocast regime, captured HERE (dynamo guards the state,
    # so under torch.compile this is the traced constant the op needs:
    # inductor runs a custom-op impl with ambient autocast disabled, and the
    # preprocessing's dtypes are the decision semantics).
    cast_code = _ambient_cast_code()
    if (not policy.ops or policy.grow or not query.is_cuda
            or initial_state is not None or initial_live is not None
            or rank_now > K5A_MAX_RANK
            or not persistent_available(rank_max, value_dim)
            or chunk_tokens < 1 or tokens < 1
            or query.shape != payload.shape):
        DISPATCH["fallbacks"] += 1
        # the per-chunk stack runs under the SAME explicit regime the op
        # would have -- the fallback must not silently change dtype either
        with _autocast_context(cast_code):
            return routed_chunk_scan(
                query, payload, decay, router=routing_diff_route_chunk,
                policy=policy, rank=rank, rank_max=rank_max,
                chunk_tokens=chunk_tokens, initial_state=initial_state,
                initial_live=initial_live, fused=False, trainable_fused=True)
    reads, state, live, _token = _whole_scan_fwd_op(
        query, payload, decay, rank_now, rank_max, chunk_tokens,
        policy.tau_match, policy.tau_merge, policy.tau_pop, policy.tie_break,
        policy.read_scale, int(cast_code), int(slot_id))
    return RoutedResult(reads, state, live, rank_now, ScanStats(), None, None)


def whole_scan_record(query, payload, decay, *, policy: RoutingPolicy,
                      rank=None, rank_max=None, chunk_tokens=None
                      ) -> RoutedResult:
    """The eager walk with the recorded decision channel (the gate instrument).

    The graph path's schedule is internal by design (recording every chunk's
    fields would be per-chunk host work on the measured path); this entry runs
    the same eager body once and assembles exactly what
    ``routed_chunk_scan(..., collect=True)`` assembles, so the parity gates
    can compare the whole-scan path's discrete channel field by field.
    """
    batch, tokens, heads, value_dim = payload.shape
    chunk_tokens = DEFAULT_CHUNK_TOKENS if chunk_tokens is None \
        else int(chunk_tokens)
    if rank_max is None:
        rank_max = int(value_dim if rank is None else rank)
    rank_max = int(rank_max)
    rank_now = rank_max if rank is None else int(rank)
    with torch.no_grad(), _autocast_context(_ambient_cast_code()):
        reads, state, live, saves = _fwd_body(query, payload, decay, policy,
                                              rank_now, rank_max, chunk_tokens)
    lanes = saves[0]["schedule"].lanes if saves else 1
    schedule_out = empty_schedule(batch, tokens, heads, value_dim, lanes,
                                  device=query.device)
    weights_out = torch.zeros(batch, tokens, heads, rank_max,
                              dtype=torch.float32, device=query.device)
    for index, start in enumerate(range(0, tokens, chunk_tokens)):
        stop = min(start + chunk_tokens, tokens)
        _write_chunk(schedule_out, start, stop, saves[index]["schedule"])
        weights_out[:, start:stop] = saves[index]["weights"].permute(0, 2, 1, 3)
    return RoutedResult(reads, state, live, rank_now, ScanStats(),
                        schedule_out, weights_out)


_COMPILE_CACHE: dict = {}


def compiled_whole_scan(mode: str = "default"):
    """The whole-scan wrapper, compiled once and shared (the ``compiled_scan``
    pattern): dynamo traces the wrapper, hosts the op pair as opaque calls and
    traces the registered backward into the joint graph, exactly as it hosts
    the K4/K5A pairs.

    ``dynamic=True`` on purpose: the one thing that varies across the model's
    24 mixers is the ``slot_id`` (each mixer owns its graph buffers), and a
    static compile would specialize on it -- 24 near-identical one-op graphs
    and a recompile-limit fallback.  One symbolic-int graph serves every
    mixer; the op stays opaque and the slot id flows through as a runtime
    argument."""
    if mode not in _COMPILE_CACHE:
        _COMPILE_CACHE[mode] = torch.compile(whole_scan, mode=mode, dynamic=True)
    return _COMPILE_CACHE[mode]


__all__ = ["DEFAULT_CHUNK_TOKENS", "DISPATCH", "MAX_SLOT_ENTRIES",
           "compiled_whole_scan", "next_slot_id", "reset_slots",
           "whole_scan", "whole_scan_record"]
