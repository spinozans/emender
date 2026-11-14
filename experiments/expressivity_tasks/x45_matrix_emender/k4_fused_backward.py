"""K4 -- the fused backward of the matrix store's block scan: the reverse walk.

Why this file exists
--------------------
Every kernel lane this far (K1, K2+K3, the persistent block scan of
``hostloop_free.py``) is **forward only**: ``check_no_grad`` refuses a graph
through them because the walk has no backward, so the registered arm keeps
training on the eager autograd path -- 1 392 launches for a 2048-token block
call (``NO_HOST_LOOP.md`` §3) -- while the tape arm's profile shows where that
class of cost lives (78 % of its step in checkpoint *replay*,
``PREREGISTRATION_MATRIX_EMENDER.md``).  This stage gives the persistent block
scan its backward, so the one-launch forward becomes trainable and the replay
the matrix arm pays for -- autograd through an eager per-chunk pipeline --
is replaced by one reverse-walk kernel.

Doctrine, unchanged
--------------------
* **Management decisions are constants in the backward.**  ``slot``, ``op``,
  ``erase`` and ``live_after`` are the recorded decision channel; gradients flow
  only through the continuous arithmetic: the read weights, the DECAY gate, the
  payloads and the state.  There are no decision logits, so there is no gradient
  through routing -- the adjoint is *linear in dS given the recorded schedule*,
  which is exactly what the Lean algebra's scan-compatibility proved.
* **The eager per-token reference under PyTorch autograd is the oracle.**  The
  fused path lands behind a flag (``trainable_block_scan(fused=True)`` /
  ``fused_block_scan``); the eager path stays the default.

The adjoint of the walk
-----------------------
Forward, per token ``t``: read against the state as it stands, then DECAY, then
every lane in order (``S[j] = erase ? p : S[j] + p``).  Reverse, per token
(lanes last-first):

======================  =========================================================
forward operation        adjoint, given the recorded schedule
======================  =========================================================
lane write, APPEND       ``dpayload[t,l] = dS[j]``; ``dS[j]`` passes through
lane write, erase        ``dpayload[t,l] = dS[j]``; ``dS[j] := 0`` (the incoming
                        content is discarded, so its gradient terminates here --
                        a POP's row is dead to everything before its death token)
DECAY ``S <- lam S``     ``dlam_t = <dS, S_t>`` (needs the *pre-decay* state);
                        ``dS <- lam * dS``
read ``o_t = w_t S_t``   ``dS[j] += w_t[j] dout_t`` and
                        ``dw_t[j] = dout_t . S_t[j]`` (needs ``S_t`` again)
======================  =========================================================

Every lane of a MERGE is one row of that table (lane 0 is an erasing write whose
payload is the blend -- its gradient reaches both source rows *through the
payload*, at the router's boundary, with detached blend weights; lane 1 is a
POP).  Inactive lanes are the identity: the registered eager path masks their
payload (``scan.chunk_scan`` multiplies by ``op != NONE``), so their payload
gradient is exactly zero here too.

THE design decision: the state trajectory
-----------------------------------------
Both ``dlam_t`` and ``dw_t`` need the per-token pre-decay state ``S_t``, so the
backward must have it.  The options, in the order they were considered:

1. **Store the trajectory in the forward** -- ``T x rank x d_v`` fp32
   (``persistent_scan_trajectory``).  Simple, exact, and the honest first
   choice; its memory cost is reported as a first-class number (gate C: 2.15 GB
   = 5.0x the block's every-other-tensor total at the bench shape).  Note the
   eager autograd baseline saves the same order of bytes as activations for its
   per-token multiplies, so this is not a new cost class -- it is the same
   trajectory, materialised once by a kernel instead of 2048 times by autograd.
2. Chunk-boundary checkpoints + in-kernel recompute -- not built: option 1
   measured fine (gate D), and recompute would re-walk the block the backward
   is already walking.
3. An analytical WY-style adjoint that avoids the trajectory (the GDN-2 trick)
   -- deliberately not attempted: that is where ``chunk_gdn2.py``'s ~900 lines
   live, and GDN-2 can do it only because its write needs no selection.

Structure
---------
* one *forward* kernel -- the persistent scan of ``hostloop_free.py`` plus a
  trajectory store of the register tile (the walk is the same walk, same
  ``mul.rn``/``add.rn`` asm, so state, liveness and reads are bit-identical;
  tested);
* one *backward* kernel on the same grid ``(rows*heads, d_v/BV)``, walking
  tokens in reverse with the adjoint tile in registers.  The value axis is the
  only axis blocked, as in the forward; the two adjoint reductions that run
  *along* the value axis (``dw_t[j] = sum_v dout . S_t[j]`` and ``dlam_t``)
  cross value blocks, so the kernel writes per-block partials and torch sums
  them over the block axis -- a fixed order, no atomics, bit-deterministic.
  Three device launches per backward: the walk, and two deterministic
  reductions;
* ``fused_block_scan`` -- the whole block call as one ``torch.autograd``
  boundary (forward + backward), and ``trainable_block_scan(fused=True)`` --
  the flag the arm would flip, with the same size gate as the forward
  (``persistent_available``: rank 2048 is refused by name and the eager
  autograd path is the fallback).

The backward is first-order: the adjoint walk is not itself differentiable
(no ``create_graph``), which matches every kernel in this lane and is stated
here so nobody has to discover it from a traceback.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .algebra import MatrixSchedule
from .hostloop_free import (MIN_VALUE_BLOCK, check_block_shapes, persistent_available,
                            value_block, warp_count)
from .k1_schedule_scan import NONE_CODE, _add_rn, _mul_rn
from .scan import chunk_scan

# ------------------------------------------------------------- the forward, + trajectory


@triton.jit(do_not_specialize=['T', 'R'])
def _persistent_scan_trajectory_kernel(
    state_in, live_in, weights, decay,
    slot, op, erase, live_after, payload,
    reads, state_out, live_out, trajectory,
    T, R,
    wbs, whs, wt, wr,
    dbs, dhs, dt,
    sbs, st, sh, sl,
    pbs, pst, psh, psl,
    obs, ohs, ot, ov,
    H: tl.constexpr, L: tl.constexpr, V: tl.constexpr,
    RP: tl.constexpr, BV: tl.constexpr, STAGES: tl.constexpr,
):
    """``hostloop_free._persistent_scan_kernel`` plus one trajectory store per token.

    The walk is the forward's walk verbatim (read, DECAY, every lane in order,
    ``mul.rn``/``add.rn`` so no FMA contraction can move it), and the extra
    ``tl.store`` writes the register tile the read has just consumed -- the
    pre-decay state ``S_t``, the one quantity the backward's DECAY and
    read-adjoints need.  Storing a value the program already holds cannot
    change it, so state, liveness and reads stay bit-identical to the forward's
    (tested in ``test_k4_fused_backward.py``).
    """
    row = tl.program_id(0)
    block = tl.program_id(1)
    batch_index = row // H
    head_index = row % H

    ranks = tl.arange(0, RP)
    keep_r = ranks < R
    values = block * BV + tl.arange(0, BV)
    keep_v = values < V
    keep = keep_r[:, None] & keep_v[None, :]

    state_base = row * R * V
    acc = tl.load(state_in + state_base + ranks[:, None] * V + values[None, :],
                  mask=keep, other=0.0)
    live = tl.load(live_in + row * R + ranks, mask=keep_r, other=0) != 0

    weights_row = weights + batch_index * wbs + head_index * whs
    decay_row = decay + batch_index * dbs + head_index * dhs
    reads_row = reads + batch_index * obs + head_index * ohs
    payload_row = payload + batch_index * pbs + head_index * psh
    # int64 base: the trajectory is the one tensor here whose element offsets
    # can pass 2^31 (rank 512 at the bench shape is 2.1e9 floats).
    traj_row = trajectory + row.to(tl.int64) * T * R * V

    for t in tl.range(0, T, num_stages=STAGES):
        w = tl.load(weights_row + t * wt + ranks * wr, mask=keep_r, other=0.0)
        read = tl.sum(w[:, None] * acc, axis=0)
        tl.store(reads_row + t * ot + values * ov, read, mask=keep_v)
        # The state exactly as this token's read saw it: pre-DECAY, pre-lanes.
        tl.store(traj_row + t * R * V + ranks[:, None] * V + values[None, :],
                 acc, mask=keep)
        lam = tl.load(decay_row + t * dt)
        acc = _mul_rn(acc, lam)
        for lane in tl.static_range(L):
            channel = batch_index * sbs + t * st + head_index * sh + lane * sl
            if tl.load(op + channel) != NONE_CODE:
                target = tl.load(slot + channel).to(tl.int32)
                hit = keep_r & (target == ranks)
                written = tl.load(payload_row + t * pst + lane * psl + values,
                                  mask=keep_v, other=0.0)
                tile = tl.broadcast_to(written[None, :], (RP, BV))
                if tl.load(erase + channel) != 0:
                    acc = tl.where(hit[:, None], tile, acc)
                else:
                    acc = tl.where(hit[:, None], _add_rn(acc, tile), acc)
                live = tl.where(hit, tl.load(live_after + channel) != 0, live)

    tl.store(state_out + state_base + ranks[:, None] * V + values[None, :], acc,
             mask=keep)
    if block == 0:
        tl.store(live_out + row * R + ranks, live.to(tl.int8), mask=keep_r)


def persistent_scan_trajectory(state: torch.Tensor, live: torch.Tensor,
                               weights: torch.Tensor, decay: torch.Tensor,
                               schedule: MatrixSchedule, block: int | None = None,
                               warps: int | None = None, stages: int = 3):
    """The persistent block scan, plus the per-token state trajectory.

    Returns ``(reads, state, live, trajectory)`` with ``trajectory``
    ``(batch, heads, tokens, rank, d_v)``: ``trajectory[:, :, t]`` is the state
    as token ``t``'s read saw it (pre-DECAY).  The first three outputs are
    bit-identical to ``hostloop_free.persistent_scan``; the trajectory is the
    K4 backward's input, and its bytes are the stage's first-class memory cost.
    """
    check_block_shapes(state, live, weights, decay, schedule)
    batch, heads, rank, value_dim = state.shape
    tokens = schedule.tokens
    block = value_block(rank, value_dim) if block is None else int(block)
    if block < MIN_VALUE_BLOCK or block & (block - 1):
        raise ValueError(f"the value block must be a power of two >= {MIN_VALUE_BLOCK}")
    device = state.device
    reads = torch.empty((batch, heads, tokens, value_dim), dtype=torch.float32,
                        device=device)
    out_state = torch.empty_like(state)
    out_live = torch.empty_like(live)
    trajectory = torch.empty((batch, heads, tokens, rank, value_dim),
                             dtype=torch.float32, device=device)
    grid = (batch * heads, triton.cdiv(value_dim, block))
    tile = triton.next_power_of_2(rank) * block
    warps = warp_count(tile) if warps is None else int(warps)
    _persistent_scan_trajectory_kernel[grid](
        state, live.view(torch.int8), weights, decay,
        schedule.slot, schedule.op, schedule.erase.view(torch.int8),
        schedule.live_after.view(torch.int8), schedule.payload,
        reads, out_state, out_live.view(torch.int8), trajectory,
        tokens, rank,
        *weights.stride(), *decay.stride(),
        *schedule.slot.stride(), *schedule.payload.stride()[:4],
        *reads.stride(),
        H=heads, L=schedule.lanes, V=value_dim,
        RP=triton.next_power_of_2(rank), BV=block, STAGES=stages, num_warps=warps)
    return reads, out_state, out_live, trajectory


# ---------------------------------------------------------------- the reverse walk


@triton.jit(do_not_specialize=['T', 'R', 'NB'])
def _persistent_bwd_kernel(
    weights, decay, dreads, dstate_out, trajectory,
    slot, op, erase,
    dstate_in, dweights_partial, ddecay_partial, dpayload,
    T, R, NB,
    wbs, whs, wt, wr,
    dbs, dhs, dt,
    gbs, ghs, gt, gv,
    sbs, st, sh, sl,
    pbs, pst, psh, psl,
    H: tl.constexpr, L: tl.constexpr, V: tl.constexpr,
    RP: tl.constexpr, BV: tl.constexpr, STAGES: tl.constexpr,
):
    """One program per ``(row, value_block)``, walking the tokens in reverse.

    The adjoint tile ``ds`` (``RP x BV``) lives in the program's registers, the
    same budget the forward's state tile uses, and every op of the walk is one
    row of the adjoint table in this module's docstring.  The loop counts up
    (``t = T - 1 - i``) so ``tl.range``'s software pipelining still works: the
    walk is a serial chain and its per-token cost is a dependent global load,
    exactly as in the forward.

    One writer per output element, no atomics anywhere:

    * ``dstate_in`` -- the adjoint tile at the end of the walk;
    * ``dpayload[t, l]`` -- the addressed row's gradient, column-block by
      column-block (the value axis is column-independent in the adjoint too);
      inactive lanes are written as exact zeros, matching the registered eager
      path's masked payload (``scan.chunk_scan``) rather than
      ``sequential_matrix_scan``'s unmasked batched write;
    * ``dweights_partial[row, block, t, :]`` and ``ddecay_partial[row, block,
      t]`` -- the per-block partials of the two adjoint reductions that run
      along the value axis; the wrapper sums them over the block axis in a
      fixed order.
    """
    row = tl.program_id(0)
    block = tl.program_id(1)
    batch_index = row // H
    head_index = row % H

    ranks = tl.arange(0, RP)
    keep_r = ranks < R
    values = block * BV + tl.arange(0, BV)
    keep_v = values < V
    keep = keep_r[:, None] & keep_v[None, :]

    state_base = row * R * V
    ds = tl.load(dstate_out + state_base + ranks[:, None] * V + values[None, :],
                 mask=keep, other=0.0)

    weights_row = weights + batch_index * wbs + head_index * whs
    decay_row = decay + batch_index * dbs + head_index * dhs
    dreads_row = dreads + batch_index * gbs + head_index * ghs
    dpayload_row = dpayload + batch_index * pbs + head_index * psh
    traj_row = trajectory + row.to(tl.int64) * T * R * V
    dw_row = dweights_partial + (row * NB + block) * T * R
    ddl_row = ddecay_partial + (row * NB + block) * T

    for i in tl.range(0, T, num_stages=STAGES):
        t = T - 1 - i
        # 1. the lanes, last lane first (the exact reverse of the forward order).
        for lane in tl.static_range(L):
            back = L - 1 - lane
            channel = batch_index * sbs + t * st + head_index * sh + back * sl
            dpay = dpayload_row + t * pst + back * psl + values
            if tl.load(op + channel) != NONE_CODE:
                target = tl.load(slot + channel).to(tl.int32)
                hit = keep_r & (target == ranks)
                # The payload gradient is the addressed row of the adjoint as it
                # stands once every later op has been reversed.
                grad_row = tl.sum(tl.where(hit[:, None], ds, 0.0), axis=0)
                tl.store(dpay, grad_row, mask=keep_v)
                if tl.load(erase + channel) != 0:
                    # An erasing write discarded the incoming row, so its
                    # gradient terminates here (a POP kills the row; a CORRECT
                    # or MERGE target replaces it).
                    ds = tl.where(hit[:, None], 0.0, ds)
            else:
                tl.store(dpay, tl.zeros((BV,), dtype=tl.float32), mask=keep_v)
        # 2. the DECAY adjoint: dlam_t = <ds, S_t> against the pre-decay state,
        #    then ds <- lam_t * ds.
        traj_tile = tl.load(traj_row + t * R * V + ranks[:, None] * V
                            + values[None, :], mask=keep, other=0.0)
        lam = tl.load(decay_row + t * dt)
        tl.store(ddl_row + t,
                 tl.sum(tl.sum(ds * traj_tile, axis=1), axis=0))
        ds = _mul_rn(ds, lam)
        # 3. the read adjoint: dw_t[j] = dout_t . S_t[j] (a per-block partial --
        #    the reduction runs along the value axis), and the direct term
        #    dS[j] += w_t[j] * dout_t.
        w = tl.load(weights_row + t * wt + ranks * wr, mask=keep_r, other=0.0)
        dout = tl.load(dreads_row + t * gt + values * gv, mask=keep_v, other=0.0)
        tl.store(dw_row + t * R + ranks,
                 tl.sum(dout[None, :] * traj_tile, axis=1), mask=keep_r)
        ds = _add_rn(ds, w[:, None] * dout[None, :])

    tl.store(dstate_in + state_base + ranks[:, None] * V + values[None, :], ds,
             mask=keep)


def check_backward_shapes(trajectory: torch.Tensor, weights: torch.Tensor,
                          decay: torch.Tensor, schedule: MatrixSchedule,
                          dreads: torch.Tensor, dstate_out: torch.Tensor) -> None:
    """Refuse a backward whose inputs do not belong to the same block, pre-launch."""
    if trajectory.dim() != 5:
        raise ValueError("the trajectory must be (batch, heads, tokens, rank, d_v)")
    batch, heads, tokens, rank, value_dim = trajectory.shape
    if not (trajectory.is_cuda and dreads.is_cuda and dstate_out.is_cuda):
        raise ValueError("the backward is a device kernel; inputs must be on CUDA")
    if weights.shape != (batch, heads, tokens, rank):
        raise ValueError(f"weights {tuple(weights.shape)} do not match the "
                         f"trajectory {tuple(trajectory.shape)}")
    if decay.shape != (batch, heads, tokens):
        raise ValueError(f"decay must be {(batch, heads, tokens)}, "
                         f"got {tuple(decay.shape)}")
    if dreads.shape != (batch, heads, tokens, value_dim):
        raise ValueError(f"dreads must be {(batch, heads, tokens, value_dim)}, "
                         f"got {tuple(dreads.shape)}")
    if dstate_out.shape != (batch, heads, rank, value_dim):
        raise ValueError(f"dstate_out must be {(batch, heads, rank, value_dim)}, "
                         f"got {tuple(dstate_out.shape)}")
    if schedule.payload.shape != (batch, tokens, heads, schedule.lanes, value_dim):
        raise ValueError("the schedule's payload does not belong to this block")
    for name, tensor in (("trajectory", trajectory), ("dreads", dreads),
                         ("dstate_out", dstate_out)):
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be fp32; the algebra is fp32")
    for name, tensor in (("trajectory", trajectory), ("dstate_out", dstate_out)):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous (the kernel indexes it directly)")


def persistent_scan_backward(trajectory: torch.Tensor, weights: torch.Tensor,
                            decay: torch.Tensor, schedule: MatrixSchedule,
                            dreads: torch.Tensor, dstate_out: torch.Tensor,
                            block: int | None = None, warps: int | None = None,
                            stages: int = 3):
    """The adjoint of the whole block: one reverse-walk kernel plus two reductions.

    Returns ``(dstate, dweights, ddecay, dpayload)`` -- the gradients of the
    block call's four continuous inputs, in their input layouts.  The decision
    channel (``slot``/``op``/``erase``/``live_after``) and the liveness output
    carry no gradient by construction, so none is returned for them.  This is a
    plain kernel entry point: call it through :func:`fused_block_scan` when the
    result must participate in autograd.
    """
    check_backward_shapes(trajectory, weights, decay, schedule, dreads, dstate_out)
    batch, heads, tokens, rank, value_dim = trajectory.shape
    block = value_block(rank, value_dim) if block is None else int(block)
    if block < MIN_VALUE_BLOCK or block & (block - 1):
        raise ValueError(f"the value block must be a power of two >= {MIN_VALUE_BLOCK}")
    nblocks = triton.cdiv(value_dim, block)
    device = trajectory.device
    dstate = torch.empty((batch, heads, rank, value_dim), dtype=torch.float32,
                         device=device)
    dweights_partial = torch.empty((batch, heads, nblocks, tokens, rank),
                                   dtype=torch.float32, device=device)
    ddecay_partial = torch.empty((batch, heads, nblocks, tokens),
                                 dtype=torch.float32, device=device)
    dpayload = torch.empty(schedule.payload.shape, dtype=torch.float32, device=device)
    tile = triton.next_power_of_2(rank) * block
    warps = warp_count(tile) if warps is None else int(warps)
    grid = (batch * heads, nblocks)
    _persistent_bwd_kernel[grid](
        weights, decay, dreads, dstate_out, trajectory,
        schedule.slot, schedule.op, schedule.erase.view(torch.int8),
        dstate, dweights_partial, ddecay_partial, dpayload,
        tokens, rank, nblocks,
        *weights.stride(), *decay.stride(), *dreads.stride(),
        *schedule.slot.stride(), *dpayload.stride()[:4],
        H=heads, L=schedule.lanes, V=value_dim,
        RP=triton.next_power_of_2(rank), BV=block, STAGES=stages, num_warps=warps)
    # The two value-axis reductions, in a fixed order over the block axis: the
    # deterministic form of a cross-program sum (no atomics anywhere).
    dweights = dweights_partial.sum(dim=2)
    ddecay = ddecay_partial.sum(dim=2)
    return dstate, dweights, ddecay, dpayload


# ------------------------------------------------------------ the autograd boundary


class _FusedBlockScan(torch.autograd.Function):
    """The whole block call as one autograd node: the trajectory forward + the walk.

    Inputs: ``state``, ``weights``, ``decay`` and the schedule's ``payload``
    carry gradient; ``live`` and the decision channel (``slot``/``erase``/
    ``op``/``live_after``) are passed through for shape checking and are
    returned ``None`` by the backward -- the recorded-schedule doctrine.  The
    bool liveness output is non-differentiable and its incoming gradient is
    ignored.
    """

    @staticmethod
    def forward(ctx, state, live, weights, decay, payload, slot, erase, op,
                live_after, block, warps, stages):
        schedule = MatrixSchedule(slot=slot, erase=erase, payload=payload, op=op,
                                  live_after=live_after)
        reads, out_state, out_live, trajectory = persistent_scan_trajectory(
            state, live, weights, decay, schedule, block, warps, stages)
        ctx.save_for_backward(weights, decay, slot, erase, op)
        ctx.trajectory = trajectory
        ctx.lanes = schedule.lanes
        ctx.tuning = (block, warps, stages)
        return reads, out_state, out_live

    @staticmethod
    def backward(ctx, dreads, dstate_out, dlive):
        weights, decay, slot, erase, op = ctx.saved_tensors
        block, warps, stages = ctx.tuning
        batch, heads, tokens, rank = weights.shape
        value_dim = ctx.trajectory.shape[-1]
        if dreads is None:
            dreads = torch.zeros((batch, heads, tokens, value_dim),
                                 dtype=torch.float32, device=weights.device)
        if dstate_out is None:
            dstate_out = torch.zeros((batch, heads, rank, value_dim),
                                     dtype=torch.float32, device=weights.device)
        # The walk indexes ``dstate_out`` directly, so it must be contiguous;
        # an autograd grad can arrive as any layout.
        if not dstate_out.is_contiguous():
            dstate_out = dstate_out.contiguous()
        schedule = MatrixSchedule(slot=slot, erase=erase,
                                  payload=torch.empty(batch, tokens, heads,
                                                      ctx.lanes, value_dim,
                                                      dtype=torch.float32,
                                                      device=weights.device),
                                  op=op,
                                  live_after=torch.zeros(batch, tokens, heads,
                                                          ctx.lanes, dtype=torch.bool,
                                                          device=weights.device))
        dstate, dweights, ddecay, dpayload = persistent_scan_backward(
            ctx.trajectory, weights, decay, schedule, dreads, dstate_out,
            block, warps, stages)
        return dstate, None, dweights, ddecay, dpayload, None, None, None, None, \
            None, None, None


def fused_block_scan(state: torch.Tensor, live: torch.Tensor, weights: torch.Tensor,
                     decay: torch.Tensor, schedule: MatrixSchedule, *,
                     block: int | None = None, warps: int | None = None,
                     stages: int = 3):
    """The whole block call with its fused backward: ``(reads, state, live)``.

    This is the opt-in path -- the flag the arm would flip after its gates
    pass.  It refuses a store that does not fit one program by name (rank 2048
    at ``d_v`` 128: ``ValueError ... does not fit, so it must take the chunked
    path``), exactly as ``persistent_scan`` does.
    """
    return _FusedBlockScan.apply(state, live, weights, decay, schedule.payload,
                                 schedule.slot, schedule.erase, schedule.op,
                                 schedule.live_after, block, warps, stages)


def eager_block_scan(state: torch.Tensor, live: torch.Tensor, weights: torch.Tensor,
                     decay: torch.Tensor, schedule: MatrixSchedule,
                     chunk_tokens: int = 128):
    """The registered training path's own form: the eager chunked scan, attached.

    Identical arithmetic to ``scan.chunked_scan(..., fused=False)`` (it *is*
    that call, chunk by chunk, in the block paths' head-major layout) and it is
    what ``trainable_block_scan`` falls back to where the persistent kernel
    does not fit.
    """
    batch, heads, tokens, _ = weights.shape
    value_dim = state.shape[-1]
    reads = torch.empty((batch, heads, tokens, value_dim), dtype=torch.float32,
                        device=state.device)
    chunk_tokens = int(chunk_tokens)
    for start in range(0, tokens, chunk_tokens):
        stop = min(start + chunk_tokens, tokens)
        chunk_reads, state, live = chunk_scan(
            state, live, weights[:, :, start:stop], decay[:, :, start:stop],
            schedule.chunk(start, stop), fused=False)
        reads[:, :, start:stop] = chunk_reads
    return reads, state, live


def trainable_block_scan(state: torch.Tensor, live: torch.Tensor, weights: torch.Tensor,
                         decay: torch.Tensor, schedule: MatrixSchedule, *,
                         fused: bool = False, chunk_tokens: int = 128):
    """One whole block, trainable, by flag.  **The default is the eager path.**

    ``fused=False`` (the default) is the registered arm's own backward --
    ordinary autograd through the eager chunked scan.  ``fused=True`` takes the
    persistent kernel with its fused backward where the store fits one
    program, and falls back to the eager path where it does not (the same
    ``persistent_available`` rule ``hostloop_free.block_scan(path='auto')``
    applies to the forward).
    """
    if fused and persistent_available(state.shape[2], state.shape[3]):
        return fused_block_scan(state, live, weights, decay, schedule)
    return eager_block_scan(state, live, weights, decay, schedule, chunk_tokens)


__all__ = ['eager_block_scan', 'fused_block_scan', 'persistent_scan_backward',
           'persistent_scan_trajectory', 'trainable_block_scan']
