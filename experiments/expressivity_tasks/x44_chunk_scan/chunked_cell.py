"""Chunked parallel scan of the Emender slot-store token loop.

Semantics are those of ``x30_tape_cell.naive_reference`` in replay mode: the
discrete decisions are INPUTS, never recomputed from rounded state.  The
recurrence they induce is the affine diagonal map of
``formal/lean/EMENDER_OPERATOR_ALGEBRA.typ``.  Per token and slot:

* strength    ``s <- (1-pop on the popped slot) then (x decay if live)``
              with ``1`` injected on append -- two affine steps per token, so
              the floating-point operation order and therefore the strength
              trajectory, the death boundaries and the ROUTING are the same as
              the sequential path;
* content     last write wins: an append/merge erases the previous value
              (``d = 0``) and injects the new one; a correction is
              ``v <- v + beta*(v_new - v)`` on the corrected slot;
* age/usage   additive channels (``+1`` per live token, ``+read weight`` per
              token, reset on append).

What is parallel over the chunk: every token's read.  The read is
``softmax_j(q.K_j) * strength_j * live_j . V_j`` over the live slots, i.e. a
matvec per token in the sequential path.  Inside a chunk it becomes two
matmuls against the chunk-entry state plus an ``C x C`` term for the slots
written inside the chunk:

    scores = q_chunk @ K_entry^T                    (C x S)
             with the columns of in-chunk appended slots replaced by
             q_chunk @ k_event^T                    (C x C)
    read   = W @ V_entry + (valid * W_event) @ (payload - V_entry[slot])
    W      = softmax(scores masked by liveness) * strength_at_read * live

so the chunk-entry state (``S x (K+V)``) is streamed once per CHUNK instead of
once per token.  That traffic reduction, not arithmetic, is the win; see
``RESULT_CHUNK_SCAN.md`` for the measurement.

The sequential path additionally recomputes the append gate from the write
residual (``sigmoid(logit + coef*error)``).  That residual is a state
function, so the chunked scan takes the gate magnitudes from the schedule --
exactly what the certified replay path does with ``Decision.error``.  With the
production ``error_coef = 0`` the gate is a pure input function, and the
schedule builder rejects a non-zero coefficient rather than silently dropping
it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .affine import scan_with_entry
from .schedule import DecisionSchedule

NEG_MASK = -1.0e30  # same dead-slot softmax mask as the X30 reference


@dataclass
class ChunkState:
    """Slot store as the scan carries it: ``[rows, slots, ...]`` per row."""

    keys: torch.Tensor      # [rows, slots, key_dim]
    values: torch.Tensor    # [rows, slots, value_dim]
    strength: torch.Tensor  # [rows, slots]
    live: torch.Tensor      # [rows, slots] bool
    age: torch.Tensor       # [rows, slots] int64
    usage: torch.Tensor     # [rows, slots]

    def clone(self) -> 'ChunkState':
        return ChunkState(*(getattr(self, name).clone() for name in
                            ('keys', 'values', 'strength', 'live', 'age', 'usage')))

    def rows(self, sl) -> 'ChunkState':
        return ChunkState(*(getattr(self, name)[sl] for name in
                            ('keys', 'values', 'strength', 'live', 'age', 'usage')))


def empty_state(rows: int, slots: int, key_dim: int, value_dim: int, device) -> ChunkState:
    return ChunkState(
        keys=torch.zeros(rows, slots, key_dim, device=device),
        values=torch.zeros(rows, slots, value_dim, device=device),
        strength=torch.zeros(rows, slots, device=device),
        live=torch.zeros(rows, slots, dtype=torch.bool, device=device),
        age=torch.zeros(rows, slots, dtype=torch.int64, device=device),
        usage=torch.zeros(rows, slots, device=device))


def _scan(multiplier: torch.Tensor, injection: torch.Tensor,
          entry: torch.Tensor):
    """``[C, R, S]`` multiplier/injection through the affine engine."""
    steps, rows, slots = multiplier.shape
    if entry.shape != (rows, slots) or injection.shape != multiplier.shape:
        raise ValueError('affine scan shape mismatch')
    lanes = rows * slots
    states, final = scan_with_entry(multiplier.reshape(steps, lanes),
                                    injection.reshape(steps, lanes),
                                    entry.reshape(lanes))
    return states.reshape(steps, rows, slots), final.reshape(rows, slots)


def _write_values(store: torch.Tensor, slot: torch.Tensor, value: torch.Tensor,
                  rows: torch.Tensor, active: torch.Tensor) -> None:
    """In-place write of ``value[r]`` into ``store[r, slot[r]]`` where active.

    Sync-free (no ``nonzero``): rows without a write re-store their current
    slot-0 value.
    """
    target = slot.clamp(min=0)
    current = store.gather(1, target.view(-1, 1, 1).expand(-1, 1, store.shape[-1]))
    store[rows, target] = torch.where(active.unsqueeze(-1), value,
                                      current.squeeze(1))


def _write_keys(store: torch.Tensor, slot: torch.Tensor, key: torch.Tensor,
                rows: torch.Tensor, active: torch.Tensor) -> None:
    _write_values(store, slot, key, rows, active)


def _chunk(kc, qc, vc, sched: DecisionSchedule, entry: ChunkState, cfg, *,
           scale: float, derive_deaths: bool, deaths: list | None = None):
    """Advance one chunk. Returns ``(read [R,C,V], exit ChunkState)``."""
    rows, steps = kc.shape[0], kc.shape[1]
    slots, value_dim = entry.strength.shape[1], entry.values.shape[2]
    device = kc.device
    index = torch.arange(slots, device=device)
    row_index = torch.arange(rows, device=device)
    decay = cfg.decay_rate
    threshold = cfg.free_threshold

    # ---- bookkeeping pass: liveness without in-chunk deaths, indices, payloads
    live = entry.live.clone()
    live_read_naive = torch.empty(rows, steps, slots, dtype=torch.bool, device=device)
    live_after_naive = torch.empty_like(live_read_naive)
    last_append = torch.full((rows, steps, slots), -1, dtype=torch.int32, device=device)
    last_write = torch.full((rows, steps, slots), -1, dtype=torch.int32, device=device)
    write_slot = torch.full((rows, steps), -1, dtype=torch.int64, device=device)
    payload = torch.zeros(rows, steps, value_dim, device=device, dtype=vc.dtype)
    running_append = torch.full((rows, slots), -1, dtype=torch.int32, device=device)
    running_write = torch.full((rows, slots), -1, dtype=torch.int32, device=device)
    values_cur = entry.values.clone()
    keys_cur = entry.keys.clone()
    append_mask = torch.empty(rows, steps, slots, dtype=torch.bool, device=device)
    for i in range(steps):
        live_read_naive[:, i] = live
        last_append[:, i] = running_append
        last_write[:, i] = running_write
        slot_a, slot_c = sched.alloc[:, i], sched.correct[:, i]
        active_a, active_c = slot_a >= 0, slot_c >= 0
        write_slot[:, i] = torch.where(active_a, slot_a, slot_c)
        before = values_cur.gather(
            1, slot_c.clamp(min=0).view(rows, 1, 1).expand(rows, 1, value_dim)).squeeze(1)
        before = torch.where(active_c.unsqueeze(-1), before, torch.zeros_like(before))
        appended = sched.gate[:, i].unsqueeze(-1) * vc[:, i]
        corrected = before + sched.beta[:, i].unsqueeze(-1) * (vc[:, i] - before)
        payload[:, i] = torch.where(active_a.unsqueeze(-1), appended, corrected)
        _write_values(values_cur, slot_a, appended, row_index, active_a)
        _write_values(values_cur, slot_c, corrected, row_index, active_c)
        _write_keys(keys_cur, slot_a, kc[:, i], row_index, active_a)
        mask_a = slot_a.unsqueeze(-1) == index
        mask_c = slot_c.unsqueeze(-1) == index
        append_mask[:, i] = mask_a
        stamp = torch.full_like(running_append, i)
        running_append = torch.where(mask_a, stamp, running_append)
        running_write = torch.where(mask_a | mask_c, stamp, running_write)
        live = live | mask_a
        live_after_naive[:, i] = live

    # ---- strength channel: two affine steps per token (pop phase, decay phase)
    ones = torch.ones(rows, steps, slots, dtype=kc.dtype, device=device)
    zeros = torch.zeros(rows, steps, slots, dtype=kc.dtype, device=device)

    def strength_maps(live_after):
        """Exact operation order of the sequential path (pop multiply, then decay)."""
        popped = torch.where(sched.pop_slot.unsqueeze(-1) == index,
                             sched.pop_factor.unsqueeze(-1), ones)
        # An append ERASES the lane and injects 1; the birth-token decay is the
        # second step, exactly as `strength*(1-onehot) + onehot*1.0` then `*decay`.
        pop_map = torch.where(append_mask, zeros, popped)
        pop_injection = torch.where(append_mask, ones, zeros)
        decay_map = torch.where(live_after, torch.full_like(ones, decay), ones)
        pair_mult = torch.stack([pop_map, decay_map], dim=2).permute(1, 2, 0, 3)
        pair_inj = torch.stack([pop_injection, zeros], dim=2).permute(1, 2, 0, 3)
        return (pair_mult.reshape(2 * steps, rows, slots),
                pair_inj.reshape(2 * steps, rows, slots))

    death_pop = torch.zeros(rows, steps, slots, dtype=torch.bool, device=device)
    death_decay = torch.zeros_like(death_pop)
    if derive_deaths:
        # Pass A ignores in-chunk deaths: a lane's strength dynamics depend only
        # on its own lane, so the FIRST threshold crossing per life is unchanged
        # by that omission.  The pop check uses the pop multiply applied to the
        # pre-pop strength directly, because an append at the same token erases
        # the lane in pass A (the reference still records that POP death before
        # the reviving append).
        mult, inj = strength_maps(live_after_naive)
        probe, _ = _scan(mult, inj, entry.strength)
        pre_pop = torch.empty(rows, steps, slots, dtype=kc.dtype, device=device)
        pre_pop[:, 0] = entry.strength
        if steps > 1:
            pre_pop[:, 1:] = probe[1::2][:-1].permute(1, 0, 2)
        popped_factor = torch.where(sched.pop_slot.unsqueeze(-1) == index,
                                    sched.pop_factor.unsqueeze(-1), ones)
        post_pop = pre_pop * popped_factor
        tracked = entry.live.clone()
        for i in range(steps):
            crossed = tracked & (post_pop[:, i] <= threshold)
            death_pop[:, i] = crossed
            tracked = (tracked & ~crossed) | append_mask[:, i]
            crossed = tracked & (probe[2 * i + 1] <= threshold)
            death_decay[:, i] = crossed
            tracked = tracked & ~crossed
        live = entry.live.clone()
        live_read = torch.empty_like(live_read_naive)
        live_after = torch.empty_like(live_read_naive)
        for i in range(steps):
            live_read[:, i] = live
            live = live & ~death_pop[:, i]
            live = live | append_mask[:, i]
            live_after[:, i] = live
            live = live & ~death_decay[:, i]
        mult, inj = strength_maps(live_after)
        strength_states, strength_final = _scan(mult, inj, entry.strength)
        live_final = live
    else:
        live_read, live_after = live_read_naive, live_after_naive
        mult, inj = strength_maps(live_after)
        strength_states, strength_final = _scan(mult, inj, entry.strength)
        live_final = live

    # strength observed by the read of token i = state BEFORE that token's pop
    # step, i.e. after the previous token's decay step.
    strength_at_read = torch.empty(rows, steps, slots, dtype=kc.dtype, device=device)
    strength_at_read[:, 0] = entry.strength
    if steps > 1:
        strength_at_read[:, 1:] = strength_states[1::2][:-1].permute(1, 0, 2)

    # ---- read: one softmax over the slots, entry state streamed once per chunk
    entry_keys = entry.keys
    scores = torch.bmm(qc, entry_keys.transpose(1, 2)) * scale
    append_keys = torch.where(sched.alloc.unsqueeze(-1) >= 0, kc,
                              torch.zeros_like(kc))
    if steps > 1:
        event_scores = torch.bmm(qc, append_keys.transpose(1, 2)) * scale
        override = event_scores.gather(2, last_append.clamp(min=0).long())
        scores = torch.where(last_append >= 0, override, scores)
    scores = scores.masked_fill(~live_read, NEG_MASK)
    weights = torch.softmax(scores, dim=-1) * strength_at_read * live_read
    read = torch.bmm(weights, entry.values)

    # slots whose value changed inside this chunk: replace the entry value by
    # the (single, last) in-chunk payload for every later token
    if steps > 1:
        column = write_slot.clamp(min=0).unsqueeze(1).expand(rows, steps, steps)
        last = last_write.gather(2, column).long()
        valid = (write_slot.unsqueeze(1) >= 0) & (last >= 0)
        valid = valid & (last == torch.arange(steps, device=device).view(1, 1, steps))
        valid = valid & (torch.arange(steps, device=device).view(1, steps, 1)
                         > torch.arange(steps, device=device).view(1, 1, steps))
        entry_at_slot = entry.values.gather(
            1, write_slot.clamp(min=0).unsqueeze(-1).expand(rows, steps, value_dim))
        delta = payload - entry_at_slot
        delta = torch.where((write_slot >= 0).unsqueeze(-1), delta,
                            torch.zeros_like(delta))
        weight_at_slot = weights.gather(2, column)
        read = read + torch.bmm((valid * weight_at_slot).to(read.dtype), delta)

    # ---- usage and age: additive channels
    # usage: u <- u*usage_decay + read_weight, then zeroed on append.
    usage_mult = torch.where(append_mask, zeros, torch.full_like(ones, cfg.usage_decay))
    usage_inj = torch.where(append_mask, zeros, weights)
    _, usage_final = _scan(usage_mult.permute(1, 0, 2), usage_inj.permute(1, 0, 2),
                           entry.usage)

    # age: age <- age*[not append] + [live after the pop/append step].
    age_mult = torch.where(append_mask, zeros, ones)
    age_inj = live_after.to(ones.dtype)
    _, age_final = _scan(age_mult.permute(1, 0, 2), age_inj.permute(1, 0, 2),
                         entry.age.to(ones.dtype))

    if deaths is not None:
        deaths.append((death_pop, death_decay))
    exit_state = ChunkState(keys=keys_cur, values=values_cur,
                            strength=strength_final, live=live_final,
                            age=age_final.round().to(torch.int64),
                            usage=usage_final)
    return read, exit_state


def chunked_tape_scan(k, q, v, schedule: DecisionSchedule, cfg, *,
                      chunk_size: int = 64, state: ChunkState | None = None,
                      row_block: int | None = None,
                      derive_deaths: bool = True,
                      deaths: list | None = None):
    """Chunked parallel scan of the slot store over a fixed decision schedule.

    Args (all ``[rows, tokens, ...]`` with ``rows = batch * heads``):
        k, q  : write key and read query ``[rows, tokens, key_dim]``
        v     : write value ``[rows, tokens, value_dim]``
        schedule : the discrete decisions (see ``DecisionSchedule``)

    ``deaths``, when a list is passed, collects the per-chunk
    ``(pop_chunk_deaths, decay_chunk_deaths)`` masks the scan derived from the
    strength channel (shape ``[rows, chunk, slots]`` per chunk).

    Returns ``(read [rows, tokens, value_dim], final ChunkState)``.
    """
    rows, tokens = k.shape[0], k.shape[1]
    key_dim, value_dim = k.shape[2], v.shape[2]
    if not (0 < cfg.decay_rate <= 1 and 0 <= cfg.free_threshold < 1):
        raise ValueError('invalid decay rate or free threshold')
    slots = cfg.slots
    if q.shape != k.shape or v.shape[:2] != (rows, tokens):
        raise ValueError('key/query/value shape mismatch')
    if any(t.dtype != torch.float32 for t in (k, q, v)):
        raise ValueError('inputs must be fp32')
    schedule.check(rows, tokens, slots).fail_on_collision()
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    if state is None:
        state = empty_state(rows, slots, key_dim, value_dim, k.device)
    if state.strength.shape != (rows, slots):
        raise ValueError('state/schedule shape mismatch')
    scale = 2.0 / math.sqrt(key_dim)   # F12b sharpened read softmax
    if row_block is None:
        # Keep the [rows, chunk, slots] trajectories inside a fixed budget.
        row_block = max(1, min(rows, 2 ** 24 // (chunk_size * slots)))
    read = torch.empty(rows, tokens, value_dim, device=k.device, dtype=v.dtype)
    pieces, death_blocks = [], []
    for start in range(0, rows, row_block):
        sl = slice(start, min(rows, start + row_block))
        carry = state.rows(sl)
        block_deaths = [] if deaths is not None else None
        for begin in range(0, tokens, chunk_size):
            stop = min(tokens, begin + chunk_size)
            piece, carry = _chunk(
                k[sl, begin:stop], q[sl, begin:stop], v[sl, begin:stop],
                DecisionSchedule(*(getattr(schedule, name)[sl, begin:stop] for name in
                                   ('alloc', 'gate', 'correct', 'beta',
                                    'pop_slot', 'pop_factor'))),
                carry, cfg, scale=scale, derive_deaths=derive_deaths,
                deaths=block_deaths)
            read[sl, begin:stop] = piece
        pieces.append(carry)
        death_blocks.append(block_deaths)
    if deaths is not None:
        for chunk_index in range(len(death_blocks[0])):
            deaths.append(tuple(
                torch.cat([block[chunk_index][phase] for block in death_blocks], dim=0)
                for phase in (0, 1)))
    final = ChunkState(*(torch.cat([getattr(piece, name) for piece in pieces], dim=0)
                         for name in ('keys', 'values', 'strength', 'live',
                                      'age', 'usage')))
    return read, final


__all__ = ['ChunkState', 'chunked_tape_scan', 'empty_state', 'NEG_MASK']
