"""Per-token loop over the same decision schedule: the sequential baseline.

Semantically identical to ``x30_tape_cell.naive_reference.tape_scan`` in replay
mode, but vectorised over rows and free of host/device synchronisation, so the
chunked-vs-sequential comparison measures the chunking rather than the Python
overhead of the certified reference.

Every token re-reads the whole ``[rows, slots, K+V]`` store: this is the
serialisation *and* the state traffic the chunked scan removes.
"""
from __future__ import annotations

import math

import torch

from .chunked_cell import NEG_MASK, ChunkState, empty_state
from .schedule import DecisionSchedule


def sequential_tape_scan(k, q, v, schedule: DecisionSchedule, cfg, *,
                         state: ChunkState | None = None,
                         collect_deaths: bool = False):
    """Sequential replay of ``schedule``: ``(read [rows, tokens, V], state)``."""
    rows, tokens, key_dim = k.shape
    value_dim = v.shape[2]
    slots = cfg.slots
    schedule.check(rows, tokens, slots).fail_on_collision()
    if any(t.dtype != torch.float32 for t in (k, q, v)):
        raise ValueError('inputs must be fp32')
    if state is None:
        state = empty_state(rows, slots, key_dim, value_dim, k.device)
    keys, values = state.keys, state.values
    strength, live = state.strength.clone(), state.live.clone()
    age, usage = state.age.clone(), state.usage.clone()
    scale = 2.0 / math.sqrt(key_dim)
    device = k.device
    index = torch.arange(slots, device=device)
    row_index = torch.arange(rows, device=device)
    read = torch.empty(rows, tokens, value_dim, device=device, dtype=v.dtype)
    deaths = [] if collect_deaths else None
    for t in range(tokens):
        qt, kt, vt = q[:, t], k[:, t], v[:, t]
        scores = (torch.bmm(qt.unsqueeze(1), keys.transpose(1, 2)).squeeze(1) * scale)
        scores = scores.masked_fill(~live, NEG_MASK)
        probs = torch.softmax(scores, dim=-1)
        weights = probs * strength * live
        read[:, t] = torch.bmm(weights.unsqueeze(1), values).squeeze(1)
        usage = usage * cfg.usage_decay + weights     # pre-write, pre-zeroing usage
        slot_a, slot_c = schedule.alloc[:, t], schedule.correct[:, t]
        active_a, active_c = slot_a >= 0, slot_c >= 0
        mask_a = slot_a.unsqueeze(-1) == index
        mask_c = slot_c.unsqueeze(-1) == index
        usage = torch.where(mask_a, torch.zeros_like(usage), usage)
        # ---- POP: strength decrement then the free threshold (order preserved)
        strength = torch.where(schedule.pop_slot[:, t].unsqueeze(-1) == index,
                               strength * schedule.pop_factor[:, t].unsqueeze(-1),
                               strength)
        freed_pop = live & (strength <= cfg.free_threshold)
        live = live & ~freed_pop
        # ---- APPEND (the only allocating op) and the delta-rule correction
        before = values.gather(1, slot_c.clamp(min=0).view(rows, 1, 1)
                               .expand(rows, 1, value_dim)).squeeze(1)
        before = torch.where(active_c.unsqueeze(-1), before, torch.zeros_like(before))
        appended = schedule.gate[:, t].unsqueeze(-1) * vt
        corrected = before + schedule.beta[:, t].unsqueeze(-1) * (vt - before)
        for slot, payload, active in ((slot_a, appended, active_a),
                                      (slot_c, corrected, active_c)):
            target = slot.clamp(min=0)
            current = values.gather(1, target.view(rows, 1, 1)
                                    .expand(rows, 1, value_dim)).squeeze(1)
            values[row_index, target] = torch.where(active.unsqueeze(-1), payload, current)
        target = slot_a.clamp(min=0)
        current_key = keys.gather(1, target.view(rows, 1, 1)
                                  .expand(rows, 1, key_dim)).squeeze(1)
        keys[row_index, target] = torch.where(active_a.unsqueeze(-1), kt, current_key)
        live = live | mask_a
        strength = torch.where(mask_a, torch.ones_like(strength), strength)
        age = torch.where(mask_a, torch.zeros_like(age), age)
        # ---- DECAY / AGE / FREE (end of token)
        strength = torch.where(live, strength * cfg.decay_rate, strength)
        age = age + live.to(torch.int64)
        freed_decay = live & (strength <= cfg.free_threshold)
        live = live & ~freed_decay
        if collect_deaths:
            deaths.append((freed_pop, freed_decay))
    final = ChunkState(keys=keys, values=values, strength=strength, live=live,
                       age=age, usage=usage)
    return read, (final, deaths) if collect_deaths else final


__all__ = ['sequential_tape_scan']
