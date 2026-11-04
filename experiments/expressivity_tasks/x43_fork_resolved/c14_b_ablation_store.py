"""B-only differentiable shared store with native key and routing geometry.

This is an eager implementation, isolated from every frozen arm. Immutable
slot versions are the autograd before-images; block checkpointing rebuilds
all fields, including suffix ownership. It is NOT the certified Triton journal
or reverse kernel. Launch validation must expose that distinction.
"""
from dataclasses import dataclass, replace
import math
import torch
from .c14_b_keys import array_append


@dataclass(frozen=True)
class Slot:
    key: torch.Tensor
    value: torch.Tensor
    strength: torch.Tensor
    born: int
    owner: object = None
    prediction: int = -1
    prediction_at: int = -1
    order: int = -1


class SharedStore:
    def __init__(self, cfg):
        self.cfg = cfg
        self.slots = [None] * cfg.slots
        self.ops = dict(APPEND=0, POP=0, DECAY=0, CORRECT=0, MERGE=0,
                        stale_retirements=0, natural_retirements=0,
                        capacity_merges=0, duplicate_merges=0,
                        array_requests=0, refused_insertions=0,
                        array_read_opportunities=0, array_slots_found=0,
                        array_slots_read=0, reads=0, peak_occupancy=0,
                        peak_array=0, peak_stream=0)
        self.occupancy = []

    def live(self):
        return [i for i,s in enumerate(self.slots) if s is not None]

    def sample(self):
        live = self.live()
        array = sum(self.slots[i].owner is not None for i in live)
        stream = len(live)-array
        self.ops['peak_occupancy'] = max(self.ops['peak_occupancy'], len(live))
        self.ops['peak_array'] = max(self.ops['peak_array'], array)
        self.ops['peak_stream'] = max(self.ops['peak_stream'], stream)
        self.occupancy.append((stream, array))

    def retire(self, i, cause):
        self.slots[i] = None
        self.ops['POP'] += 1
        self.ops[cause] += 1

    def observe(self, window, observed, byte, content, clock):
        if self.cfg.array_append_only: return
        # byte is the CURRENT INPUT, never a shifted target. A prediction made
        # at clock-1 is already scored before this observation can affect reads.
        for i in self.live():
            s = self.slots[i]
            if s.owner is None:
                continue
            if s.prediction_at == clock-1 and s.prediction != byte:
                self.slots[i] = s = replace(s, value=s.value+(content-s.value))
                self.ops['CORRECT'] += 1
            if not s.owner.is_current(window, observed):
                self.retire(i, 'stale_retirements')

    def target(self, clock):
        live = sorted(self.live(), key=lambda i: (-self.slots[i].born, -self.slots[i].order, i))
        adjacent = next((i for i in live if self.slots[i].born == clock-1), None)
        return (adjacent if adjacent is not None else live[0] if live else None,
                adjacent is not None)

    def insert_array(self, key, value, owner, prediction, clock, gate):
        payload = array_append(key, owner)
        if payload is None:
            return
        self.ops['array_requests'] += 1
        duplicate = next((i for i in self.live() if self.slots[i].owner == owner), None)
        free = next((i for i,s in enumerate(self.slots) if s is None), None)
        if duplicate is not None or free is None:
            target = duplicate if duplicate is not None else self.target(clock)[0]
            old = self.slots[target]
            self.slots[target] = replace(old, value=old.value+(value-old.value),
                                         owner=owner, prediction=prediction, prediction_at=clock)
            self.ops['MERGE'] += 1
            self.ops['duplicate_merges' if duplicate is not None else 'capacity_merges'] += 1
        else:
            self.slots[free] = Slot(payload.key, gate*value, value.new_ones(()), clock,
                                    owner, prediction, clock, 2*clock)
            self.ops['APPEND'] += 1
        self.sample()

    def read(self, query, zero):
        live = self.live()
        self.ops['reads'] += 1
        arrays = [i for i in live if self.slots[i].owner is not None]
        self.ops['array_read_opportunities'] += len(arrays)
        if not live:
            return zero, None, 0.
        keys = torch.stack([self.slots[i].key for i in live])
        scores = (keys*query).sum(-1)*(2./math.sqrt(query.numel()))
        weights = scores.softmax(0)
        strengths = torch.stack([self.slots[i].strength for i in live])
        values = torch.stack([self.slots[i].value for i in live])
        # Found means numerically addressable; read means nonzero effective
        # floating-point weight. Underflow is exposed rather than called a hit.
        self.ops['array_slots_found'] += len(arrays)
        array_indices = [j for j,i in enumerate(live) if i in arrays]
        if array_indices:
            effective = weights.detach()*strengths.detach()
            self.ops['array_slots_read'] += int((effective[array_indices] > 0).sum())
        best = int(scores.detach().argmax())
        return (weights[:,None]*strengths[:,None]*values).sum(0), live[best], float(scores[best].detach())

    def stream(self, key, content, pop, append, logit, coef, match_coef,
               error_coef, policy, best, evidence, clock, valid=True):
        cfg = self.cfg
        if best is not None and float(pop.detach()) > cfg.pop_threshold:
            s = self.slots[best]
            self.slots[best] = replace(s, strength=s.strength*(1-pop))
            self.ops['POP'] += 1
            if float(self.slots[best].strength.detach()) <= cfg.free_threshold:
                self.slots[best] = None
        target, adjacent = self.target(clock)
        previous = self.slots[target].value if adjacent else torch.zeros_like(content)
        residual = (content-previous).square().sum().detach()
        error = residual.sqrt() if target is not None else residual.new_zeros(())
        occ = len(self.live())/cfg.slots
        wire = (logit + policy.occ_slope*max(0., occ-policy.occ_onset)
                + match_coef*(policy.match_wire*evidence) + error_coef*error)
        enabled = valid and float(wire.detach().sigmoid()) > cfg.write_enable_threshold
        if enabled and self.cfg.array_append_only and target is not None and adjacent and float(residual)>=cfg.eps_residual and self.slots[target].owner is not None:
            enabled=False
        if enabled:
            allocation = target is None or not adjacent or float(residual) < cfg.eps_residual
            free = next((i for i,s in enumerate(self.slots) if s is None), None)
            gate = (append+coef*error).sigmoid()
            if allocation and free is not None:
                self.slots[free] = Slot(key, gate*content, content.new_ones(()), clock, order=2*clock+1)
                self.ops['APPEND'] += 1
            else:
                old = self.slots[target]
                self.slots[target] = replace(old, value=old.value+(content-old.value),
                                             owner=None, prediction=-1, prediction_at=-1)
                self.ops['MERGE' if allocation else 'CORRECT'] += 1
                if allocation:
                    self.ops['capacity_merges'] += 1
        self.sample()
        return wire.sigmoid()

    def decay(self, *, array_only=False):
        for i in self.live():
            s = self.slots[i]
            if self.cfg.array_append_only and s.owner is not None:
                continue
            if array_only and s.owner is None:
                continue
            strength = s.strength*self.cfg.decay_rate
            self.ops['DECAY'] += 1
            if float(strength.detach()) <= self.cfg.free_threshold:
                self.retire(i, 'natural_retirements')
            else:
                self.slots[i] = replace(s, strength=strength)

    def report(self):
        self.sample()
        result = dict(self.ops)
        result.update(budget=self.cfg.slots, occupancy=self.occupancy,
                      final_stream=self.occupancy[-1][0], final_array=self.occupancy[-1][1])
        denom = result['array_read_opportunities']
        result['array_read_hit_rate'] = result['array_slots_read']/denom if denom else None
        return result
