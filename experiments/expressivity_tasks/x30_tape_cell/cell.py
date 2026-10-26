"""Timestamp tape store, batched head reads, and checkpointed trace replay.

Slots have immutable birth times and lazy strength reference times. Management
uses host-side bitmaps, intrusive newest lists, and a cancellable timer wheel.
This is deliberately PyTorch/Python, not a claim of GPU-efficient scheduling:
scalar routing crosses the host/device boundary. No X28E code is imported.

POP follows the executable X28E oracle: multiplicative strength reduction,
then death at free_threshold (not unconditional binary kill). Module writes
are detached binary enables, making correction/merge full overwrites.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from collections import deque

import torch
from torch import nn
from .naive_reference import TapeHeadConfig
from .float_decay import advance, death, f32, unit_factors
from .routing_trace import (RoutingTrace, NO_WRITE, R_EMPTY, R_NOMATCH,
                            R_ZERO_RESIDUAL, R_CONFLICT, R_CAP)


class FreeList:
    """Lowest-free allocator, matching native argmax tie breaking.

    A hierarchical 64-bit bitmap has <=3 levels for budgets <=262144. Each
    allocation/free touches a bounded number of words, never scans slots.
    No mutex/atomic operation is needed: a store has one sequential owner.
    This is single-owner lock-free, not a concurrent allocator.
    """
    def __init__(self, size):
        if not 1 <= size <= 262144:
            raise ValueError('supported budget is 1..262144')
        self.size = size
        self.levels = []
        n = size
        while True:
            words = [(1 << min(64, n - i)) - 1 for i in range(0, n, 64)]
            self.levels.append(words)
            if len(words) == 1:
                break
            n = len(words)

    def _set(self, index, available):
        for words in self.levels:
            word, bit = divmod(index, 64)
            if available:
                words[word] |= 1 << bit
            else:
                words[word] &= ~(1 << bit)
            available = words[word] != 0
            index = word

    def take(self):
        if not self.levels[-1][0]:
            return -1
        index = 0
        for words in reversed(self.levels):
            bits = words[index]
            index = index * 64 + ((bits & -bits).bit_length() - 1)
        self._set(index, False)
        return index

    def reserve(self, slot):
        self._set(slot, False)

    def release(self, slot):
        self._set(slot, True)

    def full(self):
        return self.levels[-1][0] == 0


class IndexSet:
    """Sparse 64-ary slot-index set for birth ties, with bounded-depth min."""
    def __init__(self, size):
        self.levels = []
        while True:
            self.levels.append({})
            size = (size+63)//64
            if size == 1:
                break

    def set(self, index, present):
        for words in self.levels:
            word, bit = divmod(index, 64)
            bits = words.get(word, 0)
            bits = bits | (1 << bit) if present else bits & ~(1 << bit)
            if bits:
                words[word] = bits
            else:
                words.pop(word, None)
            present, index = bool(bits), word

    def first(self):
        if not self.levels[-1]:
            return -1
        index = 0
        for words in reversed(self.levels):
            bits = words[index]
            index = index*64 + ((bits & -bits).bit_length()-1)
        return index

    def clone(self):
        other = object.__new__(IndexSet)
        other.levels = [words.copy() for words in self.levels]
        return other


@dataclass
class Decision:
    case: list[int]
    matched: list[int]
    alloc: list[int]
    pop: list[int]
    freed_pop: list[list[int]]
    freed_decay: list[list[int]]
    w_on: list[bool]
    error: torch.Tensor
    pressure: list[float]
    pop_native: list[float] = field(default_factory=list)
    decay_native: list[float] = field(default_factory=list)


@dataclass
class Trace:
    steps: list[Decision]
    batch: int
    heads: int
    slots: int

    def dense(self, device='cpu'):
        """Explicit diagnostic conversion; never allocate dense death traces in scan."""
        def tensor(name, dtype=torch.long):
            return torch.tensor([getattr(s, name) for s in self.steps],
                                device=device, dtype=dtype).reshape(
                                    len(self.steps), self.batch, self.heads).transpose(0, 1)
        freed = torch.zeros(self.batch, len(self.steps), self.heads, self.slots,
                            device=device, dtype=torch.int8)
        for t, step in enumerate(self.steps):
            for n in range(self.batch * self.heads):
                b, h = divmod(n, self.heads)
                freed[b, t, h, step.freed_pop[n]] = 1
                freed[b, t, h, step.freed_decay[n]] += 2
        pop = tensor('pop')
        return RoutingTrace(tensor('case'), tensor('matched'), pop >= 0,
                            pop, tensor('alloc'), tensor('w_on', torch.bool), freed)


class Management:
    """Detached single-owner bookkeeping. Timer events are cancellable O(1)."""
    def __init__(self, n, cfg, window=128):
        if not (0 < cfg.decay_rate <= 1 and 0 <= cfg.free_threshold < 1 and window > 0):
            raise ValueError('require 0<decay<=1, 0<=free_threshold<1, positive window')
        self.cfg, self.n, self.t, self.window = cfg, n, 0, window
        self.free_threshold = float(torch.tensor(cfg.free_threshold, dtype=torch.float32))
        self.eps_residual = float(torch.tensor(cfg.eps_residual, dtype=torch.float32))
        s = cfg.slots
        self.free = [FreeList(s) for _ in range(n)]
        self.birth = [[0] * s for _ in range(n)]
        self.touch = [[0] * s for _ in range(n)]
        self.live = [[False] * s for _ in range(n)]
        self.dead_age = [[0] * s for _ in range(n)]
        self.newest = [-1] * n
        self.by_birth = [{} for _ in range(n)]
        self.groups = [{} for _ in range(n)]
        self.owned_groups = [set() for _ in range(n)]
        self.group_prev = [{} for _ in range(n)]
        self.group_next = [{} for _ in range(n)]
        self.newest_birth = [None] * n
        self.count = [0] * n
        self.native_ref = [[0.] * s for _ in range(n)]
        self.nonunit_count = 0
        self.pending_initial_free = [[] for _ in range(n)]
        self.due = [[None] * s for _ in range(n)]
        self.events = {}  # token-boundary -> {(store, slot): None}
        self.history = [deque() for _ in range(n)]
        self.requests = [0] * n
        self.caps = [0] * n
        # Use the actual fp32 decay constant, as the native multiply does.
        self.log_decay = math.log(float(torch.tensor(cfg.decay_rate, dtype=torch.float32)))

    def __deepcopy__(self, memo):
        # Rows contain immutable numbers. Copying them as Python objects one
        # by one is unnecessary checkpoint overhead; list.copy is a C memcpy.
        other = object.__new__(Management)
        other.__dict__ = self.__dict__.copy()
        for name in ('birth', 'touch', 'live', 'dead_age',
                     'native_ref', 'due', 'pending_initial_free'):
            setattr(other, name, [row.copy() for row in getattr(self, name)])
        for name in ('newest', 'newest_birth', 'count', 'requests', 'caps'):
            setattr(other, name, getattr(self, name).copy())
        other.by_birth = [row.copy() for row in self.by_birth]
        other.group_prev = [row.copy() for row in self.group_prev]
        other.group_next = [row.copy() for row in self.group_next]
        # Birth groups are persistent across checkpoints, copied on touch.
        other.groups = [row.copy() for row in self.groups]
        other.owned_groups = [set() for _ in range(self.n)]
        other.events = {t: row.copy() for t, row in self.events.items()}
        other.history = [deque(row) for row in self.history]
        other.free = []
        for allocator in self.free:
            cloned = copy.copy(allocator)
            cloned.levels = [row.copy() for row in allocator.levels]
            other.free.append(cloned)
        return other

    def cancel(self, n, j):
        due = self.due[n][j]
        if due is not None:
            bucket = self.events[due]
            del bucket[n, j]
            if not bucket:
                del self.events[due]
            self.due[n][j] = None

    def schedule(self, n, j, strength):
        self.cancel(n, j)
        if self.cfg.decay_rate == 1 or (self.cfg.free_threshold == 0 and self.cfg.decay_rate > .5):
            return
        # A scalar IEEE-mantissa interval jump handles native fp32 rounding;
        # analytic logarithms alone can put a death one token too early.
        dt, _ = death(strength, self.cfg.decay_rate, self.free_threshold)
        if dt is None:
            return
        due = self.t + dt
        self.events.setdefault(due, {})[n, j] = None
        self.due[n][j] = due

    def add(self, n, j, birth=None):
        self.free[n].reserve(j)
        self.live[n][j] = True
        self.count[n] += 1
        self.native_ref[n][j] = 1.
        self.birth[n][j] = self.t if birth is None else birth
        self.touch[n][j] = self.t
        born = self.birth[n][j]
        if born not in self.groups[n]:
            self.groups[n][born] = IndexSet(self.cfg.slots)
            self.owned_groups[n].add(born)
            old = self.newest_birth[n]
            self.group_prev[n][born], self.group_next[n][born] = old, None
            if old is not None:
                self.group_next[n][old] = born
            self.newest_birth[n] = born
        elif born not in self.owned_groups[n]:
            self.groups[n][born] = self.groups[n][born].clone()
            self.owned_groups[n].add(born)
        self.groups[n][born].set(j, True)
        self.by_birth[n][born] = self.groups[n][born].first()
        self.newest[n] = self.by_birth[n][self.newest_birth[n]]

    def remove(self, n, j, boundary):
        assert self.live[n][j]
        self.cancel(n, j)
        born = self.birth[n][j]
        if born not in self.owned_groups[n]:
            self.groups[n][born] = self.groups[n][born].clone()
            self.owned_groups[n].add(born)
        self.groups[n][born].set(j, False)
        first = self.groups[n][born].first()
        if first >= 0:
            self.by_birth[n][born] = first
        else:
            del self.by_birth[n][born], self.groups[n][born]
            self.owned_groups[n].discard(born)
            p, q = self.group_prev[n].pop(born), self.group_next[n].pop(born)
            if p is not None:
                self.group_next[n][p] = q
            if q is not None:
                self.group_prev[n][q] = p
            else:
                self.newest_birth[n] = p
        latest = self.newest_birth[n]
        self.newest[n] = self.by_birth[n][latest] if latest is not None else -1
        self.live[n][j] = False
        self.nonunit_count -= int(self.native_ref[n][j] != 1.)
        self.dead_age[n][j] = boundary - self.birth[n][j]
        self.touch[n][j] = boundary
        self.count[n] -= 1
        self.free[n].release(j)

    def pressure(self, n, case):
        hist = self.history[n]
        req = int(case in (R_EMPTY, R_NOMATCH, R_ZERO_RESIDUAL, R_CAP))
        cap = int(case == R_CAP)
        if len(hist) == self.window:
            r, c = hist.popleft()
            self.requests[n] -= r
            self.caps[n] -= c
        hist.append((req, cap))
        self.requests[n] += req
        self.caps[n] += cap
        return self.caps[n] / max(1, self.requests[n])


@dataclass
class State:
    keys: torch.Tensor       # [B*H, extent, K], head-parallel rows
    values: torch.Tensor     # [B*H, extent, V]
    s_ref: torch.Tensor      # fp32 strength at t_touch (lazy log exponent)
    management: Management
    batch: int
    heads: int

    def materialize(self):
        """Diagnostic/checkpoint boundary conversion to the full native layout.

        Dead-slot age and strength are frozen at death, exactly as native.
        Usage EMA is research instrumentation, not cell semantics, and is
        intentionally omitted from the efficient store.
        """
        m, width = self.management, self.keys.shape[1]
        dev = self.keys.device
        live = torch.tensor(m.live, device=dev)
        touch = torch.tensor(m.touch, device=dev)
        birth = torch.tensor(m.birth, device=dev)
        pad = m.cfg.slots - width
        keys = torch.nn.functional.pad(self.keys, (0, 0, 0, pad))
        values = torch.nn.functional.pad(self.values, (0, 0, 0, pad))
        ref = torch.nn.functional.pad(self.s_ref, (0, pad))
        dt = torch.where(live, m.t - touch, 0)
        strength = ref * unit_factors(m.cfg.decay_rate, int(dt.max()), dev)[dt]
        native = [[advance(m.native_ref[i][j], m.cfg.decay_rate,
                           m.t-m.touch[i][j]) if m.live[i][j] else m.native_ref[i][j]
                   for j in range(m.cfg.slots)] for i in range(m.n)]
        exact = ref.new_tensor(native)
        strength = strength + (exact-strength).detach()
        age = torch.where(live, m.t - birth, torch.tensor(m.dead_age, device=dev))
        def reshape(x):
            return x.reshape(self.batch, self.heads, *x.shape[1:])
        return dict(keys=reshape(keys), values=reshape(values), strength=reshape(strength),
                    live=reshape(live), age=reshape(age))


def empty_state(batch, cfg, device, window=128):
    n = batch * cfg.n_heads
    return State(torch.zeros(n, 0, cfg.key_dim, device=device, dtype=torch.float32),
                 torch.zeros(n, 0, cfg.value_dim, device=device, dtype=torch.float32),
                 torch.zeros(n, 0, device=device, dtype=torch.float32),
                 Management(n, cfg, window), batch, cfg.n_heads)


def from_native(state, cfg, t=0, window=128):
    """Import arbitrary stores once. Lowest index wins initial birth ties."""
    b, h, s = state.live.shape
    m = Management(b * h, cfg, window)
    m.t = t
    live = state.live.reshape(b*h, s).cpu().tolist()
    ages = state.age.reshape(b*h, s).cpu().tolist()
    strengths = state.strength.detach().reshape(b*h, s).cpu().tolist()
    for n in range(b*h):
        for j in sorted((j for j in range(s) if live[n][j]),
                        key=lambda j: (-ages[n][j], -j)):
            m.add(n, j, birth=t - ages[n][j])
            m.touch[n][j] = t
            m.native_ref[n][j] = strengths[n][j]
            m.nonunit_count += int(strengths[n][j] != 1.)
            if strengths[n][j] <= m.free_threshold:
                m.pending_initial_free[n].append(j)
            else:
                m.schedule(n, j, strengths[n][j])
        m.dead_age[n] = ages[n][:]
        for j in range(s):
            if not live[n][j]:
                m.native_ref[n][j] = strengths[n][j]
    return State(state.keys.reshape(b*h, s, cfg.key_dim),
                 state.values.reshape(b*h, s, cfg.value_dim),
                 state.strength.reshape(b*h, s), m, b, h)


def _put(x, rows, slots, values, replay):
    if not rows:
        return x
    idx = (torch.tensor(rows, device=x.device), torch.tensor(slots, device=x.device))
    if replay:
        return x.index_put(idx, values)
    x[idx] = values
    return x


def _run(k, q, v, beta, pop, logit, coef, keys, values, ref, initial, decisions=None):
    """No-grad sparse forward or differentiable replay of recorded decisions.

    Replays never evaluate a routing threshold, argmax, residual, allocator,
    or expiration schedule. Only the supplied trace determines management.
    """
    replay = decisions is not None
    m = copy.deepcopy(initial)
    cfg, n, length = m.cfg, m.n, k.shape[0]
    old_width = keys.shape[1]
    width = min(cfg.slots, old_width + length)
    if width > old_width:
        keys = torch.nn.functional.pad(keys, (0, 0, 0, width-old_width))
        values = torch.nn.functional.pad(values, (0, 0, 0, width-old_width))
        ref = torch.nn.functional.pad(ref, (0, width-old_width))
    elif not replay:
        keys, values, ref = keys.clone(), values.clone(), ref.clone()
    device, rows_all = k.device, list(range(n))
    device_live = torch.tensor([row[:width] for row in m.live], device=device)
    device_touch = torch.tensor([row[:width] for row in m.touch], device=device)
    earliest = min((m.touch[i][j] for i in rows_all for j in range(old_width)
                    if m.live[i][j]), default=m.t)
    factors = unit_factors(cfg.decay_rate, m.t + length - earliest, device)
    steps, outputs = [], []
    extent = old_width
    for local in range(length):
        t = m.t
        live = device_live.clone() if replay else device_live
        dt = torch.where(live, t-device_touch, 0)
        strength = ref * factors[dt]
        if m.nonunit_count:
            # Generic popped/imported strengths require exact fp32 products.
            # This is read-path work, not a decay/death scan. The common fresh
            # strength=1 path uses only the shared decay-factor lookup above.
            exact = ref.new_tensor([
                [advance(m.native_ref[i][j], cfg.decay_rate, t-m.touch[i][j])
                 if m.live[i][j] else m.native_ref[i][j] for j in range(width)]
                for i in rows_all])
            strength = strength + (exact-strength).detach()
        scores = torch.bmm(keys, q[local].unsqueeze(2)).squeeze(2) * (2 / math.sqrt(cfg.key_dim))
        probs = scores.masked_fill(~live, -1e30).softmax(-1)
        weights = probs * strength * live
        outputs.append(torch.bmm(weights.unsqueeze(1), values).squeeze(1))
        if replay:
            step = decisions[local]
            pops = step.pop
        else:
            best = scores.detach().masked_fill(~live, -1e30).argmax(-1).cpu().tolist()
            fire = (pop[local].detach() > cfg.pop_threshold).cpu().tolist()
            pops = [best[i] if fire[i] and m.count[i] else -1 for i in rows_all]
        prows = [i for i in rows_all if pops[i] >= 0]
        pslots = [pops[i] for i in prows]
        if prows:
            new_s = strength[prows, pslots] * (1-pop[local, prows])
            if replay:
                native_pop = step.pop_native
            else:
                p_cpu = pop[local, prows].detach().cpu().tolist()
                native_pop = [f32(advance(m.native_ref[i][j], cfg.decay_rate,
                                          t-m.touch[i][j]) * f32(1-p))
                              for i, j, p in zip(prows, pslots, p_cpu)]
            new_s = new_s + (new_s.new_tensor(native_pop)-new_s).detach()
            ref = _put(ref, prows, pslots, new_s, replay)
            for i, j, s in zip(prows, pslots, native_pop):
                m.nonunit_count += int(s != 1.)-int(m.native_ref[i][j] != 1.)
                m.native_ref[i][j] = s
                m.touch[i][j] = t
            device_touch[prows, pslots] = t
        if replay:
            fp = step.freed_pop
        else:
            fp = [row[:] for row in m.pending_initial_free]
            # Only imported invalid live rows and popped slots can cross the
            # threshold before decay. Imported rows are checked once at entry.
            s_cpu = new_s.detach().cpu().tolist() if prows else []
            for i, j, s in zip(prows, pslots, s_cpu):
                if s <= m.free_threshold:
                    if j not in fp[i]:
                        fp[i].append(j)
                else:
                    m.schedule(i, j, s)
        for i in rows_all:
            for j in fp[i]:
                m.remove(i, j, t)
                device_live[i, j] = False
        m.pending_initial_free = [[] for _ in rows_all]
        if not replay:
            enabled = (beta[local].detach() > cfg.write_enable_threshold).cpu().tolist()
            matched, alloc, cases = [-1]*n, [-1]*n, [NO_WRITE]*n
            adjacent = [m.by_birth[i].get(t-1, -1) for i in rows_all]
            age1 = [j >= 0 for j in adjacent]
            targets = [adjacent[i] if age1[i] else m.newest[i] for i in rows_all]
            # Initial states may have age-1 ties. Linked ordering makes newest
            # the lowest-index member of the youngest birth group.
            idx = [max(0, j) for j in targets]
            vb = values[rows_all, idx]
            vb = torch.where(torch.tensor(age1, device=device).unsqueeze(-1), vb, 0.)
            residual = ((v[local] - vb)**2).sum(-1).detach()
            residual_cpu = residual.cpu().tolist()
            for i in rows_all:
                if not enabled[i]:
                    continue
                if not m.count[i]:
                    case = R_EMPTY
                elif not age1[i]:
                    case = R_NOMATCH
                elif residual_cpu[i] < m.eps_residual:
                    case = R_ZERO_RESIDUAL
                else:
                    case = R_CONFLICT
                if age1[i]:
                    matched[i] = targets[i]
                if case != R_CONFLICT:
                    if m.free[i].full():
                        case, matched[i] = R_CAP, targets[i]
                    else:
                        alloc[i] = m.free[i].take()
                cases[i] = case
            # Native F12g residual convention: for R-CAP with no age1 the
            # forward residual is v-0. Replay consumes this recorded error.
            error = torch.where(torch.tensor([c > 0 for c in m.count], device=device),
                                residual.sqrt(), 0.).detach()
            pressure = [m.pressure(i, cases[i]) for i in rows_all]
            step = Decision(cases, matched, alloc, pops, fp, [], enabled, error, pressure)
            step.pop_native = native_pop if prows else []
        else:
            for i in rows_all:
                m.pressure(i, step.case[i])
        arows = [i for i in rows_all if step.alloc[i] >= 0]
        aslots = [step.alloc[i] for i in arows]
        gate = torch.sigmoid(logit[local] + coef * step.error)
        if arows:
            keys = _put(keys, arows, aslots, k[local, arows].detach(), replay)
            values = _put(values, arows, aslots, gate[arows, None] * v[local, arows], replay)
            ref = _put(ref, arows, aslots, torch.ones_like(ref[arows, aslots]), replay)
            for i, j in zip(arows, aslots):
                m.add(i, j)
                if not replay:
                    m.schedule(i, j, 1.)
            device_live[arows, aslots] = True
            device_touch[arows, aslots] = t
            extent = max(extent, max(aslots)+1)
        crows = [i for i in rows_all if step.case[i] in (R_CONFLICT, R_CAP)]
        cslots = [step.matched[i] for i in crows]
        if crows:
            prev = values[crows, cslots]
            corrected = prev + beta[local, crows, None].detach() * (v[local, crows] - prev)
            values = _put(values, crows, cslots, corrected, replay)
        if replay:
            fd = step.freed_decay
        else:
            fd = [[] for _ in rows_all]
            for i, j in list(m.events.get(t+1, {})):
                fd[i].append(j)
            step.freed_decay = fd
        drows = [i for i in rows_all for j in fd[i]]
        dslots = [j for i in rows_all for j in fd[i]]
        if drows:
            # Materialize at death to preserve native dead-row final state.
            ages = ref.new_tensor([t+1-m.touch[i][j] for i, j in zip(drows, dslots)])
            dead_s = ref[drows, dslots] * factors[ages.long()]
            native_dead = (step.decay_native if replay else
                           [advance(m.native_ref[i][j], cfg.decay_rate,
                                    t+1-m.touch[i][j]) for i, j in zip(drows, dslots)])
            if not replay:
                step.decay_native = native_dead
            dead_s = dead_s + (dead_s.new_tensor(native_dead)-dead_s).detach()
            ref = _put(ref, drows, dslots, dead_s, replay)
            for i, j, s in zip(drows, dslots, native_dead):
                m.remove(i, j, t+1)
                m.native_ref[i][j] = s
            device_live[drows, dslots] = False
            device_touch[drows, dslots] = t+1
        m.t += 1
        steps.append(step)
    return (torch.stack(outputs), keys[:, :extent], values[:, :extent],
            ref[:, :extent]), m, steps


class _ReplayChunk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, initial, saved, *args):
        result, final, steps = _run(*args, initial, decisions=saved.get('input_steps'))
        ctx.initial, ctx.steps = initial, steps
        ctx.save_for_backward(*args)
        saved['final'], saved['steps'] = final, steps
        if not args[7].requires_grad:
            ctx.mark_non_differentiable(result[1])  # frozen geometry needs no key adjoint
        if not args[9].requires_grad and not any(j >= 0 for step in steps for j in step.pop):
            ctx.mark_non_differentiable(result[3])  # untouched strengths are constants
        return result

    @staticmethod
    def backward(ctx, *grad_outputs):
        args = [x.detach().requires_grad_(need) for x, need in
                zip(ctx.saved_tensors, ctx.needs_input_grad[2:])]
        with torch.enable_grad():
            result, _, _ = _run(*args, ctx.initial, decisions=ctx.steps)
            active_outputs = [(x, g) for x, g in zip(result, grad_outputs)
                              if x.requires_grad and g is not None]
            active_inputs = [x for x in args if x.requires_grad]
            gradients = (torch.autograd.grad(
                [x for x, _ in active_outputs], active_inputs,
                [g for _, g in active_outputs], allow_unused=True)
                if active_outputs else [None] * len(active_inputs))
        iterator = iter(gradients)
        result = []
        for index, x in enumerate(args):
            g = next(iterator) if x.requires_grad else None
            # Sparse replay prunes zero paths that the dense oracle retains.
            # Return zeros for learned content/gates so AdamW sees the same
            # zero gradients, including closed POP. Geometry/enables stay
            # architecturally detached rather than acquiring zero gradients.
            if x.requires_grad and g is None and index not in (0, 3):
                g = torch.zeros_like(x)
            result.append(g)
        return (None, None, *result)


def tape_scan(k, q, v, write_strength, pop_strength, append_logit, state,
              cfg, append_err_coef=None, chunk_size=16, trace_override=None):
    """Inputs [B,T,H,D]. Returns read, lazy State, sparse Trace, pressure.

    Custom PyTorch checkpoints run forward without a graph, retaining only
    chunk boundaries and discrete traces. Backward replays chunks in reverse;
    replay receives decisions as inputs and never recomputes routing.
    All checkpoint tensors remain fp32. Dense tensor copies during reverse
    replay are a v1 limitation, separate from sparse forward maintenance.
    """
    b, length, h = k.shape[:3]
    if h != cfg.n_heads or k.shape[-1] != cfg.key_dim or q.shape != k.shape:
        raise ValueError('key/query/head shape does not match config')
    if v.shape != (b, length, h, cfg.value_dim):
        raise ValueError('value shape does not match config')
    if chunk_size < 1 or length < 1:
        raise ValueError('positive chunk size and nonempty token sequence required')
    if state is None:
        state = empty_state(b, cfg, k.device)
    if state.batch != b or state.heads != h or state.management.cfg != cfg:
        raise ValueError('state/config shape mismatch')
    if trace_override is not None and (trace_override.batch != b or trace_override.heads != h
                                      or trace_override.slots != cfg.slots
                                      or len(trace_override.steps) != length):
        raise ValueError('trace override shape mismatch')
    def flatten(x):
        return x.transpose(0, 1).reshape(length, b*h, *x.shape[3:]).float()
    k, q, v, beta, pop, logit = map(flatten, (k, q, v, write_strength, pop_strength, append_logit))
    if append_err_coef is None:
        coef = k.new_zeros(b*h)
    else:
        coef = append_err_coef.float().repeat(b)
    keys, values, ref = state.keys, state.values, state.s_ref
    m = state.management
    outputs, all_steps = [], []
    for start in range(0, length, chunk_size):
        stop = min(length, start + chunk_size)
        saved = {}
        if trace_override is not None:
            saved['input_steps'] = trace_override.steps[start:stop]
        initial = m
        args = (k[start:stop], q[start:stop], v[start:stop], beta[start:stop],
                pop[start:stop], logit[start:stop], coef, keys, values, ref)
        if torch.is_grad_enabled() and any(x.requires_grad for x in args):
            out, keys, values, ref = _ReplayChunk.apply(initial, saved, *args)
        else:
            with torch.no_grad():
                result, final, steps = _run(*args, initial, decisions=saved.get('input_steps'))
                saved['steps'], saved['final'] = steps, final
                out, keys, values, ref = result
        m = saved['final']
        outputs.append(out)
        all_steps.extend(saved['steps'])
    read = torch.cat(outputs).reshape(length, b, h, cfg.value_dim).transpose(0, 1)
    pressure = read.new_tensor([s.pressure for s in all_steps]).reshape(length, b, h).transpose(0, 1)
    return read, State(keys, values, ref, m, b, h), Trace(all_steps, b, h, cfg.slots), pressure


class TapeCell(nn.Module):
    def __init__(self, d_model, cfg=TapeHeadConfig(), chunk_size=16, pressure_window=128):
        super().__init__()
        self.cfg, self.chunk_size, self.pressure_window = cfg, chunk_size, pressure_window
        h, k, v = cfg.n_heads, cfg.key_dim, cfg.value_dim
        self.W_q = nn.Linear(d_model, h*k)
        self.W_k = nn.Linear(d_model, h*k)
        self.W_v = nn.Linear(d_model, h*v)
        self.W_write = nn.Linear(d_model, h)
        self.W_pop = nn.Linear(d_model, h)
        self.W_append = nn.Linear(d_model, h)
        self.append_err_coef = nn.Parameter(torch.zeros(h))
        with torch.no_grad():
            self.W_q.load_state_dict(self.W_k.state_dict())
            self.W_write.bias.fill_(2.)
            self.W_pop.bias.fill_(-4.)
        self.W_k.requires_grad_(False)
        self.W_write.requires_grad_(False)

    def forward(self, x, token_ids, state=None, trace_override=None):
        b, t = x.shape[:2]
        h, cfg = self.cfg.n_heads, self.cfg
        k = self.W_k(x).reshape(b, t, h, cfg.key_dim).detach()
        q = self.W_q(x).reshape_as(k)
        v = self.W_v(x).reshape(b, t, h, cfg.value_dim)
        beta = ((self.W_write(x).sigmoid().detach() > cfg.write_enable_threshold)
                & (token_ids != 0).unsqueeze(-1)).float()
        if state is None:
            state = empty_state(b, cfg, x.device, self.pressure_window)
        return tape_scan(k, q, v, beta, self.W_pop(x).sigmoid(), self.W_append(x),
                         state, cfg, self.append_err_coef, self.chunk_size, trace_override)


def replay_trace(k, q, v, beta, pop, logit, state, trace, coef=None):
    """Explicit replay gate, also useful for testing changed routing thresholds."""
    b, t, h = k.shape[:3]
    def flat(x):
        return x.transpose(0, 1).reshape(t, b*h, *x.shape[3:])
    if coef is None:
        coef = k.new_zeros(b*h)
    else:
        coef = coef.repeat(b)
    result, m, _ = _run(*map(flat, (k, q, v, beta, pop, logit)), coef,
                         state.keys, state.values, state.s_ref,
                         state.management, decisions=trace.steps)
    out, keys, values, ref = result
    return (out.reshape(t, b, h, -1).transpose(0, 1),
            State(keys, values, ref, m, b, h))
