"""Bounded, reusable cohort-width graphs on a shared byte-host cache. No session shares a slot."""

from collections import Counter
import torch
from .host_graph import PreparedArena, StepGraph
from .recurrent_host import RecurrentHost


def validate_buckets(capacity, widths, private_budget):
    if (
        type(capacity) is not int
        or capacity < 1
        or not isinstance(widths, (list, tuple))
        or not 1 <= len(widths) <= 4
        or any(type(b) is not int or not 1 <= b <= capacity for b in widths)
        or len(set(widths)) != len(widths)
        or type(private_budget) is not int
        or private_budget < 1
    ):
        raise ValueError("invalid bounded graph widths/budget")
    return tuple(sorted(widths))


def bounded_covers(capacity, widths, max_parts=6):
    """Exact sums only. BFS in descending-width order gives shortest, then lexicographically largest.

    At most four widths and six launches; storage is bounded by reachable sums, not capacity.
    A missing key means fallback, never padding with an idle slot.
    """
    sizes = validate_buckets(capacity, widths, 1)[::-1]
    if type(max_parts) is not int or not 1 <= max_parts <= 6:
        raise ValueError("invalid graph launch limit")
    plans = {0: ()}
    wave = [0]
    for _ in range(max_parts):
        fresh = []
        for used in wave:
            for width in sizes:
                n = used + width
                if n <= capacity and n not in plans:
                    plans[n] = (*plans[used], width)
                    fresh.append(n)
        wave = fresh
    del plans[0]
    return plans


def execute_cover(plan, buckets, arena, slots, old, ids):
    # No reordering, overlap or guessed IDs. Each graph returns owned (not borrowed) outputs.
    if sum(plan) != len(slots) or len(old) != len(slots) or ids.shape != (len(slots), 1):
        raise ValueError("graph cover does not exactly match real step batch")
    pos = 0
    parts = []
    for n in plan:
        piece = slice(pos, pos + n)
        pos += n
        parts.append(buckets[n].encode(arena, slots[piece], old[piece], ids[piece], False))
    return tuple(torch.cat(column, dim=0) for column in zip(*parts, strict=True))


class _Buckets(RecurrentHost):
    def __init__(self, net, widths, *, chunk_prefill=False, segmented_prefill=False):
        super().__init__(net, chunk_prefill=chunk_prefill, segmented_prefill=segmented_prefill)
        self.buckets = {b: StepGraph(net) for b in widths}
        self.cover_steps = False
        self.ready = False
        self.work = Counter()

    def build(self, a, budget):
        if self.ready or not a.empty:
            raise ValueError("resident build after admission")
        self.prep_stream = torch.cuda.Stream(device=self.device)
        self.pool_bytes = 0
        try:
            for b, g in self.buckets.items():
                g.build(a, b, capture=True, warm_stream=self.prep_stream)
                self.pool_bytes += g.setup["private_pool_reserved"]
                if self.pool_bytes > budget:
                    raise ValueError("resident graph pool budget exceeded")
        except BaseException:
            # Unpublished arena contains no client data. Device/stream faults are fatal.
            for g in self.buckets.values():
                if g.prepared:
                    g.close()
                else:
                    g.owner = None
            self.buckets.clear()
            self.prep_stream = None
            raise
        self.plans = bounded_covers(a.capacity, tuple(self.buckets))
        self.ready = True

    def _binding(self, slots):
        if not self.ready:
            raise ValueError("closed/unpublished resident host")
        next(iter(self.buckets.values()))._binding(slots)

    def encode(self, arena, slots, old, ids, prompt):
        self._binding(arena)
        if not prompt and len(slots) not in self.buckets and self.cover_steps and len(slots) in self.plans:
            plan = self.plans[len(slots)]
            self.work["covered_steps"] += 1
            self.work["cover_replays"] += len(plan)
            return execute_cover(plan, self.buckets, arena, slots, old, ids)
        if prompt or len(slots) not in self.buckets:
            self.work["prefill" if prompt else "eager_width_miss"] += 1
            return super().encode(arena, slots, old, ids, prompt)
        return self.buckets[len(slots)].encode(arena, slots, old, ids, False)

    def retire(self):
        for g in self.buckets.values():
            g.retire()
        self.work["retirements"] += 1

    def close(self):
        if not self.ready:
            return
        for g in self.buckets.values():
            g.close()
        self.buckets.clear()
        self.ready = False
        self.prep_stream = None


class ResidentArena(PreparedArena):
    """Prepare once, repeatedly admit/advance/release. The caller owns model lifetime and admission policy.

    `byte_budget` covers persistent tensors incl. neutral rows; `private_budget` is checked on
    actual graph pool sizes before publication. Neither caps transient GPU requests while building
    nor includes model weights, CUDA contexts, native processes, or allocator fragmentation.
    """

    def __init__(
        self,
        net,
        capacity,
        context,
        widths,
        *,
        byte_budget,
        private_budget,
        chunk_prefill=False,
        segmented_prefill=False,
    ):
        widths = validate_buckets(capacity, widths, private_budget)
        backend = _Buckets(net, widths, chunk_prefill=chunk_prefill, segmented_prefill=segmented_prefill)
        super().__init__(backend, capacity, context, byte_budget, reserve=max(widths))
        backend.build(self, private_budget)

    def set_cover(self, enabled):
        self.backend._binding(self.slots)
        if not self.empty or type(enabled) is not bool:
            raise ValueError("change graph dispatch only on an empty prepared store")
        self.backend.cover_steps = enabled

    def close(self):
        if not self.empty:
            raise ValueError("release sessions before resident shutdown")
        self.backend.close()
        self.slots.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.clear()
        self.close()
