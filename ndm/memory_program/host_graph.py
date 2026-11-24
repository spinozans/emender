"""Opt-in CUDA graph of a *known* recurrent byte-host step. No policy or reader in the graph."""

from collections import Counter
from threading import get_ident
from time import perf_counter_ns

import torch
from .host_cache import CachedByteHost
from .recurrent_host import RecurrentHost


class StepGraph(RecurrentHost):
    def __init__(self, net):
        if net.embed.weight.device.type != "cuda":
            raise ValueError("CUDA graph host requires CUDA")
        super().__init__(net)
        self.graph = None
        self.static = None
        self.output = None
        self.owner = None
        self.prepared = False
        self.work = Counter()

    def _binding(self, slots):
        if (
            not self.prepared
            or slots is not self.owner.slots
            or get_ident() != self.thread
            or torch.cuda.current_stream(self.device).cuda_stream != self.stream
        ):
            raise ValueError("foreign arena/stream/thread or closed host graph")

    def build(self, arena, b, *, capture, warm_stream=None):
        if (
            self.prepared
            or not arena.empty
            or not (arena.backend is self or getattr(arena.backend, "buckets", {}).get(b) is self)
            or not 1 <= b <= min(arena.capacity, arena.reserve)
        ):
            raise ValueError("build graph on an empty compatible arena")
        arena._check_model()
        self.owner = arena
        self.b = b
        self.thread = get_ident()
        self.stream = torch.cuda.current_stream(self.device).cuda_stream
        self.scratch_ix = torch.arange(
            arena.capacity, arena.capacity + b, device=self.device, dtype=torch.long
        )
        self.static = (self.scratch_ix.clone(), torch.zeros(b, 1, device=self.device, dtype=torch.long))
        idx, ids = self.static
        base = super().encode

        def dummy():
            return base(arena.slots, idx, [1] * b, ids, False)

        def wipe():
            for t in arena.slots.values():
                t[arena.capacity :].zero_()

        warm = torch.cuda.Stream(device=self.device) if warm_stream is None else warm_stream
        if warm.device != self.device or warm.cuda_stream == self.stream:
            raise ValueError("need a distinct same-device preparation stream")
        warm.wait_stream(torch.cuda.current_stream(self.device))
        start = perf_counter_ns()
        with torch.cuda.stream(warm):
            for _ in range(2):
                dummy()
            wipe()
        torch.cuda.current_stream(self.device).wait_stream(warm)
        torch.cuda.synchronize(self.device)
        warm_ns = perf_counter_ns() - start
        start = perf_counter_ns()
        if capture:
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=warm):
                self.output = dummy()
            self.graph.replay()
            torch.cuda.synchronize(self.device)
            for x in self.output:
                x.zero_()
        wipe()
        idx.zero_()
        ids.zero_()
        torch.cuda.synchronize(self.device)
        cap_ns = perf_counter_ns() - start
        self.stats.clear()
        self.work.clear()
        self.per_row = sum(
            t[0].numel() * t.element_size() for k, t in arena.slots.items() if k not in ("hidden", "base")
        )
        self.prepared = True
        pool_bytes = 0
        pool_active = 0
        if capture:
            pool = self.graph.pool()
            segments = torch.cuda.memory_snapshot(mempool_id=pool)
            if not segments or any(tuple(s["segment_pool_id"]) != tuple(pool) for s in segments):
                raise ValueError("unverified graph private pool")
            pool_bytes = sum(s["total_size"] for s in segments)
            pool_active = sum(s["allocated_size"] for s in segments)
        self.setup = dict(
            warmup_steps=2,
            captured=int(capture),
            warmup_ns=warm_ns,
            capture_and_init_ns=cap_ns,
            private_pool_reserved=pool_bytes,
            private_pool_allocated=pool_active,
            device_allocated_after_build=torch.cuda.memory_allocated(self.device),
            device_reserved_after_build=torch.cuda.memory_reserved(self.device),
        )

    def encode(self, arena, slots, old, ids, prompt):
        self._binding(arena)
        if prompt or self.graph is None or len(slots) != self.b:
            self.work["prefill" if prompt else "eager_step" if self.graph is None else "width_miss"] += 1
            return super().encode(arena, slots, old, ids, prompt)
        if ids.shape != (self.b, 1):
            raise ValueError("not a one-byte graph step")
        ix, token = self.static
        ix.copy_(slots)
        token.copy_(ids)
        self.graph.replay()
        self.work["real_replays"] += 1
        for k in ("gather_bytes", "scatter_bytes"):
            self.stats[k] += self.per_row * self.b
        h, p = (x.clone() for x in self.output)  # don't lend scratch across graph replays
        for x in (*self.static, *self.output):
            x.zero_()
        return h, p

    def retire(self):
        self._binding(self.owner.slots)
        if self.graph is None:
            return
        # Refresh the private workspace using only neutral, unadmittable state, even
        # on partial release. None of the surviving client indices are presented.
        ix, ids = self.static
        ix.copy_(self.scratch_ix)
        ids.zero_()
        self.graph.replay()
        self.work["neutral_retire_replays"] += 1
        for t in self.owner.slots.values():
            t[self.owner.capacity :].zero_()
        for x in (*self.static, *self.output):
            x.zero_()

    def close(self):
        if not self.prepared:
            return
        self._binding(self.owner.slots)
        if not self.owner.empty:
            raise ValueError("release admissions before graph close")
        if any(x.count_nonzero().item() for x in (*self.static, *(self.output or ()))):
            raise ValueError("user payload in graph buffers")
        if self.graph is not None:
            self.graph.reset()
        self.static = None
        self.output = None
        self.graph = None
        self.scratch_ix = None
        self.owner = None
        self.prepared = False


class PreparedArena(CachedByteHost):
    def release(self, handles):
        self.backend._binding(self.slots)
        # Retire before marking client slots admissible again. On device faults the
        # worker must fail closed, not reuse the store.
        self._indices(handles)
        self.backend.retire()
        super().release(handles)

    def admit(self, session):
        self.backend._binding(self.slots)
        return super().admit(session)
