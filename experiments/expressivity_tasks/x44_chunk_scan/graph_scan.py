"""CUDA-graph capture of one fixed-shape chunked scan.

The chunked scan is a chain of many small tensor ops (the intra-chunk passes are
Python loops over the chunk length).  Profiling the eager path showed the GPU
idle for most of the wall clock: ~100k launches for a 2048-token scan, 533 ms of
device time inside 1800 ms of wall time.  Capturing the whole loop in a CUDA
graph replaces that launch stream with one host launch per replay.

Constraints honoured here:

* every input is copied into a fixed buffer before replay (the graph reads
  static addresses);
* nothing inside the captured region branches on tensor VALUES, calls
  ``.item()`` or moves data to the host -- ``DecisionSchedule.check`` and the
  collision guard run before capture, not inside it;
* shapes are fixed per capture; a different ``(rows, tokens, slots, chunk)``
  gets a new graph object.

The state and read outputs returned by a replay live in the graph's private
memory pool and are only valid until the next replay: copy them if they must
outlive it.
"""
from __future__ import annotations

import torch

from .chunked_cell import ChunkState, chunked_tape_scan, empty_state
from .schedule import DecisionSchedule

_SCHEDULE_FIELDS = ('alloc', 'gate', 'correct', 'beta', 'pop_slot', 'pop_factor')
_STATE_FIELDS = ('keys', 'values', 'strength', 'live', 'age', 'usage')


class ChunkedScanGraph:
    """Capture ``chunked_tape_scan`` once and replay it with one host launch."""

    def __init__(self, cfg, rows: int, tokens: int, *, chunk_size: int = 64,
                 row_block: int | None = None, key_dim: int | None = None,
                 value_dim: int | None = None, device='cuda', warmup: int = 3):
        key_dim = cfg.key_dim if key_dim is None else key_dim
        value_dim = cfg.value_dim if value_dim is None else value_dim
        self.cfg = cfg
        self.rows, self.tokens = rows, tokens
        self.chunk_size = chunk_size
        self.row_block = row_block
        self.key_dim, self.value_dim = key_dim, value_dim
        self.device = device
        self._k = torch.zeros(rows, tokens, key_dim, device=device)
        self._q = torch.zeros(rows, tokens, key_dim, device=device)
        self._v = torch.zeros(rows, tokens, value_dim, device=device)
        self._schedule = {
            'alloc': torch.zeros(rows, tokens, dtype=torch.int64, device=device),
            'gate': torch.zeros(rows, tokens, device=device),
            'correct': torch.zeros(rows, tokens, dtype=torch.int64, device=device),
            'beta': torch.zeros(rows, tokens, device=device),
            'pop_slot': torch.zeros(rows, tokens, dtype=torch.int64, device=device),
            'pop_factor': torch.ones(rows, tokens, device=device),
        }
        state = empty_state(rows, cfg.slots, key_dim, value_dim, device)
        self._state = {name: getattr(state, name) for name in _STATE_FIELDS}
        # Warm up on a side stream: capture must not be the first execution.
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                self._run(validate=False)
        torch.cuda.current_stream(device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._read, self._final = self._run(validate=False)
        self.replay_count = 0

    def _schedule_view(self) -> DecisionSchedule:
        return DecisionSchedule(**self._schedule)

    def _run(self, validate: bool):
        schedule = self._schedule_view()
        return chunked_tape_scan(
            self._k, self._q, self._v, schedule, self.cfg,
            chunk_size=self.chunk_size, row_block=self.row_block,
            state=ChunkState(**self._state), validate=validate)

    def __call__(self, k, q, v, schedule: DecisionSchedule, state: ChunkState | None = None,
                 *, copy_out: bool = True):
        """Copy inputs into the static buffers, replay, return ``(read, state)``."""
        if k.shape != self._k.shape or v.shape != self._v.shape:
            raise ValueError('input shape does not match the captured graph')
        self._k.copy_(k)
        self._q.copy_(q)
        self._v.copy_(v)
        for name in _SCHEDULE_FIELDS:
            self._schedule[name].copy_(getattr(schedule, name))
        for name in _STATE_FIELDS:
            if state is None:
                self._state[name].zero_()
            else:
                self._state[name].copy_(getattr(state, name))
        self.graph.replay()
        self.replay_count += 1
        read = self._read.clone() if copy_out else self._read
        final = ChunkState(**{name: getattr(self._final, name).clone()
                              for name in _STATE_FIELDS})
        return read, final


__all__ = ['ChunkedScanGraph']
