"""Drive the chunked scan from a live X42 pure-arm forward.

The arm's routing decisions are produced by the certified Triton kernel inside
its token loop.  This adapter keeps that call (the arm is untouched, so its
losses stay bit-identical) and *replays* the same decisions through the chunked
scan, which is the path that exists today: the chunked content scan is
chunk-parallel over tokens, the routing pre-pass is not.

Per call it records

* the read-output and end-state deviation from the certified kernel,
* the certified and chunked wall times,

so a training step yields both the parity evidence at registered scale and the
measured size of the block the chunked path replaces.  ``collector.summary()``
is what ``lattice.py`` writes into the run receipt.
"""
from __future__ import annotations

import os
import time

import torch

from . import chunked_cell, graph_scan
from .schedule import DecisionSchedule


class Collector:
    """Per-call evidence for one instrumented training run.

    Calls are split by the arm's read path: ``X40_BATCHED_READS=1`` (production
    training) uses the X35 ``causal_read`` variant, whose softmax denominator
    counts every versioned slot rather than only the live ones, so the certified
    read is the ``batched=False`` one the arm uses for evaluation.
    """

    def __init__(self):
        self.pending = []
        self._sampled = {}
        self.calls = 0
        self.certified_seconds = 0.0
        self.chunked_seconds = 0.0
        self.max_read_deviation = 0.0
        self.max_state_deviation = 0.0
        self.max_strength_deviation = 0.0
        self.live_mismatches = 0
        self.calls_with_live_mismatch = 0
        self.rows = 0
        self.budget_seconds = 0.0
        self.failures = []
        self.cross_checks = []
        self.host_original = 0.0
        self.host_replay = 0.0
        self.by_read_path = {}

    def sampled_count(self, read_path):
        return self._sampled.get(read_path, 0)

    def record(self, certified_events, chunked_events, read_error, state_error,
               strength_error, live_mismatch, rows, read_path='unknown', sampled=True):
        """Queue one call. No host synchronisation happens here: the deviations
        are tensors and the timings are CUDA events, reduced in ``summary``."""
        self._sampled[read_path] = self._sampled.get(read_path, 0) + int(bool(sampled))
        self.pending.append((certified_events, chunked_events, read_error, state_error,
                             strength_error, live_mismatch, rows, read_path, sampled))

    def _flush(self):
        if not self.pending:
            return
        torch.cuda.synchronize()
        for (certified_events, chunked_events, read_error, state_error, strength_error,
             live_mismatch, rows, read_path, sampled) in self.pending:
            self._account(certified_events, chunked_events, read_error, state_error,
                          strength_error, live_mismatch, rows, read_path, sampled)
        self.pending = []

    def _account(self, certified_events, chunked_events, read_error, state_error,
                 strength_error, live_mismatch, rows, read_path, sampled):
        certified_seconds = (certified_events[0].elapsed_time(certified_events[1]) / 1e3
                             if certified_events[0] is not None else 0.0)
        chunked_seconds = (chunked_events[0].elapsed_time(chunked_events[1]) / 1e3
                           if chunked_events[0] is not None else 0.0)
        read_deviation = float(read_error)
        state_deviation = float(state_error)
        strength_deviation = float(strength_error)
        live_mismatch = int(live_mismatch)
        bucket = self.by_read_path.setdefault(
            read_path, dict(calls=0, max_read_deviation=0.0, max_state_deviation=0.0,
                            max_strength_deviation=0.0, calls_with_live_mismatch=0,
                            certified_seconds=0.0, chunked_seconds=0.0))
        bucket['calls'] += 1
        bucket['sampled'] = bucket.get('sampled', 0) + int(bool(sampled))
        bucket['certified_seconds'] += certified_seconds
        bucket['chunked_seconds'] += chunked_seconds
        bucket['max_read_deviation'] = max(bucket['max_read_deviation'], read_deviation)
        bucket['max_state_deviation'] = max(bucket['max_state_deviation'], state_deviation)
        bucket['max_strength_deviation'] = max(bucket['max_strength_deviation'],
                                               strength_deviation)
        bucket['calls_with_live_mismatch'] += int(bool(live_mismatch))
        self.calls += 1
        self.certified_seconds += certified_seconds
        self.chunked_seconds += chunked_seconds
        self.rows += rows
        self.max_read_deviation = max(self.max_read_deviation, read_deviation)
        self.max_state_deviation = max(self.max_state_deviation, state_deviation)
        self.max_strength_deviation = max(self.max_strength_deviation, strength_deviation)
        if live_mismatch:
            self.calls_with_live_mismatch += 1
            self.live_mismatches += live_mismatch

    def summary(self) -> dict:
        self._flush()
        total = self.certified_seconds + self.chunked_seconds
        return dict(
            calls=self.calls, rows=self.rows,
            certified_seconds=self.certified_seconds,
            chunked_seconds=self.chunked_seconds,
            chunked_overhead_fraction=(self.chunked_seconds / total) if total else 0.0,
            max_read_deviation=self.max_read_deviation,
            max_state_deviation=self.max_state_deviation,
            max_strength_deviation=self.max_strength_deviation,
            calls_with_live_mismatch=self.calls_with_live_mismatch,
            live_mismatches=self.live_mismatches,
            by_read_path=self.by_read_path,
            cross_checks=self.cross_checks[:6],
            failures=self.failures[:8],
            host_original_seconds=self.host_original,
            host_replay_seconds=self.host_replay,
        )


USE_GRAPHS = True
SAMPLE_CALLS = int(os.environ.get('X44_SAMPLE', '0')) or 10 ** 9
COLLECTOR = Collector()
_DUMPED = []


def _dump(k, q, v, append_logit, pop, error_coef, cfg, result, *, read=None,
          final=None, call=None):
    """Save one call's inputs and the certified outputs for offline diagnosis."""
    path = os.environ.get('X44_DUMP')
    torch.save(dict(k=k.detach().cpu(), q=q.detach().cpu(), v=v.detach().cpu(),
                    append_logit=append_logit.detach().cpu(), pop=pop.detach().cpu(),
                    error_coef=error_coef.detach().cpu(), out=result[0].detach().cpu(),
                    trace=dict(alloc=result[2].alloc_target.cpu(),
                               matched=result[2].matched.cpu(),
                               pop_fired=result[2].pop_fired.cpu(),
                               pop_target=result[2].pop_target.cpu(),
                               w_on=result[2].w_on.cpu()),
                    keys=result[1].native.keys.detach().cpu(),
                    values=result[1].native.values.detach().cpu(),
                    strength=result[1].native.strength.detach().cpu(),
                    live=result[1].native.live.detach().cpu(),
                    chunked_read=None if read is None else read.detach().cpu(),
                    chunked_values=None if final is None else final.values.detach().cpu(),
                    call=call,
                    cfg=dict(key_dim=cfg.key_dim, value_dim=cfg.value_dim,
                             slots=cfg.slots, decay_rate=cfg.decay_rate,
                             free_threshold=cfg.free_threshold,
                             usage_decay=cfg.usage_decay)), path)
    print(f'  X44 dump written: {path}', flush=True)
_INSTALLED = []


_ERROR_COEF_CHECKED = {}


def _error_coef_nonzero(error_coef) -> bool:
    """Residual coefficient check, memoised: ``.item()`` inside the training loop
    would stall the pipeline once per scan call."""
    key = (error_coef.data_ptr(), tuple(error_coef.shape), error_coef.dtype)
    if key not in _ERROR_COEF_CHECKED:
        _ERROR_COEF_CHECKED[key] = bool(float(error_coef.abs().max()) != 0.0)
    return _ERROR_COEF_CHECKED[key]


def schedule_from_call(k, append_logit, pop, error_coef, trace) -> DecisionSchedule:
    """Pack the certified kernel's decisions into a ``DecisionSchedule``.

    The three magnitudes are the ones the kernel actually applied:

    * ``gate = sigmoid(append_logit + error_coef * error)`` -- production sets
      ``error_coef = 0`` (a zero buffer in the arm), so the gate is
      ``sigmoid(append_logit)``; a non-zero coefficient makes the gate depend on
      the write residual, which this path does not reproduce and therefore
      refuses rather than approximates;
    * ``beta = trace.w_on`` -- the kernel overwrites the write-magnitude buffer
      with the write-enable flag and interpolates with that;
    * ``pop_factor = 1 - pop`` -- the kernel's ``exact_strength * (1 - pop)``.
    """
    batch, tokens = k.shape[0], k.shape[1]
    def flat(x):
        return x.reshape(batch, tokens, -1)[:, :, 0].contiguous()
    if _error_coef_nonzero(error_coef):
        raise ValueError('append gate depends on the write residual '
                         '(error_coef != 0): the chunked replay does not model it')
    fired = flat(trace.pop_fired)
    target = flat(trace.pop_target)
    pop_slot = torch.where(fired, target, torch.full_like(target, -1))
    pop_factor = torch.where(fired, 1.0 - flat(pop), torch.ones_like(flat(pop)))
    return DecisionSchedule(alloc=flat(trace.alloc_target),
                            gate=torch.sigmoid(flat(append_logit)),
                            correct=flat(trace.matched),
                            beta=flat(trace.w_on).to(torch.float32),
                            pop_slot=pop_slot, pop_factor=pop_factor)


_GRAPHS = {}


def graph_for(cfg, batch, tokens, chunk, key_dim, value_dim):
    """Captured chunked scan for one shape: the arm repeats the same shapes, so a
    replay costs one host launch instead of ~3 000."""
    if not USE_GRAPHS or not torch.cuda.is_available():
        return None
    key = (batch, tokens, chunk, cfg.slots, key_dim, value_dim)
    if key not in _GRAPHS:
        _GRAPHS[key] = graph_scan.ChunkedScanGraph(
            cfg, batch, tokens, chunk_size=chunk,
            row_block=max(1, min(batch, 2 ** 24 // (chunk * cfg.slots))),
            key_dim=key_dim, value_dim=value_dim)
    return _GRAPHS[key]


def replay(k, q, v, append_logit, pop, error_coef, cfg, result,
           chunk_size=None) -> tuple:
    """Replay one certified scan call through the chunked scan and record parity."""
    read_certified, state_certified = result[0], result[1]
    trace = result[2]
    batch, tokens = k.shape[0], k.shape[1]
    schedule = schedule_from_call(k, append_logit, pop, error_coef, trace)
    key_flat = k.reshape(batch, tokens, -1).contiguous()
    query_flat = q.reshape(batch, tokens, -1).contiguous()
    value_flat = v.reshape(batch, tokens, -1).contiguous()
    native = getattr(state_certified, 'native', state_certified)
    # The arm scans each window from a RESET store, so the replay starts empty;
    # the kernel's returned store is the exit state it must reproduce.
    chunk = int(chunk_size or schedule_recommended_chunk(cfg.slots, batch))
    chunk_start = torch.cuda.Event(enable_timing=True)
    chunk_end = torch.cuda.Event(enable_timing=True)
    graph = graph_for(cfg, batch, tokens, chunk, cfg.key_dim, cfg.value_dim)
    chunk_start.record()
    if graph is not None:
        read, final = graph(key_flat, query_flat, value_flat, schedule, copy_out=False)
    else:
        read, final = chunked_cell.chunked_tape_scan(
            key_flat, query_flat, value_flat, schedule, cfg, chunk_size=chunk,
            row_block=max(1, min(batch, 2 ** 24 // (chunk * cfg.slots))))
    chunk_end.record()
    # Parity is measured on a sample of calls: the reductions are extra device
    # work outside the timed interval and would otherwise dominate the overhead
    # of the harness rather than measure it.
    read_path = ('batched' if os.environ.get('X40_BATCHED_READS', '0') == '1'
                 else 'certified')
    sampled = COLLECTOR.sampled_count(read_path) < SAMPLE_CALLS
    if sampled:
        # Order the comparison against the certified path explicitly: a
        # deviation measured across an unsynchronised producer stream is a
        # measurement artefact, and only sampled calls pay for the sync.
        torch.cuda.synchronize()
        read_error = (read.reshape(batch, 1, tokens, cfg.value_dim) - read_certified
                      ).abs().max()
        state_error = (final.values - native.values.reshape(batch, cfg.slots, -1)
                       ).abs().max()
        strength_error = (final.strength - native.strength.reshape(batch, cfg.slots)
                          ).abs().max()
        live_mismatch = (final.live != native.live.reshape(batch, cfg.slots)).sum()
    else:
        read_error = state_error = strength_error = torch.zeros((), device=k.device)
        live_mismatch = torch.zeros((), dtype=torch.int64, device=k.device)
    if os.environ.get('X44_DUMP') and not _DUMPED and float(read_error) > 1e-2:
        _dump(k, q, v, append_logit, pop, error_coef, cfg, result,
              read=read, final=final, call=COLLECTOR.calls)
        _DUMPED.append(1)
    if os.environ.get('X44_CROSSCHECK') and float(read_error) > 1e-2:
        # A large read deviation is either a real divergence or the arm's
        # returned buffer being reused; re-run the buffer-independent per-token
        # implementation to tell the two apart.
        from . import sequential_replay
        reference_read, _ = sequential_replay.sequential_tape_scan(
            key_flat, query_flat, value_flat, schedule, cfg)
        COLLECTOR.cross_checks.append(dict(
            call=COLLECTOR.calls,
            chunked_vs_certified=float(read_error),
            chunked_vs_sequential=float((read - reference_read).abs().max()),
            sequential_vs_certified=float((reference_read.reshape(
                batch, 1, tokens, cfg.value_dim) - read_certified).abs().max())))
    return (read_error, state_error, strength_error, live_mismatch,
            (chunk_start, chunk_end), sampled)


def schedule_recommended_chunk(slots: int, rows: int) -> int:
    """Measured optimum (see RESULT_CHUNK_SCAN.md): small stores want long chunks,
    large stores want short ones because the per-chunk trajectories are
    ``rows x chunk x slots`` and stop fitting in cache."""
    footprint = max(1, rows) * slots
    if footprint <= 32_768:          # <= ~2 MB of fp32 per trajectory
        return 256
    if footprint <= 131_072:
        return 128
    if footprint <= 524_288:
        return 48
    return 16


def instrument(max_calls: int | None = None, chunk_size: int | None = None) -> Collector:
    """Patch ``velocity_candidate.scan`` so every forward is also replayed chunked.

    Idempotent; returns the collector that accumulates the evidence.
    """
    from ..x42_pile_scale import velocity_candidate as arm

    if any(module is arm for module in _INSTALLED):
        return COLLECTOR
    original = arm.scan

    def wrapped(k, q, v, beta, pop, append, cfg, append_coef, logits, match_coef,
                error_coef, policy):
        host_original_start = time.perf_counter()
        certified_start = torch.cuda.Event(enable_timing=True)
        certified_end = torch.cuda.Event(enable_timing=True)
        certified_start.record()
        result, occupancy, gates = original(k, q, v, beta, pop, append, cfg,
                                            append_coef, logits, match_coef,
                                            error_coef, policy)
        certified_end.record()
        COLLECTOR.host_original += time.perf_counter() - host_original_start
        host_replay_start = time.perf_counter()
        if max_calls is None or COLLECTOR.calls + len(COLLECTOR.pending) < max_calls:
            try:
                with torch.no_grad():
                    deviation = replay(k, q, v, append, pop, error_coef, cfg, result,
                                       chunk_size)
                COLLECTOR.record((certified_start, certified_end), deviation[4],
                                 deviation[0], deviation[1], deviation[2], deviation[3],
                                 k.shape[0], read_path=('batched' if os.environ.get(
                                     'X40_BATCHED_READS', '0') == '1' else 'certified'),
                                 sampled=deviation[5])
            except Exception as failure:            # pragma: no cover - evidence path
                COLLECTOR.failures.append(f'{type(failure).__name__}: {failure}')
        COLLECTOR.host_replay += time.perf_counter() - host_replay_start
        return result, occupancy, gates

    wrapped._x44_chunk_replay = True
    arm.scan = wrapped
    _INSTALLED.append(arm)
    return COLLECTOR


__all__ = ['Collector', 'COLLECTOR', 'instrument', 'replay', 'schedule_from_call',
           'schedule_recommended_chunk']
