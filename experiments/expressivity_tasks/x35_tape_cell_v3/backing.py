"""Elastic empty-origin backing; untouched logical rows are implicit zeros.

This is a prefix representation, NOT live-row pooling: dead touched rows remain
resident, preserving their exact payload/strength/age bits. Imported states and
explicit recording scans use the established dense representation. Training
from an empty origin uses prefix-sized recorded replay. Public access
promotes the entire handle, including metadata, once, before returning tensors.
"""
from dataclasses import replace
import weakref
import torch
from ..x30_tape_cell.naive_reference import TapeState
from ..x30_tape_cell.routing_trace import RoutingTrace

FIELDS = ('keys', 'values', 'strength', 'live', 'age', 'usage')


def configuration(cfg):
    return (cfg.slots, cfg.key_dim, cfg.value_dim, cfg.n_heads,
            cfg.decay_rate, cfg.free_threshold)


def pad_rows(x, slots):
    if x.shape[2] == slots:
        return x
    shape = list(x.shape)
    shape[2] = slots - x.shape[2]
    return torch.cat((x, x.new_zeros(shape)), dim=2)


class PrefixNative(TapeState):
    """Public TapeState facade whose first tensor access promotes its owner.

    Keeping the TapeState base preserves existing import/type checks. Internal
    scans explicitly unwrap ``compact`` and never inspect public fields.
    """
    def __init__(self, compact, cfg):
        self.compact = compact
        self.cfg = cfg
        self.owner = None
        self.public = None
        self.storage = None

    def __getattribute__(self, name):
        if name in FIELDS:
            owner_ref = object.__getattribute__(self, "owner")
            owner = None if owner_ref is None else owner_ref()
            if owner is not None:
                promote(owner)
                self.public = owner.native
            elif self.public is None:
                self.public = TapeState(*(pad_rows(getattr(self.compact, n), self.cfg.slots) for n in FIELDS))
                if self.storage is not None:
                    self.storage.watch(tuple(getattr(self.public,n) for n in FIELDS))
            return getattr(self.public, name)
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if name in FIELDS:
            getattr(self, name)  # Materialize before replacing a public field.
            setattr(self.public, name, value)
        else:
            object.__setattr__(self, name, value)


def promote(state):
    """Preserve native tensor aliasing after public materialization.

    Persistent metadata is expanded at the same boundary, so subsequent dense
    continuation reuses public allocations and the same metadata addresses.
    """
    facade = state.native
    if not isinstance(facade, PrefixNative):
        return state
    from .kernels import initialize_free_words
    import triton
    cfg = facade.cfg
    compact = facade.compact
    old = compact.live.shape[2]
    native = TapeState(*(pad_rows(getattr(compact, name), cfg.slots) for name in FIELDS))
    if state.metadata is not None:
        meta = list(state.metadata)
        # Slot-strided owners; unused tail fields are initialized defensively.
        for i in (0, 1, 2, 3, 4, 5, 9, 10):
            x = meta[i]
            fill = -1 if i in (0, 1, 9, 10) else (state.tokens if i == 5 else 0)
            tail = x.new_full((x.shape[0], cfg.slots - old), fill)
            meta[i] = torch.cat((x, tail), dim=1)
        for i in (13, 14):
            meta[i] = pad_rows(meta[i], cfg.slots)
        stores = native.live.shape[0] * cfg.n_heads
        meta[6] = torch.empty((stores, triton.cdiv(cfg.slots, 64)), device=native.live.device, dtype=torch.uint64)
        meta[7] = torch.empty((stores, triton.cdiv(triton.cdiv(cfg.slots, 64), 64)), device=native.live.device, dtype=torch.uint64)
        meta[8] = torch.empty(stores, device=native.live.device, dtype=torch.uint64)
        initialize_free_words[(stores,)](native.live, *meta[6:9], cfg.slots,
            triton.next_power_of_2(triton.cdiv(cfg.slots, 64)),
            triton.next_power_of_2(triton.cdiv(triton.cdiv(cfg.slots, 64), 64)))
        state.metadata = tuple(meta)
    if state.replay_storage is not None:
        state.replay_storage.watch((*(getattr(native, name) for name in FIELDS),
                                    state.history, *state.metadata))
    state.native = native
    state.configuration = configuration(cfg)
    return state


class PrefixTrace(RoutingTrace):
    """Defer only the implicit-zero tail of the public death trace."""
    def __init__(self, trace, slots):
        self._trace = trace
        self._slots = slots
        self._freed = None

    @property
    def freed(self):
        if self._freed is None:
            x = self._trace.freed
            self._freed = torch.cat((x, x.new_zeros((*x.shape[:-1], self._slots-x.shape[-1]))), dim=-1)
        return self._freed

    @freed.setter
    def freed(self, value):
        self._freed = value

    def __getattr__(self, name):
        return getattr(self._trace, name)


def eligible(inputs, state, coef, record_payload, record_all):
    if record_payload or record_all:
        return False
    return state is None or (not isinstance(state, TapeState) and isinstance(state.native, PrefixNative))


def scan(scan_impl, inputs, state, cfg, coef):
    """Run with a proven allocation horizon, retaining architectural capacity.

    A chunk allocates at most T rows and lowest-free allocation cannot exceed
    the previous prefix plus T. Thus a smaller physical capacity cannot trigger
    R-CAP unless it is the architectural S itself. No pressure substitution is
    needed, and no token may exhaust a smaller physical prefix before its last
    possible append. Growth imports the complete exact touched-row state.
    """
    from .cell_fused import FusedState
    k = inputs[0]
    b, t, h = k.shape[:3]
    if state is not None:
        if state.consumed:
            raise ValueError('FusedState handle was already consumed')
        if state.replay_storage is not None:
            state.replay_storage.freeze()
        if configuration(cfg) != configuration(state.native.cfg):
            raise ValueError('persistent metadata/config mismatch')
    old = 0 if state is None else state.native.compact.live.shape[2]
    capacity = min(cfg.slots, max(1, old + t))
    physical_cfg = replace(cfg, budget=capacity)
    if state is None:
        physical = None
    else:
        compact = state.native.compact
        if capacity == old:
            physical = FusedState(compact, state.history, state.tokens,
                                  state.metadata, configuration(physical_cfg),
                                  live_bound=state.live_bound)
        else:
            native = TapeState(*(pad_rows(getattr(compact, name), capacity) for name in FIELDS))
            # Reconstruct from exact native strengths/ages, keeping the absolute
            # clock and pressure history. This does not change logical slots.
            physical = FusedState(native, state.history, state.tokens,
                                  live_bound=state.live_bound, owns_import=True)
    out, final, trace, pressure = scan_impl(*inputs, physical, physical_cfg, coef)
    if state is not None:
        state.consumed = True
    if capacity < cfg.slots:
        facade = PrefixNative(final.native, cfg)
        final.native = facade
        facade.owner = weakref.ref(final)
        facade.storage = final.replay_storage
        final.configuration = configuration(cfg)
        trace = PrefixTrace(trace, cfg.slots)
    return out, final, trace, pressure
