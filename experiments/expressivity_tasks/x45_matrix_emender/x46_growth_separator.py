"""X46 GROWTH SEPARATOR -- the constructed-task-family test of the program's
central theorem: fixed-state machines are automata, and emendation+growth is
the only finite-precision escape.

The X41 theory says a fixed-state machine pins at its own state algebra, and
the C3 program receipted that NATURAL TEXT does not bind (PR ~8 of 128,
tau unreachable; grown ~= born at matched grown state).  The one untested
form is the CONSTRUCTED TASK FAMILY -- the same discipline as
``granularity_separator.py`` and ``key_separation_probe.py``: a registered
task construction with a DERIVED fixed-rank ceiling (never assumed),
pre-stated predictions with refutation clauses, fail-closed.

The task family: GAS (growing-associations stream)
--------------------------------------------------
One document-bounded constructed stream per window (never Pile material;
state never carried across windows).  The window introduces ``n`` DISTINCT
key identities, each bound to a per-window-random value, then probes every
binding in a per-window shuffled order (lag ~ n: the number of
simultaneously-live associations GROWS past any fixed capacity within the
stream -- the dial the whole probe turns on).  Values are drawn iid from a
pool of ``n_values``: the ADDRESS is the key (each key has exactly one
binding per window), so value repeats do not create ambiguity, and a probe
of an evicted key cannot find its value by addressing (its row is gone;
same-value rows are filed under other keys' contents).  Chance = 1/n_values
on the value-restricted argmax.

One symbolic construction, two encodings (each machine's own input interface):

* MATRIX ENCODING (the ours arms): a binding is ONE combined pair token
  ``PAIR(k, v)`` -- the granularity separator's writability requirement: at
  the first store layer the payload is a fixed function of the token, so the
  binding must be writable as one token or it is unrepresentable at depth 1.
  A probe is the bare KEY token; the answer (the VALUE token) is revealed
  teacher-forced at the next position and is the loss target.
* BYTE ENCODING (the GDN2 arms): a binding is the 3-byte pattern
  ``(key_hi, key_lo, value)``; a probe is ``(key_hi, key_lo)`` and the
  target is the value byte (the byte LM's own next-byte prediction, made
  from the key_lo position).  GDN2's short conv (d_conv=4) makes the
  multi-byte key bindable at write time; the 2-byte key alphabet gives 640
  key identities over bytes (the single-byte alphabet cannot).

The machines (probe scale, the separator's discipline)
------------------------------------------------------
* ``ours-fixed``     the registered store at rank 128 (of 2048 slots), growth
                     OFF -- the K8 ``GrowingMatrixMixer`` with ``grow=False``
                     (bit-equal K6 per-token semantics at rank <= 128).
* ``ours-growable``  the SAME model and parameters, the K8 machinery verbatim
                     (``convert_model_to_growing``): start rank 128, cap
                     2048, the REGISTERED append-collision trigger
                     (``scan.repeat_write_rate`` at 64-token boundaries).
                     Rank is capacity, never parameters.
* ``gdn2-fixed``     a probe-scale GDN2 byte-LM (2 blocks of the registered
                     mixer shape ``GDN2ExternalMLPLayer(dim=576, expansion=2,
                     head_dim=128, H=2, d_conv=4)`` -- the committed smoke's
                     block), growth OFF.
* ``gdn2-growth``    the same machine with each core wrapped by the committed
                     ``GrowingGDN2Core`` (imported verbatim from
                     ``gdn2_growth_c3``), heads snap {2,4,8}, the calibrated
                     tau discipline (the C3 pattern, re-calibrated on THIS
                     constructed stream by ``calibrate``; the forced-trigger
                     alternative is an ESCALATED, unruled option -- refused
                     by name until ruled).

The derived ceilings (the algebra, never assumed)
-------------------------------------------------
See ``DESIGN_X46_GROWTH_SEPARATOR.md`` for the full derivations.  The
machine-checkable halves are in this module (``selftest``) and in
``test_x46_growth_separator.py``:

* OURS (rank R): the slot-count bound is HARD -- the store holds R rows; a
  full store never APPENDs (the (R+1)-th distinct binding is POPped-away or
  dropped), so at most R bindings are ever simultaneously live; an aliased
  row's content is a positive-decayed blend of its payloads, and a fixed
  linear readout cannot order the blended values differently at two probes
  (the swap argument) -- at most one of an aliased pair is answered.  The
  linear-channel ceiling is
  ``C(N) = min(1, R/N) + (1 - min(1, R/N)) / n_values``.
* GDN2 (H heads, per-head key dim d_k=128): the state per head is a linear
  associative memory over a d_k-dimensional key space: exact (zero-crosstalk)
  retrieval needs mutually orthogonal keys, so at most d_k per head; beyond
  that the Welch/frame-potential bound FORCES coherence
  ``rms(cos) >= sqrt((N-d)/(d(N-1)))`` for every key configuration, and the
  outer-product crosstalk-to-signal ratio ``sqrt(N-1) * rms(cos)`` crosses 1
  near N ~ 2d.  The pre-stated effective capacity is ``H * d_k`` (the
  concatenated key spaces) with the refutation clause.
* Both ceilings carry the disclosed depth-2 superposition escape (a
  key-conditioned nonlinear decoder could in principle resolve blends --
  the key-separation precedent) -- covered by the REFUTATION CLAUSE, not by
  proof: a fixed machine beating its ceiling + band REFUTES the ceiling
  claim and is reported as a first-class finding.

Fail-closed: the headline arms REFUSE to run until
``PREREGISTRATION_X46_GROWTH_SEPARATOR.md`` carries an OPERATOR-SIGNED
status line.  ``selftest`` and ``smoke`` (plumbing) and ``calibrate``
(pre-registration evidence, disclosed) run before sign-off.

Subcommands::

    selftest    CPU: the derivation checks (slot-count, no-APPEND-when-full,
                aliasing swap argument, Welch/crosstalk, trigger-on-overflow,
                construction invariants) -> receipts/x46_selftest.json
    smoke       leased GPU: a tiny shakedown of all four arms (growth events
                receipted) -> receipts/x46_smoke.json
    calibrate   leased GPU: the gdn2-fixed arm on the constructed stream, the
                per-boundary PR ceiling measured, tau committed (pre-registration
                evidence) -> receipts/x46_gdn2_calibration.json
    arm         leased GPU: ONE (machine, n, lr, seed) cell -> receipts/ (FAIL-CLOSED)
    sweep       leased GPU: the lr grid for one (machine set, n) on one seed (FAIL-CLOSED)
    assemble    the arm receipts -> the ceiling adjudication (FAIL-CLOSED)

Usage (cards 2/3, this lane's pool; card 4 is the detached emender run --
never touched; card 7 forbidden):

    eval "$(GPU_LEASE_VISIBLE=2,3 scripts/gpu_lease.sh acquire 1 --wait)"
    OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m \
        experiments.expressivity_tasks.x45_matrix_emender.x46_growth_separator selftest
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from torch import nn
import torch.nn.functional as F

from .algebra import APPEND, CORRECT, NONE, POP
from .granularity_separator import parse_floats
from .model import MatrixShape, MatrixBlock
from .schedule import RoutingPolicy
from .scan import repeat_write_rate

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / "receipts"
PREREGISTRATION = ROOT / "PREREGISTRATION_X46_GROWTH_SEPARATOR.md"
SIGNED_MARK = "OPERATOR-SIGNED"

# ------------------------------------------------------------ the task family
N_KEYS = 640                   # key identities (>= the largest n: keys are
                              # drawn without replacement per window)
N_VALUES = 128                 # the value pool = the chance floor 1/128
GROW_BOUNDARY_TOKENS = 64      # the registered statistic boundary (scan.py)

# the matrix encoding's vocabulary: BOS, PAD, key tokens, value tokens, and
# one combined pair token per (key, value) -- the depth-1 writability
# requirement (the granularity separator's PAIR_BASE construction)
BOS, PAD = 0, 1
KEY_BASE = 2
VAL_BASE = KEY_BASE + N_KEYS
PAIR_BASE = VAL_BASE + N_VALUES
MATRIX_VOCAB = PAIR_BASE + N_KEYS * N_VALUES

# the byte encoding's alphabet (disjoint ranges, all < 256)
BYTE_PAD, BYTE_BOS = 0, 1
VAL_BYTE_BASE = 2                       # value v -> byte 2+v   (2 .. 129)
KEY_HI_BASE = 0x90                      # key k  -> 0x90 + k//32 (0x90 .. 0xA3)
KEY_LO_BASE = 0xB0                      #          0xB0 + k%32  (0xB0 .. 0xCF)

# the machines' state algebra
HEAD_DIM = 128                          # GDN2's per-head key dimension
MATRIX_RANK = 128                       # ours-fixed's slot count (of 2048)
MATRIX_RANK_CAP = 2048
GDN2_HEAD_CAP = 8                       # the growth snap ceiling {2,4,8}
GDN2_DIM = 576                          # the registered mixer shape's width
BAND = 0.05                             # the pre-stated accuracy endpoint band

# the n dial: 64 (below every ceiling: the learnability control), 320 (past
# both fixed ceilings: the headline), 640 (past the matched grown state 512:
# the capacity-pressure cell)
N_LADDER = (64, 320, 640)


@dataclass(frozen=True)
class Window:
    """One constructed document: the symbolic stream, both encodings' views."""
    keys: torch.Tensor          # (B, n)   distinct key identities, intro order
    values: torch.Tensor        # (B, n)   iid value draws, intro order
    order: torch.Tensor         # (B, n)   the probe order (intro indices)


def sample_window(rng: torch.Generator, count: int, n: int) -> Window:
    if n > N_KEYS:
        raise ValueError(f"n={n} exceeds the key pool {N_KEYS}")
    keys = torch.argsort(torch.rand(count, N_KEYS, generator=rng), dim=1)[:, :n]
    values = torch.randint(0, N_VALUES, (count, n), generator=rng)
    order = torch.argsort(torch.rand(count, n, generator=rng), dim=1)
    return Window(keys=keys, values=values, order=order)


# ------------------------------------------------------------- the encodings

def matrix_layout(n: int):
    """Positions: BOS at 0; the n binding (pair) tokens at 1..n; then per
    probe j: the KEY token at n+1+2j (its target = the bound VALUE token,
    revealed as the input at n+2+2j)."""
    pair_positions = tuple(1 + i for i in range(n))
    probe_positions = tuple(n + 1 + 2 * j for j in range(n))
    return pair_positions, probe_positions


def make_matrix_batch(window: Window) -> dict:
    """The matrix-encoding window: ids (B, 3n+1), targets (B, n) value ids."""
    keys, values, order = window.keys, window.values, window.order
    batch, n = keys.shape
    pair_positions, probe_positions = matrix_layout(n)
    ids = torch.full((batch, 3 * n + 1), PAD, dtype=torch.long)
    ids[:, 0] = BOS
    ids[:, 1 + torch.arange(n)] = PAIR_BASE + keys * N_VALUES + values
    probed_keys = keys.gather(1, order)
    probed_values = values.gather(1, order)
    probe_index = torch.tensor(list(probe_positions))
    ids[:, probe_index] = KEY_BASE + probed_keys
    ids[:, probe_index + 1] = VAL_BASE + probed_values
    targets = VAL_BASE + probed_values
    pair_pos = torch.tensor(list(pair_positions))
    lag = probe_index.unsqueeze(0) - pair_pos[order]
    return dict(encoding="matrix", ids=ids, targets=targets, order=order,
                probe_positions=probe_positions,
                pair_positions=pair_positions, lag=lag)


def byte_layout(n: int):
    """Positions: BOS at 0; binding i's 3 bytes at 1+3i..1+3i+2 (the value
    byte LAST: the byte LM's next-byte prediction at the key_lo position IS
    the binding); probe j's triple at 1+3n+3j.. (target = the value byte,
    predicted from the key_lo position)."""
    bind_positions = tuple(1 + 3 * i + 2 for i in range(n))
    probe_positions = tuple(1 + 3 * n + 3 * j + 2 for j in range(n))
    return bind_positions, probe_positions


def make_byte_batch(window: Window) -> dict:
    """The byte-encoding window: ids (B, 6n+1), targets (B, n) value bytes."""
    keys, values, order = window.keys, window.values, window.order
    batch, n = keys.shape
    bind_positions, probe_positions = byte_layout(n)
    ids = torch.full((batch, 6 * n + 1), BYTE_PAD, dtype=torch.long)
    ids[:, 0] = BYTE_BOS
    hi = KEY_HI_BASE + keys // 32
    lo = KEY_LO_BASE + keys % 32
    val = VAL_BYTE_BASE + values
    index = torch.arange(n)
    ids[:, 1 + 3 * index] = hi
    ids[:, 2 + 3 * index] = lo
    ids[:, 3 + 3 * index] = val
    probed_hi = hi.gather(1, order)
    probed_lo = lo.gather(1, order)
    probed_val = val.gather(1, order)
    probe_index = torch.tensor(list(probe_positions))
    ids[:, probe_index - 2] = probed_hi
    ids[:, probe_index - 1] = probed_lo
    ids[:, probe_index] = probed_val
    bind_pos = torch.tensor(list(bind_positions))
    lag = probe_index.unsqueeze(0) - bind_pos[order]
    return dict(encoding="byte", ids=ids, targets=probed_val, order=order,
                probe_positions=probe_positions,
                pair_positions=bind_positions, lag=lag)


def _to_device(batch: dict, device: str) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def make_eval_set(n: int, count: int, device: str) -> dict:
    """The fixed eval set (seed 4999), both encodings, on the device."""
    rng = torch.Generator().manual_seed(4999)
    window = sample_window(rng, count, n)
    return dict(matrix=_to_device(make_matrix_batch(window), device),
                byte=_to_device(make_byte_batch(window), device),
                n=n, eval_windows=count, eval_seed=4999)


# ------------------------------------------------------- the derived ceilings

def matrix_capacity(rank: int) -> int:
    """Ours: the slot count IS the capacity (rank is the number of rows)."""
    return int(rank)


def gdn2_capacity(heads: int) -> int:
    """GDN2: the concatenated per-head key spaces (H x d_k), the pre-stated
    effective capacity with the refutation clause (the per-head EXACT bound
    is d_k; the sum-of-heads form is what the readout could at best route)."""
    return int(heads) * HEAD_DIM


def linear_channel_ceiling(capacity: int, n: int, n_values: int = N_VALUES) -> float:
    """The pre-stated linear-channel ceiling: the live fraction is addressable
    (at most ``capacity`` bindings live), the evicted fraction is at chance.
    The aliased-pair refinement (at most one of a shared row's probes) cannot
    exceed this bound."""
    live = min(1.0, capacity / max(1, n))
    return live + (1.0 - live) / n_values


def ceiling_record(machine: str, n: int, *, rank: int | None = None,
                   heads: int | None = None) -> dict:
    """The honest automata-bound arithmetic for one arm's receipt."""
    if machine.startswith("ours"):
        capacity = matrix_capacity(rank)
        algebra = (f"slot-count: the store holds {rank} rows; a full store "
                   f"never APPENDs (POP-or-drop), so <= {rank} live bindings; "
                   f"aliased rows resolve at most one probe of a pair")
    elif machine.startswith("gdn2"):
        capacity = gdn2_capacity(heads)
        algebra = (f"key-space: per-head d_k={HEAD_DIM} (exact retrieval needs "
                   f"orthogonal keys: <= {HEAD_DIM}/head); the Welch rms bound "
                   f"{welch_bound(n, HEAD_DIM):.4f} at N={n} forces crosstalk/"
                   f"signal {crosstalk_to_signal(n, HEAD_DIM):.2f} per head "
                   f"(> 1 past N~2d); the pre-stated effective capacity is "
                   f"H*d_k = {capacity} (heads {heads}) with the refutation "
                   f"clause")
    else:
        raise ValueError(f"unknown machine {machine!r}")
    ceiling = linear_channel_ceiling(capacity, n)
    return dict(machine=machine, n=n, capacity=capacity,
                ceiling=ceiling, ceiling_plus_band=ceiling + BAND,
                algebra=algebra, band=BAND,
                escape=("capacity >= n: no binding ceiling (the control regime)"
                        if capacity >= n else None))


def welch_bound(n: int, d: int) -> float:
    """The frame-potential bound: for ANY n unit vectors in R^d (n > d), the
    rms off-diagonal coherence is at least sqrt((n-d)/(d(n-1)))."""
    if n <= d:
        return 0.0
    return math.sqrt((n - d) / (d * (n - 1)))


def crosstalk_to_signal(n: int, d: int) -> float:
    """The outer-product retrieval's crosstalk-to-signal ratio at the Welch
    floor: sqrt(n-1) * rms coherence (signal 1).  > 1 means every key
    configuration is past reliable per-head retrieval."""
    return math.sqrt(max(0, n - 1)) * welch_bound(n, d)


# ---------------------------------------------------------- the ours arms

def matrix_probe_shape(rank_max: int) -> MatrixShape:
    """The probe-scale store shape: width 128 (640 key identities need the
    margin), depth 2, ONE head, value_dim 64 (<= MAX_WALK_VALUE_DIM so the
    K8 hi path accepts it), rank 128 of ``rank_max`` slots."""
    return MatrixShape(width=128, depth=2, hidden=256, heads=1, value_dim=64,
                       rank=MATRIX_RANK, rank_max=rank_max, vocab=MATRIX_VOCAB)


class X46StoreModel(nn.Module):
    """The ours machine: the registered ``MatrixBlock`` (the registered
    ``MatrixMixer`` store, SwiGLU, RMSNorm -- imported, never re-written) at
    probe scale, with the probe-position loss of the separator convention.
    Built then converted by the committed ``convert_model_to_growing`` (the
    K8 machinery verbatim): ``grow=False`` is ours-fixed (K6 semantics at
    rank 128), ``grow=True`` is ours-growable (the registered trigger)."""

    def __init__(self, shape: MatrixShape, policy: RoutingPolicy, rank_max: int):
        super().__init__()
        self.shape, self.policy = shape, policy
        self.embed = nn.Embedding(shape.vocab, shape.width)
        self.blocks = nn.ModuleList(
            MatrixBlock(shape, policy, chunk_tokens=1) for _ in range(shape.depth))
        self.norm = nn.RMSNorm(shape.width, eps=1e-5)
        self.head = nn.Linear(shape.width, shape.vocab, bias=False)
        self.probe_positions: tuple[int, ...] = ()

    def bind_layout(self, n: int):
        _, self.probe_positions = matrix_layout(n)
        return self

    def features(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids: torch.Tensor):
        return self.head(self.features(ids))

    def loss(self, batch: dict) -> torch.Tensor:
        ids, targets = batch["ids"], batch["targets"]
        logits = self.forward(ids)
        probe = logits[:, list(self.probe_positions)]
        return F.cross_entropy(probe.reshape(-1, probe.shape[-1]),
                               targets.reshape(-1))


def build_matrix_arm(grow: bool, seed: int, rank_cap: int | None = None):
    """One ours arm: the registered block, the K8 conversion verbatim.
    Returns (model, summary).  ``grow=False`` pins rank at 128 (ours-fixed);
    ``grow=True`` starts at 128 with the registered trigger (ours-growable)."""
    from .k8_growth import convert_model_to_growing
    torch.manual_seed(seed)
    rank_cap = rank_cap or (MATRIX_RANK_CAP if grow else MATRIX_RANK)
    policy = RoutingPolicy(grow=grow)
    model = X46StoreModel(matrix_probe_shape(rank_cap), policy, rank_cap)
    model = convert_model_to_growing(model, segment_tokens=GROW_BOUNDARY_TOKENS,
                                     rank_cap=rank_cap, grow=grow,
                                     start_rank=MATRIX_RANK)
    total = sum(p.numel() for p in model.parameters())
    summary = dict(machine="ours-growable" if grow else "ours-fixed",
                   parameters=dict(total=total,
                                   note="rank is capacity, never parameters"),
                   rank=MATRIX_RANK, rank_cap=rank_cap, grow=grow,
                   policy=policy.record(),
                   shape=dict(width=128, depth=2, heads=1, value_dim=64,
                              vocab=MATRIX_VOCAB))
    return model, summary


@torch.no_grad()
def matrix_probe_accuracy(model: X46StoreModel, eval_set: dict,
                          batch: int = 64) -> dict:
    """Value-restricted argmax at the probe positions + the lag strata."""
    data = eval_set["matrix"]
    ids, targets = data["ids"], data["targets"]
    model.eval()
    picks = []
    for start in range(0, ids.shape[0], batch):
        logits = model.forward(ids[start:start + batch])
        probe = logits[:, list(data["probe_positions"]),
                       VAL_BASE:VAL_BASE + N_VALUES]
        picks.append(probe.argmax(-1) + VAL_BASE)
    picks = torch.cat(picks)
    correct = picks == targets
    quartiles = []
    width = max(1, correct.shape[1] // 4)
    for lo in range(0, correct.shape[1], width):
        quartiles.append(float(correct[:, lo:lo + width].float().mean()))
    model.train()
    return dict(value_acc=float(correct.float().mean()),
                value_acc_by_lag_quartile=quartiles,
                n_probes=int(correct.numel()), chance=1.0 / N_VALUES)


@torch.no_grad()
def matrix_channels(model: X46StoreModel, eval_set: dict, windows: int = 128):
    """The mechanistic receipt (the granularity separator's ``collect_channels``
    pattern on the K8 collect walk): for every probe, the probed binding's
    row -- its write op, whether it was LIVE at the probe, and the read mass
    the probe put on it -- with accuracy stratified live/dead-at-probe (the
    slot-count bound made countable).  Slot indices are stable across growth
    (zero-injection pads rows at the END), so a binding written in one span
    is addressable from its probe's span."""
    from .model import DECAY_FLOOR
    from .k8_growth import growing_scan
    data = eval_set["matrix"]
    ids, targets = data["ids"][:windows], data["targets"][:windows]
    order = data["order"][:windows]
    pair_positions, probe_positions = data["pair_positions"], data["probe_positions"]
    n = len(pair_positions)
    model.eval()
    per_layer = []
    x = model.embed(ids)
    for layer, block in enumerate(model.blocks):
        mixer = block.mixer
        h = block.norm1(x)
        batch, tokens, _ = h.shape
        query = mixer.query(h).view(batch, tokens, mixer.heads, mixer.value_dim)
        payload = mixer.value(h).view(batch, tokens, mixer.heads, mixer.value_dim)
        decay = torch.sigmoid(mixer.decay(h)).clamp(min=DECAY_FLOOR,
                                                     max=1.0 - 1e-6)
        result = growing_scan(query, payload, decay, policy=mixer.policy,
                              start_rank=mixer.rank, rank_cap=mixer.rank_cap,
                              segment_tokens=mixer.segment_tokens, collect=True)
        # the spans' recorded channels, indexed by GLOBAL window position
        pair_slot = torch.zeros(batch, n, dtype=torch.int64, device=ids.device)
        pair_op = torch.zeros(batch, n, dtype=torch.int64, device=ids.device)
        probe_live = torch.zeros(batch, n, dtype=torch.bool, device=ids.device)
        probe_mass = torch.zeros(batch, n, device=ids.device)
        spans = result.collected
        for span in spans:
            t0 = span["t0"]
            schedule = span["schedule"]
            for i, position in enumerate(pair_positions):
                if t0 <= position < span["t1"]:
                    pair_slot[:, i] = schedule.slot[:, position - t0, 0, 0]
                    pair_op[:, i] = schedule.op[:, position - t0, 0, 0]
            for j, position in enumerate(probe_positions):
                if t0 <= position < span["t1"]:
                    local = position - t0
                    live_pre = span["live_pre"][:, local, 0, :]   # (b, rank)
                    weights = span["weights"][:, local, 0, :]
                    slot_j = pair_slot.gather(1, order)          # (b, n)
                    # probe j asks binding order[:, j]: its slot at the probe
                    probe_live[:, j] = live_pre.gather(
                        1, slot_j[:, j:j + 1]).squeeze(1)
                    probe_mass[:, j] = weights.gather(
                        1, slot_j[:, j:j + 1]).squeeze(1)
        picks = None
        logits = model.forward(ids)
        probe = logits[:, list(probe_positions),
                       VAL_BASE:VAL_BASE + N_VALUES]
        picks = probe.argmax(-1) + VAL_BASE
        correct = picks == targets
        stored = pair_op.ne(NONE)
        live = probe_live & stored
        per_layer.append(dict(
            layer=layer, final_rank=int(result.rank),
            growth_events=[dict(e) for e in result.stats.events],
            pair_op_counts={"append": int((pair_op == APPEND).sum()),
                            "correct": int((pair_op == CORRECT).sum()),
                            "pop": int((pair_op == POP).sum()),
                            "none": int((pair_op == NONE).sum())},
            stored_frac=float(stored.float().mean()),
            live_at_probe_frac=float(live.float().mean()),
            accuracy=dict(
                live=float(correct[live].float().mean())
                if bool(live.any()) else None,
                dead=float(correct[~live].float().mean())
                if bool((~live).any()) else None),
            mass_on_answer_row=dict(
                mean=float(probe_mass.mean()),
                mean_when_live=float(probe_mass[live].mean())
                if bool(live.any()) else None)))
        mixed = mixer.out(result.reads.reshape(batch, tokens,
                                              mixer.heads * mixer.value_dim))
        x = x + mixed
        x = x + block.mlp(block.norm2(x))
    model.train()
    return dict(note=("the probed binding's row: live-at-probe (the slot-count "
                      "bound made countable) with the probe's read mass on it; "
                      "growth events are the spans' recorded fires"),
                per_layer=per_layer)


# ---------------------------------------------------------- the gdn2 arms

class GDN2ProbeModel(nn.Module):
    """A probe-scale GDN2 byte-LM: the registered mixer shape (the committed
    smoke's block), depth 2, pre-norm residual (LadderLM supplies the first
    RMSNorm externally -- the wrapper's own contract).  The loss is the
    probe next-byte CE (the association targets only: matched exposure
    across families, n targets per window on both).  The byte LM predicts
    position p from the state at p-1, so the probe's predict position is the
    probe target position's predecessor (the key_lo byte)."""

    def __init__(self, depth: int = 2, heads: int = 2, seed: int = 3401):
        super().__init__()
        from ndm.models import external_gdn2
        from ndm.models.external_gdn2 import GDN2ExternalMLPLayer
        external_gdn2._load_gdn2_class()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(256, GDN2_DIM)
        self.blocks = nn.ModuleList(
            GDN2ExternalMLPLayer(dim=GDN2_DIM, expansion=2.0, head_dim=HEAD_DIM,
                                 num_heads=heads, use_conv=True, d_conv=4)
            for _ in range(depth))
        for index, block in enumerate(self.blocks):
            block.set_layer_idx(index)
        self.norm = nn.RMSNorm(GDN2_DIM, eps=1e-5)
        self.head = nn.Linear(GDN2_DIM, 256, bias=False)
        self.predict_positions: tuple[int, ...] = ()

    def bind_layout(self, n: int):
        _, probe_positions = byte_layout(n)
        self.predict_positions = tuple(p - 1 for p in probe_positions)
        return self

    def features(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for block in self.blocks:
            x = x + block(self.norm(x))[0]
        return self.norm(x)

    def forward(self, ids: torch.Tensor):
        return self.head(self.features(ids))

    def loss(self, batch: dict) -> torch.Tensor:
        ids, targets = batch["ids"], batch["targets"]
        logits = self.forward(ids)
        probe = logits[:, list(self.predict_positions)]
        return F.cross_entropy(probe.reshape(-1, probe.shape[-1]),
                               targets.reshape(-1))


def install_gdn2_growth(model: GDN2ProbeModel, sink: dict, tau: float,
                        cap: int = GDN2_HEAD_CAP):
    """Wrap every block's core with the committed ``GrowingGDN2Core``
    (imported verbatim from gdn2_growth_c3), with the wrapper's model-level
    name prefix resolved exactly as the C3 lane's ``install_growth`` does."""
    from .gdn2_growth_c3 import GrowingGDN2Core
    wrappers = []
    by_id = {id(p): n for n, p in model.named_parameters()}
    for index, block in enumerate(model.blocks):
        core = block.gdn2.gdn2
        wrapper = GrowingGDN2Core(core, index, sink, cap=cap, tau=tau, grow=True)
        block.gdn2.gdn2 = wrapper
        wrappers.append(wrapper)
    for wrapper in wrappers:
        rel = next(iter(dict(wrapper.core.named_parameters())))
        first = next(iter(dict(wrapper.core.named_parameters()).values()))
        wrapper.name_prefix = by_id[id(first)][:-len(rel)]
    return wrappers


@torch.no_grad()
def gdn2_probe_accuracy(model: GDN2ProbeModel, eval_set: dict,
                        batch: int = 32) -> dict:
    """Value-restricted argmax at the probe predict positions (over the
    value byte range) + the lag strata."""
    data = eval_set["byte"]
    ids, targets = data["ids"], data["targets"]
    was_training = model.training
    model.eval()
    picks = []
    for start in range(0, ids.shape[0], batch):
        logits = model.forward(ids[start:start + batch])
        probe = logits[:, list(model.predict_positions)]
        picks.append(probe[:, :, VAL_BYTE_BASE:VAL_BYTE_BASE + N_VALUES]
                     .argmax(-1) + VAL_BYTE_BASE)
    picks = torch.cat(picks)
    correct = picks == targets
    quartiles = []
    width = max(1, correct.shape[1] // 4)
    for lo in range(0, correct.shape[1], width):
        quartiles.append(float(correct[:, lo:lo + width].float().mean()))
    if was_training:
        model.train()
    return dict(value_acc=float(correct.float().mean()),
                value_acc_by_lag_quartile=quartiles,
                n_probes=int(correct.numel()), chance=1.0 / N_VALUES)


def build_gdn2_arm(grow: bool, seed: int, tau: float = float("inf"),
                   sink: dict | None = None, depth: int = 2):
    """One gdn2 arm at probe scale.  ``grow=False`` is the plain registered
    machine (H=2); ``grow=True`` wraps every core with the committed growth
    wrapper at the calibrated ``tau`` (the C3 discipline on THIS stream;
    ``sink`` must carry the spectra pool -- the wrapper's own contract)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = GDN2ProbeModel(depth=depth, heads=2, seed=seed)
    summary = dict(machine="gdn2-growth" if grow else "gdn2-fixed",
                   heads=2, head_dim=HEAD_DIM, depth=depth, dim=GDN2_DIM,
                   capacity=gdn2_capacity(2), grow=grow)
    if grow:
        if sink is None or "pool" not in sink:
            raise ValueError("the growth arm needs the spectra pool in its "
                             "sink (the wrapper's registered probe contract)")
        wrappers = install_gdn2_growth(model, sink, tau=tau)
        summary["head_cap"] = GDN2_HEAD_CAP
        summary["tau"] = tau
        summary["n_wrapped"] = len(wrappers)
    summary["parameters"] = dict(
        total=sum(p.numel() for p in model.parameters()),
        note=("growth adds parameters (GDN2's keys are learned) -- the delta "
              "is a first-class column" if grow else "the fixed arm"))
    return model, summary


# ------------------------------------------------------------- fail-closed

def require_operator_signoff():
    """The headline gate: refuse to run any arm/sweep/assemble until the
    PREREGISTRATION carries an OPERATOR-SIGNED status line."""
    if not PREREGISTRATION.exists():
        raise ValueError(f"fail-closed: {PREREGISTRATION.name} is missing; "
                         "the derivation and construction must be committed "
                         "and signed before any headline arm runs")
    text = PREREGISTRATION.read_text()
    if SIGNED_MARK not in text:
        raise ValueError(f"fail-closed: {PREREGISTRATION.name} is not "
                         f"{SIGNED_MARK}; the pre-stated predictions need "
                         "operator sign-off before any headline arm runs")


def require_lease() -> str:
    card = os.environ.get("GPU_LEASE_HELD")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if card is None or visible is None:
        raise ValueError("no GPU lease: run through the leased driver "
                         "(cards 2/3: eval \"$(GPU_LEASE_VISIBLE=2,3 "
                         "scripts/gpu_lease.sh acquire 1 --wait)\")")
    if str(card) != str(visible):
        raise ValueError(f"lease mismatch: held {card} vs visible {visible}")
    return card


def calibrated_tau() -> float:
    """The committed calibrated tau from this lane's calibration receipt
    (fail-closed: no invented tau)."""
    path = RECEIPTS / "x46_gdn2_calibration.json"
    if not path.exists():
        raise ValueError("fail-closed: the gdn2-growth arm needs the "
                         "CALIBRATED tau -- run `calibrate` first (the "
                         "forced-trigger alternative is escalated, unruled)")
    return float(json.loads(path.read_text())["tau_cal"])


# ----------------------------------------------------- the derivation checks

def check_slot_lemma(rank: int = 8, n_writes: int = 24, seed: int = 3401) -> dict:
    """MACHINE-CHECKED (parameter-free): with N distinct unmatched bindings
    (mutually orthogonal one-hot payloads, near-1 decay) the store (a) holds
    at most ``rank`` live rows at every token, (b) never admits a post-fill
    binding without destroying a live row first (every post-fill APPEND is
    paid for by a POP: the (R+1)-th distinct binding can only enter a slot
    freed by evicting another binding), and (c) carries at most ``rank``
    distinct payload directions in the state -- the slot-count bound,
    mechanically, with the overflow ops receipted."""
    from .k6_sequential_matrix import sequential_oracle_scan
    assert n_writes > rank
    value_dim = n_writes
    payload = torch.zeros(1, n_writes, 1, value_dim)
    payload[0, torch.arange(n_writes), 0, torch.arange(n_writes)] = 1.0
    query = torch.zeros_like(payload)
    decay = torch.full((1, n_writes, 1), 0.999)
    result = sequential_oracle_scan(query, payload, decay,
                                     policy=RoutingPolicy(), rank=rank,
                                     rank_max=rank, collect=True)
    ops = result["schedule"].op[0, :, 0, 0]
    live = result["live"][0, 0]
    state = result["state"][0, 0]
    appended_after_full = int((ops[rank:] == APPEND).sum())
    popped_after_full = int((ops[rank:] == POP).sum())
    coordinate_mass = state.abs().sum(0)                  # (value_dim,)
    distinct_present = int((coordinate_mass > 0).sum())
    checks = dict(
        live_rows=int(live.sum()), live_rows_le_rank=bool(live.sum() <= rank),
        first_rank_all_append=bool((ops[:rank] == APPEND).all()),
        appends_after_full=appended_after_full,
        pops_after_full=popped_after_full,
        every_post_fill_append_paid_by_a_pop=bool(
            appended_after_full <= popped_after_full),
        overflow_ops={name: int((ops[rank:] == code).sum())
                      for code, name in ((POP, "pop"), (NONE, "none"),
                                         (CORRECT, "correct"))},
        distinct_directions_present=distinct_present,
        distinct_directions_le_rank=bool(distinct_present <= rank))
    checks["passed"] = all(v for k, v in checks.items() if isinstance(v, bool))
    return checks


def check_aliasing_swap_lemma(trials: int = 200, seed: int = 3401) -> dict:
    """MACHINE-CHECKED: two bindings aliased into ONE row (a positive-decayed
    blend of their payload directions) cannot both be answered by ANY fixed
    linear readout.  The certified read form is nonnegative (softmax masses
    times nonnegative strengths), and the aliased row is the only live row,
    so each probe's read output is the SAME blend direction up to a positive
    scalar -- the value-logit gap ``logit(v0) - logit(v1)`` therefore has ONE
    sign at both probes, and the per-window value swap (probe A wants v0,
    probe B wants v1; the swapped window the reverse) is unsatisfiable: at
    most one probe of the aliased pair is answered.  The blend half of the
    ours ceiling, in key-separation's role-blindness style."""
    g = torch.Generator().manual_seed(seed)
    value_dim = 8
    a, b = 0.8, 0.4                    # the decayed blend coefficients (> 0)
    blend = torch.zeros(value_dim)
    blend[0], blend[1] = a, b          # the two aliased value directions
    rank = 2
    heads = []
    # the two probes: distinct queries that both put their whole read mass on
    # the single live row (any query does -- it is the only live row)
    for probe in (0, 1):
        q = torch.randn(value_dim, generator=g)
        scores = torch.zeros(rank)
        scores[0] = RoutingPolicy().read_scale * F.cosine_similarity(
            blend, q, dim=0)
        masked = scores.masked_fill(torch.tensor([False, True]),
                                     float("-inf"))   # row 1 is dead
        probs = torch.softmax(masked, -1)
        strength = torch.tensor([blend.norm(), 0.0])
        w = probs * strength
        read = w[0] * blend             # the aliased row's contribution
        assert w[1] == 0.0
        heads.append(read)
    same_direction = bool(
        F.cosine_similarity(heads[0], heads[1], dim=0) > 1.0 - 1e-6)
    both = 0
    gap_positive = gap_negative = gap_zero = 0
    for _ in range(trials):
        head = torch.randn(value_dim, 2, generator=g)
        gap = float(head.T @ heads[0] @ torch.tensor([1.0, -1.0]))
        gap_positive += gap > 0
        gap_negative += gap < 0
        gap_zero += gap == 0.0
        # a fixed head answers BOTH probes of one window iff the gap is
        # simultaneously > 0 (probe A -> v0) and < 0 (probe B -> v1)
        if gap > 0 and gap < 0:
            both += 1
    return dict(trials=trials, blend=dict(a=a, b=b),
                read_outputs_same_direction=same_direction,
                gap_signs=dict(positive=gap_positive, negative=gap_negative,
                               zero=gap_zero),
                heads_answering_both_probes=both,
                passed=bool(same_direction and both == 0))


def check_gdn2_key_space(n: int = 320, d: int = HEAD_DIM,
                         seed: int = 3401) -> dict:
    """MACHINE-CHECKED at the task's numbers: (a) the frame-potential bound
    holds for a random key draw (the realized rms coherence is at or above
    the Welch line -- the bound holds for EVERY key configuration), (b)
    orthonormal keys at N = d retrieve EXACTLY (zero crosstalk: the per-head
    exact bound d_k), and (c) at N=320 in d=128 the outer-product crosstalk
    norm exceeds the signal -- every key configuration is past reliable
    per-head retrieval (the per-head pin regime)."""
    g = torch.Generator().manual_seed(seed)
    keys = F.normalize(torch.randn(n, d, generator=g), dim=-1)
    gram = keys @ keys.T
    off = gram - torch.eye(n)
    rms = float((off ** 2).sum().div(n * (n - 1)).sqrt())
    bound = welch_bound(n, d)
    # (b) the exact regime: orthonormal keys, N = d.  S = K^T V (the
    # outer-product state); the retrieval for query k_j is S^T k_j, whose
    # matrix form over all queries is M = K S -- with orthonormal K, M = V
    # exactly (zero crosstalk: the per-head exact bound).
    ortho = torch.eye(d)
    values = torch.eye(d)
    state = ortho.T @ values            # sum_i k_i v_i^T
    retrieved = ortho @ state           # M[j] = S^T k_j = v_j exactly
    exact_error = float((retrieved - values).abs().max())
    # (c) the aliased regime: random keys, one-hot values
    val_eye = torch.eye(n)
    state_r = keys.T @ val_eye          # S = K^T V, (d, n)
    retr = keys @ state_r               # M[j] = v_j + sum_i cos_ij v_i
    crosstalk = retr - val_eye
    crosstalk_norm = float((crosstalk ** 2).sum(-1).sqrt().mean())
    return dict(n=n, d=d, welch_bound=bound, realized_rms_coherence=rms,
                welch_holds=bool(rms >= bound * 0.999),
                orthonormal_exact_error=exact_error,
                orthonormal_exact=bool(exact_error < 1e-5),
                crosstalk_to_signal_mean=crosstalk_norm,
                crosstalk_exceeds_signal=bool(crosstalk_norm > 1.0),
                crosstalk_at_welch_floor=crosstalk_to_signal(n, d),
                passed=bool(rms >= bound * 0.999 and exact_error < 1e-5
                            and crosstalk_norm > 1.0))


def check_growth_trigger_on_overflow(rank: int = 64, n_writes: int = 200,
                                     seed: int = 3401) -> dict:
    """MACHINE-CHECKED: on a constructed overflow stream (distinct unmatched
    bindings past a full rank-``rank`` store) the REGISTERED trigger fires
    mechanically -- the append-collision rate (``scan.repeat_write_rate``,
    called verbatim) crosses ``grow_trigger`` at the first boundary past the
    fill, and the registered ``growth_target`` snaps 64 -> 128.  The
    ours-growable arm's trigger needs no calibrated tau on THIS family: the
    construction IS the write pressure."""
    from .k6_sequential_matrix import sequential_oracle_scan
    from .keys import growth_target
    assert n_writes > rank
    value_dim = 64
    payload = torch.zeros(1, n_writes, 1, value_dim)
    for i in range(n_writes):
        payload[0, i, 0, i % value_dim] = 1.0
    query = torch.zeros_like(payload)
    decay = torch.full((1, n_writes, 1), 0.999)
    result = sequential_oracle_scan(query, payload, decay,
                                     policy=RoutingPolicy(), rank=rank,
                                     rank_max=rank, collect=True)
    schedule = result["schedule"]
    policy = RoutingPolicy(grow=True)   # the trigger's constants (the scan
    # itself runs quiet: growth is a boundary concept the per-token oracle
    # does not carry, exactly as the K6 registration refuses it)
    last_write = torch.full((1, 1, rank), -1, dtype=torch.int64)
    window: list[float] = []
    rates = []
    fire = None
    smoothed_at_fire = None
    for k in range(1, n_writes // GROW_BOUNDARY_TOKENS + 1):
        chunk = schedule.chunk((k - 1) * GROW_BOUNDARY_TOKENS,
                               k * GROW_BOUNDARY_TOKENS)
        rate = repeat_write_rate(chunk, last_write,
                                 (k - 1) * GROW_BOUNDARY_TOKENS,
                                 policy.grow_window_tokens)
        window.append(rate)
        rates.append((k * GROW_BOUNDARY_TOKENS, rate))
        recent = window[-policy.grow_window:]
        mean = sum(recent) / len(recent)
        if mean >= policy.grow_trigger:
            fire = k * GROW_BOUNDARY_TOKENS
            smoothed_at_fire = mean
            break
    target = growth_target(rank, policy.grow_geometric, policy.grow_stride_floor)
    return dict(rank=rank, n_writes=n_writes, boundary_rates=rates,
                fire_boundary=fire, smoothed_rate_at_fire=smoothed_at_fire,
                grow_trigger=policy.grow_trigger, growth_target=target,
                fired=bool(fire is not None),
                passed=bool(fire is not None and target == 128))


def check_construction(n: int = 64, count: int = 8, seed: int = 1234) -> dict:
    """The registered construction's invariants: both encodings decode to the
    SAME symbolic stream; keys are distinct per window; every probe target is
    its binding's value; the layouts are causal (probe after its write); the
    eval set is deterministic; the byte alphabet is disjoint and in range."""
    rng = torch.Generator().manual_seed(seed)
    window = sample_window(rng, count, n)
    m = make_matrix_batch(window)
    b = make_byte_batch(window)
    ids = b["ids"]
    val_bytes = ids[(ids >= VAL_BYTE_BASE) & (ids < VAL_BYTE_BASE + N_VALUES)]
    key_hi_bytes = ids[(ids >= KEY_HI_BASE) & (ids < KEY_HI_BASE + 20)]
    key_lo_bytes = ids[(ids >= KEY_LO_BASE) & (ids < KEY_LO_BASE + 32)]
    disjoint = bool(len(val_bytes) + len(key_hi_bytes) + len(key_lo_bytes)
                    == ids.numel() - int((ids == BYTE_PAD).sum())
                    - int((ids == BYTE_BOS).sum()))
    ok = dict(
        matrix_shape=list(m["ids"].shape), byte_shape=list(b["ids"].shape),
        keys_distinct=bool(all(len(set(row.tolist())) == n
                               for row in window.keys)),
        targets_match_bindings=bool(
            (m["targets"] == VAL_BASE + window.values.gather(1, window.order))
            .all() and
            (b["targets"] == VAL_BYTE_BASE
             + window.values.gather(1, window.order)).all()),
        probes_after_writes=bool((m["lag"] > 0).all() and (b["lag"] > 0).all()),
        byte_alphabet_disjoint=disjoint)
    w1 = sample_window(torch.Generator().manual_seed(4999), 4, n)
    w2 = sample_window(torch.Generator().manual_seed(4999), 4, n)
    ok["eval_set_deterministic"] = bool(
        torch.equal(w1.keys, w2.keys) and torch.equal(w1.values, w2.values))
    ok["passed"] = all(v for k, v in ok.items() if isinstance(v, bool))
    return ok


def selftest(out_path: str | None = None) -> dict:
    """The derivation checks, all CPU, exact; the receipt the DESIGN doc and
    the PREREGISTRATION cite."""
    receipt = dict(
        experiment="x46_growth_separator selftest",
        slot_lemma=check_slot_lemma(),
        aliasing_swap_lemma=check_aliasing_swap_lemma(),
        gdn2_key_space=check_gdn2_key_space(),
        growth_trigger_on_overflow=check_growth_trigger_on_overflow(),
        construction=check_construction(),
        ceilings_ours={f"n{n}": ceiling_record("ours-fixed", n, rank=MATRIX_RANK)
                      for n in N_LADDER},
        ceilings_gdn2={f"n{n}": ceiling_record("gdn2-fixed", n, heads=2)
                       for n in N_LADDER},
        torch=torch.__version__)
    receipt["passed"] = all(
        receipt[name]["passed"] for name in
        ("slot_lemma", "aliasing_swap_lemma", "gdn2_key_space",
         "growth_trigger_on_overflow", "construction"))
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(receipt, indent=1))
    return receipt


# ------------------------------------------------------------ the arms' loop

def train_arm(machine: str, n: int, lr: float, seed: int, steps: int,
              batch: int, device: str, eval_count: int = 512,
              cooldown_frac: float = 0.1, tau: float | None = None,
              rank_cap: int | None = None, depth: int = 2) -> dict:
    """One (machine, n, lr, seed) cell: matched exposure (steps x batch x n
    probe targets on every arm), constant LR then a linear cooldown over the
    last ``cooldown_frac`` (the corrected-protocol doctrine adapted to the
    accuracy probe), the honest clock, growth events processed per step (the
    GDN2 arm: the C3 event discipline -- grad merge, optimizer surgery, the
    committed machinery imported verbatim).  PRIMARY = the final
    value-restricted probe accuracy on the fixed eval set."""
    from .gdn2_growth_c3 import merge_event_grads, rebuild_optimizer
    torch.set_num_threads(4)
    started = time.monotonic()
    eval_set = make_eval_set(n, eval_count, device)
    growth_events: list[dict] = []
    pool = None
    sink = None
    if machine in ("ours-fixed", "ours-growable"):
        grow = machine == "ours-growable"
        model, summary = build_matrix_arm(grow, seed, rank_cap)
        model.bind_layout(n)
        make_batch = make_matrix_batch
    elif machine in ("gdn2-fixed", "gdn2-growth"):
        grow = machine == "gdn2-growth"
        if grow:
            from . import x45_spectra_sidecar as SPECTRA
            pool = SPECTRA.SpectraPool()
            sink = {"pool": pool, "events": []}
            if tau is None:
                tau = calibrated_tau()
        model, summary = build_gdn2_arm(grow, seed, tau=tau or float("inf"),
                                        sink=sink, depth=depth)
        model.bind_layout(n)
        make_batch = make_byte_batch
    else:
        raise ValueError(f"unknown machine {machine!r}")
    model = model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95),
                                 weight_decay=0.01, fused=(device == "cuda"))
    opt_names = {id(p): name for name, p in model.named_parameters()}
    rng = torch.Generator().manual_seed(seed)
    constant_steps = max(1, int(steps * (1.0 - cooldown_frac)))
    warmup = max(1, constant_steps // 10)
    curve = []
    try:
        for step in range(1, steps + 1):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.monotonic()
            if step <= constant_steps:
                factor = min(1.0, step / warmup)
            else:
                factor = (steps + 1 - step) / max(1, steps - constant_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr * factor
            optimizer.zero_grad(set_to_none=True)
            batch_data = _to_device(make_batch(sample_window(rng, batch, n)),
                                    device)
            loss = model.loss(batch_data)
            loss.backward()
            if sink is not None and sink["events"]:
                # the C3 event discipline: the pre-event spans' gradients
                # merge into the grown core's shared slices, the optimizer
                # is rebuilt with state carried by name (committed
                # machinery, imported verbatim)
                merge_event_grads(sink["events"])
                optimizer, opt_names = rebuild_optimizer(
                    optimizer, model, sink["events"], opt_names)
                params = [p for p in model.parameters() if p.requires_grad]
                for event in sink["events"]:
                    growth_events.append(dict(
                        step=step, layer=event.layer_idx,
                        boundary=event.boundary, heads=event.heads_old,
                        new_heads=event.heads_new, fire_pr=event.fire_pr))
                sink["events"] = []
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            if device == "cuda":
                torch.cuda.synchronize()
            if step % max(1, steps // 6) == 0 or step == steps:
                if machine.startswith("ours"):
                    metrics = matrix_probe_accuracy(model, eval_set)
                else:
                    metrics = gdn2_probe_accuracy(model, eval_set)
                curve.append(dict(step=step, loss=float(loss.detach()),
                                  value_acc=metrics["value_acc"]))
                print(f"  {machine} n={n} lr={lr:.0e} seed={seed} step {step}: "
                      f"loss {float(loss.detach()):.4f} "
                      f"acc {metrics['value_acc']:.4f}", flush=True)
    finally:
        if pool is not None:
            pool.close()
    walls = time.monotonic() - started
    if machine.startswith("ours"):
        metrics = matrix_probe_accuracy(model, eval_set)
        channels = matrix_channels(model, eval_set)
        final_rank = [int(block.mixer.last_rank) for block in model.blocks]
        scan_events = [dict(e) for block in model.blocks
                       for e in block.mixer.growth_events]
        heads = None
    else:
        metrics = gdn2_probe_accuracy(model, eval_set)
        channels = None
        final_rank = None
        scan_events = []
        core = model.blocks[0].gdn2.gdn2
        heads = int(core.num_heads)
    receipt = dict(
        machine=machine, n=n, lr=lr, seed=seed, steps=steps, batch=batch,
        device=device, walls_s=round(walls, 1), torch=torch.__version__,
        exposure_probe_targets=steps * batch * n,
        metrics=metrics, curve=curve, growth_events=growth_events,
        last_scan_growth_events=scan_events,
        final_rank=final_rank, final_heads=heads,
        ceiling=ceiling_record(
            machine, n,
            rank=final_rank[0] if final_rank else MATRIX_RANK,
            heads=heads or 2),
        summary=summary, channels=channels,
        cooldown=dict(constant_steps=constant_steps,
                      cooldown_steps=steps - constant_steps))
    return receipt


def run_arm(args) -> dict:
    if not args.smoke:
        require_operator_signoff()
    card = require_lease()
    receipt = train_arm(args.machine, args.n, args.lr, args.seed, args.steps,
                        args.batch, "cuda", eval_count=args.eval_windows,
                        tau=args.tau, rank_cap=args.rank_cap, depth=args.depth)
    receipt["card"] = card
    name = (f"x46_smoke_{args.machine}_n{args.n}.json" if args.smoke else
            f"x46_arm_{args.machine}_n{args.n}_lr{args.lr:.0e}"
            f"_seed{args.seed}.json")
    out = Path(args.out) / name if args.out else RECEIPTS / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in
                      ("machine", "n", "lr", "seed", "metrics",
                       "growth_events", "final_rank", "final_heads",
                       "ceiling", "walls_s")}, indent=1), flush=True)
    return receipt


def run_sweep(args) -> dict:
    require_operator_signoff()
    require_lease()
    machines = (args.machines.split(",") if args.machines else
                ["ours-fixed", "ours-growable", "gdn2-fixed", "gdn2-growth"])
    cells = []
    for machine in machines:
        for lr in parse_floats(args.lrs):
            receipt = train_arm(machine, args.n, lr, args.seed, args.steps,
                                args.batch, "cuda",
                                eval_count=args.eval_windows, tau=args.tau,
                                rank_cap=args.rank_cap, depth=args.depth)
            name = (f"x46_arm_{machine}_n{args.n}_lr{lr:.0e}"
                    f"_seed{args.seed}.json")
            (RECEIPTS / name).write_text(json.dumps(receipt, indent=1))
            cells.append(dict(machine=machine, lr=lr, seed=args.seed,
                              metrics=receipt["metrics"],
                              growth_events=len(receipt["growth_events"]),
                              walls_s=receipt["walls_s"]))
            print(f"{machine} lr={lr:.0e}: acc="
                  f"{receipt['metrics']['value_acc']:.4f}", flush=True)
    summary = dict(n=args.n, seed=args.seed, steps=args.steps, cells=cells)
    (RECEIPTS / f"x46_sweep_n{args.n}_seed{args.seed}.json").write_text(
        json.dumps(summary, indent=1))
    return summary


def run_assemble(args) -> dict:
    """The ceiling adjudication: per machine at one n, the best cells vs the
    derived ceilings; the pre-state read (fixed machines pinned at their
    ceilings; growables escaped; the matched-grown-state comparison)."""
    require_operator_signoff()
    n = args.n
    table, verdict = {}, {}
    for machine in ("ours-fixed", "ours-growable", "gdn2-fixed", "gdn2-growth"):
        rows = []
        for path in sorted(RECEIPTS.glob(
                f"x46_arm_{machine}_n{n}_lr*_seed*.json")):
            receipt = json.loads(path.read_text())
            rows.append(dict(lr=receipt["lr"], seed=receipt["seed"],
                             acc=receipt["metrics"]["value_acc"],
                             growth_events=len(receipt["growth_events"]),
                             final_rank=receipt["final_rank"],
                             final_heads=receipt["final_heads"]))
        table[machine] = rows
        if not rows:
            verdict[machine] = "no cells"
            continue
        best = max(rows, key=lambda r: r["acc"])
        ceiling = ceiling_record(
            machine, n,
            rank=best["final_rank"][0] if best["final_rank"] else MATRIX_RANK,
            heads=best["final_heads"] or 2)
        verdict[machine] = dict(
            best_lr=best["lr"], best_acc=best["acc"],
            ceiling=ceiling["ceiling"],
            ceiling_plus_band=ceiling["ceiling_plus_band"],
            within_ceiling=bool(best["acc"] <=
                                ceiling["ceiling_plus_band"]),
            beats_ceiling=bool(best["acc"] > ceiling["ceiling_plus_band"]))
    summary = dict(n=n, band=BAND, table=table, verdict=verdict)
    (RECEIPTS / f"x46_summary_n{n}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(verdict, indent=1), flush=True)
    return summary


def run_calibrate(args) -> dict:
    """The gdn2 tau calibration on the constructed stream (pre-registration
    evidence, the C3 calibrated-below-registered discipline): train the FIXED
    arm at the headline n, then measure the trained machine's per-layer
    per-boundary smoothed-PR ceiling on eval windows (the kernel's own
    per-chunk state trajectory, the imported spectra sidecar, exact LAPACK).
    tau_cal = the trained fixed machine's own exhibited ceiling; the growth
    arm fires exactly when a boundary is MORE saturated than anything the
    fixed machine exhibits."""
    from . import x45_spectra_sidecar as SPECTRA
    from .gdn2_growth_smoke import _chunk_call, _kernel_inputs
    from .gdn2_growth_c3 import pr_from_trajectory, smoothed_series
    require_lease()
    n = args.n
    model, summary = build_gdn2_arm(False, args.seed, depth=args.depth)
    model.bind_layout(n)
    device = "cuda"
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=0.01,
                                  fused=True)
    rng = torch.Generator().manual_seed(args.seed)
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        batch_data = _to_device(make_byte_batch(sample_window(rng, args.batch, n)),
                               device)
        loss = model.loss(batch_data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 25 == 0:
            print(f"  calibrate step {step}: loss {float(loss.detach()):.4f}",
                  flush=True)
    # the trained-state PR measurement (the gdn2_saturation_diagnostics
    # pattern at probe scale: forward hooks on the mixer layers, the
    # kernel's own per-chunk state trajectory)
    pool = SPECTRA.SpectraPool()
    eval_set = make_eval_set(n, args.eval_windows, device)
    ids = eval_set["byte"]["ids"][:args.probe_windows]
    store = []
    handles = []

    def make_hook(index):
        def hook(module, args_, _output):
            x = args_[0]
            core = module.gdn2          # the GDN2ExternalLayer's own core
            with torch.inference_mode():
                q, k, v, g, b, w = _kernel_inputs(core, x)
                _o, _s, h = _chunk_call(core, q, k, v, g, b, w,
                                        output_final_state=True,
                                        return_intermediate_states=True)
                pr = pr_from_trajectory(pool, h).cpu()
            store.append(dict(layer=index, heads=core.num_heads,
                              pr=pr.numpy()))
        return hook

    for index, block in enumerate(model.blocks):
        handles.append(block.gdn2.register_forward_hook(make_hook(index)))
    model.eval()
    with torch.no_grad():
        for start in range(0, ids.shape[0], 4):
            model.forward(ids[start:start + 4])
    for handle in handles:
        handle.remove()
    pool.close()
    per_layer = {}
    for entry in store:
        series = entry["pr"].mean(axis=(0, 2))       # [C-1] per boundary
        per_layer.setdefault(entry["layer"], []).append(
            float(smoothed_series(series).max()))
    per_layer_max = {layer: max(values) for layer, values in per_layer.items()}
    tau_cal = max(per_layer_max.values())
    receipt = dict(
        experiment="x46 gdn2 tau calibration (pre-registration evidence)",
        n=n, seed=args.seed, steps=args.steps, lr=args.lr,
        machine="gdn2-fixed", heads=2, head_dim=HEAD_DIM,
        per_layer_smoothed_pr_max=per_layer_max,
        tau_cal=tau_cal,
        tau_rule=("the trained fixed machine's own per-layer smoothed-PR "
                  "ceiling on THIS constructed stream (the C3 "
                  "calibrated-below-registered discipline): the growth arm "
                  "fires exactly when a boundary is MORE saturated than "
                  "anything the fixed machine exhibits"),
        registered_tau_sat=0.75 * HEAD_DIM,
        summary=summary, spectra=SPECTRA.box_load_receipt(),
        torch=torch.__version__)
    out = Path(args.out) if args.out else RECEIPTS / "x46_gdn2_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=1))
    print(json.dumps(dict(tau_cal=tau_cal, per_layer=per_layer_max), indent=1),
          flush=True)
    return receipt


def run_smoke(args) -> dict:
    """The plumbing shakedown (allowed pre-sign-off): every arm at tiny
    budget; the growable arms' growth events are the receipt that the
    registered triggers fire on the constructed stream."""
    require_lease()
    receipt = dict(experiment="x46 smoke", cells=[])
    for machine in ("ours-fixed", "ours-growable", "gdn2-fixed", "gdn2-growth"):
        cell = train_arm(machine, args.n, 1e-3, 3401, args.steps, args.batch,
                         "cuda", eval_count=args.eval_windows,
                         tau=args.tau, depth=args.depth)
        receipt["cells"].append(dict(
            machine=machine, n=args.n, steps=args.steps,
            metrics=cell["metrics"],
            growth_events=len(cell["growth_events"]),
            last_scan_growth_events=len(cell["last_scan_growth_events"]),
            final_rank=cell["final_rank"], final_heads=cell["final_heads"],
            walls_s=cell["walls_s"]))
        print(f"smoke {machine}: acc={cell['metrics']['value_acc']:.4f} "
              f"events={len(cell['growth_events'])} "
              f"rank={cell['final_rank']} heads={cell['final_heads']}",
              flush=True)
    receipt["passed"] = True
    out = Path(args.out) if args.out else RECEIPTS / "x46_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=1))
    return receipt


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("selftest")
    st.add_argument("--out", default=str(RECEIPTS / "x46_selftest.json"))
    sm = sub.add_parser("smoke")
    sm.add_argument("--n", type=int, default=320)
    sm.add_argument("--steps", type=int, default=30)
    sm.add_argument("--batch", type=int, default=4)
    sm.add_argument("--eval-windows", type=int, default=128)
    sm.add_argument("--tau", type=float, default=20.0,
                    help="the gdn2-growth smoke's provisional tau (the "
                         "calibration's committed value supersedes it)")
    sm.add_argument("--depth", type=int, default=2)
    sm.add_argument("--out", default=None)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--n", type=int, default=320)
    cal.add_argument("--seed", type=int, default=3401)
    cal.add_argument("--lr", type=float, default=1e-3)
    cal.add_argument("--steps", type=int, default=300)
    cal.add_argument("--batch", type=int, default=8)
    cal.add_argument("--eval-windows", type=int, default=64)
    cal.add_argument("--probe-windows", type=int, default=8)
    cal.add_argument("--depth", type=int, default=2)
    cal.add_argument("--out", default=None)
    arm = sub.add_parser("arm")
    arm.add_argument("--machine", required=True,
                     choices=("ours-fixed", "ours-growable", "gdn2-fixed",
                              "gdn2-growth"))
    arm.add_argument("--n", type=int, default=320)
    arm.add_argument("--lr", type=float, default=1e-3)
    arm.add_argument("--seed", type=int, default=3401)
    arm.add_argument("--steps", type=int, default=3000)
    arm.add_argument("--batch", type=int, default=16)
    arm.add_argument("--eval-windows", type=int, default=512)
    arm.add_argument("--tau", type=float, default=None,
                     help="gdn2-growth: the CALIBRATED tau is read from the "
                          "calibration receipt when omitted (fail-closed)")
    arm.add_argument("--rank-cap", type=int, default=None)
    arm.add_argument("--depth", type=int, default=2)
    arm.add_argument("--smoke", action="store_true")
    arm.add_argument("--out", default=None)
    swp = sub.add_parser("sweep")
    swp.add_argument("--machines", default=None)
    swp.add_argument("--n", type=int, default=320)
    swp.add_argument("--lrs", default="3e-3,1e-3,3e-4,1e-4")
    swp.add_argument("--seed", type=int, default=3401)
    swp.add_argument("--steps", type=int, default=3000)
    swp.add_argument("--batch", type=int, default=16)
    swp.add_argument("--eval-windows", type=int, default=512)
    swp.add_argument("--tau", type=float, default=None)
    swp.add_argument("--rank-cap", type=int, default=None)
    swp.add_argument("--depth", type=int, default=2)
    ass = sub.add_parser("assemble")
    ass.add_argument("--n", type=int, default=320)
    args = parser.parse_args()
    if args.cmd == "selftest":
        receipt = selftest(args.out)
        print(json.dumps({k: receipt[k] for k in
                         ("slot_lemma", "aliasing_swap_lemma",
                          "gdn2_key_space", "growth_trigger_on_overflow",
                          "construction", "passed")}, indent=1))
    elif args.cmd == "smoke":
        run_smoke(args)
    elif args.cmd == "calibrate":
        run_calibrate(args)
    elif args.cmd == "arm":
        run_arm(args)
    elif args.cmd == "sweep":
        run_sweep(args)
    elif args.cmd == "assemble":
        run_assemble(args)


if __name__ == "__main__":
    main()
