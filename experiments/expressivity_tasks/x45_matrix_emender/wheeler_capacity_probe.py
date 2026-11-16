"""WHEELER-ORDER capacity probe (SPAN-RUN) -- does an ordered, run-coalescing
store hold Theta(r x avg-run) of context mass where a pointwise store holds r?

The design (``DESIGN_WHEELER_ORDER.md``, semantics confirmed by the operator
lane) in one paragraph: the corpus spine (chiSA suffix array) is a static
compressed index -- order is the address, intervals are the retrieval, runs are
the compression.  This probe tests the SAME mathematics maintained online by
the model over its own context, at oracle level, tiny models, no kernels.

Arms (both on the SEP-KV minimal separated-key base, rank r=8, one address
budget):
* **POINTWISE** -- the SEP-KV separated arm VERBATIM (``SeparatedModel`` /
  ``separated_oracle_scan`` imported read-only): free 16-D learned keys, the
  registered five-op policy (CORRECT = erase-and-replace, run length 1 -- the
  identity cache; MERGE's tau_merge=0.85 double-match makes it effectively
  inert), the certified softmax*strength read.  One address per written item.
* **ORDERED** -- the same base with the three deltas of the design: scalar 1-D
  keys (derived order: argsort of key values, insertion writes a key between
  its neighbours, no relabeling), the interval read (query key opens
  [q-w, q+w]; FULL-POPULATION softmax denominator over all live rows, rows
  outside the interval get numerator weight exactly 0 -- the Lean truncation
  law, K7 precedent; omitted mass receipted per read), and run-coalescing op
  semantics (update-in-run ACCUMULATES mass-preservingly; under capacity
  pressure MERGE coalesces the tightest adjacent pair BEFORE any POP).

Task SPAN-RUN (the operator-confirmed v6; why: a single span-wide
continuation is solvable by the identity cache alone -- one row, serial
CORRECT, the latest write carries the whole answer -- so it cannot separate
the arms; per-member continuations force run-mass).  Window:
``BOS, PAD, PAD, (m_1,v_1)..(m_S,v_S), d_1, d_2, m_j -> v_j`` -- a contiguous
SPAN of S combined (member, value) binding tokens (members drawn without
replacement from a 64-token pool, values drawn per window; no fixed token
association is memorizable), a 2-token distractor run from a separate
8-token pool, then ONE probe: a bare member token at a uniform random span
position j, whose target is the registered continuation v_j (revealed
teacher-forced after the probe).  Accuracy is the value-restricted argmax
over the 32 value tokens; chance = 1/32.  Every binding is probeable, so the
pointwise arm cannot drop span writes: it needs S distinct live rows (the
SEP-KV collision lemma), and collapses under the r=8 budget once
S + 2 (distractors) + 2 (BOS/PAD rows) > 8.

Pre-registered predictions (saturation, dilution, refutation clauses) live in
``RESULT_WHEELER_CAPACITY.md`` and are committed BEFORE any training run.
Receipts: the pointwise arm's occupancy/eviction logs (the probed binding
filed / evicted / overwritten before the probe, accuracy per stratum), the
ordered arm's run-length histogram and Wheeler-ness contiguity (does the
trained store actually build runs?), and the interval read's omitted mass.

    eval "$(GPU_LEASE_VISIBLE=4,5 scripts/gpu_lease.sh acquire 1 --wait)"
    /home/erikg/emender/.venv/bin/python -m \
        experiments.expressivity_tasks.x45_matrix_emender.wheeler_capacity_probe \
        --quick        # smoke
    /home/erikg/emender/.venv/bin/python -m \
        experiments.expressivity_tasks.x45_matrix_emender.wheeler_capacity_probe \
        --arm both    # the sweep + mechanism pass
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .key_separation_probe import (SeparatedModel, TaskSpec, causal_context)
from .model import DECAY_FLOOR, DECAY_HORIZON_TOKENS

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / "receipts"

EPS = 1e-6
CONTEXT = 4                 # the write-context window (the SEP-KV base's)
N_VALUES = 32
CHANCE = 1.0 / N_VALUES
RANK = 8                    # both arms: the one address budget under test

# ------------------------------------------------------- the ordered policy
# Registration-level constants of the ordered variant (DESIGN_WHEELER_ORDER.md
# §3-§4; new radii in key space, not inherited cosines; sensitivity open).
WIDTH = 0.25                # the interval read's radius around the query key
READ_SCALE = 2.0            # inherited registered read scale
TAU_MATCH_DIST = 0.10       # a write joins a live row's run within this radius
# tau_run = WIDTH by principle: coalesce under pressure exactly what the
# interval read would blend anyway (two rows closer than w are never resolved
# apart by ANY read, so merging them is lossless where it matters).  Note the
# ordering constraint this settles: tau_run must EXCEED tau_match_dist --
# with tau_run < tau_match_dist a coalescible pair can never exist (the
# second write of any pair closer than tau_match_dist matches the first row
# and accumulates instead of inserting two rows), the same self-deduplication
# impossibility the separator lane receipted for the registered MERGE's
# tau_merge > tau_match ordering.  Machine-checked in
# test_merge_coalesces_before_pop.
TAU_RUN = 0.25              # adjacent live rows this close coalesce (MERGE)
TAU_POP_DIST = 0.5          # POP fires only if the farthest row is this far

# op codes of the ordered scan (index terms: insert / update-in-run / coalesce
# / close-interval; see the design's op table for the mapping to the five).
O_NONE, O_INSERT, O_UPDATE, O_MERGE, O_POP = 0, 1, 2, 3, 4
O_NAMES = ("none", "insert", "update", "merge", "pop")

# The mimetic-init variant (operator-directed extension; the X42 precedent --
# experiments/expressivity_tasks/x42_pile_scale/receipts/mimetic_geometry/
# VERDICT.md, "mimicry improves materially while unfreezing does not").
# The key/query-key projections are born AT the fixed-family geometry: only
# the CURRENT-token block is non-zero, projecting onto one fixed seeded unit
# direction, so at step 0 the emitted scalar key is tanh(gain <e_token, u>) --
# a FIXED function of token identity (the registered fixed-key behaviour,
# address = identity) whose spread over the vocabulary is an invariant of the
# embeddings, NOT a training outcome.  Training may then drift the projections
# freely.  Safe birth, free drift -- the fork vs the rotation-constrained
# family (spread invariant by construction) is a registration decision with
# receipts on both sides (DESIGN_WHEELER_ORDER.md §8).
MIMETIC_GAIN = 1.5
MIMETIC_SEED = 3401


def ordered_policy_record() -> dict:
    return dict(width=WIDTH, read_scale=READ_SCALE, tau_match_dist=TAU_MATCH_DIST,
                tau_run=TAU_RUN, tau_pop_dist=TAU_POP_DIST, rank=RANK,
                mimetic=dict(gain=MIMETIC_GAIN, seed=MIMETIC_SEED))


def ordered_policy() -> dict:
    return dict(width=WIDTH, read_scale=READ_SCALE, tau_match_dist=TAU_MATCH_DIST,
                tau_run=TAU_RUN, tau_pop_dist=TAU_POP_DIST)


# ------------------------------------------------------------------ vocabulary
BOS, PAD = 0, 1
MEM_BASE, N_MEMBERS = 2, 64              # the span member pool
VAL_BASE = MEM_BASE + N_MEMBERS          # 66: the 32 value tokens
COMB_BASE = VAL_BASE + N_VALUES          # 98: combined (member, value) tokens
DIST_BASE = COMB_BASE + N_MEMBERS * N_VALUES   # 2146: the distractor pool
N_DISTRACTORS = 8
VOCAB = DIST_BASE + N_DISTRACTORS        # 2154

SPANS = (2, 4, 8, 16, 32, 64)
VALUE_SLICE = slice(VAL_BASE, VAL_BASE + N_VALUES)


def comb_id(member: int, value: int) -> int:
    return COMB_BASE + member * N_VALUES + value


def make_task(span: int) -> TaskSpec:
    """The SPAN-RUN window layout for one span length.

    ``BOS, PAD, PAD, (m_1,v_1)..(m_S,v_S), d_1, d_2, probe(m_j), reveal(v_j)``
    with the probe's 4-token write context = (last span token, d_1, d_2, m_j)
    -- the member-identity bridge the arms must learn (the SEP-AR bare-key
    doctrine).  Members are drawn without replacement (S <= 64), values are
    drawn per window with replacement (S may exceed the 32 values; the
    registered answer is still unambiguous -- it is the PROBED member's value).
    """
    window = 7 + span
    span_positions = tuple(3 + i for i in range(span))
    distractor_positions = (3 + span, 4 + span)
    probe_position = 5 + span
    reveal_position = 6 + span

    def make_batch(rng: torch.Generator, count: int):
        members = torch.argsort(torch.rand(count, N_MEMBERS, generator=rng),
                                dim=1)[:, :span]
        values = torch.randint(0, N_VALUES, (count, span), generator=rng)
        j = torch.randint(0, span, (count,), generator=rng)
        d1 = torch.randint(0, N_DISTRACTORS, (count,), generator=rng)
        d2 = (d1 + 1 + torch.randint(0, N_DISTRACTORS - 1, (count,),
                                     generator=rng)) % N_DISTRACTORS
        ids = torch.empty(count, window, dtype=torch.long)
        ids[:, 0], ids[:, 1], ids[:, 2] = BOS, PAD, PAD
        for i, position in enumerate(span_positions):
            ids[:, position] = COMB_BASE + members[:, i] * N_VALUES + values[:, i]
        ids[:, distractor_positions[0]] = DIST_BASE + d1
        ids[:, distractor_positions[1]] = DIST_BASE + d2
        m_j = members.gather(1, j.unsqueeze(1)).squeeze(1)
        v_j = values.gather(1, j.unsqueeze(1)).squeeze(1)
        ids[:, probe_position] = MEM_BASE + m_j
        ids[:, reveal_position] = VAL_BASE + v_j
        targets = (VAL_BASE + v_j).unsqueeze(1)
        return ids, targets, dict(j=j, members=members, values=values)

    return TaskSpec(f"span{span}", VOCAB, window, span_positions,
                    (reveal_position,), (probe_position,), make_batch)


TASKS = {span: make_task(span) for span in SPANS}


# ------------------------------------------------------- the ordered read
def ordered_read_weights(query: torch.Tensor, keys: torch.Tensor,
                         state: torch.Tensor, live: torch.Tensor, *,
                         read_scale: float, width: float):
    """The ONE interval read of the ordered store (``read.py``'s role).

    ``query`` (b, h), ``keys`` (b, h, r), ``state`` (b, h, r, d_v), ``live``
    (b, h, r).  Returns ``(w, omitted)``: ``w`` (b, h, r) is the
    full-population softmax weight (denominator over ALL live rows) times the
    row strength, ZEROED outside the interval -- the survivor coefficients are
    interval-widening-invariant (Lean ``full_coefficients_truncation_invariant``),
    the omitted mass drops output mass monotonically and is receipted as
    ``omitted`` (b, h).  An all-dead read is exactly zero (empty is a value).
    """
    strength = state.norm(dim=-1) * live.to(state.dtype)          # (b, h, r)
    dist = (query.unsqueeze(-1) - keys).abs()                    # (b, h, r)
    scores = (-read_scale * dist).masked_fill(~live, float("-inf"))
    any_live = live.any(-1, keepdim=True)
    safe = torch.where(any_live, scores, torch.zeros_like(scores))
    probs = torch.where(any_live, torch.softmax(safe, -1),
                        torch.zeros_like(safe))
    w_full = probs * strength
    in_interval = dist <= width
    w = w_full * in_interval
    omitted = (w_full * (~in_interval)).sum(-1)
    return w, omitted


def _rows_at(tensor: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
    """``tensor`` (b, h, r, X), ``slot`` (b, h) -> the (b, h, X) rows at slot."""
    index = slot.unsqueeze(-1).unsqueeze(-1).expand(
        *slot.shape, 1, tensor.shape[-1])
    return tensor.gather(2, index).squeeze(2)


def _scatter_rows(tensor: torch.Tensor, slot: torch.Tensor, src: torch.Tensor):
    """Write ``src`` (b, h, X) into ``tensor`` (b, h, r, X) at ``slot`` (b, h)."""
    index = slot.unsqueeze(-1).unsqueeze(-1).expand(
        *slot.shape, 1, tensor.shape[-1])
    return tensor.scatter(2, index, src.unsqueeze(2))


def _scatter_scalars(tensor: torch.Tensor, slot: torch.Tensor, src: torch.Tensor):
    """Scalar-channel variant of :func:`_scatter_rows` for (b, h, r) tensors."""
    return tensor.scatter(2, slot.unsqueeze(-1), src.unsqueeze(-1))


def ordered_oracle_scan(payload, write_keys, query_keys, decay, *, rank: int,
                        policy=None, collect: bool = True):
    """The ordered arm's eager per-token oracle, differentiable.

    Walks the window token by token, vectorised over (batch, head).  Per
    token: the interval READ (attached, full-population denominator), the
    DECISIONS (detached, on the scalar key channel), then DECAY (values only
    -- keys are addresses and do not age) and the selected op:

    * ``|k_t - k_j| <= tau_match_dist`` for the nearest live row ->
      UPDATE-IN-RUN: ``S_j += p_t`` (mass-preserving accumulate, the
      registered ``erase=False`` lane applied in-run), run length +1; the row
      KEEPS its key (a fixed quantisation centre -- a drifting centre parks
      the run's mass at its last address where no interior query can open it).
    * else a free slot -> INSERT at ``k_t`` (the key takes its derived sort
      position; nothing relabels).
    * else the tightest adjacent live pair with key gap ``<= tau_run`` ->
      MERGE (mass-preserving sum into the lower slot index, run lengths add),
      then the write INSERTS into the freed slot.  MERGE has priority over
      POP: pressure is resolved losslessly first.
    * else if the farthest live row is ``>= tau_pop_dist`` away -> POP
      (close-interval) and INSERT the write into the freed slot.
    * else NONE: the write is dropped, nothing is destroyed.
    """
    policy = ordered_policy() if policy is None else policy
    batch, tokens, heads, value_dim = payload.shape
    device = payload.device
    state = torch.zeros(batch, heads, rank, value_dim, dtype=torch.float32,
                        device=device)
    keys = torch.zeros(batch, heads, rank, dtype=torch.float32, device=device)
    live = torch.zeros(batch, heads, rank, dtype=torch.bool, device=device)
    runlen = torch.zeros(batch, heads, rank, dtype=torch.int64, device=device)
    reads = torch.empty(batch, tokens, heads, value_dim, dtype=torch.float32,
                        device=device)
    channels = dict(
        slot=torch.zeros(batch, tokens, heads, dtype=torch.int64, device=device),
        op=torch.zeros(batch, tokens, heads, dtype=torch.int64, device=device),
        merge_src=torch.zeros(batch, tokens, heads, dtype=torch.int64,
                              device=device),
        weights=torch.zeros(batch, tokens, heads, rank, dtype=torch.float32,
                            device=device),
        live_pre=torch.zeros(batch, tokens, heads, rank, dtype=torch.bool,
                             device=device),
        runlen_pre=torch.zeros(batch, tokens, heads, rank, dtype=torch.int64,
                               device=device),
        emitted_keys=torch.zeros(batch, tokens, heads, dtype=torch.float32,
                                 device=device),
        emitted_payload=torch.zeros(batch, tokens, heads, value_dim,
                                    dtype=torch.float32, device=device),
        omitted=torch.zeros(batch, tokens, heads, dtype=torch.float32,
                            device=device),
        retained=torch.zeros(batch, tokens, heads, dtype=torch.float32,
                             device=device))
    tie = torch.arange(rank, dtype=torch.float32, device=device)

    for t in range(tokens):
        q = query_keys[:, t]                                        # (b, h)
        k_t = write_keys[:, t]
        p_t = payload[:, t]                                         # (b, h, d_v)
        lam = decay[:, t]                                           # (b, h)
        # ---- the read (attached): full-population softmax, interval numerator.
        w, omitted = ordered_read_weights(q, keys, state, live,
                                          read_scale=policy["read_scale"],
                                          width=policy["width"])
        reads[:, t] = torch.einsum("bhr,bhrv->bhv", w, state)
        if collect:
            channels["live_pre"][:, t] = live
            channels["runlen_pre"][:, t] = runlen
            channels["weights"][:, t] = w
            channels["omitted"][:, t] = omitted
            channels["retained"][:, t] = w.sum(-1)
            channels["emitted_keys"][:, t] = k_t.detach()
            channels["emitted_payload"][:, t] = p_t.detach()
        # ---- the decisions: the registered chain's index-term form, detached.
        kd = k_t.detach()
        dist = (kd.unsqueeze(-1) - keys).abs().masked_fill(~live, float("inf"))
        best_idx = (dist + tie * 1e-6).argmin(-1)          # lowest index wins ties
        best_d = dist.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
        any_live = live.any(-1)
        match = any_live & (best_d <= policy["tau_match_dist"])
        # free slots: the lowest-index free slot (deterministic ordinal)
        free = ~live
        has_free = free.any(-1)
        slots = torch.arange(rank, device=device)
        first_free = torch.where(free, slots, torch.full_like(slots, rank)).amin(-1)
        # the tightest adjacent live pair in key order (MERGE candidate)
        sort_keys = keys.masked_fill(~live, 2.0)
        _, order = torch.sort(sort_keys, dim=-1)           # (b, h, r) slot ids
        lo_live = live.gather(-1, order[..., :-1])
        hi_live = live.gather(-1, order[..., 1:])
        gaps = (sort_keys.gather(-1, order[..., 1:])
                - sort_keys.gather(-1, order[..., :-1]))
        gaps = torch.where(lo_live & hi_live, gaps,
                           torch.full_like(gaps, float("inf")))
        has_merge = (gaps <= policy["tau_run"]).any(-1)
        m_star = (gaps + torch.arange(rank - 1, dtype=torch.float32,
                                      device=device) * 1e-6).argmin(-1)
        merge_a = order.gather(-1, m_star.unsqueeze(-1)).squeeze(-1)
        merge_b = order.gather(-1, (m_star + 1).unsqueeze(-1)).squeeze(-1)
        merge_lo = torch.minimum(merge_a, merge_b)         # target: lower slot id
        merge_hi = torch.maximum(merge_a, merge_b)         # source: the other
        # POP: the farthest live row, only if it is genuinely far
        far_idx = (dist - tie * 1e-6).argmax(-1)
        far_d = dist.gather(-1, far_idx.unsqueeze(-1)).squeeze(-1)
        pop_ok = any_live & (far_d >= policy["tau_pop_dist"])

        op = torch.where(match, torch.full_like(best_idx, O_UPDATE),
                        torch.where(has_free, torch.full_like(best_idx, O_INSERT),
                                    torch.where(has_merge,
                                                torch.full_like(best_idx, O_MERGE),
                                                torch.where(pop_ok,
                                                            torch.full_like(best_idx, O_POP),
                                                            torch.full_like(best_idx, O_NONE)))))
        slot = torch.where(match, best_idx,
                           torch.where(has_free, first_free,
                                       torch.where(has_merge, merge_hi, far_idx)))
        if collect:
            channels["slot"][:, t] = slot
            channels["op"][:, t] = op
            channels["merge_src"][:, t] = torch.where(op == O_MERGE, merge_lo, slot)
        # ---- apply: DECAY (values only), then the op lanes.
        state = state * lam[:, :, None, None]
        ones = torch.ones(batch, heads, dtype=torch.bool, device=device)
        # MERGE first: the mass-preserving coalesce of the adjacent pair
        # (target = the lower slot index -- deterministic; the freed source
        # slot takes the incoming write below).
        do_merge = op == O_MERGE
        if bool(do_merge.any()):
            lo_row = _rows_at(state, merge_lo)
            merged = lo_row + _rows_at(state, merge_hi)
            state = _scatter_rows(state, merge_lo,
                                 torch.where(do_merge.unsqueeze(-1), merged,
                                             lo_row))
            lo_run = runlen.gather(-1, merge_lo.unsqueeze(-1)).squeeze(-1)
            hi_run = runlen.gather(-1, merge_hi.unsqueeze(-1)).squeeze(-1)
            runlen = _scatter_scalars(runlen, merge_lo,
                                      torch.where(do_merge, lo_run + hi_run,
                                                  lo_run))
            live = _scatter_scalars(
                live, merge_lo,
                torch.where(do_merge, ones,
                            live.gather(-1, merge_lo.unsqueeze(-1)).squeeze(-1)))
        # the write itself: accumulate on update-in-run, else file into the
        # chosen slot (insert / the merge-freed slot / the pop victim's slot)
        old_row = _rows_at(state, slot)
        written = torch.where((op == O_UPDATE).unsqueeze(-1), old_row + p_t, p_t)
        do_write = op != O_NONE
        state = _scatter_rows(state, slot,
                              torch.where(do_write.unsqueeze(-1), written,
                                          old_row))
        # the stored key stays attached (the SEP-KV base's doctrine: gradients
        # reach the key channel through the reads); decisions used kd only.
        # On update-in-run the row KEEPS its key -- a fixed quantisation centre
        # (a drifting centre parks the run's mass at its last address, where
        # no interior query can open it; DESIGN_WHEELER_ORDER.md §4).
        old_key = keys.gather(-1, slot.unsqueeze(-1)).squeeze(-1)
        keys = _scatter_scalars(keys, slot,
                                torch.where(do_write,
                                            torch.where(op == O_UPDATE, old_key,
                                                        k_t),
                                            old_key))
        runlen = _scatter_scalars(
            runlen, slot,
            torch.where(do_write,
                        torch.where(op == O_UPDATE,
                                    runlen.gather(-1, slot.unsqueeze(-1)).squeeze(-1)
                                    + 1,
                                    torch.ones_like(best_idx)),
                        runlen.gather(-1, slot.unsqueeze(-1)).squeeze(-1)))
        live = _scatter_scalars(
            live, slot,
            torch.where(do_write, ones,
                        live.gather(-1, slot.unsqueeze(-1)).squeeze(-1)))
    return dict(reads=reads, state=state, live=live, keys=keys, runlen=runlen,
                **channels)


# ------------------------------------------------------- the ordered arm model
class WheelerMixer(nn.Module):
    """The ordered arm's mixer: payloads/DECAY/output unchanged from the
    SEP-KV base; the key channel is a learned SCALAR (tanh-squashed) key per
    head, for both the write and the query, and the scan is the ordered
    oracle."""

    def __init__(self, width: int, value_dim: int, heads: int, rank: int,
                 ctx: int = CONTEXT, mimetic: bool = False):
        super().__init__()
        self.heads, self.value_dim, self.rank = heads, value_dim, rank
        self.ctx, self.mimetic = ctx, mimetic
        self.value = nn.Linear(width, heads * value_dim, bias=False)
        self.key = nn.Linear(ctx * width, heads, bias=False)
        self.query_key = nn.Linear(ctx * width, heads, bias=False)
        self.out = nn.Linear(heads * value_dim, width, bias=False)
        self.decay = nn.Linear(width, heads, bias=True)
        self.ctx_norm = nn.RMSNorm(width, eps=1e-5)
        with torch.no_grad():
            lam0 = math.exp(-1.0 / DECAY_HORIZON_TOKENS)
            self.decay.bias.fill_(math.log(lam0 / (1.0 - lam0)))
            if mimetic:
                generator = torch.Generator().manual_seed(MIMETIC_SEED)
                u = torch.randn(width, generator=generator)
                u = u / u.norm().clamp_min(EPS)
                born = torch.zeros(heads, ctx * width)
                born[:, (ctx - 1) * width:] = MIMETIC_GAIN * u
                self.key.weight.copy_(born)
                self.query_key.weight.copy_(born.clone())

    def forward(self, x, collect: bool = False):
        batch, tokens, _ = x.shape
        ctx = causal_context(x, self.ctx)
        ctx_flat = self.ctx_norm(ctx.reshape(batch * tokens * self.ctx, -1)) \
            .reshape(batch, tokens, self.ctx * x.shape[-1])
        payload = self.value(x).view(batch, tokens, self.heads, self.value_dim)
        write_keys = torch.tanh(self.key(ctx_flat)).view(batch, tokens, self.heads)
        query_keys = torch.tanh(self.query_key(ctx_flat)).view(batch, tokens,
                                                               self.heads)
        decay = torch.sigmoid(self.decay(x)).clamp(min=DECAY_FLOOR,
                                                   max=1.0 - 1e-6)
        result = ordered_oracle_scan(payload, write_keys, query_keys, decay,
                                     rank=self.rank, collect=collect)
        reads = result["reads"].reshape(batch, tokens, self.heads * self.value_dim)
        return self.out(reads), result


class WheelerBlock(nn.Module):
    def __init__(self, width: int, hidden: int, value_dim: int, heads: int,
                 rank: int, ctx: int = CONTEXT, mimetic: bool = False):
        super().__init__()
        self.norm1 = nn.RMSNorm(width, eps=1e-5)
        self.mixer = WheelerMixer(width, value_dim, heads, rank, ctx, mimetic)
        self.norm2 = nn.RMSNorm(width, eps=1e-5)
        self.w1 = nn.Linear(width, hidden, bias=False)
        self.w2 = nn.Linear(width, hidden, bias=False)
        self.w3 = nn.Linear(hidden, width, bias=False)

    def forward(self, x, collect: bool = False):
        mixed, result = self.mixer(self.norm1(x), collect=collect)
        h = x + mixed
        gate = F.silu(self.w1(self.norm2(h))) * self.w2(self.norm2(h))
        return h + self.w3(gate), result


class WheelerModel(nn.Module):
    """The ordered arm: the registered block shape, scalar keys, interval read,
    run-coalescing ops (DESIGN_WHEELER_ORDER.md §3-§5)."""

    def __init__(self, task: TaskSpec, width: int = 64, hidden: int = 128,
                 depth: int = 2, heads: int = 1, value_dim: int = 32,
                 rank: int = RANK, mimetic: bool = False):
        super().__init__()
        self.task, self.rank = task, rank
        self.embed = nn.Embedding(task.vocab, width)
        self.blocks = nn.ModuleList(
            WheelerBlock(width, hidden, value_dim, heads, rank, mimetic=mimetic)
            for _ in range(depth))
        self.norm = nn.RMSNorm(width, eps=1e-5)
        self.head = nn.Linear(width, task.vocab, bias=False)

    def forward(self, ids, collect: bool = False):
        x = self.embed(ids)
        results = [] if collect else None
        for block in self.blocks:
            x, result = block(x, collect=collect)
            if collect:
                results.append(result)
        return self.head(self.norm(x)), results

    def loss(self, ids, targets):
        logits, _ = self.forward(ids)
        probe = logits[:, self.task.probe_positions]
        return F.cross_entropy(probe.reshape(-1, probe.shape[-1]),
                               targets.reshape(-1))


def make_model(task: TaskSpec, arm: str) -> nn.Module:
    """arm == "pointwise": the SEP-KV separated form VERBATIM at rank 8 (free
    16-D keys, registered policy, run length 1).  arm == "ordered": the
    Wheeler arm above."""
    if arm == "pointwise":
        return SeparatedModel(task, key_dim=16, rank=RANK)
    if arm == "ordered":
        return WheelerModel(task, rank=RANK)
    if arm == "ordered_mimetic":
        return WheelerModel(task, rank=RANK, mimetic=True)
    raise ValueError(f"unknown arm {arm!r}")


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def probe_predictions(model, ids, task: TaskSpec, batch: int = 1024):
    """Value-restricted argmax at the probe position: (N,) value-token ids.
    The task is "which value"; chance = 1/32."""
    model.eval()
    picks = []
    for start in range(0, ids.shape[0], batch):
        logits, _ = model.forward(ids[start:start + batch])
        probe = logits[:, task.probe_positions[0], VALUE_SLICE]     # (b, 32)
        picks.append(probe.argmax(-1) + VAL_BASE)
    return torch.cat(picks)


@torch.no_grad()
def evaluate(model, ids, targets, task: TaskSpec) -> dict:
    picks = probe_predictions(model, ids, task)
    correct = (picks == targets[:, 0])
    return dict(value_acc=float(correct.float().mean()),
                n_probes=int(correct.numel()), chance=CHANCE)


# ------------------------------------------------------------------- receipts
PT_APPEND, PT_CORRECT, PT_MERGE, PT_POP, PT_NONE = 1, 2, 4, 3, 0  # algebra codes


def _pointwise_binding_fate(result, aux, task: TaskSpec, count: int):
    """The pointwise saturation receipt: for every (window, probed binding) --
    was the binding's write FILED (append/correct/merge: it entered a row),
    and was that row destroyed (POP, or an erasing CORRECT by a later write)
    strictly between the write and the probe.  Blind spot, stated: a MERGE
    whose SOURCE row is the binding's row is not visible in the single-lane
    channel (tau_merge=0.85 makes MERGE ~inert in this policy; receipted)."""
    span = len(task.copy_positions)
    probe = task.probe_positions[0]
    slot, op = result["slot"][:, :, 0], result["op"][:, :, 0]   # (N, T)
    j = aux["j"]
    write_pos = 3 + j                                            # (N,)
    binding_slot = slot.gather(1, write_pos.unsqueeze(1)).squeeze(1)
    binding_op = op.gather(1, write_pos.unsqueeze(1)).squeeze(1)
    filed = (binding_op == PT_APPEND) | (binding_op == PT_CORRECT) \
        | (binding_op == PT_MERGE)
    destroyed = torch.zeros(count, dtype=torch.bool, device=slot.device)
    distance = (probe - write_pos).clamp(min=1)
    for offset in range(1, int(distance.max().item()) + 1):
        raw = write_pos + offset
        active = raw < probe
        if not bool(active.any()):
            break
        # CLAMP before the gather: the loop bound is the batch-wide max
        # distance, so per-window indices can run past the probe/window; the
        # clamped reads for inactive windows are discarded by the mask (an
        # unclamped gather past the window asserted on CUDA at S=4).
        at = raw.clamp(max=probe - 1).unsqueeze(1)
        hit = active & (op.gather(1, at).squeeze(1) == PT_POP) \
            & (slot.gather(1, at).squeeze(1) == binding_slot)
        hit = hit | (active
                     & (op.gather(1, at).squeeze(1) == PT_CORRECT)
                     & (slot.gather(1, at).squeeze(1) == binding_slot))
        destroyed |= hit
    return dict(binding_slot=binding_slot, binding_op=binding_op, filed=filed,
                destroyed=destroyed)


@torch.no_grad()
def collect_pointwise_channels(model, ids, targets, aux, task: TaskSpec) -> dict:
    """The pointwise arm's mechanistic receipts, from the imported SEP-KV
    oracle's own recorded channels."""
    model.eval()
    logits, results = model.forward(ids, collect=True)
    picks = probe_predictions(model, ids, task)
    correct = picks == targets[:, 0]
    count = ids.shape[0]
    probe = task.probe_positions[0]
    per_layer = []
    for layer, result in enumerate(results):
        fate = _pointwise_binding_fate(result, aux, task, count)
        weights = result["weights"][:, probe, 0]                # (N, r)
        live_pre = result["live_pre"][:, probe, 0]               # (N, r)
        slot, op = result["slot"][:, :, 0], result["op"][:, :, 0]
        mass_on_row = weights.gather(1, fate["binding_slot"].unsqueeze(1)) \
            .squeeze(1)
        total_mass = weights.sum(-1).clamp_min(EPS)
        intact = fate["filed"] & ~fate["destroyed"] \
            & live_pre.gather(1, fate["binding_slot"].unsqueeze(1)).squeeze(1)
        span_slots = slot[:, list(task.copy_positions)]          # (N, S)
        span_live = live_pre.gather(1, span_slots)               # (N, S)
        per_layer.append(dict(
            layer=layer,
            op_histogram={name: int((op == code).sum())
                          for code, name in ((PT_APPEND, "append"),
                                             (PT_CORRECT, "correct"),
                                             (PT_MERGE, "merge"),
                                             (PT_POP, "pop"), (PT_NONE, "none"))},
            occupancy_mean=float(live_pre.sum(-1).float().mean()),
            span_pos_live_frac=float(span_live.float().mean()),
            probed_binding=dict(
                filed_frac=float(fate["filed"].float().mean()),
                destroyed_frac=float(fate["destroyed"].float().mean()),
                intact_frac=float(intact.float().mean()),
                acc_when_filed=float(correct[fate["filed"]].float().mean())
                if bool(fate["filed"].any()) else None,
                acc_when_intact=float(correct[intact].float().mean())
                if bool(intact.any()) else None,
                acc_when_destroyed=float(correct[fate["destroyed"]].float()
                                         .mean())
                if bool(fate["destroyed"].any()) else None,
                acc_when_not_filed=float(correct[~fate["filed"]].float()
                                         .mean())
                if bool((~fate["filed"]).any()) else None,
                probe_mass_on_row=dict(
                    mean=float(mass_on_row.mean()),
                    mean_share=float((mass_on_row / total_mass).mean())))))
    return dict(arm="pointwise", value_acc=float(correct.float().mean()),
                per_layer=per_layer)


@torch.no_grad()
def collect_ordered_channels(model, ids, targets, aux, task: TaskSpec) -> dict:
    """The ordered arm's run-structure receipts: the run-length histogram at
    the probe (does the store actually build runs?), the Wheeler-ness
    contiguity of the span's writes in the learned key order, the true-value
    mass share of the interval read (the dilution receipt), and the interval's
    omitted mass."""
    model.eval()
    logits, results = model.forward(ids, collect=True)
    picks = probe_predictions(model, ids, task)
    correct = picks == targets[:, 0]
    count = ids.shape[0]
    probe = task.probe_positions[0]
    reveal = task.reveal_positions[0]
    span_positions = list(task.copy_positions)
    j = aux["j"]
    write_pos = 3 + j
    per_layer = []
    for layer, result in enumerate(results):
        op, slot = result["op"][:, :, 0], result["slot"][:, :, 0]    # (N, T)
        runlen_pre = result["runlen_pre"][:, probe, 0]           # (N, r)
        live_pre = result["live_pre"][:, probe, 0]                # (N, r)
        weights = result["weights"][:, probe, 0]                  # (N, r)
        emitted = result["emitted_keys"][:, :, 0]                  # (N, T)
        payload = result["emitted_payload"][:, :, 0]              # (N, T, d_v)
        reads = result["reads"][:, :, 0]                          # (N, T, d_v)
        # run-length histogram over live rows (the coalescing receipt)
        live_runlens = runlen_pre[live_pre]
        histogram = torch.bincount(live_runlens.to("cpu"),
                                   minlength=int(live_runlens.max().item()) + 1) \
            if live_runlens.numel() else torch.zeros(1, dtype=torch.long)
        # the probed binding's row and the read's mass on it
        binding_slot = slot.gather(1, write_pos.unsqueeze(1)).squeeze(1)
        binding_op = op.gather(1, write_pos.unsqueeze(1)).squeeze(1)
        mass_on_row = weights.gather(1, binding_slot.unsqueeze(1)).squeeze(1)
        retained = result["retained"][:, probe, 0].clamp_min(EPS)
        row_live = live_pre.gather(1, binding_slot.unsqueeze(1)).squeeze(1)
        # the dilution receipt: the probe read's magnitude along the TRUE
        # value's payload direction (the bare reveal token's emitted payload
        # is the pure value direction at this layer).
        read = reads[:, probe]                                    # (N, d_v)
        value_dir = payload[:, reveal]                           # (N, d_v)
        unit_read = read / read.norm(dim=-1, keepdim=True).clamp_min(EPS)
        unit_val = value_dir / value_dir.norm(dim=-1, keepdim=True).clamp_min(EPS)
        share = (unit_read * unit_val).sum(-1).clamp_min(0)
        # the Wheeler-ness diagnostic: contiguity of the span's writes in the
        # learned key order among this window's writes.
        order = emitted.argsort(dim=1)                            # (N, T)
        ranks = order.argsort(dim=1)                              # (N, T)
        span_ranks = ranks[:, span_positions]                    # (N, S)
        lo, hi = span_ranks.min(-1).values, span_ranks.max(-1).values
        span = len(span_positions)
        contiguity = span / (hi - lo + 1).clamp_min(span)
        interleaved = (lo.unsqueeze(1) <= ranks) & (ranks <= hi.unsqueeze(1))
        interleaved[:, span_positions] = False
        span_slot_rows = slot[:, span_positions]
        per_layer.append(dict(
            layer=layer,
            op_histogram={name: int((op == code).sum())
                          for code, name in enumerate(O_NAMES)},
            occupancy_mean=float(live_pre.sum(-1).float().mean()),
            run_length_histogram=[int(x) for x in histogram],
            run_length_mean=float(live_runlens.float().mean())
            if live_runlens.numel() else None,
            run_length_max=int(live_runlens.max().item())
            if live_runlens.numel() else 0,
            probed_binding=dict(
                op_counts={name: int((binding_op == code).sum())
                           for code, name in enumerate(O_NAMES)},
                row_live_frac=float(row_live.float().mean()),
                probe_mass_on_row=dict(
                    mean=float(mass_on_row.mean()),
                    mean_share=float((mass_on_row / retained).mean())),
                true_value_share=dict(mean=float(share.mean()),
                                      median=float(share.median()))),
            wheeler_contiguity=dict(
                mean=float(contiguity.mean()),
                frac_at_least_half=float((contiguity >= 0.5).float().mean()),
                interleaved_foreign_writes_mean=float(
                    interleaved.sum(-1).float().mean())),
            key_scale=dict(
                span_key_std=float(emitted[:, span_positions].std()),
                span_key_range_mean=float(
                    (emitted[:, span_positions].max(-1).values
                     - emitted[:, span_positions].min(-1).values).mean())),
            interval_omitted_mass_mean=float(
                result["omitted"][:, probe, 0].mean()),
            accuracy_per_probe=float(correct.float().mean())))
        top = share >= share.quantile(0.75)
        per_layer[-1]["probed_binding"]["acc_when_share_top_quartile"] = \
            float(correct[top].float().mean()) if bool(top.any()) else None
    return dict(arm="ordered", value_acc=float(correct.float().mean()),
                per_layer=per_layer)


def collect_channels(model, ids, targets, aux, task: TaskSpec) -> dict:
    if isinstance(model, SeparatedModel):
        return collect_pointwise_channels(model, ids, targets, aux, task)
    return collect_ordered_channels(model, ids, targets, aux, task)


def is_pointwise_arm(arm: str) -> bool:
    return arm == "pointwise"


# ------------------------------------------------------------------- one run
def cell_receipt_name(arm: str, span: int, lr: float, seed: int) -> str:
    return f"wheeler_{arm}_s{span}_lr{lr:.0e}_seed{seed}.json"


def run_one(arm: str, span: int, lr: float, seed: int, steps: int, batch: int,
            device: str, eval_sets, quick: bool = False, resume: bool = False):
    """Train one (arm, span, lr, seed) cell; every cell of the sweep shares the
    fixed eval set (seed 4999) so arms are compared on identical probes.
    With ``resume`` an existing per-cell receipt is reused instead of
    retraining (cells are deterministic; the receipt is the record of the
    committed run) -- an engineering recovery path, never a protocol change:
    the reused receipt is byte-identical to what its own run wrote."""
    task = TASKS[span]
    if resume and not quick:
        path = RECEIPTS / cell_receipt_name(arm, span, lr, seed)
        if path.exists():
            receipt = json.loads(path.read_text())
            receipt["resumed"] = True
            state = torch.load(RECEIPTS /
                               f"wheeler_{arm}_s{span}_best_seed{seed}.pt",
                               map_location=device, weights_only=True) \
                if (RECEIPTS / f"wheeler_{arm}_s{span}_best_seed{seed}.pt").exists() \
                else None
            print(f"{arm} S={span} lr={lr:.0e} seed={seed}: "
                  f"value_acc={receipt['metrics']['value_acc']:.4f} [resumed]",
                  flush=True)
            return receipt, state
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = make_model(task, arm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                                  weight_decay=0.01)
    rng = torch.Generator().manual_seed(seed)
    eval_ids, eval_targets, eval_aux = eval_sets
    eval_aux = {k: v.to(device) for k, v in eval_aux.items()}
    curve = []
    started = time.monotonic()
    model.train()
    for step in range(1, steps + 1):
        ids, targets, _ = task.make_batch(rng, batch)
        loss = model.loss(ids.to(device), targets.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 500 == 0 or step == steps:
            picks = probe_predictions(model, eval_ids[:1024], task)
            curve.append(dict(step=step, loss=float(loss.detach()),
                              value_acc=float(
                                  (picks == eval_targets[:1024, 0]).float().mean())))
    walls = time.monotonic() - started
    metrics = evaluate(model, eval_ids, eval_targets, task)
    receipt_ids, receipt_targets = eval_ids[:512], eval_targets[:512]
    receipt_aux = {k: v[:512] for k, v in eval_aux.items()}
    channels = collect_channels(model, receipt_ids, receipt_targets, receipt_aux,
                                task)
    receipt = dict(
        arm=arm, span=span, lr=lr, seed=seed, rank=RANK, steps=steps,
        batch=batch, device=device, window=task.window,
        probe_positions=list(task.probe_positions), chance=CHANCE,
        policy=ordered_policy_record(), metrics=metrics,
        curve_tail=curve[-4:], walls_s=round(walls, 1), channels=channels,
        params=sum(p.numel() for p in model.parameters()),
        torch=torch.__version__)
    state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    return receipt, state


def parse_floats(text):
    return [float(x) for x in text.split(",")]


def parse_ints(text):
    return [int(x) for x in text.split(",")]


def sweep(arm: str, spans, lrs, seeds, steps: int, batch: int, device: str,
          quick: bool = False, resume: bool = False):
    """The (span, lr, seed) grid for one arm; per-cell receipts, the best model
    per (arm, span) kept for the mechanism pass."""
    runs, best, best_state = [], {}, {}
    for span in spans:
        task = TASKS[span]
        eval_rng = torch.Generator().manual_seed(4999)
        eval_ids, eval_targets, eval_aux = task.make_batch(eval_rng, 512 if quick
                                                           else 4096)
        eval_aux = {k: v.to(device) for k, v in eval_aux.items()}
        eval_sets = (eval_ids.to(device), eval_targets.to(device), eval_aux)
        for lr in lrs:
            for seed in seeds:
                receipt, state = run_one(arm, span, lr, seed, steps, batch,
                                        device, eval_sets, quick, resume)
                if not quick and not receipt.get("resumed"):
                    name = cell_receipt_name(arm, span, lr, seed)
                    (RECEIPTS / name).write_text(json.dumps(receipt, indent=1))
                runs.append({k: receipt[k] for k in
                             ("arm", "span", "lr", "seed", "metrics")})
                key = (arm, span)
                if (key not in best or \
                        receipt["metrics"]["value_acc"] \
                        > best[key]["metrics"]["value_acc"]) \
                        and state is not None:
                    best[key] = receipt
                    best_state[key] = state
                    if not quick:
                        # NOT committed (program rule); the merge step and the
                        # mechanism pass reload them for the run-structure
                        # receipts.
                        torch.save(state, RECEIPTS /
                                   f"wheeler_{arm}_s{span}_best_seed{seed}.pt")
                print(f"{arm} S={span} lr={lr:.0e} seed={seed}: "
                      f"value_acc={receipt['metrics']['value_acc']:.4f} "
                      f"[{receipt['walls_s']}s]", flush=True)
        if not quick:
            del eval_ids
    summary = dict(arm=arm, rank=RANK, steps=steps, batch=batch,
                   seeds=list(seeds), chance=CHANCE, runs=runs,
                   best={f"s{span}": dict(lr=v["lr"], seed=v["seed"],
                                          metrics=v["metrics"],
                                          channels_summary={
                                              k: v["channels"][k]
                                              for k in ("value_acc",)
                                          }) for (a, span), v in best.items()
                         if a == arm})
    return summary, best, best_state


def merge_seeds(arm: str, spans, seeds, steps: int, batch: int, device: str):
    """Merge the per-seed split sweep into the canonical arm receipt, then run
    the mechanism pass from the overall best models (the split-by-seed,
    split-by-card execution of the registered protocol -- the cells themselves
    are exactly the registered grid)."""
    runs, best = [], {}
    for seed in seeds:
        for tag in ("", "_phaseB"):
            path = RECEIPTS / f"wheeler_{arm}_sweep_seed{seed}{tag}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            runs.extend(data["runs"])
            for key_s, v in data["best"].items():
                span = int(key_s[1:])
                if span not in best or v["metrics"]["value_acc"] \
                        > best[span]["metrics"]["value_acc"]:
                    best[span] = v
    best_state = {}
    for span, v in best.items():
        best_state[(arm, span)] = torch.load(
            RECEIPTS / f"wheeler_{arm}_s{span}_best_seed{v['seed']}.pt",
            map_location=device, weights_only=True)
    summary = dict(arm=arm, rank=RANK, steps=steps, batch=batch,
                   seeds=list(seeds), chance=CHANCE, runs=runs,
                   best={f"s{span}": v for span, v in best.items()})
    (RECEIPTS / f"wheeler_{arm}_sweep.json").write_text(json.dumps(summary,
                                                                   indent=1))
    print(json.dumps(summary["best"], indent=1))
    return mechanism_pass(None, best_state, spans, device, arm)


def mechanism_pass(best, best_state, spans, device: str, arm: str,
                   windows: int = 512):
    """The headline receipts from the sweep's best models per (arm, span):
    the pointwise saturation witness and the ordered run-structure /
    Wheeler-ness receipts, on a fresh receipt batch (seed 6999).  When the
    sweep is split by arm across cards, each process receipts its own arm."""
    mechanism = dict(experiment="wheeler_capacity_mechanism", chance=CHANCE,
                    policy=ordered_policy_record(),
                    note=("pointwise: probed-binding filed/evicted strata from "
                          "the imported SEP-KV oracle's own channels; ordered: "
                          "run-length histogram, key-order contiguity (the "
                          "Wheeler-ness diagnostic), interval omitted mass and "
                          "true-value share of the interval read."))
    arms_present = sorted({a for a, _ in best_state})
    for span in spans:
        task = TASKS[span]
        rng = torch.Generator().manual_seed(6999)
        ids, targets, aux = task.make_batch(rng, windows)
        ids, targets = ids.to(device), targets.to(device)
        aux = {k: v.to(device) for k, v in aux.items()}
        for arm in arms_present:
            key = (arm, span)
            if key not in best_state:
                continue
            model = make_model(task, arm).to(device)
            model.load_state_dict(best_state[key])
            mechanism[f"{arm}_s{span}"] = collect_channels(
                model, ids, targets, aux, task)
            print(f"mechanism {arm} S={span}: "
                  f"acc={mechanism[f'{arm}_s{span}']['value_acc']:.4f}",
                  flush=True)
    (RECEIPTS / f"wheeler_mechanism_{arm}.json").write_text(
        json.dumps(mechanism, indent=1))
    return mechanism


def main():
    parser = argparse.ArgumentParser(
        description="WHEELER-ORDER capacity probe (SPAN-RUN): pointwise vs "
                    "ordered run-coalescing at rank 8")
    parser.add_argument("--arm",
                        choices=["pointwise", "ordered", "ordered_mimetic",
                                 "both"],
                        default="both")
    parser.add_argument("--spans", default=",".join(str(s) for s in SPANS))
    parser.add_argument("--lrs", default="3e-3,1e-3,3e-4")
    parser.add_argument("--seeds", default="3401,3402,3403")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--quick", action="store_true",
                        help="smoke: one arm, one span, one lr, one seed, "
                             "200 steps, no receipts")
    parser.add_argument("--run-tag", default="",
                        help="suffix for the sweep summary file (the "
                             "split-by-seed execution; per-cell receipt "
                             "names are already unique by seed)")
    parser.add_argument("--resume", action="store_true",
                        help="reuse existing per-cell receipts instead of "
                             "retraining deterministic cells (recovery path)")
    parser.add_argument("--merge-seeds", action="store_true",
                        help="merge the per-seed sweep summaries for --arm "
                             "into the canonical receipt and run the "
                             "mechanism pass from the overall best models")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # CPU thread cap (operator housekeeping, receipted with the driver change:
    # the eager oracle is GPU-dispatch-bound, so this costs single-digit %
    # walls; without it each worker spin-waits on ~30 cores).  Env-overridable.
    torch.set_num_threads(int(__import__("os").environ.get("WHEELER_CPU_THREADS",
                                                          "4")))

    spans = parse_ints(args.spans)
    if args.quick:
        args.arm, args.lrs, args.seeds = "ordered", "1e-3", "3401"
        args.steps, spans = 200, [8]

    if args.merge_seeds:
        seeds = parse_ints(args.seeds)
        print(json.dumps(merge_seeds(args.arm, spans, seeds, args.steps,
                                     args.batch, args.device), indent=1))
        return

    arms = ["pointwise", "ordered"] if args.arm == "both" else [args.arm]
    all_best, all_best_state = {}, {}
    for arm in arms:
        summary, best, best_state = sweep(arm, spans, parse_floats(args.lrs),
                                          parse_ints(args.seeds), args.steps,
                                          args.batch, args.device, args.quick,
                                          args.resume)
        all_best.update(best)
        all_best_state.update(best_state)
        if not args.quick:
            (RECEIPTS / f"wheeler_{arm}_sweep{args.run_tag}.json").write_text(
                json.dumps(summary, indent=1))
            print(json.dumps(summary["best"], indent=1))
            if not args.run_tag:
                print(json.dumps(mechanism_pass(all_best, all_best_state,
                                                 spans, args.device, arm),
                                 indent=1))


if __name__ == "__main__":
    main()
