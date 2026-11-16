"""GRANULARITY SEPARATOR -- gated.

The parity and mechanism discipline for the separator experiment
(``granularity_separator.py``, pre-stated predictions in
``RESULT_GRANULARITY_SEPARATOR.md``):

* **The binding receipt** (``test_per_token_arm_matches_k6_oracle``): the
  separator's per-token arm is ``routed_chunk_scan(chunk_tokens=1)`` through the
  registered eager scan; it must reproduce K6's ``sequential_oracle_scan``
  decision-for-decision (integer channels bit-equal; float channels within the
  fp32-vs-fp64 floor measured on the same fixture, the K6 NOISE_FACTOR=8
  discipline -- the two paths differ in reduction order, not semantics).  This
  binds the separator's per-token arm and the K6 lane into one story.
* **P2 is exact** (``test_chunked_read_blindness_is_exact``): perturbing the
  payload of a mid-chunk APPEND changes the chunked arm's reads NOT AT ALL --
  bit-identical -- while the state does change and the per-token arm's reads do
  change.  The in-chunk write-then-read channel does not exist at granularity C,
  for any payload.  This is the expressivity limitation the experiment is a
  witness for, proven on a fixture.
* **P1 is exact** (``test_freed_slot_not_appendable_same_chunk``): a slot freed
  mid-chunk by POP cannot be APPENDed into before the next chunk boundary; the
  per-token arm appends into it one token later.
* The task generator's layout, binding and determinism, and the
  pre-registered same-chunk prediction table, are asserted by name.
"""
from __future__ import annotations

import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    APPEND, MERGE, NONE, POP)
from experiments.expressivity_tasks.x45_matrix_emender.granularity_separator import (
    BOS, CHANCE, KEY_BASE, N_KEYS, N_PAIRS, N_PROBES, N_VALUES, PAIR_BASE,
    PAIR_POSITIONS, PAD, PROBE_POSITIONS, SeparatorModel, VAL_BASE, VOCAB,
    WINDOW, make_batch, probe_predictions, same_chunk, same_chunk_table)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy, route_chunk)
from experiments.expressivity_tasks.x45_matrix_emender.scan import routed_chunk_scan

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
POLICY = RoutingPolicy()


# ------------------------------------------------------------ task generator
def test_window_layout_and_binding():
    ids, targets, probe_pairs = make_batch(torch.Generator().manual_seed(7), 64)
    assert ids.shape == (64, WINDOW) and targets.shape == (64, N_PROBES)
    assert (ids[:, 0] == BOS).all()
    assert (ids[:, 1:4] == PAD).all()
    pairs = ids[:, PAIR_POSITIONS]
    assert (pairs >= PAIR_BASE).all() and (pairs < PAIR_BASE + N_KEYS * N_VALUES).all()
    # keys and values drawn without replacement: the answer is unambiguous
    keys = (pairs - PAIR_BASE) // N_VALUES
    values = (pairs - PAIR_BASE) % N_VALUES
    assert all(len(set(row.tolist())) == N_PAIRS for row in keys)
    assert all(len(set(row.tolist())) == N_PAIRS for row in values)
    for j, position in enumerate(PROBE_POSITIONS):
        assert (ids[:, position] >= KEY_BASE).all() \
            and (ids[:, position] < KEY_BASE + N_KEYS).all()
        # the target is the revealed next token, and the probed key is the key
        # of the pair the probe index names
        assert (ids[:, position + 1] == targets[:, j]).all()
        assert (targets[:, j] >= VAL_BASE).all() \
            and (targets[:, j] < VAL_BASE + N_VALUES).all()
        picked = probe_pairs[:, j]
        assert (ids[:, position] - KEY_BASE ==
                keys.gather(1, picked.unsqueeze(1)).squeeze(1)).all()
        assert (ids[:, position + 1] - VAL_BASE ==
                values.gather(1, picked.unsqueeze(1)).squeeze(1)).all()
    assert all(len(set(row.tolist())) == N_PROBES for row in probe_pairs)


def test_generator_determinism():
    a = make_batch(torch.Generator().manual_seed(11), 8)
    b = make_batch(torch.Generator().manual_seed(11), 8)
    c = make_batch(torch.Generator().manual_seed(12), 8)
    for x, y in zip(a, b):
        assert torch.equal(x, y)
    assert not torch.equal(a[0], c[0])


def test_same_chunk_predicate_is_the_prediction_table():
    # C=1: every decision is fresh -- nothing is blind.
    assert not same_chunk_table(1).any()
    # C=8: chunks [0..7] and [8..15]; pairs 0..3 (positions 4..7) are
    # entry-live at the probes' chunk, pairs 4,5 (positions 8,9) are written
    # inside it.  This split IS the pre-registered middle-of-the-curve test.
    table8 = same_chunk_table(8)
    assert not table8[:4].any()
    assert table8[4:].all()
    # C=32 and C=128: the whole window is one chunk with an empty entry store;
    # every pair is written after the (single) decision point.
    for chunk_tokens in (32, 128):
        assert same_chunk_table(chunk_tokens).all()
    # the predicate itself is position arithmetic, checked once by hand:
    # pair 5 sits at position 9, probe 1 at position 12; both in chunk 1 at C=8.
    assert same_chunk(5, 1, 8)
    assert not same_chunk(3, 1, 8)


# -------------------------------------------------------- the exact properties
def test_chunked_read_blindness_is_exact():
    """P2: a mid-chunk APPEND into a dead-at-entry row contributes EXACTLY
    nothing to any read of its chunk, whatever its payload magnitude -- while
    the per-token arm reads it one token later.  The chunked schedule's read is
    therefore invariant to information that exists in the store at read time:
    the write-then-read-inside-the-decision-interval channel does not exist."""
    torch.manual_seed(5)
    batch, tokens, heads, value_dim, rank = 1, 16, 1, 8, 16
    eye = torch.eye(value_dim)
    # chunk 0 (tokens 0..7) fills rows 0..7 with basis directions; token 10
    # (chunk 1) carries the uniform direction: cosine 1/sqrt(8) < tau_match to
    # every row, so it APPENDs into row 8 (dead at chunk-1 entry).
    payload = torch.zeros(batch, tokens, heads, value_dim)
    for t in range(8):
        payload[:, t, 0] = eye[t % value_dim]
    payload[:, 8, 0] = eye[0]        # entry-matching CORRECTs: leave the free
    payload[:, 9, 0] = eye[1]        # slots for the token under test
    payload[:, 10, 0] = torch.full((value_dim,), value_dim ** -0.5)
    payload[:, 11:] = eye[1]
    query = torch.randn(batch, tokens, heads, value_dim)
    decay = torch.full((batch, tokens, heads), 0.99)

    def run(chunk_tokens, pay):
        with torch.no_grad():
            return routed_chunk_scan(query, pay, decay, router=route_chunk,
                                      policy=POLICY, rank=rank, rank_max=rank,
                                      chunk_tokens=chunk_tokens, collect=True,
                                      fused=False)

    base = run(8, payload)
    scaled = payload.clone()
    scaled[:, 10] *= 7.0            # cosine-invariant: the same op, same slot
    perturbed = run(8, scaled)
    # identical decision channels (magnitude is not a decision input)
    for name in ("slot", "erase", "op", "live_after"):
        assert torch.equal(getattr(base.schedule, name),
                           getattr(perturbed.schedule, name)), name
    assert int(base.schedule.op[0, 10, 0, 0]) == APPEND
    # P2: the reads are bit-identical; the store is not.
    assert torch.equal(base.reads, perturbed.reads)
    assert not torch.equal(base.state, perturbed.state)
    assert int(base.live[0, 0, 8]) == 1, "the appended row IS live at the chunk exit"
    # the recorded read channel puts exactly zero on that live row
    assert float(base.weights[0, :, 0, 8].abs().max()) == 0.0
    # the per-token arm reads the write at the very next token
    per_token_base = run(1, payload)
    per_token_perturbed = run(1, scaled)
    assert torch.equal(per_token_base.reads[:, :11], per_token_perturbed.reads[:, :11])
    assert not torch.equal(per_token_base.reads[:, 11], per_token_perturbed.reads[:, 11])
    assert float(per_token_perturbed.weights[0, 11, 0, 8]) > 0.0


def test_freed_slot_not_appendable_same_chunk():
    """P1: a slot freed mid-chunk by POP is invisible to the chunk's APPEND
    enumeration (free slots are counted at the chunk entry); the per-token arm
    appends into it one token later."""
    value_dim, rank, tokens = 4, 2, 4
    eye = torch.eye(value_dim)
    initial_state = torch.zeros(1, 1, rank, value_dim)
    initial_state[0, 0, 0] = eye[0]
    initial_state[0, 0, 1] = eye[1]
    initial_live = torch.ones(1, 1, rank, dtype=torch.bool)
    payload = torch.zeros(1, tokens, 1, value_dim)
    payload[:, 0] = eye[2]          # orthogonal to both rows: POPs the worst row
    payload[:, 1] = eye[3]          # needs a slot
    payload[:, 2] = eye[0]          # harmless CORRECT of row 0 afterwards
    payload[:, 3] = eye[0]
    query = torch.randn(1, tokens, 1, value_dim)
    decay = torch.full((1, tokens, 1), 0.999)

    def run(chunk_tokens):
        return routed_chunk_scan(query, payload, decay, router=route_chunk,
                                 policy=POLICY, rank=rank, rank_max=rank,
                                 chunk_tokens=chunk_tokens, collect=True,
                                 fused=False, initial_state=initial_state,
                                 initial_live=initial_live)

    chunked = run(4)
    assert int(chunked.schedule.op[0, 0, 0, 0]) == POP, "the store is full: POP fires"
    assert int(chunked.schedule.op[0, 1, 0, 0]) != APPEND, \
        "a mid-chunk-freed slot is not in the chunk's APPEND enumeration (P1)"
    # the freed slot was never APPENDed into (delta-solve roundoff ~1e-7
    # stays on the POPped row; the op channel is the assertion that matters)
    assert float(chunked.state[0, 0, 1].abs().max()) < 1e-5
    per_token = run(1)
    assert int(per_token.schedule.op[0, 0, 0, 0]) == POP
    assert int(per_token.schedule.op[0, 1, 0, 0]) == APPEND, \
        "the fresh decision sees the freed slot one token later"
    assert int(per_token.schedule.slot[0, 1, 0, 0]) == 1
    assert per_token.state[0, 0, 1].norm() > 0.9 * eye[3].norm()


# --------------------------------------------------- the K6 binding receipt
def _distinct_stream(batch, tokens, heads, value_dim, distinct, seed, device,
                     decay=0.95):
    g = torch.Generator().manual_seed(seed)
    query = torch.randn(batch, tokens, heads, value_dim, generator=g).to(device) * 0.5
    directions = torch.randn(distinct, value_dim, generator=g).to(device)
    choice = torch.randint(0, distinct, (batch, tokens, heads), generator=g).to(device)
    payload = directions[choice]
    decay = torch.full((batch, tokens, heads), decay, device=device)
    return query, payload, decay


@needs_cuda
def test_per_token_arm_matches_k6_oracle():
    """The binding receipt: the separator's per-token arm
    (``routed_chunk_scan(chunk_tokens=1)`` through the registered eager scan)
    IS the K6 sequential arm's semantic -- the same decisions (integer channels
    bit-equal), the same reads/weights/state within the fp32-vs-fp64 floor
    measured on the same fixture (K6's NOISE_FACTOR=8 discipline; the two paths
    differ in reduction order, not semantics)."""
    from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix \
        import sequential_oracle_scan
    device = "cuda"
    batch, tokens, heads, value_dim, rank = 2, 16, 2, 8, 5
    query, payload, decay = _distinct_stream(batch, tokens, heads, value_dim, 6,
                                             23, device)
    eager = routed_chunk_scan(query, payload, decay, router=route_chunk,
                              policy=POLICY, rank=rank, rank_max=rank,
                              chunk_tokens=1, collect=True, fused=False)
    with torch.no_grad():
        ref = sequential_oracle_scan(query, payload, decay, policy=POLICY,
                                     rank=rank, rank_max=rank)
        ref64 = sequential_oracle_scan(query.double(), payload.double(),
                                       decay.double(), policy=POLICY,
                                       rank=rank, rank_max=rank)
    for name in ("slot", "erase", "op", "live_after"):
        assert torch.equal(getattr(eager.schedule, name),
                           getattr(ref["schedule"], name)), name
    assert torch.equal(eager.schedule.payload, ref["schedule"].payload)
    assert torch.equal(eager.live, ref["live"])
    for name in ("reads", "weights", "state"):
        floor = (ref[name].float() - ref64[name].float()).abs().max().item()
        deviation = (getattr(eager, name) - ref[name]).abs().max().item()
        assert deviation <= max(8.0 * max(floor, 1e-30), 1e-5), (name, deviation, floor)
    import experiments.expressivity_tasks.x45_matrix_emender.algebra as algebra
    hist = algebra.op_histogram(ref["schedule"])
    assert hist["append"] > 0 and hist["correct"] > 0, hist


@needs_cuda
def test_per_token_arm_matches_k6_oracle_merge_and_pop():
    """The binding receipt on crafted stores that force MERGE and POP: the
    registered eager path at chunk granularity 1 carries all five ops
    decision-for-decision like the K6 oracle."""
    from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix \
        import sequential_oracle_scan
    device = "cuda"
    value_dim, rank = 8, 4
    eye = torch.eye(value_dim, device=device)
    row1 = 0.9 * eye[0] + (1 - 0.81) ** 0.5 * eye[1]

    def run(payload, state_rows):
        state = torch.zeros(1, 1, rank, value_dim, device=device)
        live = torch.zeros(1, 1, rank, dtype=torch.bool, device=device)
        for r, row in enumerate(state_rows):
            state[0, 0, r] = row
            live[0, 0, r] = True
        eager = routed_chunk_scan(query, payload, decay, router=route_chunk,
                                  policy=POLICY, rank=rank, rank_max=rank,
                                  chunk_tokens=1, collect=True, fused=False,
                                  initial_state=state, initial_live=live)
        with torch.no_grad():
            ref = sequential_oracle_scan(query, payload, decay, policy=POLICY,
                                         rank=rank, rank_max=rank,
                                         initial_state=state, initial_live=live)
        return eager, ref

    g = torch.Generator().manual_seed(31)
    query = torch.randn(1, 16, 1, value_dim, generator=g).to(device)
    decay = torch.full((1, 16, 1), 0.999, device=device)
    # MERGE: rows 0 and 1 are 0.9-cosine; the payload is row 0's direction
    payload = eye[0].view(1, 1, 1, value_dim).repeat(1, 16, 1, 1)
    eager, ref = run(payload, [eye[0], row1, eye[2], eye[3]])
    assert int(ref["schedule"].op[0, 0, 0, 0]) == MERGE
    for name in ("slot", "erase", "op", "live_after"):
        assert torch.equal(getattr(eager.schedule, name),
                           getattr(ref["schedule"], name)), name
    # POP: a full live store, a payload orthogonal to every row
    payload = eye[4].view(1, 1, 1, value_dim).repeat(1, 16, 1, 1)
    eager, ref = run(payload, [eye[i] for i in range(4)])
    assert int(ref["schedule"].op[0, 0, 0, 0]) == POP
    for name in ("slot", "erase", "op", "live_after"):
        assert torch.equal(getattr(eager.schedule, name),
                           getattr(ref["schedule"], name)), name


# ----------------------------------------------------------------- the model
def test_model_smoke_forward_backward():
    """Both arms run, the loss is finite, and the gradient reaches the write
    channel (the payload projection) through the discrete schedule."""
    ids, targets, _ = make_batch(torch.Generator().manual_seed(3), 4)
    for chunk_tokens in (1, 8):
        model = SeparatorModel(chunk_tokens=chunk_tokens)
        loss = model.loss(ids, targets)
        assert torch.isfinite(loss)
        loss.backward()
        gradient = model.blocks[0].mixer.value.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
        picks = probe_predictions(model, ids)
        assert picks.shape == (4, N_PROBES)
        assert float(((picks >= VAL_BASE) & (picks < VAL_BASE + N_VALUES)).float().mean()) == 1.0


def test_chance_baseline_of_untrained_predictions():
    """An untrained model's value-restricted accuracy scatters around 1/32 on a
    large batch (the chance level every prediction is stated against)."""
    ids, targets, _ = make_batch(torch.Generator().manual_seed(19), 512)
    model = SeparatorModel(chunk_tokens=8)
    picks = probe_predictions(model, ids)
    acc = float((picks == targets).float().mean())
    assert abs(acc - CHANCE) < 0.05, acc
