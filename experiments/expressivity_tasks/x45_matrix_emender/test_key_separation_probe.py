"""SEP-KV -- the key-separation necessity probe, gated.

The machine-checks behind the pre-registered predictions
(``key_separation_probe.py``, predictions in ``RESULT_KEY_SEPARATION.md``):

* **The collision lemma is exact and parameter-free.**  Two occurrences of the
  same token produce bit-identical layer-1 payloads
  (``test_identical_payloads_bit_equal``), and while the first copy's row is
  live the second copy's routing similarity to it is exactly 1, so the
  registered router files the second copy onto the first's row -- never a
  separate APPEND (``test_identity_rewrite_is_mechanical``).  Two live rows
  that DO share content receive exactly equal read masses
  (``test_equal_mass_on_identical_rows_is_exact``): the read channel's
  mechanical role-blindness.
* **M2's order feature is absent from the content channel** (the supervisor's
  receipt requirement, checked before the sweep): at layer 1 the (P, Q) and
  (Q, P) frames produce the same row directions and the same read
  probabilities (up to the decay scale-invariance rounding), the two appends
  do not interact under tau_match, and the only state differential is the
  strength norms -- the accumulated per-token decay factors, a fixed function
  of token identities that cannot carry a per-window-random binding
  (``test_m2_order_blindness_is_content_exact``).
* **The separated oracle is correct and differentiable**: keys are filed and
  re-filed per the five-op chain, the read retrieves by key cosine, gradients
  reach the key, query and value projections
  (``test_separated_oracle_files_and_retrieves_by_key``,
  ``test_separated_oracle_read_channel_guards``,
  ``test_separated_oracle_gradient_reaches_key_and_value_channels``).
* **The payload run-integral identity** stated in the design notes: a slot's
  value between re-files IS the geometrically decayed run-integral of
  everything written to it (``test_slot_value_is_decayed_run_integral``).
"""
from __future__ import annotations

import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    APPEND, CORRECT, MERGE, sequential_matrix_scan)
from experiments.expressivity_tasks.x45_matrix_emender.key_separation_probe import (
    BOS, CHANCE, M1_COPY_POSITIONS, M1_PROBE, M1_PROBE_POSITIONS, M1_R0, M1_R1,
    M1_REVEAL_POSITIONS, M1_WINDOW, M2_P, M2_Q, M2_COPY_POSITIONS,
    M2_PROBE_POSITIONS, M2_REVEAL_POSITIONS, M2_WINDOW, N_VALUES, PAD, TASKS,
    VAL_BASE, X_BASE, FusedModel, SeparatedModel, causal_context,
    collect_channels, probe_predictions, separated_oracle_scan)
from experiments.expressivity_tasks.x45_matrix_emender.read import (
    attention_probs, read_weights)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy, route_chunk)
from experiments.expressivity_tasks.x45_matrix_emender.scan import (
    routed_chunk_scan)

POLICY = RoutingPolicy()


# ------------------------------------------------------------ task generators
@pytest.mark.parametrize("name", ["m1", "m2"])
def test_window_layout_and_binding(name):
    task = TASKS[name]
    ids, targets, x_id = task.make_batch(torch.Generator().manual_seed(7), 64)
    assert ids.shape == (64, task.window) and targets.shape == (64, 2)
    assert (ids[:, 0] == BOS).all() and (ids[:, 1] == PAD).all()
    copy1, copy2 = task.copy_positions
    # the collision premise: the SAME ambiguous token in both roles
    assert (ids[:, copy1] == ids[:, copy2]).all()
    assert (ids[:, copy1] >= X_BASE).all() and (ids[:, copy1] < X_BASE + 4).all()
    # the two continuations are distinct value tokens (binding unambiguous)
    assert (targets[:, 0] != targets[:, 1]).all()
    assert (targets >= VAL_BASE).all() and (targets < VAL_BASE + N_VALUES).all()
    # the reveals carry the targets (teacher-forced)
    reveal1, reveal2 = task.reveal_positions
    assert (ids[:, reveal1] == targets[:, 0]).all()
    assert (ids[:, reveal2] == targets[:, 1]).all()
    if name == "m1":
        # probes are the combined (role, x) tokens
        assert (ids[:, M1_PROBE_POSITIONS[0]] == M1_PROBE(0, x_id)).all()
        assert (ids[:, M1_PROBE_POSITIONS[1]] == M1_PROBE(1, x_id)).all()
        assert (ids[:, 3] == M1_R0).all() and (ids[:, 6] == M1_R1).all()
    else:
        # the role is the ORDER of the shared frame tokens, absent from content
        assert (ids[:, 3] == M2_P).all() and (ids[:, 4] == M2_Q).all()
        assert (ids[:, 7] == M2_Q).all() and (ids[:, 8] == M2_P).all()
        assert (ids[:, 11] == M2_P).all() and (ids[:, 12] == M2_Q).all()
        assert (ids[:, 15] == M2_Q).all() and (ids[:, 16] == M2_P).all()


@pytest.mark.parametrize("name", ["m1", "m2"])
def test_generator_determinism(name):
    make = TASKS[name].make_batch
    a = make(torch.Generator().manual_seed(11), 8)
    b = make(torch.Generator().manual_seed(11), 8)
    c = make(torch.Generator().manual_seed(12), 8)
    for x, y in zip(a, b):
        assert torch.equal(x, y)
    assert not torch.equal(a[0], c[0])


# ------------------------------------------------------- the collision lemma
def test_identical_payloads_bit_equal():
    """Layer 1 of the fused arm: the payload is a fixed function of the token,
    so the two X copies' payloads are BIT-IDENTICAL for every parameter
    setting -- the premise of the collision lemma."""
    ids, targets, _ = TASKS["m1"].make_batch(torch.Generator().manual_seed(3), 8)
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        model = FusedModel(TASKS["m1"])
        block = model.blocks[0]
        with torch.no_grad():
            h = block.norm1(model.embed(ids))
            mixer = block.mixer
            payload = mixer.value(h).view(*h.shape[:2], mixer.heads,
                                         mixer.value_dim)
            query = mixer.query(h).view(*h.shape[:2], mixer.heads,
                                       mixer.value_dim)
        copy1, copy2 = M1_COPY_POSITIONS
        assert torch.equal(payload[:, copy1], payload[:, copy2]), seed
        assert torch.equal(query[:, copy1], query[:, copy2]), seed


def test_identity_rewrite_is_mechanical():
    """The collision lemma's write half: with the first copy's row live (any
    content direction, any decay whose accumulated product stays above the
    registered strength clamp), the second copy's similarity to it is EXACTLY
    1, so the registered router files it onto that row -- CORRECT or MERGE,
    never a separate APPEND -- whatever the parameters."""
    value_dim, rank = 32, 16
    # orthonormal token directions: no intervening CORRECT can steal the X row
    # (the geometry lottery is excluded so the lemma's premise is pure)
    directions = torch.linalg.qr(torch.randn(value_dim, 7, generator=(
        torch.Generator().manual_seed(5)))).Q.T      # (7, 32), rows orthonormal
    tokens = [BOS, PAD, M1_R0, X_BASE, VAL_BASE, M1_R1, VAL_BASE + 1]
    layout = [BOS, PAD, PAD, M1_R0, X_BASE, VAL_BASE, M1_R1, X_BASE,
              VAL_BASE + 1] + [PAD] * (M1_WINDOW - 9)
    for decay_value in (0.999, 0.9, 0.5, 0.05):
        payload = torch.empty(1, M1_WINDOW, 1, value_dim)
        for t, tok in enumerate(layout):
            payload[0, t, 0] = directions[tokens.index(tok)]
        query = torch.randn(1, M1_WINDOW, 1, value_dim)
        decay = torch.full((1, M1_WINDOW, 1), decay_value)
        result = routed_chunk_scan(query, payload, decay, router=route_chunk,
                                   policy=POLICY, rank=rank, rank_max=rank,
                                   chunk_tokens=1, collect=True, fused=False)
        op = result.schedule.op[0, :, 0, 0]
        slot = result.schedule.slot[0, :, 0, 0]
        assert int(op[M1_COPY_POSITIONS[0]]) == APPEND
        assert int(op[M1_COPY_POSITIONS[1]]) in (CORRECT, MERGE)
        assert int(slot[M1_COPY_POSITIONS[1]]) == int(slot[M1_COPY_POSITIONS[0]]), \
            "the second copy must file onto the first's row, at any decay"


def test_equal_mass_on_identical_rows_is_exact():
    """The collision lemma's read half: two live rows with identical content
    get EXACTLY equal read weights (cosine is content-only), so no read can
    prefer one -- the mechanical indistinguishability of identical memories.
    The separated read, given the same two values under distinct keys,
    separates them."""
    query = torch.randn(1, 1, 1, 32)
    rows = torch.zeros(1, 1, 4, 32)
    rows[0, 0, 0] = rows[0, 0, 1] = torch.randn(32)      # identical content
    rows[0, 0, 2] = torch.randn(32)
    live = torch.tensor([[[True, True, True, False]]])
    w = read_weights(query, rows, live, scale=2.0)
    assert torch.equal(w[0, 0, 0, 0], w[0, 0, 0, 1]), \
        "identical content must draw exactly equal read mass"
    # and the separated oracle separates the same values under distinct keys
    payload = torch.zeros(1, 3, 1, 32)
    payload[0, 0, 0] = payload[0, 1, 0] = rows[0, 0, 0]   # the same value twice
    keys = torch.zeros(1, 3, 1, 4)
    keys[0, 0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])   # copy 1's address
    keys[0, 1, 0] = torch.tensor([0.0, 1.0, 0.0, 0.0])   # copy 2's address
    keys[0, 2, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])   # the probe asks copy 1
    decay = torch.full((1, 3, 1), 0.999)
    result = separated_oracle_scan(payload, keys, keys, decay, policy=POLICY,
                                   rank=4, collect=True)
    weights = result["weights"][0, 2, 0]                  # (rank,)
    slots = result["slot"][0, :, 0]
    assert int(slots[0]) != int(slots[1]), "distinct keys must file apart"
    ratio = float(weights[int(slots[0])]) / max(float(weights[int(slots[1])]), 1e-12)
    assert ratio > 5.0, "the read must address the role-matching copy via its key"


def _layer1_state(model, ids):
    """The layer-1 registered scan of one window prefix (the fused arm's own
    mixer, chunk granularity 1)."""
    block = model.blocks[0]
    mixer = block.mixer
    with torch.no_grad():
        h = block.norm1(model.embed(ids))
        query = mixer.query(h).view(1, -1, 1, mixer.value_dim)
        payload = mixer.value(h).view(1, -1, 1, mixer.value_dim)
        decay = torch.sigmoid(mixer.decay(h)).clamp(min=1e-4, max=1 - 1e-6)
        result = routed_chunk_scan(query, payload, decay, router=route_chunk,
                                   policy=POLICY, rank=mixer.rank,
                                   rank_max=mixer.rank, chunk_tokens=1,
                                   collect=True, fused=False)
    return result


def _decay_at(model, ids, position):
    block = model.blocks[0]
    with torch.no_grad():
        h = block.norm1(model.embed(ids))
        return torch.sigmoid(block.mixer.decay(h))[0, position]


def test_m2_order_blindness_is_content_exact():
    """M2's order feature is absent from the content channel (checked before
    the sweep, per the supervisor's receipt requirement): the (P, Q) and (Q, P)
    frames produce the same row directions and the same read probabilities (up
    to the decay scale-invariance rounding), both frames' writes APPEND
    without interacting, and the only state differential is the strength norms
    -- the per-token decay factors, a fixed function of token identities that
    cannot carry a per-window-random binding."""
    torch.manual_seed(13)
    model = FusedModel(TASKS["m2"])
    ids0 = torch.tensor([[BOS, PAD, PAD, M2_P, M2_Q, X_BASE, VAL_BASE]])
    ids1 = torch.tensor([[BOS, PAD, PAD, M2_Q, M2_P, X_BASE, VAL_BASE]])
    r0, r1 = _layer1_state(model, ids0), _layer1_state(model, ids1)
    # the two frame writes do not interact: both APPEND under tau_match 0.45
    for result in (r0, r1):
        assert int(result.schedule.op[0, 3, 0, 0]) == APPEND
        assert int(result.schedule.op[0, 4, 0, 0]) == APPEND
    # the row directions are order-blind: the live rows match one-to-one
    def unit_rows(result):
        rows = result.state[0, 0]
        live = result.live[0, 0]
        return rows[live] / rows[live].norm(dim=-1, keepdim=True).clamp_min(1e-6)
    u0, u1 = unit_rows(r0), unit_rows(r1)
    assert u0.shape == u1.shape
    cos = u0 @ u1.T                                        # (4, 4)
    match = cos.argmax(-1)
    assert float(cos.max(-1).values.min()) > 1 - 1e-4, \
        "every frame-0 row must find its frame-1 twin"
    assert len(set(match.tolist())) == cos.shape[0], "the match must be a bijection"
    # read probabilities for arbitrary queries agree up to the row permutation
    # (the STRENGTH channel is the one differential, compared below)
    probe = torch.randn(1, 1, 16, 32)          # (b, h, 16 probe queries, d_v)
    with torch.no_grad():
        p0 = attention_probs(probe, r0.state[:, :, :4], r0.live[:, :, :4], 2.0)
        p1 = attention_probs(probe, r1.state[:, :, :4], r1.live[:, :, :4], 2.0)
    s0 = p0[0, 0].sort(-1).values
    s1 = p1[0, 0].sort(-1).values
    assert float((s0 - s1).abs().max()) < 1e-5, \
        "the read's probability channel must not see the frame order"
    # the only differential: the strength norms carry the per-token decay
    # factors (the P row misses lambda_Q in frame-0, the Q row misses
    # lambda_P in frame-1; everything else cancels in the ratios)
    lam_p = float(_decay_at(model, ids0, 3))
    lam_q = float(_decay_at(model, ids0, 4))
    n0 = r0.state[0, 0].norm(dim=-1)[r0.live[0, 0]]
    n1 = r1.state[0, 0].norm(dim=-1)[r1.live[0, 0]]
    # frame-0 slots: [BOS, PAD, P, Q]; frame-1 slots: [BOS, PAD, Q, P]
    assert abs(float(n0[0] / n1[0]) - 1.0) < 1e-4, "the BOS row is order-blind"
    assert abs(float(n0[2] / n1[3]) - lam_q) < 1e-3, \
        "the P row's strength differential is exactly the lambda_Q factor"
    assert abs(float(n1[2] / n0[3]) - lam_p) < 1e-3, \
        "the Q row's strength differential is exactly the lambda_P factor"


# ------------------------------------------------- the separated arm (oracle)
def test_separated_oracle_files_and_retrieves_by_key():
    """The separated oracle's core mechanics: APPEND files (key, value); a
    second write whose key-cosine exceeds tau_match CORRECTs the same row and
    re-files its address; the read retrieves by key cosine, not content."""
    value_dim, key_dim, rank = 8, 4, 8
    payload = torch.zeros(1, 3, 1, value_dim)
    payload[0, 0, 0] = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0])
    payload[0, 1, 0] = torch.tensor([0, 1.0, 0, 0, 0, 0, 0, 0])   # new value
    payload[0, 2, 0] = torch.tensor([0, 0, 1.0, 0, 0, 0, 0, 0])   # probe payload
    keys = torch.zeros(1, 3, 1, key_dim)
    keys[0, 0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    keys[0, 1, 0] = torch.tensor([0.9, 0.435889894354, 0.0, 0.0])  # cos ~0.9 to key 0
    keys[0, 2, 0] = torch.tensor([0.0, 1.0, 0.0, 0.0])            # asks the re-filed row
    decay = torch.full((1, 3, 1), 0.999)
    result = separated_oracle_scan(payload, keys, keys, decay, policy=POLICY,
                                   rank=rank, collect=True)
    assert int(result["op"][0, 1, 0]) == CORRECT, "key-cos >= tau_match must CORRECT"
    assert int(result["slot"][0, 1, 0]) == int(result["slot"][0, 0, 0])
    # the re-filed row now answers the probe: the read returns the NEW value
    assert float(result["reads"][0, 2, 0, 1]) > 0.9
    assert float(result["reads"][0, 2, 0, 0]) < 0.1


def test_separated_oracle_read_channel_guards():
    """The all-dead read is EXACTLY zero (no rows live yet), and rows whose
    content is exactly zero read exactly zero whatever their keys."""
    payload = torch.zeros(1, 3, 1, 8)
    keys = torch.randn(1, 3, 1, 4)
    decay = torch.full((1, 3, 1), 0.99)
    result = separated_oracle_scan(payload, keys, keys, decay, policy=POLICY,
                                   rank=4, collect=True)
    assert float(result["weights"][:, 0].abs().max()) == 0.0, \
        "the all-dead read must put exactly zero mass everywhere"
    assert float(result["reads"].abs().max()) == 0.0


def test_separated_oracle_gradient_reaches_key_and_value_channels():
    """The oracle is differentiable in the registered doctrine's split: the
    read weights are attached (gradient reaches the key projections through
    the stored keys), decisions are detached, payloads carry gradient."""
    torch.manual_seed(23)
    payload = torch.randn(2, 6, 1, 8).detach().requires_grad_(True)
    keys = torch.randn(2, 6, 1, 4).detach().requires_grad_(True)
    qkeys = torch.randn(2, 6, 1, 4).detach().requires_grad_(True)
    decay = torch.full((2, 6, 1), 0.99)
    result = separated_oracle_scan(payload, keys, qkeys, decay, policy=POLICY,
                                   rank=4, collect=False)
    result["reads"].sum().backward()
    assert keys.grad is not None and float(keys.grad.abs().sum()) > 0
    assert qkeys.grad is not None and float(qkeys.grad.abs().sum()) > 0
    assert payload.grad is not None and float(payload.grad.abs().sum()) > 0


def test_causal_context_window():
    x = torch.arange(24, dtype=torch.float32).view(1, 6, 4)
    ctx = causal_context(x, 4)
    assert ctx.shape == (1, 6, 4, 4)
    # position t sees x[t-3..t], clamped at 0
    for t in range(6):
        expected = [float(x[0, max(0, t - 3 + i), 0]) for i in range(4)]
        got = [float(ctx[0, t, i, 0]) for i in range(4)]
        assert expected == got


def test_separated_write_keys_separate_mechanically():
    """The two X copies' emitted keys differ MECHANICALLY (their write contexts
    differ), whatever the parameters -- the contrast with the fused arm's
    bit-identical payloads."""
    ids, targets, _ = TASKS["m1"].make_batch(torch.Generator().manual_seed(29), 16)
    for seed in (0, 1):
        torch.manual_seed(seed)
        model = SeparatedModel(TASKS["m1"])
        with torch.no_grad():
            _, results = model.forward(ids, collect=True)
        emitted = results[0]["emitted_keys"][:, :, 0, :]   # layer 0, head 0
        k1 = emitted[:, M1_COPY_POSITIONS[0]]
        k2 = emitted[:, M1_COPY_POSITIONS[1]]
        cos = (k1 * k2).sum(-1) / (k1.norm(dim=-1) * k2.norm(dim=-1)).clamp_min(1e-6)
        assert float(cos.max()) < 0.999, \
            "contexts differ, so the emitted keys cannot coincide"


# ------------------------------------------------------ the payload identity
def test_slot_value_is_decayed_run_integral():
    """The design note's provable characterization, machine-checked: between
    re-files, a slot's value IS the geometrically decayed run-integral of
    everything written to it -- each payload survives to the next key
    weighted by the product of the decays since its write."""
    torch.manual_seed(31)
    batch, tokens, rank, value_dim = 1, 12, 3, 8
    payload = torch.randn(batch, tokens, 1, value_dim)
    decay = torch.rand(batch, tokens, 1) * 0.2 + 0.8
    # force every write into slot 1 with APPEND semantics (no erase)
    from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
        MatrixSchedule)
    schedule = MatrixSchedule(
        slot=torch.ones(batch, tokens, 1, 1, dtype=torch.int64),
        erase=torch.zeros(batch, tokens, 1, 1, dtype=torch.bool),
        payload=payload.unsqueeze(3),
        op=torch.full((batch, tokens, 1, 1), APPEND, dtype=torch.int64),
        live_after=torch.ones(batch, tokens, 1, 1, dtype=torch.bool))
    weights = torch.zeros(batch, tokens, 1, rank)
    _, state = sequential_matrix_scan(weights, decay, schedule)
    # closed form: S_1(T) = sum_u payload_u * prod_{u < s < T} lam_s
    closed = torch.zeros(value_dim)
    for u in range(tokens):
        closed = closed + payload[0, u, 0] * float(
            decay[0, u + 1:, 0].prod()) if u + 1 < tokens \
            else closed + payload[0, u, 0]
    assert torch.allclose(state[0, 0, 1], closed, atol=1e-5), \
        "the slot's value must be the geometrically decayed run-integral"


# ------------------------------------------------------------------- the arms
@pytest.mark.parametrize("name", ["m1", "m2"])
@pytest.mark.parametrize("arm", ["fused", "separated"])
def test_model_smoke_forward_backward(name, arm):
    ids, targets, _ = TASKS[name].make_batch(torch.Generator().manual_seed(3), 4)
    model = FusedModel(TASKS[name]) if arm == "fused" \
        else SeparatedModel(TASKS[name])
    loss = model.loss(ids, targets)
    assert torch.isfinite(loss)
    loss.backward()
    mixer = model.blocks[0].mixer
    assert mixer.value.weight.grad is not None
    assert float(mixer.value.weight.grad.abs().sum()) > 0
    if arm == "separated":
        assert mixer.key.weight.grad is not None
        assert float(mixer.key.weight.grad.abs().sum()) > 0
        assert mixer.query_key.weight.grad is not None
        assert float(mixer.query_key.weight.grad.abs().sum()) > 0
    picks = probe_predictions(model, ids, TASKS[name])
    assert picks.shape == (4, 2)
    assert bool(((picks >= VAL_BASE) & (picks < VAL_BASE + N_VALUES)).all())


@pytest.mark.parametrize("name", ["m1", "m2"])
def test_chance_baseline_of_untrained_predictions(name):
    ids, targets, _ = TASKS[name].make_batch(torch.Generator().manual_seed(19),
                                            512)
    torch.manual_seed(3401)
    for arm in ("fused", "separated"):
        model = FusedModel(TASKS[name]) if arm == "fused" \
            else SeparatedModel(TASKS[name])
        picks = probe_predictions(model, ids, TASKS[name])
        acc = float((picks == targets).float().mean())
        assert abs(acc - CHANCE) < 0.05, (arm, acc)


@pytest.mark.parametrize("name", ["m1", "m2"])
def test_channel_collectors_run(name):
    """Both arms' receipt collectors run on the untrained models and report
    the pre-registered channel fields (the receipts the sweep writes)."""
    ids, targets, _ = TASKS[name].make_batch(torch.Generator().manual_seed(41),
                                             16)
    torch.manual_seed(3401)
    fused = collect_channels(FusedModel(TASKS[name]), ids, targets, TASKS[name])
    assert fused["arm"] == "fused"
    layer0 = fused["per_layer"][0]
    assert "identity_rewrite_frac" in layer0
    assert "identical_live_row_pairs" in layer0
    assert "answer_row_mass" in layer0
    torch.manual_seed(3401)
    separated = collect_channels(SeparatedModel(TASKS[name]), ids, targets,
                                 TASKS[name])
    assert separated["arm"] == "separated"
    layer0 = separated["per_layer"][0]
    assert "copy_key_cos" in layer0
    assert "role_addressing" in layer0
    assert "answer_row_mass" in layer0


def test_composite_receipt_runs_at_depth_2():
    """The depth-substitution mechanism receipt runs on the depth-2 fused
    model and reports the layer-2 payload decomposition (value component +
    context components + the layer-1 read injection)."""
    from experiments.expressivity_tasks.x45_matrix_emender.key_separation_probe \
        import composite_receipt
    ids, targets, _ = TASKS["m1"].make_batch(
        torch.Generator().manual_seed(43), 8)
    torch.manual_seed(3401)
    model = FusedModel(TASKS["m1"], depth=2)
    receipt = composite_receipt(model, ids, TASKS["m1"])
    assert len(receipt["reveals"]) == 2
    for row in receipt["reveals"]:
        for field in ("cos_to_value", "cos_to_prev_1", "cos_to_prev_2",
                      "cos_to_prev_3", "cos_to_l1_read"):
            assert isinstance(row[field], float) and -1.0 <= row[field] <= 1.0


def test_arms_default_to_depth_one():
    """The necessity arms are DEPTH 1 by construction: one registered block,
    the airtight layer-1 scope of the collision lemma."""
    from experiments.expressivity_tasks.x45_matrix_emender.key_separation_probe \
        import make_model
    assert len(make_model("m1", "fused").blocks) == 1
    assert len(make_model("m2", "separated").blocks) == 1
    assert len(make_model("m1", "fused", depth=2).blocks) == 2
