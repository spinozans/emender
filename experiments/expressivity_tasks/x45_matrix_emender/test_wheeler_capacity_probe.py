"""WHEELER-ORDER capacity probe -- the machine-checks behind the
pre-registered predictions (``wheeler_capacity_probe.py``, predictions in
``RESULT_WHEELER_CAPACITY.md``):

* **The interval read obeys the Lean truncation law, as numbers** (the design
  doc §3): the full-population denominator makes a survivor's weight EXACTLY
  invariant under interval widening
  (``test_interval_read_survivor_invariance`` -- Lean
  ``full_coefficients_truncation_invariant``), the retained mass is monotone
  in the width and the omitted mass is exactly the difference
  (``test_interval_read_omitted_mass_monotone`` --
  ``full_truncation_mass_monotone``), and an empty interval / all-dead read is
  exactly zero, never renormalised noise
  (``test_interval_read_empty_is_zero``).
* **Update-in-run ACCUMULATES (the registration-level semantic)**: a second
  write within ``tau_match_dist`` of a live row adds its mass -- the row is
  the SUM of the payloads, run length 2 -- while the pointwise arm's
  registered CORRECT REPLACES the row's content (run length 1, the identity
  cache), and the accumulate is exactly the registered ``erase=False`` lane
  (``test_update_in_run_accumulates_mass``).
* **MERGE coalesces before POP under pressure** (the design doc §4): a full
  store with a coalescible adjacent pair merges mass-preservingly and NO
  content is lost, while a full store with no coalescible pair POPs the
  farthest row and its mass is gone (``test_merge_coalesces_before_pop``,
  ``test_pop_closes_interval_when_no_pair``).
* **The pointwise arm is the SEP-KV module verbatim** -- structural
  (``test_pointwise_arm_is_sepkv_verbatim``) -- and both arms train, backward
  and produce finite gradients through the probe loss
  (``test_both_arms_backward_is_finite``), with the ordered arm's gradient
  reaching the key channel through the read
  (``test_ordered_gradient_reaches_key_projection``).
* **The task family is what the pre-registration says**
  (``test_span_run_task_construction``): the window layout, the probe = the
  bare probed member, the target = that member's registered value, distinct
  members, and probe-position coverage.
"""
from __future__ import annotations

import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import CORRECT
from experiments.expressivity_tasks.x45_matrix_emender.key_separation_probe import (
    SeparatedModel, separated_oracle_scan)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy)
from experiments.expressivity_tasks.x45_matrix_emender.wheeler_capacity_probe import (
    BOS, CHANCE, COMB_BASE, DIST_BASE, MEM_BASE, N_MEMBERS, N_VALUES, O_INSERT,
    O_MERGE, O_NONE, O_POP, O_UPDATE, PAD, RANK, READ_SCALE, TAU_MATCH_DIST,
    VAL_BASE, VOCAB, WIDTH, make_model, make_task, ordered_oracle_scan,
    ordered_read_weights)

RNG = 3401


def _state_fixture(device="cpu", seed=RNG):
    """Five live rows at spread keys, distinct payloads, full strength."""
    g = torch.Generator().manual_seed(seed)
    keys = torch.tensor([[[-0.8, -0.3, 0.0, 0.3, 0.8, 0.0, 0.0, 0.0]]],
                        device=device)
    payload = torch.randn(1, 5, 32, generator=g).to(device)
    state = torch.zeros(1, 1, 8, 32, device=device)
    state[0, 0, :5] = payload
    live = torch.zeros(1, 1, 8, dtype=torch.bool, device=device)
    live[0, 0, :5] = True
    return keys, state, live


def test_interval_read_survivor_invariance():
    keys, state, live = _state_fixture()
    query = torch.zeros(1, 1)
    narrow, _ = ordered_read_weights(query, keys, state, live,
                                     read_scale=2.0, width=0.2)
    wide, _ = ordered_read_weights(query, keys, state, live,
                                  read_scale=2.0, width=0.6)
    # row 2 (key 0.0) is inside BOTH intervals: its weight must be exactly
    # invariant (the denominator spans the full population either way).
    assert narrow[0, 0, 2].item() == wide[0, 0, 2].item()


def test_interval_read_omitted_mass_monotone():
    keys, state, live = _state_fixture()
    query = torch.zeros(1, 1)
    retained, omitted = [], []
    for width in (0.1, 0.25, 0.45, 0.7, 1.0):
        w, om = ordered_read_weights(query, keys, state, live,
                                     read_scale=2.0, width=width)
        retained.append(w.sum().item())
        omitted.append(om.item())
    for a, b in zip(retained, retained[1:]):
        assert b >= a - 1e-6          # monotone: widening never drops mass
    # the omitted mass is exactly what the widening recovered
    for i in range(len(retained) - 1):
        assert pytest.approx(omitted[i] - omitted[i + 1], abs=1e-6) == \
            retained[i + 1] - retained[i]


def test_interval_read_empty_is_zero():
    keys, state, live = _state_fixture()
    far = torch.ones(1, 1) * 5.0      # no live row within any width
    w, omitted = ordered_read_weights(far, keys, state, live,
                                      read_scale=2.0, width=0.25)
    assert float(w.abs().sum()) == 0.0
    assert omitted.item() > 0.0       # the mass is disclosed, not renormalised
    dead = torch.zeros(1, 1, 8, dtype=torch.bool)
    w2, om2 = ordered_read_weights(far, keys, state, dead,
                                   read_scale=2.0, width=0.25)
    assert float(w2.abs().sum()) == 0.0 and float(om2.abs().sum()) == 0.0


def _ordered_write(batch, tokens, keys, payloads, decay=None):
    """A tiny helper scan: per-token scalar write keys/payloads."""
    heads, value_dim = 1, payloads.shape[-1]
    write_keys = keys.to(payloads)
    query_keys = keys.to(payloads)
    if decay is None:
        decay = torch.ones(batch, tokens, heads)
    return ordered_oracle_scan(payloads.view(batch, tokens, heads, value_dim),
                               write_keys.view(batch, tokens, heads),
                               query_keys.view(batch, tokens, heads),
                               decay, rank=RANK, collect=True)


def test_update_in_run_accumulates_mass():
    # one row at key 0.0 with payload p1; a second write within tau_match_dist
    p1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    p2 = torch.tensor([[0.0, 2.0, 0.0, 0.0]])
    keys = torch.tensor([[0.0, 0.05]])
    payloads = torch.stack((p1, p2), dim=1)                 # (1, 2, 4)
    result = _ordered_write(1, 2, keys, payloads)
    assert result["op"][0, 0, 0].item() == O_INSERT
    assert result["op"][0, 1, 0].item() == O_UPDATE
    row = result["state"][0, 0, 0]
    assert torch.allclose(row, torch.tensor([1.0, 2.0, 0.0, 0.0]),
                          atol=1e-6)         # the SUM: mass preserved
    assert result["runlen"][0, 0, 0].item() == 2
    assert result["keys"][0, 0, 0].item() == pytest.approx(0.0)  # fixed centre


def test_pointwise_correct_replaces_where_ordered_accumulates():
    # the SAME two matching writes through the pointwise arm's imported
    # oracle: the registered CORRECT REPLACES the row (run length 1).
    g = torch.Generator().manual_seed(RNG)
    payload = torch.randn(1, 2, 1, 8, generator=g)
    key = torch.randn(1, 2, 1, 16, generator=g)
    key[:, 1] = key[:, 0]                     # identical keys -> CORRECT
    decay = torch.ones(1, 2, 1)
    result = separated_oracle_scan(payload, key, key, decay,
                                   policy=RoutingPolicy(), rank=RANK,
                                   collect=False)
    result2 = separated_oracle_scan(payload, key, key, decay,
                                     policy=RoutingPolicy(), rank=RANK,
                                     collect=True)
    assert result2["op"][0, 1, 0].item() == CORRECT
    row = result["state"][0, 0, result2["slot"][0, 1, 0].item()]
    assert torch.allclose(row, payload[0, 1, 0], atol=1e-6)  # replaced, not summed


def test_merge_coalesces_before_pop():
    # full store (rank 8, 8 live rows); all 8 pre-writes pairwise farther
    # than tau_match_dist (so each INSERTs -- the ordering constraint that
    # keeps a coalescible pair creatable), tightest adjacent gap 0.16 <=
    # tau_run; the pressured write at 0.44 matches nothing
    keys = torch.zeros(1, 9)
    keys[0, :8] = torch.tensor([-0.9, -0.62, -0.34, -0.18, 0.02, 0.3, 0.58, 0.86])
    keys[0, 8] = 0.44                          # the pressured insert
    payloads = torch.randn(1, 9, 4, generator=torch.Generator()
                           .manual_seed(RNG))  # distinct payloads
    result = _ordered_write(1, 9, keys, payloads)
    assert result["op"][0, 8, 0].item() == O_MERGE
    state, runlen = result["state"][0, 0], result["runlen"][0, 0]
    # mass preserved: the tightest pair (keys -0.34/-0.18, slots 2/3) now
    # lives in ONE row (the lower slot id), the source slot holds the new
    # write, and nothing was destroyed
    assert torch.allclose(state[2], payloads[0, 2] + payloads[0, 3], atol=1e-6)
    assert torch.allclose(state[3], payloads[0, 8], atol=1e-6)
    assert runlen[2].item() == 2
    assert result["live"][0, 0].all()


def test_pop_closes_interval_when_no_pair():
    # full store, NO adjacent pair within tau_run (all gaps 0.26 > 0.25 -- the
    # sparsest 8-row layout the tanh key range admits), far victim: POP fires
    # and the victim's mass is gone (close-interval is lossy by design)
    keys = torch.zeros(1, 9)
    keys[0, :8] = torch.linspace(-0.91, 0.91, 8)
    keys[0, 8] = 0.0
    payloads = torch.randn(1, 9, 4, generator=torch.Generator()
                           .manual_seed(RNG))
    result = _ordered_write(1, 9, keys, payloads)
    assert result["op"][0, 8, 0].item() == O_POP
    # the write filed into the popped slot; the victim's payload is gone
    slot = result["slot"][0, 8, 0].item()
    assert torch.allclose(result["state"][0, 0, slot], payloads[0, 8], atol=1e-6)
    assert result["live"][0, 0].all()
    # no row holds the old victim payload any more
    assert not any(torch.allclose(result["state"][0, 0, j], payloads[0, slot],
                                  atol=1e-6)
                   for j in range(RANK))


def test_none_drops_when_nothing_is_far_enough():
    # the registered tau_pop guard's index-term form: pressure with no match,
    # no coalescible pair and no genuinely-far victim -> the write is DROPPED
    # and nothing is destroyed.  (At the registered radii a full rank-8 store
    # always has a coalescible pair or a far-enough victim, so this branch is
    # exercised with an explicit policy: tau_run tightened below the key
    # spacing, tau_pop_dist raised so no victim qualifies.)
    keys = torch.zeros(1, 9)
    keys[0, :8] = torch.linspace(-0.42, 0.42, 8)
    keys[0, 8] = 0.55
    payloads = torch.randn(1, 9, 4, generator=torch.Generator()
                           .manual_seed(RNG))
    write_keys = keys.view(1, 9, 1)
    result = ordered_oracle_scan(
        payloads.view(1, 9, 1, 4), write_keys, write_keys,
        torch.ones(1, 9, 1), rank=RANK,
        policy=dict(width=WIDTH, read_scale=READ_SCALE,
                    tau_match_dist=TAU_MATCH_DIST, tau_run=0.05,
                    tau_pop_dist=2.0))
    assert result["op"][0, 8, 0].item() == O_NONE
    assert torch.allclose(result["state"][0, 0, :8],
                          payloads[0, :8], atol=1e-6)   # nothing touched


def test_pointwise_arm_is_sepkv_verbatim():
    task = make_task(4)
    model = make_model(task, "pointwise")
    assert isinstance(model, SeparatedModel)
    mixers = [b.mixer for b in model.blocks]
    assert all(m.rank == RANK for m in mixers)
    assert all(m.key_dim == 16 for m in mixers)      # free 16-D vector keys
    assert model.policy.ops and model.policy.tau_merge == 0.85


def _tiny_batch(span=4, count=3):
    task = make_task(span)
    g = torch.Generator().manual_seed(RNG)
    return task, *task.make_batch(g, count)


def test_span_run_task_construction():
    for span in (2, 4, 8, 64):
        task = make_task(span)
        g = torch.Generator().manual_seed(RNG)
        ids, targets, aux = task.make_batch(g, 16)
        window = 7 + span
        assert ids.shape == (16, window) and targets.shape == (16, 1)
        assert (ids[:, 0] == BOS).all() and (ids[:, 1] == PAD).all() \
            and (ids[:, 2] == PAD).all()
        members = aux["members"]
        ordered_members = members.sort(dim=1).values
        assert (ordered_members[:, 1:] != ordered_members[:, :-1]).all() \
            # members drawn without replacement, in every window
        for i in range(span):
            expected = COMB_BASE + members[:, i] * N_VALUES + aux["values"][:, i]
            assert (ids[:, 3 + i] == expected).all()
        probe = 5 + span
        j = aux["j"]
        m_j = members.gather(1, j.unsqueeze(1)).squeeze(1)
        v_j = aux["values"].gather(1, j.unsqueeze(1)).squeeze(1)
        assert (ids[:, probe] == MEM_BASE + m_j).all()      # the bare member
        assert (ids[:, probe + 1] == VAL_BASE + v_j).all()  # the teacher-forced
        assert (targets[:, 0] == VAL_BASE + v_j).all()
        assert CHANCE == 1.0 / 32


def test_probe_position_covers_the_span():
    # over many draws the probe hits early and late span positions alike
    task = make_task(16)
    g = torch.Generator().manual_seed(RNG)
    _, _, aux = task.make_batch(g, 4096)
    j = aux["j"]
    assert int(j.min()) == 0 and int(j.max()) == 15
    assert j.float().mean().item() == pytest.approx(7.5, abs=0.5)


def test_both_arms_backward_is_finite():
    for arm in ("pointwise", "ordered"):
        task, ids, targets, _ = _tiny_batch()
        model = make_model(task, arm)
        loss = model.loss(ids, targets)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert float(loss) > 0.0


def test_ordered_gradient_reaches_key_projection():
    task, ids, targets, _ = _tiny_batch()
    model = make_model(task, "ordered")
    loss = model.loss(ids, targets)
    loss.backward()
    for block in model.blocks:
        assert block.mixer.key.weight.grad is not None
        assert torch.isfinite(block.mixer.key.weight.grad).all()
        assert block.mixer.key.weight.grad.abs().sum().item() > 0.0


def test_ordered_scan_is_deterministic():
    g = torch.Generator().manual_seed(RNG)
    payloads = torch.randn(2, 6, 1, 8, generator=g)
    keys = (torch.rand(2, 6, 1, generator=g) * 1.6 - 0.8)
    decay = torch.rand(2, 6, 1, generator=g) * 0.2 + 0.8
    first = ordered_oracle_scan(payloads, keys, keys, decay, rank=RANK)
    second = ordered_oracle_scan(payloads, keys, keys, decay, rank=RANK)
    assert torch.equal(first["state"], second["state"])
    assert torch.equal(first["reads"], second["reads"])
    assert torch.equal(first["op"], second["op"])


def test_make_batch_helpers_export():
    # the module's public constants line up (the vocabulary layout the
    # pre-registration states)
    assert VAL_BASE == MEM_BASE + N_MEMBERS
    assert COMB_BASE == VAL_BASE + N_VALUES
    assert DIST_BASE == COMB_BASE + N_MEMBERS * N_VALUES
    assert VOCAB == DIST_BASE + 8
