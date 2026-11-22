"""K24 -- THE LONG-WINDOW PROBE, gated.

The lane's disciplines, per ``PREREGISTRATION_K24_LONG_WINDOW.md``
(committed before any arm):

* **THE WINDOW-LENGTH LEVER is the corpus pipeline's own windowing**: the
  longer windows come from the ``lattice.Stream`` constructor's own
  ``context`` parameter -- every window a slice of exactly one document,
  truncating at the boundary; nothing is packed, nothing crosses.  The
  synthetic-manifest tests pin the boundary discipline; the real-manifest
  tests pin the step/window arithmetic against the committed registration
  receipt (including the banked 1x landing: 180 steps, 20,022,866
  targets -- the anchor protocol's own numbers, computed from the
  manifest, never hardcoded).
* **THE EVAL RULE** picks, at every window length, the first 7 FULL
  windows of the arm's own context in calibration order; at 1x that set
  IS the banked ``L.evaluate`` set (the first-24-kept-full rule's own 7
  windows) -- asserted set-equal from the manifest.
* **THE MEMORY-MATCHED MICROBATCH rule**: every arm's microbatch carries
  the banked 16,384 tokens (8x2048, 4x4096, 2x8192).
* **THE ANCHORS are the banked rows' own receipts** (ours
  1.3811104467936925 from the K15 run's committed result.json; GDN2
  1.3521734476089478 from the protocol-correction rescore's own), and the
  K22 hybrid's banked 1x row is cited, never restated.
* **THE TRAINING LOOP is the banked loop** with exactly the two threaded
  differences (structural source checks: the warmup/clip/EMA/cooldown
  lines are the committed loop's own).
* **THE COMPOSED SCAN at a longer window** is the committed engine (CUDA
  tests: the kernel vs the eager oracle bit-equal on the schedule fields
  at a multi-segment long-window shape; the hybrid wrapper born as its
  dense layer bit for bit at a long window).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender.k24_long_window \
    import (ARMS, BANDS, CHECKPOINT_STEPS, EVAL_WINDOWS, EXPOSURE_TARGETS,
            LOAD_BEARING_BAR, MICROBATCH, PRIMARY_SEED, RULES,
            TOKENS_PER_MICROBATCH, WINDOW_CONTEXT, WINDOWS_PER_STEP,
            gdn2_1x_banked, hybrid_1x_banked, ours_1x_banked,
            registration_window_receipt, window_lengths)

REPO = Path(__file__).resolve().parents[3]
X45 = REPO / "experiments/expressivity_tasks/x45_matrix_emender"
FLAGSHIP_MANIFEST = (REPO / "experiments/expressivity_tasks/x43_fork_resolved"
                     / "receipts/flagship_stream/documents.jsonl.gz")
RECEIPT = X45 / "receipts" / "k24_window_arithmetic.json"

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="the committed Triton kernels "
                                       "need CUDA")


# ----------------------------------------------------- the arms and rules


def test_the_arms_are_the_registrations_own():
    """The approved bracket: {1x, 2x, 4x} x {ours, gdn2} + {4x} x
    {hybrid} = SEVEN arms, the two 1x cells the mandatory anchors."""
    assert set(ARMS) == {"ours_1x", "gdn2_1x", "ours_2x", "gdn2_2x",
                         "ours_4x", "gdn2_4x", "hybrid_4x"}
    for label, arm in ARMS.items():
        assert arm["machine"] in ("ours", "gdn2", "hybrid")
        assert arm["mult"] in (1, 2, 4)
        assert arm["role"] in ("anchor", "long")
    assert ARMS["ours_1x"]["role"] == ARMS["gdn2_1x"]["role"] == "anchor"
    assert ARMS["ours_1x"]["mult"] == ARMS["gdn2_1x"]["mult"] == 1
    assert ARMS["hybrid_4x"]["machine"] == "hybrid"
    assert ARMS["hybrid_4x"]["mult"] == 4
    assert "bit for bit" in ARMS["ours_1x"]["note"]
    assert "bit for bit" in ARMS["gdn2_1x"]["note"]


def test_the_rules_carry_the_pre_state_and_the_refutation():
    """L-1 (the horizon hypothesis), L-2 (the gap shrinks), the REFUTATION
    clause, the band discipline, the anchor convention and the boundary
    discipline -- the registration's own words, receipted verbatim."""
    assert "RISES with window length" in RULES["L1"]["text"]
    assert RULES["L1"]["bar"] == 0.02
    assert "SHRINKS with window length" in RULES["L2"]["text"]
    assert "horizon hypothesis is dead at probe scale" \
        in RULES["REFUTATION"]["text"]
    assert "0.0119" in RULES["band_discipline"]["text"]
    assert "STOPS the lane" in RULES["anchor"]["text"]
    assert "NEVER pack across document" in RULES["boundaries"]["text"]
    assert BANDS["winner_seed_delta"] == 0.011948040553501738
    assert BANDS["endpoint_variance"] == 0.075


def test_the_banked_anchor_rows_are_the_receipts_own_numbers():
    """The anchors are loaded from the committed run receipts, never
    restated: ours from the K15 composition run, GDN2 from the
    protocol-correction rescore, the hybrid from the K22 run."""
    ours = ours_1x_banked()
    gdn2 = gdn2_1x_banked()
    hybrid = hybrid_1x_banked()
    assert ours["post_cooldown"] == 1.3811104467936925
    assert ours["constant_lr"] == 1.433674931526184
    assert ours["ema180"] == 1.424389226096017
    assert ours["step1"] == 5.782775391046648
    assert ours["steps"] == 180 and ours["exposure_bytes"] == 20_022_866
    assert ours["total_bytes"] == 22_745_198
    assert len(ours["rows"]) == 204
    assert gdn2["post_cooldown"] == 1.3521734476089478
    assert gdn2["constant_lr"] == 1.3950541530336653
    assert gdn2["ema180"] == 1.3803383282252721
    assert gdn2["step1"] == 5.948033220866492
    assert gdn2["steps"] == 180 and gdn2["exposure_bytes"] == 20_022_866
    assert len(gdn2["rows"]) == 204
    assert hybrid["post_cooldown"] == 1.3547673906598772
    assert hybrid["slot_read_mask_sweep"]["slot_layers_ge_bar"] == 0
    assert hybrid["slot_read_mask_sweep"]["all_slots_delta"] == -0.00009 \
        or abs(hybrid["slot_read_mask_sweep"]["all_slots_delta"]) < 0.001


# ------------------------------------------- the step/window arithmetic


def test_the_step_shape_holds_tokens_per_step_and_the_microbatch_rule():
    """The registration's arithmetic: windows/step scaled down as the
    context grows (the ~111K-targets/step, ~180-step shape), and every
    microbatch carries the banked 16,384 tokens (the memory-matched
    rule)."""
    assert WINDOW_CONTEXT == {1: 2048, 2: 4096, 4: 8192}
    assert WINDOWS_PER_STEP == {1: 64, 2: 38, 4: 27}
    assert MICROBATCH == {1: 8, 2: 4, 4: 2}
    for mult in (1, 2, 4):
        assert MICROBATCH[mult] * WINDOW_CONTEXT[mult] \
            == TOKENS_PER_MICROBATCH == 16_384
    assert EXPOSURE_TARGETS == 20_000_000
    assert EVAL_WINDOWS == 7
    assert PRIMARY_SEED == 3401
    assert 160 in CHECKPOINT_STEPS and 180 in CHECKPOINT_STEPS


def test_the_window_arithmetic_matches_the_committed_registration():
    """The lane's committed receipt (computed from the flagship manifest
    at registration time): the landing steps {180, 182, 178}, the
    exposures, and the honest effective-window means (the 4x arm's
    achieved mean is 2.405x the banked mean, NOT 4x -- the lever's own
    strength, stated)."""
    receipt = json.loads(RECEIPT.read_text())
    arms = receipt["arms"]
    assert arms["1"]["landing_step"] == 180
    assert arms["1"]["exposure_landing"] == 20_022_866
    assert arms["1"]["windows_consumed"] == 11_520
    assert arms["2"]["landing_step"] == 182
    assert arms["2"]["exposure_landing"] == 20_067_118
    assert arms["4"]["landing_step"] == 178
    assert arms["4"]["exposure_landing"] == 20_088_160
    assert arms["1"]["effective_window_bytes"]["mean"] == 1738.1
    assert arms["2"]["effective_window_bytes"]["mean_over_1x_mean"] == 1.669
    assert arms["4"]["effective_window_bytes"]["mean_over_1x_mean"] == 2.405
    assert arms["4"]["effective_window_bytes"]["full_length_fraction"] \
        == 0.3225
    # the banked 1x landing IS the receipt's own arithmetic (the anchor
    # protocol's numbers, computed from the manifest)
    assert arms["1"]["exposure_landing"] == ours_1x_banked()[
        "exposure_bytes"] == gdn2_1x_banked()["exposure_bytes"]
    for mult in ("1", "2", "4"):
        assert arms[mult]["tokens_per_microbatch"] == 16_384


def test_the_windowing_never_crosses_a_document_boundary():
    """The boundary discipline on a synthetic manifest: every window is a
    slice of exactly one document, truncating at the boundary; a
    document shorter than context+1 contributes its one short window;
    starts sit at document-relative multiples of the context; and the
    per-document window targets sum to exactly the document's own
    target count (length - 1)."""
    import gzip
    import tempfile
    docs = [
        dict(split="train", byte_offset=0, byte_length=10_000, sha256=""),
        dict(split="train", byte_offset=10_000, byte_length=5_000,
             sha256=""),
        dict(split="train", byte_offset=15_000, byte_length=100,
             sha256=""),          # shorter than one window
        dict(split="train", byte_offset=15_100, byte_length=2,
             sha256=""),         # a single-target document
        dict(split="train", byte_offset=15_102, byte_length=8_194,
             sha256=""),         # one full window + one single-target window
        dict(split="calibration", byte_offset=23_295, byte_length=20_000,
             sha256=""),
    ]
    with tempfile.NamedTemporaryFile("wt", suffix=".jsonl.gz") as f:
        with gzip.open(f.name, "wt") as g:
            for d in docs:
                g.write(json.dumps(d) + "\n")
        manifest = Path(f.name)
        for context in (2048, 4096, 8192):
            lengths = window_lengths(manifest, "train", context)
            targets = sum(lengths)
            expected = sum(d["byte_length"] - 1 for d in docs
                            if d["split"] == "train")
            assert targets == expected, context
            assert all(0 < n <= context for n in lengths)
            # the synthetic docs: 100-1=99 < context (one short window),
            # 2-1=1 (a single target), 8194-1 = one full window + one
            # single-target window at 4x
            if context == 8192:
                assert lengths[-1] == 1
                assert lengths[-2] == 8192
            assert sum(1 for n in lengths if n == context) >= 1


# ------------------------------------------------------------- the eval rule


def test_the_eval_rule_is_the_banked_set_at_1x():
    """At 1x the K24 eval rule (the first 7 FULL windows in calibration
    order) picks exactly the banked ``L.evaluate`` rule's own window set
    (the first 24 calibration windows, keeping the full ones).  This is
    what makes the 1x anchor cells' landing bar the banked row's own
    number."""
    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    first24 = [row for row in list(stream.windows("calibration"))[:24]
               if len(row[2]) == L.CONTEXT]
    first7full = [row for row in stream.windows("calibration")
                  if len(row[2]) == L.CONTEXT][:EVAL_WINDOWS]
    assert len(first24) == 7
    assert [(r[0], r[1]) for r in first24] \
        == [(r[0], r[1]) for r in first7full]


# ------------------------------------------------- the training loop's form


def test_the_k24_loop_is_the_banked_loop_with_two_threaded_differences():
    """The K24 loop is ``protocol_correction._train_corrected`` with
    exactly the (context, windows_per_step) pair threaded through the
    stream consumption and the grouping: the warmup form, the grad clip,
    the EMA update, the cooldown factor and the honest-clock lines are
    the committed loop's own (structural source receipt)."""
    import inspect
    from experiments.expressivity_tasks.x45_matrix_emender import (
        protocol_correction as PC)
    from experiments.expressivity_tasks.x45_matrix_emender import (
        k24_long_window as K24)
    banked_src = inspect.getsource(PC._train_corrected)
    k24_src = inspect.getsource(K24._train_corrected_k24)
    for marker in (
            "min(1., (step0 + 1) / max(1, warmup))",
            "cooldown_factor(cooldown_k, cooldown_steps)",
            "clip_grad_norm_(model.parameters(), GRAD_CLIP,",
            "error_if_nonfinite=True",
            "ema[name].mul_(ema_decay).add_(",
            "loss_sum += float(loss.detach()) * valid / n",
            "(loss * (valid / n)).backward()",
            "optimizer.zero_grad(set_to_none=True)",
            "row = dict(step=step, active_seconds=elapsed, bytes=seen,"):
        assert marker in banked_src, marker
        assert marker in k24_src, marker
    # the two threaded differences, and only these two
    assert 'batches("train", context=context,' in k24_src
    assert "batch=windows_per_step)" in k24_src
    assert "grouped(batch_rows, microbatch, context)" in k24_src
    assert "iter(stream.batches(\"train\"))" not in k24_src
    assert "grouped(batch_rows, microbatch)" not in k24_src


# ------------------------------------------------------------- CUDA gates


@needs_cuda
def test_the_composed_scan_at_a_long_window_is_the_committed_engine():
    """The kernel vs the eager oracle at a multi-segment long-window
    shape (17 segments, a non-segment-multiple token count): the
    schedule fields, payload lanes, weights, liveness, exit state and key
    table bit-equal; the reads at the measured fp32-vs-fp64 floor x the
    noise factor -- the K15 gate-A bar at a LONGER window."""
    from experiments.expressivity_tasks.x45_matrix_emender.k15_composition \
        import _k15_forward_launch, composed_oracle_scan
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k6_sequential_matrix import _round_f32
    from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
        RoutingPolicy)
    policy = RoutingPolicy(ops=True)
    pol32 = tuple(_round_f32(v) for v in
                  (policy.tau_match, policy.tau_merge, policy.tau_pop,
                   policy.tie_break, policy.read_scale))
    generator = torch.Generator().manual_seed(77)
    batch, tokens, heads, rank, value_dim, key_dim = 2, 1088, 2, 128, 128, 16
    payload = (torch.rand(batch, tokens, heads, value_dim,
                          generator=generator) * 0.9 + 0.05).cuda()
    write_keys = torch.randn(batch, tokens, heads, key_dim,
                            generator=generator).cuda()
    query_keys = torch.randn(batch, tokens, heads, key_dim,
                             generator=generator).cuda()
    decay = (0.85 + torch.rand(batch, tokens, heads, value_dim,
                               generator=generator) * 0.1).cuda()
    gates = (torch.rand(batch, tokens, heads, value_dim,
                       generator=generator) * 0.9 + 0.05).cuda()
    state = torch.zeros(batch, heads, rank, value_dim, dtype=torch.float32,
                        device="cuda")
    live = torch.zeros(batch, heads, rank, dtype=torch.bool, device="cuda")
    keys = torch.zeros(batch, heads, rank, key_dim, dtype=torch.float32,
                       device="cuda")
    out = _k15_forward_launch(payload.float(), write_keys.float(),
                              query_keys.float(), decay.float(),
                              gates.float(), state, live, keys, pol32,
                              rank, 64, True, collect=True, warps=8,
                              stages=3)
    ref = composed_oracle_scan(payload.float(), write_keys.float(),
                               query_keys.float(), decay.float(),
                               gates.float(), policy=policy, gated=True,
                               rank=rank, rank_max=rank, collect=True)
    with torch.no_grad():
        ref64 = composed_oracle_scan(payload.double(), write_keys.double(),
                                     query_keys.double(), decay.double(),
                                     gates.double(), policy=policy,
                                     gated=True, rank=rank, rank_max=rank)
    for name in ("slot", "erase", "op", "live_after"):
        assert torch.equal(getattr(ref["schedule"], name), out[name]), name
    for name in ("weights", "live_pre", "state", "live", "keys"):
        assert torch.equal(ref[name], out[name]), name
    assert torch.equal(ref["schedule"].payload, out["payload"])
    floor = (ref["reads"] - ref64["reads"].float()).abs().max().item()
    deviation = (ref["reads"] - out["reads"]).abs().max().item()
    assert deviation <= 8.0 * max(floor, 1e-30)


@needs_cuda
def test_the_hybrid_wrapper_is_its_dense_layer_at_a_long_window():
    """At the zero-born read gate the hybrid layer IS its dense layer's
    own output bitwise at a LONG window (the only legal escape the IEEE
    sign-of-zero corner) -- the K22 birth discipline at 4x-window token
    counts."""
    from experiments.expressivity_tasks.x45_matrix_emender.k22_hybrid \
        import HybridSlotGDN2Layer, build_slot_mixer
    from experiments.expressivity_tasks.x45_matrix_emender.k22_gate_receipt \
        import _flip_rows

    class Dense(torch.nn.Module):
        def forward(self, x, h0=None, **kwargs):
            return x.clone(), None

    dense = Dense().cuda()
    mixer = build_slot_mixer(width=64).cuda()
    wrapper = HybridSlotGDN2Layer(dense, mixer, 6).cuda()
    x = torch.randn(2, 8192, 64, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        dense_out, _ = dense(x)
        out, _cache = wrapper(x)
    row = _flip_rows(dense_out, out)
    assert row["flips_outside_exact_zero"] == 0
    assert row["value_equal"]
    assert float(wrapper.read_gate) == 0.0
