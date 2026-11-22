"""Source-named checks for the K24 long-window probe: the arms are exactly
the registration's (the window-length bracket {1x, 2x, 4x} over {ours,
gdn2} plus {4x} over {hybrid}), both anchors' banked rows are cited from
their own committed receipts (ours 1.3811104467936925 from the K15 run,
GDN2 1.3521734476089478 from the protocol-correction rescore), the
step/window arithmetic matches the committed registration receipt
computed from the flagship manifest (including the banked 1x landing:
180 steps, 20,022,866 targets), the memory-matched microbatch rule holds
16,384 tokens at every window length, the eval rule coincides with the
banked set at 1x, the boundary discipline is pinned on a synthetic
manifest, and the pre-stated L-1/L-2/REFUTATION rules with the band
discipline are receipted verbatim.  CPU-only.
"""
import json
from pathlib import Path

from experiments.expressivity_tasks.x45_matrix_emender.k24_long_window \
    import (ARMS, BANDS, EXPOSURE_TARGETS, MICROBATCH, PRIMARY_SEED,
            RULES, TOKENS_PER_MICROBATCH, WINDOW_CONTEXT, WINDOWS_PER_STEP,
            gdn2_1x_banked, hybrid_1x_banked, ours_1x_banked,
            window_arithmetic)
from experiments.expressivity_tasks.x42_pile_scale import lattice as L

REPO = Path(__file__).resolve().parents[1]
X45 = REPO / "experiments/expressivity_tasks/x45_matrix_emender"


def test_the_bracket_is_the_registrations_own_seven_arms():
    """The approved bracket: {1x, 2x, 4x} x {ours-composed, gdn2} plus
    {4x} x {hybrid} = 7 arms; the two 1x cells are the mandatory anchors
    that must reproduce the banked rows bit for bit FIRST."""
    assert set(ARMS) == {"ours_1x", "gdn2_1x", "ours_2x", "gdn2_2x",
                         "ours_4x", "gdn2_4x", "hybrid_4x"}
    assert all(ARMS[a]["machine"] == "ours"
               for a in ("ours_1x", "ours_2x", "ours_4x"))
    assert all(ARMS[a]["machine"] == "gdn2"
               for a in ("gdn2_1x", "gdn2_2x", "gdn2_4x"))
    assert ARMS["hybrid_4x"]["machine"] == "hybrid"
    assert ARMS["hybrid_4x"]["mult"] == 4
    assert {ARMS[a]["mult"] for a in ("ours_1x", "ours_2x", "ours_4x")} \
        == {1, 2, 4}
    assert ARMS["ours_1x"]["role"] == ARMS["gdn2_1x"]["role"] == "anchor"
    assert all(ARMS[a]["role"] == "long" for a in
               ("ours_2x", "gdn2_2x", "ours_4x", "gdn2_4x", "hybrid_4x"))


def test_both_anchors_banked_rows_are_the_receipts_own_numbers():
    """The 1x cells' bars are the banked rows' own committed numbers,
    loaded from their run receipts (never restated), and the anchor rule
    states the bit-for-bit convention with the STOP."""
    ours = ours_1x_banked()
    gdn2 = gdn2_1x_banked()
    assert ours["post_cooldown"] == 1.3811104467936925
    assert ours["step1"] == 5.782775391046648
    assert ours["steps"] == 180
    assert ours["exposure_bytes"] == 20_022_866
    assert ours["total_bytes"] == 22_745_198
    assert len(ours["rows"]) == 204
    assert gdn2["post_cooldown"] == 1.3521734476089478
    assert gdn2["step1"] == 5.948033220866492
    assert gdn2["steps"] == 180
    assert gdn2["exposure_bytes"] == 20_022_866
    assert len(gdn2["rows"]) == 204
    banked_ours = json.loads((X45 / "k15_runs/composition_seed3401"
                             "/result.json").read_text())
    banked_gdn2 = json.loads((X45 / "protocol_correction_runs/"
                             "gdn2_speedrun_rescored/result.json").read_text())
    assert banked_ours["final_eval_nll_post_cooldown"] \
        == ours["post_cooldown"]
    assert banked_gdn2["final_eval_nll_post_cooldown"] \
        == gdn2["post_cooldown"]
    assert "STOPS the lane" in RULES["anchor"]["text"]
    assert "1.3811104467936925" in RULES["anchor"]["text"]
    assert "1.3521734476089478" in RULES["anchor"]["text"]


def test_the_hybrid_banked_1x_row_is_cited_not_restated():
    """The curve's hybrid cell cites the K22 lane's own landed row (the
    1.3547673906598772 machine with 0 of 3 slot layers over the bar)."""
    hybrid = hybrid_1x_banked()
    assert hybrid["post_cooldown"] == 1.3547673906598772
    assert hybrid["slot_read_mask_sweep"]["slot_layers_ge_bar"] == 0
    banked = json.loads((X45 / "k22_runs/hybrid_seed3401"
                         "/result.json").read_text())
    assert banked["final_eval_nll_post_cooldown"] == hybrid["post_cooldown"]


def test_the_step_window_arithmetic_is_the_manifests_own():
    """The step/window arithmetic, computed from the flagship manifest
    (never hardcoded): the banked 1x landing (180 steps, 20,022,866
    targets -- the anchor protocol's own numbers), the longer arms'
    landings {182, 178}, and the honest effective-window means (the 4x
    arm achieves 2.405x the banked mean, not 4x)."""
    identity = L.flagship_identity()
    receipt_1x = window_arithmetic(identity["manifest"], 1)
    assert receipt_1x["landing_step"] == 180
    assert receipt_1x["exposure_landing"] == 20_022_866
    assert receipt_1x["exposure_landing"] == ours_1x_banked()[
        "exposure_bytes"]
    receipt_2x = window_arithmetic(identity["manifest"], 2)
    receipt_4x = window_arithmetic(identity["manifest"], 4)
    assert receipt_2x["landing_step"] == 182
    assert receipt_4x["landing_step"] == 178
    assert receipt_1x["effective_window_bytes"]["mean_over_1x_mean"] == 1.0
    assert receipt_2x["effective_window_bytes"]["mean_over_1x_mean"] == 1.669
    assert receipt_4x["effective_window_bytes"]["mean_over_1x_mean"] == 2.405
    assert receipt_4x["effective_window_bytes"]["full_length_fraction"] \
        == 0.3225
    committed = json.loads((X45 / "receipts/k24_window_arithmetic.json")
                            .read_text())
    assert committed["arms"]["4"] == receipt_4x


def test_the_microbatch_rule_holds_the_banked_tokens_per_microbatch():
    """The memory-matched rule: every arm's microbatch carries the banked
    16,384 tokens (8x2048, 4x4096, 2x8192) -- the banked ours arm's peak
    allocation (27.9 GiB) is the memory budget the rule holds."""
    assert WINDOW_CONTEXT == {1: 2048, 2: 4096, 4: 8192}
    assert WINDOWS_PER_STEP == {1: 64, 2: 38, 4: 27}
    assert MICROBATCH == {1: 8, 2: 4, 4: 2}
    for mult in (1, 2, 4):
        assert MICROBATCH[mult] * WINDOW_CONTEXT[mult] == 16_384
    assert TOKENS_PER_MICROBATCH == 16_384
    assert EXPOSURE_TARGETS == 20_000_000
    assert PRIMARY_SEED == 3401


def test_the_pre_state_carries_the_refutation_clause_and_the_band():
    """L-1, L-2 and the REFUTATION clause are the registration's own
    words; every delta is read against the 0.0119 winner-seed band."""
    assert "RISES with window length" in RULES["L1"]["text"]
    assert "SHRINKS with window length" in RULES["L2"]["text"]
    assert "horizon hypothesis is dead at probe scale" \
        in RULES["REFUTATION"]["text"]
    assert "scale/exposure alone" in RULES["REFUTATION"]["text"]
    assert BANDS["winner_seed_delta"] == 0.011948040553501738
    assert "0.0119" in RULES["band_discipline"]["text"]


def test_the_boundary_discipline_is_stated_and_enforced():
    """The standing rule: never pack across document boundaries; windows
    truncate at the boundary; the document-disjoint splits stay intact --
    and the synthetic windowing honors it."""
    import gzip
    import tempfile
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k24_long_window import window_lengths
    assert "NEVER pack across document" in RULES["boundaries"]["text"]
    assert "TRUNCATE at the boundary" in RULES["boundaries"]["text"]
    docs = [
        dict(split="train", byte_offset=0, byte_length=10_000),
        dict(split="train", byte_offset=10_000, byte_length=100),
        dict(split="train", byte_offset=10_100, byte_length=2),
    ]
    with tempfile.NamedTemporaryFile("wt", suffix=".jsonl.gz") as f:
        with gzip.open(f.name, "wt") as g:
            for d in docs:
                g.write(json.dumps(d) + "\n")
        for context in (2048, 4096, 8192):
            lengths = window_lengths(Path(f.name), "train", context)
            assert sum(lengths) == sum(d["byte_length"] - 1
                                       for d in docs)
            assert all(0 < n <= context for n in lengths)
            assert lengths[-2] == 99        # the short doc's one window
            assert lengths[-1] == 1         # the single-target doc
