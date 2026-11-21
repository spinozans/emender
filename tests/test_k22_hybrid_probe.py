"""Source-named checks for the K22 hybrid probe: the arms are exactly the
registration's (the live hybrid + the mandatory zero-read frozen
falsifier), both parents' banked rows are cited from their own committed
receipts (GDN2 1.3522 from the protocol-correction rescore; ours
1.3811/1.3832 from the K15 assemble receipt), the parameter accounting is
purely additive and computed not hardcoded, the pre-stated H-1/H-2/
REFUTATION rules and the falsifier's non-negotiable bar are receipted
verbatim, the recipe is the banked GDN2 row's own (5x, option B), and the
slot branch routes per the K18 convention mirrored.  CPU-only.
"""
import json
from pathlib import Path

from experiments.expressivity_tasks.x45_matrix_emender.k22_probe_hybrid \
    import (ARMS, BANDS, GDN2_CONSTANT_LR, GDN2_EMA180, GDN2_MACHINE_PARAMETERS,
            GDN2_POST_COOLDOWN, PEAK_LR, RULES, WIN_BAR, composed_4x_banked,
            evaluate_clauses, gdn2_banked_step1, gdn2_banked_trajectory)

REPO = Path(__file__).resolve().parents[1]
X45 = REPO / "experiments" / "expressivity_tasks" / "x45_matrix_emender"


def test_the_arms_are_the_registrations_own():
    """The approved arms: the live hybrid (born zero-read, the gates
    learn to open) and the mandatory frozen falsifier; no third arm."""
    assert set(ARMS) == {"hybrid", "hybrid_frozen"}
    assert ARMS["hybrid"] == dict(ARMS["hybrid"], kind="hybrid", frozen=False)
    assert ARMS["hybrid_frozen"] == dict(ARMS["hybrid_frozen"],
                                         kind="hybrid", frozen=True)
    assert "ZERO-BORN read gate" in ARMS["hybrid"]["note"]
    assert "FALSIFIER" in ARMS["hybrid_frozen"]["note"]
    assert "bit-identical" in ARMS["hybrid_frozen"]["note"]


def test_both_parents_banked_rows_are_the_receipts_own_numbers():
    """GDN2's cited row is the protocol-correction rescore's own numbers
    (loaded from the banked run's committed result.json, never restated);
    ours are the K15 assemble receipt's own."""
    banked = json.loads((X45 / "protocol_correction_runs/"
                         "gdn2_speedrun_rescored/result.json").read_text())
    assert GDN2_POST_COOLDOWN == 1.3521734476089478
    assert banked["final_eval_nll_post_cooldown"] == GDN2_POST_COOLDOWN
    assert banked["seed"] == 3401
    assert GDN2_CONSTANT_LR == 1.3950541530336653
    assert GDN2_EMA180 == 1.3803383282252721
    assert GDN2_MACHINE_PARAMETERS == banked["parameters"] == 149_855_670
    composed = composed_4x_banked()
    assert composed["post_cooldown"][3401] == 1.3811104467936925
    assert composed["post_cooldown"][3402] == 1.383232252938407


def test_the_falsifiers_reference_is_the_banked_trajectory():
    """The falsifier's reference is the banked GDN2 row's OWN training log
    (204 rows at seed 3401: the constant phase plus the cooldown), and its
    step-1 row is the protocol-identical start the arms must reproduce bit
    for bit."""
    trajectory = gdn2_banked_trajectory(3401)
    assert len(trajectory) == 204
    assert [r["step"] for r in trajectory] == list(range(1, 205))
    assert trajectory[179]["phase"] == "constant"
    assert trajectory[180]["phase"] == "cooldown"
    assert gdn2_banked_step1(3401) == 5.948033220866492


def test_the_recipe_is_the_banked_gdn2_rows_own():
    """The optimizer recipe is the banked 1.3522 row's own VERBATIM: the
    5x speedrun point over the searched nomination's AdamW knobs, option
    B routing, the probe's own grad clip."""
    from experiments.expressivity_tasks.x45_matrix_emender.muon_control \
        import GDN2_BANKED_ADAMW
    start = json.loads((X45 / "protocol_correction_runs/"
                        "gdn2_speedrun_rescored/start.json").read_text())
    assert start["optimizer"]["peak_lr"] == PEAK_LR
    assert PEAK_LR == 5 * GDN2_BANKED_ADAMW["lr"]
    assert start["optimizer"]["lr_multiplier"] == 5
    assert tuple(start["optimizer"]["betas"]) == tuple(
        GDN2_BANKED_ADAMW["betas"])
    for key in ("weight_decay", "warmup_steps", "microbatch"):
        assert start["optimizer"][key] == GDN2_BANKED_ADAMW[key]


def test_the_parameter_accounting_is_purely_additive():
    """The hybrid's delta is the registration's computed accounting: 3
    composed slot mixers at the host width 640 + 3 read gate scalars,
    nothing removed (the dense mixers stay -- the falsifier's own
    construction)."""
    from experiments.expressivity_tasks.x45_matrix_emender.k22_hybrid \
        import (SLOT_LAYERS, hybrid_slot_parameter_delta,
                slot_mixer_parameter_count)
    assert slot_mixer_parameter_count(640) == 887_304
    assert hybrid_slot_parameter_delta(640) == 2_661_915
    assert GDN2_MACHINE_PARAMETERS + hybrid_slot_parameter_delta(640) \
        == 152_517_585
    assert len(SLOT_LAYERS) == 3


def test_the_prestated_rules_are_receipted_verbatim():
    """H-1 (the bankable-win bar vs GDN2), H-2 (vs the composed machine's
    own row), the REFUTATION clause, the standing second-seed rule, the
    band discipline, the falsifier's non-negotiable bar and the birth
    parity rule are pre-stated in the registration and receipted in every
    arm."""
    assert WIN_BAR == 0.01
    for key in ("H1", "H2", "REFUTATION", "second_seed",
                "band_discipline", "falsifier", "birth_parity"):
        assert key in RULES and RULES[key]["text"]
    assert RULES["H1"]["bar"] == WIN_BAR
    assert RULES["H1"]["reference"] == GDN2_POST_COOLDOWN
    assert str(GDN2_POST_COOLDOWN) in RULES["band_discipline"]["text"]
    assert "non-negotiable" in RULES["falsifier"]["text"]
    assert "1.3811104467936925" in RULES["H2"]["text"]
    assert BANDS["winner_seed_delta"] == 0.011948040553501738
    assert BANDS["endpoint_variance"] == 0.075


def test_the_clause_read_is_the_prestated_rule():
    """evaluate_clauses applies the pre-stated rules per seed exactly as
    stated: a >= 0.01 improvement vs GDN2 is a BANKABLE win (the second
    seed owed); a landing >= GDN2's row is the REFUTATION clause; a
    sub-band delta gets the parity phrasing; the frozen falsifier carries
    its own bar."""
    def receipt(arm, post, **extra):
        return dict(arm=arm, seed=3401,
                   final_eval_nll_post_cooldown=post, **extra)

    out = evaluate_clauses(receipt("hybrid", GDN2_POST_COOLDOWN - 0.012))
    assert out["clause_at_this_seed"].startswith("H-1_WIN")
    assert out["second_seed_owed"] is True
    assert out["gdn2_contact_flag"] is True

    out = evaluate_clauses(receipt("hybrid", GDN2_POST_COOLDOWN - 0.004))
    assert out["clause_at_this_seed"].startswith("H-1_PARITY")
    assert out["second_seed_owed"] is False

    out = evaluate_clauses(receipt("hybrid", GDN2_POST_COOLDOWN + 0.02))
    assert out["clause_at_this_seed"].startswith("REFUTED")
    assert out["gdn2_contact_flag"] is False

    # the win bar's edge: 0.011 >= 0.01 is a bankable win (no band-worship)
    out = evaluate_clauses(receipt("hybrid", GDN2_POST_COOLDOWN - 0.011))
    assert out["clause_at_this_seed"].startswith("H-1_WIN")
    assert out["second_seed_owed"] is True

    # H-2 read against the composed machine's own same-seed row
    out = evaluate_clauses(receipt("hybrid", 1.3700))
    assert out["H2_held"] is True
    out = evaluate_clauses(receipt("hybrid", 1.3900))
    assert out["H2_held"] is False

    # the falsifier's own bar
    out = evaluate_clauses(receipt(
        "hybrid_frozen", GDN2_POST_COOLDOWN,
        falsifier_divergence=dict(steps_compared=204,
                                  bit_identical_steps=204,
                                  max_abs_loss_delta=0.0),
        falsifier_landing_bit_identical=True))
    assert out["clause_at_this_seed"] == "FALSIFIER_PASSED"
    out = evaluate_clauses(receipt(
        "hybrid_frozen", GDN2_POST_COOLDOWN + 0.001,
        falsifier_divergence=dict(steps_compared=204,
                                  bit_identical_steps=203,
                                  max_abs_loss_delta=1e-9),
        falsifier_landing_bit_identical=False))
    assert out["clause_at_this_seed"].startswith("FALSIFIER_DIVERGED")
