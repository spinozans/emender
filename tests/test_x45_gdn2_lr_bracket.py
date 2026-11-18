"""Source-named checks for the x45 GDN2 LR bracket: the bracket's LR points
are exact multiples of GDN2's banked AdamW LR, the arms run the control's own
option-B routing (imported verbatim, never restated), the pre-stated
second-seed rule's thresholds are receipted against the banked artifacts, and
the symmetric best-vs-best inputs are the prior lanes' receipts' own numbers.
CPU-only (the real inventory is enumerated on the meta device).
"""
import json
from pathlib import Path

import torch

from experiments.expressivity_tasks.x45_matrix_emender.gdn2_lr_bracket import (
    ARMS, ENDPOINT_VARIANCE_BAND, GDN2_MUON3X_GATES_ADAMW_BANKED_FINAL,
    GDN2_MUON3X_MAX_BANKED_FINAL, OUR_BEST_FINAL, OUR_BEST_LABEL,
    SECOND_SEED_RULE, WINNER_SEED_DELTA, _sibling_step1_loss,
    split_parameters_gdn2)
from experiments.expressivity_tasks.x45_matrix_emender.muon_control import (
    GDN2_BANKED_ADAMW, GDN2_BANKED_STEP1_LOSS)

REPO = Path(__file__).resolve().parents[1]
X45 = REPO / "experiments" / "expressivity_tasks" / "x45_matrix_emender"


def test_the_bracket_is_exactly_two_upward_arms_at_option_b():
    """The approved bracket: 4x and 5x only (upward, the shape our machine's
    bracket took), both at GDN2's best routing (option B, gates-on-AdamW),
    both hybrid-Muon family."""
    assert set(ARMS) == {"gdn2_muon4x_gates_adamw", "gdn2_muon5x_gates_adamw"}
    for label, arm in ARMS.items():
        assert arm["family"] == "hybrid-muon", label
        assert arm["gates_on_adamw"] is True, label
        assert arm["lr_multiplier"] in (4, 5), label


def test_the_bracket_lr_points_are_exact_multiples_of_the_banked_lr():
    """4x = 2.1680742e-3 and 5x = 2.7100927e-3 -- exact multiples of GDN2's
    banked AdamW LR (5.4201855e-4), the bracket's own axis and nothing else
    moved."""
    lr = GDN2_BANKED_ADAMW["lr"]
    assert lr == 5.420185524740253e-4
    assert 4 * lr == 2.1680742098961013e-3
    assert 5 * lr == 2.7100927623701266e-3
    assert ARMS["gdn2_muon4x_gates_adamw"]["lr_multiplier"] * lr \
        == 4 * lr
    assert ARMS["gdn2_muon5x_gates_adamw"]["lr_multiplier"] * lr \
        == 5 * lr


def test_the_routing_is_the_controls_own_option_b_imported_verbatim():
    """The bracket arms' routing string is the control's banked option-B arm's
    own routing (imported, not restated), and the split function is the
    control's own -- GDN2's model code and routing inventory untouched."""
    from experiments.expressivity_tasks.x45_matrix_emender import (
        gdn2_lr_bracket, muon_control)
    assert gdn2_lr_bracket.split_parameters_gdn2 \
        is muon_control.split_parameters_gdn2
    banked_routing = muon_control.ARMS["gdn2_muon3x_gates_adamw"]["routing"]
    for label, arm in ARMS.items():
        assert arm["routing"] == banked_routing, label
    # and the option-B routing is unchanged on the real inventory (the
    # control's own receipted counts)
    from unittest.mock import patch

    from experiments.expressivity_tasks.x40_emender_lm import (
        part8_surface, scale_surface)
    with patch.object(scale_surface, "VOCAB", 256), \
            patch.object(part8_surface, "VOCAB", 256), \
            torch.device("meta"):
        part8_surface.gdn_config.cache_clear()
        model = part8_surface.GDNModel("150M", 150_003_984)
    _m, _a, muon, adamw = split_parameters_gdn2(model, gates_on_adamw=True)
    params = dict(model.named_parameters())
    assert len(muon) == 191
    assert sum(params[n].numel() for n in muon) == 122_255_360
    assert all(params[n].ndim == 2 for n in muon)


def test_the_second_seed_rule_thresholds_are_the_banked_artifacts_numbers():
    """The pre-stated second-seed rule must key on the banked option-B final
    (the control's own receipt) and the winner's measured seed band (the
    routing-symmetry assemble's own number), not restated constants."""
    control = json.loads((X45 / "receipts/muon_control_summary.json")
                         .read_text())
    assert GDN2_MUON3X_GATES_ADAMW_BANKED_FINAL == \
        control["finals"]["gdn2_muon3x_gates_adamw"]["final_eval_nll"]
    assert GDN2_MUON3X_MAX_BANKED_FINAL == \
        control["finals"]["gdn2_muon3x"]["final_eval_nll"]
    routing = json.loads((X45 / "receipts/routing_symmetry_summary.json")
                         .read_text())
    assert WINNER_SEED_DELTA == routing["symmetric_best_vs_best"][
        "winner_seed_band"]
    assert OUR_BEST_FINAL == routing["symmetric_best_vs_best"]["our_best"][
        "final"]
    assert OUR_BEST_LABEL == routing["symmetric_best_vs_best"]["our_best"][
        "label"]
    assert ENDPOINT_VARIANCE_BAND == \
        routing["symmetric_best_vs_best"]["endpoint_variance_band"]
    assert SECOND_SEED_RULE["reference"] == GDN2_MUON3X_GATES_ADAMW_BANKED_FINAL
    assert SECOND_SEED_RULE["seed_band"] == WINNER_SEED_DELTA


def test_the_step1_receipt_keys_on_the_banked_probe_row_at_seed_3401():
    """Seed 3401's step-1 reference is the banked gdn2 probe's own first row
    (the optimizer cannot touch the first forward); other seeds have no
    banked GDN2 row, so the reference is the first same-seed bracket arm's
    own first row -- receipted against that arm's training.jsonl, never a
    restated constant."""
    assert _sibling_step1_loss(3401, "gdn2_muon4x_gates_adamw") \
        == GDN2_BANKED_STEP1_LOSS == 5.948033220866492
    # the 5x arm's seed-3402 row has landed: it is the same-seed reference
    first3402 = json.loads(
        (X45 / "gdn2_lr_bracket_runs/gdn2_muon5x_gates_adamw_seed3402"
         "/training.jsonl").read_text().splitlines()[0])["loss"]
    assert _sibling_step1_loss(3402, "gdn2_muon4x_gates_adamw") == first3402
    # and the arm that IS the reference excludes itself honestly
    assert _sibling_step1_loss(3402, "gdn2_muon5x_gates_adamw") is None
