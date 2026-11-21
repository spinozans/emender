"""Source-named checks for the K18 short-conv probe on the composed machine:
the arms are exactly the registration's (the conv transplant born identity,
the mandatory frozen-conv falsifier control, the fixed-shift control), the
composed machine's cited baseline rows are the K15 assemble receipt's own
numbers (never re-run), the parameter accounting is the registration's own
(the composed machine's own + one depthwise 256x4 kernel per mixer, 24
mixers), the pre-stated THREE-OUTCOME rule and the falsifier/shift clauses
are receipted verbatim, and the live conv's kernel lands on the AdamW side
(the GDN2 convention mirrored).  CPU-only.
"""
import json
from pathlib import Path

from experiments.expressivity_tasks.x45_matrix_emender.k18_probe_short_conv \
    import (ARMS, BANDS, CONV_MACHINE_PARAMETERS, CONV_PARAMETER_DELTA,
            COMPOSED_PARAMETERS, GDN2_POST_COOLDOWN,
            INHERITED_LR_MULTIPLIER, RULES, WIN_BAR, composed_4x_banked,
            evaluate_clauses)

REPO = Path(__file__).resolve().parents[1]
X45 = REPO / "experiments" / "expressivity_tasks" / "x45_matrix_emender"


def test_the_arms_are_the_registrations_own():
    """The approved arms: the conv transplant (born identity, learns), the
    mandatory frozen-conv falsifier control, and the fixed-shift control
    (a frozen non-identity kernel, never learned, no birth parity)."""
    assert set(ARMS) == {"conv", "conv_frozen", "shift_fixed"}
    assert ARMS["conv"] == dict(ARMS["conv"], kind="conv", birth="identity",
                                frozen=False)
    assert ARMS["conv_frozen"] == dict(ARMS["conv_frozen"], kind="conv",
                                       birth="identity", frozen=True)
    assert ARMS["shift_fixed"] == dict(ARMS["shift_fixed"], kind="shift",
                                       birth="shift", frozen=True)
    assert ARMS["conv"]["note"].find("[0,0,0,1]") >= 0, \
        "the identity birth must be quoted in the registered tap layout"
    assert "identity birth" in ARMS["conv_frozen"]["note"]
    assert "LEARNS to deviate" in ARMS["conv"]["note"]
    assert ARMS["shift_fixed"]["note"].find("[0,0,0.5,0.5]") >= 0


def test_the_conv_births_are_the_registered_kernels():
    """The engine's births registry is the registration's: the identity
    kernel [0,0,0,1] and the fixed shift [0,0,0.5,0.5], both in the
    STANDARD nn.Conv1d layout (the LAST tap is the current token -- the
    escalated correction of the task's [1,0,0,0] spec, approved by the
    supervisor)."""
    from experiments.expressivity_tasks.x45_matrix_emender.k18_short_conv \
        import (IDENTITY_SHORT_CONV_BIRTH, SHIFT_SHORT_CONV_BIRTH,
                SHORT_CONV_BIRTHS, SHORT_CONV_KERNEL_SIZE)
    assert SHORT_CONV_KERNEL_SIZE == 4
    assert IDENTITY_SHORT_CONV_BIRTH == (0.0, 0.0, 0.0, 1.0)
    assert SHIFT_SHORT_CONV_BIRTH == (0.0, 0.0, 0.5, 0.5)
    assert SHORT_CONV_BIRTHS == {"identity": IDENTITY_SHORT_CONV_BIRTH,
                                 "shift": SHIFT_SHORT_CONV_BIRTH}


def test_the_composed_baseline_rows_are_the_k15_receipts_own_numbers():
    """The cited baseline must be the K15 composition arm's OWN landed rows
    (the assemble receipt's numbers, both seeds), not restated constants."""
    summary = json.loads((X45 / "receipts/k15_composition_summary.json")
                         .read_text())
    banked = composed_4x_banked()
    for seed in (3401, 3402):
        row = summary["arms"][f"composition_seed{seed}"]
        assert banked["post_cooldown"][seed] == row["post_cooldown"]
        assert banked["constant_lr"][seed] == row["constant_lr"]
        assert banked["ema_180"][seed] == row["ema_180"]
        assert banked["step1_train_loss"][seed] == row["step1_train_loss"]
    assert banked["post_cooldown"][3401] == 1.3811104467936925
    assert banked["post_cooldown"][3402] == 1.383232252938407
    assert GDN2_POST_COOLDOWN == 1.3521734476089478


def test_the_parameter_accounting_is_the_registrations_own():
    """The conv is GDN2's own v_conv1d shape: one depthwise kernel of
    heads*value_dim = 256 channels x 4 taps per mixer, 24 mixers -- 24,576
    on top of the composed machine's own 158,638,848 (0.015% of the
    machine, quoted in every receipt)."""
    from experiments.expressivity_tasks.x45_matrix_emender.k18_short_conv \
        import short_conv_parameter_delta
    from experiments.expressivity_tasks.x45_matrix_emender.k15_composition \
        import composed_parameter_delta
    assert CONV_PARAMETER_DELTA == 24 * 256 * 4 == 24_576
    assert COMPOSED_PARAMETERS == 158_638_848
    assert CONV_MACHINE_PARAMETERS == 158_663_424
    assert short_conv_parameter_delta(24, 576, 2, 128) \
        == composed_parameter_delta(24, 576, 2, 128) + CONV_PARAMETER_DELTA


def test_the_prestated_rules_are_receipted_verbatim():
    """The THREE-OUTCOME rule (SC-1 local-features / SC-2 parity / SC-3
    hurts), the shift control's SH-1 decomposition clause, the standing
    0.01 win-bar second-seed rule, the band discipline and the falsifier's
    own bar are pre-stated in the registration and receipted in every
    arm."""
    assert WIN_BAR == 0.01
    assert INHERITED_LR_MULTIPLIER == 4
    for key in ("SC1", "SC2", "SC3", "SH1", "second_seed",
                "band_discipline", "falsifier"):
        assert key in RULES and RULES[key]["text"]
    assert RULES["SC1"]["bar"] == WIN_BAR
    assert RULES["second_seed"]["bar"] == WIN_BAR
    assert str(GDN2_POST_COOLDOWN) in RULES["band_discipline"]["text"]
    assert BANDS["winner_seed_delta"] == 0.011948040553501738   # "0.0119"
    assert BANDS["endpoint_variance"] == 0.075
    # the three-outcome rule's own pre-state: the parity clause names the
    # heterogeneous published effects (the small-end possibility, stated
    # BEFORE the result), and the hurts clause names the slot-semantics
    # reading (a machine-family difference, not a failure)
    assert "heterogeneous" in RULES["SC2"]["text"]
    assert "machine-family difference" in RULES["SC3"]["text"]


def test_the_clause_read_is_the_three_outcome_rule():
    """evaluate_clauses applies the registered three-outcome rule per seed
    exactly as stated: >= 0.01 improvement is SC-1 (the second seed owed),
    inside the band is SC-2, worse beyond the band is SC-3; the falsifier
    and the shift control carry their own bars."""
    base = composed_4x_banked()["post_cooldown"][3401]

    def receipt(arm, post):
        return dict(arm=arm, seed=3401, final_eval_nll_post_cooldown=post)

    out = evaluate_clauses(receipt("conv", base - 0.012))
    assert out["clause_at_this_seed"].startswith("SC-1_at_this_seed")
    assert out["improvement_vs_same_seed_baseline"] >= WIN_BAR

    out = evaluate_clauses(receipt("conv", base - 0.004))
    assert out["clause_at_this_seed"].startswith("SC-2_at_this_seed")

    out = evaluate_clauses(receipt("conv", base + 0.02))
    assert out["clause_at_this_seed"].startswith("SC-3_at_this_seed")

    out = evaluate_clauses(receipt("conv_frozen", base))
    assert "FALSIFIER" in out["clause_at_this_seed"]

    out = evaluate_clauses(receipt("shift_fixed", base - 0.006))
    assert "FIXED-SHIFT" in out["clause_at_this_seed"]
    assert out["gdn2_contact_flag"] is False
    # GDN2 contact: the flag fires at or past the reference row
    out = evaluate_clauses(receipt("conv", GDN2_POST_COOLDOWN - 1e-9))
    assert out["gdn2_contact_flag"] is True


def test_the_conv_kernel_routes_to_the_adamw_side():
    """The routing inventory is closed: the p3 gate tensors AND the live
    conv's kernel land on the hybrid's AdamW side (the GDN2 convention
    mirrored -- a depthwise conv kernel is not a hidden-map matrix); a
    frozen control's kernel is excluded by requires_grad and the split
    equals the composed machine's own."""
    import torch
    from experiments.expressivity_tasks.x45_matrix_emender.k11b_probe_channel_gates \
        import gate_tensor_names
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k18_probe_short_conv import _conv_tensor_names, split_parameters_k18
    from experiments.expressivity_tasks.x45_matrix_emender.k18_short_conv \
        import ShortConvComposedMixer, convert_model_to_short_conv
    from experiments.expressivity_tasks.x45_matrix_emender.k15_composition \
        import convert_model_to_composition
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        MatrixMixer, MatrixShape)
    from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
        RoutingPolicy)

    assert _conv_tensor_names() == ("v_conv.weight",)
    assert "v_conv.weight" not in gate_tensor_names("p3")

    block = torch.nn.Module()
    block.add_module("mixer", MatrixMixer(MatrixShape(
        width=64, depth=1, hidden=128, heads=2, value_dim=32,
        rank=16, rank_max=16), RoutingPolicy(ops=True), 64))
    stub = torch.nn.Module()
    stub.add_module("blocks", torch.nn.ModuleList([block]))
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k10_separated_keys import convert_model_to_separated_keys
    model = convert_model_to_short_conv(
        convert_model_to_composition(convert_model_to_separated_keys(stub)))
    assert isinstance(model.blocks[0].mixer, ShortConvComposedMixer)
    muon, adamw, muon_names, adamw_names = split_parameters_k18(model, False)
    assert any(n.endswith("v_conv.weight") for n in adamw_names)
    assert not any("v_conv" in n for n in muon_names)
    # the frozen controls: the kernel drops out of the inventory
    for block in model.blocks:
        block.mixer.v_conv.weight.requires_grad_(False)
    _m, _a, _mn, frozen_adamw = split_parameters_k18(model, True)
    assert not any("v_conv" in n for n in frozen_adamw)
