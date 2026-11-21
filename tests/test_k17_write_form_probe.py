"""Source-named checks for the K17 write-form probe on the composed machine:
the arms are exactly the registration's (the dense write, the frozen-write
falsifier control, the read-gain bracket {1.0, 2.0, 4.0}, the conditional
combined arm), the composed machine's cited baseline rows are the K15
assemble receipt's own numbers (never re-run), the machine's ACTUAL
inherited read gain is the tuned 2.787 point (the registration's factual
correction -- the "2.0" of the K13 registration text is the RoutingPolicy
class default the banked build overrides), the parameter accounting is the
registration's own, the pre-stated rules are receipted verbatim, and the
dense channel's routing lands on the AdamW side.  CPU-only.
"""
import json
from pathlib import Path

from experiments.expressivity_tasks.x45_matrix_emender.k17_probe_write_form \
    import (ARMS, BANDS, GAIN_ARMS, GDN2_POST_COOLDOWN,
            INHERITED_LR_MULTIPLIER, INHERITED_READ_SCALE, RULES, WIN_BAR,
            composed_4x_banked)

REPO = Path(__file__).resolve().parents[1]
X45 = REPO / "experiments" / "expressivity_tasks" / "x45_matrix_emender"


def test_the_arms_are_the_registrations_own():
    """The approved arms: the dense gated write, the mandatory frozen-write
    falsifier control, the read-gain bracket {1.0, 2.0, 4.0} (the inherited
    2.787 cell is NOT an arm -- its rows are the K15 composition's own,
    cited), and the conditional combined arm."""
    assert set(ARMS) == {"dense", "dense_frozen", "gain1x", "gain2x",
                         "gain4x", "combined"}
    assert ARMS["dense"]["kind"] == "dense"
    assert not ARMS["dense"].get("frozen", False)
    assert ARMS["dense_frozen"]["frozen"] is True
    assert GAIN_ARMS == {"gain1x": 1.0, "gain2x": 2.0, "gain4x": 4.0}
    for label in ("gain1x", "gain2x", "gain4x"):
        assert ARMS[label]["kind"] == "gain"
        assert ARMS[label]["read_scale"] == GAIN_ARMS[label]
    assert ARMS["combined"]["kind"] == "dense"


def test_the_inherited_read_gain_is_the_tuned_point_not_the_class_default():
    """The registration's factual correction, pinned: the composed machine's
    ACTUAL inherited read_scale is the tuned softmax-machine point
    2.7870458737509636 (optimizer_bakeoff.TUNED_WINNER -- every banked
    build passes it to velocity_ladder._build_k10); the "2.0" of the K13
    registration text is the RoutingPolicy CLASS DEFAULT, which the banked
    build overrides."""
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import TUNED_WINNER
    from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
        RoutingPolicy)
    assert INHERITED_READ_SCALE == TUNED_WINNER["read_scale"] \
        == 2.7870458737509636
    assert RoutingPolicy().read_scale == 2.0, \
        "the class default is 2.0 -- the folklore the correction names"
    assert INHERITED_READ_SCALE not in GAIN_ARMS.values(), \
        "the inherited point is NOT a bracket cell: its rows are the K15 " \
        "composition's own at both seeds, cited never re-run"


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
    """The dense channel is the b_proj form: one bias-free Linear(width,
    heads*value_dim) + one per-head zero-born scalar per mixer, ON TOP of
    the composed machine's own p3 stack (the committed k15 accounting)."""
    from experiments.expressivity_tasks.x45_matrix_emender.k15_composition \
        import composed_parameter_delta
    from experiments.expressivity_tasks.x45_matrix_emender.k17_dense_write \
        import LANES, dense_parameter_delta
    assert LANES == 3
    assert dense_parameter_delta(1, 64, 2, 32) \
        - composed_parameter_delta(1, 64, 2, 32) == 64 * 2 * 32 + 2
    assert dense_parameter_delta(24, 576, 2, 128) \
        - composed_parameter_delta(24, 576, 2, 128) == 3_538_992
    assert dense_parameter_delta(24, 576, 2, 128) == 12_196_032
    # the machine totals: the banked 149,981,808 + p3's 8,657,040 + the
    # dense channel's 3,538,992
    assert 149_981_808 + 8_657_040 + 3_538_992 == 162_177_840


def test_the_prestated_rules_are_receipted_verbatim():
    """The W-1/W-2/W-3/G-1 clauses, the conditional combined arm, the
    standing 0.01 win-bar second-seed rule, the band discipline and the
    falsifier's own bar are pre-stated in the registration and receipted
    in every arm."""
    assert WIN_BAR == 0.01
    assert INHERITED_LR_MULTIPLIER == 4
    for key in ("W1", "W2", "W3", "G1", "combined", "second_seed",
                "band_discipline", "falsifier"):
        assert key in RULES and RULES[key]["text"]
    assert RULES["W1"]["bar"] == WIN_BAR
    assert RULES["second_seed"]["bar"] == WIN_BAR
    assert str(GDN2_POST_COOLDOWN) in RULES["band_discipline"]["text"]
    assert BANDS["winner_seed_delta"] == 0.011948040553501738   # "0.0119"
    assert BANDS["endpoint_variance"] == 0.075


def test_the_dense_channel_routes_to_the_adamw_side():
    """The routing inventory is closed: the p3 gate tensors AND the dense
    channel's tensors land on the hybrid's AdamW side (the DECAY gate's own
    convention); a frozen control's dense tensors are excluded by
    requires_grad and the split equals the composed machine's own."""
    import torch
    from experiments.expressivity_tasks.x45_matrix_emender.k11b_probe_channel_gates \
        import gate_tensor_names
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k17_probe_write_form import _dense_tensor_names, split_parameters_k17
    from experiments.expressivity_tasks.x45_matrix_emender.k17_dense_write \
        import (DenseWriteComposedMixer, convert_model_to_dense_write)
    from experiments.expressivity_tasks.x45_matrix_emender.k15_composition \
        import convert_model_to_composition
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        MatrixMixer, MatrixShape)
    from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
        RoutingPolicy)

    assert _dense_tensor_names() == ("d_proj.weight", "dense_scalar")
    assert "dense_scalar" not in gate_tensor_names("p3")

    block = torch.nn.Module()
    block.add_module("mixer", MatrixMixer(MatrixShape(
        width=64, depth=1, hidden=128, heads=2, value_dim=32,
        rank=16, rank_max=16), RoutingPolicy(ops=True), 64))
    stub = torch.nn.Module()
    stub.add_module("blocks", torch.nn.ModuleList([block]))
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k10_separated_keys import convert_model_to_separated_keys
    model = convert_model_to_dense_write(
        convert_model_to_composition(convert_model_to_separated_keys(stub)))
    assert isinstance(model.blocks[0].mixer, DenseWriteComposedMixer)
    muon, adamw, muon_names, adamw_names = split_parameters_k17(model, False)
    assert any(n.endswith("d_proj.weight") for n in adamw_names)
    assert any(n.endswith("dense_scalar") for n in adamw_names)
    assert not any("d_proj" in n or "dense_scalar" in n for n in muon_names)
    # the frozen control: the dense tensors drop out of the inventory
    for block in model.blocks:
        block.mixer.d_proj.weight.requires_grad_(False)
        block.mixer.dense_scalar.requires_grad_(False)
    _m, _a, _mn, frozen_adamw = split_parameters_k17(model, True)
    assert not any("d_proj" in n or "dense_scalar" in n
                   for n in frozen_adamw)
