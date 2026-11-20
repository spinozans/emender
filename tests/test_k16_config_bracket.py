"""Source-named checks for the K16 config bracket on the composed machine:
the bracket's LR points are exact multiples of the tuned AdamW base, the
inherited 4x cell's cited rows are the K15 assemble receipt's own numbers
(never re-run), the machine and the mechanism instrument are the K15 lane's
own imports (never restated), the pre-stated second-seed rule keys on the
receipted anchors, and the assemble applies the rule exactly as stated.
CPU-only.
"""
import json
from pathlib import Path

from experiments.expressivity_tasks.x45_matrix_emender.k16_config_bracket \
    import (ARMS, BANDS, COMPOSED_4X_FALLBACK, GDN2_POST_COOLDOWN,
            INHERITED_LR_MULTIPLIER, SECOND_SEED_RULE, TUNED_ADAMW_LR,
            composed_4x_banked)

REPO = Path(__file__).resolve().parents[1]
X45 = REPO / "experiments" / "expressivity_tasks" / "x45_matrix_emender"


def test_the_bracket_is_three_lr_cells_plus_the_family_check():
    """The approved bracket: {2x, 3x, 6x} around the INHERITED 4x point
    (cited from the K15 arm's own landed rows, never re-run -- 4x is not an
    arm), plus the plain-AdamW family check at 1x."""
    assert set(ARMS) == {"muon2x", "muon3x", "muon6x", "adamw"}
    assert "muon4x" not in ARMS, \
        "the inherited 4x point is the K15 arm's own banked rows -- cited, " \
        "never re-run"
    assert {a["lr_multiplier"] for a in ARMS.values()} == {1, 2, 3, 6}
    assert INHERITED_LR_MULTIPLIER == 4
    assert ARMS["muon2x"]["family"] == "hybrid-muon"
    assert ARMS["muon3x"]["family"] == "hybrid-muon"
    assert ARMS["muon6x"]["family"] == "hybrid-muon"
    assert ARMS["adamw"]["family"] == "adamw"
    assert ARMS["adamw"]["lr_multiplier"] == 1


def test_the_bracket_lr_points_are_exact_multiples_of_the_tuned_base():
    """2x/3x/6x of the machine's own tuned AdamW LR (3.3427141575371284e-4),
    the prior bracket lanes' own axis, and the adamw arm AT the tuned point."""
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import TUNED_WINNER
    assert TUNED_ADAMW_LR == TUNED_WINNER["lr"] == 3.3427141575371284e-4
    assert 2 * TUNED_ADAMW_LR == 6.685428315074257e-4
    assert 3 * TUNED_ADAMW_LR == 1.0028142472611385e-3
    assert 6 * TUNED_ADAMW_LR == 2.005628494522277e-3
    for label, arm in ARMS.items():
        assert arm["lr_multiplier"] * TUNED_ADAMW_LR \
            == arm["lr_multiplier"] * TUNED_WINNER["lr"], label


def test_the_inherited_4x_cited_rows_are_the_k15_receipts_own_numbers():
    """The cited 4x cell must be the K15 composition arm's OWN landed rows
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
        # the fallback constants are pinned to the same receipt numbers
        assert COMPOSED_4X_FALLBACK["post_cooldown"][seed] \
            == row["post_cooldown"]


def test_the_machine_and_instrument_are_the_k15_lanes_own_imports():
    """The composed machine is the K15 composition VERBATIM (imported, never
    restated): the conversion and the parameter accounting are k15's own,
    the mechanism instrument is the K15 probe's own function object."""
    from experiments.expressivity_tasks.x45_matrix_emender import (
        k15_composition, k15_probe_composition)
    from experiments.expressivity_tasks.x45_matrix_emender \
        import k16_config_bracket
    assert k16_config_bracket.k15_probe_diagnostics \
        is k15_probe_composition.k15_probe_diagnostics
    assert k16_config_bracket.K13_ADDITIVE \
        is k15_probe_composition.K13_ADDITIVE
    assert k16_config_bracket.BANKED is k15_probe_composition.BANKED
    # the parameter delta is the committed accounting, not a restatement
    # (the registered flagship shape: width 576, heads 2, value_dim 128,
    # depth 24 -- the K15 receipt's own 8,657,040)
    assert k15_composition.composed_parameter_delta(24, 576, 2, 128) \
        == 8_657_040


def test_the_prestated_rules_key_on_the_receipted_anchors():
    """The second-seed rule keys on the bracket's own convention (best cell
    at 3401 across the run cells and the cited 4x row) and the bands are the
    standing amendments' own numbers; the GDN2 contact row is the banked gap
    reference."""
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k15_probe_composition import BANKED
    assert SECOND_SEED_RULE["bracket_seed"] == 3401
    assert SECOND_SEED_RULE["second_seed"] == 3402
    assert SECOND_SEED_RULE["cited_cell"].startswith("muon4x")
    assert GDN2_POST_COOLDOWN == BANKED["gdn2"]["post_cooldown"] \
        == 1.3521734476089478
    assert BANDS["endpoint_variance"] == 0.075
    assert BANDS["winner_seed_delta"] == 0.011948040553501738


def test_the_assemble_applies_the_second_seed_rule_as_stated(tmp_path,
                                                          monkeypatch):
    """The assemble's adjudication is the rule applied exactly as stated:
    a non-4x best cell owes its second seed; the 4x cell's pair is already
    banked; and a cell at/past GDN2's row raises the both-scorings-both-seeds
    escalation flag."""
    from experiments.expressivity_tasks.x45_matrix_emender \
        import k16_config_bracket as K16

    runs = tmp_path / "k16_runs"
    runs.mkdir()

    def land(label, seed, post, constant=1.44, ema=1.43):
        out = runs / f"{label}_seed{seed}"
        out.mkdir()
        (out / "result.json").write_text(json.dumps(dict(
            arm=label, seed=seed, family=ARMS[label]["family"],
            lr_multiplier=ARMS[label]["lr_multiplier"], peak_lr=1e-3,
            status="completed", final_eval_nll_post_cooldown=post,
            final_eval_nll_constant_lr=constant, ema_eval_nll=ema,
            step1_train_loss=5.78, step1_receipt=dict(k13_parity=True),
            train_seconds_constant=1900.0, parameters=158_638_848,
            diagnostics_post_cooldown=None,
            muon_update_rms=None)))

    # a bracket where 6x wins at 3401: the second seed is OWED
    land("muon2x", 3401, 1.395)
    land("muon3x", 3401, 1.390)
    land("muon6x", 3401, 1.375)
    land("adamw", 3401, 1.410)
    monkeypatch.setattr(K16, "RUNS", runs)
    summary = K16.assemble(type("A", (), {"out": str(tmp_path / "s.json")})())
    assert summary["best_cell_at_seed_3401"] == "muon6x"
    assert summary["second_seed_adjudication"]["second_seed_owed"] is True
    assert summary["second_seed_adjudication"][
        "second_seed_already_banked"] is False
    assert summary["search_symmetric_ours_best"]["cell"] == "muon6x"
    assert summary["gap_quote_vs_gdn2"]["remaining_gap"] \
        == 1.375 - GDN2_POST_COOLDOWN

    # the 6x second seed lands: the pair is complete, no run owed
    land("muon6x", 3402, 1.380)
    summary = K16.assemble(type("A", (), {"out": str(tmp_path / "s2.json")})())
    assert summary["second_seed_adjudication"]["second_seed_owed"] is False
    assert summary["second_seed_adjudication"]["second_seed_already_run"] \
        is True

    # a bracket where the inherited 4x row stays best: no run owed (banked)
    runs2 = tmp_path / "k16_runs_4x_best"
    runs2.mkdir()
    for label, post in (("muon2x", 1.395), ("muon3x", 1.390),
                         ("muon6x", 1.385), ("adamw", 1.410)):
        out = runs2 / f"{label}_seed3401"
        out.mkdir()
        (out / "result.json").write_text(json.dumps(dict(
            arm=label, seed=3401, family=ARMS[label]["family"],
            lr_multiplier=ARMS[label]["lr_multiplier"], peak_lr=1e-3,
            status="completed", final_eval_nll_post_cooldown=post,
            final_eval_nll_constant_lr=1.44, ema_eval_nll=1.43,
            step1_train_loss=5.78, step1_receipt=dict(k13_parity=True),
            train_seconds_constant=1900.0, parameters=158_638_848,
            diagnostics_post_cooldown=None, muon_update_rms=None)))
    monkeypatch.setattr(K16, "RUNS", runs2)
    summary = K16.assemble(type("A", (), {"out": str(tmp_path / "s3.json")})())
    assert summary["best_cell_at_seed_3401"] == "muon4x"
    assert summary["second_seed_adjudication"]["second_seed_owed"] is False
    assert summary["second_seed_adjudication"]["second_seed_already_banked"] \
        is True
    assert summary["search_symmetric_ours_best"]["cell"] == "muon4x"

    # the adamw arm winning inverts the optimizer-family ranking: flagged
    runs3 = tmp_path / "k16_runs_family"
    runs3.mkdir()
    for label, post in (("muon2x", 1.395), ("muon3x", 1.390),
                         ("muon6x", 1.398), ("adamw", 1.370)):
        out = runs3 / f"{label}_seed3401"
        out.mkdir()
        (out / "result.json").write_text(json.dumps(dict(
            arm=label, seed=3401, family=ARMS[label]["family"],
            lr_multiplier=ARMS[label]["lr_multiplier"], peak_lr=1e-3,
            status="completed", final_eval_nll_post_cooldown=post,
            final_eval_nll_constant_lr=1.44, ema_eval_nll=1.43,
            step1_train_loss=5.78, step1_receipt=dict(k13_parity=True),
            train_seconds_constant=1900.0, parameters=158_638_848,
            diagnostics_post_cooldown=None, muon_update_rms=None)))
    monkeypatch.setattr(K16, "RUNS", runs3)
    summary = K16.assemble(type("A", (), {"out": str(tmp_path / "s4.json")})())
    assert summary["optimizer_family_inversion"] is True
    assert any("OPTIMIZER-FAMILY RANKING INVERTED" in f
               for f in summary["escalation_flags"])

    # a cell at/past GDN2's row raises the band-discipline escalation flag
    runs4 = tmp_path / "k16_runs_contact"
    runs4.mkdir()
    for label, post in (("muon2x", 1.395), ("muon3x", 1.390),
                         ("muon6x", 1.350), ("adamw", 1.410)):
        out = runs4 / f"{label}_seed3401"
        out.mkdir()
        (out / "result.json").write_text(json.dumps(dict(
            arm=label, seed=3401, family=ARMS[label]["family"],
            lr_multiplier=ARMS[label]["lr_multiplier"], peak_lr=1e-3,
            status="completed", final_eval_nll_post_cooldown=post,
            final_eval_nll_constant_lr=1.44, ema_eval_nll=1.43,
            step1_train_loss=5.78, step1_receipt=dict(k13_parity=True),
            train_seconds_constant=1900.0, parameters=158_638_848,
            diagnostics_post_cooldown=None, muon_update_rms=None)))
    monkeypatch.setattr(K16, "RUNS", runs4)
    summary = K16.assemble(type("A", (), {"out": str(tmp_path / "s5.json")})())
    assert summary["search_symmetric_ours_best"]["cell"] == "muon6x"
    assert any("at/past GDN2's" in f for f in summary["escalation_flags"])
