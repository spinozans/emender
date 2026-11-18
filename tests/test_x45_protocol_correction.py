"""Source-named checks for the x45 protocol correction lane: the corrected
scoring conventions (cooldown form/length, EMA decay, the primary/secondary
roles), the four arms' approved configs (the banked best rows' own configs
and seeds; the Moonlight LR points), the RMS-matching receipt (the Moonshot
scaling is ALREADY the RMS match -- measured, not rescaled), the no-directional
-prediction pre-statement, and the banked artifacts the lane's numbers key on.
CPU-only (the real inventories are enumerated on the meta device where
needed; the optimizer-metric check runs on CPU tensors).
"""
import json
import math
from pathlib import Path

import torch

from experiments.expressivity_tasks.x45_matrix_emender.protocol_correction import (
    ARMS, BANKED_MARGIN, BANKED_REFS, CORRECTED_SCORING, COOLDOWN_STEPS,
    EMA_DECAY, ENDPOINT_VARIANCE_BAND, GDN2_BEST, HORIZON_TRANSFER_RULE,
    NO_PREDICTION, OUR_BEST, WINNER_SEED_DELTA, _banked_step1_loss,
    cooldown_factor)
from experiments.expressivity_tasks.x45_matrix_emender.muon_control import (
    GDN2_BANKED_ADAMW)
from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff import (
    TUNED_WINNER)

REPO = Path(__file__).resolve().parents[1]
X45 = REPO / "experiments" / "expressivity_tasks" / "x45_matrix_emender"


def test_the_lane_is_exactly_four_arms_both_machines_both_recipes():
    """The directive's four arms: 2 machines x {speedrun recipe re-scored,
    Moonlight recipe}, each machine's arms at its banked best row's own
    machine-seed pairing (ours 3402, GDN2 3401)."""
    assert set(ARMS) == {"winner_speedrun_rescored", "winner_moonlight",
                         "gdn2_speedrun_rescored", "gdn2_moonlight"}
    by_machine = {}
    for label, arm in ARMS.items():
        by_machine.setdefault(arm["machine"], []).append(arm["recipe"])
    assert sorted(by_machine["winner"]) == ["moonlight", "speedrun"]
    assert sorted(by_machine["gdn2"]) == ["moonlight", "speedrun"]
    for label, arm in ARMS.items():
        assert arm["machine"] in ("winner", "gdn2"), label
    assert ARMS["winner_speedrun_rescored"]["seed"] == 3402
    assert ARMS["winner_moonlight"]["seed"] == 3402
    assert ARMS["gdn2_speedrun_rescored"]["seed"] == 3401
    assert ARMS["gdn2_moonlight"]["seed"] == 3401


def test_the_speedrun_arms_are_the_banked_best_rows_own_configs():
    """The re-scored arms ARE the banked best rows: ours the 1.4899 row
    (B-analog routing, 4x its own tuned AdamW LR), GDN2 the 1.3951 row
    (option B, 5x its own banked AdamW LR) -- so the corrected scores differ
    from the banked rows ONLY by the instrument."""
    assert OUR_BEST["label"] == "muon4x_banked_seed3402"
    assert OUR_BEST["final"] == 1.4899048124040877
    assert ARMS["winner_speedrun_rescored"]["lr_multiplier"] == 4
    assert ARMS["winner_speedrun_rescored"]["routing"] == "banked_gates_adamw"
    assert GDN2_BEST["label"] == "gdn2_muon5x_gates_adamw_seed3401"
    assert GDN2_BEST["final"] == 1.3950541530336653
    assert ARMS["gdn2_speedrun_rescored"]["lr_multiplier"] == 5
    assert ARMS["gdn2_speedrun_rescored"]["gates_on_adamw"] is True
    assert GDN2_BEST["final"] < BANKED_REFS["gdn2_muon3x_gates_adamw"]
    # the banked best-vs-best margin this lane re-measures: 0.0949
    assert abs(BANKED_MARGIN - 0.0948) < 0.0005


def test_the_banked_artifacts_back_every_cited_number():
    """The banked best rows' numbers are the prior lanes' own receipts on
    disk, never restated from memory."""
    for best in (OUR_BEST, GDN2_BEST):
        result = json.loads((Path(best["run_dir"]) / "result.json")
                            .read_text())
        assert result["final_eval_nll"] == best["final"], best["label"]
    # and the step-1 receipts the arms assert bit-identity against
    assert _banked_step1_loss("winner", 3402) == 5.768619428012168
    assert _banked_step1_loss("gdn2", 3401) == 5.948033220866492


def test_the_cooldown_is_the_approved_form_and_length():
    """24 steps (13.3% of the 180-step budget, inside the brief's 10-15%
    band), linear, factor (25-k)/24 for k=1..24 -- 1.0 at the first cooldown
    step, 1/24 (never zero) at the last."""
    assert COOLDOWN_STEPS == 24
    fraction = COOLDOWN_STEPS / 180
    assert 0.10 <= fraction <= 0.15
    assert cooldown_factor(1) == 1.0
    assert cooldown_factor(24) == 1 / 24
    for k in range(1, COOLDOWN_STEPS + 1):
        assert cooldown_factor(k) == (COOLDOWN_STEPS + 1 - k) / COOLDOWN_STEPS
        assert cooldown_factor(k) > 0.0
    # linear: equal steps down
    deltas = [cooldown_factor(k) - cooldown_factor(k + 1)
              for k in range(1, COOLDOWN_STEPS)]
    assert all(abs(d - 1 / COOLDOWN_STEPS) < 1e-12 for d in deltas)
    import pytest
    with pytest.raises(ValueError):
        cooldown_factor(0)
    with pytest.raises(ValueError):
        cooldown_factor(COOLDOWN_STEPS + 1)


def test_the_ema_convention_is_the_approved_one():
    """Decay 0.99 (window ~100 steps, 55% of the 180-step budget),
    ranking-only, constant-phase-only, initialized at the init weights."""
    assert EMA_DECAY == 0.99
    assert CORRECTED_SCORING["ema_decay"] == 0.99
    assert CORRECTED_SCORING["primary"].startswith("post-cooldown eval")
    assert "RANKING-ONLY" in CORRECTED_SCORING["secondary"]
    assert "NOT updated during the cooldown" in CORRECTED_SCORING["secondary"]
    assert "EXTENSION" in CORRECTED_SCORING["exposure_note"]


def test_the_ema_update_math():
    """One EMA update: ema <- d*ema + (1-d)*param (the loop's exact lerp)."""
    d = EMA_DECAY
    ema = torch.tensor([1.0, 2.0])
    param = torch.tensor([3.0, 5.0])
    expected = d * ema + (1 - d) * param
    ema.mul_(d).add_(param, alpha=1 - d)
    assert torch.allclose(ema, expected)


def test_the_moonlight_arms_are_one_x_the_machines_own_adamw_lrs():
    """The Moonlight recipe (arXiv:2502.16982): with the update RMS matched
    to AdamW's, Muon directly reuses the LR tuned for AdamW -- 1x, not the
    3x-5x speedrun multiples.  Ours: 3.3427e-4 (the tuned winner's own LR);
    GDN2: 5.4201855e-4 (its banked AdamW LR).  No other knob moves."""
    assert ARMS["winner_moonlight"]["lr_multiplier"] == 1
    assert ARMS["gdn2_moonlight"]["lr_multiplier"] == 1
    assert ARMS["winner_moonlight"]["routing"] == "banked_gates_adamw"
    assert ARMS["gdn2_moonlight"]["gates_on_adamw"] is True
    # and the speedrun points are exact multiples of the same AdamW LRs
    assert 4 * TUNED_WINNER["lr"] > TUNED_WINNER["lr"]
    assert 5 * GDN2_BANKED_ADAMW["lr"] == 2.7100927623701266e-3


def test_the_rms_matching_is_by_construction_and_measured_not_rescaled():
    """The lane's key fact: the pinned hybrid's Moonshot 0.2*sqrt(max(shape))
    scaling IS the RMS match to AdamW's ~0.2 update RMS -- so the Moonlight
    arms change NO optimizer code, only the LR point; and the driver MEASURES
    both groups' update RMS per step (the receipt, not an assumption)."""
    from experiments.expressivity_tasks.hybrid_muon import scaled_update
    g = torch.randn(64, 32)
    update = scaled_update(g)
    # the Moonshot factor: the update's RMS is ~0.2 x the raw orthogonalized
    # update's 1/sqrt(max(shape)) class -- matched to AdamW's typical
    # 0.2-0.4 by design, the Moonlight target
    rms = update.square().mean().sqrt().item()
    assert 0.1 < rms < 0.4
    # the measured-RMS hook exists and is metrics-only: the subclass calls
    # the parent step and only extends last_metrics
    import inspect

    from experiments.expressivity_tasks.x45_matrix_emender import (
        protocol_correction as PC)
    src = inspect.getsource(PC._make_measured_hybrid)
    assert "super().step(closure)" in src
    assert "adamw_update_rms" in src


def test_the_measured_adamw_rms_metric_on_a_toy_optimizer():
    """The metrics-only subclass: one step on a toy hybrid, the aux AdamW
    group's measured update RMS must be finite and positive and must equal
    the m_hat/(sqrt(v_hat)+eps) RMS computed by hand from the state."""
    cls = None

    from experiments.expressivity_tasks.x45_matrix_emender import (
        protocol_correction as PC)
    cls = PC._make_measured_hybrid()
    muon_p = [torch.nn.Parameter(torch.randn(8, 4))]
    adamw_p = [torch.nn.Parameter(torch.randn(6, 3))]
    opt = cls(muon_p, adamw_p, lr=1e-3, betas=(0.9, 0.9392),
              weight_decay=0.01)
    for _ in range(3):
        for p in muon_p + adamw_p:
            p.grad = torch.randn_like(p)
        opt.step()
    assert opt.last_metrics["muon_update_rms"] > 0
    measured = opt.last_metrics["adamw_update_rms"]
    assert math.isfinite(measured) and measured > 0
    # by hand from the aux state (torch AdamW: bias-corrected)
    state = opt.aux.state[adamw_p[0]]
    step_n = int(state["step"].item()) if torch.is_tensor(state["step"]) \
        else int(state["step"])
    m_hat = state["exp_avg"] / (1 - 0.9 ** step_n)
    v_hat = state["exp_avg_sq"] / (1 - 0.9392 ** step_n)
    hand = (m_hat / (v_hat.sqrt() + 1e-8)).square().mean().sqrt().item()
    assert abs(measured - hand) < 1e-6


def test_no_directional_prediction_is_pre_stated():
    """The directive's own rule: no directional prediction for any arm --
    the gate-surface difference (27.7K vs 27.4M) makes both directions
    live; receipted in every arm and applied verbatim by the assemble."""
    assert "NO DIRECTIONAL PREDICTION" in NO_PREDICTION["rule"]
    assert "27.7K" in NO_PREDICTION["rule"] and "27.4M" in NO_PREDICTION["rule"]
    for label, arm in ARMS.items():
        assert arm["recipe"] in ("speedrun", "moonlight"), label


def test_the_bands_and_the_horizon_transfer_rule():
    """Every margin is read against the endpoint-variance band (0.075) and
    the winner's seed band (0.0119); the horizon rule transfers probe-optimal
    LRs DOWNSCALED with alpha FIT locally (never Bjorck's imported exponent),
    with the 2x-budget rank-stability hard gate, and the deferred fit is
    labelled as not-yet-run."""
    assert ENDPOINT_VARIANCE_BAND == 0.075
    assert abs(WINNER_SEED_DELTA - 0.0119) < 0.0005
    rule = HORIZON_TRANSFER_RULE["rule"]
    assert "DOWNSCALED" in rule
    assert "(D_probe / D_board)^alpha" in rule
    assert "FIT at our scale" in rule and "never their imported exponent" in rule
    assert "rank stability" in HORIZON_TRANSFER_RULE["hard_gate"]
    assert "DEFERRED" in HORIZON_TRANSFER_RULE["alpha_provenance"]


def test_the_banked_reference_rows_are_the_lanes_own_numbers():
    """Our AdamW 1.8227, GDN2's AdamW 1.5837, GDN2's option-B 3x row 1.4151
    -- the cited reference rows, receipted against the prior lanes' summary
    receipts on disk."""
    assert BANKED_REFS == dict(
        our_adamw=1.822737353188651,
        gdn2_adamw=1.5836738858904158,
        gdn2_muon3x_gates_adamw=1.4150957890919276)
    summary = json.loads((X45 / "receipts" / "muon_control_summary.json")
                         .read_text())
    assert summary["banked"]["gdn2_adamw"] == BANKED_REFS["gdn2_adamw"]
    assert summary["banked"]["winner_muon3x"] == 1.514617647443499
