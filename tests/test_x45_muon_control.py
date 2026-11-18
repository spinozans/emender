"""Source-named checks for the x45 symmetric Muon control: the GDN2 parameter
routing (options A and B, the enumerated inventory, the fail-closed guards),
the conventions held at GDN2's banked AdamW row, and the verdict rule's
thresholds receipted against the banked artifacts.  CPU-only (the real
inventory is enumerated on the meta device).
"""
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from experiments.expressivity_tasks.x45_matrix_emender.muon_control import (
    ARMS, GDN2_BANKED_ADAMW, GDN2_BANKED_FINAL, GDN2_BANKED_STEP1_LOSS,
    VERDICT_RULE, WINNER_MUON3X_BANKED_FINAL, split_parameters_gdn2)

REPO = Path(__file__).resolve().parents[1]
X45 = REPO / "experiments/expressivity_tasks/x45_matrix_emender"


class TinyGDN2Core(nn.Module):
    """The external GatedDeltaNet2 core's own parameter names (not its
    arithmetic): projections, gate projections, the decay-gate parameters,
    the depthwise short convs, o_norm."""

    def __init__(self, d=8, heads=2, key_dim=6, value_dim=8):
        super().__init__()
        self.q_proj = nn.Linear(d, key_dim, bias=False)
        self.k_proj = nn.Linear(d, key_dim, bias=False)
        self.v_proj = nn.Linear(d, value_dim, bias=False)
        self.q_conv1d = nn.Conv1d(key_dim, key_dim, 4, groups=key_dim,
                                  bias=False)
        self.k_conv1d = nn.Conv1d(key_dim, key_dim, 4, groups=key_dim,
                                  bias=False)
        self.v_conv1d = nn.Conv1d(value_dim, value_dim, 4, groups=value_dim,
                                  bias=False)
        self.f_proj = nn.Sequential(nn.Linear(d, 4, bias=False),
                                    nn.Linear(4, key_dim, bias=False))
        self.b_proj = nn.Linear(d, key_dim, bias=False)
        self.w_proj = nn.Linear(d, value_dim, bias=False)
        self.A_log = nn.Parameter(torch.zeros(heads))
        self.dt_bias = nn.Parameter(torch.zeros(key_dim))
        self.g_proj = nn.Sequential(nn.Linear(d, 4, bias=False),
                                    nn.Linear(4, value_dim, bias=True))
        self.o_norm = nn.RMSNorm(4, eps=1e-5)
        self.o_proj = nn.Linear(value_dim, d, bias=False)


class _CoreWrapper(nn.Module):
    """GDN2ExternalLayer's own nesting (block.layer.gdn2.gdn2.<param>)."""

    def __init__(self, core):
        super().__init__()
        self.gdn2 = core


class TinyGDN2Layer(nn.Module):
    """GDN2ExternalMLPLayer's own names: gdn2, norm_2, the SwiGLU mlp."""

    def __init__(self, core, d=8, hidden=12):
        super().__init__()
        self.gdn2 = _CoreWrapper(core)
        self.norm_2 = nn.RMSNorm(d, eps=1e-5)
        self.mlp = nn.ModuleDict({
            name: nn.Linear(d if name != "w3" else hidden,
                            hidden if name != "w3" else d, bias=False)
            for name in ("w1", "w2", "w3")})


class TinyGDNBlock(nn.Module):
    def __init__(self, core, d=8):
        super().__init__()
        self.norm = nn.RMSNorm(d, eps=1e-5)
        self.layer = TinyGDN2Layer(core, d)


class TinyGDNModel(nn.Module):
    def __init__(self, d=8, vocab=16, depth=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.norm = nn.RMSNorm(d, eps=1e-5)
        self.head = nn.Linear(d, vocab, bias=False)
        self.blocks = nn.ModuleList(
            [TinyGDNBlock(TinyGDN2Core(d), d) for _ in range(depth)])


class TinyBaseline(nn.Module):
    """matched_arms' Baseline wrapper: the GDNModel hangs off ``self.model``,
    so every parameter name carries the ``model.`` prefix."""

    def __init__(self):
        super().__init__()
        self.model = TinyGDNModel()


def _tiny():
    torch.manual_seed(0)
    return TinyBaseline().eval()


def _names(depth):
    muon = {"model.embed.weight", "model.head.weight"}
    gates = set()
    for i in range(depth):
        muon |= {f"model.blocks.{i}.layer.gdn2.gdn2.{p}.weight"
                 for p in ("q_proj", "k_proj", "v_proj", "o_proj",
                           "b_proj", "w_proj")}
        muon |= {f"model.blocks.{i}.layer.gdn2.gdn2.f_proj.{j}.weight"
                 for j in (0, 1)}
        muon |= {f"model.blocks.{i}.layer.gdn2.gdn2.g_proj.{j}.weight"
                 for j in (0, 1)}
        muon |= {f"model.blocks.{i}.layer.mlp.{w}.weight"
                 for w in ("w1", "w2", "w3")}
        gates |= {f"model.blocks.{i}.layer.gdn2.gdn2.{p}.weight"
                  for p in ("b_proj", "w_proj")}
        gates |= {f"model.blocks.{i}.layer.gdn2.gdn2.{f}.{j}.weight"
                  for f in ("f_proj", "g_proj") for j in (0, 1)}
    adamw = {"model.norm.weight"}
    for i in range(depth):
        adamw |= {f"model.blocks.{i}.norm.weight",
                  f"model.blocks.{i}.layer.norm_2.weight",
                  f"model.blocks.{i}.layer.gdn2.gdn2.o_norm.weight",
                  f"model.blocks.{i}.layer.gdn2.gdn2.A_log",
                  f"model.blocks.{i}.layer.gdn2.gdn2.dt_bias",
                  f"model.blocks.{i}.layer.gdn2.gdn2.g_proj.1.bias"}
        adamw |= {f"model.blocks.{i}.layer.gdn2.gdn2.{c}_conv1d.weight"
                  for c in ("q", "k", "v")}
    return muon, adamw, gates


def test_option_a_routing_is_every_2d_hidden_layer_matrix():
    """Option A (the approved control): Muon owns every 2D matrix of GDN2's
    inventory -- the gate-PRODUCING projections included; AdamW owns exactly
    the non-matrix params (norms, A_log/dt_bias, the bias, the 3D convs)."""
    model = _tiny()
    _m, _a, muon, adamw = split_parameters_gdn2(model)
    expected_muon, expected_adamw, _ = _names(len(model.model.blocks))
    assert set(muon) == expected_muon
    assert set(adamw) == expected_adamw
    params = dict(model.named_parameters())
    for name in adamw:                 # nothing 2D stays on the AdamW side
        assert params[name].ndim != 2, name
    for name in muon:
        assert params[name].ndim == 2, name


def test_option_b_routing_mirrors_the_k10_decay_gate_exclusion():
    """The informative appendix: the gate-producing projections move to the
    AdamW group; everything else is option A's split."""
    model = _tiny()
    _m, _a, muon, adamw = split_parameters_gdn2(model, gates_on_adamw=True)
    expected_muon, expected_adamw, gates = _names(len(model.model.blocks))
    assert set(muon) == expected_muon - gates
    assert set(adamw) == expected_adamw | gates


def test_routing_fails_closed_on_drift_and_guards():
    model = _tiny()
    # an unclassified name (inventory drift) must fail loudly
    model.model.extra = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="unclassified GDN2 parameter"):
        split_parameters_gdn2(model)
    del model.model.extra
    # a non-2D parameter under a Muon-owned name must be rejected
    model.model.head.weight = nn.Parameter(torch.randn(2, 3, 4))
    with pytest.raises(ValueError, match="not a full 2D matrix"):
        split_parameters_gdn2(model)
    # distinct parameters sharing storage must be rejected (the prior line's
    # own guard)
    model = _tiny()
    model.model.head.weight = nn.Parameter(model.model.embed.weight.data)
    with pytest.raises(ValueError, match="share storage"):
        split_parameters_gdn2(model)


def test_control_conventions_hold_gdn2s_banked_row():
    """The control's knobs must be GDN2's own banked AdamW row (the searched
    nomination), receipted against the side-by-side armed run's start.json,
    and the Muon LR must be 3x that LR (the bake-off's own convention)."""
    start = json.loads((X45 / "side_by_side_runs/gdn2_armed_seed3401"
                        "/start.json").read_text())
    banked = start["optimizer"]
    assert GDN2_BANKED_ADAMW["lr"] == banked["lr"]
    assert GDN2_BANKED_ADAMW["betas"] == tuple(banked["betas"])
    assert GDN2_BANKED_ADAMW["weight_decay"] == banked["weight_decay"]
    assert GDN2_BANKED_ADAMW["warmup_steps"] == banked["warmup_steps"]
    assert GDN2_BANKED_ADAMW["microbatch"] == banked["microbatch"]
    for label, arm in ARMS.items():
        assert arm["lr_multiplier"] == 3, label
        assert arm["family"] == "hybrid-muon", label
    assert ARMS["gdn2_muon3x"]["gates_on_adamw"] is False
    assert ARMS["gdn2_muon3x_gates_adamw"]["gates_on_adamw"] is True


def test_verdict_thresholds_are_receipted_against_the_banked_artifacts():
    """The verdict rule's thresholds must be the banked artifacts' own
    numbers, not restated constants: the winner's muon3x final from the
    bake-off summary, GDN2's AdamW final and its step-1 train loss from the
    banked armed/probe runs."""
    bakeoff = json.loads((X45 / "receipts/optimizer_bakeoff_summary.json")
                         .read_text())
    assert WINNER_MUON3X_BANKED_FINAL == \
        bakeoff["arms"]["muon3x"]["final_eval_nll"]
    result = json.loads((X45 / "side_by_side_runs/gdn2_armed_seed3401"
                         "/result.json").read_text())
    assert GDN2_BANKED_FINAL == result["final_eval_nll"]
    assert GDN2_BANKED_FINAL == result["banked_final_nll"]
    first = json.loads((X45 / "velocity_runs/gdn2_probe_seed3401"
                        "/training.jsonl").read_text().splitlines()[0])
    assert GDN2_BANKED_STEP1_LOSS == first["loss"]
    assert VERDICT_RULE["banked_margin"] == pytest.approx(0.0691, abs=1e-4)


def test_the_real_gdn2_inventory_classifies_completely():
    """The banked build's own inventory (meta device, the matched-arms build
    path's module tree): every parameter classifies under both routings, and
    the group sizes are the receipted counts (Option A: 353 Muon tensors /
    149,682,176 params; the AdamW side is the non-matrix residue)."""
    from unittest.mock import patch

    from experiments.expressivity_tasks.x40_emender_lm import (
        part8_surface, scale_surface)
    with patch.object(scale_surface, "VOCAB", 256), \
            patch.object(part8_surface, "VOCAB", 256), \
            torch.device("meta"):
        part8_surface.gdn_config.cache_clear()
        model = part8_surface.GDNModel("150M", 150_003_984)
    total = sum(p.numel() for p in model.parameters())
    assert total == 149_855_670                      # the banked row's count
    _m, _a, muon, adamw = split_parameters_gdn2(model)
    params = dict(model.named_parameters())
    assert len(muon) == 353
    assert sum(params[n].numel() for n in muon) == 149_682_176
    assert len(muon) + len(adamw) == len(params)
    assert all(params[n].ndim == 2 for n in muon)
    assert all(params[n].ndim != 2 for n in adamw)
    _m, _a, muon_b, _ = split_parameters_gdn2(model, gates_on_adamw=True)
    assert len(muon_b) == 191
    assert sum(params[n].numel() for n in muon_b) == 122_255_360
