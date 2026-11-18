"""Source-named checks for the x45 symmetric routing extension: the winner's
parameter routings (the approved M1 mapping, the banked-B-analog identity, the
max-Muon A-analog, the broad-mapping appendix), the fail-closed guards, the
conventions held at the banked rows, and the pre-stated prediction thresholds
receipted against the banked artifacts.  CPU-only (the real inventory is
enumerated on the meta device).
"""
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from experiments.expressivity_tasks.x45_matrix_emender.routing_symmetry import (
    ARMS, BANKED, CLAIM_RULE, PREDICTIONS, ROUTING_SPREAD_BAND,
    WINNER_SEED_DELTA, _banked_step1_loss, split_parameters_winner)

REPO = Path(__file__).resolve().parents[1]
X45 = REPO / "experiments/expressivity_tasks/x45_matrix_emender"


class TinySeparatedMixer(nn.Module):
    """The converted winner's own mixer names (SeparatedSequentialMixer):
    value/key/query_key/out plus the DECAY pair."""

    def __init__(self, d=8, heads=2, value_dim=8, key_dim=4, ctx=4):
        super().__init__()
        self.value = nn.Linear(d, heads * value_dim, bias=False)
        self.key = nn.Linear(ctx * d, heads * key_dim, bias=False)
        self.query_key = nn.Linear(ctx * d, heads * key_dim, bias=False)
        self.out = nn.Linear(heads * value_dim, d, bias=False)
        self.decay = nn.Linear(d, heads, bias=True)


class TinyBlock(nn.Module):
    def __init__(self, d=8, hidden=12):
        super().__init__()
        self.norm1 = nn.RMSNorm(d, eps=1e-5)
        self.mixer = TinySeparatedMixer(d)
        self.norm2 = nn.RMSNorm(d, eps=1e-5)
        self.mlp = nn.ModuleDict({
            name: nn.Linear(d if name != "w3" else hidden,
                            hidden if name != "w3" else d, bias=False)
            for name in ("w1", "w2", "w3")})


class TinyWinner(nn.Module):
    """MatrixModel's own names: embed, the final norm, head, the blocks."""

    def __init__(self, d=8, vocab=16, depth=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.norm = nn.RMSNorm(d, eps=1e-5)
        self.head = nn.Linear(d, vocab, bias=False)
        self.blocks = nn.ModuleList([TinyBlock(d) for _ in range(depth)])


def _tiny():
    torch.manual_seed(0)
    return TinyWinner().eval()


def _expected(depth):
    """(per-matrix muon set, decay pair, address producers, norms)."""
    matrices = {"embed.weight", "head.weight"}
    addresses, decay, norms = set(), set(), {"norm.weight"}
    for i in range(depth):
        b = f"blocks.{i}"
        matrices |= {f"{b}.mixer.{p}.weight"
                     for p in ("value", "key", "query_key", "out")}
        matrices |= {f"{b}.mlp.{w}.weight" for w in ("w1", "w2", "w3")}
        addresses |= {f"{b}.mixer.{p}.weight" for p in ("key", "query_key")}
        decay |= {f"{b}.mixer.decay.weight", f"{b}.mixer.decay.bias"}
        norms |= {f"{b}.norm1.weight", f"{b}.norm2.weight"}
    return matrices, decay, addresses, norms


def test_banked_routing_is_the_bakeoffs_own_split():
    """The approved M1 mapping's consequence, proven in code: the
    banked_gates_adamw routing reproduces bakeoff_optimizers.split_parameters
    EXACTLY -- the banked bake-off arm already ran our B-analog, so its rows
    are cited, not re-run (a re-run would be bit-identical by construction)."""
    from experiments.expressivity_tasks.x45_matrix_emender import (
        bakeoff_optimizers as BO)
    model = _tiny()
    _m, _a, muon, adamw = split_parameters_winner(model, "banked_gates_adamw")
    _m, _a, bo_muon, bo_adamw = BO.split_parameters(model)
    assert set(muon) == set(bo_muon)
    assert set(adamw) == set(bo_adamw)
    matrices, decay, _addresses, norms = _expected(len(model.blocks))
    assert set(muon) == matrices
    assert set(adamw) == decay | norms


def test_max_muon_routing_puts_every_2d_matrix_under_muon():
    """The A-analog: Muon on every 2D matrix (the banked routing plus
    decay.weight); AdamW on the true non-matrix residue ONLY (the 1D norms +
    decay.bias) -- nothing 2D stays on the AdamW side."""
    model = _tiny()
    _m, _a, muon, adamw = split_parameters_winner(model, "max_muon")
    matrices, decay, _addresses, norms = _expected(len(model.blocks))
    params = dict(model.named_parameters())
    assert set(muon) == matrices | {n for n in decay
                                    if n.endswith("weight")}
    assert set(adamw) == {n for n in decay if n.endswith("bias")} | norms
    assert all(params[n].ndim == 2 for n in muon)
    assert all(params[n].ndim != 2 for n in adamw)   # the true residue only


def test_broad_mapping_routing_moves_the_address_producers_to_adamw():
    """The M2 appendix: the gate producer AND the address producers
    (key/query_key) on AdamW; the core matrices stay on Muon."""
    model = _tiny()
    _m, _a, muon, adamw = split_parameters_winner(model, "gates_adamw_broad")
    matrices, decay, addresses, norms = _expected(len(model.blocks))
    assert set(muon) == matrices - addresses
    assert set(adamw) == decay | addresses | norms


def test_routing_fails_closed_on_drift_and_guards():
    model = _tiny()
    # an unclassified name (inventory drift) must fail loudly
    model.extra = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="unclassified winner parameter"):
        split_parameters_winner(model, "max_muon")
    del model.extra
    # a non-2D parameter under a Muon-owned name must be rejected
    model.head.weight = nn.Parameter(torch.randn(2, 3, 4))
    with pytest.raises(ValueError, match="not a full 2D matrix"):
        split_parameters_winner(model, "max_muon")
    # distinct parameters sharing storage must be rejected (the prior line's
    # own guard)
    model = _tiny()
    model.head.weight = nn.Parameter(model.embed.weight.data)
    with pytest.raises(ValueError, match="share storage"):
        split_parameters_winner(model, "max_muon")
    # an unregistered routing must be refused by name
    with pytest.raises(ValueError, match="unregistered routing"):
        split_parameters_winner(_tiny(), "option_c")


def test_the_real_winner_inventory_classifies_completely():
    """The banked build's own inventory (meta device, the converted K10 build
    path's module tree): every parameter classifies under every routing, and
    the group sizes are the receipted counts (banked 170 Muon tensors /
    149,925,888 params; max-Muon 194 / 149,953,536; broad 122 / 146,386,944 --
    149,981,808 total, the audited count)."""
    from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys \
        import (DEFAULT_CONTEXT, DEFAULT_KEY_DIM, DEFAULT_SEGMENT_TOKENS,
                MIMETIC_SEED as K10_MIMETIC_SEED,
                convert_model_to_separated_keys)
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        MatrixModel, matrix_config)
    from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
        RoutingPolicy)

    with torch.device("meta"):
        model = MatrixModel(matrix_config(), RoutingPolicy(ops=True),
                             chunk_tokens=64, compile_scan=False)
        model = convert_model_to_separated_keys(
            model, key_dim=DEFAULT_KEY_DIM, ctx=DEFAULT_CONTEXT,
            segment_tokens=DEFAULT_SEGMENT_TOKENS, seed=K10_MIMETIC_SEED)
    params = dict(model.named_parameters())
    total = sum(p.numel() for p in model.parameters())
    assert total == 149_981_808                       # the audited count
    expected = dict(banked_gates_adamw=(170, 149_925_888, 97, 55_920),
                    max_muon=(194, 149_953_536, 73, 28_272),
                    gates_adamw_broad=(122, 146_386_944, 145, 3_594_864))
    for routing, (nm, pm, na, pa) in expected.items():
        _m, _a, muon, adamw = split_parameters_winner(model, routing)
        assert len(muon) == nm, routing
        assert sum(params[n].numel() for n in muon) == pm, routing
        assert len(adamw) == na, routing
        assert sum(params[n].numel() for n in adamw) == pa, routing
        assert all(params[n].ndim == 2 for n in muon), routing


def test_conventions_hold_the_banked_rows():
    """The arms' knobs must be the banked rows' own: every arm at 3x unless
    it is a bracket arm, the winner's tuned LR/decay/warmup the base, and the
    banked finals receipted against the bake-off and muon-control summaries."""
    bakeoff = json.loads((X45 / "receipts/optimizer_bakeoff_summary.json")
                         .read_text())
    assert BANKED["winner_muon3x_seed3401"] == \
        bakeoff["arms"]["muon3x"]["final_eval_nll"]
    control = json.loads((X45 / "receipts/muon_control_summary.json")
                         .read_text())
    assert BANKED["gdn2_muon3x_max"] == control["finals"]["gdn2_muon3x"][
        "final_eval_nll"]
    assert BANKED["gdn2_muon3x_gates_adamw"] == \
        control["finals"]["gdn2_muon3x_gates_adamw"]["final_eval_nll"]
    side_by_side = json.loads(
        (X45 / "side_by_side_runs/gdn2_armed_seed3401/result.json").read_text())
    assert BANKED["gdn2_adamw"] == side_by_side["final_eval_nll"]
    assert BANKED["winner_adamw"] == 1.822737353188651
    for label, arm in ARMS.items():
        assert arm["lr_multiplier"] in (2, 3, 4), label
        if label.startswith("muon3x"):
            assert arm["lr_multiplier"] == 3, label
    assert ARMS["muon3x_max"]["routing"] == "max_muon"
    assert ARMS["muon3x_gates_adamw"]["routing"] == "gates_adamw_broad"


def test_step1_receipt_is_the_banked_same_seed_row():
    """The protocol-identical start receipt's reference is the BANKED muon3x
    arm's own first training row at the same seed, not a restated constant."""
    for seed in (3401, 3402):
        first = json.loads(
            (X45 / f"bakeoff_runs/muon3x_seed{seed}/training.jsonl")
            .read_text().splitlines()[0])
        assert _banked_step1_loss(seed) == first["loss"]
    assert _banked_step1_loss(3401) == 5.785070835425993
    assert _banked_step1_loss(3402) == 5.768619428012168


def test_predictions_and_claim_rule_are_pre_stated():
    """The supervisor-approved pre-stated predictions and the symmetric claim
    rule: the routing-spread band (0.02) against the banked same-seed
    B-analog row, the broad-mapping neutral-to-worse direction, and the
    best-vs-best-only claim rule with its bands."""
    spread = PREDICTIONS["routing_spread"]
    assert spread["band"] == ROUTING_SPREAD_BAND == 0.02
    assert spread["reference"] == BANKED["winner_muon3x_seed3401"]
    assert "confirmed_if" in spread and "outside_band_is" in spread
    broad = PREDICTIONS["broad_mapping"]
    assert broad["arm"] == "muon3x_gates_adamw"
    assert "NEUTRAL-TO-WORSE" in broad["prediction"]
    assert WINNER_SEED_DELTA == pytest.approx(0.0119, abs=1e-4)
    assert "best-vs-best" in CLAIM_RULE["rule"].lower()
    assert "routing search symmetric" in CLAIM_RULE["rule"].lower()
    assert "parity-within-band" in CLAIM_RULE["rule"].lower()
