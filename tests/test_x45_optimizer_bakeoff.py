"""Source-named checks for the x45 optimizer bake-off: the x45 parameter
routing, the hybrid Muon step's fidelity to the prior line's pinned
KellerJordan convention (hybrid_muon), the schedule-free eval-point snapshot
against the package's own eval()/train() swap, and the arms' symmetry rules.
CPU-only.
"""
import math

import pytest
import torch
import torch.nn as nn

from experiments.expressivity_tasks.hybrid_muon import scaled_update
from experiments.expressivity_tasks.x45_matrix_emender.bakeoff_optimizers import (
    HybridMuonX45, sf_x_snapshot, split_parameters)
from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff import (
    ARMS, TUNED_WINNER)


class TinyMixer(nn.Module):
    """The converted K10 mixer's own parameter names (not its arithmetic)."""

    def __init__(self, width=8, heads=2, value_dim=4, key_dim=3, ctx=2):
        super().__init__()
        self.value = nn.Linear(width, heads * value_dim, bias=False)
        self.key = nn.Linear(ctx * width, heads * key_dim, bias=False)
        self.query_key = nn.Linear(ctx * width, heads * key_dim, bias=False)
        self.out = nn.Linear(heads * value_dim, width, bias=False)
        self.decay = nn.Linear(width, heads, bias=True)


class TinyBlock(nn.Module):
    def __init__(self, width=8, hidden=12):
        super().__init__()
        self.norm1 = nn.RMSNorm(width, eps=1e-5)
        self.mixer = TinyMixer(width)
        self.norm2 = nn.RMSNorm(width, eps=1e-5)
        self.mlp = nn.ModuleDict({name: nn.Linear(width if name != "w3"
                                                  else hidden,
                                                  hidden if name != "w3"
                                                  else width, bias=False)
                                  for name in ("w1", "w2", "w3")})


class TinyModel(nn.Module):
    """The converted winner's parameter-name tree (MatrixModel + the K10
    conversion: mixer.{value,key,query_key,out,decay}, embed/head/norm)."""

    def __init__(self, width=8, vocab=16, depth=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.norm = nn.RMSNorm(width, eps=1e-5)
        self.head = nn.Linear(width, vocab, bias=False)
        self.blocks = nn.ModuleList([TinyBlock(width) for _ in range(depth)])


def _tiny_model():
    torch.manual_seed(0)
    return TinyModel().eval()          # eval: no dropout/init RNG surprises


def test_routing_is_embeddings_and_projections_only():
    model = _tiny_model()
    _muon_params, _adamw_params, muon, adamw = split_parameters(model)
    expected_muon = {"embed.weight", "head.weight"}
    for i in range(len(model.blocks)):
        expected_muon |= {
            f"blocks.{i}.mixer.{part}.weight"
            for part in ("value", "key", "query_key", "out")}
        expected_muon |= {f"blocks.{i}.mlp.{w}.weight" for w in ("w1", "w2", "w3")}
    expected_adamw = {"norm.weight"}
    for i in range(len(model.blocks)):
        expected_adamw |= {f"blocks.{i}.norm1.weight",
                           f"blocks.{i}.norm2.weight",
                           f"blocks.{i}.mixer.decay.weight",
                           f"blocks.{i}.mixer.decay.bias"}
    assert set(muon) == expected_muon
    assert set(adamw) == expected_adamw
    # the approved routing is "embeddings + projections": every muon-owned
    # name is a 2D embedding/linear weight, every gate/norm stays on AdamW.
    params = dict(model.named_parameters())
    for name in muon:
        assert params[name].ndim == 2


def test_routing_fails_closed_on_non2d_muon_parameter_and_shared_storage():
    model = _tiny_model()
    # a non-2D parameter under a Muon-owned name must be rejected (the NS
    # convention requires full matrices).
    model.blocks[0].mixer.value.weight = nn.Parameter(
        torch.randn(2, 3, 4))
    with pytest.raises(ValueError, match="not a full 2D matrix"):
        split_parameters(model)
    # distinct parameters sharing storage must be rejected (the prior line's
    # own guard: a shared-storage parameter would be updated twice).
    model = _tiny_model()
    model.head.weight = nn.Parameter(model.embed.weight.data)
    with pytest.raises(ValueError, match="share storage"):
        split_parameters(model)


def test_hybrid_muon_step_matches_pinned_convention_exactly():
    """One step of HybridMuonX45 must equal the prior line's HybridMuon math
    (hybrid_muon.scaled_update, lerp momentum .05 / Nesterov .95, decoupled
    decay on the matrix group) bit-for-bit, and the aux AdamW group must be
    the probe's betas/decay at the facade's LR."""
    muon = [nn.Parameter(torch.randn(6, 4)), nn.Parameter(torch.randn(3, 7))]
    aux = [nn.Parameter(torch.randn(4)), nn.Parameter(torch.randn(2, 6))]
    lr, betas, wd = 1e-2, (0.9, 0.9392004369596886), 0.05
    opt = HybridMuonX45(muon, aux, lr=lr, betas=betas, weight_decay=wd)
    reference = torch.optim.AdamW(aux, lr=lr, betas=betas, eps=1e-8,
                                  weight_decay=wd)
    grads_muon = [torch.randn_like(p) for p in muon]
    grads_aux = [torch.randn_like(p) for p in aux]
    before_muon = [p.detach().clone() for p in muon]
    for p, g in zip(muon, grads_muon):
        p.grad = g.clone()
    for p, g in zip(aux, grads_aux):
        p.grad = g.clone()
    opt.step()
    for p, g, before in zip(muon, grads_muon, before_muon):   # the pinned math
        buffer = torch.zeros_like(p, dtype=torch.float32)
        buffer.lerp_(g, .05)
        direction = g.lerp(buffer, .95)
        update = scaled_update(direction)
        expected = before * (1 - lr * wd)
        expected.add_(update.to(p.dtype), alpha=-lr)
        assert torch.equal(p, expected)
    for p, g in zip(aux, grads_aux):             # the reference aux step
        reference.param_groups[0]["params"][0].grad = None
    ref_params = reference.param_groups[0]["params"]
    for p, g in zip(ref_params, grads_aux):
        p.grad = g.clone()
    reference.step()
    for mine, theirs in zip(aux, ref_params):
        assert torch.equal(mine, theirs)
    assert math.isfinite(opt.last_metrics["muon_update_rms"])


def test_sf_x_snapshot_is_the_package_eval_point_and_round_trips():
    """sf_x_snapshot must equal the package's own eval() swap (the primal
    average x), must not perturb the live training parameters, and the
    package's train() must restore them."""
    import schedulefree
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(5, 4), nn.Linear(4, 3))
    opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=1e-2,
                                         betas=(0.9, 0.98), weight_decay=0.01)
    opt.train()
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        model(torch.randn(2, 5)).sum().backward()
        opt.step()
    y_live = {k: v.detach().clone() for k, v in model.state_dict().items()}
    snapshot = sf_x_snapshot(model, opt, 0.9)
    assert not any(torch.equal(snapshot[k], y_live[k]) for k in snapshot), \
        "the primal average must differ from the live training iterate"
    after = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for k in y_live:                             # the live path is untouched
        assert torch.equal(y_live[k], after[k])
    opt.eval()                                   # the package's own swap
    package_eval = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for k in snapshot:
        assert torch.allclose(snapshot[k], package_eval[k], atol=1e-6), k
    opt.train()
    restored = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for k in y_live:
        assert torch.allclose(y_live[k], restored[k], atol=1e-6), k


def test_arm_conventions_hold_the_symmetry_rules():
    """The approved conventions, as spec'd: the SF arms carry NO schedule by
    design; the Muon/AdamW arms share the incumbent's warmup-45 schedule; the
    matched-to-incumbent arms hold the incumbent's lr/betas/decay; the prior
    bracket is the ONLY arm at the prior line's knobs."""
    for label in ("sf", "muon1x", "muon3x", "sf_muon"):
        assert label in ARMS
    for label in ("sf", "sf_prior", "sf_muon"):
        assert ARMS[label]["spec"]["warmup_steps"] == 0, label
    for label in ("adamw", "muon1x", "muon3x"):
        assert ARMS[label]["spec"]["warmup_steps"] == TUNED_WINNER["warmup_steps"], label
    for label in ("adamw", "sf", "muon1x", "muon3x", "sf_muon"):
        spec = ARMS[label]["spec"]
        assert spec["betas"] == (0.9, 0.9392004369596886), label
        if label != "sf_muon":           # the wrapper carries decay at y instead
            assert spec["weight_decay"] == TUNED_WINNER["weight_decay"], label
    assert ARMS["muon1x"]["spec"]["lr"] == TUNED_WINNER["lr"]
    assert ARMS["muon3x"]["spec"]["lr"] == 3 * TUNED_WINNER["lr"]
    assert ARMS["sf"]["spec"]["lr"] == TUNED_WINNER["lr"]
    prior = ARMS["sf_prior"]["spec"]
    assert (prior["lr"], prior["betas"], prior["weight_decay"]) == \
        (1e-3, (0.9, 0.95), 0.01)
