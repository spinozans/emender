"""Transplant boundaries and unfrozen-gradient graph structure."""
import ast
import inspect
import textwrap
import torch
from experiments.expressivity_tasks.x42_pile_scale import x42_mimetic_geometry as experiment
from experiments.expressivity_tasks.x42_pile_scale.velocity_candidate import Candidate, Policy


def test_transplant_preserves_frozen_and_unrelated_parameters(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(3401)
    model=Candidate(Policy(N=2,w_in_width=58),layers=1,target=82_000_000)
    state={}
    hidden=128
    for name,shape in {
        'qkv.weight':(2304,768),'qkv.bias':(2304,),
        'out.weight':(768,768),'out.bias':(768,),
        'ff.0.weight':(hidden,768),'ff.0.bias':(hidden,),
        'ff.2.weight':(768,hidden),'ff.2.bias':(768,),
    }.items():state['model.blocks.0.'+name]=torch.randn(shape)*.02
    donor=tmp_path/'weights.pt';torch.save(state,donor)
    monkeypatch.setattr(experiment,'OUT',tmp_path)
    monkeypatch.setattr(experiment,'DONOR',donor)
    frozen={n:p.clone() for n,p in model.named_parameters() if not p.requires_grad}
    unrelated=model.embed.weight.clone()
    experiment.transplant(model,{'state_dict':state})
    assert torch.equal(model.embed.weight,unrelated)
    assert all(torch.equal(dict(model.named_parameters())[n],p) for n,p in frozen.items())
    block=model.blocks[0]
    assert torch.linalg.matrix_rank(block.cell.W_in[1].weight@block.cell.W_in[0].weight)<=58
    # Neuron replication preserves the donor MLP function when recipient is wider.
    x=torch.randn(3,768)
    expected=torch.nn.functional.linear(torch.nn.functional.gelu(
        torch.nn.functional.linear(x,state['model.blocks.0.ff.0.weight'],state['model.blocks.0.ff.0.bias'])),
        state['model.blocks.0.ff.2.weight'],state['model.blocks.0.ff.2.bias'])
    torch.testing.assert_close(block.ff(x),expected,atol=2e-6,rtol=2e-5)


def test_diagnostic_removes_only_key_detach():
    forward=experiment.diagnostic_forward()
    # Execute the compiled forward with a minimal differentiable scan stub.
    assert forward.__code__.co_names.count('detach')==0
    original=ast.parse(textwrap.dedent(inspect.getsource(experiment.Cell.forward)))
    assert 'detach' in ast.unparse(original)
