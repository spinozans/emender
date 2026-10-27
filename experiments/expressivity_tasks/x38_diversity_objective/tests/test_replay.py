"""Checks address differentiation, gradient ownership and scalar policy composition."""
import os
import pytest
import torch
from experiments.expressivity_tasks.x37_head_differentiation.model import DifferentiatedCell, input_snapshot, input_metrics
from experiments.expressivity_tasks.x34_lm_pilot.cli import replay_present
from experiments.expressivity_tasks.x36_tape_perf.cell_fused import tape_scan

pytestmark = pytest.mark.skipif(os.environ.get('GPU_LEASE_HELD') not in ('0','1','5','6'),
                                reason='broker lease required')


@pytest.mark.parametrize('arm', ['b','c','d'])
def test_head_geometry_and_replay(arm):
    torch.manual_seed(3401)
    cell = DifferentiatedCell(arm,32).cuda()
    assert not torch.equal(cell.features[0].weight,cell.features[1].weight)
    assert not torch.equal(cell.geometry[0]['key'].weight,cell.geometry[1]['key'].weight)
    x = torch.randn(2,17,256,device='cuda',requires_grad=True)
    ids = torch.arange(17,device='cuda').repeat(2,1)+1
    initial = input_snapshot(cell)
    output = cell(x,ids)
    loss = output.square().sum()
    assert replay_present(loss) == 'ReplayBackward'
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for layer in cell.W_in:
        assert torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.norm() > 0
    assert all(p.grad is None for p in cell.features.parameters())
    assert all(p.grad is None for p in cell.geometry.parameters())
    if cell.W_in:
        grads = [layer.weight.grad for layer in cell.W_in]
        assert all(not torch.equal(grads[0],g) for g in grads[1:])
        opt = torch.optim.AdamW(cell.parameters(),lr=3e-4)
        opt.step()
        assert all(d>0 for d in input_metrics(cell,initial)['distance_from_init'])
    cell.eval()
    with torch.no_grad():
        cell(x,ids)
    assert len(cell.telemetry['routing_per_head']) == 4
    assert all(sum(row.values()) == 34 for row in cell.telemetry['routing_per_head'])


def test_batched_scan_equals_independent_heads():
    from dataclasses import replace
    from experiments.expressivity_tasks.x31_tape_cell_fused.cell_fused import TapeHeadConfig
    torch.manual_seed(37)
    cfg = TapeHeadConfig(n_heads=4,budget=16)
    tensors = [torch.randn(2,9,4,d,device='cuda') for d in (64,64,128)]
    tensors += [torch.ones(2,9,4,device='cuda'),torch.zeros(2,9,4,device='cuda'),
                torch.randn(2,9,4,device='cuda')]
    coef = torch.zeros(4,device='cuda')
    with torch.no_grad():
        full = tape_scan(*tensors,None,cfg,coef)[0]
        parts = [tape_scan(*(v[:,:,h:h+1].contiguous() for v in tensors),None,
                           replace(cfg,n_heads=1),coef[h:h+1])[0] for h in range(4)]
    assert torch.equal(full,torch.cat(parts,2))


def test_diversity_gradient_reaches_input_projection():
    from experiments.expressivity_tasks.x38_diversity_objective.model import read_diversity
    cell = DifferentiatedCell('c',32).cuda()
    x = torch.randn(2,17,256,device='cuda',requires_grad=True)
    ids = torch.arange(17,device='cuda').repeat(2,1)+1
    penalty,_,coverage = read_diversity(cell(x,ids))
    assert replay_present(penalty) == 'ReplayBackward'
    penalty.backward()
    assert coverage.min() > 0
    assert all(layer.weight.grad.norm()>0 for layer in cell.W_in)
    assert all(p.grad is None for p in cell.geometry.parameters())


def test_slot_shadow_replay():
    from experiments.expressivity_tasks.x38_diversity_objective.model import slot_usage
    cell = DifferentiatedCell('c',128).cuda().eval()
    x = torch.randn(1,97,256,device='cuda')
    ids = torch.arange(97,device='cuda').view(1,-1)+1
    with torch.no_grad():
        expected = cell(x,ids)
        metrics = slot_usage(cell,x,ids,expected)
    assert len(metrics['slot_usage_js']) == 6
    assert all(0 <= v <= 1 for v in metrics['slot_usage_js'].values())
