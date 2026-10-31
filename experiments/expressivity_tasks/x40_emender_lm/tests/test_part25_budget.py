"""Mechanism receipt tests: native occupancy and conditional cap rates."""
from types import SimpleNamespace
import torch
from experiments.expressivity_tasks.x40_emender_lm import part25_depth as depth
from experiments.expressivity_tasks.x30_tape_cell.routing_trace import CASE_NAMES


def test_native_occupancy_does_not_reconstruct_from_events(monkeypatch):
    monkeypatch.setattr(depth,'BUDGET',4)
    cap=next(k for k,v in CASE_NAMES.items() if v=='R-CAP')
    state=SimpleNamespace(native=SimpleNamespace(live=torch.tensor([[[True,True,False,False]]])))
    trace=SimpleNamespace(case=torch.tensor([[[cap],[0],[0]]]),
        alloc_target=torch.tensor([[[-1],[0],[-1]]]),w_on=torch.ones(1,3,1))
    report=depth.depth_profile(state,trace,torch.zeros(1,3,1))
    assert report['final_live_mean_per_head']==[2.0]
    assert report['final_occupancy_fraction_per_head']==[0.5]
    assert report['rcap_per_request_per_head']==[0.5]
    assert abs(report['rcap_fraction_per_head'][0]-1/3)<1e-6
