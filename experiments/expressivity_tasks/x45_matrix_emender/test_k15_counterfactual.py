from experiments.expressivity_tasks.x45_matrix_emender.k15_counterfactual import alternatives
from .test_k15_relation import get


def test_single_stays_dead_while_latest_survivor_falls_back():
    sites=get(b'xyABCxyAXY'+b'0'*20+b'xyABCZZ','full'); rows=list(alternatives(sites))
    assert [(single,alt) for single,alt,_ in rows[:3]]==[(True,65),(True,66),(False,67)]
    assert sites[1][1][-1]==88


def test_controller_decoder_switches_without_labels_and_ties_keep_latest():
    import torch
    from experiments.expressivity_tasks.x45_matrix_emender.k15_counterfactual import copy_logprobs
    sites=get(b'xyABCxyAXY'+b'0'*20+b'xyABCZZ','full'); z=torch.zeros(1,37,256)
    args=(torch.ones(len(sites)),list(alternatives(sites)))
    choices=copy_logprobs(z,sites,args); noop=copy_logprobs(z,sites,args,noop=True)
    assert choices[32].argmax()==66 and noop[32].argmax()==88
    assert torch.allclose(choices.exp().sum(-1),torch.ones(37))
    assert torch.equal(noop,copy_logprobs(z,sites,(args[0]*0,args[1])))
    assert torch.equal(noop[:31],z[0,:31].log_softmax(-1))
