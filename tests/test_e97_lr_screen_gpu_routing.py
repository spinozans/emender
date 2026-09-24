import pytest
from scripts import eval_e97_lr_screen as evaluator


@pytest.mark.parametrize('rank',range(8))
def test_torchrun_rank_selects_its_own_visible_gpu(monkeypatch,rank):
    monkeypatch.setenv('LOCAL_RANK',str(rank))
    monkeypatch.setattr(evaluator.torch.cuda,'device_count',lambda:8)
    selected=[]
    monkeypatch.setattr(evaluator.torch.cuda,'set_device',selected.append)
    assert evaluator.select_cuda_device()==f'cuda:{rank}'
    assert selected==[rank]


@pytest.mark.parametrize('rank',[-1,8])
def test_invalid_cuda_rank_fails_closed(monkeypatch,rank):
    monkeypatch.setenv('LOCAL_RANK',str(rank))
    monkeypatch.setattr(evaluator.torch.cuda,'device_count',lambda:8)
    with pytest.raises(RuntimeError,match='outside visible'):
        evaluator.select_cuda_device()
