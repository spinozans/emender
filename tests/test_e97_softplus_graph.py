"""X24 source-named protocol reuse and single-card resource guard tests."""
import json
import signal
from types import SimpleNamespace

import pytest
import torch

from experiments.expressivity_tasks import run_e97_softplus_graph as arm
from scripts import watch_e97_softplus_graph as watcher


def test_softplus_graph_has_only_state_activation_difference():
    torch.manual_seed(0)
    stock = arm.STOCK_BUILD_MODEL('e97','graph')
    torch.manual_seed(0)
    model = arm.build_model('e97','graph')
    assert model.state_dict().keys() == stock.state_dict().keys()
    assert all(torch.equal(v,stock.state_dict()[k]) for k,v in model.state_dict().items())
    for a,b in zip(model.layers,stock.layers):
        assert a.state_activation == 'softplus' and b.state_activation == 'tanh'
        assert a.recurrent_state_precision == b.recurrent_state_precision == 'legacy'
        assert a.use_split_edit and a.use_triton and a.fused_inference
    with pytest.raises(ValueError):
        arm.build_model('gdn2','graph')


def test_shared_dataset_link_refuses_independent_pools(tmp_path,monkeypatch):
    root,comparison = tmp_path/'x24',tmp_path/'x23e'
    root.mkdir(); (comparison/'datasets').mkdir(parents=True)
    monkeypatch.setattr(arm,'ROOT',root)
    monkeypatch.setattr(arm,'COMPARISON_ROOT',comparison)
    arm.link_datasets(); arm.link_datasets()
    assert (root/'datasets').is_symlink()
    assert (root/'datasets').resolve() == (comparison/'datasets').resolve()
    (root/'datasets').unlink(); (root/'datasets').mkdir()
    with pytest.raises(ValueError,match='X23E dataset'):
        arm.link_datasets()


@pytest.mark.parametrize('alive', [False, True])
def test_single_rung_guard_signals_only_identity_verified_process_group(tmp_path,monkeypatch,alive):
    monkeypatch.setattr(watcher,'ROOT',tmp_path)
    monkeypatch.setattr(watcher,'configure_watcher',lambda:None)
    monkeypatch.setattr(watcher.protocol,'rung_status',lambda name:'running')
    monkeypatch.setattr(watcher.protocol,'alive',lambda record:alive)
    monkeypatch.setattr(watcher.shutil,'disk_usage',lambda path:SimpleNamespace(free=0))
    receipt = tmp_path/'receipts'/watcher.NAME; receipt.mkdir(parents=True)
    (receipt/'worker.json').write_text(json.dumps({'pid':123,'launched_at':0}))
    signals=[]
    monkeypatch.setattr(watcher.os,'killpg',lambda pid,sig:signals.append((pid,sig)))
    assert watcher.watch(once=True) == 2
    assert signals == ([(123,signal.SIGTERM)] if alive else [])
    assert 'infrastructure censoring' in json.loads((receipt/'guard.json').read_text())['decision']


def test_launch_disk_guard_precedes_remote_receipt_or_process_spawn(tmp_path,monkeypatch):
    monkeypatch.setattr(watcher,'ROOT',tmp_path)
    monkeypatch.setattr(watcher,'configure_watcher',lambda:None)
    monkeypatch.setattr(watcher.shutil,'disk_usage',lambda path:SimpleNamespace(free=0))
    def forbidden():
        raise AssertionError('launch mutation before disk guard')
    monkeypatch.setattr(watcher.protocol,'launch',forbidden)
    with pytest.raises(OSError,match='disk guard'):
        watcher.watch(initial=True)
