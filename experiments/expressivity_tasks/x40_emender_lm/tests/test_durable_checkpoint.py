"""Exercise the actual depth sampler, optimizer and best-weight resume loop on CPU."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.expressivity_tasks.x40_emender_lm import depth, parity
from experiments.expressivity_tasks.x42_pile_scale.resume_probe import state_hash


class TinyProbe(torch.nn.Module):
    def __init__(self,lm):
        super().__init__();self.linear=torch.nn.Linear(1,2)
    def forward(self,ids,lengths):
        return self.linear(ids[:,:1].float())


def test_depth_resume_inside_epoch_restores_optimizer_sampler_and_best(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self:self)
    monkeypatch.setattr(depth,'Probe',TinyProbe)
    monkeypatch.setattr(depth,'require_replays',lambda loss:None)
    monkeypatch.setattr(parity,'certify',lambda **kw:{'status':'passed'})
    pool=(np.arange(320,dtype=np.int64).reshape(320,1)%5,np.arange(320)%2,np.ones(320,dtype=np.int64))
    monkeypatch.setattr(depth.ladder,'ladder_train_pool',lambda:pool)
    monkeypatch.setattr(depth.ladder,'ladder_validation_pool',lambda:pool)
    monkeypatch.setattr(depth.ladder,'ladder_bin_pool',lambda name:pool)
    monkeypatch.setattr(depth.ladder,'ladder_hash',lambda *args:'fixture')
    monkeypatch.setattr(depth,'evaluate',lambda probe,pool:{'balanced_accuracy':float(probe.linear.weight.detach().sum())})
    monkeypatch.setattr(depth,'write',lambda *args:None)
    for name in ('reference','interrupted'):(tmp_path/name).mkdir()
    def run(name,resume=False):
        monkeypatch.setenv('X40_RECEIPT_DIR',str(tmp_path/name))
        depth.fine_tune(None,tmp_path/name,resume,updates=8,checkpoint_every=2,validation_every=2)
    run('reference')
    save=depth.save
    def crash(*args,**kwargs):
        save(*args,**kwargs)
        if args[4]==4:raise RuntimeError('simulated post-checkpoint process loss')
    monkeypatch.setattr(depth,'save',crash)
    with pytest.raises(RuntimeError,match='process loss'):run('interrupted')
    monkeypatch.setattr(depth,'save',save)
    run('interrupted',True)
    expected=torch.load(tmp_path/'reference/depth.ckpt',weights_only=False)
    actual=torch.load(tmp_path/'interrupted/depth.ckpt',weights_only=False)
    for key in ('model','optimizer','cursor','rows','torch_rng'):
        assert state_hash(expected[key])==state_hash(actual[key]),key


def test_checkpoint_rejects_changed_registration(tmp_path):
    from experiments.expressivity_tasks.x42_pile_scale.checkpoint import save,restore
    model=torch.nn.Linear(1,1);opt=torch.optim.AdamW(model.parameters())
    save(tmp_path/'state.ckpt',model,opt,{'seed':0},0,0,[],0.)
    with pytest.raises(ValueError,match='registration mismatch'):
        restore(tmp_path/'state.ckpt',model,opt,{'seed':1})
