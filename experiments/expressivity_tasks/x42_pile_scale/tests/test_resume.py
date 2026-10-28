import pytest
import torch
from experiments.expressivity_tasks.x42_pile_scale.checkpoint import save,restore
from experiments.expressivity_tasks.x42_pile_scale.resume_probe import roundtrip


def test_kill_resume_exact_continuation():
    receipt=roundtrip(dict(width=16,layers=1,heads=2,activation_checkpointing=True,
                           context=8,lr=.0003))
    assert receipt['bit_identical']
    assert receipt['kill_exit_code']==-9
    assert receipt['updates'][-1]['cursor']==6


def test_checkpoint_rejects_changed_registration(tmp_path):
    model=torch.nn.Linear(2,2)
    optimizer=torch.optim.AdamW(model.parameters())
    path=tmp_path/'latest.ckpt'
    save(path,model,optimizer,dict(seed=1),1,2,[],0.)
    with pytest.raises(ValueError,match='registration mismatch'):
        restore(path,model,optimizer,dict(seed=2))


def test_attestation_waits_and_rejects_bad_hash(tmp_path):
    import json
    from experiments.expressivity_tasks.x42_pile_scale.attest_corpus import attest
    from experiments.expressivity_tasks.x42_pile_scale.data import build
    class Encoded:
        ids=[1,2]*16
    class Tokenizer:
        def encode(self,text): return Encoded()
        def save(self,path):
            from pathlib import Path
            Path(path).write_text('fixture')
    root=tmp_path/'corpus';root.mkdir()
    (root/'progress.json').write_text(json.dumps(dict(status='running')))
    assert not attest(root)['final_attestation']
    root=tmp_path/'complete'
    build((dict(text=f'doc {i}') for i in range(1000)),Tokenizer(),root,4096,{},.5)
    assert attest(root)['final_attestation']
    with (root/'train.bin').open('r+b') as f:
        f.write(b'XX')
    with pytest.raises(AssertionError):
        attest(root)
