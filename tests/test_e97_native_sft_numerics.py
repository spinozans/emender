import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scripts.qualify_e97_native_sft_numerics import (
    INDEX, inputs, main, record_arrays, run, select_panel,
)


def test_panel_is_deterministic_training_only_and_exercises_actual_mlp_chunks():
    rows=[dict(split=split,tokens=n,assistant_units=3,trajectory_identity=f'{split}-{n}')
          for split in (0,1) for n in (8000,12289,14841,62000,65536)]
    chosen=select_panel(rows)
    assert select_panel(rows[::-1])==chosen
    assert all(x['split']==0 for x in chosen)
    assert 12289<=chosen[0]['tokens']<=16384
    assert 60000<=chosen[1]['tokens']<=65536
    with pytest.raises(ValueError,match='length bin'):select_panel(rows[:1])


def test_record_reads_match_index_and_identify_real_mask_openings(tmp_path):
    tokens=np.full(12,17,dtype='<u4');tokens[[1,5,9]]=32750
    mask=np.zeros(12,dtype='u1');mask[[1,2,5,6,9,10]]=1
    row=dict(offset=2,tokens=12,targets=6,split=0,record_index=0,assistant_units=3)
    (tmp_path/'tokens.bin').write_bytes(bytes(8)+tokens.tobytes())
    (tmp_path/'loss_mask.bin').write_bytes(bytes(2)+mask.tobytes())
    (tmp_path/'records.idx').write_bytes(INDEX.pack(2,12,6,0))
    actual,positions=record_arrays(tmp_path,row)
    assert np.array_equal(actual,tokens) and positions==[1,5,9]
    with pytest.raises(ValueError,match='index'):record_arrays(tmp_path,{**row,'offset':3})
    tokens[5]=17;(tmp_path/'tokens.bin').write_bytes(bytes(8)+tokens.tobytes())
    with pytest.raises(ValueError,match='Analysis'):record_arrays(tmp_path,row)


def test_complete_record_padding_does_not_add_targets_or_history_resets():
    raw=np.arange(12,dtype=np.uint32)
    tokens,valid,reset,mask=inputs(raw,[1,5,9],'cpu')
    assert tokens.shape==(1,17) and mask.shape==(1,16)
    assert torch.equal(tokens[0,:12],torch.arange(12))
    assert valid.sum()==12 and reset.sum()==1 and reset[0,0]
    assert mask.sum()==3 and mask[0,[0,4,8]].all()
    assert not mask[0,11:].any() and not valid[0,12:].any()
    with pytest.raises(ValueError):inputs(raw,[1,1,9],'cpu')
    with pytest.raises(ValueError):inputs(raw,[0,5,9],'cpu')
    with pytest.raises(ValueError):inputs(np.zeros(65537),[1,5,9],'cpu')


def test_bad_recipe_identity_fails_before_cuda(tmp_path,monkeypatch):
    path=tmp_path/'recipe.json';path.write_text('{}')
    def forbidden(*a,**kw):raise AssertionError('must not initialize CUDA')
    monkeypatch.setattr(torch.cuda,'set_device',forbidden)
    with pytest.raises(ValueError,match='recipe identity'):
        run(SimpleNamespace(recipe=path,recipe_sha256='0'*64))


def test_existing_output_is_not_modified_even_on_failure(tmp_path,monkeypatch):
    out=tmp_path/'existing';out.mkdir();(out/'sentinel').write_text('unchanged')
    monkeypatch.setattr('sys.argv',['qualifier','run','--recipe',str(tmp_path/'absent'),
                                   '--recipe-sha256','0'*64,'--output',str(out)])
    with pytest.raises(FileExistsError):main()
    assert [p.name for p in out.iterdir()]==['sentinel']
