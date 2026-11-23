import pytest
from experiments.expressivity_tasks.x45_matrix_emender import k15_bookmark_joint as joint
from experiments.expressivity_tasks.x45_matrix_emender.k15_bookmark_payload import LastByteSession


def test_last_byte_only_after_locate():
    native = joint.BookmarkSession(joint.EXE)
    try:
        s = LastByteSession(native)
        s.rpc('append 41424341'); assert s.rpc('byte 1')['byte'] == 66
        s.rpc('locate 41 0 4 4'); assert s.rpc('byte 1')['byte'] == 65
        assert s.write_pairs == [[66,65]] and s.reads[-1][1]['byte'] == 66
    finally: native.close()


def test_decode_pin(tmp_path, monkeypatch):
    path = tmp_path/'joint.pt'; joint.torch.save({'step':4095,'model':{}}, path)
    with pytest.raises(ValueError,match='digest'): joint.load_d_model(path)
    monkeypatch.setattr(joint,'D_SHA256',joint.digest(path))
    with pytest.raises(ValueError,match='step'): joint.load_d_model(path)
