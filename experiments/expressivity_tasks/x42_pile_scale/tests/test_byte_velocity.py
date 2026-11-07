import hashlib
import json
import numpy as np
import pytest
from experiments.expressivity_tasks.x42_pile_scale.byte_stream import ByteStream
from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import dimensions,decode,model_for,CAPACITY


def test_boundaries_and_tampering(tmp_path):
    source=tmp_path/'pile'; source.write_bytes(b'abc!WXYZ')
    index=tmp_path/'offsets'
    rows=[dict(source_index=0,split='train',byte_offset=0,byte_length=3,sha256=hashlib.sha256(b'abc').hexdigest()),dict(source_index=1,split='valid',byte_offset=4,byte_length=4,sha256=hashlib.sha256(b'WXYZ').hexdigest())]
    index.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    stream=ByteStream(source,index)
    train=list(stream.windows('train',2)); valid=list(stream.windows('valid',2))
    assert [(r[2].tolist(),r[3].tolist()) for r in train]==[([97,98],[98,99])]
    assert [(r[2].tolist(),r[3].tolist()) for r in valid]==[([87,88],[88,89]),([89],[90])]
    source.write_bytes(b'abd!WXYZ')
    with pytest.raises(ValueError,match='hash mismatch'): list(stream.windows('train'))


@pytest.mark.parametrize('arm',['transformer','gdn2','pure'])
def test_byte_capacity_and_space(arm):
    space=dimensions(arm)
    center=np.array([np.log(c/lo)/np.log(hi/lo) if log else (c-lo)/(hi-lo) for _,lo,hi,c,log in space])
    policy=decode(arm,center)
    for vector in [np.zeros(len(space)), np.ones(len(space))]:
        endpoint=decode(arm,vector)
        assert 1e-5<=endpoint.lr<=3e-3 and 10<=endpoint.warmup_steps<=8000
    model=model_for(arm,policy)
    assert abs(sum(p.numel() for p in model.parameters())/CAPACITY-1)<.001
    embedding=model.embed if arm=='pure' else model.model.embed
    assert embedding.num_embeddings==256


def test_masked_chunked_loss_has_real_byte_normalization():
    import torch
    from experiments.expressivity_tasks.x36_tape_perf.loss import ChunkedLoss
    torch.manual_seed(3)
    hidden=torch.randn(2,4,3,requires_grad=True)
    weight=torch.randn(256,3,requires_grad=True)
    labels=torch.tensor([[0,255,-100,-100],[4,2,8,-100]])
    valid=labels.ne(-100).sum()
    actual=ChunkedLoss.apply(hidden,weight,labels,4)*labels.numel()/valid
    expected=torch.nn.functional.cross_entropy(torch.nn.functional.linear(hidden,weight).reshape(-1,256),labels.reshape(-1))
    torch.testing.assert_close(actual,expected)
    gradients=torch.autograd.grad(actual,(hidden,weight))
    reference=torch.autograd.grad(expected,(hidden,weight))
    for a,b in zip(gradients,reference): torch.testing.assert_close(a,b)
    assert gradients[0][labels.eq(-100)].eq(0).all()


def test_occupancy_slopes_use_one_document_prefix(tmp_path):
    source=tmp_path/'pile'; index=tmp_path/'offsets'
    texts=[b'a'*700,b'b'*1200,bytes(range(256))*20]
    source.write_bytes(b''.join(texts)); offset=0; rows=[]
    for i,text in enumerate(texts):
        rows.append(dict(source_index=i,split='valid',byte_offset=offset,byte_length=len(text),sha256=hashlib.sha256(text).hexdigest())); offset+=len(text)
    index.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    probes=ByteStream(source,index).slope_windows()
    assert [len(r[2]) for r in probes]==[512,1024,2048,4096]
    assert all(r[0]==2 and r[1]==0 for r in probes)
    for row in probes:
        assert row[2].tobytes()==probes[-1][2][:len(row[2])].tobytes()
        assert row[3].tobytes()==probes[-1][3][:len(row[3])].tobytes()
