from .k15_bookmarks import BookmarkSession
from pathlib import Path
EXE = Path.home() / "sxgc/xsa/target/debug/xsa"
from .k15_coupled_port import candidates


def test_native_suffix_extension_and_multiple_historical_continuations():
    raw=b'abcdefghijX_abcdefghijY_abcdefghij'; s=BookmarkSession(EXE)
    try:
        s.rpc('append '+raw.hex()); rows=candidates(raw,s,True)
        assert {b for n,_,b in rows if n>=10} >= {88,89}
        assert all(pos<len(raw) for _,pos,_ in rows)
    finally: s.close()


def test_reserved_row_against_direct_oracle_and_gated_gradients():
    from types import SimpleNamespace
    import torch
    from torch.nn import functional as F
    from .k15_coupled_port import CoupledPort

    torch.manual_seed(3411)
    # Two candidate occurrence sets, including a historical continuation that
    # remains valid for the first five future inputs. Only one decision here.
    raw=(b'abcd12XYZ---ef12Q---'+b'x'*40+b'ab12')+b'XYZ---0123456789'
    hidden=torch.randn(1,len(raw),6); ids=torch.tensor([list(raw)])
    mixer=SimpleNamespace(ctx=1, key_dim=6, value_dim=4, query_key=lambda x:x)
    port=CoupledPort(6,4)
    def run(write):
        s=BookmarkSession(EXE)
        try:return port(mixer,hidden,ids,s,forced={64:write},audit=True)
        finally:s.close()
    zw,tr,gate,_=run(True); zs,*_=run(False)
    prefix=raw[:64]; opts={}
    for k in (2,4,8):
        hits=[j for j in range(64-k) if prefix[j:j+k]==prefix[-k:]]
        for j in hits[::max(1,len(hits)//3)]+hits[-1:]:
            a=j+k
            length=max(t for t in range(k,min(24,a)+1) if prefix[a-t:a]==prefix[-t:]);opts[a]=(length,a,prefix[a])
    options=sorted(opts.values(), reverse=True);assert tr[0]['candidates']==options
    h=hidden[0,63]; b=port.byte.weight[[row[2] for row in options]]
    logits=(b*port.query(h)).sum(-1)/2+.6*torch.tensor([row[0] for row in options]); joint=torch.cat((h,(logits.softmax(0)[:,None]*b).sum(0)))
    val=port.update(joint).sigmoid()*port.fuse(joint).tanh(); a=options[int(logits.argmax())][1]
    expected=zs.detach().clone(); assert torch.equal(zs[0,64:],torch.zeros_like(zs[0,64:]))
    for t in range(64,len(raw)):
        k=t-63; r=F.cosine_similarity(h,hidden[0,t],dim=0)*val
        if a+k<64 and prefix[a:a+k]==raw[64:t+1]:r=r+port.byte.weight[prefix[a+k]]
        expected[0,t]=.05*r
    torch.testing.assert_close(zw,expected); (zw.square().sum()+gate[0][1]).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in port.parameters())


def test_stage_restart_requires_receipted_checkpoint(tmp_path):
    import hashlib
    import pytest
    from .k15_coupled_launch import stage_restart
    prev=tmp_path/'prev'; out=tmp_path/'out';prev.mkdir();out.mkdir()
    (prev/'launch.json').write_text('{"pid":99999999,"starttime":"0"}');(prev/'model.pt').write_bytes(b'snapshot');(prev/'oracle').write_text('exact')
    with pytest.raises(ValueError,match='hash mismatch'):stage_restart(out,prev,'bad')
    assert not (out/'model.pt').exists()
    stage_restart(out,prev,hashlib.sha256(b'snapshot').hexdigest()); assert (out/'model.pt').read_bytes()==b'snapshot' and (out/'oracle').read_text()=='exact'
