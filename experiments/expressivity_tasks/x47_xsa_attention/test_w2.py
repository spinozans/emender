"""W2 acceptance + negative cases. The entire four-case production-shaped trace is in panel()."""
from unittest.mock import patch

import numpy as np
import pytest
import torch

from .workbench import Handle, Meter
from .workbench_native import Live
from .workbench_witness import close
from .w2_join import Cell, Episode, FixedJoin, Neural, Reply, ReplicatedMount, bind
from .w2_panel import WORLDS, initial_model, panel, prefill, prompt


def test_complete_cross():
    assert len(panel(initial_model())['cases'])==16


@torch.inference_mode()
def test_arbitrary_symbols_and_response_order_do_not_change_semantics():
    # Reverse completion order, unrelated code and tag bytes, ordinary NUL and 0xff.
    docs=(b'key=\xff\x00',b'key=\x00\x1e')
    with ReplicatedMount(docs) as bank:
        read=bank.children
        with patch.object(bank,'children',side_effect=lambda keys:read(keys)[::-1]):
            model=initial_model()
            for tag,answer in ((0,255),(30,0)):
                with Live() as live:
                    d=Episode(Neural(model,'full'),live,bank); m=Meter(trace=True)
                    prefill(d,prompt(tag),m); p,c=d.predict(m)
                    assert c==answer and m.views[-2]['rank']==[0,255]
                    assert abs(p[c].item()-(4+1/256)/5)<3e-5


def test_missing_stale_and_duplicate_reply_refused():
    x,y=(Handle('f','v',k) for k in (b'key=a',b'key=b'))
    def r(h): return Reply(h,(0,)*257,(0,0))
    assert [v.locus for v in bind((x,y),(r(y),r(x)))]==[x,y]
    for bad in ((r(x),r(x)),(r(x),),(r(x),r(Handle('f','other',y.key)))):
        with pytest.raises(ValueError,match='binding'):
            bind((x,y),bad)


@torch.inference_mode()
def test_zero_end_overcap_and_no_record():
    model=initial_model()
    for docs,p_expected in (((b'key=',),1/256),((b'miss',),1/256),(WORLDS[0],(2+1/256)/5)):
        with ReplicatedMount(docs) as bank, Live() as live:
            d=Episode(Neural(model,'full'),live,bank); m=Meter(trace=True)
            prefill(d,b'junk=M;0123456789;key=',m); p,_=d.predict(m)
            assert d.record is None and abs(p[ord('a')].item()-p_expected)<3e-5
            if docs!=(WORLDS[0]):
                assert m.reads==1  # never mistake END for an outgoing byte
    with ReplicatedMount((*WORLDS[0],b'key=cQ')) as bank, Live(b'key=') as live:
        with pytest.raises(ValueError,match='fanout'):
            Episode(Neural(model,'full'),live,bank).predict(Meter())


def test_wrong_session_record_and_denied_scope():
    with patch('subprocess.Popen',side_effect=AssertionError('must not open')):
        with pytest.raises(PermissionError):
            ReplicatedMount(WORLDS[0],excluded_docs=(1,))
    with ReplicatedMount(WORLDS[0]) as bank, Live(b'key=') as live:
        d=Episode(FixedJoin(),live,bank); d.record=Cell('another-document',1,ord('M'))
        with pytest.raises(ValueError,match='stale'):
            d.predict(Meter())


def test_writer_receives_later_credit_without_an_update():
    model=initial_model()
    with ReplicatedMount(WORLDS[0]) as bank, Live() as live:
        d=Episode(Neural(model,'full'),live,bank); m=Meter()
        prefill(d,prompt(ord('M')),m); assert live.tail==b'789;key=' and d.record.written_at==6
        p,_=d.predict(m); grad,=torch.autograd.grad(-p[ord('a')].log(), model.writer.weight)
        assert torch.isfinite(grad).all() and abs(grad[ord('S'),ord('M')].item())>.1


@torch.inference_mode()
def test_payload_transport_is_load_bearing():
    model=initial_model(); model.out.weight[:256].zero_()   # disable reader payload, preserve op flags
    with ReplicatedMount(WORLDS[0]) as bank, Live() as live:
        d=Episode(Neural(model,'full'),live,bank); m=Meter()
        prefill(d,prompt(ord('M')),m); p,_=d.predict(m)
        assert m.reads==3
        close(p.numpy(),np.full(256,1/256))
