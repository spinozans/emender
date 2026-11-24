"""W1 gates; intended CPU invocation has CUDA_VISIBLE_DEVICES='' and OMP_NUM_THREADS=4."""
import math
from unittest import mock

import numpy as np
import pytest
import torch

from .workbench import Decoder, Direct, Meter
from .workbench_native import Archive, Live
from .workbench_witness import PREFIX, WORLDS, close, initial_model, oracle, scan, witness


def test_eight_native_replays():
    assert len(witness(initial_model())['cases']) == 8


def test_boundary_semantics_binary_and_overlaps():
    docs = (b'aaa\x00\xff', b'\x1e\xff', b'\xff')
    with Archive(docs) as a, Live(docs[0]) as live:
        for key in (b'aa', b'a\x00', b'\xff', b'\xff\x1e', b'\x1e', b'absent'):
            assert list(a.branch(key)) == scan(docs,key,closed=True)
            assert list(live.branch(key)) == scan(docs[:1],key,closed=False)
        assert a.branch(b'\xff')[-1] == 3 and not any(live.branch(b'\xff'))
        with pytest.raises(ValueError,match='depth'):
            a.branch(b'x'*17)


def test_scope_refused_before_open():
    with mock.patch('subprocess.Popen', side_effect=AssertionError('opened')):
        for kwargs in ({'allowed':False}, {'excluded_docs':(0,)}):
            with pytest.raises(PermissionError):
                Archive(WORLDS[0],**kwargs)


@torch.inference_mode()
def test_attention_is_not_a_decorative_forward():
    model = initial_model()
    model.out.weight[:256].zero_()  # disable evidence transport, keep opcode metadata route intact
    with Live(PREFIX) as live, Archive(WORLDS[0]) as a:
        d = Decoder(model,live,a); m = Meter(trace=True); p,_ = d.predict(m)
        assert [v['op'] for v in m.views] == ['BASE','EMIT'] and d.record is None
        close(p.numpy(),np.full(256,1/256))
        assert [bytes.fromhex(r['key']) for r in m.requests if r['source']=='archive'] == [PREFIX]


@torch.inference_mode()
def test_model_can_select_other_legal_edge_with_identical_native_answer():
    model = initial_model()
    model.edge.bias[ord('c')] = math.log(4)
    # Prove the address comes from model scores, not host selection by native count.
    with Live(PREFIX) as live, Archive(WORLDS[0]) as a:
        d = Decoder(model,live,a); m = Meter(trace=True); p,_ = d.predict(m)
        assert [bytes.fromhex(r['key']) for r in m.requests if r['source']=='archive'] == [PREFIX,b'retry_c']
        assert m.views[-1]['op']=='EMIT' and d.record is None  # selected child END-only
        close(p.numpy(),oracle(PREFIX,WORLDS[0]))  # did not vote a byte early


def test_gradient_reaches_real_tensor_writer_without_address_label_or_update():
    model = initial_model()
    with Live(PREFIX) as live, Archive(WORLDS[0]) as a:
        d=Decoder(model,live,a); meter=Meter(); d.predict(meter)
        d.observe(ord('d'),meter); p,_=d.predict(meter)
        grad, = torch.autograd.grad(-p[ord('e')].log(),model.writer.weight)
        assert torch.isfinite(grad).all() and grad.abs().max() > .01


@torch.inference_mode()
def test_reset_and_remount_are_not_old_record_addresses():
    model = initial_model()
    with Archive(WORLDS[0]) as a, Archive(WORLDS[1]) as b, Live(PREFIX) as live:
        d=Decoder(model,live,a); d.predict(Meter()); old=d.record
        with pytest.raises(ValueError,match='already issued'):
            d.predict(Meter())
        d.observe(ord('d'),Meter()); assert old.matches(live,a)
        assert not old.matches(live,b) and not old.matches(live,None)
        with Live(live.tail) as reset:
            assert not old.matches(reset,a)
            fresh=Decoder(model,reset,a); m=Meter(trace=True); p,_=fresh.predict(m)
            assert fresh.model is model and m.views[0]['op']=='BASE'
            close(p.numpy(),oracle(b'retry_d',WORLDS[0]))
        d.archive = b; m=Meter(trace=True); p,_=d.predict(m)
        assert m.views[0]['op']=='BASE' and not m.views[0]['keys_by_slot']
        close(p.numpy(),oracle(b'retry_d',WORLDS[1]))


@torch.inference_mode()
def test_live_support_backoff_empty_removed_and_end_only():
    model = initial_model()
    for prefix,docs in ((b'',None),(b'totallyZ', None),(b'ry_e..retry_',None),
                        (b'retry_e...retry_',WORLDS[0]),(b'retry_c',WORLDS[0])):
        from contextlib import nullcontext
        with Live(prefix) as live, (Archive(docs) if docs else nullcontext()) as a:
            meter=Meter(trace=True); p,_=Decoder(model,live,a).predict(meter)
            expected=oracle(prefix,docs)
            close(p.numpy(),expected)
            direct,_=Direct(live,a).predict(Meter()); close(direct,expected)
            assert meter.reads<=7 and meter.passes<=3
            if a is None:
                assert not any(r['source']=='archive' for r in meter.requests)
            if prefix == b'ry_e..retry_':
                assert p[ord('e')]>.5  # live support at suffix y_
            if prefix == b'retry_e...retry_':
                assert p[ord('d')]>.49 and p[ord('e')]>.16  # both sources, not pooled counts


@torch.inference_mode()
def test_ties_and_zero_edge_are_not_padding():
    model=initial_model()
    with Archive((b'a\x00\xff',b'a\xff\x1e')) as a, Live(b'a') as live:
        d=Decoder(model,live,a); d.predict(Meter())
        assert [h.key for h in d.handles] == [b'a\0',b'a\xff']
        d.observe(0,Meter()); p,_=d.predict(Meter()); assert int(p.argmax())==255
