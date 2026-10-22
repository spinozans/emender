"""X21E source-named guards: accounting, censoring, alignment and state codecs."""
import json

import numpy as np
import pytest
import torch

from experiments.expressivity_tasks import eval_horizon_ladder as ladder
from experiments.expressivity_tasks import horizon_state_quantization as quant


def columns(depth, survivors):
    return [{'driver_position': i + 1, 'correct': 512 if i + 1 in survivors else 0}
            for i in range(depth)]


def test_alignment_start_suffix_ambiguous_and_unclassified():
    base = columns(4, [1, 2])
    mapped = ladder.alignment(base, columns(8, [1, 2, 3, 5, 6]))
    assert [r['classification'] for r in mapped] == [
        'start-anchored', 'start-anchored', 'unclassified', 'suffix-relative', 'suffix-relative']
    assert all('no designated events' in r['structure_relative'] for r in mapped)
    assert ladder.alignment(base, base)[0]['classification'] == 'ambiguous'


def test_horizon_failure_censoring_and_threshold():
    base = columns(4, [1, 2])
    rungs = {m: {'status': 'complete', 'metrics': {'positions': columns(4*m, [1, 2])}}
             for m in ladder.MULTIPLIERS}
    assert ladder.failure(base, rungs) == {'status': 'right-censored', 'beyond_multiplier': 32}
    rungs[4]['metrics']['positions'][1]['correct'] = 479
    assert ladder.failure(base, rungs) == {'status': 'observed-failure', 'multiplier': 4}
    rungs[1]['status'] = 'infrastructure_censored'
    assert ladder.failure(base, rungs)['status'] == 'infrastructure_censored'
    assert ladder.failure(columns(4, []), rungs)['status'] == 'already-failed-1x'


def test_register_and_generate_require_frozen_registration(tmp_path, monkeypatch):
    monkeypatch.setattr(ladder, 'ROOT', tmp_path)
    with pytest.raises(FileNotFoundError):
        ladder.generate()
    (tmp_path / 'registration.json').write_text('{}')
    with pytest.raises(FileExistsError):
        ladder.register()
    source = tmp_path / 'source.py'
    source.write_text('before')
    (tmp_path / 'registration.json').write_text(json.dumps({'source_sha256': {str(source): ladder.x7.digest(source)}}))
    source.write_text('after')
    with pytest.raises(ValueError, match='hash mismatch'):
        ladder.registration()


def test_metrics_match_canonical_position_instrument():
    pool = ladder.x18.fresh_training_batch(ladder.x18.x12.x10.InvertibleQuadraticTask(61),
        512, 16, np.random.default_rng(10))
    pred = pool[1].copy()
    pred[1, 2] = 3721
    pred[2, 3] = (pred[2, 3] + 1) % 3721
    losses = np.ones((512, 16), dtype=np.float32)
    measured = ladder.metrics(pred, pool[1], losses)
    canonical = ladder.x7.position_counts(pred, pool[1], pool[2], losses)
    assert all(all(a[k] == b[k] for k in a) for a, b in zip(measured['positions'], canonical))
    assert measured['aggregate']['correct'] == 512 * 16 - 2
    assert measured['aggregate']['out_of_range'] == 1


@pytest.mark.parametrize('precision', ('bf16', 'fp8', 'int8', 'int4'))
def test_state_codec_zero_and_per_head_scale(precision):
    state = torch.tensor([[[[0., 0.], [0., 0.]], [[.01, -.02], [1., -1.]]]])
    result = quant.quantize_state(state, precision)
    assert torch.isfinite(result).all()
    assert not result[0, 0].any()
    if precision.startswith('int'):
        assert result[0, 1, 1, 0] == 1.
        assert result[0, 1, 1, 1] == -1.


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA kernel parity')
@pytest.mark.parametrize('precision', ('wrapper-fp32', 'bf16', 'fp8', 'int8', 'int4'))
def test_e97_quantization_every_token_matches_scalar_recurrence(precision):
    torch.manual_seed(12)
    b, t, h, n, nv = 2, 16, 2, 32, 32
    k, q = [torch.randn((b,t,h,n), device='cuda', dtype=torch.bfloat16) * .1 for _ in range(2)]
    v, g = [torch.randn((b,t,h,nv), device='cuda', dtype=torch.bfloat16) * .1 for _ in range(2)]
    e = torch.full_like(k, .6)
    w = torch.full_like(v, .7)
    decay = torch.full((b,t,h), .8, device='cuda', dtype=torch.bfloat16)
    state = torch.zeros((b,h,n,nv), device='cuda')
    expected = []
    with torch.no_grad():
        final, output = quant.e97_recurrence(precision, False, k, v, q, decay, g,
            state, h, erase_gate=e, value_write_gate=w)
        for j in range(t):
            kk, qq, vv = k[:,j].float(), q[:,j].float(), v[:,j].float()
            delta = vv * w[:,j].float() - (state * (kk * e[:,j].float())[...,None]).sum(-2)
            # Match the native kernel's sigmoid tanh identity.
            pre = decay[:,j].float()[...,None,None] * state + kk[...,None] * delta[...,None,:]
            state = quant.quantize_state(2 * torch.sigmoid(2 * pre) - 1, precision)
            expected.append(((state * qq[...,None]).sum(-2) * torch.nn.functional.silu(g[:,j].float())).to(v.dtype))
    if precision == 'wrapper-fp32':
        torch.testing.assert_close(final.float(), state, atol=2e-5, rtol=2e-4)
    else:
        # Fused multiply/add and sigmoid approximations can cross discrete bins;
        # compare numerical error, not bitwise equality after sixteen transitions.
        assert (final.float() - state).abs().mean() < 2e-5
        torch.testing.assert_close(quant.quantize_state(final.float(), precision), final.float())
    torch.testing.assert_close(output, torch.stack(expected, 1), atol=1e-4, rtol=.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA kernel parity')
@pytest.mark.parametrize('precision', ('wrapper-fp32', 'bf16', 'fp8', 'int8', 'int4'))
def test_gdn_quantization_global_head_scale_not_tile_scale(precision):
    torch.manual_seed(13)
    b, t, h, n, nv = 2, 5, 2, 16, 80
    q, k = [torch.randn((b,t,h,n), device='cuda', dtype=torch.bfloat16) * .1 for _ in range(2)]
    v = torch.randn((b,t,h,nv), device='cuda', dtype=torch.bfloat16)
    g = torch.full((b,t,h), -.2, device='cuda')
    beta = torch.full_like(g, .7)
    state = torch.zeros((b,h,n,nv), device='cuda')
    expected = []
    with torch.no_grad():
        output, final = quant.gdn_recurrence(precision, q, k, v, g, beta, output_final_state=True)
        for j in range(t):
            state = state * g[:,j].exp()[...,None,None]
            delta = (v[:,j].float() - (state * k[:,j].float()[...,None]).sum(-2)) * beta[:,j,:,None]
            state = quant.quantize_state(state + k[:,j].float()[...,None] * delta[...,None,:], precision)
            expected.append((state * (q[:,j].float() * n**-.5)[...,None]).sum(-2).to(v.dtype))
    if precision == 'fp8':
        # FMA crosses a few fp8 bin boundaries; the codec remains idempotent.
        assert (final - state).abs().mean() < 2e-4
        torch.testing.assert_close(quant.quantize_state(final, precision), final)
    else:
        torch.testing.assert_close(final, state, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(output, torch.stack(expected, 1), atol=2e-3, rtol=.03)


def test_quantization_patch_restored_on_exception():
    from ndm.triton import e97_sequential
    original = e97_sequential.e97_split_edit_triton_apply
    with pytest.raises(RuntimeError):
        with quant.state_quantization(None, 'e97', 'bf16'):
            assert e97_sequential.e97_split_edit_triton_apply is not original
            raise RuntimeError('infrastructure')
    assert e97_sequential.e97_split_edit_triton_apply is original
