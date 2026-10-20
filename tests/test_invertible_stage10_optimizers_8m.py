"""Source-named X10 routing/NS/RMS/state tests; must pass before task launch."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

from experiments.expressivity_tasks import hybrid_muon as h
from experiments.expressivity_tasks import run_invertible_stage10_optimizers_8m as r


def tiny():
    model = nn.Module()
    model.embed = nn.Embedding(7, 4)
    model.out_proj = nn.Linear(4, 7, bias=False)
    model.out_proj.weight = model.embed.weight
    layer = nn.Module()
    layer.qkv_proj = nn.Linear(4, 8, bias=False)
    layer.value_residual = nn.Parameter(torch.ones(2, 4))
    layer.bias2d = nn.Parameter(torch.ones(2, 4))
    layer.norm = nn.LayerNorm(4)
    model.layers = nn.ModuleList([layer])
    return model


def test_routing_exhaustive_tied_and_semantic_not_ndim():
    model = tiny()
    routes = h.parameter_routes(model)
    assert len(routes) == len(list(model.parameters()))
    assert sum(row['numel'] for row in routes) == sum(p.numel() for p in model.parameters())
    assert [row['name'] for row in routes if row['owner'] == 'muon'] == ['layers.0.qkv_proj.weight']
    assert routes[0]['aliases'] == ['embed.weight', 'out_proj.weight']
    assert all(row['owner'] == 'adamw' for row in routes if row['name'].endswith(('value_residual', 'bias2d', 'bias')))
    model.other = nn.Parameter(model.embed.weight.data)
    with pytest.raises(ValueError, match='share storage'):
        h.parameter_routes(model)


def test_full_model_routes_and_exact_matched_initialization():
    torch.manual_seed(0)
    model = r.build_model('muon')
    torch.manual_seed(0)
    ref = r.x7.build_model('e97', 'bf16')
    assert all(torch.equal(value, ref.state_dict()[name]) for name, value in model.state_dict().items())
    routes = h.parameter_routes(model)
    assert sum(row['numel'] for row in routes) == 9900800
    assert len([row for row in routes if row['owner'] == 'muon']) == 36
    assert len([row for row in routes if row['no_weight_decay_tag_ignored']]) == 8
    assert all(row['weight_decay'] == .01 and row['lr_multiplier'] == 1 for row in routes)
    assert routes[0]['aliases'] == ['embed.weight', 'out_proj.weight']
    assert all(layer.fused_inference and layer.recurrent_state_precision == 'legacy' for layer in model.layers)


@pytest.mark.parametrize('shape', [(4, 4), (9, 3), (3, 9)])
def test_newton_schulz_against_singular_value_polynomial_reference(shape):
    # Independent SVD reference: each singular value follows the quintic.
    generator = torch.Generator().manual_seed(12)
    matrix = torch.randn(shape, generator=generator)
    u, s, vh = np.linalg.svd(matrix.numpy().astype(np.float64), full_matrices=False)
    s /= np.linalg.norm(matrix.numpy().astype(np.float64)) + 1e-7
    for _ in range(5):
        s = 3.4445 * s - 4.7750 * s**3 + 2.0315 * s**5
    ref = torch.from_numpy((u * s) @ vh).float()
    actual = h.newton_schulz(matrix)
    torch.testing.assert_close(actual, ref, atol=3e-5, rtol=3e-5)
    scaled = h.scaled_update(matrix)
    torch.testing.assert_close(scaled, ref * (.2 * math.sqrt(max(shape))), atol=3e-5, rtol=3e-5)
    assert scaled.square().mean().sqrt().item() == pytest.approx((ref * (.2 * math.sqrt(max(shape)))).square().mean().sqrt().item(), rel=3e-5)
    torch.testing.assert_close(h.scaled_update(matrix.T), scaled.T, atol=3e-5, rtol=3e-5)
    assert torch.count_nonzero(h.newton_schulz(torch.zeros(shape))) == 0
    assert torch.count_nonzero(h.scaled_update(torch.zeros(shape))) == 0


def test_ns_not_claimed_exact_polar_and_bad_inputs_rejected():
    x = h.newton_schulz(torch.eye(4))
    assert torch.linalg.vector_norm(x.T @ x - torch.eye(4)).item() > .01
    with pytest.raises(ValueError):
        h.newton_schulz(torch.ones(3))
    with pytest.raises(FloatingPointError):
        h.newton_schulz(torch.full((3, 3), float('nan')))


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_update_rms_decay_zero_grad_finite_and_aux_adamw_parity(dtype):
    torch.manual_seed(3)
    model = tiny().to(dtype)
    opt = h.HybridMuon(model, lr=.001)
    ref = nn.Parameter(model.embed.weight.detach().clone())
    adam = torch.optim.AdamW([ref], lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=.01)
    matrix = model.layers[0].qkv_proj.weight
    grad = torch.randn_like(matrix)
    before = matrix.detach().clone()
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    matrix.grad = grad
    ref.grad = model.embed.weight.grad.clone()
    opt.step()
    adam.step()
    torch.testing.assert_close(model.embed.weight, ref, rtol=0, atol=0)
    direction = grad.float().lerp(grad.float() * .05, .95)
    update = h.scaled_update(direction)
    expected = before.clone().mul_(1 - .001 * .01).add_(update.to(dtype), alpha=-.001)
    torch.testing.assert_close(matrix, expected, rtol=0, atol=0)
    assert opt.last_metrics['muon_update_rms'] == pytest.approx(update.square().mean().sqrt().item(), rel=1e-6)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    # A fresh zero-gradient optimizer isolates decay, no stale momentum.
    opt = h.HybridMuon(model, lr=.001)
    before = [p.detach().clone() for p in model.parameters()]
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    for p, old in zip(model.parameters(), before):
        torch.testing.assert_close(p, old * (1 - .001 * .01), rtol=0, atol=0)


def test_two_updates_nesterov_reference_scheduler_and_state_parity(tmp_path):
    torch.manual_seed(7)
    model = tiny()
    opt = h.HybridMuon(model)
    matrix = model.layers[0].qkv_proj.weight
    buf = torch.zeros_like(matrix)
    for step in (1, 5000, 252500, 500000):
        r.rates(opt, 'muon', step)
        assert [g['lr'] for g in opt.param_groups] == [r.learning_rate(step)] * 2
        grad = torch.randn_like(matrix)
        matrix.grad = grad
        before = matrix.detach().clone()
        buf = .95 * buf + .05 * grad
        expected = before * (1 - r.learning_rate(step) * .01) - r.learning_rate(step) * h.scaled_update(.05 * grad + .95 * buf)
        opt.step()
        torch.testing.assert_close(matrix, expected, rtol=1e-5, atol=1e-7)
        assert opt.aux.param_groups[0]['lr'] == r.learning_rate(step)
    assert r.learning_rate(500000) == 1e-5
    saved = copy.deepcopy(opt.state_dict())
    restored = copy.deepcopy(model)
    opt2 = h.HybridMuon(restored)
    opt2.load_state_dict(saved)
    for p, q in zip(model.parameters(), restored.parameters()):
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
    opt.step()
    opt2.step()
    for p, q in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    with pytest.raises(ValueError):
        opt2.load_state_dict({'family': 'adamw'})
    wrong_routes = copy.deepcopy(saved)
    wrong_routes['routes'][0]['name'] = 'wrong'
    with pytest.raises(ValueError):
        opt2.load_state_dict(wrong_routes)


def test_fresh_process_reload_next_update(tmp_path):
    # Entire E97 state, freshly constructed model/optimizer in a new process.
    torch.manual_seed(0)
    model = r.build_model('muon')
    opt = h.HybridMuon(model, lr=.001)
    for p in model.parameters():
        p.grad = torch.full_like(p, .01)
    opt.step()
    path = tmp_path / 'restart.pt'
    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict()}, path)
    for p in model.parameters():
        p.grad = torch.full_like(p, .02)
    opt.step()
    code = '''import sys, torch
from experiments.expressivity_tasks import run_invertible_stage10_optimizers_8m as r
from experiments.expressivity_tasks.hybrid_muon import HybridMuon
torch.set_num_threads(2)
s=torch.load(sys.argv[1],weights_only=False)
m=r.build_model('muon'); m.load_state_dict(s['model']); o=HybridMuon(m); o.load_state_dict(s['optimizer'])
for p in m.parameters(): p.grad=torch.full_like(p,.02)
o.step(); torch.save(m.state_dict(),sys.argv[2])
'''
    target = tmp_path / 'next.pt'
    subprocess.run([sys.executable, '-c', code, str(path), str(target)], check=True, timeout=90)
    actual = torch.load(target, weights_only=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, actual[name], rtol=0, atol=0)


def test_x7b_stock_bf16_not_fp32_adapter():
    torch.manual_seed(0)
    model = r.build_model('gdn2_uniform')
    torch.manual_seed(0)
    ref = r.stock_model('gdn2', 61)
    assert sum(p.numel() for p in model.parameters()) == 9887056
    assert all(torch.equal(v, ref.state_dict()[k]) for k, v in model.state_dict().items())
    for layer in model.layers:
        assert layer.gdn.forward.__func__.__globals__['chunk_gated_delta_rule'] is not r.x7.fp32_gdn_recurrence
    opt = r.make_optimizer(model, 'gdn2_uniform')
    assert opt.param_groups[0]['betas'] == (.9, .95)
    assert opt.param_groups[0]['lr'] == .001
    assert opt.param_groups[0]['warmup_steps'] == 0


def test_adamw_sf_controls_exact_recipes():
    model = tiny()
    adam = r.make_optimizer(model, 'adamw')
    assert type(adam) is torch.optim.AdamW
    assert adam.param_groups[0]['betas'] == (.9, .999)
    for arm in ('adamw', 'sf'):
        opt = r.make_optimizer(model, arm)
        assert opt.param_groups[0]['weight_decay'] == .01
        assert opt.param_groups[0]['eps'] == 1e-8
    assert 'eval-x' in r.point('sf') and r.point('muon') == r.point('adamw')


def test_retention_unconditionally_bounded_and_hashed(tmp_path):
    cps = {}
    for step in range(5):
        p = tmp_path / f'{step}.pt'
        p.write_text(str(step))
        cps[str(p)] = {'path': str(p), 'sha256': r.x7.digest(p), 'step': step}
    keep = {str(tmp_path / f'{step}.pt') for step in (0, 3, 4)}
    r.prune(tmp_path, cps, keep)
    assert {str(p) for p in tmp_path.glob('*.pt')} == keep
    manifest = json.loads((tmp_path / 'pruning.jsonl').read_text())
    assert len(manifest['removed']) == 2
    assert all(r.x7.digest(cp['path']) == cp['sha256'] for cp in manifest['retained'])


@pytest.mark.parametrize('stage', ['10_optimizers', '7b_stock'])
def test_watchers_identity_pending_done_fail(tmp_path, stage):
    spec = importlib.util.spec_from_file_location('watch', f'scripts/watch_invertible_stage{stage}.py')
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    for name in w.NAMES:
        r.x7.write_json(tmp_path / 'receipts' / name / 'worker.json', {'pid': os.getpid(), 'host': socket.gethostname(),
            'starttime': Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(') ', 1)[1].split()[19]})
    assert w.status(tmp_path)[0] is None
    for name in w.NAMES:
        r.x7.write_json(tmp_path / name / 'result.json', {})
        r.x7.write_json(tmp_path / 'markers' / f'{name}.DONE', {'name': name, 'exit_code': 0})
    assert w.status(tmp_path)[0] == 0
    (tmp_path / 'markers' / f'{w.NAMES[0]}.FAIL').touch()
    assert w.status(tmp_path)[0] == 1


@pytest.mark.parametrize('arm', ['muon', 'adamw', 'sf', 'gdn2_uniform'])
def test_runner_ceiling_point_semantics_and_frozen_pool(tmp_path, monkeypatch, arm):
    pool = r.x7.frozen_pools(0, 16, 61)['val']
    import hashlib
    registration = {'authorized_runs': [f'{arm}_seed0'], 'pool_array_sha256': {'0': [hashlib.sha256(a.tobytes()).hexdigest() for a in pool]},
                    'parameter_routes': h.parameter_routes(r.build_model('muon')),
                    'resource_bounds': {arm: {'wall_seconds': 3600, 'peak_allocated_bytes': 8 * 1024**3}}}
    root = tmp_path / 'root'
    r.x7.write_json(root / 'registration.json', registration)
    monkeypatch.setitem(r.ROOTS, arm, root)
    monkeypatch.setattr(r, 'verify_registration', lambda _: registration)
    monkeypatch.setattr(r, 'CEILING', 3)
    monkeypatch.setattr(r, 'EVAL_EVERY', 1)
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self: self)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    lengths = []
    def forward(model, batch):
        lengths.append(batch[0].shape[1])
        y, mask = torch.from_numpy(batch[1]) * 0, torch.from_numpy(batch[2])
        return model.embed.weight[0, :2].expand(*y.shape, 2), y, mask
    monkeypatch.setattr(r.x7, 'forward_batch', forward)
    def evaluate(model, pool):
        return {'aggregate': {'accuracy': 1., 'loss': 4. - len(lengths)},
                'positions': [{'driver_position': j, 'correct': 512} for j in range(1, 17)], 'perfect_prefix_depth': 16}
    monkeypatch.setattr(r.x7, 'evaluate', evaluate)
    from argparse import Namespace
    r.run(Namespace(arm=arm, seed=0))
    assert lengths == [18, 18, 18]  # Perfect mocked metrics must NOT stop training.
    out = root / f'{arm}_seed0'
    result = json.loads((out / 'result.json').read_text())
    assert result['total_steps'] == 3 and result['stop_reason'] == 'safety_ceiling'
    validations = json.loads((out / 'validation.json').read_text())
    assert [v['step'] for v in validations] == [0, 1, 2, 3]
    final = torch.load(validations[-1]['checkpoint']['path'], weights_only=False)
    assert final['weights'] == r.point(arm)
    assert len(list(out.glob('*.pt'))) <= 3
    rows = [json.loads(line) for line in (out / 'training.jsonl').read_text().splitlines()]
    assert all(np.isfinite(row['loss']) and row['full_update_rms'] is not None for row in rows)
    if arm in ('sf', 'gdn2_uniform'):
        assert final['optimizer']['param_groups'][0]['train_mode'] is False
    else:
        assert 'ordinary live' in final['weights']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='exclusive target GPU required')
@pytest.mark.parametrize('arm', ['muon', 'adamw', 'sf', 'gdn2_uniform'])
def test_target_gpu_full_position_evaluation_finite(arm):
    torch.manual_seed(99)
    model = r.build_model(arm).cuda()
    # Disposable synthetic seed99 pool, not the confirmatory frozen pools.
    pool = r.InvertibleQuadraticTask(61).generate_batch(512, 18, np.random.default_rng(99))
    opt = r.make_optimizer(model, arm)
    metrics = r.train_update(model, opt, arm, tuple(a[:32] for a in pool), 1)
    assert math.isfinite(metrics['loss'])
    if arm in ('sf', 'gdn2_uniform'):
        opt.eval()
    positions = r.x7.evaluate(model, pool)
    assert len(positions['positions']) == 16
    assert sum(p['correct'] for p in positions['positions']) == positions['aggregate']['correct']
    assert math.isfinite(positions['aggregate']['loss'])
