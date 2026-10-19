"""X7 precision-only adapter, T16 safety-only protocol and watcher gates."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import socket

import numpy as np
import pytest
import torch

from experiments.expressivity_tasks import run_invertible_stage7_deepfirst_8m as runner


def test_registered_scratch_t16_recipe_and_precision_scope():
    assert runner.DEPTH == 16 and runner.CEILING == 500000 and runner.EVAL_EVERY == 500
    assert runner.NAMES == ('e97_bf16_seed0', 'e97_fp32_seed0', 'gdn2_fp32_seed0')
    torch.manual_seed(0)
    bf16 = runner.build_model('e97', 'bf16')
    torch.manual_seed(0)
    fp32 = runner.build_model('e97', 'fp32')
    assert all(torch.equal(v, fp32.state_dict()[k]) for k, v in bf16.state_dict().items())
    assert all(l.recurrent_state_precision == 'legacy' for l in bf16.layers)
    assert all(l.recurrent_state_precision == 'fp32' for l in fp32.layers)
    assert all(l.fused_inference for l in (*fp32.layers, *bf16.layers))
    assert fp32.out_proj.weight.shape == (3721, 256)
    assert sum(p.numel() for p in fp32.parameters()) == 9900800
    assert fp32.out_proj.weight is fp32.embed.weight


def test_gdn_adapter_is_instance_local_and_preserves_parameters():
    model = runner.base_model('gdn2', 61)
    gdn = model.layers[0].gdn
    old = gdn.forward.__func__
    state = {k: v.clone() for k, v in model.state_dict().items()}
    runner.install_gdn_adapter(gdn)
    assert gdn.forward.__func__.__code__ is old.__code__
    for name in ('chunk_gated_delta_rule', 'fused_recurrent_gated_delta_rule'):
        assert gdn.forward.__func__.__globals__[name] is runner.fp32_gdn_recurrence
        assert old.__globals__[name] is not runner.fp32_gdn_recurrence
    assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
    assert sum(p.numel() for p in model.parameters()) == 9887056
    assert gdn.allow_neg_eigval


def test_fp32_adapter_matches_independent_naive_reference_and_gradients():
    from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule
    torch.manual_seed(73)
    q, k = [torch.randn(2, 18, 2, 8, requires_grad=True) for _ in range(2)]
    v = torch.randn(2, 18, 2, 12, requires_grad=True)
    beta = torch.randn(2, 18, 2, requires_grad=True)
    g = torch.randn(2, 18, 2, requires_grad=True)
    a = torch.randn(2, requires_grad=True)
    dt = torch.randn(2, requires_grad=True)
    initial = torch.randn(2, 2, 12, 8)
    kwargs = dict(use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                  use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True,
                  state_v_first=True, output_final_state=True, initial_state=initial,
                  A_log=a, dt_bias=dt)
    output, state = runner.fp32_gdn_recurrence(q, k, v, g, beta, **kwargs)
    qn, kn = [x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6) for x in (q, k)]
    ref, rs = naive_recurrent_gated_delta_rule(qn, kn, v, beta.sigmoid() * 2,
        -a.exp() * torch.nn.functional.softplus(g + dt),
        initial_state=initial.transpose(-1, -2), output_final_state=True)
    torch.testing.assert_close(output, ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(state, rs.transpose(-1, -2), rtol=1e-5, atol=1e-6)
    variables = (q, k, v, g, beta, a, dt)
    actual = torch.autograd.grad(output.square().sum() + state.square().sum(), variables, retain_graph=True)
    expected = torch.autograd.grad(ref.square().sum() + rs.square().sum(), variables)
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, rtol=2e-4, atol=2e-5)
    low, final = runner.fp32_gdn_recurrence(q.bfloat16(), k.bfloat16(), v.bfloat16(), g.bfloat16(), beta.bfloat16(), **kwargs)
    assert low.dtype == torch.bfloat16 and final.dtype == torch.float32
    with pytest.raises(ValueError):
        runner.fp32_gdn_recurrence(q, k, v, g, beta, cu_seqlens=torch.tensor([0, 18]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires leased GPU for stock Triton')
@pytest.mark.parametrize('length', [32, 96])
def test_fp32_adapter_stock_chunk_equivalence(length):
    """Full-fp32 stock kernel vs sequential, including gradients and cross-chunk state."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    torch.manual_seed(791 + length)
    q, k = [torch.randn(2, length, 2, 128, device='cuda', requires_grad=True) for _ in range(2)]
    v = torch.randn(2, length, 2, 1104, device='cuda', requires_grad=True)
    g, beta = [torch.randn(2, length, 2, device='cuda', requires_grad=True) for _ in range(2)]
    a = torch.randn(2, device='cuda', requires_grad=True)
    dt = torch.randn(2, device='cuda', requires_grad=True)
    kwargs = dict(use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                  use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True,
                  state_v_first=True, output_final_state=True, A_log=a, dt_bias=dt)
    actual, state = runner.fp32_gdn_recurrence(q, k, v, g, beta, **kwargs)
    expected, stock_state = chunk_gated_delta_rule(q, k, v, g, beta, **kwargs)
    variables = (q, k, v, g, beta, a, dt)
    # Shared random VJP checks every recurrence input, not just output agreement.
    do, ds = torch.randn_like(actual), torch.randn_like(state)
    grads = torch.autograd.grad((actual * do).sum() + (state * ds).sum(), variables, retain_graph=True)
    ref_grads = torch.autograd.grad((expected * do).sum() + (stock_state * ds).sum(), variables)
    # Stock FLA explicitly uses TF32 dots even for fp32 tensors. Supervisor-approved
    # normwise guard, paired with the stricter independent fp32 naive test above.
    comparisons = {}
    for name, x, y in zip(('output', 'state', 'dq', 'dk', 'dv', 'dg', 'dbeta', 'dA_log', 'ddt_bias'),
                          (actual, state, *grads), (expected, stock_state, *ref_grads)):
        error = (x - y).detach()
        comparisons[name] = {'max_abs': error.abs().max().item(),
            'relative_l2': (error.norm() / y.detach().norm().clamp_min(1e-12)).item(),
            'max_abs_over_stock_max': (error.abs().max() / y.detach().abs().max().clamp_min(1e-12)).item()}
        assert comparisons[name]['relative_l2'] <= .01
        assert comparisons[name]['max_abs_over_stock_max'] <= .01
    receipt = {'length': length, 'shape': list(q.shape), 'value_dim': 1104,
        'dtype': 'fp32 input tensors; stock chunk TF32 dots vs sequential IEEE fp32',
        'comparisons': comparisons,
        'tolerance': 'relative L2 <=1% AND max_abs/stock_max <=1% for every output/state/VJP',
        'passed': True}
    print('X7_EQUIVALENCE=' + json.dumps(receipt))
    if os.environ.get('X7_EQUIVALENCE_RECEIPT'):
        path = Path(os.environ['X7_EQUIVALENCE_RECEIPT'])
        rows = json.loads(path.read_text()) if path.exists() else []
        runner.write_json(path, rows + [receipt])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires leased GPU for model smoke')
def test_full_models_finite_and_e97_eval_backend_parity():
    import time
    batch = runner.InvertibleQuadraticTask(61).generate_batch(32, 18, np.random.default_rng(0))
    receipts = []
    for arm, precision in (('e97', 'bf16'), ('e97', 'fp32'), ('gdn2', 'fp32')):
        torch.manual_seed(0)
        model = runner.build_model(arm, precision).cuda()
        times = []
        for _ in range(3):
            model.train()
            model.zero_grad()
            start = time.monotonic()
            logits, y, mask = runner.forward_batch(model, batch)
            loss = torch.nn.functional.cross_entropy(logits[mask].float(), y[mask])
            loss.backward()
            torch.cuda.synchronize()
            times.append(time.monotonic() - start)
        assert torch.isfinite(logits).all()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        model.eval()
        with torch.no_grad():
            logits, y, mask = runner.forward_batch(model, batch)
            assert torch.isfinite(logits).all()
            if arm == 'e97' and precision == 'bf16':
                for layer in model.layers:
                    layer.fused_inference = False
                eager, _, _ = runner.forward_batch(model, batch)
                error = logits.float() - eager.float()
                parity = {'relative_l2': (error.norm() / eager.float().norm()).item(),
                    'max_abs': error.abs().max().item(), 'eager_max': eager.abs().max().item(),
                    'argmax_agreement': (logits.argmax(-1) == eager.argmax(-1)).float().mean().item(),
                    'fused_loss': torch.nn.functional.cross_entropy(logits[mask].float(), y[mask]).item(),
                    'eager_loss': torch.nn.functional.cross_entropy(eager[mask].float(), y[mask]).item(),
                    'scope': 'seed0 initialization, full model, 32x18 identical tokens; not trained-model parity'}
                assert parity['relative_l2'] < .01 and parity['argmax_agreement'] >= .99
                if os.environ.get('X7_EQUIVALENCE_RECEIPT'):
                    runner.write_json(Path(os.environ['X7_EQUIVALENCE_RECEIPT']).with_name('e97_eval_backend_parity.json'), parity)
                print('X7_E97_BACKEND_PARITY=' + json.dumps(parity))
        receipts.append({'arm': arm, 'precision': precision, 'finite': True,
                         'loss': loss.item(), 'forward_backward_seconds': times})
        del model, logits, loss
        torch.cuda.empty_cache()
    print('X7_GPU_SMOKE=' + json.dumps(receipts))


def test_safety_only_run_never_threshold_stops_and_never_ladders(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'CEILING', 2)
    monkeypatch.setattr(runner, 'EVAL_EVERY', 1)
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self: self)
    monkeypatch.setattr(runner, 'prune', lambda *args: None)
    lengths = []
    def forward(model, batch):
        lengths.append(batch[0].shape[1])
        y, mask = torch.from_numpy(batch[1]) * 0, torch.from_numpy(batch[2])
        return model.embed.weight[0, :2].expand(*y.shape, 2), y, mask
    monkeypatch.setattr(runner, 'forward_batch', forward)
    monkeypatch.setattr(runner, 'evaluate', lambda *_: {'aggregate': {'accuracy': 1., 'loss': 0.},
        'positions': [{'driver_position': j, 'correct': 512} for j in range(1, 17)], 'perfect_prefix_depth': 16})
    out = tmp_path / 'run'
    runner.run(argparse.Namespace(arm='e97', precision='bf16', seed=0, out=str(out)))
    assert lengths == [18, 18]  # Only T16 from initialization, despite 100% validation.
    config = json.loads((out / 'config.json').read_text())
    assert not config['ladder'] and not config['carry_supervision'] and not config['stagnation_kill']
    rows = json.loads((out / 'validation.json').read_text())
    assert [r['step'] for r in rows] == [0, 1, 2]
    assert len(rows[0]['positions']) == 16
    assert json.loads((out / 'result.json').read_text())['stop_reason'] == 'safety_ceiling'
    cp = torch.load(rows[0]['checkpoint']['path'], weights_only=False)
    assert cp['step'] == 0 and not cp['optimizer']['state']


def test_per_position_evaluation_reconciles(tmp_path, monkeypatch):
    pool = runner.InvertibleQuadraticTask(61).generate_batch(512, 18, np.random.default_rng(1))
    def forward(model, batch):
        y, mask = torch.from_numpy(batch[1]), torch.from_numpy(batch[2])
        logits = torch.zeros(*y.shape, 3721)
        logits.scatter_(-1, y[..., None], 1.)
        return logits, y, mask
    monkeypatch.setattr(runner, 'forward_batch', forward)
    result = runner.evaluate(torch.nn.Linear(1, 1), pool)
    assert result['perfect_prefix_depth'] == 16
    assert [p['correct'] for p in result['positions']] == [512] * 16
    assert result['aggregate']['correct'] == 8192


def test_three_run_watcher_pending_done_fail(tmp_path):
    spec = importlib.util.spec_from_file_location('x7watch', 'scripts/watch_invertible_stage7_deepfirst.py')
    watcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watcher)
    assert tuple(watcher.NAMES) == runner.NAMES
    (tmp_path / 'markers').mkdir()
    stat = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(') ', 1)[1].split()
    for name in watcher.NAMES:
        receipt = tmp_path / 'receipts' / name
        receipt.mkdir(parents=True)
        runner.write_json(receipt / 'worker.json', {'pid': os.getpid(), 'starttime': stat[19], 'host': socket.gethostname()})
    assert watcher.status(tmp_path)[0] is None
    for name in watcher.NAMES:
        (tmp_path / name).mkdir()
        runner.write_json(tmp_path / name / 'result.json', {})
        runner.write_json(tmp_path / 'markers' / f'{name}.DONE', {'name': name, 'exit_code': 0})
    assert watcher.status(tmp_path)[0] == 0
    (tmp_path / 'markers' / f'{watcher.NAMES[0]}.FAIL').touch()
    assert watcher.status(tmp_path)[0] == 1
