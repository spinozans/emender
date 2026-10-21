"""X13 audit executes historical paths and rejects token/mask/target corruption."""
import numpy as np
import pytest

from scripts import audit_invertible_stage13_zerocheck as a


def test_cpu_t64_historical_paths_and_independent_scalar_oracle():
    pool = a.x10.x7.frozen_pools(0, 64, 61)['val']
    result = a.equivalence(pool)
    assert result['clean'] and result['differences'] == []
    assert all(result[name] for name in ('same_forward_callable', 'same_pool_callable',
                                         'same_task_class', 'same_padder_callable'))
    assert len(result['examples']) == 8
    for example in result['examples']:
        assert len(example['tokens']) == 66
        assert len(example['stage12_padded_input']) == 80
        assert example['tokens'][2]['driver_position'] == 1
        assert example['tokens'][-1]['driver_position'] == 64


@pytest.mark.parametrize('field,index,value', [(0, 0, 0), (1, 2, -1), (2, 2, False)])
def test_scalar_oracle_rejects_bad_prefix_target_or_mask(field, index, value):
    batch = a.x10.InvertibleQuadraticTask(61).generate_batch(1, 66, np.random.default_rng(0))
    row = [array[0].copy() for array in batch]
    row[field][index] = value
    with pytest.raises(AssertionError):
        a.scalar_decode(*row)


def test_cpu_equivalence_receipts_exact_divergent_tokens(monkeypatch):
    pool = a.x10.x7.frozen_pools(0, 64, 61)['val']
    original = a.cpu_path
    def changed(module, batch):
        result = original(module, batch)
        if module is a.x12:
            result[0][0, 2] += 1
        return result
    monkeypatch.setattr(a, 'cpu_path', changed)
    result = a.equivalence(pool)
    assert not result['clean']
    assert result['differences'][0]['field'] == 'padded_input'
    assert result['differences'][0]['stage12'][0][2] != result['differences'][0]['stage10'][0][2]


def test_replication_reuses_exact_x12_recipe_and_only_authorized_seeds():
    from experiments.expressivity_tasks import run_invertible_stage13_zerocheck_8m as r
    assert r.SEEDS == (1, 2) and r.CEILING == 1000000 and r.EVAL_EVERY == 500
    for name in ('build_model', 'train_update', 'evaluate', 'make_optimizer', 'learning_rate'):
        assert getattr(r, name) is getattr(r.x12, name)


def test_clean_audit_gate_and_receipt_tampering(tmp_path):
    from experiments.expressivity_tasks import run_invertible_stage13_zerocheck_8m as r
    def write(value):
        r.x7.write_json(tmp_path / 'audit/verdict.json', value)
    base = {'verdict': 'BROKEN', 'seed_replication_authorized': False, 'a_p1_p4': [0]*4,
            'b_token_equivalence': False}
    write(base)
    with pytest.raises(ValueError, match='not clean'):
        r.require_clean_audit(tmp_path)
    r.x7.write_json(tmp_path / 'audit/example.json', {'original': True})
    write({**base, 'verdict': 'CLEAN', 'seed_replication_authorized': True,
           'a_p1_p4': [512]*4, 'b_token_equivalence': True,
           'receipt_sha256': {'example.json': r.x7.digest(tmp_path / 'audit/example.json')}})
    assert r.require_clean_audit(tmp_path)['verdict'] == 'CLEAN'
    r.x7.write_json(tmp_path / 'audit/example.json', {'original': False})
    with pytest.raises(ValueError, match='changed audit receipt'):
        r.require_clean_audit(tmp_path)


def test_replication_scratch_seed_pool_schedule_ceiling_and_pruning(tmp_path, monkeypatch):
    import json
    import torch
    from experiments.expressivity_tasks import run_invertible_stage13_zerocheck_8m as r
    seed, name = 2, 'e97_bf16_seed2'
    pool = r.x7.frozen_pools(seed, 64, 61)['val']
    reg = {'authorized_runs': [name], 'pool_array_sha256': {'2': r.array_hashes(pool)}}
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    monkeypatch.setattr(r, 'require_clean_audit', lambda _: {})
    monkeypatch.setattr(r.x10, 'verify_registration', lambda _: reg)
    r.x7.write_json(tmp_path / 'registration.json', reg)
    r.x7.write_json(tmp_path / 'audit/verdict.json', {})
    r.x7.write_json(tmp_path / 'receipts' / name / 'admission.json',
                   {'registration_sha256': r.x7.digest(tmp_path / 'registration.json'),
                    'audit_verdict_sha256': r.x7.digest(tmp_path / 'audit/verdict.json'), 'gpu': 1})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    monkeypatch.setattr(r, 'CEILING', 3)
    monkeypatch.setattr(r, 'EVAL_EVERY', 1)
    model = torch.nn.Linear(1, 2, bias=False)
    model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])
    monkeypatch.setattr(r, 'build_model', lambda _: model)
    monkeypatch.setattr(r.x12, 'PARAMETERS', {'e97': 2})
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self: self)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    batches = []
    def update(model, opt, batch, step):
        assert type(opt) is torch.optim.AdamW
        batches.append(batch)
        return {'loss': 1., 'grad_norm': 1., 'lr': r.learning_rate(step)}
    monkeypatch.setattr(r, 'train_update', update)
    monkeypatch.setattr(r, 'evaluate', lambda *_: {'aggregate': {'accuracy': 1., 'loss': 1.},
        'positions': [{'driver_position': j, 'correct': 512, 'count': 512} for j in range(1, 65)],
        'perfect_prefix_depth': 64})
    r.run(seed)
    expected_rng = np.random.default_rng(seed)
    for batch in batches:
        expected = r.x10.InvertibleQuadraticTask(61).generate_batch(32, 66, expected_rng)
        assert all(np.array_equal(a, b) for a, b in zip(batch, expected))
    out = tmp_path / name
    result = json.loads((out / 'result.json').read_text())
    assert result['total_steps'] == 3 and result['stop_reason'] == 'safety_ceiling'
    assert len(batches) == 3  # Perfect validation is not a stop.
    assert len(list(out.glob('*.pt'))) <= 3
    saved = torch.load(out / 'step_000003.pt', weights_only=False)
    assert saved['config']['seed'] == seed and not saved['optimizer']['state']
    assert saved['optimizer_family'] == 'adamw'
    assert json.loads((out / 'data_receipt.json').read_text())['array_sha256'] == r.array_hashes(pool)


def test_worker_fails_closed_before_lease_on_unclean_audit(tmp_path, monkeypatch):
    import json
    from scripts import invertible_stage13_control as c
    monkeypatch.setattr(c.run, 'ROOT', tmp_path)
    monkeypatch.setattr(c.run, 'require_clean_audit', lambda _: (_ for _ in ()).throw(ValueError('unclean audit')))
    monkeypatch.setattr(c.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(c.subprocess, 'run', lambda *args, **kw: pytest.fail('no broker before audit'))
    assert c.worker(1) == 1
    marker = json.loads((tmp_path / 'markers/e97_bf16_seed1.FAIL').read_text())
    assert 'unclean audit' in marker['error']


def test_watcher_zero_pending_vs_final_and_p1_vs_composition(tmp_path):
    import json
    import os
    from scripts import watch_invertible_stage13_zerocheck as w
    from scripts.invertible_stage10_control import identity
    from experiments.expressivity_tasks import run_invertible_stage13_zerocheck_8m as r
    for name in w.NAMES:
        r.x7.write_json(tmp_path / 'receipts' / name / 'worker.json', identity(os.getpid()))
        row = {'step': 500, 'positions': [{'driver_position': j, 'correct': 0, 'count': 512}
                                         for j in range(1, 65)], 'perfect_prefix_depth': 0}
        r.x7.write_json(tmp_path / name / 'validation.json', [row])
    assert w.status(tmp_path)[0] is None
    assert 'verdict PENDING' in w.growth_report(tmp_path)
    row['positions'][0]['correct'] = 50
    r.x7.write_json(tmp_path / w.NAMES[0] / 'validation.json', [row])
    report = w.growth_report(tmp_path)
    assert 'marginal escape observed=True; composition observed=False' in report
    assert 'seed0 learned-nothing zero was lottery' in report
    assert '| P64 |' in report and '| P1 | 50 | 0 | 50 | 0 | 500 / — |' in report
    for name in w.NAMES:
        row = {'step': 1000000, 'positions': [{'driver_position': j, 'correct': 0, 'count': 512}
                                             for j in range(1, 65)], 'perfect_prefix_depth': 0}
        result = {'name': name, 'status': 'complete', 'total_steps': 1000000,
                  'stop_reason': 'safety_ceiling', 'final': row}
        r.x7.write_json(tmp_path / name / 'result.json', result)
        r.x7.write_json(tmp_path / name / 'validation.json', [row])
        terminal = {'name': name, 'exit_code': 0, 'result_sha256': r.x7.digest(tmp_path / name / 'result.json')}
        r.x7.write_json(tmp_path / 'receipts' / name / 'terminal.json', terminal)
        r.x7.write_json(tmp_path / 'markers' / f'{name}.DONE', terminal)
    assert w.status(tmp_path)[0] == 0
    assert 'replicates as3/3' in w.growth_report(tmp_path)
    (tmp_path / w.NAMES[0] / 'result.json').write_text('{}')
    assert w.status(tmp_path)[0] == 1


def test_worker_owner_bound_lease_heartbeat_terminal_and_release(tmp_path, monkeypatch):
    import json
    import os
    from scripts import invertible_stage13_control as c
    monkeypatch.setattr(c.run, 'ROOT', tmp_path)
    monkeypatch.setattr(c.run, 'require_clean_audit', lambda _: {})
    monkeypatch.setattr(c.run.x10, 'verify_registration', lambda _: {'authorized_runs': c.run.NAMES})
    monkeypatch.setattr(c.signal, 'signal', lambda *args: None)
    name = c.run.NAMES[0]
    c.run.x7.write_json(tmp_path / 'registration.json', {})
    c.run.x7.write_json(tmp_path / 'audit/verdict.json', {})
    c.run.x7.write_json(tmp_path / 'receipts' / name / 'admission.json',
        {'gpu': 3, 'registration_sha256': c.run.x7.digest(tmp_path / 'registration.json'),
         'audit_verdict_sha256': c.run.x7.digest(tmp_path / 'audit/verdict.json')})
    c.run.x7.write_json(tmp_path / name / 'result.json', {'status': 'complete', 'total_steps': 1000000})
    calls = []
    def broker(cmd, **kwargs):
        assert kwargs['env']['GPU_LEASE_PID'] == str(os.getpid())
        assert kwargs['env']['GPU_LEASE_VISIBLE'] == kwargs['env']['CUDA_VISIBLE_DEVICES'] == '3'
        calls.append(cmd[2:])
    class Child:
        pid = os.getpid()
        waits = 0
        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise c.subprocess.TimeoutExpired('trainer', timeout)
            return 0
        def poll(self):
            return 0
    def spawn(cmd, **kwargs):
        assert cmd[-2:] == ['--seed', '1']
        return Child()
    monkeypatch.setattr(c.subprocess, 'run', broker)
    monkeypatch.setattr(c.subprocess, 'Popen', spawn)
    assert c.worker(1) == 0
    assert calls == [['acquire', '1', '--no-wait'], ['heartbeat', '3'], ['release', '3']]
    marker = json.loads((tmp_path / 'markers' / f'{name}.DONE').read_text())
    assert marker == json.loads((tmp_path / 'receipts' / name / 'terminal.json').read_text())
    assert marker['result_sha256'] == c.run.x7.digest(tmp_path / name / 'result.json')
