"""Independent X5 receipt-audit rejection tests, using tiny checkpoint payloads."""
import json

import numpy as np
import pytest

from experiments.expressivity_tasks import aggregate_invertible_stage5_curriculum_8m as agg


@pytest.fixture
def audited_run(tmp_path):
    row = {'name': 'e97_seed0', 'arm': 'e97', 'seed': 0}
    out = tmp_path / row['name']
    out.mkdir()
    receipt = tmp_path / 'receipts' / row['name']
    receipt.mkdir(parents=True)
    (receipt / 'exit.txt').write_text('exit_code=0\n')
    config = {'arm': 'e97', 'seed': 0, 'trainable_parameters': 9900800,
              'depths': list(agg.DEPTHS), 'stage_budget': 10000, 'threshold': .9,
              'eval_every': 500, 'lr': .001, 'batch': 32, 'betas': [.9, .95],
              'weight_decay': .01, 'optimizer': 'AdamWScheduleFree', 'warmup': 0,
              'clip': 1., 'initialization': 'from scratch; no checkpoint loaded'}
    agg.write_json(out / 'config.json', config)
    pools = {}
    for d in agg.DEPTHS:
        pools[f'T{d}_0'] = np.zeros((512, d + 2), dtype=np.int64)
        mask = np.ones((512, d + 2), dtype=bool)
        mask[:, :2] = False
        pools[f'T{d}_2'] = mask
    path = out / 'data.npz'
    np.savez_compressed(path, **pools)
    agg.write_json(out / 'data_receipt.json', {'path': str(path), 'sha256': agg.digest(path)})
    cp_path = out / 'checkpoint.pt'
    cp_path.write_bytes(b'fixture checkpoint')
    cp = {'path': str(cp_path), 'sha256': agg.digest(cp_path)}
    agg.write_json(out / 'checkpoints.json', {str(cp_path): cp})
    metric = {'correct': 1, 'count': 2048, 'accuracy': 1/2048, 'loss': 8.22}
    history = [{'depth': 4, 'stage_step': s, 'decision': 'budget_exhausted' if s == 10000 else 'continue',
                'validation_by_depth': {'4': metric}, 'marginal_reference_nats': agg.MARGINAL,
                'suffix_floor_percent': 100*agg.FLOOR} for s in range(0, 10001, 500)]
    agg.write_json(out / 'validation.json', history)
    with (out / 'training.jsonl').open('w') as f:
        for step in range(1, 10001):
            f.write(json.dumps({'step': step, 'depth': 4, 'stage_step': step, 'loss': 8.22}) + '\n')
    mean = float(np.mean([8.22]*500))
    stage = {'depth': 4, 'entry_step': 0, 'exit_step': 10000, 'steps': 10000,
             'decision': 'budget_exhausted', 'retention': {'4': metric},
             'final_accuracy': metric['accuracy'], 'final_loss': metric['loss'],
             'train_loss_first500': mean, 'train_loss_final500': mean,
             'checkpoint': cp, 'best': {'checkpoint': cp, 'accuracy': metric['accuracy']},
             'steps_to_threshold': None}
    agg.write_json(out / 'stages.json', [stage])
    transitions = [{'event': 'stage_entry', 'optimizer_reset': False, 'checkpoint': cp, 'previous_stage_exit': None},
                   {'event': 'stage_exit', 'checkpoint': cp}]
    (out / 'transitions.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in transitions))
    result = {**row, 'status': 'complete', 'frontier': 0, 'total_steps': 10000,
              'stop_reason': 'budget_exhausted',
              'stopping_depth': 4, 'stopping_accuracy': metric['accuracy'], 'stages': [stage]}
    agg.write_json(out / 'result.json', result)
    return tmp_path, row, out


def test_independent_audit_accepts_exact_receipts(audited_run):
    root, row, _ = audited_run
    result = agg.audit_run(root, row)
    assert result['frontier'] == 0 and result['retained_checkpoints_verified'] == 1


@pytest.mark.parametrize('fault', ['frontier', 'checkpoint', 'missing_panel', 'skipped_crossing', 'optimizer_reset', 'update_gap', 'missing_exit'])
def test_independent_audit_rejects_receipt_corruption(audited_run, fault):
    root, row, out = audited_run
    if fault == 'frontier':
        r = agg.load(out / 'result.json')
        r['frontier'] = 4
        agg.write_json(out / 'result.json', r)
    elif fault == 'checkpoint':
        (out / 'checkpoint.pt').write_bytes(b'corrupt')
    elif fault in ('missing_panel', 'skipped_crossing'):
        h = agg.load(out / 'validation.json')
        if fault == 'missing_panel':
            h[1]['validation_by_depth'] = {}
        else:
            h[1]['validation_by_depth']['4'].update(correct=1844, accuracy=1844/2048)
        agg.write_json(out / 'validation.json', h)
    elif fault == 'optimizer_reset':
        p = out / 'transitions.jsonl'
        p.write_text(p.read_text().replace('"optimizer_reset": false', '"optimizer_reset": true'))
    elif fault == 'update_gap':
        p = out / 'training.jsonl'
        p.write_text('\n'.join(p.read_text().splitlines()[1:])+'\n')
    else:
        (root / 'receipts' / row['name'] / 'exit.txt').unlink()
    with pytest.raises((AssertionError, FileNotFoundError)):
        agg.audit_run(root, row)


def test_report_is_explicit_about_unattempted_depths_and_nonmechanistic_scope(audited_run):
    _, _, out = audited_run
    text = agg.report({'runs': [agg.load(out / 'result.json')]})
    assert 'none (<4)' in text
    assert 'unattempted' in text and 'intrinsic architecture capability' in text
    assert '1/3721' in text and '8.221748' in text
    assert 'no architectural superiority claim' in text


def test_checkpoint_payload_audit_checks_optimizer_clocks_and_exact_carry(tmp_path):
    import copy
    import torch
    transitions = []
    state = {'model': {'embed.weight': torch.zeros(1, 2), 'out_proj.weight': torch.zeros(1, 2)},
             'optimizer': {'param_groups': [{'k': 0, 'train_mode': False}], 'state': {}},
             'numpy_rng': np.random.default_rng(0).bit_generator.state,
             'torch_rng': torch.get_rng_state(), 'cuda_rng': [torch.get_rng_state()],
             'step': 0, 'depth': 4, 'stage_step': 0, 'weights': 'ScheduleFree eval x'}
    payloads = []
    for depth, local_step, total_step in [(4, 0, 0), (4, 500, 500), (8, 0, 500), (8, 500, 1000)]:
        raw = copy.deepcopy(state)
        raw.update(depth=depth, stage_step=local_step, step=total_step)
        raw['optimizer']['param_groups'][0]['k'] = total_step
        raw['optimizer']['state'] = {0: {'z': torch.zeros(1)}} if total_step else {}
        path = tmp_path / f'{depth}_{local_step}.pt'
        torch.save(raw, path)
        cp = {'path': str(path), 'depth': depth, 'stage_step': local_step, 'step': total_step}
        transitions.append({'checkpoint': cp})
        payloads.append((path, raw))
    (tmp_path / 'transitions.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in transitions))
    assert agg.audit_checkpoint_payloads(tmp_path)
    path, corrupt = payloads[2]
    corrupt['model']['embed.weight'][0, 0] = 1
    corrupt['model']['out_proj.weight'][0, 0] = 1
    torch.save(corrupt, path)
    with pytest.raises(AssertionError, match='discontinuous model'):
        agg.audit_checkpoint_payloads(tmp_path)
    corrupt['model']['embed.weight'].zero_()
    corrupt['model']['out_proj.weight'].zero_()
    corrupt['optimizer']['param_groups'][0]['k'] = 0
    torch.save(corrupt, path)
    with pytest.raises(AssertionError):
        agg.audit_checkpoint_payloads(tmp_path)
