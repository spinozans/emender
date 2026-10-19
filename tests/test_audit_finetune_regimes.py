"""Receipt-only audit: no training or checkpoint deserialization."""
import json

from scripts import audit_finetune_regimes as audit


def test_sample_schedule_is_not_lr_annealing(monkeypatch):
    monkeypatch.setattr(audit, 'historical_source', lambda *_: 'AdamWScheduleFree')
    klass, regime, _ = audit.classify(
        {'source_commit': 'abc1234', 'schedule': {'path': 'expected-schedule.json'}},
        'train_e97_4b_pi_sft.py --warmup-steps 8', {'start_update': 0, 'parent_checkpoint': 'x'})
    assert klass == 'AdamWScheduleFree'
    assert regime == 'constant-lr-SF'


def test_explicit_adamw_source_schedule(monkeypatch):
    monkeypatch.setattr(audit, 'historical_source', lambda *_: 'lr_scale_at(step)')
    assert audit.classify({'optimizer': 'adamw'}, '', {})[1] == 'annealed'


def test_unknown_snapshot_is_not_silently_sf(monkeypatch):
    monkeypatch.setattr(audit, 'historical_source', lambda *_: '')
    assert audit.classify({}, '', {'start_update': 0, 'parent_checkpoint': 'x'})[1] == 'unknown'
    assert audit.classify({}, '', {'sft_precision': {'optimizer': 'bf16-sr-candidate'}})[1] == 'constant-lr-SF'


def test_resume_dose_counts_only_observed_updates_and_deduplicates(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'historical_source', lambda *_: 'AdamWScheduleFree')
    root = tmp_path / 'runs' / 'u256'
    (root / 'identity').mkdir(parents=True)
    (root / 'logs').mkdir()
    (root / 'identity' / 'launch.json').write_text(json.dumps({'steps': 256}))
    rows = [{'event': 'start', 'start_update': 64, 'parent_checkpoint': 'x', 'lr': 1e-5},
            {'event': 'step', 'update': 65, 'global_tokens': 100, 'global_targets': 7, 'loss': 2},
            {'event': 'step', 'update': 65, 'global_tokens': 100, 'global_targets': 7, 'loss': 2},
            {'event': 'step', 'update': 66, 'global_tokens': 120, 'global_targets': 8, 'loss': 1}]
    (root / 'logs' / 'training.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    (root / 'checkpoints').mkdir()
    (root / 'checkpoints' / 'unsafe.pt').write_bytes(b'not a pickle; never opened')
    clone = tmp_path / 'eval-worktree'
    clone.mkdir()
    (clone / 'args.json').write_text('{"optimizer":"adamw"}')
    data = audit.census([tmp_path])
    assert len(data['runs']) == 1
    r = data['runs'][0]
    assert r['observed_updates'] == 2
    assert r['dose_input_tokens'] == 220
    assert r['dose_target_tokens'] == 15
    assert r['update_range'] == [65, 66]
    assert r['planned_steps'] == 256


def test_online_cycles_do_not_deduplicate_equal_update_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'historical_source', lambda *_: 'AdamWScheduleFree')
    paths = []
    for n in range(2):
        path = tmp_path / 'e97-rl-loop-v1' / 'training' / f'cycle-{n}' / 'log.jsonl'
        path.parent.mkdir(parents=True)
        rows = [{'event': 'start', 'trainer': 'rl_train_step', 'lr': 1e-5},
                {'event': 'step', 'update': 1, 'global_targets': 9, 'loss': 1}]
        path.write_text('\n'.join(map(json.dumps, rows)))
        paths.append(path)
    group, = audit.online_census(paths)
    assert group['observed_updates'] == 2
    assert group['dose_target_tokens'] == 18
    assert group['regimes'] == {'constant-lr-SF': 2}
    assert all(r['snapshot_bytes'] > 0 for r in group['logs'])


def test_scalar_metrics_drops_case_content_but_keeps_scores():
    assert audit.scalar_metrics({'rows': [{'secret': 'prompt'}], 'passed': 3,
                                 'checkpoint_sha256': 'abc', 'loss_quartiles': [2.0, 1.0]}) == {
        'passed': 3, 'checkpoint_sha256': 'abc', 'loss_quartiles': [2.0, 1.0]}
