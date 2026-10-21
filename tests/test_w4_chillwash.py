"""W4 source-named checks: exact dose, isolated amendment, trip-and-continue."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import w4_chillwash as control
from scripts import w4_stage_b_chill as stage
from scripts.w4_train_chill import learning_rate


def test_chill_cosine_fixed_endpoints_and_dose():
    rates = [learning_rate(i) for i in range(1, 513)]
    assert rates[0] == pytest.approx(2e-6)
    assert rates[-1] == pytest.approx(2e-8)
    assert all(a > b for a, b in zip(rates, rates[1:]))
    from scripts.w3_train_arm import learning_rate as historical
    assert rates == pytest.approx([historical(i, 'adamw-cosine') / 5 for i in range(1, 513)])
    for invalid in (0, 513):
        with pytest.raises(ValueError):
            learning_rate(invalid)


def test_boundary_collisions_exact_lexical_predicate_and_digests():
    private = ['', 'public earlier', 'secret', 'public earlier', 'é']
    events = 'public earlier public earlier é'
    trips = stage.boundary_collisions('choice-write-file', private, events)
    assert [t['private_field_index'] for t in trips] == [1, 3, 4]
    assert [t['event_occurrences'] for t in trips] == [2, 2, 1]
    for trip in trips:
        content = private[trip['private_field_index']]
        assert trip['case_id'] == 'choice-write-file'
        assert trip['content_sha256'] == hashlib.sha256(content.encode()).hexdigest()
        assert trip['content_utf8_bytes'] == len(content.encode())
        assert content not in json.dumps(trip)
    assert not stage.boundary_collisions('clean', ['', 'missing'], events)


@pytest.mark.parametrize('collisions, expected', [(0, 'a'), (1, 'b'), (14, 'b')])
def test_registered_classification(collisions, expected):
    assert control.classification(14, collisions) == expected


@pytest.mark.parametrize('cases,collisions', [(10, 0), (13, 1), (14, 15), (14, -1)])
def test_no_partial_or_inconsistent_classification(cases, collisions):
    with pytest.raises(ValueError):
        control.classification(cases, collisions)


def test_pin_checked_and_create_once(tmp_path):
    source = tmp_path / 'checkpoint.pt'
    source.write_bytes(b'fixed model')
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    pinned = control.pin(source, tmp_path / 'pin', sha)
    assert pinned.read_bytes() == source.read_bytes()
    receipt = json.loads((pinned.parent / 'pin-receipt.json').read_text())
    assert receipt['pinned_before_any_evaluation'] is True
    with pytest.raises(FileExistsError):
        control.pin(source, tmp_path / 'pin', sha)
    with pytest.raises(ValueError, match='identity mismatch'):
        control.pin(source, tmp_path / 'bad-pin', '0' * 64)


def test_registration_must_be_committed(tmp_path, monkeypatch):
    path = tmp_path / 'registration.json'
    path.write_text(json.dumps({'status': 'registered-before-launch', 'bindings': []}))
    monkeypatch.setattr(control, 'REPO', tmp_path)
    monkeypatch.setattr(control.subprocess, 'check_output', lambda *a, **k: b'other committed bytes')
    with pytest.raises(ValueError, match='committed before launch'):
        control.verify_registration(path)


def test_frozen_w3_untouched_and_grading_identical():
    from scripts.w3_matcheddose_preflight import checked
    preflight = json.loads((control.W3 / 'preflight.json').read_text())
    for binding in preflight['instruments']['bindings']:
        if binding['path'].endswith('eval_e97_pi_native_stage_b.py'):
            checked(binding['path'], binding['sha256'])
    old = ast.parse(Path('scripts/eval_e97_pi_native_stage_b.py').read_text())
    new = ast.parse(Path('scripts/w4_stage_b_chill.py').read_text())
    for name in ('grade', 'generate'):
        old_node = next(n for n in ast.walk(old) if isinstance(n, ast.FunctionDef) and n.name == name)
        new_node = next(n for n in ast.walk(new) if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(old_node) == ast.dump(new_node)
    assert 'private reasoning crossed boundary' in Path('scripts/eval_e97_pi_native_stage_b.py').read_text()
    assert "'gate':" not in Path('scripts/w4_stage_b_chill.py').read_text()


def test_stage_b_collision_case11_continues_through_case14(tmp_path, monkeypatch):
    import ndm.e97
    import scripts.audit_e97_live_actor_capture as audit

    cases = [dict(id=f'case{i}', stage='B', files={}, prompt='do it', family='test', delay_tokens=0)
             for i in range(1, 15)]
    panel = tmp_path / 'panel.json'
    panel.write_text(json.dumps(dict(cases=cases, tools=[], episode_seconds=1)))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(model_visible_tools=[])))
    bound = tmp_path / 'bound'
    bound.write_text('bound')
    plan = dict(panel=str(panel), panel_sha256=stage.sha(panel), case_ids=[c['id'] for c in cases],
                tool_manifest=str(manifest), tool_manifest_sha256=stage.sha(manifest),
                tokenizer_vocabulary_sha256='tokenizer', pi_bin='pi')
    for path_key, sha_key in [('checkpoint','checkpoint_sha256'),
            ('parent_model_source_panel','parent_model_source_panel_sha256'), ('args_json','args_sha256'),
            ('pi_inventory','pi_inventory_sha256'), ('stage_b_tools','stage_b_tools_sha256'),
            ('cli_image','cli_image_sha256')]:
        plan[path_key] = str(bound)
        plan[sha_key] = stage.sha(bound)
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    model = SimpleNamespace(eval=lambda: None, parameters=lambda: [torch.zeros(1, dtype=torch.bfloat16)])
    monkeypatch.setattr(ndm.e97, 'load_e97_checkpoint', lambda *a, **k: SimpleNamespace(model=model))
    monkeypatch.setattr(torch.cuda, 'set_device', lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 1)
    monkeypatch.setattr(audit, 'fingerprint', lambda model: 'unchanged')
    monkeypatch.setattr(audit, 'runtime', lambda *a: {})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2')
    monkeypatch.setattr(stage, 'PACKAGES', {})
    monkeypatch.setattr(stage.tiktoken, 'get_encoding', lambda *a: None)
    monkeypatch.setattr(stage, 'vocabulary', lambda *a: (None, 'tokenizer'))
    monkeypatch.setattr(stage, 'snapshot', lambda *a: {})
    monkeypatch.setattr(stage, 'NativePiToolBridge', lambda *a: SimpleNamespace(
        generations=[], reason='finished', final='done', failed=False, closed=True, close_verified=True,
        history=[], episode=SimpleNamespace(text=lambda: '', source_messages=lambda: [])))
    seen = []
    def transport(bridge, directory, **kwargs):
        seen.append(directory.parent.name)
        directory.mkdir()
        (directory / 'pi-events-private.jsonl').write_text('reused public text' if len(seen) == 11 else '')
        return dict(model_failure_verified=False)
    monkeypatch.setattr(stage, 'serve_pi_native_tools', transport)
    monkeypatch.setattr(stage, 'actions', lambda *a: [dict(reasoning_content='reused public text',
                                                        name='finish', arguments={})])
    monkeypatch.setattr(stage, 'grade', lambda *a: dict(success=False,
        checks=dict(valid_first_frame=True, first_action_correct=False)))
    out = tmp_path / 'read'
    stage.run(SimpleNamespace(plan=plan_path, plan_sha=stage.sha(plan_path), output=out))
    summary = json.loads((out / 'summary.json').read_text())
    assert seen == [c['id'] for c in cases]
    assert summary['boundary_collision_cases'] == 1
    assert summary['rows'][10]['boundary_collision'] is True
    assert summary['rows'][13]['id'] == 'case14'
    assert len((out / 'boundary-trips.jsonl').read_text().splitlines()) == 1
    assert 'gate' not in summary
    assert summary['automatic_retries'] == 0


def test_runner_one_card_never_x12_and_no_execution_or_repetitions():
    runner = Path('scripts/w4_run_chillwash.sh').read_text()
    assert 'GPU_LEASE_VISIBLE=2,3,4,5,6,7' in runner
    assert 'acquire 1 --no-wait' in runner
    assert 'for repetition' not in runner
    assert 'torch.distributed.run' not in runner
    assert 'ppid' in runner and "identity['ppid']!=1" in runner
    assert runner.index('scripts.w4_chillwash prepare') < runner.index('scripts.w4_stage_b_chill freeze')
