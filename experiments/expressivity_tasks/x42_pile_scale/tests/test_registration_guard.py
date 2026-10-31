import json
import subprocess
import pytest
from scripts.registration_guard import diagnose, digest, precommit, rebind


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, 'init', '-q')
    git(tmp_path, 'config', 'user.email', 'test@example.org')
    git(tmp_path, 'config', 'user.name', 'Test')
    (tmp_path/'source.py').write_text('original\n')
    (tmp_path/'registration.json').write_text(json.dumps({'sources': {
        'source.py': digest(b'original\n')}, 'protocol': 'immutable'}))
    git(tmp_path, 'add', '.')
    git(tmp_path, 'commit', '-qm', 'original binding')
    return tmp_path


def test_matches(repo):
    assert diagnose('registration.json', repo)['matches']
    assert diagnose(repo/'registration.json', repo)['matches']


def test_stale_names_commits_and_recipe(repo):
    (repo/'source.py').write_text('promoted\n')
    git(repo, 'add', 'source.py')
    git(repo, 'commit', '-qm', 'part 22 promotion')
    report = diagnose('registration.json', repo)
    assert report['sources'][0]['status'] == 'REGISTRATION STALE vs HEAD'
    assert 'part 22 promotion' in report['sources'][0]['commits'][0]
    assert 'rebind registration.json --receipt' in report['rebind_command']
    rebind('registration.json', repo/'receipt.json', 'part 22', repo)
    assert diagnose('registration.json', repo)['matches']
    assert json.loads((repo/'registration.json').read_text())['protocol']=='immutable'


def test_dirty(repo):
    (repo/'source.py').write_text('local edit\n')
    assert diagnose('registration.json', repo)['sources'][0]['status']=='WORKING TREE DIRTY vs HEAD'
    with pytest.raises(ValueError):
        rebind('registration.json', repo/'receipt.json', 'bad', repo)


def test_tamper(repo):
    (repo/'registration.json').write_text(json.dumps({'sources': {'source.py':'0'*64}}))
    assert diagnose('registration.json', repo)['sources'][0]['status']=='TAMPER SUSPECT'


def test_hook_uses_index_and_requires_matching_rebind(repo):
    (repo/'source.py').write_text('new\n')
    git(repo, 'add', 'source.py')
    assert len(precommit(repo))==1
    (repo/'registration.json').write_text(json.dumps({'sources': {'source.py':digest(b'new\n')}}))
    assert len(precommit(repo))==1  # Unstaged rebind is insufficient.
    git(repo, 'add', 'registration.json')
    assert precommit(repo)==[]


def test_deleted_source_warns(repo):
    git(repo, 'rm', 'source.py')
    assert len(precommit(repo))==1


def test_anchor_preserves_transformer_and_rejects_historical_pure(tmp_path):
    from experiments.expressivity_tasks.x42_pile_scale.velocity_anchor_resume import complete
    row = dict(status='completed', score_eligible=True, train_seconds=2701,
               train_budget_seconds=2700, batch=64, context=2048, eval_windows=24, score=4.)
    transformer = tmp_path/'transformer'
    transformer.mkdir()
    result = transformer/'result.json'
    result.write_text(json.dumps(dict(row, arm='transformer')))
    (transformer/'length_sweep.json').write_text('{"status":"completed"}')
    historical = tmp_path/'c06_result.json'
    historical.write_text(json.dumps(row))
    manifest = {arm: dict(path=str(path), sha256=digest(path.read_bytes()))
                for arm, path in [('transformer', result), ('pure', historical)]}
    (tmp_path/'completed_rows.json').write_text(json.dumps(manifest))
    assert complete('transformer', tmp_path)
    assert not complete('pure', tmp_path)
    assert not complete('gdn2', tmp_path)
    result.write_text('{}')
    with pytest.raises(ValueError, match='receipt changed'):
        complete('transformer', tmp_path)


def test_external_repository_source(repo, tmp_path):
    external = tmp_path/'external'
    external.mkdir()
    git(external, 'init', '-q')
    git(external, 'config', 'user.email', 'test@example.org')
    git(external, 'config', 'user.name', 'Test')
    source = external/'kernel.py'
    source.write_text('external\n')
    git(external, 'add', '.')
    git(external, 'commit', '-qm', 'upstream kernel')
    (repo/'registration.json').write_text(json.dumps({'sources': {str(source): digest(source.read_bytes())}}))
    assert diagnose('registration.json', repo)['matches']
    source.write_text('external promoted\n')
    git(external, 'commit', '-qam', 'external promotion')
    assert diagnose('registration.json', repo)['sources'][0]['status']=='REGISTRATION STALE vs HEAD'


def test_resume_amendment_rejects_identity_change(tmp_path, monkeypatch):
    from experiments.expressivity_tasks.x42_pile_scale import velocity_parent as mod
    root = tmp_path/'root'
    root.mkdir()
    out = tmp_path/'run'
    out.mkdir()
    (root/'velocity_regime.json').write_text('{}')
    (root/'velocity_registration.json').write_text('{}')
    (out/'smoke.json').write_text(json.dumps(dict(batch=64, context=2048, regime_sha256='old')))
    (out/'generation.json').write_text(json.dumps(dict(registration_hash=digest(b'{}'))))
    amendment = dict(status='attended_source_rebind', smoke_sha256=digest((out/'smoke.json').read_bytes()),
                     regime_sha256=digest(b'{}'), registration_sha256=digest(b'{}'),
                     retained_banked_candidates=7, cause_commit='promotion', residual_risks=['historical throughput'], evidence={})
    path=out/'resume_amendment.json'
    path.write_text(json.dumps(amendment))
    monkeypatch.setattr(mod, 'ROOT', root)
    monkeypatch.chdir(tmp_path)
    import experiments.expressivity_tasks.x42_pile_scale.velocity_train as train
    monkeypatch.setattr(train, 'subprocess_bytes', lambda commit, name: (tmp_path/name).read_bytes())
    mod.validate_smoke(out)
    assert not json.loads((out/'smoke_compatibility.json').read_text())['current_throughput_certified']
    amendment['registration_sha256']='wrong'
    path.write_text(json.dumps(amendment))
    with pytest.raises(RuntimeError, match='identity mismatch'):
        mod.validate_smoke(out)


def test_missing_external_history_is_suspect(repo):
    source='/nonexistent-registration-test-repo/source.py'
    (repo/'registration.json').write_text(json.dumps({'sources': {source: '0'*64}}))
    assert diagnose('registration.json', repo)['sources'][0]['status']=='TAMPER SUSPECT'
