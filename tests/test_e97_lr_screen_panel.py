import json
import pytest
from scripts.e97_lr_screen_panel import make_panel, score, read_fixture


def test_frozen_panel_is_deterministic_balanced_and_not_training():
    panel = make_panel()
    assert panel == make_panel()
    assert len(panel['tasks']) == len({t['marker'] for t in panel['tasks']}) == 24
    assert not panel['training_eligible'] and not panel['final_holdout']
    assert panel['decode']['max_output_tokens'] == 4096
    for kind in {t['kind'] for t in panel['tasks']}:
        assert sum(t['kind'] == kind for t in panel['tasks']) == 4


def test_declared_answers_and_required_observations(tmp_path):
    for task in make_panel()['tasks']:
        root = tmp_path / task['id']
        root.mkdir()
        for path, text in task['fixtures'].items():
            p = root / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        for path in task['required_reads']:
            observation, missing = read_fixture(root, {'path': path, 'offset': 1, 'limit': 10})
            assert observation.startswith('1: approved=') and not missing
        for path in task['required_missing_reads']:
            observation, missing = read_fixture(root, {'path': path, 'offset': 1, 'limit': 10})
            assert observation.startswith('FileNotFoundError:') and missing
        assert score(task, 'Final: ' + task['answer'], task['required_reads'], task['required_missing_reads'])
        assert not score(task, 'wrong', task['required_reads'], task['required_missing_reads'])
        if task['required_reads']:
            assert not score(task, task['answer'], [], task['required_missing_reads'])


@pytest.mark.parametrize('arguments', [
    {'path': '../outside', 'offset': 1, 'limit': 10},
    {'path': '/etc/passwd', 'offset': 1, 'limit': 10},
    {'path': 'file', 'offset': True, 'limit': 10},
    {'path': 'file', 'offset': 1, 'limit': 2000},
    {'path': 'file', 'offset': 1, 'limit': 10, 'command': 'anything'},
])
def test_read_rejects_unsafe_or_unbounded_arguments(tmp_path, arguments):
    with pytest.raises(ValueError):
        read_fixture(tmp_path, arguments)


def test_read_refuses_symlink(tmp_path):
    (tmp_path / 'target').write_text('contents')
    (tmp_path / 'link').symlink_to(tmp_path / 'target')
    with pytest.raises(ValueError):
        read_fixture(tmp_path, {'path': 'link', 'offset': 1, 'limit': 10})
