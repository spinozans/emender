"""External W3G safety guard: frozen instruments remain unchanged."""
import os

import pytest

from scripts import watch_w3_gate_window as guard


def state():
    return dict(gpus=[[str(i), f'uuid-{i}', '4', '0'] for i in range(8)], processes=[])


def test_window_requires_all_eight_unleased_idle_cards_and_no_process():
    snapshot = state()
    assert guard.window_free(snapshot, '0,1,2,3,4,5,6,7')
    assert not guard.window_free(snapshot, '0,1,2,3,4,5,6')
    snapshot['processes'] = [dict(pid=123)]
    assert not guard.window_free(snapshot, '0,1,2,3,4,5,6,7')
    snapshot['processes'] = []
    snapshot['gpus'][7][2] = '256'
    assert not guard.window_free(snapshot, '0,1,2,3,4,5,6,7')
    snapshot['gpus'][7][2:4] = ['4', '10']
    assert not guard.window_free(snapshot, '0,1,2,3,4,5,6,7')


def test_occupancy_identity_checks_ancestry_and_root_starttime(monkeypatch):
    tree = {100: (1, 'root-start'), 101: (100, 'child'), 102: (101, 'grandchild'),
            200: (1, 'foreign')}
    monkeypatch.setattr(guard, 'process_fields', lambda pid: tree[pid])
    assert guard.belongs_to(102, 100, 'root-start')
    assert not guard.belongs_to(102, 100, 'reused-pid')
    assert not guard.belongs_to(200, 100, 'root-start')


def test_actual_proc_identity():
    pid = os.getpid()
    parent, start = guard.process_fields(pid)
    assert parent == os.getppid()
    assert guard.belongs_to(pid, pid, start)
    assert not guard.belongs_to(pid, pid, 'wrong-starttime')


def test_evidence_is_create_once(tmp_path):
    path = tmp_path / 'receipt.json'
    guard.write_once(path, dict(value=1))
    with pytest.raises(FileExistsError):
        guard.write_once(path, dict(value=2))
    assert '1' in path.read_text()


def test_failed_arm_prevents_any_pin_hash_or_read(tmp_path):
    directory = tmp_path / 'arms/A'
    directory.mkdir(parents=True)
    guard.write_once(directory / 'complete-receipt.json', dict(
        status='complete', updates=511, inputs=3571182, targets=154386))
    guard.write_once(tmp_path / 'arms/A-terminal.json', dict(exit_code=0))
    with pytest.raises(ValueError, match='incomplete/different arm'):
        guard.verify_entries(tmp_path)
