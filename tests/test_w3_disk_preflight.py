"""Supervisor-amended root capacity launch gate: fail closed before BOTH arms."""
import json
from types import SimpleNamespace

import pytest

from scripts.w3_disk_preflight import MIN_ROOT_FREE_BYTES, capacity_receipt, main


@pytest.mark.parametrize('arm', ['A', 'B'])
def test_root_capacity_floor_is_inclusive_and_applies_to_both_arms(arm):
    assert capacity_receipt(arm, MIN_ROOT_FREE_BYTES - 1)['status'] == 'blocked'
    assert capacity_receipt(arm, MIN_ROOT_FREE_BYTES)['status'] == 'passed'


def test_blocked_gate_records_evidence_then_exits_nonzero(tmp_path, monkeypatch):
    output = tmp_path / 'gate.json'
    monkeypatch.setattr('sys.argv', ['w3_disk_preflight', '--arm', 'B', '--output', str(output)])
    def usage(volume):
        assert volume == '/'
        return SimpleNamespace(free=23 * (1 << 30))
    monkeypatch.setattr('scripts.w3_disk_preflight.shutil.disk_usage', usage)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    report = json.loads(output.read_text())
    assert report['arm'] == 'B' and report['status'] == 'blocked'
    assert report['free_bytes'] == 23 * (1 << 30)
