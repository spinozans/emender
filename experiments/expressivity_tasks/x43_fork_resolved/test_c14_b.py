"""Registration audit regression checks. These do not exercise a B1 head."""
import copy
import json
import pytest
from experiments.expressivity_tasks.x43_fork_resolved.validate_c14_b import REGISTRATION, audit, check_bindings


def test_committed_registration_and_standing_geometry():
    result = audit()
    assert result['status'] == 'passed_registration_audit'
    assert result['trainable_parameters'] == 2133700
    assert not result['launch_ready']
    assert not result['training_started']
    assert len(result['remaining_gates']) >= 6


def test_audit_rejects_dependency_tampering():
    registration = copy.deepcopy(json.loads(REGISTRATION.read_text()))
    name = next(iter(registration['files']))
    registration['files'][name]['sha256'] = '0'*64
    with pytest.raises(ValueError, match='frozen dependency changed'):
        check_bindings(registration)


def test_audit_rejects_git_object_tampering():
    registration = copy.deepcopy(json.loads(REGISTRATION.read_text()))
    name = next(iter(registration['files']))
    registration['files'][name]['git_blob'] = '0'*40
    with pytest.raises(ValueError, match='frozen Git object changed'):
        check_bindings(registration)
