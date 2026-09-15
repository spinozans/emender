from copy import deepcopy
import pytest
from scripts.e97_pi_repository_probes import repository_probes, inspect_snapshot


def test_fixtures_and_repairs_are_bounded_and_structurally_admitted():
    cases = repository_probes()
    assert len(cases) == 4 and len({c['id'] for c in cases}) == 4
    assert len({c['prompt'] for c in cases}) == 1
    for c in cases:
        assert c['training_eligible'] is False and c['independent_repository_claim'] is False
        assert c['max_turns'] == 8
        original = inspect_snapshot(c, c['files'])
        assert original['semantic_tests_required']
        repaired = {**c['files'], c['implementation']: c['expected_source']}
        assert inspect_snapshot(c, repaired)['host_code_execution'] is False
        assert c['files'][c['implementation']] != c['expected_source']


@pytest.mark.parametrize('mutation', ['test', 'readme', 'missing', 'extra'])
def test_immutable_files_and_coverage(mutation):
    c = repository_probes()[0]; snapshot = deepcopy(c['files'])
    if mutation == 'test': snapshot['tests/test_threshold.py'] = 'print("OK")'
    elif mutation == 'readme': snapshot['README.md'] = 'different'
    elif mutation == 'missing': snapshot.pop(c['implementation'])
    else: snapshot['sitecustomize.py'] = 'pass'
    with pytest.raises(ValueError): inspect_snapshot(c, snapshot)


@pytest.mark.parametrize('source', [
    'import os\ndef qualifies(a,b): return True\n',
    'def qualifies(a,b): return eval("True")\n',
    'def qualifies(a,b): return sorted(["print(1)"], key=eval)\n',
    'def qualifies(a,b): return globals().get("x")\n',
    'def qualifies(a,b): return a.__class__\n',
    'def qualifies(a,b):\n    while True: pass\n',
    '@print\ndef qualifies(a,b): return True\n',
    'def dummy(eval): pass\ndef qualifies(a,b): return sorted(["print(1)"], key=eval)\n',
    'def qualifies(a,b, fn=eval): return sorted(["print(1)"], key=fn)\n',
])
def test_unsupported_shapes_rejected_before_external_verifier(source):
    c = repository_probes()[0]
    with pytest.raises(ValueError): inspect_snapshot(c, {**c['files'], c['implementation']: source})
