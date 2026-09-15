"""Frozen case selection and fail-closed compatibility gate; no model loads."""
from copy import deepcopy
import pytest

from scripts.eval_e97_pi_native_compatibility import FAMILIES, gate, select_cases
from scripts.qualify_e97_pi_native_transport import scripted_controls, texts


def fixture_panel():
    return dict(cases=[dict(id=f'bridge-fresh-{f}-{i:04d}-world-{w}', family=f,
                            cohort='fresh', pair_index=j*4+i)
                       for j, f in enumerate(sorted(FAMILIES)) for i in range(4) for w in (0, 1)])


def rows():
    return [dict(id=c['id'], family=c['family'], direct_success=True, pi_success=True,
                 pi_close_verified=True, first_prompt_equal=True, pi_prompt_replay_exact=True)
            for c in select_cases(fixture_panel())]


def test_family_local_first_pair_not_global_zero():
    cases = select_cases(fixture_panel())
    assert len(cases) == 8
    assert all('-0000-world-' in c['id'] for c in cases)
    assert {c['family'] for c in cases} == FAMILIES


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'assisted'])
def test_case_coverage_fail_closed(mutation):
    panel = fixture_panel()
    if mutation == 'missing': panel['cases'].pop(0)
    elif mutation == 'duplicate': panel['cases'][1] = deepcopy(panel['cases'][0])
    else: panel['cases'][0]['supplied_calls'] = [dict(name='str_replace_editor')]
    with pytest.raises(ValueError): select_cases(panel)


@pytest.mark.parametrize('field', ['direct_success', 'pi_success', 'pi_close_verified',
                                  'first_prompt_equal', 'pi_prompt_replay_exact'])
def test_gate_rejects_missing_false_nonboolean(field):
    assert all(gate(rows()).values())
    r = rows(); r[0][field] = False
    assert not all(gate(r).values())
    r[0][field] = 1
    with pytest.raises(ValueError): gate(r)
    del r[0][field]
    with pytest.raises(KeyError): gate(r)


def test_gate_coverage():
    with pytest.raises(ValueError): gate(rows()[:-1])
    r = rows(); r[1] = deepcopy(r[0])
    with pytest.raises(ValueError): gate(r)


def test_scripted_controls_are_not_model_evidence():
    controls = scripted_controls()
    assert [len(texts(c)) for c in controls] == [7, 7]
    assert all(c['steps'][-1][0] == 'finish' for c in controls)
    assert sum(name not in ('finish', 'think') for c in controls for name, args in c['steps']) == 11
