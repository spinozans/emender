"""F9R3/F9G permanent admission and synthetic sealed-grade regressions.

Negative command candidates never execute. Borrowed historical end-state fields
are validator inputs only, not evidence that the boundary commands ran.
"""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.e97_offline_shell import MAX_SHELL_LENGTH, parse_shell, serialize_shell, shell_allowed
from test_f9r2_grammar import gates, test_f9r2_four_gate_matrix as four_gate_matrix

S = Path(__file__).resolve().parent
CASES = json.loads((S / 'F9R3-grammar-cases.json').read_text())['cases']
# Exact source-named F9R3-boundary-qa.py probes; lengths derive from the bound.
BOUND_CASES = [
    ('ordinary-raw-4091', 'printf %s ' + 'x' * (MAX_SHELL_LENGTH - 15), 'REJECT', 'canonical'),
    ('ordinary-raw-4096', 'printf %s ' + 'x' * (MAX_SHELL_LENGTH - 10), 'REJECT', 'canonical'),
    ('apostrophes-817', 'printf %s "' + "'" * 817 + '"', 'REJECT', 'canonical'),
    ('apostrophes-1000', 'printf %s "' + "'" * 1000 + '"', 'REJECT', 'canonical'),
    ('raw-exact-bound', 'printf %s x' + ' ' * (MAX_SHELL_LENGTH - 11), 'ACCEPT', 'raw'),
    ('raw-one-over', 'printf %s x' + ' ' * (MAX_SHELL_LENGTH - 10), 'REJECT', 'raw'),
    ('canonical-exact-bound', 'printf %s ' + 'x' * (MAX_SHELL_LENGTH - 16), 'ACCEPT', 'canonical'),
    ('canonical-one-over', 'printf %s ' + 'x' * (MAX_SHELL_LENGTH - 15), 'REJECT', 'canonical'),
    ('apostrophes-exact-bound', 'printf %s "' + "'" * 816 + '"', 'ACCEPT', 'canonical'),
    ('apostrophes-one-over', 'printf %s "' + "'" * 816 + 'x"', 'REJECT', 'canonical'),
]


def test_f9r3_retained_case_counts_and_independent_baseline():
    baseline = json.loads((S / 'F9R2-grammar-results.json').read_text())['cases']
    assert len(CASES) == 237
    assert sum(c['expected'] == 'REJECT' for c in CASES) == 233
    assert sum(c['expected'] == 'ACCEPT' for c in CASES) == 4
    assert len({c['command'] for c in CASES}) == 237
    assert not {c['command'] for c in baseline} & {c['command'] for c in CASES}


@pytest.mark.parametrize('case', CASES, ids=[f"f9r3-{c['category']}-{i:03}" for i, c in enumerate(CASES)])
def test_f9r3_four_gate_regression(case, gates):
    four_gate_matrix(case['category'], case['command'], case['expected'], gates)


def test_f9g_single_source_constant_and_embedded_shell_ast(gates):
    direct = ast.parse((S / 'scripts/e97_offline_shell.py').read_text())
    sealed = ast.parse((S / 'scripts/e97_first_party_validator_diversity.py').read_text())
    def definitions(tree):
        return {n.name: ast.dump(n, include_attributes=False)
                for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name, definition in definitions(direct).items():
        assert definitions(sealed)[name] == definition, name
    def constant(tree):
        assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'MAX_SHELL_LENGTH' for t in n.targets)]
        assert len(assignments) == 1
        return assignments[0]
    assert ast.dump(constant(direct)) == ast.dump(constant(sealed))
    assert gates[0]['MAX_SHELL_LENGTH'] == MAX_SHELL_LENGTH
    # No shell bound literal in either implementation outside the generated binding.
    for tree in (direct, sealed):
        binding = constant(tree).value
        assert [n for n in ast.walk(tree) if isinstance(n, ast.Constant)
                and n.value == MAX_SHELL_LENGTH] == [binding]


@pytest.mark.parametrize('name,command,expected,bound', BOUND_CASES, ids=[c[0] for c in BOUND_CASES])
def test_f9g_admission_bound_closure(name, command, expected, bound, gates):
    four_gate_matrix('f9g-bound', command, expected, gates)
    reason = ('bounded offline shell required' if bound == 'raw'
              else 'canonical offline shell exceeds length bound')
    for check in (shell_allowed, gates[0]['shell_allowed']):
        if expected == 'REJECT':
            with pytest.raises(ValueError, match=reason):
                check(command)
        else:
            executed = check(command)
            assert len(executed) <= MAX_SHELL_LENGTH
            assert check(executed) == executed
    if expected == 'ACCEPT':
        four_gate_matrix('f9g-canonical-bound', shell_allowed(command), expected, gates)
    if name.startswith('raw-'):
        assert len(command) == MAX_SHELL_LENGTH + (expected == 'REJECT')
    elif name.endswith('exact-bound') or name.endswith('one-over'):
        assert len(serialize_shell(parse_shell(command))) == MAX_SHELL_LENGTH + (expected == 'REJECT')


@pytest.fixture(scope='module')
def historical_grade_input():
    evidence = json.loads((S / 'F9-evidence.json').read_text())
    root = Path(evidence['private_evidence_root']) / 'author-work/teacher-authored/tranche-test-diversity-test_real_pi_new_families_and_0/quarantine'
    identity = '18c99597ae7d714c927c6362f59339e241abe6310299e971b00a22e6d9fc0cb9'
    spec = root / 'private_validators' / (identity + '.json')
    terminal = json.loads((root / 'era8-proofs' / identity / 'gym-terminal-projection-proof.json').read_bytes())
    return spec, terminal


@pytest.mark.parametrize('mode', ['focused', 'regression'])
@pytest.mark.parametrize('name,command,expected,bound', BOUND_CASES, ids=[c[0] for c in BOUND_CASES])
def test_f9g_sealed_grading_agrees_with_admission(name, command, expected, bound, mode,
                                                 gates, historical_grade_input, tmp_path):
    spec, historical = historical_grade_input
    # For raw-one-over the canonical form alone is short and legitimately admitted;
    # all canonical-expansion probes reject in both representations.
    canonical_command = serialize_shell(parse_shell(command)) if len(command) <= MAX_SHELL_LENGTH else None
    forms = [('raw', command)]
    if canonical_command is not None:
        forms.append(('canonical', canonical_command))
    for representation, form in forms:
        terminal = deepcopy(historical)
        terminal['actions'][1]['arguments']['command'] = form
        path = tmp_path / (representation + '.json')
        path.write_text(json.dumps(terminal))
        result = subprocess.run([sys.executable, str(Path(os.environ['F9_STAGE_ROOT']) /
            'scripts/e97_first_party_validator_diversity.py'), '--mode', mode,
            '--spec', str(spec), '--terminal', str(path)], capture_output=True, text=True, timeout=10)
        if expected == 'ACCEPT':
            assert result.returncode == 0, result.stderr
            assert gates[0]['shell_allowed'](form) == shell_allowed(form)
        else:
            assert result.returncode != 0
            assert 'sealed invalid-action outcome:' in result.stderr
            reason = ('canonical offline shell exceeds length bound'
                      if representation == 'raw' and bound == 'canonical'
                      else 'bounded offline shell required')
            assert reason in result.stderr
