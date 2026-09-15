"""First-party miniature repository probes; no downloaded or protected code.

These target discovery, test-guided editing and verification of small pure
functions, NOT independent/general repository competence. Fixtures and all tests
must be frozen before model sampling. No model call or executor lives here.
"""
import ast
from copy import deepcopy


def repository_probes():
    cases = []
    for world in (0, 1):
        inclusive = world == 0
        intended = '>=' if inclusive else '>'
        broken = '>' if inclusive else '>='
        module = f'def qualifies(score, minimum):\n    return score {broken} minimum\n'
        cases.append(dict(id=f'pi-repository-boundary-world-{world}', family='boundary', world=world,
            prompt='The tests in /testbed are failing. Discover the project, fix the implementation, and run the tests to verify your change. Do not change tests or documentation.',
            implementation='threshold.py', expected_source=f'def qualifies(score, minimum):\n    return score {intended} minimum\n',
            files={'README.md': f'# Threshold utility\n\nqualifies(score, minimum) returns whether score {intended} minimum.\nRun: python -m unittest discover -s tests -v\n',
                   'threshold.py': module,
                   'tests/test_threshold.py': 'import unittest\nfrom threshold import qualifies\n\nclass ThresholdTests(unittest.TestCase):\n    def test_boundaries(self):\n' +
                    ''.join(f'        self.assertIs(qualifies({a}, {b}), {a >= b if inclusive else a > b})\n'
                            for a,b in [(-4,-4), (0,0), (7,7), (8,7), (6,7), (-3,-4), (-5,-4)])}))
        key, value = ('code','amount') if world == 0 else ('label','price')
        source = f'def lookup(rows, wanted):\n    return rows[0]["{value}"]\n'
        repair = (f'def lookup(rows, wanted):\n    for row in rows:\n        if row["{key}"] == wanted:\n'
                  f'            return row["{value}"]\n    return None\n')
        checks = []
        for rows, wanted, expected in [([{key:'a',value:13},{key:'b',value:41}],'b',41),
                                        ([{key:'b',value:41},{key:'a',value:13}],'b',41),
                                        ([{key:'a',value:13}],'absent',None), ([], 'anything',None)]:
            checks.append(f'        self.assertEqual(lookup({rows!r}, {wanted!r}), {expected!r})\n')
        cases.append(dict(id=f'pi-repository-selection-world-{world}', family='selection', world=world,
            prompt='The tests in /testbed are failing. Discover the project, fix the implementation, and run the tests to verify your change. Do not change tests or documentation.',
            implementation='catalog.py', expected_source=repair,
            files={'README.md': f'# Catalog utility\n\nlookup(rows, wanted) selects the row whose {key} equals wanted and returns its {value}. Return None for no match, including empty input.\nRun: python -m unittest discover -s tests -v\n',
                   'catalog.py': source,
                   'tests/test_catalog.py': 'import unittest\nfrom catalog import lookup\n\nclass CatalogTests(unittest.TestCase):\n    def test_selection(self):\n'+''.join(checks)}))
    for c in cases:
        c.update(origin='first-party-authored-repository-probe', training_eligible=False,
                 independent_repository_claim=False, max_turns=8, generation_budget=4096,
                 episode_generation_budget=8192, episode_seconds=600)
    return cases


def inspect_snapshot(case, snapshot):
    """Read-only structure/immutability guard before execution in a fresh sandbox.

    This is NOT a semantic oracle. Unsupported implementation shapes must be
    reported separately. No returned model code is ever executed on the host.
    The deliberately narrow pure-function policy prevents fixture code replacing
    the verifier, importing OS facilities, or monkeypatching unittest.
    """
    if set(snapshot) != set(case['files']):
        raise ValueError('snapshot_coverage')
    for name, original in case['files'].items():
        if name != case['implementation'] and snapshot[name] != original:
            raise ValueError('tests_or_documentation_changed')
    source = snapshot[case['implementation']]
    if not isinstance(source, str) or len(source.encode()) > 65536:
        raise ValueError('implementation_size_or_type')
    tree = ast.parse(source)
    if not tree.body or any(not isinstance(node, ast.FunctionDef) for node in tree.body):
        raise ValueError('unsupported_module_shape')
    forbidden = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef,
                 ast.With, ast.AsyncWith, ast.AsyncFunctionDef, ast.Await, ast.While,
                 ast.Lambda, ast.Raise, ast.Try, ast.Delete)
    safe_calls = {'len', 'range', 'sum', 'next', 'enumerate', 'sorted', 'list', 'tuple',
                  'int', 'str', 'bool', 'min', 'max', 'abs'}
    functions = {n.name for n in tree.body}
    for function in tree.body:
        if (function.args.defaults or any(x is not None for x in function.args.kw_defaults) or
                function.returns is not None or any(n.annotation is not None for n in ast.walk(function) if isinstance(n, ast.arg))):
            raise ValueError('unsupported_default_or_annotation')
        if any(isinstance(n, ast.FunctionDef) and n is not function for n in ast.walk(function)):
            raise ValueError('unsupported_nested_function')
        bound = {n.arg for n in ast.walk(function) if isinstance(n, ast.arg)}
        bound |= {n.id for n in ast.walk(function) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        for node in ast.walk(function):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in bound | safe_calls | functions:
                raise ValueError('unsupported_free_name')
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            raise ValueError('unsupported_implementation_shape')
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            raise ValueError('unsupported_decorator')
        if isinstance(node, ast.Name) and (node.id.startswith('__') or node.id in {
                'eval', 'exec', 'compile', 'open', 'globals', 'locals', 'vars', 'getattr',
                'setattr', 'delattr', 'breakpoint', 'input', 'print', 'help', 'dir', 'type', 'object'}):
            raise ValueError('unsupported_reflection_or_io_name')
        if isinstance(node, ast.Attribute) and node.attr not in ('get', 'append'):
            raise ValueError('unsupported_attribute')
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in safe_calls:
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr in ('get', 'append'):
                continue
            raise ValueError('unsupported_call')
    return dict(structure_admitted=True, semantic_tests_required=True,
                verified_files=deepcopy(snapshot), host_code_execution=False)
