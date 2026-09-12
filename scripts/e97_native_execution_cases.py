"""Frozen, non-training diagnostic tasks and host-only outcome oracles."""
import hashlib
import json

SCHEMA = 'emender-e97-native-execution-panel-v1'
SYSTEM = ('Complete the task with the declared source-native tools in /testbed. '
          'Use tool observations, including errors, to decide what to do next. '
          'Finish only when the task is complete. Keep workspace operations inside /testbed.')


def context(role, text):
    return dict(role=role, content=text, reasoning_content=None, think=None, tool_calls=None)


def cases(seed):
    """Each pair has an identical prompt, but different environment-only answers."""
    result = []
    for family in range(4):
        for variant in range(2):
            h = hashlib.sha256(f'{seed}:{family}:{variant}'.encode()).hexdigest()
            token = h[:20]
            left, right = int(h[20:26], 16), int(h[26:32], 16)
            if family == 0:
                name = 'lookup'
                prompt = ('Read /testbed/config.json. Return exactly the value selected by its active key '
                          'using finish. Do not change any files.')
                files = {'config.json': json.dumps({'active': 'blue' if variant else 'green',
                         'values': {'blue': token if variant else h[32:52],
                                    'green': h[32:52] if variant else token}})}
                answer, output = token, None
            elif family == 1:
                name = 'sum'
                prompt = ('Read /testbed/numbers.json. Return exactly the decimal sum of left and right '
                          'using finish. Do not change any files.')
                files = {'numbers.json': json.dumps(dict(left=left, right=right))}
                answer, output = str(left + right), None
            elif family == 2:
                name = 'edit'
                prompt = ('Read /testbed/state.json. Write /testbed/result.json containing the same JSON '
                          'object but with count increased by 7. Preserve every other field and the original '
                          'file. Verify the result, then finish with exactly done.')
                state = dict(ticket=token, count=left, owner='fixture-owner')
                files = {'state.json': json.dumps(state)}
                answer, output = 'done', {**state, 'count': left + 7}
            else:
                name = 'recovery'
                prompt = ('First attempt to read /testbed/missing.json using str_replace_editor view; '
                          'wait for its result before trying other paths. If it is absent, read '
                          '/testbed/catalog.json to find active_path, then read that file and return '
                          'exactly its value field using finish. Do not change any files.')
                files = {'catalog.json': json.dumps({'active_path': '/testbed/items/record.json'}),
                         'items/record.json': json.dumps({'value': token})}
                answer, output = token, None
            result.append(dict(id=f'{name}-{variant}', family=name, pair_index=family,
                               variant=variant, prompt=prompt, files=files,
                               answer=answer, expected_output=output))
    return result


def grade(case, final, calls, snapshot):
    """Snapshot comes from the isolated trusted reader while all agent processes are paused.

    Success requires finish, actual external execution and unchanged input files.
    Pairwise success on lookup/sum/recovery tests environment-conditioned answers;
    edit success alone does not prove the model itself inspected an observation.
    """
    unchanged = all(snapshot.get(path) == text for path, text in case['files'].items())
    output_ok = True
    if case['expected_output'] is not None:
        try:
            output_ok = json.loads(snapshot.get(case.get('output_path','result.json'), '')) == case['expected_output']
        except (ValueError, TypeError):
            output_ok = False
    recovery = True
    if case['family'] == 'recovery':
        recovery = False
        if calls:
            first = calls[0]
            a = first['request']
            first_error = (a['name'] == 'str_replace_editor' and
                           a['arguments'].get('command') == 'view' and
                           a['arguments'].get('path') == case.get('missing_path','/testbed/missing.json') and
                           first.get('result', {}).get('message', {}).get('content', '').startswith('ERROR:'))
            recovery = first_error and any(case['answer'] in c.get('result', {}).get('message', {}).get('content', '')
                                           for c in calls[1:])
    checks = dict(finished_correctly=final == case['answer'], external_call=bool(calls),
                  source_files_unchanged=unchanged, output_correct=output_ok,
                  required_recovery_observed=recovery)
    return dict(success=all(checks.values()), checks=checks)


def aggregate_results(panel, reports):
    models = {}
    for index, target in enumerate(panel['models']):
        rows = []
        for rank in (2 * index, 2 * index + 1):
            report = reports[rank]
            if report['rank'] != rank or report['model'] != target:
                raise ValueError('rank/model identity')
            expected = {c['id'] for c in panel['cases'] if c['pair_index'] % 2 == rank % 2}
            actual = [r['id'] for r in report['results']]
            if len(actual) != len(expected) or set(actual) != expected:
                raise ValueError('case coverage')
            rows.extend(report['results'])
        families = {}
        for family in sorted({c['family'] for c in panel['cases']}):
            selected = [r for r in rows if r['family'] == family]
            families[family] = dict(episodes=len(selected), successes=sum(r['grade']['success'] for r in selected),
                                    paired_success=all(r['grade']['success'] for r in selected))
        models[target['name']] = dict(episodes=len(rows), successes=sum(r['grade']['success'] for r in rows),
                                      families=families, outcomes=[{k: r[k] for k in ('id', 'reason', 'grade')} for r in rows])
    return dict(schema=SCHEMA, status='measurements-complete', models=models, training_eligible=False,
                checkpoint_promotion=False, scope='Authored paired diagnostic tasks; not repository benchmark or independent final holdout.')
