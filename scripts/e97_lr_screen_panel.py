#!/usr/bin/env python3
"""Freeze a small fresh-instance LR-screening panel; not a final holdout."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from ndm.e97_atomic import publish_bytes_no_replace
from ndm.e97_agent_protocol import E97_PI_AGENT_ANALYSIS_SYSTEM_V1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def make_panel():
    tasks = []
    for kind in ('instruction', 'document', 'document_sum', 'read', 'two_reads', 'recovery'):
        for i in range(4):
            marker = 'screen_' + hashlib.sha256(f'e97-lr5e5-dev-20260909-v1/{kind}/{i}'.encode()).hexdigest()[:24]
            first, second = f'{marker}/primary.txt', f'{marker}/secondary.txt'
            alpha, beta = marker + '_alpha', marker + '_beta'
            a = int(hashlib.sha256(marker.encode()).hexdigest()[:7], 16)
            b = a + 173
            fixtures, required, missing = {}, [], []
            if kind == 'instruction':
                user = f'For ticket {marker}, return exactly {alpha}. No explanation and no tools.'
                answer = alpha
            elif kind == 'document':
                user = f'Read this supplied document. Ticket: {marker}. Draft label: {beta}. Approved label: {alpha}. Return only the approved label. No tools.'
                answer = alpha
            elif kind == 'document_sum':
                user = f'Ticket {marker}: Batch A contains {a} items; batch B contains {b} items. Return their combined count as decimal digits only. No tools.'
                answer = str(a+b)
            elif kind == 'read':
                fixtures = {first: f'approved={alpha}\n'}
                required = [first]
                user = f'Read {first} with offset=1 and limit=10. Return only its approved value.'
                answer = alpha
            elif kind == 'two_reads':
                fixtures = {first: f'approved={alpha}\n', second: f'approved={beta}\n'}
                required = [first, second]
                user = f'Read {first}, then {second}, each with offset=1 and limit=10. Return their approved values separated by a comma, first then second, without spaces.'
                answer = alpha + ',' + beta
            else:
                absent = f'{marker}/retired.txt'
                fixtures = {second: f'approved={beta}\n'}
                required, missing = [second], [absent]
                user = f'First attempt to read {absent}, offset=1 and limit=10. If it is missing, read {second} with the same bounds instead. Return only the approved value you actually observe.'
                answer = beta
            tasks.append({'id': f'{kind}-{i:02d}', 'kind': kind, 'marker': marker,
                          'user': 'Only the read tool is available. ' + user,
                          'fixtures': fixtures, 'required_reads': required,
                          'required_missing_reads': missing, 'answer': answer})
    return {'schema': 'emender-e97-lr-screen-panel-v1', 'purpose': 'fresh-instance development only',
            'training_eligible': False, 'final_holdout': False,
            'limitations': ['synthetic familiar families; not unseen repositories or unseen task families',
                            'exact-instance novelty is not semantic non-overlap',
                            'direct tokenwise engine evaluation; not HTTP/Pi qualification'],
            'system': E97_PI_AGENT_ANALYSIS_SYSTEM_V1, 'tasks': tasks,
            'decode': {'temperature': 0, 'max_output_tokens': 4096,
                       'max_analysis_tokens': 2048, 'max_turns': 6},
            'milestones': [64, 128, 240],
            'decision': {'retention_floors_each_weight_mode': {'core': 116, 'compositional': 228},
                         'numerical_q8_is_not_behavioral_selection': True,
                         'continue_if_retention_passes_even_without_acquisition': True,
                         'screening_signal': {'complete_tasks_at_least': 12, 'tool_tasks_at_least': 4,
                                              'first_turn_protocol_valid_at_least': 22,
                                              'paired_success_gain_over_parent_at_least': 4},
                         'terminal_selection': 'screening evidence only; no promotion without matched control and independent holdout',
                         'no_midrun_lr_prompt_or_mixture_changes': True}}


def normalized_final(text):
    text = text.strip()
    return text[len('Final:'):].strip() if text.startswith('Final:') else text


def score(task, final, reads, missing):
    return (isinstance(final, str) and normalized_final(final) == task['answer']
            and reads == task['required_reads'] and missing == task['required_missing_reads'])


def read_fixture(root, arguments):
    if (not isinstance(arguments, dict) or set(arguments) != {'path', 'offset', 'limit'}
            or not isinstance(arguments['path'], str)
            or type(arguments['offset']) is not int or arguments['offset'] < 1
            or type(arguments['limit']) is not int or not 1 <= arguments['limit'] <= 10):
        raise ValueError('invalid bounded read arguments')
    path = Path(arguments['path'])
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('read escaped fixture')
    target = root / path
    if any(p.is_symlink() for p in [target, *target.parents]):
        raise ValueError('symlink read forbidden')
    try:
        with target.open('rb') as f:
            payload = f.read(65537)
    except FileNotFoundError:
        return f'FileNotFoundError: {path.as_posix()}', True
    if len(payload) > 65536:
        raise ValueError('oversize fixture observation')
    start = arguments['offset'] - 1
    lines = payload.decode().splitlines()
    text = '\n'.join(f'{i+1}: {line}' for i, line in enumerate(lines)
                     if start <= i < start + arguments['limit']) or '(no lines)'
    return text, False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    panel = make_panel()
    publish_bytes_no_replace(args.output, (json.dumps(panel, indent=2, sort_keys=True)+'\n').encode(), mode=0o600)
    print(json.dumps({'tasks': len(panel['tasks']), 'panel_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}))

if __name__ == '__main__':
    main()
