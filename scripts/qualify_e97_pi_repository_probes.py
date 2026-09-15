#!/usr/bin/env python3
"""Verify authored repository workflows through real Pi/OpenHands; zero models."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import signal
import tiktoken
from scripts.e97_native_execution_sandbox import NativeSandbox
from scripts.e97_pi_repository_probes import repository_probes, inspect_snapshot
from scripts.qualify_e97_pi_native_transport import run_control
from scripts.eval_e97_native_execution import publish, sha

TEST_COMMAND = 'python -m unittest discover -s tests -v'


def control(case):
    implementation = case['implementation']
    test = next(p for p in case['files'] if p.startswith('tests/'))
    return dict(name=case['id'], files=deepcopy(case['files']),
                expected={**case['files'], implementation: case['expected_source']}, steps=[
        ('execute_bash', dict(command='ls')),
        ('str_replace_editor', dict(command='view', path='/testbed/README.md')),
        ('execute_bash', dict(command=TEST_COMMAND)),
        ('str_replace_editor', dict(command='view', path='/testbed/'+test)),
        ('str_replace_editor', dict(command='view', path='/testbed/'+implementation)),
        ('str_replace_editor', dict(command='str_replace', path='/testbed/'+implementation,
                                  old_str=case['files'][implementation], new_str=case['expected_source'])),
        ('execute_bash', dict(command=TEST_COMMAND)),
        ('finish', dict(message='Authored repository control verified.')),
    ])


def verifier(panel, case, files, output):
    admitted = inspect_snapshot(case, files)
    with NativeSandbox(panel, output) as sandbox:
        sandbox.request('setup', files=admitted['verified_files'])
        result = sandbox.request('execute', call=dict(name='execute_bash', arguments=dict(command=TEST_COMMAND, timeout=15)))
        if 'dispatch_error' in result:
            raise ValueError('verifier_dispatch_failure')
        publish(output/'test-result-private.json', result)
        return result['result']['exit_code']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--panel', type=Path, required=True); p.add_argument('--panel-sha', required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--pi', default=shutil.which('pi'))
    a = p.parse_args(); os.umask(0o077)
    def interrupted(signum, frame): raise TimeoutError('repository preflight interrupted')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or sha(a.panel) != a.panel_sha:
        raise ValueError('CPU/panel identity')
    panel = json.loads(a.panel.read_text())
    cases = repository_probes()
    a.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    publish(a.output/'plan-private.json', dict(cases=cases, model_generations=0, optimizer_updates=0,
            training_eligible=False, independent_repository_claim=False, panel_sha256=a.panel_sha))
    results = []
    try:
        for case in cases:
            row = run_control(panel, control(case), tiktoken.get_encoding('p50k_base'), a.output/case['id'], a.pi,
                              Path(__file__).resolve().parents[1]/'configs/pi/e97-openhands-compat.ts')
            episode = json.loads((a.output/case['id']/'episode-private.json').read_text())
            tests = [c['result']['exit_code'] for c in episode['calls'] if c['request']['name']=='execute_bash'
                     and c['request']['arguments']['command']==TEST_COMMAND]
            if tests != [1,0]: raise ValueError('failing_then_passing_tests_not_observed')
            actual = json.loads((a.output/case['id']/'sandbox/reader.stdout').read_text())
            repaired_exit = verifier(panel, case, actual, a.output/(case['id']+'-verifier'))
            original_exit = verifier(panel, case, case['files'], a.output/(case['id']+'-negative-verifier'))
            if repaired_exit != 0 or original_exit != 1:
                raise ValueError('independent_semantic_verifier_controls_failed')
            row.update(original_tests_failed=True, repaired_tests_passed=True,
                       fresh_sandbox_verifier=True, verifier_negative_control=True)
            results.append(row)
        publish(a.output/'summary.json', dict(status='authored-repository-workflow-preflight-passed',
                results=results, model_generations=0, optimizer_updates=0, training_eligible=False,
                independent_repository_claim=False))
        print('PI_REPOSITORY_PREFLIGHT_PASSED', len(results), flush=True)
    except BaseException as exc:
        publish(a.output/'failure.json', dict(type=type(exc).__name__, message=str(exc), completed=results,
                                            model_generations=0, optimizer_updates=0))
        raise


if __name__ == '__main__': main()
