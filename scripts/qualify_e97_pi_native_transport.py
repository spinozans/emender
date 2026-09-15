#!/usr/bin/env python3
"""Authored CPU controls through real Pi and the qualified native sandbox.

This is executor/transport qualification, NOT E97 sampling or training evidence.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal

import tiktoken

from scripts.e97_native_execution_cases import context
from scripts.e97_native_execution_sandbox import NativeSandbox
from scripts.e97_open_swe_native_codec import compact, native_turn
from scripts.e97_open_swe_native_runtime_protocol import NativeEpisode
from scripts.e97_pi_native_bridge import NativePiBridge
from scripts.e97_pi_native_transport import serve_pi
from scripts.eval_e97_native_execution import publish, sha


def scripted_controls():
    return [dict(name='persistent-shell-edit', files={}, expected={'check.py': 'value = 24\n'}, steps=[
        ('think', dict(thought='PRIVATE_THOUGHT_CONTROL_SENTINEL')),
        ('execute_bash', dict(command="export E97_COMPAT_MARK='persisted'; cd /testbed")),
        ('execute_bash', dict(command="printf '%s|%s\\n' \"$E97_COMPAT_MARK\" \"$PWD\"")),
        ('str_replace_editor', dict(command='create', path='/testbed/check.py', file_text='value = 17\n')),
        ('str_replace_editor', dict(command='str_replace', path='/testbed/check.py', old_str='17', new_str='24')),
        ('str_replace_editor', dict(command='view', path='/testbed/check.py')),
        ('finish', dict(message='Scripted persistent-shell/edit transport checked.')),
    ]), dict(name='native-errors-recovery', files={'recover.py': 'value = 4\n'}, expected={'recover.py': 'value = 29\n'}, steps=[
        ('str_replace_editor', dict(command='view', path='/testbed/absent.py')),
        ('execute_bash', dict(command="printf 'real failure\\n'; false")),
        ('str_replace_editor', dict(command='view', path='/testbed/recover.py')),
        ('str_replace_editor', dict(command='str_replace', path='/testbed/recover.py', old_str='no such text', new_str='29')),
        ('str_replace_editor', dict(command='str_replace', path='/testbed/recover.py', old_str='4', new_str='29')),
        ('str_replace_editor', dict(command='view', path='/testbed/recover.py')),
        ('finish', dict(message='Scripted native-error/recovery transport checked.')),
    ])]


def texts(control):
    return [native_turn(dict(role='assistant', content=f'Authored transport control {i+1}.',
              reasoning_content='PRIVATE_ANALYSIS_CONTROL_SENTINEL', think=None,
              tool_calls=[dict(type='function', function=dict(name=name, arguments=compact(args)))]))
            for i, (name, args) in enumerate(control['steps'])]


def run_control(panel, control, encoding, output, pi_bin, extension):
    output.mkdir(mode=0o700)
    prompt = 'Run the authored native transport control; this is not a model evaluation.'
    scripted = texts(control)
    observed = []
    prefixes = []
    index = 0
    with NativeSandbox(panel, output/'sandbox') as sandbox:
        sandbox.request('setup', files=control['files'])
        def execute(call):
            reply = sandbox.request('execute', call=call)
            if 'dispatch_error' in reply:
                raise ValueError('scripted native dispatch failed')
            observed.append(reply['result']['message'])
            return reply
        def generate(actual, budget, deadline):
            nonlocal index
            # Independent direct-codec replay of authored turns and actual
            # executor observations, not a projection of the bridge's history.
            direct = NativeEpisode(panel['tools'], encoding)
            direct.append_source_message(panel.get('system_message', context('system', panel['system'])))
            direct.append_source_message(context('user', prompt))
            obs_index = 0
            for previous in scripted[:index]:
                turn = direct.accept_generated_turn(previous)
                if turn.backend_call() is None:
                    direct.append_observation(context('tool', 'Your thought has been logged.'))
                else:
                    direct.append_observation(observed[obs_index]); obs_index += 1
            if direct.prompt() != actual:
                raise ValueError('direct/Pi native prompt bytes differ')
            text = scripted[index]
            ids = encoding.encode_ordinary(text)
            if len(ids) > budget:
                raise ValueError('scripted turn budget')
            prefixes.append(dict(turn=index, exact=True))
            index += 1
            return text, ids, 'valid'
        bridge = NativePiBridge(panel, prompt, encoding, generate, execute)
        try:
            terminal = serve_pi(bridge, output/'pi', pi_bin=pi_bin, extension=extension, seconds=180)
            actual_files = sandbox.snapshot(list(control['expected']))
            if actual_files != control['expected']:
                raise ValueError('external snapshot oracle failed')
            if index != len(scripted):
                raise ValueError('scripted turn coverage')
        finally:
            publish(output/'episode-private.json', dict(origin='authored-transport-control', training_eligible=False,
                    source_messages=bridge.episode.source_messages(), generations=bridge.generations,
                    calls=bridge.calls, public_history=bridge.history, reason=bridge.reason))
    for filename in ('pi-events-private.jsonl', 'requests-private.jsonl'):
        public_wire = (output/'pi'/filename).read_text()
        if 'PRIVATE_ANALYSIS_CONTROL_SENTINEL' in public_wire or 'PRIVATE_THOUGHT_CONTROL_SENTINEL' in public_wire:
            raise ValueError('private field crossed Pi boundary')
    errors = sum(m['isError'] for m in bridge.history if m['role'] == 'toolResult')
    if control['name'] == 'native-errors-recovery' and errors != 3:
        raise ValueError('native error flags not preserved')
    if control['name'] == 'persistent-shell-edit' and not any('persisted|/testbed' in m['content'] for m in observed):
        raise ValueError('native shell state not preserved')
    for filename in ('cleanup.json', 'reader-cleanup.json'):
        if json.loads((output/'sandbox'/filename).read_text())['removed'] is not True:
            raise ValueError('owned cleanup missing')
    return dict(name=control['name'], scripted_turns=index, model_generations=0, full_prompt_byte_checks=prefixes,
                native_calls=len(bridge.calls), native_errors=errors, oracle=True, pi_close_verified=terminal['close_verified'],
                agent_cleanup=True, reader_cleanup=True, private_fields_absent_from_pi=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--panel', type=Path, required=True)
    p.add_argument('--panel-sha', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pi', default=shutil.which('pi'))
    args = p.parse_args()
    os.umask(0o077)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: (_ for _ in ()).throw(SystemExit(128+signum)))
    if sha(args.panel) != args.panel_sha:
        raise ValueError('native panel identity')
    panel = json.loads(args.panel.read_text())
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU controls must hide GPUs')
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    controls = scripted_controls()
    publish(args.output/'plan.json', dict(schema='emender-e97-pi-native-scripted-controls-v1', controls=controls,
            panel=str(args.panel), panel_sha256=args.panel_sha, training_eligible=False, model_generations=0))
    extension = Path(__file__).resolve().parents[1]/'configs/pi/e97-openhands-compat.ts'
    results = []
    try:
        for control in controls:
            results.append(run_control(panel, control, tiktoken.get_encoding('p50k_base'),
                                       args.output/control['name'], args.pi, extension))
        publish(args.output/'summary.json', dict(status='scripted-transport-controls-passed', results=results,
                model_generations=0, updates=0, checkpoint_promotion=False, pi_native_tools=False))
        print('PI_NATIVE_SCRIPTED_CONTROLS_PASSED', len(results), flush=True)
    except BaseException as exc:
        publish(args.output/'failure.json', dict(type=type(exc).__name__, message=str(exc), completed=results,
                                               model_generations=0, updates=0))
        raise


if __name__ == '__main__':
    main()
