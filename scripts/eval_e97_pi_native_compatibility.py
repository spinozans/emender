#!/usr/bin/env python3
"""Frozen, unchanged-live-y matched native versus Pi-fronted diagnostic."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import time

from scripts.e97_native_execution_cases import context, grade
from scripts.e97_native_execution_sandbox import NativeSandbox
from scripts.e97_open_swe_native_codec import compact, vocabulary
from scripts.e97_open_swe_native_runtime_protocol import NativeEpisode
from scripts.e97_pi_native_bridge import NativePiBridge
from scripts.e97_pi_native_transport import serve_pi
from scripts.eval_e97_native_execution import episode, generate_turn, publish, sha

R = Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining')
CHECKPOINT_SHA = '9b78628d47c48c304c14de50399fe1853d8c83386c7cf5d9bd4a0e878bf679fa'
FAMILIES = {'lookup', 'sum', 'edit', 'recovery'}


def select_cases(panel):
    wanted = {f'bridge-fresh-{family}-0000-world-{world}' for family in FAMILIES for world in (0, 1)}
    selected = [deepcopy(c) for c in panel['cases'] if c['cohort'] == 'fresh' and c['id'] in wanted]
    if (len(selected) != 8 or len({c['id'] for c in selected}) != 8 or
            {c['family'] for c in selected} != FAMILIES or
            any(c.get('supplied_calls') for c in selected) or
            any(sum(c['family'] == f for c in selected) != 2 for f in FAMILIES)):
        raise ValueError('fixed eight-case paired coverage')
    return sorted(selected, key=lambda c: c['id'])


def gate(rows):
    if len(rows) != 8 or len({r['id'] for r in rows}) != 8:
        raise ValueError('matched row coverage')
    for row in rows:
        if any(type(row[k]) is not bool for k in ('direct_success', 'pi_success', 'pi_close_verified', 'first_prompt_equal', 'pi_prompt_replay_exact')):
            raise ValueError('nonboolean matched evidence')
    return dict(eight_direct_successes=all(r['direct_success'] for r in rows),
                eight_pi_successes=all(r['pi_success'] for r in rows),
                per_case_success_parity=all(r['direct_success'] == r['pi_success'] for r in rows),
                pi_closures_verified=all(r['pi_close_verified'] for r in rows),
                initial_prompt_bytes_equal=all(r['first_prompt_equal'] for r in rows),
                full_pi_native_prompt_replay_exact=all(r['pi_prompt_replay_exact'] for r in rows))


def freeze(args):
    source = R/'representation-bridge-v1-train/evaluation/execution/panel.json'
    if sha(source) != '8e70382d1e941e4c03d3c137699d8a1b6f76dfb9170619a5e308a6346d7d40eb':
        raise ValueError('audited bridge execution panel identity')
    panel = json.loads(source.read_text())
    target = next(m for m in panel['models'] if m['name'] == 'bridge-y')
    if target['sha256'] != CHECKPOINT_SHA or target['mode'] != 'train' or sha(target['checkpoint']) != CHECKPOINT_SHA:
        raise ValueError('unchanged exact live-y checkpoint identity')
    control = R/'pi-native-compatibility-v1-control'
    scripted = R/'pi-native-compatibility-v1-scripted'
    if (sha(control/'terminal.json') != 'ddfe92ed5b190f7bfb2a592fbc1529f532fdfb6e632f5745b9e8b563b3a0eeeb' or
            sha(scripted/'summary.json') != '9cdc588caaf749cc3ea9ee2d4b35188f43035d8d60d99f257607d7b5b3efa610'):
        raise ValueError('scripted receipt identity')
    terminal = json.loads((control/'terminal.json').read_text())
    results = json.loads((scripted/'summary.json').read_text())
    if terminal != dict(original_exit=0, audited_exit=0, model_generations=0, optimizer_updates=0):
        raise ValueError('scripted source/launch audit required')
    if results['status'] != 'scripted-transport-controls-passed' or len(results['results']) != 2:
        raise ValueError('scripted native controls required')
    # The exercised adapter and its native dependencies must be unchanged, not
    # merely accompanied by old passing receipts from a different implementation.
    for name in ('configs/pi/e97-openhands-compat.ts', 'scripts/e97_pi_native_bridge.py',
                 'scripts/e97_pi_native_transport.py', 'scripts/e97_open_swe_native_runtime_protocol.py',
                 'scripts/e97_open_swe_native_codec.py', 'scripts/e97_openhands_native_backend.py'):
        if sha(name) != sha(control/'worktree'/name):
            raise ValueError('adapter differs from scripted qualification')
    panel = {**panel, 'models': [target], 'cases': select_cases(panel)}
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    plan = dict(schema='emender-e97-pi-native-matched-v1', panel=panel, source_panel=str(source),
                source_panel_sha256=sha(source), scripted_summary=str(scripted/'summary.json'),
                scripted_summary_sha256=sha(scripted/'summary.json'),
                scripted_terminal=str(control/'terminal.json'), scripted_terminal_sha256=sha(control/'terminal.json'),
                pi_bin=str(control/'bin/pi'), pi_inventory=str(control/'pi-files.sha256'),
                pi_inventory_sha256=sha(control/'pi-files.sha256'),
                episode_routes=[['direct', 'pi'] if i % 2 == 0 else ['pi', 'direct'] for i in range(8)],
                max_model_episodes=16, gpu_count=1, run_seconds=2400, optimizer_updates=0,
                training_eligible=False, automatic_retry=False, checkpoint_promotion=False,
                independent_generalization=False, pi_native_tools=False,
                gates='eight successes per route; per-case parity; initial prompt, complete Pi replay and closure exact',
                cross_run_full_transcript_and_token_identity='reported descriptively, not a blanket numerical gate')
    publish(args.output/'plan-private.json', plan)
    print('PI_NATIVE_MATCHED_PLAN_FROZEN', sha(args.output/'plan-private.json'), flush=True)


def pi_episode(loaded, case, panel, encoding, output, pi_bin, extension):
    output.mkdir(mode=0o700)
    began = time.monotonic()
    bridge = None
    with NativeSandbox(panel, output/'sandbox') as sandbox:
        sandbox.request('setup', files=case['files'])
        def generate(prompt, budget, deadline):
            return generate_turn(loaded, prompt, encoding, budget, deadline)
        def execute(call):
            return sandbox.request('execute', call=call)
        bridge = NativePiBridge(panel, case['prompt'], encoding, generate, execute)
        try:
            terminal = serve_pi(bridge, output/'pi', pi_bin=pi_bin, extension=extension,
                                seconds=panel['episode_seconds']+60)
            names = list(case['files']) + ([case.get('output_path', 'result.json')] if case['expected_output'] is not None else [])
            snapshot = sandbox.snapshot(names)
            verdict = grade(case, bridge.final, bridge.calls, snapshot)
        finally:
            publish(output/'bridge-private.json', dict(source_messages=bridge.episode.source_messages(),
                    generations=bridge.generations, calls=bridge.calls, public_history=bridge.history,
                    final=bridge.final, reason=bridge.reason, closed=bridge.closed, close_verified=bridge.close_verified))
    replay = NativeEpisode(panel['tools'], encoding)
    turn_index = 0
    for message in bridge.episode.source_messages():
        if message['role'] == 'assistant':
            actual = hashlib.sha256(replay.prompt().encode()).hexdigest()
            if actual != bridge.generations[turn_index]['prompt_sha256']:
                raise ValueError('Pi full native prompt replay changed')
            turn_index += 1
        replay.append_source_message(message)
    if turn_index != len(bridge.generations):
        raise ValueError('Pi generation replay coverage')
    result = dict(id=case['id'], family=case['family'], final=bridge.final, reason=bridge.reason,
                  grade=verdict, autonomous_success=verdict['success'], assisted=False, supplied_calls=[],
                  calls=bridge.calls, generations=bridge.generations, snapshot=snapshot,
                  prompt_and_history=bridge.episode.text(), pi_close_verified=terminal['close_verified'],
                  pi_prompt_replay_exact=True, seconds=time.monotonic()-began)
    publish(output/'episode-private.json', result)
    return result


def run(args):
    import torch
    import tiktoken
    from ndm.e97 import load_e97_checkpoint
    from scripts.audit_e97_live_actor_capture import fingerprint, runtime
    if sha(args.plan) != args.plan_sha:
        raise ValueError('matched plan identity')
    plan = json.loads(args.plan.read_text())
    panel = plan['panel']; target = panel['models'][0]
    if (plan['max_model_episodes'] != 16 or target['sha256'] != CHECKPOINT_SHA or target['mode'] != 'train' or
            sha(target['checkpoint']) != CHECKPOINT_SHA or sha(panel['args_json']) != panel['args_sha256']):
        raise ValueError('fixed checkpoint/load mode/args identity')
    if len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) != 1 or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('exactly one leased visible GPU required')
    for pathkey, hashkey in [('source_panel', 'source_panel_sha256'), ('scripted_summary', 'scripted_summary_sha256'),
                             ('scripted_terminal', 'scripted_terminal_sha256'), ('pi_inventory', 'pi_inventory_sha256')]:
        if sha(plan[pathkey]) != plan[hashkey]:
            raise ValueError('qualification input changed')
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    loaded = load_e97_checkpoint(target['checkpoint'], args_json=panel['args_json'], device=torch.device('cuda', 0),
                                 dtype=torch.bfloat16, weight_mode='train', use_triton=True, mmap=True)
    loaded.model.eval()
    encoding = tiktoken.get_encoding('p50k_base')
    if vocabulary(encoding)[1] != panel['tokenizer_vocabulary_sha256']:
        raise ValueError('tokenizer identity')
    before = fingerprint(loaded.model)
    publish(args.output/'model-before-private.json', dict(fingerprint=before, runtime=runtime(loaded, 0), target=target))
    rows = []
    extension = Path(__file__).resolve().parents[1]/'configs/pi/e97-openhands-compat.ts'
    try:
        for index, case in enumerate(panel['cases']):
            pair = {}
            for route in plan['episode_routes'][index]:
                out = args.output/(route+'-'+case['id'])
                if route == 'direct':
                    result = episode(loaded, case, panel, encoding, out)
                else:
                    result = pi_episode(loaded, case, panel, encoding, out, plan['pi_bin'], extension)
                pair[route] = result
                print('PI_NATIVE_MATCHED_EPISODE', route, case['id'], bool(result['autonomous_success']), result['reason'], flush=True)
            d, p = pair['direct'], pair['pi']
            direct_initial = NativeEpisode(panel['tools'], encoding)
            direct_initial.append_source_message(panel.get('system_message', context('system', panel['system'])))
            direct_initial.append_source_message(context('user', case['prompt']))
            expected_prefix = direct_initial.prompt()
            first_equal = (d['prompt_and_history'].startswith(expected_prefix) and
                           p['generations'][0]['prompt_sha256'] == hashlib.sha256(expected_prefix.encode()).hexdigest())
            rows.append(dict(id=case['id'], family=case['family'], direct_success=d['autonomous_success'],
                             pi_success=p['autonomous_success'], first_prompt_equal=first_equal,
                             pi_close_verified=p['pi_close_verified'], pi_prompt_replay_exact=p['pi_prompt_replay_exact'],
                             native_transcript_equal=d['prompt_and_history'] == p['prompt_and_history'],
                             generated_tokens_equal=[g['token_ids'] for g in d['generations']] == [g['token_ids'] for g in p['generations']],
                             direct_episode_sha256=sha(args.output/('direct-'+case['id'])/'episode-private.json'),
                             pi_episode_sha256=sha(args.output/('pi-'+case['id'])/'episode-private.json')))
        checks = gate(rows)
        publish(args.output/'summary.json', dict(schema=plan['schema'], plan_sha256=args.plan_sha, rows=rows,
                checks=checks, compatibility_passed=all(checks.values()), optimizer_updates=0,
                checkpoint_promotion=False, independent_generalization=False, pi_native_tools=False))
        if not all(checks.values()):
            raise ValueError('matched compatibility gate failed')
    finally:
        after = fingerprint(loaded.model)
        no_gradients = all(p.grad is None for p in loaded.model.parameters())
        bf16 = all(p.dtype == torch.bfloat16 for p in loaded.model.parameters())
        memory = torch.cuda.max_memory_allocated()
        publish(args.output/'model-after-private.json', dict(fingerprint=after, unchanged=before == after,
                no_gradients=no_gradients, all_parameters_bf16=bf16, peak_hbm_allocated=memory))
        if before != after or not no_gradients or not bf16 or memory >= 40*1024**3:
            raise ValueError('unchanged model/gradient/dtype/HBM guard failed')


def main():
    os.umask(0o077)
    def interrupted(signum, frame):
        raise TimeoutError('matched qualification interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('freeze'); f.add_argument('--output', type=Path, required=True)
    r = sub.add_parser('run'); r.add_argument('--plan', type=Path, required=True)
    r.add_argument('--plan-sha', required=True); r.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'freeze':
        freeze(args)
    else:
        try:
            run(args)
        except BaseException as exc:
            publish(args.output/'failure.json', dict(type=type(exc).__name__, message=str(exc), optimizer_updates=0))
            raise


if __name__ == '__main__':
    main()
