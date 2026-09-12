#!/usr/bin/env python3
"""Bounded generated native-tool episodes against fresh host-owned paired oracles."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

from ndm.e97_atomic import publish_bytes_no_replace
from scripts.e97_native_execution_cases import SCHEMA, SYSTEM, aggregate_results, cases, context, grade
from scripts.e97_native_execution_sandbox import NativeSandbox
from scripts.e97_open_swe_native_runtime_protocol import NativeEpisode, validate_generated_turn


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def publish(path, value):
    publish_bytes_no_replace(Path(path), (json.dumps(value, indent=2, sort_keys=True)+'\n').encode(), mode=0o400)


def freeze(args):
    out = args.output
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, mode=0o700)
    r = args.training
    final = json.loads((r/'segment-000880/evaluation/panel.json').read_text())
    completed = json.loads((r/'program-completion.json').read_text())
    if not completed['fixed_budget_complete'] or completed['updates'] != 880:
        raise ValueError('training not complete')
    early = json.loads((r/'segment-000128/summary.json').read_text())['checkpoint']
    parent = dict(final['models'][0]); parent['name'] = 'parent-y'
    models = [parent, dict(name='u128-y', checkpoint=early['path'], sha256=early['sha256'], mode='train'),
              {**final['models'][2], 'name': 'u880-y'}, {**final['models'][3], 'name': 'u880-x'}]
    for m in models:
        if sha(m['checkpoint']) != m['sha256']:
            raise ValueError('checkpoint identity')
    runtime = args.runtime
    receipt = runtime/'native-runtime-execution-v2/results/summary.json'
    if sha(receipt) != '46584aa4538682e4575ad7d78905b3390392f2023aa5aaf289d400db7ecb3f59':
        raise ValueError('executor qualification identity')
    manifest = runtime/'native-runtime-image-build-v2/manifest.json'
    if sha(manifest) != 'eaab33b7d60b2989e05344e60d092990f04f3f4547f521d9e343b3361c38a98d':
        raise ValueError('image manifest identity')
    image = json.loads(manifest.read_text())
    for path, digest in image['files'].items():
        if sha(manifest.parent/path) != digest:
            raise ValueError('image evidence identity')
    tools_path = runtime/'native-runtime-execution-v2/bundle/expected-tools.json'
    tools = json.loads(tools_path.read_text())
    panel = dict(schema=SCHEMA, training_eligible=False, models=models, cases=cases('native-execution-diagnostic-20260911-v1'),
                 args_json=final['args_json'], args_sha256=final['args_sha256'], tools=tools['tools'],
                 tools_sha256=sha(tools_path), tokenizer_vocabulary_sha256=tools['tokenizer_vocabulary_sha256'],
                 image_id=image['image_id'], image_manifest=str(manifest), image_manifest_sha256=sha(manifest),
                 executor_receipt_sha256=sha(receipt), training_completion_sha256=sha(r/'program-completion.json'),
                 snapshot_method='paused-pid-namespace-reader-v2',
                 snapshot_reader_sha256=sha('scripts/e97_native_snapshot_reader.py'),
                 bundle_files={p: sha(p) for p in ('scripts/e97_native_execution_rpc.py', 'scripts/e97_openhands_native_backend.py')},
                 max_turns=8, generation_budget=4096, episode_generation_budget=8192, episode_seconds=600,
                 system=SYSTEM, automatic_retry=False, checkpoint_promotion=False,
                 scope='Four authored paired families, eight cases/model. No training, tool translation, source admission or repository-benchmark claim.')
    publish(out/'panel.json', panel)
    print('NATIVE_EXECUTION_PANEL_FROZEN', sha(out/'panel.json'), flush=True)


def load_panel(args):
    if sha(args.panel) != args.panel_sha:
        raise ValueError('panel identity')
    p = json.loads(args.panel.read_text())
    if p['schema'] != SCHEMA or p['training_eligible'] is not False:
        raise ValueError('panel schema')
    return p


def generate_turn(loaded, prompt, encoding, budget, deadline):
    import torch
    from ndm.e97 import advance_e97_cache_segment, generate_e97_from_cache
    prefix = encoding.encode_ordinary(prompt)
    if len(prefix)+budget > 65536:
        return None, [], 'context_budget'
    ids = []; reason = 'generation_budget'
    with torch.no_grad():
        # Exact whole causal prompt replay at each turn, no compaction or implicit cache splice.
        cache = advance_e97_cache_segment(loaded, prefix)
        for _ in range(budget):
            if time.monotonic() >= deadline:
                reason = 'episode_deadline'; break
            new, cache = generate_e97_from_cache(loaded, cache, max_new_tokens=1, temperature=0.,
                                                 top_k=0, top_p=0., stop_token_ids=(218,))
            if not new:
                reason = 'empty'; break
            ids.extend(new); text = encoding.decode(ids)
            if not ('Analysis: '.startswith(text) or text.startswith('Analysis: ')):
                reason = 'invalid_opening'; break
            if text.count('\n') > 4:
                reason = 'invalid_frame'; break
            if text.count('\n') == 4 and text.endswith('}'):
                try:
                    validate_generated_turn(text, encoding)
                except ValueError:
                    pass
                else:
                    return text, ids, 'valid'
            if 218 in new:
                reason = 'separator_before_valid_turn'; break
    return None, ids, reason


def episode(loaded, case, panel, encoding, output):
    e = NativeEpisode(panel['tools'], encoding)
    e.append_source_message(panel.get('system_message', context('system', panel['system'])))
    e.append_source_message(context('user', case['prompt']))
    supplied_calls = []; calls = []; generations = []; final = None; tokens = 0; reason = 'turn_budget'
    began = time.monotonic()
    interventions = case.get('supplied_calls', [])
    if len(interventions) > 1:
        raise ValueError('at most one supplied read')
    for call in interventions:
        if (call.get('name') != 'str_replace_editor' or
            call.get('arguments') not in [dict(command='view', path='/testbed/'+p) for p in case['files']]):
            raise ValueError('only a complete fixture read may be supplied')
    with NativeSandbox(panel, output) as sandbox:
        sandbox.request('setup', files=case['files'])
        for supplied in case.get('supplied_calls', []):
            from scripts.e97_open_swe_native_codec import native_turn, compact
            message = dict(role='assistant', content=None, reasoning_content=None, think=None,
                           tool_calls=[{'type':'function', 'function':{'name':supplied['name'],
                                        'arguments':compact(supplied['arguments'])}}])
            turn = e.accept_generated_turn(native_turn(message))
            if turn.backend_call() != supplied:
                raise ValueError('supplied call changed')
            reply = sandbox.request('execute', call=supplied)
            if 'dispatch_error' in reply or reply['result']['message']['content'].startswith('ERROR:'):
                raise ValueError('supplied read did not succeed')
            supplied_calls.append(dict(request=supplied, result=reply['result'], origin='authored-intervention'))
            e.append_observation(reply['result']['message'])
        deadline = time.monotonic()+panel['episode_seconds']
        for index in range(panel['max_turns']):
            budget = min(panel['generation_budget'], panel['episode_generation_budget']-tokens)
            if budget <= 0:
                reason = 'episode_generation_budget'; break
            text, ids, reason = generate_turn(loaded, e.prompt(), encoding, budget, deadline)
            generations.append(dict(turn=index, token_ids=ids, reason=reason)); tokens += len(ids)
            if text is None:
                break
            try:
                turn = e.accept_generated_turn(text)
                call = turn.backend_call()
                if e.finished:
                    final = turn.public_events()[-1]['text']; reason = 'finished'; break
                if call is None:
                    # Upstream Runtime base.py AgentThinkObservation uses this exact text.
                    e.append_observation(context('tool', 'Your thought has been logged.'))
                    continue
                try:
                    reply = sandbox.request('execute', call=call)
                except TimeoutError:
                    calls.append(dict(request=call, dispatch_error='host_tool_deadline'))
                    reason = 'host_tool_deadline'; break
                if 'dispatch_error' in reply:
                    calls.append(dict(request=call, **reply)); reason = 'invalid_tool_request'; break
                result = reply['result']; calls.append(dict(request=call, result=result))
                e.append_observation(result['message'])
            except ValueError:
                reason = 'protocol_error'; break
        else:
            reason = 'turn_budget'
        names = list(case['files']) + (['result.json'] if case['expected_output'] is not None else [])
        snapshot = sandbox.snapshot(names)
    verdict = grade(case, final, supplied_calls+calls, snapshot)
    result = dict(id=case['id'], family=case['family'], reason=reason, final=final, calls=calls,
                  supplied_calls=supplied_calls, assisted=bool(supplied_calls),
                  autonomous_success=verdict['success'] and not supplied_calls,
                  generations=generations, prompt_and_history=e.text(), snapshot=snapshot,
                  grade=verdict, seconds=time.monotonic()-began)
    publish(Path(output)/'episode-private.json', result)
    return result


def run(args):
    import torch
    import tiktoken
    from ndm.e97 import load_e97_checkpoint
    from scripts.e97_open_swe_native_codec import vocabulary
    panel = load_panel(args)
    if any(c.get('supplied_calls') for c in panel['cases']):
        raise ValueError('interventions require the separately labeled grounding diagnostic')
    rank = int(os.environ['RANK']); local = int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != 8:
        raise ValueError('eight evaluation ranks required')
    target = panel['models'][rank//2]
    if sha(target['checkpoint']) != target['sha256'] or sha(panel['args_json']) != panel['args_sha256']:
        raise ValueError('model input identity')
    torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    loaded = load_e97_checkpoint(target['checkpoint'], args_json=panel['args_json'], device=torch.device('cuda', local),
                                 dtype=torch.bfloat16, weight_mode=target['mode'], use_triton=True, mmap=True)
    loaded.model.eval(); encoding = tiktoken.get_encoding('p50k_base')
    if vocabulary(encoding)[1] != panel['tokenizer_vocabulary_sha256']:
        raise ValueError('tokenizer identity')
    results = []
    for c in panel['cases']:
        if c['pair_index'] % 2 != rank % 2:
            continue
        result = episode(loaded, c, panel, encoding, args.output/f'rank-{rank}-{c["id"]}')
        results.append({k: result[k] for k in ('id', 'family', 'reason', 'grade', 'seconds')})
        print(json.dumps(dict(event='native_episode_complete', model=target['name'], id=c['id'],
                              success=result['grade']['success'], reason=result['reason'])), flush=True)
    publish(args.output/f'rank-{rank}.json', dict(rank=rank, model=target, panel_sha256=args.panel_sha, results=results,
                                                 peak_hbm_allocated=torch.cuda.max_memory_allocated()))


def smoke(args):
    panel = load_panel(args); results = []
    for c in panel['cases']:
        calls = []
        with NativeSandbox(panel, args.output/('smoke-'+c['id'])) as sandbox:
            sandbox.request('setup', files=c['files'])
            if c['family'] == 'recovery':
                call = dict(name='str_replace_editor', arguments=dict(command='view', path='/testbed/missing.json'))
                reply = sandbox.request('execute', call=call); calls.append(dict(request=call, result=reply['result']))
            command = 'cat ' + ' '.join('/testbed/'+p for p in c['files'])
            if c['family'] == 'edit':
                command += '; python -c \'import json; p=json.load(open("/testbed/state.json")); p["count"]+=7; json.dump(p,open("/testbed/result.json","w"))\''
            call = dict(name='execute_bash', arguments=dict(command=command, timeout=10))
            reply = sandbox.request('execute', call=call); calls.append(dict(request=call, result=reply['result']))
            snapshot = sandbox.snapshot(list(c['files']) + (['result.json'] if c['expected_output'] is not None else []))
        verdict = grade(c, c['answer'], calls, snapshot)
        publish(args.output/('smoke-'+c['id'])/'oracle-check.json',
                dict(id=c['id'], calls=calls, snapshot=snapshot, verdict=verdict, panel_sha256=args.panel_sha))
        if not verdict['success']:
            raise ValueError('authored oracle/executor preflight: '+c['id']+': '+str(verdict))
        results.append(dict(id=c['id'], **verdict))
    publish(args.output/'smoke-summary.json', dict(status='passed', checks=results, model_loaded=False, panel_sha256=args.panel_sha))
    print('NATIVE_EXECUTION_SMOKE_PASSED', flush=True)


def aggregate(args):
    panel = load_panel(args); reports = {}
    for rank in range(8):
        r = json.loads((args.output/f'rank-{rank}.json').read_text())
        if r['panel_sha256'] != args.panel_sha:
            raise ValueError('rank panel identity')
        reports[rank] = r
    summary = aggregate_results(panel, reports); summary['panel_sha256'] = args.panel_sha
    publish(args.output/'summary.json', summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


def main():
    def interrupted(signum, frame):
        raise TimeoutError('evaluation interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('freeze')
    for name in ('training', 'runtime', 'output'):
        f.add_argument('--'+name, type=Path, required=True)
    for name in ('run', 'smoke', 'aggregate'):
        q = sub.add_parser(name); q.add_argument('--panel', type=Path, required=True)
        q.add_argument('--panel-sha', required=True); q.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); globals()[a.command](a)


if __name__ == '__main__':
    main()
