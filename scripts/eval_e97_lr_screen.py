#!/usr/bin/env python3
"""Direct dense tokenwise screening, deliberately not an HTTP/Pi admission gate."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import torch
import tiktoken
from ndm.e97 import load_e97_checkpoint
from ndm.e97_agent_server import TorchE97AgentEngine
from ndm.e97_agent_protocol import serialize_pi_messages, parse_agent_turn, AgentProtocolError
from ndm.e97_atomic import publish_bytes_no_replace
from scripts.e97_lr_screen_panel import read_fixture, score
from scripts.serve_e97_agent_openai import _archived_controller_source_sha256


def evaluate_task(engine, encoding, panel, task):
    messages = [{'role': 'system', 'content': panel['system']}, {'role': 'user', 'content': task['user']}]
    reads, missing, turns = [], [], []
    previous, cache, final, error = [], None, None, None
    began = time.monotonic()
    first_valid = False
    seen = set()
    with tempfile.TemporaryDirectory(prefix='e97-lr-screen-') as directory:
        root = Path(directory)
        for path, content in task['fixtures'].items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        for n in range(panel['decode']['max_turns']):
            ids = engine.encode(serialize_pi_messages(messages, private_analysis=True))
            prefix_hit = bool(previous) and ids[:len(previous)] == previous
            cache = engine.advance(ids[len(previous):] if prefix_hit else ids, cache if prefix_hit else None)
            generated, _ = engine.generate(cache, max_new_tokens=panel['decode']['max_output_tokens'], temperature=0, top_p=1)
            text = engine.decode(generated)
            record = {'completion_tokens': len(generated), 'prompt_tokens': len(ids),
                      'suffix_tokens': len(ids)-len(previous) if prefix_hit else len(ids),
                      'cache_event': 'hit' if prefix_hit else ('miss' if not previous else 'replay'),
                      'generated_text': text}
            turns.append(record)
            try:
                if not 1 <= len(generated) <= panel['decode']['max_output_tokens']:
                    raise ValueError('invalid generation token count')
                turn = parse_agent_turn(text, private_analysis=True)
                if len(encoding.encode(turn.private_analysis, disallowed_special=())) > panel['decode']['max_analysis_tokens']:
                    raise ValueError('analysis token cap exceeded')
                if n == 0:
                    first_valid = True
                assistant = {'role': 'assistant', 'content': turn.final_text if turn.kind == 'final' else None,
                             'reasoning_content': turn.private_analysis}
                if turn.kind == 'final':
                    final = turn.final_text
                    messages.append(assistant)
                    break
                if turn.tool_name != 'read':
                    raise ValueError('tool unavailable in read-only screen')
                key = json.dumps(turn.arguments, sort_keys=True)
                # Fixture is immutable: identical reads cannot gain information here.
                if key in seen:
                    raise ValueError('no_progress: identical read in unchanged fixture')
                seen.add(key)
                observation, absent = read_fixture(root, turn.arguments)
                (missing if absent else reads).append(turn.arguments['path'])
                call_id = f'call_{n}'
                assistant['tool_calls'] = [{'id': call_id, 'type': 'function', 'function': {
                    'name': 'read', 'arguments': turn.arguments_json}}]
                messages.extend([assistant, {'role': 'tool', 'tool_call_id': call_id, 'content': observation}])
                record['observation'] = observation
                previous = ids
            except (AgentProtocolError, ValueError, OSError) as exc:
                error = str(exc)
                break
    return {'id': task['id'], 'kind': task['kind'], 'success': error is None and score(task, final, reads, missing),
            'first_turn_protocol_valid': first_valid, 'error': error, 'final': final,
            'reads': reads, 'missing_reads': missing, 'turns': turns,
            'elapsed_seconds': time.monotonic()-began}


def select_cuda_device():
    # NUMA placement does not select a CUDA device. torchrun exposes all leased
    # devices to each worker, so plain 'cuda' otherwise sends every rank to 0.
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if not 0 <= local_rank < torch.cuda.device_count():
        raise RuntimeError(f'LOCAL_RANK={local_rank} is outside visible CUDA devices')
    torch.cuda.set_device(local_rank)
    return f'cuda:{local_rank}'


def main():
    p = argparse.ArgumentParser()
    for name in ('checkpoint', 'args-json', 'panel', 'output-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--panel-sha256', required=True)
    p.add_argument('--weight-mode', choices=['saved', 'train'], required=True)
    args = p.parse_args()
    panel_bytes = args.panel.read_bytes()
    assert hashlib.sha256(panel_bytes).hexdigest() == args.panel_sha256
    panel = json.loads(panel_bytes)
    with args.checkpoint.open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == args.checkpoint_sha256
    closure = _archived_controller_source_sha256()
    rank, world = int(os.environ.get('RANK', '0')), int(os.environ.get('WORLD_SIZE', '1'))
    output = args.output_root / f'rank-{rank:02d}.json'
    if output.exists():
        raise FileExistsError(output)
    device = select_cuda_device()
    print(f'LR_SCREEN_DEVICE rank={rank} local_rank={os.environ.get("LOCAL_RANK", "0")} device={device}', flush=True)
    loaded = load_e97_checkpoint(args.checkpoint, args_json=args.args_json, device=device,
                                 dtype=torch.bfloat16, weight_mode=args.weight_mode, use_triton=True, mmap=True)
    loaded.model.eval()
    engine = TorchE97AgentEngine(loaded, ingest_mode='tokenwise', weight_mode=args.weight_mode,
                                 device=device, dtype='bfloat16', use_triton=True,
                                 checkpoint_sha256=args.checkpoint_sha256, private_analysis=True)
    encoding = tiktoken.get_encoding('p50k_base')
    results = []
    for task in panel['tasks'][rank::world]:
        result = evaluate_task(engine, encoding, panel, task)
        results.append(result)
        print(json.dumps({'rank': rank, 'task': result['id'], 'success': result['success']}), flush=True)
    artifact = {'schema': 'emender-e97-lr-screen-shard-v1', 'checkpoint_sha256': args.checkpoint_sha256,
                'weight_mode': args.weight_mode, 'panel_sha256': args.panel_sha256,
                'rank': rank, 'world_size': world, 'cuda_device': device,
                'controller_closure_sha256': closure,
                'evaluator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'claims': 'direct tokenwise engine only; not HTTP/Pi or production admission', 'results': results}
    publish_bytes_no_replace(output, (json.dumps(artifact, indent=2, sort_keys=True)+'\n').encode(), mode=0o600)
    print(f'LR_SCREEN_SHARD_COMPLETE rank={rank}', flush=True)

if __name__ == '__main__':
    main()
