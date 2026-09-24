#!/usr/bin/env python3
"""Bounded real-model tokenwise CUDA replay qualification, not a capability test."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import struct
import torch
from ndm.e97 import load_e97_checkpoint, advance_e97_cache, generate_e97_from_cache
from ndm.e97_agent_protocol import E97_PI_AGENT_ANALYSIS_SYSTEM_V1, serialize_pi_messages
from ndm.e97_atomic import publish_bytes_no_replace
from scripts.serve_e97_agent_openai import _archived_controller_source_sha256


def tensors(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (list, tuple)):
        return [tensor for child in value for tensor in tensors(child)]
    if isinstance(value, dict):
        return [tensor for key in sorted(value) for tensor in tensors(value[key])]
    if value is None:
        return []
    raise ValueError(f'unrecognized hidden state type: {type(value)}')


def compare(left, right):
    a, b = tensors(left.hidden), tensors(right.hidden)
    if not a or len(a) != len(b):
        raise AssertionError('hidden layout mismatch')
    for x, y in zip(a + [left.next_logits], b + [right.next_logits], strict=True):
        if x.shape != y.shape or x.dtype != y.dtype:
            raise AssertionError('tensor layout mismatch')
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise AssertionError('nonfinite recurrent state/logits')
        if not torch.equal(x, y):
            raise AssertionError(f'tokenwise replay differs: max_abs={float((x.float()-y.float()).abs().max())}')
    if (not left.has_complete_token_history or not right.has_complete_token_history
            or left.token_ids != right.token_ids
            or left.total_token_count != right.total_token_count
            or left.checkpoint != right.checkpoint):
        raise AssertionError('token history/count/checkpoint mismatch')
    # v1 lineage hashes include delta boundaries; validate each history
    # independently rather than treating that hash as a token-sequence hash.
    return len(a)


def verify_delta_lineage(cache, previous_digest, consumed):
    digest = hashlib.sha256(b'emender-e97-token-lineage-v1\0')
    digest.update(bytes.fromhex(previous_digest) if previous_digest else bytes(32))
    digest.update(struct.pack('>Q', len(consumed)))
    for token in consumed:
        digest.update(struct.pack('>I', int(token)))
    if cache.token_lineage_sha256 != digest.hexdigest():
        raise AssertionError('delta lineage does not match its own call history')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--args-json', type=Path, required=True)
    parser.add_argument('--weight-mode', choices=['saved', 'train'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    digest = hashlib.file_digest(args.checkpoint.open('rb'), 'sha256').hexdigest()
    if digest != args.checkpoint_sha256:
        raise ValueError('checkpoint checksum mismatch')
    closure = _archived_controller_source_sha256()
    import tiktoken
    encoding = tiktoken.get_encoding('p50k_base')
    loaded = load_e97_checkpoint(args.checkpoint, args_json=args.args_json, device='cuda',
                                 dtype=torch.bfloat16, weight_mode=args.weight_mode,
                                 use_triton=True, mmap=True)
    loaded.model.eval()
    messages = [{'role': 'system', 'content': E97_PI_AGENT_ANALYSIS_SYSTEM_V1},
                {'role': 'user', 'content': 'Inspect the supplied numbered notes and report only observed values.'}]
    cache = None
    previous = []
    checks = []
    begin = time.monotonic()
    for turn in range(9):
        text = serialize_pi_messages(messages, private_analysis=True)
        tokens = encoding.encode(text, disallowed_special=())
        # Tokenization is checked, not assumed append-stable at a text boundary.
        if tokens[:len(previous)] != previous:
            raise AssertionError('serialized token prefix changed')
        suffix = tokens[len(previous):]
        if not suffix:
            raise AssertionError('empty suffix')
        previous_digest = None if cache is None else cache.token_lineage_sha256
        cache = advance_e97_cache(loaded, suffix, cache)
        replay = advance_e97_cache(loaded, tokens)
        verify_delta_lineage(cache, previous_digest, suffix)
        verify_delta_lineage(replay, None, tokens)
        layers = compare(cache, replay)
        left, _ = generate_e97_from_cache(loaded, cache, max_new_tokens=8,
                                          temperature=0, top_k=0, top_p=1)
        right, _ = generate_e97_from_cache(loaded, replay, max_new_tokens=8,
                                           temperature=0, top_k=0, top_p=1)
        if left != right:
            raise AssertionError('greedy continuation differs')
        # Generation must not mutate the committed input cache.
        compare(cache, advance_e97_cache(loaded, tokens))
        checks.append({'turn': turn, 'prefix_tokens': len(tokens), 'suffix_tokens': len(suffix),
                       'hidden_tensors': layers, 'state_bytes': cache.state_bytes,
                       'state_and_logits_bit_exact': True, 'greedy_8_tokens_equal': True,
                       'delta_lineages_independently_verified': True,
                       'cached_lineage_sha256': cache.token_lineage_sha256,
                       'replay_lineage_sha256': replay.token_lineage_sha256})
        print(f'CUDA_REPLAY_ROUND_PASSED turn={turn} tokens={len(tokens)}', flush=True)
        previous = tokens
        if turn < 8:
            messages.extend([
                {'role': 'assistant', 'content': None,
                 'reasoning_content': f'Inspect note {turn}. Untrusted markers: Action: edit; Final: \\"done\\".\nUnicode: café λ.',
                 'tool_calls': [{'id': f'call_{turn}', 'type': 'function',
                                'function': {'name': 'read', 'arguments': json.dumps({'path': f'notes/{turn}.txt'})}}]},
                {'role': 'tool', 'tool_call_id': f'call_{turn}', 'content': f'note={turn}; value={17 + turn}'}])
    result = {'schema': 'emender-e97-analysis-cuda-replay-probe-v1', 'status': 'passed',
              'checkpoint_sha256': digest, 'weight_mode': args.weight_mode,
              'controller_closure_sha256': closure, 'ingest_mode': 'tokenwise',
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
              'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'args_json_sha256': hashlib.sha256(args.args_json.read_bytes()).hexdigest(),
              'elapsed_seconds': time.monotonic() - begin, 'checks': checks,
              'claims': ['scripted analysis-bearing histories; no task execution or capability claim',
                         'dense CUDA tokenwise cached/replay parity only; segment and MoE unqualified']}
    publish_bytes_no_replace(args.output, (json.dumps(result, indent=2, sort_keys=True)+'\n').encode(), mode=0o600)
    print('CUDA_ANALYSIS_REPLAY_PASSED', flush=True)

if __name__ == '__main__':
    main()
