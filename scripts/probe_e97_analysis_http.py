#!/usr/bin/env python3
"""Diagnostic real-model HTTP round trip with a two-file, read-only fixture.

Not a formal collection controller, training authority, or held-out capability panel.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request
import urllib.error

from ndm.e97_acquisition_controller import READ_OBSERVE_TOOLS
from ndm.e97_agent_protocol import E97_PI_AGENT_ANALYSIS_SYSTEM_V1
from ndm.e97_onpolicy_records import sha256_json, validate_service_attestation
from ndm.e97_atomic import publish_bytes_no_replace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weight-mode', choices=['saved', 'train'], required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    messages = [
        {'role': 'system', 'content': E97_PI_AGENT_ANALYSIS_SYSTEM_V1},
        {'role': 'user', 'content': 'Runtime plumbing test. Read notes/first.txt, then notes/second.txt using read with offset=1 and limit=10. Report the two values as first=value, second=value. Do not write or change anything.'},
    ]
    receipts = []
    seen = []
    outcome = 'turn_limit'
    error = None
    final = None
    with tempfile.TemporaryDirectory(prefix='e97-http-readonly-fixture-') as directory:
        fixture = Path(directory)
        (fixture / 'notes').mkdir()
        (fixture / 'notes/first.txt').write_text('value=41\n')
        (fixture / 'notes/second.txt').write_text('value=83\n')
        try:
            for turn in range(8):
                request = {'model': 'e97-dense-agent', 'messages': messages,
                           'tools': list(READ_OBSERVE_TOOLS), 'temperature': 0, 'max_tokens': 4096}
                body = json.dumps(request).encode()
                wire = urllib.request.Request('http://127.0.0.1:24980/v1/chat/completions', data=body,
                                              headers={'Content-Type': 'application/json',
                                                       'Authorization': 'Bearer local',
                                                       'x-session-id': 'analysis-http-plumbing-v1'})
                with urllib.request.urlopen(wire, timeout=600) as response:
                    payload = response.read((4 << 20) + 1)
                    if len(payload) > 4 << 20:
                        raise ValueError('HTTP response exceeds bound')
                    cache = response.headers.get('x-emender-cache')
                value = json.loads(payload)
                assistant = value['choices'][0]['message']
                attestation = validate_service_attestation(value['emender_service_attestation'])
                if (attestation['schema'] != 'emender-e97-agent-service-attestation-v2'
                        or attestation['weight_mode'] != args.weight_mode
                        or attestation['checkpoint_sha256'] != '6881acf1d79f60ea910752277ac4809bdd39d6d7598d47489c7e9bc3a1df6fcd'):
                    raise ValueError('service attestation identity mismatch')
                if value.get('emender_assistant_message_sha256') != sha256_json(assistant):
                    raise ValueError('server-bound assistant digest mismatch')
                reasoning = assistant.get('reasoning_content')
                if not isinstance(reasoning, str) or not reasoning.strip():
                    raise ValueError('missing private reasoning')
                count = value['usage']['completion_tokens']
                if type(count) is not int or not 1 <= count <= 4096:
                    raise ValueError('invalid completion token accounting')
                if cache != ('miss' if turn == 0 else 'hit'):
                    raise ValueError(f'unexpected cache outcome: {cache}')
                receipts.append({'request_sha256': hashlib.sha256(body).hexdigest(),
                                 'response': value, 'cache': cache})
                # Preserve the exact server assistant object, including reasoning_content.
                messages.append(assistant)
                calls = assistant.get('tool_calls') or []
                if not calls:
                    final = assistant.get('content')
                    outcome = 'completed'
                    break
                if len(calls) != 1 or calls[0]['function']['name'] != 'read':
                    raise ValueError('unexpected tool call; no dispatch performed')
                call = calls[0]
                arguments = json.loads(call['function']['arguments'])
                if (set(arguments) != {'path', 'offset', 'limit'}
                        or arguments['path'] not in {'notes/first.txt', 'notes/second.txt'}
                        or type(arguments['offset']) is not int or arguments['offset'] < 1
                        or type(arguments['limit']) is not int or not 1 <= arguments['limit'] <= 10):
                    raise ValueError('arguments outside the read-only fixture bounds')
                path = fixture / arguments['path']
                lines = path.read_text().splitlines()
                start = arguments['offset'] - 1
                observation = '\n'.join(f'{i+1}: {line}' for i, line in enumerate(lines)
                                        if start <= i < start + arguments['limit']) or '(no lines)'
                seen.append(arguments['path'])
                messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': observation})
        except urllib.error.HTTPError as exc:
            outcome = 'http_rejected'
            error = {'status': exc.code, 'body': exc.read(1 << 20).decode(errors='replace')}
        except Exception as exc:
            outcome = 'probe_failed'
            error = {'type': type(exc).__name__, 'message': str(exc)}
    passed = outcome == 'completed' and seen == ['notes/first.txt', 'notes/second.txt'] and len(receipts) == 3
    result = {'schema': 'emender-e97-analysis-http-probe-v1', 'status': 'passed' if passed else 'failed',
              'outcome': outcome, 'error': error, 'training_eligible': False,
              'weight_mode': args.weight_mode,
              'purpose': 'runtime plumbing; not held-out capability evidence',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'tool_paths': seen, 'final': final, 'turns': receipts}
    publish_bytes_no_replace(args.output, (json.dumps(result, indent=2, sort_keys=True)+'\n').encode(), mode=0o600)
    print(json.dumps({'status': result['status'], 'outcome': outcome, 'turns': len(receipts),
                      'tool_calls': len(seen), 'output': str(args.output)}), flush=True)
    if not passed:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
