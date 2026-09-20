#!/usr/bin/env python3
"""Run the E1 chat-probe acceptance panel (the operator's conversational tests).

A small deterministic panel executed per checkpoint on the CPU llama.cpp port
runner (greedy decode; no GPU is touched). For every case the harness builds
the canonical Pi-native episode prompt (Protocol header + System + User +
"Assistant:"), samples greedily via the e97-runner --panel mode, extracts the
emitted five-line frame, and applies the same verification machinery as the
hybrid-collection collector (scripts/e97_pi_native_codec.py: parse_turn,
semantic_turn, canonical round-trip). Multi-turn cases continue the episode by
feeding ToolResults from the case's frozen fixtures or the harness's own `date`
invocation — the harness NEVER executes model-chosen commands.

Verdict rules (frozen in configs/pi/e97-e1-chat-probe-panel-v1.json):
- greeting: PASS iff a valid frame with Action finish (tool==none) carries a
  non-empty finish.message, OR a text-only response (no five-line frame) that
  is non-empty and contains no tool-call Action line.
- date-tool: PASS iff a valid bash frame whose command mentions `date` is
  followed (after the harness feeds the real `date` output) by a valid finish
  frame whose message states the weekday computed from the harness clock at
  eval time (deterministic per run).
- two-tool-mini-task: PASS iff a valid read frame on the fixture path, then a
  valid bash frame mentioning both observed operands, then a valid finish
  frame stating the expected sum.

Frame validity: the canonical check (validate_generated_turn) is recorded; the
verdict uses semantic validity (parse_turn + semantic_turn against the frozen
eleven-tool manifest), because the property under test is the tool/plain-text
routing, not JSON whitespace.
"""
import argparse, datetime, hashlib, json, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tiktoken  # noqa: E402
from scripts.e97_pi_native_codec import (PiNativeEpisode, semantic_turn,  # noqa: E402
                                         parse_turn, validate_generated_turn)

EOT_ID = 50256
FRAME_LABELS = ('Analysis: ', 'Commentary: ', 'Think: ', 'Action: ', 'Arguments: ')
THINK_RESULT = 'Your thought has been logged.'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def extract_frame(text):
    """Return the first five-line frame in the emission, or None."""
    segment = text.split('\x1e', 1)[0]
    candidate = []
    for line in segment.split('\n'):
        candidate.append(line)
        if len(candidate) == 5:
            break
    if len(candidate) == 5 and all(c.startswith(l) for c, l in zip(candidate, FRAME_LABELS)):
        return '\n'.join(candidate)
    return None


def semantic_frame(frame_text, tools):
    """parse_turn + semantic_turn; returns the semantic message or raises."""
    parsed = parse_turn(frame_text)
    source = dict(role='assistant', content=parsed['content'],
                  reasoning_content=parsed['reasoning_content'], think=parsed['think'],
                  tool_calls=[{'type': 'function',
                               'function': {'name': parsed['name'],
                                            'arguments': json.dumps(parsed['arguments'],
                                                                   sort_keys=True,
                                                                   separators=(',', ':'))}}])
    return semantic_turn(source, tools)


def canonical_frame(frame_text, tools, encoding):
    try:
        validate_generated_turn(frame_text, tools, encoding)
        return True
    except ValueError:
        return False


def decode_emission(ids, encoding):
    return encoding.decode([i for i in ids if i != EOT_ID]).split('\x1e')[0]


class ProbeEpisode:
    """PiNativeEpisode with a recorded lenient continuation path for
    non-canonical frames (the verdict still fails on the property under test;
    the accumulated prompt text stays byte-exact)."""

    def __init__(self, tools, encoding, system_message):
        self.tools = tools
        self.episode = PiNativeEpisode(tools, encoding, 65536)
        self.episode.append_context({'role': 'system', 'content': system_message})
        self.noncanonical_frames = 0

    def open(self, user_message):
        self.episode.append_context({'role': 'user', 'content': user_message})
        return self.episode.prompt()

    def accept(self, frame_text):
        try:
            self.episode.accept_generated_turn(frame_text)
            return True
        except ValueError:
            self.noncanonical_frames += 1
            message = semantic_frame(frame_text, self.tools)
            self.episode._text = self.episode._text + '\n\nAssistant:\n' + frame_text
            if message['name'] == 'finish':
                self.episode._finished = True
            else:
                self.episode._pending = message['name']
                self.episode._check(self.episode._text)
            return False

    def feed_tool_result(self, name, text):
        self.episode.append_context({'role': 'toolResult', 'toolCallId': f'probe-{name}',
                                     'toolName': name,
                                     'content': [{'type': 'text', 'text': text}],
                                     'isError': False})
        return self.episode.prompt()

    def prompt(self):
        return self.episode.prompt()


def weekday_today():
    return datetime.datetime.now().strftime('%A')


def check_greeting(emission, tools):
    """Deterministic greeting verdict: (pass, path-or-reason)."""
    frame = extract_frame(emission)
    if frame is not None:
        try:
            message = semantic_frame(frame, tools)
        except ValueError as error:
            return False, f'invalid-frame:{error}'
        if message['name'] == 'finish':
            text = message['arguments']['message']
            if isinstance(text, str) and text.strip():
                return True, 'finish-frame'
            return False, 'finish-frame-empty-message'
        return False, f'tool-call-frame:{message["name"]}'
    if emission.strip():
        for line in emission.split('\n'):
            if line.startswith('Action: '):
                return False, 'text-only-with-action-line'
        return True, 'text-only'
    return False, 'empty-response'


def run_greeting(case, tools, encoding, sampler):
    result = {'id': case['id'], 'kind': case['kind'], 'turns': []}
    episode = ProbeEpisode(tools, encoding, PANEL_SYSTEM)
    prompt = episode.open(case['user_message'])
    emission = sampler(case['id'], prompt, 1)
    frame = extract_frame(emission)
    canonical = bool(frame and canonical_frame(frame, tools, encoding))
    passed, path = check_greeting(emission, tools)
    result.update({'pass': passed, 'path': path if passed else None,
                   'reason': None if passed else path,
                   'canonical_frame': canonical,
                   'noncanonical_frames': episode.noncanonical_frames,
                   'turns': [{'turn': 1, 'prompt': prompt, 'emission': emission}]})
    return result


def run_date_case(case, tools, encoding, sampler, max_turns):
    expected_weekday = weekday_today()
    date_output = subprocess.run(['date'], check=True, capture_output=True,
                                 text=True).stdout.strip()
    result = {'id': case['id'], 'kind': case['kind'], 'pass': False, 'reason': None,
              'expected_weekday': expected_weekday, 'harness_date_output': date_output,
              'turns': [], 'canonical_frames': [], 'noncanonical_frames': 0}
    episode = ProbeEpisode(tools, encoding, PANEL_SYSTEM)
    prompt = episode.open(case['user_message'])
    saw_bash_date = False
    reason = 'no-frame'
    for turn in range(1, max_turns + 1):
        emission = sampler(case['id'], prompt, turn)
        result['turns'].append({'turn': turn, 'prompt': prompt, 'emission': emission})
        frame = extract_frame(emission)
        if frame is None:
            reason = 'no-frame'
            break
        try:
            message = semantic_frame(frame, tools)
        except ValueError as error:
            reason = f'invalid-frame:{error}'
            break
        result['canonical_frames'].append(canonical_frame(frame, tools, encoding))
        episode.accept(frame)
        result['noncanonical_frames'] = episode.noncanonical_frames
        name = message['name']
        if name == 'finish':
            text = str(message['arguments'].get('message', ''))
            if not saw_bash_date:
                reason = 'finish-without-bash-date'
            elif expected_weekday.lower() not in text.lower():
                reason = f'finish-missing-weekday:{expected_weekday}:{text!r}'
            else:
                result['pass'] = True
                result['path'] = 'bash-date-then-finish-weekday'
            result['finish_message'] = text
            result['reason'] = None if result['pass'] else reason
            return result
        if name == 'think':
            prompt = episode.feed_tool_result('think', THINK_RESULT)
            continue
        if name == 'bash':
            command = str(message['arguments'].get('command', ''))
            if 'date' not in command:
                reason = f'bash-without-date:{command!r}'
                break
            saw_bash_date = True
            prompt = episode.feed_tool_result('bash', date_output)
            continue
        reason = f'unexpected-tool:{name}'
        break
    result['reason'] = reason
    return result


def run_two_tool_case(case, tools, encoding, sampler, max_turns):
    fixtures = case['fixtures']
    fixture_path = [key.split(':', 1)[1] for key in fixtures if key.startswith('read:')][0]
    operands = ['157', '284']
    expected_sum = case['expected_sum']
    result = {'id': case['id'], 'kind': case['kind'], 'pass': False, 'reason': None,
              'turns': [], 'canonical_frames': [], 'noncanonical_frames': 0}
    episode = ProbeEpisode(tools, encoding, PANEL_SYSTEM)
    prompt = episode.open(case['user_message'])
    saw_read = saw_bash = False
    reason = 'no-frame'
    for turn in range(1, max_turns + 1):
        emission = sampler(case['id'], prompt, turn)
        result['turns'].append({'turn': turn, 'prompt': prompt, 'emission': emission})
        frame = extract_frame(emission)
        if frame is None:
            reason = 'no-frame'
            break
        try:
            message = semantic_frame(frame, tools)
        except ValueError as error:
            reason = f'invalid-frame:{error}'
            break
        result['canonical_frames'].append(canonical_frame(frame, tools, encoding))
        episode.accept(frame)
        result['noncanonical_frames'] = episode.noncanonical_frames
        name = message['name']
        if name == 'finish':
            text = str(message['arguments'].get('message', ''))
            result['finish_message'] = text
            if not saw_read:
                reason = 'finish-without-read'
            elif not saw_bash:
                reason = 'finish-without-bash'
            elif expected_sum not in text:
                reason = f'finish-missing-sum:{expected_sum}:{text!r}'
            else:
                result['pass'] = True
                result['path'] = 'read-then-bash-then-finish-sum'
            result['reason'] = None if result['pass'] else reason
            return result
        if name == 'think':
            prompt = episode.feed_tool_result('think', THINK_RESULT)
            continue
        if name == 'read':
            path = str(message['arguments'].get('path', ''))
            if path != fixture_path:
                reason = f'unexpected-read-path:{path!r}'
                break
            saw_read = True
            prompt = episode.feed_tool_result('read', fixtures[f'read:{fixture_path}'])
            continue
        if name == 'bash':
            command = str(message['arguments'].get('command', ''))
            if not all(o in command for o in operands):
                reason = f'bash-missing-operands:{command!r}'
                break
            saw_bash = True
            prompt = episode.feed_tool_result('bash', fixtures['bash'])
            continue
        reason = f'unexpected-tool:{name}'
        break
    result['reason'] = reason
    return result


PANEL_SYSTEM = ('Complete the task using the declared Pi-native tools. Use actual '
                'observations, recover from errors, and finish only when complete.')


def run_case(case, tools, encoding, sampler, max_turns):
    if case['kind'] == 'greeting':
        return run_greeting(case, tools, encoding, sampler)
    if case['kind'] == 'date-tool':
        return run_date_case(case, tools, encoding, sampler, max_turns)
    if case['kind'] == 'two-tool-mini-task':
        return run_two_tool_case(case, tools, encoding, sampler, max_turns)
    raise ValueError(f'unknown case kind: {case["kind"]}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--panel', type=Path,
                  default=Path(__file__).resolve().parents[1] / 'configs' / 'pi' / 'e97-e1-chat-probe-panel-v1.json')
    p.add_argument('--panel-sha256', required=True)
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--gguf-sha256', required=True)
    p.add_argument('--runner', type=Path, required=True)
    p.add_argument('--threads', type=int, default=32)
    p.add_argument('--gen-tokens', type=int, default=256)
    p.add_argument('--checkpoint', default=None,
                   help='provenance binding: the checkpoint this GGUF was quantized from')
    p.add_argument('--checkpoint-sha256', default=None)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()

    if sha(args.panel) != args.panel_sha256:
        raise SystemExit('panel identity')
    if sha(args.gguf) != args.gguf_sha256:
        raise SystemExit('gguf identity')
    panel = json.loads(args.panel.read_text())
    repo = Path(__file__).resolve().parents[1]
    manifest_path = repo / panel['tool_manifest']
    if sha(manifest_path) != panel['tool_manifest_sha256']:
        raise SystemExit('tool manifest identity')
    manifest = json.loads(manifest_path.read_text())
    tools = manifest['model_visible_tools']
    encoding = tiktoken.get_encoding(panel['tokenizer'])
    args.output.mkdir(parents=True, exist_ok=True)
    workdir = args.output / 'passes'
    workdir.mkdir(exist_ok=True)

    def sampler(case_id, prompt, turn):
        return run_panel_pass(args.runner, args.gguf, prompt, f'{case_id}-t{turn}',
                               encoding, args.threads, args.gen_tokens, workdir)

    cases = [run_case(case, tools, encoding, sampler, panel['max_tool_turns'])
             for case in panel['cases']]
    summary = {
        'schema': 'emender-e97-e1-chat-probe-panel-run-v1',
        'status': 'measured',
        'panel': str(args.panel), 'panel_sha256': args.panel_sha256,
        'gguf': str(args.gguf), 'gguf_sha256': args.gguf_sha256,
        'checkpoint': args.checkpoint, 'checkpoint_sha256': args.checkpoint_sha256,
        'runner_sha256': sha(args.runner), 'checker_sha256': sha(Path(__file__)),
        'gen_tokens': args.gen_tokens,
        'determinism': 'greedy decode; identical weights and panel reproduce identical emissions; the date case binds the weekday to the eval-time clock',
        'cases': [{k: v for k, v in case.items() if k != 'turns'} for case in cases],
        'verdicts': {case['id']: case['pass'] for case in cases},
        'pass_count': sum(1 for case in cases if case['pass']),
        'case_count': len(cases),
    }
    for case in cases:
        (args.output / f'{case["id"]}.json').write_text(json.dumps(case, indent=2, sort_keys=True) + '\n')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print('E1_CHAT_PROBE_PANEL', summary['pass_count'], '/', summary['case_count'],
          json.dumps(summary['verdicts']), sha(args.output / 'summary.json'))
    return 0


def run_panel_pass(runner, gguf, prompt_text, case_id, encoding, threads, gen_tokens, workdir):
    ids = encoding.encode_ordinary(prompt_text)
    panel_path = workdir / f'{case_id}.panel.jsonl'
    out_path = workdir / f'{case_id}.out.json'
    panel_path.write_text(json.dumps({'id': case_id, 'token_ids': ids}) + '\n')
    subprocess.run([str(runner), '--gguf', str(gguf), '--panel', str(panel_path),
                    '--out', str(out_path), '--threads', str(threads),
                    '--gen-tokens', str(gen_tokens)], check=True)
    record = json.loads(out_path.read_text())['records'][0]
    return decode_emission(record['greedy_ids'], encoding)


if __name__ == '__main__':
    sys.exit(main())
