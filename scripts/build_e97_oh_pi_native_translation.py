#!/usr/bin/env python3
"""Translate OpenHands source-native trajectories into canonical Pi-native records.

Phase 1 of the fail-closed OH-repair pipeline (plan:
docs/validation/e97-oh-translation-and-reasoning-integration-plan-v1.md, T1).

Each raw OpenHands trajectory (nvidia/Open-SWE-Traces derivative, previously
rendered under profile e97-open-swe-source-native-v1 with the OH tool
vocabulary) is re-rendered under profile e97-pi-native-v1 with the canonical
eleven-tool surface extracted from the gate-passing collections:

  str_replace_editor view (file)   -> read  {path, offset?, limit?}
  str_replace_editor view (dir)    -> bash  {command: "find <path> -maxdepth 2 ..."}
  str_replace_editor str_replace   -> edit  {path, edits:[{oldText,newText}]}
  str_replace_editor create        -> write {path, content}
  execute_bash                     -> bash  {command, timeout?}
  think / finish                   -> think / finish (pseudo actions)

The five-line turn frame, block layout, compaction and record separator are
byte-compatible with scripts/e97_pi_native_codec.py. Assistant turns are the
supervised targets; context blocks are never targets.

Outputs (under --output-dir):
  candidates/   one JSON per trajectory: ordered episode blocks (assistant turn
               texts final; ToolResult texts provisional), the mapped action
               plan with parsed observation expectations for the replay
               verifier, and provenance. Oracle metadata (reference/model
               patches) is never copied into candidates.
  think-export.jsonl  grounded pre-action commentary rows for the
                reasoning-rehearsal cohort (T2).
  translate-report.json  counts and drop reasons.

No record is training-eligible until scripts/verify_e97_oh_translation_replay.py
replays it against the recorded repository state and it passes; the verifier
assembles the final record text from the verified observations.
"""
import argparse, hashlib, json, sys
from pathlib import Path

RS = '\x1e'
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from scripts.e97_open_swe_native_codec import compact, strict_json  # noqa: E402

SYSTEM_PROMPT = ('Complete the task using the declared Pi-native tools. Use actual '
                 'observations, recover from errors, and finish only when complete.')
DIR_LISTING_MARKER = "Here's the files and directories up to"


def sha8(text):
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def parse_catn_body(body):
    """Parse a `cat -n` observation body into {lineno: content}."""
    lines = {}
    for line in body.split('\n'):
        m = re.match(r'^\s*(\d+)\t(.*)$', line) if (re := __import__('re')) else None
        if m:
            lines[int(m.group(1))] = m.group(2)
    return lines


def split_view_observation(text):
    """Split an OH view observation into (header, {lineno: content}); None if not a cat -n view."""
    marker = '`cat -n` on'
    idx = text.find(marker)
    if idx < 0:
        return None, {}
    nl = text.find('\n', idx)
    body = text[nl + 1:] if nl >= 0 else ''
    return text[:idx], parse_catn_body(body)


def parse_bash_observation(text):
    """Split an OH bash observation into (core_output, exit_code)."""
    import re
    core = text
    exit_code = 0
    m = re.search(r'\[The command (?:completed|exited) with (?:exit code|code) (\d+)\.\]', text)
    if m:
        exit_code = int(m.group(1))
        core = text[:m.start()] + text[m.end():]
    core = re.sub(r'\[Current working directory: [^\]]*\]\n?', '', core)
    core = re.sub(r'\[The command exceeded[^\]]*\]\n?', '', core)
    core = core.strip('\n')
    if core and not core.endswith('\n'):
        core += '\n'
    return core, exit_code


def map_view_args(args, recorded_observation):
    """Map an OH view to read args, or to a bash find for directory listings."""
    import re
    if recorded_observation and recorded_observation.startswith(DIR_LISTING_MARKER):
        path = args['path']
        command = f"find {path} -maxdepth 2 -not -path '*/.*' | sort"
        return 'bash', {'command': command}, {'kind': 'dir_listing', 'path': path}
    vr = args.get('view_range')
    if vr is None:
        return 'read', {'path': args['path']}, {'kind': 'file_read'}
    if (not isinstance(vr, list) or len(vr) != 2 or not all(isinstance(v, int) for v in vr)
            or vr[0] < 1):
        raise ValueError('invalid_view_range')
    if vr[1] == -1:
        return 'read', {'path': args['path'], 'offset': vr[0]}, {'kind': 'file_read'}
    if vr[1] < vr[0]:
        raise ValueError('invalid_view_range')
    return 'read', {'path': args['path'], 'offset': vr[0], 'limit': vr[1] - vr[0] + 1}, {'kind': 'file_read'}


def map_action(name, args, recorded_observation=None):
    """Map one OH action to (pi_name, pi_args, expectation). Raises when untranslatable."""
    if name == 'execute_bash':
        if 'command' not in args:
            raise ValueError('malformed_bash_args')
        out = {'command': args['command']}
        if 'timeout' in args:
            out['timeout'] = args['timeout']
        core, exit_code = parse_bash_observation(recorded_observation or '')
        return 'bash', out, {'kind': 'bash', 'recorded_core': core, 'recorded_exit': exit_code}
    if name == 'think':
        if 'thought' not in args:
            raise ValueError('malformed_think_args')
        return 'think', {'thought': args['thought']}, {'kind': 'think'}
    if name == 'finish':
        if 'message' not in args:
            raise ValueError('malformed_finish_args')
        return 'finish', {'message': args['message']}, {'kind': 'finish'}
    if name == 'str_replace_editor':
        command = args.get('command')
        if command == 'view':
            return map_view_args(args, recorded_observation)
        if command == 'str_replace':
            if 'old_str' not in args or 'new_str' not in args:
                raise ValueError('malformed_str_replace_args')
            return 'edit', {'path': args['path'],
                            'edits': [{'oldText': args['old_str'], 'newText': args['new_str']}]}, \
                   {'kind': 'edit'}
        if command == 'create':
            if 'file_text' not in args:
                raise ValueError('malformed_create_args')
            return 'write', {'path': args['path'], 'content': args['file_text']}, {'kind': 'write'}
        raise ValueError(f'untranslatable_str_replace_editor_command:{command}')
    raise ValueError(f'untranslatable_action:{name}')


READ_MAX_LINES = 2000
READ_MAX_BYTES = 50 * 1024


def render_read_output(lines_text, offset, limit):
    """Render the sandbox read tool output for a verified file state.

    Mirrors pi-runtime/dist/core/tools/read.js: raw content slice, truncation
    notices at 2000 lines / 50KB, and the "[N more lines in file...]" notice
    when a user limit stops early.
    """
    all_lines = lines_text.split('\n')
    start = (offset - 1) if offset else 0
    if start >= len(all_lines):
        return None, f'Offset {offset} is beyond end of file ({len(all_lines)} lines total)'
    if limit is not None:
        selected = all_lines[start:start + limit]
        remaining = len(all_lines) - (start + len(selected))
    else:
        selected = all_lines[start:]
        remaining = 0
    out = '\n'.join(selected)
    if len(selected) > READ_MAX_LINES or len(out.encode()) > READ_MAX_BYTES:
        kept = []
        used = 0
        for line in selected[:READ_MAX_LINES]:
            if used + len(line.encode()) + 1 > READ_MAX_BYTES:
                break
            kept.append(line)
            used += len(line.encode()) + 1
        shown = start + len(kept)
        out = '\n'.join(kept)
        out += (f'\n\n[Showing lines {start + 1}-{shown} of {len(all_lines)}. '
                f'Use offset={shown + 1} to continue.]')
    elif limit is not None and remaining > 0:
        out += f'\n\n[{remaining} more lines in file. Use offset={start + len(selected) + 1} to continue.]'
    return out, None


class Translator:
    def __init__(self, surface):
        self.header = compact({'instructions': surface['instructions'],
                               'profile': surface['profile'],
                               'pseudo_actions': surface['pseudo_actions'],
                               'tools': surface['tools']})

    def translate(self, row):
        """Translate one raw trajectory row into a candidate block episode."""
        import re
        messages = row['messages'] if 'messages' in row else row['source']['messages']
        trajectory_id = row.get('trajectory_id') or row.get('source', {}).get('trajectory_id')

        # Pass 1: pair each assistant action with its observation (OH delivers the
        # tool message after the assistant message; think actions carry only an
        # acknowledgment that we render ourselves).
        events = []
        seen_user = False
        finished = False
        pending = None
        for msg in messages:
            role = msg.get('role')
            if pending not in (None, 'think-ack') and role != 'tool':
                return None, f'observation_order:{pending}'
            if role == 'system':
                continue
            if role == 'user':
                seen_user = True
                events.append(('user', msg))
            elif role == 'assistant':
                if not seen_user:
                    return None, 'assistant_without_user'
                calls = msg.get('tool_calls') or []
                if len(calls) != 1:
                    return None, f'call_count:{len(calls)}'
                events.append(('assistant', msg))
                call_name = (calls[0].get('function') or {}).get('name')
                pending = None if call_name == 'finish' else 'awaiting'
            elif role == 'tool':
                if pending is None:
                    return None, 'orphan_observation'
                if pending == 'think-ack':
                    pending = None
                    continue
                if pending != 'awaiting' or not events or events[-1][0] != 'assistant':
                    return None, 'observation_order'
                raw = msg.get('content')
                events[-1] = ('assistant', events[-1][1],
                              raw if isinstance(raw, str) else json.dumps(raw))
                pending = None
            else:
                return None, f'unknown_role:{role}'
        if not seen_user:
            return None, 'no_user_message'
        if pending is not None:
            return None, 'missing_final_observation'

        # Pass 2: map actions and emit blocks.
        blocks = [{'kind': 'protocol', 'header': self.header},
                  {'kind': 'system', 'content': SYSTEM_PROMPT}]
        actions = []
        think_rows = []
        finished = False
        fid = sha8(str(trajectory_id))
        for event in events:
            kind = event[0]
            if kind == 'user':
                blocks.append({'kind': 'user', 'content': event[1].get('content')})
                continue
            msg = event[1]
            observation = event[2] if len(event) > 2 else None
            calls = msg.get('tool_calls') or []
            fn = calls[0].get('function') or {}
            raw_name = fn.get('name')
            try:
                raw_args = strict_json(fn.get('arguments') or '{}')
            except Exception:
                return None, 'bad_arguments_json'
            try:
                pi_name, pi_args, expectation = map_action(raw_name, raw_args, observation)
            except ValueError as exc:
                return None, str(exc)
            reasoning = msg.get('reasoning_content')
            content = msg.get('content')
            think_flag = msg.get('think')
            if reasoning is not None and not isinstance(reasoning, str):
                return None, 'bad_reasoning'
            turn = '\n'.join((
                'Analysis: ' + compact(reasoning if reasoning else None),
                'Commentary: ' + compact(content if content else ''),
                'Think: ' + compact(think_flag if isinstance(think_flag, bool) else None),
                'Action: ' + pi_name,
                'Arguments: ' + compact(pi_args)))
            call_counter = len(actions) + 1
            tool_call_id = f'{fid}-{call_counter}'
            blocks.append({'kind': 'assistant', 'turn': turn})
            if pi_name == 'think':
                blocks.append({'kind': 'toolresult', 'tool_call_id': tool_call_id,
                               'tool_name': 'think', 'action_index': call_counter,
                               'verified': True, 'is_error': False})
                actions.append({'index': call_counter, 'tool': pi_name, 'arguments': pi_args,
                                'tool_call_id': tool_call_id, 'source_tool': raw_name,
                                'expectation': expectation})
                if reasoning:
                    think_rows.append({'trajectory_id': trajectory_id, 'turn': call_counter,
                                       'reasoning_content': reasoning, 'action': pi_name,
                                       'arguments': pi_args})
                thought = raw_args.get('thought')
                if isinstance(thought, str) and thought:
                    think_rows.append({'trajectory_id': trajectory_id, 'turn': call_counter,
                                       'reasoning_content': thought, 'action': pi_name,
                                       'arguments': pi_args})
                continue
            action = {'index': call_counter, 'tool': pi_name, 'arguments': pi_args,
                      'tool_call_id': tool_call_id, 'source_tool': raw_name,
                      'expectation': expectation}
            if observation is not None:
                action['recorded_observation'] = observation
            actions.append(action)
            if pi_name == 'finish':
                finished = True
                if reasoning:
                    think_rows.append({'trajectory_id': trajectory_id, 'turn': call_counter,
                                       'reasoning_content': reasoning, 'action': pi_name,
                                       'arguments': pi_args})
                break
            blocks.append({'kind': 'toolresult', 'tool_call_id': tool_call_id,
                           'tool_name': pi_name, 'action_index': call_counter})
            if reasoning:
                think_rows.append({'trajectory_id': trajectory_id, 'turn': call_counter,
                                   'reasoning_content': reasoning, 'action': pi_name,
                                   'arguments': pi_args})
        if not finished:
            return None, 'no_finish'
        return {'trajectory_id': trajectory_id, 'blocks': blocks, 'actions': actions,
                'think_rows': think_rows}, None


def assemble(blocks, observation_texts):
    """Assemble final record text from blocks + verified observation texts."""
    parts = []
    for block in blocks:
        kind = block['kind']
        if kind == 'protocol':
            parts.append('Protocol:\n' + block['header'])
        elif kind == 'system':
            parts.append('\n\nSystem:\n' + compact({'role': 'system', 'content': block['content']}))
        elif kind == 'user':
            parts.append('\n\nUser:\n' + compact({'role': 'user', 'content': block['content']}))
        elif kind == 'assistant':
            parts.append('\n\nAssistant:\n' + block['turn'])
        elif kind == 'toolresult':
            text = observation_texts[block['action_index']]
            parts.append('\n\nToolResult:\n' + compact({
                'role': 'toolResult', 'toolCallId': block['tool_call_id'],
                'toolName': block['tool_name'],
                'content': [{'type': 'text', 'text': text}], 'isError': block.get('is_error', False)}))
    return ''.join(parts) + RS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-messages', type=Path, required=True)
    p.add_argument('--surface', type=Path, required=True)
    p.add_argument('--instance-meta', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--limit', type=int, default=None)
    args = p.parse_args()

    surface = json.loads(args.surface.read_text())
    meta = json.loads(args.instance_meta.read_text())
    out = args.output_dir
    (out / 'candidates').mkdir(parents=True, exist_ok=True)
    translator = Translator(surface)
    counts = {'records': 0, 'translated': 0, 'dropped': 0, 'think_rows': 0}
    drops = {}
    think_path = out / 'think-export.jsonl'
    report_path = out / 'translate-report.json'
    with open(args.source_messages) as src, open(think_path, 'w') as think_out:
        for i, line in enumerate(src):
            if args.limit is not None and i >= args.limit:
                break
            counts['records'] += 1
            row = json.loads(line)
            source = row.get('source') or row
            record, reason = translator.translate(source)
            if record is None:
                counts['dropped'] += 1
                drops[reason] = drops.get(reason, 0) + 1
                continue
            pk = row.get('problem_key')
            instance_id = pk[1] if isinstance(pk, list) and len(pk) > 1 else source.get('instance_id')
            imeta = meta.get(instance_id) or {}
            record['instance_id'] = instance_id
            record['problem_key'] = pk
            record['repo'] = imeta.get('repo')
            record['base_commit'] = imeta.get('base_commit')
            record['image_name'] = imeta.get('image_name')
            if not record.get('base_commit'):
                counts['dropped'] += 1
                drops['missing_base_commit'] = drops.get('missing_base_commit', 0) + 1
                continue
            path = out / 'candidates' / f'{record["trajectory_id"]}.json'
            path.write_text(json.dumps(record, sort_keys=True) + '\n')
            for trow in record['think_rows']:
                think_out.write(json.dumps(trow, sort_keys=True) + '\n')
            counts['translated'] += 1
            counts['think_rows'] += len(record['think_rows'])
    counts['drop_reasons'] = drops
    report_path.write_text(json.dumps(counts, indent=2, sort_keys=True) + '\n')
    print('OH_PI_NATIVE_TRANSLATED', json.dumps(counts, sort_keys=True))


if __name__ == '__main__':
    main()
