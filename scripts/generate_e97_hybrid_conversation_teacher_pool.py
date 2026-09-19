#!/usr/bin/env python3
"""Bounded teacher pool for the E97 hybrid-conversation cohort (extension-prep step 2).

Operator design (from their interactive findings): conversation records where a
user asks a fact question, the model reasons privately in Analysis, calls the
minimal tool (e.g. bash date), reads the observation, and answers in plain
text - plus the pure-chat mirror that answers WITHOUT tools from facts supplied
in the conversation.

This script renders a bounded teacher prompt, requests one completion from the
validated teacher model (lunaroute/deepseek-4.1-flash, the standing
teacher-pilot authority), and validates the returned task specifications
fail-closed. Teacher output is task specifications only: no claimed tool
results, no self-attestation - every answer is re-observed through real Pi at
collection time (deterministic outcome verification).

Commands:
  render    write the teacher prompt
  request   call the teacher API once (bounded), save the raw response
  validate  validate the raw response, publish the canonical task pool
"""
import argparse, hashlib, json, os, re, sys, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ndm.e97_atomic import publish_bytes_no_replace

FORBIDDEN = re.compile(r'(?i)\b(api[_ -]?key|password|secret|credentials?|purchase|production deploy)\b')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    publish_bytes_no_replace(path, (json.dumps(value, indent=2, sort_keys=True) + '\n').encode(), mode=0o400)


def render_prompt(config):
    tool_fams = ', '.join(config['tool_flow_families'])
    chat_fams = ', '.join(config['pure_chat_families'])
    n_tool = config['tool_flow_tasks']
    n_chat = config['pure_chat_tasks']
    return f'''You are the parent curriculum designer for a small conversation-cohort task pool. Return one raw JSON object with exactly two keys: schema and tasks. No markdown, no explanation, raw JSON only.

schema must be "emender-e97-hybrid-conversation-teacher-pool-v1".

tasks must contain exactly {n_tool + n_chat} objects, ordered: first the {n_tool} tool-flow tasks, then the {n_chat} pure-chat tasks. Each object has exactly these fields: id, kind, family, user_question, workspace_files, expected_answer, observed_literal, expected_tool, bash_command, analysis_focus, answer_commentary, diversity_tags.

TOOL-FLOW TASKS (kind "tool-flow", {n_tool} of them):
- family comes from: {tool_fams}.
- The user asks a short natural-language question about a fact stored in a small workspace fixture (a config value, a count, a sum of two recorded operands, a status word). The model must call exactly one minimal tool to observe the fact, then answer in plain text.
- workspace_files: one or two relative paths (no leading /, no ..) with complete file contents. Keep each file under 20 lines. The fact the question asks about must appear verbatim in the file contents.
- expected_answer: the exact final answer string (a number, word, or short token) - it MUST appear verbatim inside the workspace_files contents.
- observed_literal: a short literal (at least 5 characters) copied verbatim from the file contents that the model's private reasoning should quote when it observes the file.
- expected_tool: "read" for file-content facts, "bash" for counts/sums. For "bash", bash_command is one simple deterministic command from: {', '.join(config['bash_command_whitelist'])}. For "read", bash_command is the empty string.
- analysis_focus: one sentence describing what the private reasoning step should note (what fact is being looked up and why the tool is needed).
- answer_commentary: the full plain-text answer sentence the user will see; it must contain expected_answer verbatim.
- diversity_tags: at least three distinct axes (topic, phrasing style, file kind).

PURE-CHAT MIRROR TASKS (kind "pure-chat", {n_chat} of them):
- family comes from: {chat_fams}.
- The user supplies the fact directly inside user_question and explicitly asks for a plain-text answer WITHOUT using any tools.
- workspace_files: empty object. expected_tool: empty string. bash_command: empty string.
- expected_answer and observed_literal must each appear verbatim inside user_question (observed_literal at least 5 characters). expected_answer is the exact final answer string.
- analysis_focus: one sentence about why no tool is needed (the fact is already in the conversation).
- answer_commentary: the full plain-text answer sentence; it must contain expected_answer verbatim.
- diversity_tags: at least three distinct axes.

ALL TASKS:
- id: unique, of the form teacher-hybrid-<kind>-<zero-padded index> (e.g. teacher-hybrid-toolflow-01).
- Never use real credentials, purchases, or production systems. Never reference evaluation entities: {config['constraints']['evaluation_entities_forbidden']}.
- Every answer must be deterministically checkable from the workspace file or the supplied conversation text - never from world knowledge.
- Vary phrasing, topics, and file kinds; keep every question under 60 words.'''


def validate(config, raw_text):
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f'non-JSON teacher output: {exc}')
    if set(data) != {'schema', 'tasks'} or data['schema'] != 'emender-e97-hybrid-conversation-teacher-pool-v1':
        raise ValueError('teacher envelope')
    tasks = data['tasks']
    if not isinstance(tasks, list) or len(tasks) != config['tool_flow_tasks'] + config['pure_chat_tasks']:
        raise ValueError('teacher task count')
    fields = set(config['task_fields'])
    forbidden_entities = config['constraints']['evaluation_entities_forbidden']
    seen_ids = set()
    tool_seen = chat_seen = 0
    for t in tasks:
        if set(t) != fields:
            raise ValueError('teacher task fields')
        for key in ('id', 'kind', 'family', 'user_question', 'expected_answer', 'observed_literal',
                    'expected_tool', 'bash_command', 'analysis_focus', 'answer_commentary'):
            if not isinstance(t[key], str):
                raise ValueError('teacher strings')
        if not isinstance(t['workspace_files'], dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in t['workspace_files'].items()):
            raise ValueError('teacher workspace files')
        if not isinstance(t['diversity_tags'], list) or len(t['diversity_tags']) < 3 or \
                any(not isinstance(x, str) for x in t['diversity_tags']):
            raise ValueError('teacher diversity tags')
        if t['id'] in seen_ids:
            raise ValueError('teacher id collision')
        seen_ids.add(t['id'])
        flat = ' '.join([t['id'], t['kind'], t['family'], t['user_question'], t['expected_answer'],
                         t['observed_literal'], t['expected_tool'], t['bash_command'], t['analysis_focus'],
                         t['answer_commentary']] + list(t['workspace_files'].values()))
        if FORBIDDEN.search(flat) or any(e in flat for e in forbidden_entities):
            raise ValueError('forbidden teacher content')
        if t['kind'] == 'tool-flow':
            tool_seen += 1
            if tool_seen > config['tool_flow_tasks'] or t['family'] not in config['tool_flow_families']:
                raise ValueError('tool-flow family/count')
            if not t['id'].startswith('teacher-hybrid-tool'):
                raise ValueError('tool-flow id')
            if not t['workspace_files']:
                raise ValueError('tool-flow workspace files required')
            for path in t['workspace_files']:
                if path.startswith('/') or '..' in path.split('/') or path.endswith('/'):
                    raise ValueError('teacher path')
            contents = '\n'.join(t['workspace_files'].values())
            if t['expected_answer'] not in contents:
                raise ValueError('tool-flow answer not in workspace')
            # >=5 characters: accommodates compact deterministic literals
            # such as 'sum=2' from the derive family (documented relaxation).
            if len(t['observed_literal']) < 5 or t['observed_literal'] not in contents:
                raise ValueError('tool-flow observed literal')
            # Documented relaxation: natural-language references ("the job
            # file" for status/job.status) are accepted; a path-verbatim
            # question is not required because (a) each case workspace holds
            # only the authored fixtures and (b) the runtime oracles re-observe
            # the answer through real Pi before any record is admitted.
            if t['expected_tool'] == 'read':
                if t['bash_command']:
                    raise ValueError('read flow bash command')
            elif t['expected_tool'] == 'bash':
                if not t['bash_command'] or not any(
                        t['bash_command'].startswith(w) for w in config['bash_command_whitelist']):
                    raise ValueError('bash command whitelist')
                if re.search(r'[;&|`$<>]', t['bash_command']):
                    raise ValueError('bash command metacharacters')
            else:
                raise ValueError('tool-flow expected tool')
        elif t['kind'] == 'pure-chat':
            chat_seen += 1
            if chat_seen > config['pure_chat_tasks'] or t['family'] not in config['pure_chat_families']:
                raise ValueError('pure-chat family/count')
            if not (t['id'].startswith('teacher-hybrid-chat') or t['id'].startswith('teacher-hybrid-purechat')):
                raise ValueError('pure-chat id')
            if t['workspace_files'] or t['expected_tool'] or t['bash_command']:
                raise ValueError('pure-chat must be tool-free')
            if t['expected_answer'] not in t['user_question'] or \
                    len(t['observed_literal']) < 5 or t['observed_literal'] not in t['user_question']:
                raise ValueError('pure-chat fact must be supplied in the question')
        else:
            raise ValueError('teacher kind')
        # Documented relaxation: short numeric answers (counts, sums) are the
        # point of the count/derive families; the degeneracy guard rejects
        # empty or null-like answers, not short deterministic values.
        if not t['expected_answer'].strip() or t['expected_answer'].strip() in ('null', 'None', '...'):
            raise ValueError('degenerate answer')
        if t['expected_answer'] not in t['answer_commentary']:
            raise ValueError('answer commentary must contain the answer')
        if len(t['user_question'].split()) > 80:
            raise ValueError('question length')
    return data


def request(config, prompt_path, raw_path):
    token = json.loads(Path(os.path.expanduser('~/.pi/agent/auth.json')).read_text())['lunaroute']['access']
    url = os.environ.get('LUNAROUTE_ROUTING_URL', 'https://gw.lunaroute.com/v1') + '/chat/completions'
    body = json.dumps({
        'model': config['model'],
        'messages': [{'role': 'user', 'content': Path(prompt_path).read_text()}],
        'temperature': 0.7,
        'max_tokens': 24000,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        payload = json.load(response)
    raw = payload['choices'][0]['message']['content']
    raw_path.write_text(raw)
    return payload


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('render')
    r.add_argument('--config', type=Path, required=True)
    r.add_argument('--output', type=Path, required=True)
    q = sub.add_parser('request')
    q.add_argument('--config', type=Path, required=True)
    q.add_argument('--prompt', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    q.add_argument('--meta', type=Path, required=True)
    v = sub.add_parser('validate')
    v.add_argument('--config', type=Path, required=True)
    v.add_argument('--raw', type=Path, required=True)
    v.add_argument('--output', type=Path, required=True)
    v.add_argument('--summary', type=Path, required=True)
    args = p.parse_args()
    os.umask(0o077)
    config = json.loads(args.config.read_text())
    if config['schema'] != 'emender-e97-hybrid-conversation-teacher-v1':
        raise ValueError('teacher config authority')
    if args.command == 'render':
        args.output.write_text(render_prompt(config))
        print('TEACHER_PROMPT_RENDERED', sha(args.output))
    elif args.command == 'request':
        payload = request(config, args.prompt, args.output)
        publish(args.meta, {'status': 'raw-teacher-response-retained', 'model': config['model'],
                            'provider': config['provider'], 'raw_sha256': sha(args.output),
                            'usage': payload.get('usage'), 'prompt_sha256': sha(args.prompt),
                            'real_pi_executions': 0, 'model_generations': 1, 'optimizer_updates': 0,
                            'training_eligible': False})
        print('TEACHER_RESPONSE_RETAINED', sha(args.output))
    else:
        canonical = validate(config, args.raw.read_text())
        publish(args.output, canonical)
        publish(args.summary, {
            'status': 'hybrid-conversation-teacher-tasks-validated',
            'tasks': len(canonical['tasks']),
            'tool_flow': sum(1 for t in canonical['tasks'] if t['kind'] == 'tool-flow'),
            'pure_chat': sum(1 for t in canonical['tasks'] if t['kind'] == 'pure-chat'),
            'config_sha256': sha(args.config), 'raw_sha256': sha(args.raw),
            'tasks_sha256': sha(args.output), 'real_pi_executions': 0,
            'model_generations': 0, 'optimizer_updates': 0, 'training_eligible': False})
        print('HYBRID_TEACHER_VALIDATED', len(canonical['tasks']), sha(args.output))


if __name__ == '__main__':
    main()
