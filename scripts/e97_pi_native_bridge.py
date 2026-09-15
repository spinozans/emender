"""Single-episode Pi-facing transport for the unchanged OpenHands-native model.

The native transcript is authoritative and private. Pi receives public events
and lossless *transport envelopes*, not a translation to read/bash/edit/write.
In particular, Pi must not validate/coerce native arguments before OpenHands.
No model, executor, retry, checkpoint loading, or persistence lives in this file.
"""
from copy import deepcopy
import hashlib
import secrets
import time

from scripts.e97_native_execution_cases import context
from scripts.e97_open_swe_native_codec import compact, semantic_message
from scripts.e97_open_swe_native_runtime_protocol import NativeEpisode

PROVIDER = 'e97-openhands-compat'
MODEL = 'e97-4b-native-bridge'
API = 'e97-native-unix-v1'
TRANSPORT_SCHEMA = {
    'type': 'object',
    'properties': {'call_ref': {'type': 'string'}, 'native_arguments_json': {'type': 'string'}},
    'required': ['call_ref'], 'additionalProperties': False,
}


class BridgeStopped(ValueError):
    """Only fixed reason codes may cross the public transport boundary."""


def transport_tools(tools):
    return [dict(name=t['function']['name'], label='OpenHands: '+t['function']['name'],
                 description='OpenHands-native transport; arguments are validated by its executor.',
                 parameters=deepcopy(TRANSPORT_SCHEMA)) for t in tools]


def normalized_messages(messages):
    """Discard transport accounting only, never text, blocks, roles, or errors."""
    if not isinstance(messages, list):
        raise BridgeStopped('invalid_pi_history')
    result = []
    for m in messages:
        role = m.get('role')
        if role == 'user':
            content = m['content']
            if isinstance(content, str):
                content = [dict(type='text', text=content)]
            if (not isinstance(content, list) or len(content) != 1 or
                    set(content[0]) != {'type', 'text'} or content[0]['type'] != 'text' or
                    not isinstance(content[0]['text'], str)):
                raise BridgeStopped('unsupported_pi_user_content')
            result.append(dict(role=role, content=deepcopy(content)))
        elif role == 'assistant':
            if m.get('stopReason', 'toolUse') != 'toolUse':
                raise BridgeStopped('unexpected_pi_assistant_stop')
            for key, value in [('provider', PROVIDER), ('model', MODEL), ('api', API)]:
                if m.get(key, value) != value:
                    raise BridgeStopped('pi_model_changed')
            content = m.get('content')
            if not isinstance(content, list):
                raise BridgeStopped('invalid_pi_assistant_content')
            for b in content:
                keys = {'type', 'text'} if b.get('type') == 'text' else {'type', 'id', 'name', 'arguments'}
                if b.get('type') not in ('text', 'toolCall') or set(b) != keys:
                    raise BridgeStopped('unsupported_pi_assistant_block')
            result.append(dict(role=role, content=deepcopy(content)))
        elif role == 'toolResult':
            if type(m.get('isError')) is not bool:
                raise BridgeStopped('invalid_pi_error_flag')
            result.append({k: deepcopy(m[k]) for k in
                           ('role', 'toolCallId', 'toolName', 'content', 'isError')})
        else:
            raise BridgeStopped('unsupported_pi_history_role')
    return result


class NativePiBridge:
    """Owns one full-history episode and at most one outstanding Pi tool call.

    generate(prompt, budget, deadline) returns the native generator's triple.
    execute(call) returns the qualified sandbox RPC reply. Neither callback is
    allowed to replace failed model outputs with a teacher or synthesized result.
    """
    def __init__(self, panel, prompt, encoding, generate, execute):
        self.panel = deepcopy(panel)
        self.encoding = encoding
        self.generate = generate
        self.executor = execute
        self.episode = NativeEpisode(panel['tools'], encoding)
        self.episode.append_source_message(panel.get('system_message', context('system', panel['system'])))
        self.episode.append_source_message(context('user', prompt))
        self.history = [dict(role='user', content=[dict(type='text', text=prompt)])]
        self.tools = transport_tools(panel['tools'])
        self.pending = None
        self.generations = []
        self.calls = []
        self.tokens = 0
        self.final = None
        self.reason = None
        self.failed = False
        self.closed = False
        self.close_verified = False
        self.deadline = time.monotonic() + panel['episode_seconds']
        self.nonce = secrets.token_hex(16)

    def stop(self, reason):
        self.failed = True
        self.reason = reason
        raise BridgeStopped(reason)

    def verify_history(self, messages):
        if compact(normalized_messages(messages)) != compact(self.history):
            self.stop('pi_history_mismatch')

    def next(self, request):
        if self.failed or self.closed:
            raise BridgeStopped('bridge_not_active')
        if self.pending is not None or self.episode.finished:
            self.stop('unexpected_pi_generation')
        if (request.get('systemPrompt') != self.panel['system'] or
                request.get('model') != MODEL or request.get('provider') != PROVIDER or
                compact(request.get('tools')) != compact(self.tools)):
            self.stop('pi_contract_mismatch')
        self.verify_history(request.get('messages'))
        if len(self.generations) >= self.panel['max_turns']:
            self.stop('turn_budget')
        budget = min(self.panel['generation_budget'], self.panel['episode_generation_budget']-self.tokens)
        if budget <= 0:
            self.stop('episode_generation_budget')
        if time.monotonic() >= self.deadline:
            self.stop('episode_deadline')
        prompt = self.episode.prompt()
        text, ids, reason = self.generate(prompt, budget, self.deadline)
        self.generations.append(dict(turn=len(self.generations), token_ids=ids, reason=reason,
                                     prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
        self.tokens += len(ids)
        if text is None:
            self.stop(reason)
        turn = self.episode.accept_generated_turn(text)
        native = semantic_message(turn.source_message())
        ref = f'{self.nonce}-{len(self.generations)}'
        # A JSON string envelope preserves large integers and 2.0 versus 2 across
        # JavaScript. Only the opaque reference is exposed for a private think.
        arguments = dict(call_ref=ref)
        if native['name'] != 'think':
            arguments['native_arguments_json'] = compact(native['arguments'])
        call = dict(type='toolCall', id=ref, name=native['name'], arguments=arguments)
        blocks = []
        if native['content'] is not None:
            blocks.append(dict(type='text', text=native['content']))
        blocks.append(call)
        message = dict(role='assistant', content=blocks)
        self.history.append(deepcopy(message))
        self.pending = (turn, deepcopy(call))
        return dict(message=message, input_tokens=len(self.encoding.encode_ordinary(prompt)),
                    output_tokens=len(ids))

    def execute(self, request):
        if self.failed or self.closed or self.pending is None:
            raise BridgeStopped('no_pending_native_call')
        turn, expected = self.pending
        actual = dict(type='toolCall', id=request.get('id'), name=request.get('name'),
                      arguments=request.get('arguments'))
        if compact(actual) != compact(expected):
            self.stop('pi_call_mismatch')
        if time.monotonic() >= self.deadline:
            self.stop('episode_deadline')
        native = semantic_message(turn.source_message())
        if native['name'] == 'finish':
            self.final = native['arguments']['message']
            self.reason = 'finished'
            text = self.final
            is_error = False
            # Pi gets a terminating acknowledgement; native finish gets no
            # invented observation and never triggers an extra model generation.
        elif native['name'] == 'think':
            text = 'Your thought has been logged.'
            self.episode.append_observation(context('tool', text))
            is_error = False
        else:
            call = turn.backend_call()
            reply = self.executor(deepcopy(call))
            self.calls.append(dict(request=call, **deepcopy(reply)))
            if 'dispatch_error' in reply:
                self.stop('invalid_tool_request')
            result = reply['result']
            self.episode.append_observation(result['message'])
            text = result['message']['content']
            is_error = text.startswith('ERROR:') or result.get('exit_code', 0) not in (0, -1, None)
        result = dict(content=[dict(type='text', text=text)],
                      details=dict(native_error=is_error), terminate=native['name'] == 'finish')
        self.history.append(dict(role='toolResult', toolCallId=expected['id'],
                                 toolName=expected['name'], content=deepcopy(result['content']), isError=is_error))
        self.pending = None
        return result

    def close(self, messages):
        if self.closed:
            raise BridgeStopped('duplicate_close')
        try:
            # Failed Pi messages are retained externally, not admitted into the
            # native transcript as invented recovery or completion evidence.
            if not self.failed:
                self.verify_history(messages)
                if self.pending is not None or self.reason != 'finished':
                    self.stop('pi_ended_before_native_finish')
                self.close_verified = True
        finally:
            self.closed = True
        return dict(closed=True, verified=self.close_verified)
