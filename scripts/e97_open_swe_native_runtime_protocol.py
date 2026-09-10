"""Source-native episode protocol boundary; no shell/editor execution here.

Uses the exact published dataset codec. A separate isolated, qualified backend
must execute admitted calls. This module never maps source calls to Pi tools.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib

from scripts.e97_open_swe_native_codec import (
    PROFILE, TOOLS, FRAME, compact, strict_json, semantic_message, native_turn, parse_turn,
)


@dataclass(frozen=True)
class NativeTurn:
    """Keep private fields out of repr and explicitly separate public events."""
    text_sha256: str
    _message: dict

    def __repr__(self):
        return f'NativeTurn(text_sha256={self.text_sha256!r})'

    def source_message(self):
        return deepcopy(self._message)

    def public_events(self):
        m=semantic_message(self._message);events=[]
        if m['content'] is not None:events.append({'kind':'commentary','text':m['content']})
        if m['name']=='finish':events.append({'kind':'final','text':m['arguments']['message']})
        return events

    def private_fields(self):
        m=semantic_message(self._message)
        return {'reasoning_content':m['reasoning_content'],
                'thought':m['arguments'].get('thought') if m['name']=='think' else None,
                'think':m['think']}

    def backend_call(self):
        """Only actual shell/editor calls leave this boundary. Think is private."""
        m=semantic_message(self._message)
        if m['name'] in ('think','finish'):return None
        return {'name':m['name'],'arguments':deepcopy(m['arguments'])}


def validate_generated_turn(text,encoding):
    if not isinstance(text,str):raise ValueError('native_turn_must_be_text')
    # A valid UTF-8 p50k record cannot exceed this conservative byte bound.
    if len(text.encode())>65536*128:raise ValueError('native_turn_byte_cap')
    parsed=parse_turn(text)
    source={'role':'assistant','content':parsed['content'],'reasoning_content':parsed['reasoning_content'],
            'think':parsed['think'],'tool_calls':[{'type':'function','function':{
                'name':parsed['name'],'arguments':compact(parsed['arguments'])}}]}
    m=semantic_message(source)
    if native_turn(source)!=text:raise ValueError('noncanonical_native_turn')
    private=[m['reasoning_content'] or '']
    if m['name']=='think':
        thought=m['arguments'].get('thought')
        if not isinstance(thought,str):raise ValueError('missing_thought')
        private.append(thought)
    if sum(len(s.encode()) for s in private)>65536 or sum(len(encoding.encode_ordinary(s)) for s in private)>2048:
        raise ValueError('analysis_cap')
    if m['name']=='finish':
        message=m['arguments'].get('message')
        if not isinstance(message,str) or not message.strip():raise ValueError('missing_final_message')
    if len(encoding.encode_ordinary(text))>65536:raise ValueError('native_turn_token_cap')
    return NativeTurn(hashlib.sha256(text.encode()).hexdigest(),source)


class NativeEpisode:
    """Causal, transactional full-history framing for one source-native task.

    A finish ends the episode, not an arbitrary ongoing chat session. Context is
    never compacted. Source argument type/range errors belong to the source
    executor: this boundary checks framing, not rewrites of invalid requests.
    """
    def __init__(self,tools,encoding,max_context_tokens=65536):
        if type(max_context_tokens) is not int or not 1<=max_context_tokens<=65536:
            raise ValueError('invalid_native_context_bound')
        specs=[strict_json(s) if isinstance(s,str) else deepcopy(s) for s in tools]
        try:names=[s['function']['name'] for s in specs]
        except (KeyError,TypeError):raise ValueError('source_tool_declarations') from None
        if len(names)!=len(set(names)) or set(names)!=TOOLS:raise ValueError('source_tool_declarations')
        self.encoding=encoding;self.max_context_tokens=max_context_tokens
        self._header='Protocol:\n'+compact({'profile':PROFILE,'instructions':FRAME,'tools':specs})
        self._text=self._header;self._messages=[];self._seen_user=False;self._pending=False;self._finished=False
        self._check_size(self._text)

    def _check_size(self,text):
        if len(self.encoding.encode_ordinary(text))>self.max_context_tokens:raise ValueError('native_context_exhausted')

    def _append(self,message):
        raw=deepcopy(message);role=raw.get('role')
        if role not in ('system','user','assistant','tool'):raise ValueError('unknown_role')
        if self._finished and role!='tool':raise ValueError('content_after_finish')
        if self._pending and role!='tool':raise ValueError('missing_observation')
        seen_user=self._seen_user;pending=self._pending;finished=self._finished
        if role=='assistant':
            if not seen_user:raise ValueError('assistant_without_user')
            text=native_turn(raw);turn=validate_generated_turn(text,self.encoding)
            finished=semantic_message(turn.source_message())['name']=='finish';pending=True
            suffix='\n\nAssistant:\n'+text
        else:
            if role=='tool':
                if not pending:raise ValueError('orphan_observation')
                pending=False
            if role=='user':seen_user=True
            suffix='\n\n'+role.title()+':\n'+compact(raw)
        candidate=self._text+suffix
        self._check_size(candidate)
        # Only commit after every check; failed append leaves the complete prefix intact.
        self._text=candidate;self._messages.append(raw)
        self._seen_user=seen_user;self._pending=pending;self._finished=finished

    def append_source_message(self,message):
        self._append(message)

    def prompt(self):
        if self._finished:raise ValueError('episode_finished')
        if self._pending:raise ValueError('missing_observation')
        if not self._seen_user:raise ValueError('missing_user')
        text=self._text+'\n\nAssistant:\n';self._check_size(text)
        return text

    def accept_generated_turn(self,text):
        self.prompt()
        turn=validate_generated_turn(text,self.encoding)
        self._append(turn.source_message())
        return turn

    def append_observation(self,message):
        if message.get('role')!='tool':raise ValueError('observation_requires_tool_role')
        self._append(message)

    @property
    def finished(self):return self._finished

    def text(self):return self._text

    def source_messages(self):return deepcopy(self._messages)
