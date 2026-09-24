import json

import tiktoken

from ndm.e97_agent_protocol import E97_PI_AGENT_ANALYSIS_SYSTEM_V1, serialize_pi_messages
from scripts import build_e97_open_swe_private_analysis_sft as builder


def test_training_and_serving_analysis_fixture_bytes_tokens_and_target_starts(monkeypatch):
    encoding = tiktoken.get_encoding('p50k_base')
    monkeypatch.setattr(builder.codec, '_WORKER_ENCODING', encoding)
    rationale = 'Read café.\nQuoted marker: Action: edit; λ.'
    prefix = 'Analysis: ' + json.dumps(rationale, ensure_ascii=False, separators=(',', ':')) + '\n'
    assistant_body = prefix + 'Action: read\nArguments: {"path":"note.txt","offset":1,"limit":10}'
    final_body = 'Analysis: "Use only the observation."\nFinal: observed'
    training = [('system', E97_PI_AGENT_ANALYSIS_SYSTEM_V1, False),
                ('user', 'Read note.txt.', False),
                ('assistant', assistant_body, True),
                ('tool', '1: observed', False),
                ('assistant', final_body, True)]
    serving = [{'role': 'system', 'content': E97_PI_AGENT_ANALYSIS_SYSTEM_V1},
               {'role': 'user', 'content': 'Read note.txt.'},
               {'role': 'assistant', 'content': None, 'reasoning_content': rationale,
                'tool_calls': [{'id': 'fixture', 'type': 'function', 'function': {
                    'name': 'read', 'arguments': '{"path":"note.txt","offset":1,"limit":10}'}}]},
               {'role': 'tool', 'tool_call_id': 'fixture', 'content': '1: observed'},
               {'role': 'assistant', 'content': 'Final: observed',
                'reasoning_content': 'Use only the observation.'}]
    tokens, mask, text = builder.encode_record(training)
    live = serialize_pi_messages(serving, private_analysis=True, append_assistant_header=False)
    assert text == live + builder.codec.RS
    assert tokens == encoding.encode_ordinary(text)
    for turn in (2, 4):
        expected_prompt = serialize_pi_messages(serving[:turn], private_analysis=True)
        assert text.startswith(expected_prompt + 'Analysis: ')
        prefix_tokens = encoding.encode_ordinary(expected_prompt)
        assert tokens[:len(prefix_tokens)] == prefix_tokens
        assert mask[len(prefix_tokens)] == 1
        assert mask[len(prefix_tokens)-1] == 0
