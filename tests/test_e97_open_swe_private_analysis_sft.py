from __future__ import annotations

import json

import pytest

from scripts import build_e97_open_swe_private_analysis_sft as private_swe


class ByteEncoding:
    def encode(self, text, disallowed_special=()):
        return list(text.encode("utf-8"))

    def encode_ordinary(self, text):
        return list(text.encode("utf-8"))

    def decode_single_token_bytes(self, token):
        return bytes([token])


def tool_call(name, arguments):
    return {"function": {"name": name, "arguments": json.dumps(arguments)}}


def source_row(*, reasoning="inspect Action: and Final: safely", final="done"):
    return {
        "messages": [
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "reasoning_content": "plan first", "tool_calls": [
                tool_call("think", {"thought": "plan first"}),
            ]},
            {"role": "tool", "content": "Your thought has been logged."},
            {"role": "assistant", "reasoning_content": reasoning, "tool_calls": [
                tool_call("execute_bash", {"command": "pytest -q"}),
            ]},
            {"role": "tool", "content": "1 passed"},
            {"role": "assistant", "reasoning_content": "report exactly", "tool_calls": [
                tool_call("finish", {"message": final}),
            ]},
        ]
    }


def test_normalizer_folds_think_reasoning_and_uses_canonical_json_framing(monkeypatch):
    monkeypatch.setattr(private_swe, "_ANALYSIS_TOKEN_CAP", 10_000)
    messages, counts, think_turns = private_swe.normalize_private_messages(
        source_row(), encoding=ByteEncoding())

    first = messages[2][1]
    assert first.startswith('Analysis: "plan first\\n\\ninspect Action: and Final: safely"\n')
    assert first.endswith('Action: bash\nArguments: {"command":"pytest -q"}')
    assert messages[3] == ("tool", "1 passed")
    assert messages[-1][1] == 'Analysis: "report exactly"\nFinal: done'
    assert think_turns == 1
    assert counts == [len("plan first\n\ninspect Action: and Final: safely".encode()), len("report exactly".encode())]


def test_matched_action_only_removes_exact_analysis_frame_only(monkeypatch):
    monkeypatch.setattr(private_swe, "_ANALYSIS_TOKEN_CAP", 10_000)
    private_messages, _, _ = private_swe.normalize_private_messages(
        source_row(), encoding=ByteEncoding())
    action_messages = private_swe.action_only_messages(private_messages)

    assert action_messages[0][1] == private_swe.E97_PI_AGENT_SYSTEM_V2
    assert action_messages[2][1] == 'Action: bash\nArguments: {"command":"pytest -q"}'
    assert action_messages[-1][1] == "Final: done"
    assert all("Analysis:" not in content for role, content in action_messages if role == "assistant")


def test_final_text_is_not_silently_truncated(monkeypatch):
    monkeypatch.setattr(private_swe, "_ANALYSIS_TOKEN_CAP", 10_000)
    final = "z" * 2_000
    messages, _, _ = private_swe.normalize_private_messages(
        source_row(final=final), encoding=ByteEncoding())
    assert messages[-1][1].endswith("Final: " + final)


def test_analysis_cap_excludes_instead_of_truncating(monkeypatch):
    monkeypatch.setattr(private_swe, "_ANALYSIS_TOKEN_CAP", 8)
    with pytest.raises(ValueError, match="private_analysis_token_cap"):
        private_swe.normalize_private_messages(source_row(), encoding=ByteEncoding())


def test_segmentation_preserves_each_complete_target_unit_once(monkeypatch):
    monkeypatch.setattr(private_swe.codec, "_WORKER_ENCODING", ByteEncoding())
    base = [("system", "s"), ("user", "u")]
    messages = base + [
        ("assistant", 'Analysis: "a"\nAction: read\nArguments: {}'),
        ("tool", "x" * 40),
        ("assistant", 'Analysis: "b"\nAction: bash\nArguments: {}'),
        ("tool", "y" * 40),
        ("assistant", 'Analysis: "c"\nFinal: done'),
    ]
    records = private_swe.segment_complete_units(messages, 180)
    assert len(records) >= 2
    assert all(len(private_swe.encode_record(record)[0]) <= 180 for record in records)
    assert sum(sum(target for _, _, target in record) for record in records) == 3
    for record in records:
        for index, (role, _, targeted) in enumerate(record):
            if role == "tool":
                assert index > 0 and record[index - 1][0] == "assistant"
                assert record[index - 1][2] in {True, False}


def test_oversize_complete_unit_fails_closed(monkeypatch):
    monkeypatch.setattr(private_swe.codec, "_WORKER_ENCODING", ByteEncoding())
    messages = [
        ("system", "s"), ("user", "u"),
        ("assistant", 'Analysis: "a"\nAction: read\nArguments: {}'),
        ("tool", "x" * 1_000),
        ("assistant", 'Analysis: "b"\nFinal: done'),
    ]
    with pytest.raises(ValueError, match="oversize_complete_logical_unit"):
        private_swe.segment_complete_units(messages, 180)
