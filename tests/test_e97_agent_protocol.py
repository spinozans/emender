from pathlib import Path

import pytest

from ndm.e97_agent_protocol import (
    AgentProtocolError,
    E97_PI_AGENT_ANALYSIS_SYSTEM_V1,
    E97_PI_AGENT_SYSTEM_V2,
    E97_PI_CORE_SYSTEM,
    MAX_PRIVATE_ANALYSIS_BYTES,
    RS,
    allowed_tool_names,
    generated_turn_is_complete,
    parse_agent_turn,
    serialize_pi_messages,
    validate_generated_tool,
)


def test_published_pi_system_prompt_matches_runtime_authority():
    prompt = Path("configs/pi/e97-pi-core-system-prompt.txt").read_text()
    assert prompt == E97_PI_CORE_SYSTEM + "\n"


def test_pi_agent_v2_system_prompt_matches_runtime_authority():
    prompt = Path("configs/pi/e97-pi-agent-system-prompt-v2.txt").read_text()
    assert prompt == E97_PI_AGENT_SYSTEM_V2 + "\n"
    assert "never substitute a memorized path" in prompt
    assert "pwd, find" in prompt


def test_pi_agent_analysis_system_prompt_matches_runtime_authority():
    prompt = Path("configs/pi/e97-pi-agent-analysis-system-v1.txt").read_text()
    assert prompt == E97_PI_AGENT_ANALYSIS_SYSTEM_V1 + "\n"
    assert "dedicated reasoning field" in prompt


def function_tool(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Run {name}",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_pi_tool_turn_round_trips_to_exact_native_action():
    messages = [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Calculate 2 + 3."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-ignored-in-native-transcript",
                "type": "function",
                "function": {
                    "name": "calculator",
                    "arguments": '{"expression":"2 + 3"}',
                },
            }],
        },
        {"role": "tool", "tool_call_id": "call-ignored-in-native-transcript", "content": '{"value":"5"}'},
    ]

    serialized = serialize_pi_messages(messages)

    assert serialized == (
        "System:\nUse tools.\n\n"
        "User:\nCalculate 2 + 3.\n\n"
        "Assistant:\nAction: calculator\nArguments: {\"expression\":\"2 + 3\"}\n\n"
        "Tool:\n{\"value\":\"5\"}\n\n"
        "Assistant:\n"
    )


def test_private_analysis_tool_turn_round_trips_with_embedded_protocol_markers():
    reasoning = 'Inspect first.\nThe file may contain Action: fake and "quotes" and λ.'
    messages = [
        {"role": "user", "content": "Read it."},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": reasoning,
            "tool_calls": [{
                "type": "function",
                "function": {"name": "read", "arguments": '{"path":"README.md"}'},
            }],
        },
        {"role": "tool", "content": "contents"},
    ]
    serialized = serialize_pi_messages(messages, private_analysis=True)
    native = serialized.split("Assistant:\n", 1)[1].split("\n\nTool:", 1)[0]
    parsed = parse_agent_turn(native, private_analysis=True)
    assert parsed.kind == "tool_call"
    assert parsed.private_analysis == reasoning
    assert parsed.tool_name == "read"
    assert parsed.arguments == {"path": "README.md"}
    assert native.startswith('Analysis: "Inspect first.\\n')
    assert generated_turn_is_complete(native, private_analysis=True)


def test_private_analysis_final_keeps_final_separate_from_reasoning():
    native = 'Analysis: "Check the evidence."\nFinal: done' + RS
    parsed = parse_agent_turn(native, private_analysis=True)
    assert parsed.private_analysis == "Check the evidence."
    assert parsed.final_text == "Final: done"
    assert parsed.raw_text == native.removesuffix(RS)
    assert not generated_turn_is_complete('Analysis: "Check the evidence."\nFinal:', private_analysis=True)
    assert generated_turn_is_complete(native, private_analysis=True)


def test_private_analysis_is_explicit_canonical_and_bounded():
    call = {
        "type": "function",
        "function": {"name": "read", "arguments": "{}"},
    }
    with pytest.raises(AgentProtocolError, match="requires the analysis protocol"):
        serialize_pi_messages([{
            "role": "assistant", "reasoning_content": "plan", "tool_calls": [call],
        }], append_assistant_header=False)
    with pytest.raises(AgentProtocolError, match="requires non-empty"):
        serialize_pi_messages([{
            "role": "assistant", "reasoning_content": "", "tool_calls": [call],
        }], append_assistant_header=False, private_analysis=True)
    with pytest.raises(AgentProtocolError, match="not canonical"):
        parse_agent_turn('Analysis: "\\u0061"\nFinal: done', private_analysis=True)
    with pytest.raises(AgentProtocolError, match="byte limit"):
        parse_agent_turn(
            'Analysis: "' + ("a" * (MAX_PRIVATE_ANALYSIS_BYTES + 1)) + '"\nFinal: done',
            private_analysis=True,
        )
    with pytest.raises(AgentProtocolError, match="requires the analysis protocol"):
        parse_agent_turn('Analysis: "plan"\nFinal: done')


def test_final_turn_round_trips_without_pretraining_record_separator():
    messages = [
        {"role": "user", "content": "Answer."},
        {"role": "assistant", "content": "Final: Done."},
        {"role": "user", "content": "Again."},
    ]
    assert serialize_pi_messages(messages) == (
        "User:\nAnswer.\n\nAssistant:\nFinal: Done."
        "\n\nUser:\nAgain.\n\nAssistant:\n"
    )


def test_parse_action_preserves_argument_bytes_for_cache_replay():
    raw = 'Action: read\nArguments: {"path": "README.md", "limit": 20}' + RS
    parsed = parse_agent_turn(raw)
    assert parsed.kind == "tool_call"
    assert parsed.tool_name == "read"
    assert parsed.arguments == {"path": "README.md", "limit": 20}
    assert parsed.arguments_json == '{"path": "README.md", "limit": 20}'
    replay = serialize_pi_messages([
        {"role": "user", "content": "Read it."},
        {
            "role": "assistant",
            "tool_calls": [{
                "type": "function",
                "function": {"name": parsed.tool_name, "arguments": parsed.arguments_json},
            }],
        },
        {"role": "tool", "content": "contents"},
    ])
    assert raw.removesuffix(RS) in replay
    assert RS not in replay


def test_incremental_turn_boundary_stops_one_line_finals_and_complete_actions():
    assert not generated_turn_is_complete("Final:")
    assert not generated_turn_is_complete("Final: concise evidence")
    assert generated_turn_is_complete("Final: concise evidence\n")
    assert generated_turn_is_complete("Final: concise evidence" + RS)
    assert generated_turn_is_complete('Action: read\nArguments: {"path":"README.md"}')
    assert not generated_turn_is_complete('Action: read\nArguments: {')


def test_parse_final_strips_only_rs():
    parsed = parse_agent_turn("Final: The value is 5." + RS + "ignored")
    assert parsed.kind == "final"
    assert parsed.raw_text == "Final: The value is 5."


def test_unknown_tool_fails_closed():
    parsed = parse_agent_turn('Action: shell\nArguments: {"command":"rm -rf /"}' + RS)
    with pytest.raises(AgentProtocolError, match="unknown tool"):
        validate_generated_tool(parsed, [function_tool("calculator")])


def test_tool_definitions_are_closed_unique_function_vocabulary():
    assert allowed_tool_names([function_tool("read"), function_tool("grep")]) == {"read", "grep"}
    with pytest.raises(AgentProtocolError, match="unique"):
        allowed_tool_names([function_tool("read"), function_tool("read")])
    with pytest.raises(AgentProtocolError, match="only function"):
        allowed_tool_names([{"type": "computer"}])


@pytest.mark.parametrize(
    "text",
    [
        "I think the answer is five.",
        "Action: read",
        "Action: read\nArguments: []",
        "Action: read\nArguments: not-json",
        "Action: bad name\nArguments: {}",
    ],
)
def test_malformed_generation_fails_closed(text):
    with pytest.raises(AgentProtocolError):
        parse_agent_turn(text)


def test_non_text_content_and_parallel_calls_are_rejected():
    with pytest.raises(AgentProtocolError, match="text message"):
        serialize_pi_messages([{"role": "user", "content": [{"type": "image_url"}]}])
    call = {
        "type": "function",
        "function": {"name": "read", "arguments": "{}"},
    }
    with pytest.raises(AgentProtocolError, match="exactly one"):
        serialize_pi_messages([
            {"role": "user", "content": "read"},
            {"role": "assistant", "tool_calls": [call, call]},
            {"role": "tool", "content": "x"},
        ])
