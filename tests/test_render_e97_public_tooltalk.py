"""Unit tests for the E97 public tooltalk codec spike renderer (CPU only)."""
import tiktoken

from scripts.e97_open_swe_native_codec import compact, strict_json
from scripts.e97_pi_native_codec import parse_turn
from scripts.render_e97_public_tooltalk import (Renderer, parse_call_block,
                                                smoltalk2_tools, _call_result, _strict_object)

ENCODING = tiktoken.get_encoding("p50k_base")
LT = chr(60)
CALL_OPEN = LT + "tool_call" + chr(62)
CALL_CLOSE = LT + "/tool_call" + chr(62)
TOOLS_OPEN = LT + "tools" + chr(62)
TOOLS_CLOSE = LT + "/tools" + chr(62)


def tool(name):
    return {"name": name, "label": name, "description": f"tool {name}",
            "parameters": {"type": "object", "properties": {}}}


def call(name, arguments_raw=None, arguments=None, call_id=None):
    return {"id": call_id, "name": name, "arguments": arguments if arguments_raw is None else arguments,
            "arguments_raw": arguments_raw}


def assistant(reasoning=None, text=None, calls=(), final_text=None):
    return {"role": "assistant", "reasoning": reasoning, "text": text,
            "calls": list(calls), "final_text": final_text}


def result(call_id, content, name="tool"):
    return {"role": "tool", "tool_call_id": call_id, "tool_name": name, "content": content}


def test_call_result_verbatim_json_string():
    parsed = _call_result("t", '{"a": 1}')
    assert parsed["verbatim"] and parsed["arguments_raw"] == '{"a": 1}'
    assert parsed["arguments"] == {"a": 1}


def test_call_result_python_literal_canonicalized():
    parsed = _call_result("t", {"a": 1})
    assert not parsed["verbatim"] and parsed["arguments_raw"] is None


def test_parse_call_block_strict_json_keeps_verbatim_span():
    block = '{"name": "f", "arguments": {"b": 2, "a": 1}}'
    parsed = parse_call_block(block)
    assert parsed["verbatim"] and parsed["arguments_raw"] == '{"b": 2, "a": 1}'
    assert strict_json(parsed["arguments_raw"]) == {"b": 2, "a": 1}


def test_parse_call_block_python_literal_parses_and_canonicalizes():
    block = "{'name': 'f', 'arguments': {'x': 'it'}}"
    parsed = parse_call_block(block)
    assert parsed["name"] == "f" and parsed["arguments"] == {"x": "it"}
    assert not parsed["verbatim"] and parsed["arguments_raw"] is None


def test_parse_call_block_hermes_mixed_quotes():
    block = '{"arguments": {"queries": [\'What is a?\', \'Why b?\'], "name": "ExpertQA"}'
    # unterminated objects in this family are upstream corruption and must raise
    try:
        parse_call_block(block)
        raised = False
    except (ValueError, IndexError):
        raised = True
    assert raised


def test_parse_call_block_requires_name():
    for bad in ('{"arguments": {}}', '{"name": "", "arguments": {}}', '[]'):
        try:
            parse_call_block(bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_smoltalk2_tools_three_registry_shapes():
    json_lines = '{"type": "function", "function": {"name": "a", "parameters": {}}}\n' \
                 '{"type": "function", "function": {"name": "b", "parameters": {}}}'
    assert [t["name"] for t in smoltalk2_tools({"xml_tools": [json_lines]})] == ["a", "b"]
    wrapped_json = TOOLS_OPEN + '\n[{"type": "function", "function": {"name": "c", "parameters": {}}}]' + "\n" + TOOLS_CLOSE
    assert [t["name"] for t in smoltalk2_tools({"xml_tools": [wrapped_json]})] == ["c"]
    wrapped_literal = TOOLS_OPEN + "\n[{'type': 'function', 'function': {'name': 'd', 'parameters': {}}}]" + "\n" + TOOLS_CLOSE
    assert [t["name"] for t in smoltalk2_tools({"xml_tools": [wrapped_literal]})] == ["d"]


def test_smoltalk2_tools_registry_after_prose_marker_mention():
    prose = ("You are an expert. The registry is within XML tags " + TOOLS_OPEN + TOOLS_CLOSE +
             ". Don't assume. " + TOOLS_OPEN + '\n[{"type": "function", "function": '
             '{"name": "late", "parameters": {}}}]' + "\n" + TOOLS_CLOSE + " closing prose.")
    assert [t["name"] for t in smoltalk2_tools({"xml_tools": [prose]})] == ["late"]


def test_renderer_frames_and_seam():
    renderer = Renderer(ENCODING)
    record = {"tools": [tool("search"), tool("finish")], "turns": [
        {"role": "user", "content": "find the answer"},
        assistant(text="I will search.", calls=[call("search", arguments={"q": "x"}, call_id="c1")]),
        result("c1", "answer=42", name="search"),
        assistant(reasoning="The result says 42.", text="The answer is 42."),
    ]}
    out = renderer.render(record)
    metrics, transcript = out["metrics"], out["transcript"]
    assert metrics["seam_finish_after_result"] == 1
    assert metrics["orphan_calls"] == 0 and metrics["unmatched_results"] == 0
    assert metrics["finish_frames"] == 1
    frames = [s for s in transcript.split("\n\n") if s.startswith("Assistant:\n")]
    assert len(frames) == 2
    for frame in frames:
        parsed = parse_turn(frame[len("Assistant:\n"):])
        assert isinstance(parsed["arguments"], dict)
    first = strict_json(frames[0][len("Assistant:\n"):].split("\n")[4][len("Arguments: "):])
    assert first == {"q": "x"}
    assert strict_json(frames[0][len("Assistant:\n"):].split("\n")[1][len("Commentary: "):]) == "I will search."
    finish = frames[1][len("Assistant:\n"):].split("\n")
    assert finish[3] == "Action: finish"
    assert strict_json(finish[4][len("Arguments: "):])["message"] == "The answer is 42."
    assert strict_json(finish[0][len("Analysis: "):]) == "The result says 42."
    assert metrics["tokens"] > metrics["assistant_tokens"] > 0
    assert metrics["context_tokens"] == metrics["tokens"] - metrics["assistant_tokens"]


def test_renderer_parallel_calls_serialize_frame_result_pairs():
    renderer = Renderer(ENCODING)
    record = {"tools": [tool("a"), tool("b")], "turns": [
        {"role": "user", "content": "go"},
        assistant(calls=[call("a", arguments={"x": 1}, call_id="p1"),
                         call("b", arguments={"y": 2}, call_id="p2")]),
        result("p1", "one", name="a"), result("p2", "two", name="b"),
        assistant(text="both done."),
    ]}
    out = renderer.render(record)
    m = out["metrics"]
    assert m["n_calls"] == 2 and m["n_results"] == 2
    assert m["orphan_calls"] == 0 and m["seam_finish_after_result"] == 1
    sections = out["transcript"].split("\n\n")
    kinds = [s.split("\n", 1)[0] for s in sections]
    assert kinds == ["Protocol:", "User:", "Assistant:", "ToolResult:", "Assistant:", "ToolResult:", "Assistant:"]


def test_renderer_multiline_verbatim_arguments_canonicalized():
    renderer = Renderer(ENCODING)
    raw = '{\n  "x": 1\n}'
    record = {"tools": [tool("a")], "turns": [
        {"role": "user", "content": "go"},
        assistant(calls=[call("a", arguments_raw=raw, arguments={"x": 1})]),
        result(None, "ok", name="a"),
    ]}
    m = renderer.render(record)["metrics"]
    assert m["canonicalized_multiline_argument_frames"] == 1
    assert m["verbatim_argument_frames"] == 0 and m["malformed_frames"] == 0


def test_renderer_single_line_verbatim_arguments_kept():
    renderer = Renderer(ENCODING)
    raw = '{"x": 1, "z": 2}'
    record = {"tools": [tool("a")], "turns": [
        {"role": "user", "content": "go"},
        assistant(calls=[call("a", arguments_raw=raw, arguments={"x": 1, "z": 2})]),
        result(None, "ok", name="a"),
    ]}
    out = renderer.render(record)
    assert out["metrics"]["verbatim_argument_frames"] == 1
    frame = next(s for s in out["transcript"].split("\n\n") if s.startswith("Assistant:\n"))
    assert frame[len("Assistant:\n"):].split("\n")[4] == "Arguments: " + raw


def test_renderer_orphan_accounting_and_terminal_orphan():
    renderer = Renderer(ENCODING)
    dangling_terminal = {"tools": [tool("a"), tool("b")], "turns": [
        {"role": "user", "content": "go"},
        assistant(calls=[call("a", arguments={})]),
        result(None, "ok", name="a"),
        assistant(calls=[call("b", arguments={})]),
    ]}
    m = renderer.render(dangling_terminal)["metrics"]
    assert m["orphan_calls"] == 1 and m["orphan_terminal_calls"] == 1 and m["orphan_mid_calls"] == 0

    dangling_mid = {"tools": [tool("a"), tool("b")], "turns": [
        {"role": "user", "content": "go"},
        assistant(calls=[call("a", arguments={}, call_id="x")]),
        assistant(calls=[call("b", arguments={}, call_id="y")]),
        result("y", "for b", name="b"),
    ]}
    m = renderer.render(dangling_mid)["metrics"]
    assert m["orphan_calls"] == 1 and m["orphan_mid_calls"] == 1 and m["orphan_terminal_calls"] == 0


def test_renderer_undeclared_action_flagged():
    renderer = Renderer(ENCODING)
    record = {"tools": [tool("a")], "turns": [
        {"role": "user", "content": "go"},
        assistant(calls=[call("undeclared_tool", arguments={})]),
    ]}
    m = renderer.render(record)["metrics"]
    assert m["invalid_frames"] == 1 and m["undeclared_action_frames"] == 1 and m["malformed_frames"] == 0


def test_strict_object_rejects_non_object():
    assert _strict_object('{"a": 1}') == {"a": 1}
    assert _strict_object('"text"') is None
    assert _strict_object('not json') is None
