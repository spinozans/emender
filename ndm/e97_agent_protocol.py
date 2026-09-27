"""Canonical OpenAI/Pi message serialization for the bounded E97 agent."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

RS = "\x1e"
DENSE_AGENT_V1_SYSTEM = (
    'You are a precise tool-using agent. Respond with either "Action:" and one JSON '
    '"Arguments:" object, or "Final:". Never invent tool results.'
)
DENSE_AGENT_V2_SYSTEM = (
    "Use one registered tool to obtain the requested value. Then call submit_answer "
    "with exactly the value returned by that tool. Respond only with Action and Arguments."
)
DENSE_AGENT_CLI_DIRECT_SYSTEM = (
    "Work only in the current directory. Use cli with an argv array and run the appropriate "
    "repo subcommand directly. Then call submit_answer with an exact value and exact evidence "
    "copied from successful CLI stdout. Respond only with Action and Arguments."
)
DENSE_AGENT_CLI_SYSTEM = (
    "Work only in the current directory. Use cli with an argv array. Use repo --help when "
    "you need to discover repository commands. Then call submit_answer with an exact value "
    "and exact evidence copied from successful CLI stdout. Respond only with Action and Arguments."
)
E97_PI_CORE_SYSTEM = (
    "You are a coding agent operating in Pi in the current working directory. "
    "Use read, bash, edit, and write to inspect, change, and verify repository files. "
    "For a tool call respond with exactly Action and one JSON Arguments object. "
    "Never invent tool results. After the work is verified, respond with Final and a "
    "concise evidence-grounded summary."
)
E97_PI_AGENT_SYSTEM_V2 = (
    "You are a coding agent operating in Pi in the current working directory. "
    "Repository paths are unknown unless the user, environment context, or a tool result "
    "establishes them; never substitute a memorized path. If a needed path is unspecified "
    "or unverified, inspect with bash using pwd, find, or another bounded search before "
    "reading or changing it. Use exactly one tool per turn: read(path,offset,limit), "
    "bash(command), edit(path,oldText,newText), or write(path,content). For a tool call "
    "respond with exactly Action and one JSON Arguments object. Treat environment listings "
    "as bounded hints and recover from stale hints or tool failures instead of repeating an "
    "identical failed call. Never invent tool results. Test or otherwise verify changes, then "
    "respond with Final and a concise evidence-grounded summary."
)
E97_PI_AGENT_ANALYSIS_SYSTEM_V1 = (
    E97_PI_AGENT_SYSTEM_V2 + " Before every tool call or final answer, produce a bounded "
    "private rationale in the dedicated reasoning field. The runtime will serialize it as "
    "one canonical Analysis JSON string. Never place private rationale inside tool arguments "
    "or the visible final answer."
)
ROLE_LABELS = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool",
}
_TOOL_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
MAX_PRIVATE_ANALYSIS_BYTES = 65_536


class AgentProtocolError(ValueError):
    """An incoming message or generated action violates the bounded protocol."""


@dataclass(frozen=True)
class ParsedAgentTurn:
    kind: str
    raw_text: str
    tool_name: str | None = None
    arguments_json: str | None = None
    arguments: Mapping[str, Any] | None = None
    private_analysis: str | None = None
    final_text: str | None = None


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        pieces = []
        for item in content:
            if not isinstance(item, Mapping) or item.get("type") != "text":
                raise AgentProtocolError("E97 accepts text message content only")
            text = item.get("text")
            if not isinstance(text, str):
                raise AgentProtocolError("text content items require string text")
            pieces.append(text)
        return "".join(pieces)
    raise AgentProtocolError("message content must be text, text items, or null")


def _private_analysis_prefix(message: Mapping[str, Any], *, enabled: bool) -> str:
    present = "reasoning_content" in message
    reasoning = message.get("reasoning_content")
    if not enabled:
        if present:
            raise AgentProtocolError("private analysis requires the analysis protocol")
        return ""
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise AgentProtocolError("analysis protocol requires non-empty reasoning_content")
    if len(reasoning.encode("utf-8")) > MAX_PRIVATE_ANALYSIS_BYTES:
        raise AgentProtocolError("private analysis exceeds the decoded byte limit")
    return "Analysis: " + json.dumps(reasoning, ensure_ascii=False, separators=(",", ":")) + "\n"


def _assistant_body(message: Mapping[str, Any], *, private_analysis: bool) -> str:
    tool_calls = message.get("tool_calls")
    content = _text_content(message.get("content"))
    prefix = _private_analysis_prefix(message, enabled=private_analysis)
    if tool_calls:
        if content:
            raise AgentProtocolError("assistant tool calls cannot also contain visible text")
        if not isinstance(tool_calls, list) or len(tool_calls) != 1:
            raise AgentProtocolError("E97 supports exactly one tool call per turn")
        call = tool_calls[0]
        if not isinstance(call, Mapping) or call.get("type", "function") != "function":
            raise AgentProtocolError("assistant tool call must be a function")
        function = call.get("function")
        if not isinstance(function, Mapping):
            raise AgentProtocolError("assistant tool call requires function metadata")
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not _TOOL_NAME.fullmatch(name):
            raise AgentProtocolError("assistant tool name is invalid")
        if isinstance(arguments, Mapping):
            arguments = json.dumps(arguments, separators=(",", ":"), ensure_ascii=False)
        if not isinstance(arguments, str):
            raise AgentProtocolError("assistant tool arguments must be a JSON string or object")
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise AgentProtocolError("assistant tool arguments are invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise AgentProtocolError("assistant tool arguments must decode to an object")
        return prefix + f"Action: {name}\nArguments: {arguments}"
    if not content:
        raise AgentProtocolError("assistant message requires text or one tool call")
    return prefix + content.removesuffix(RS)


def serialize_pi_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    append_assistant_header: bool = True,
    private_analysis: bool = False,
) -> str:
    """Serialize a Pi/OpenAI message list into the native E97 transcript."""

    if not messages:
        raise AgentProtocolError("at least one message is required")
    sections: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise AgentProtocolError(f"message {index} must be an object")
        role = message.get("role")
        if role not in ROLE_LABELS:
            raise AgentProtocolError(f"unsupported message role: {role!r}")
        if role == "assistant":
            # RS was a pretraining record boundary, so never place it between
            # coherent agent turns. Role headers provide transcript framing.
            body = _assistant_body(message, private_analysis=private_analysis)
        else:
            body = _text_content(message.get("content"))
            if role == "tool" and not body:
                raise AgentProtocolError("tool result content cannot be empty")
        sections.append(f"{ROLE_LABELS[role]}:\n{body}")
    if append_assistant_header:
        if messages[-1].get("role") == "assistant":
            raise AgentProtocolError("cannot generate immediately after an assistant message")
        sections.append("Assistant:\n")
    return "\n\n".join(sections)


def parse_agent_turn(text: str, *, private_analysis: bool = False) -> ParsedAgentTurn:
    """Parse one complete E97 assistant turn (legacy terminal RS is accepted)."""

    if not isinstance(text, str):
        raise AgentProtocolError("generated turn must be text")
    raw = text.split(RS, 1)[0]
    body = raw
    analysis = None
    if private_analysis:
        first_line, separator, body = raw.partition("\n")
        if not separator or not first_line.startswith("Analysis: "):
            raise AgentProtocolError("analysis turn requires a canonical Analysis JSON string")
        encoded = first_line[len("Analysis: "):]
        try:
            analysis = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise AgentProtocolError("private analysis is invalid JSON") from exc
        if not isinstance(analysis, str) or not analysis.strip():
            raise AgentProtocolError("private analysis must decode to a non-empty string")
        if len(analysis.encode("utf-8")) > MAX_PRIVATE_ANALYSIS_BYTES:
            raise AgentProtocolError("private analysis exceeds the decoded byte limit")
        canonical = json.dumps(analysis, ensure_ascii=False, separators=(",", ":"))
        if encoded != canonical:
            raise AgentProtocolError("private analysis JSON string is not canonical")
    elif raw.startswith("Analysis: "):
        raise AgentProtocolError("private analysis requires the analysis protocol")
    if body.startswith("Action: "):
        first_line, separator, arguments_json = body.partition("\nArguments: ")
        if not separator or not arguments_json:
            raise AgentProtocolError("action requires one Arguments JSON object")
        name = first_line[len("Action: "):]
        if not _TOOL_NAME.fullmatch(name):
            raise AgentProtocolError("generated tool name is invalid")
        try:
            arguments = json.loads(arguments_json)
        except json.JSONDecodeError as exc:
            raise AgentProtocolError("generated tool arguments are invalid JSON") from exc
        if not isinstance(arguments, dict):
            raise AgentProtocolError("generated tool arguments must be an object")
        return ParsedAgentTurn(
            kind="tool_call",
            raw_text=raw,
            tool_name=name,
            arguments_json=arguments_json,
            arguments=arguments,
            private_analysis=analysis,
        )
    if body.startswith("Final:"):
        return ParsedAgentTurn(
            kind="final", raw_text=raw, private_analysis=analysis, final_text=body)
    expected = "analysis plus Action:/Final:" if private_analysis else "Action: or Final:"
    raise AgentProtocolError(f"generated turn must begin with {expected}")


def generated_turn_is_complete(text: str, *, private_analysis: bool = False) -> bool:
    """Return whether incremental generation reached a safe Pi turn boundary."""
    try:
        turn = parse_agent_turn(text, private_analysis=private_analysis)
    except AgentProtocolError:
        return False
    if turn.kind == "tool_call":
        return True
    # Canonical SFT finals are one concise line. Stop at their first newline so
    # an otherwise correct final cannot drift into a memorized next transcript.
    final_text = turn.final_text or turn.raw_text
    return turn.kind == "final" and ("\n" in final_text or RS in text)


def allowed_tool_names(tools: Any) -> frozenset[str]:
    """Extract the closed function-tool vocabulary from an OpenAI request."""

    if tools is None:
        return frozenset()
    if not isinstance(tools, list):
        raise AgentProtocolError("tools must be an array")
    names = []
    for tool in tools:
        if not isinstance(tool, Mapping) or tool.get("type") != "function":
            raise AgentProtocolError("only function tools are supported")
        function = tool.get("function")
        name = function.get("name") if isinstance(function, Mapping) else None
        if not isinstance(name, str) or not _TOOL_NAME.fullmatch(name):
            raise AgentProtocolError("tool definition requires a valid function name")
        names.append(name)
    if len(names) != len(set(names)):
        raise AgentProtocolError("tool names must be unique")
    return frozenset(names)


def validate_generated_tool(turn: ParsedAgentTurn, tools: Any) -> None:
    """Reject unknown model-selected tools before Pi can execute them."""

    if turn.kind != "tool_call":
        return
    allowed = allowed_tool_names(tools)
    if turn.tool_name not in allowed:
        raise AgentProtocolError(f"model selected unknown tool: {turn.tool_name}")
