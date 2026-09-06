"""Strict Phase C ingestion for authentic Pi JSON event traces.

This module deliberately accepts only the small event vocabulary emitted by
``pi --mode json``.  Model text is parsed by the existing E97 protocol parser;
tool execution and state receipts are joined by Pi's tool-call id, never by
position alone.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Iterable, Mapping, Sequence

from ndm.e97_agent_protocol import AgentProtocolError, parse_agent_turn
from ndm.e97_onpolicy_records import (
    ActionProgressReceipt,
    NoProgressDecision,
    NoProgressDetector,
    canonical_action,
    canonical_json,
    sha256_json,
    sha256_text,
)
from ndm.e97_task_lake import validate_task_bundle, validate_source_registry


PHASE_C_RECEIPT_SCHEMA = "emender-e97-authentic-rollout-receipt-v1"
NO_TOOL_OUTPUT = "(no tool output)"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_CALL_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_ALLOWED_EVENT_TYPES = {
    "session", "agent_start", "agent_end", "agent_settled", "turn_start", "turn_end",
    "message_start", "message_update", "message_end", "tool_execution_start",
    "tool_execution_end",
}


class PiEventError(ValueError):
    """A Pi event trace is malformed, ambiguous, or incomplete."""


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise PiEventError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _call_id(value: Any, name: str = "tool call id") -> str:
    if not isinstance(value, str) or not _CALL_ID.fullmatch(value):
        raise PiEventError(f"{name} is invalid")
    return value


def _text_parts(content: Any, *, where: str, allow_empty: bool = False) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list) or (not content and not allow_empty):
        raise PiEventError(f"{where} content must be a non-empty Pi content list")
    pieces: list[str] = []
    for item in content:
        if not isinstance(item, Mapping) or item.get("type") != "text" or not isinstance(item.get("text"), str):
            raise PiEventError(f"{where} contains non-text content")
        pieces.append(item["text"])
    return "".join(pieces)


def _result_text(value: Any, *, is_error: bool) -> str:
    """Render Pi's result content while retaining the raw result separately."""
    if isinstance(value, Mapping) and "content" in value:
        text = _text_parts(value["content"], where="tool execution result", allow_empty=True)
    elif isinstance(value, list):
        text = _text_parts(value, where="tool execution result", allow_empty=True)
    elif isinstance(value, str):
        text = value
    else:
        text = canonical_json(value)
    if not text:
        text = NO_TOOL_OUTPUT
    # Error status is retained as data, not inferred from text.
    return text


def _assistant_message(message: Mapping[str, Any]) -> tuple[str, str | None]:
    content = message.get("content")
    if not isinstance(content, list) or not content:
        raise PiEventError("Pi assistant message requires a content list")
    calls = [item for item in content if isinstance(item, Mapping) and item.get("type") == "toolCall"]
    texts = [item for item in content if isinstance(item, Mapping) and item.get("type") == "text"]
    unknown = [item for item in content if not isinstance(item, Mapping) or item.get("type") not in {"toolCall", "text"}]
    if unknown or (calls and texts) or len(calls) > 1:
        raise PiEventError("assistant message must contain one toolCall or text only")
    if calls:
        call = calls[0]
        call_id = _call_id(call.get("id"))
        name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, Mapping):
            raise PiEventError("Pi toolCall requires name and object arguments")
        try:
            text = canonical_action(name, arguments)
        except (ValueError, AgentProtocolError) as exc:
            raise PiEventError("Pi toolCall is not canonical E97 protocol") from exc
        return text, call_id
    text = _text_parts(content, where="assistant")
    try:
        turn = parse_agent_turn(text)
    except AgentProtocolError as exc:
        raise PiEventError("assistant text is not canonical E97 protocol") from exc
    if turn.kind != "final":
        raise PiEventError("assistant text without toolCall must be a terminal Final")
    return turn.raw_text, None


def _tool_result_message(message: Mapping[str, Any]) -> tuple[str, str, bool]:
    call_id = _call_id(message.get("toolCallId"))
    is_error = message.get("isError")
    if not isinstance(is_error, bool):
        raise PiEventError("Pi tool result requires boolean isError")
    if "content" not in message:
        raise PiEventError("Pi tool result requires content")
    text = _text_parts(message["content"], where="tool result", allow_empty=True)
    return call_id, text or NO_TOOL_OUTPUT, is_error


@dataclass(frozen=True)
class PiAction:
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    action_text: str
    observation_text: str
    observation_raw: Any
    is_error: bool


@dataclass(frozen=True)
class PiTranscript:
    messages: list[dict[str, Any]]
    actions: list[PiAction]
    event_count: int
    terminal_status: str
    terminal_error: Any = None


def parse_pi_events(
    events: Iterable[Mapping[str, Any]],
    *,
    system_prompt: str,
    user_prompt: str,
) -> PiTranscript:
    """Parse a complete authentic Pi event sequence into canonical messages.

    ``tool_execution_start`` and ``tool_execution_end`` are mandatory for every
    action.  A Pi ``toolResult`` message, when present, must agree with the end
    event.  Metadata/update events are accepted only from Pi's known JSON mode;
    unknown events, duplicate ids, missing ends, and actions after Final fail.
    """
    if not isinstance(system_prompt, str) or not system_prompt:
        raise PiEventError("system prompt is required")
    if not isinstance(user_prompt, str) or not user_prompt:
        raise PiEventError("user prompt is required")
    rows = list(events)
    if not rows:
        raise PiEventError("Pi event trace is empty")
    messages: list[dict[str, Any]] = [
        {"role": "system", "text": system_prompt, "loss": 0},
        {"role": "user", "text": user_prompt, "loss": 0},
    ]
    actions: list[PiAction] = []
    started: dict[str, dict[str, Any]] = {}
    ended: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    pending_action: tuple[str, str, dict[str, Any], str] | None = None
    tool_result_ids: set[str] = set()
    saw_session = False
    saw_agent_start = False
    saw_user = False
    saw_final = False
    saw_agent_end = False
    saw_agent_settled = False
    terminal_status: str | None = None
    terminal_error: Any = None

    for index, event in enumerate(rows):
        if not isinstance(event, Mapping):
            raise PiEventError(f"event {index} is not an object")
        if saw_agent_end and event.get("type") != "agent_settled":
            raise PiEventError("events appeared after agent_end")
        kind = event.get("type")
        if kind not in _ALLOWED_EVENT_TYPES:
            raise PiEventError(f"event {index} has unsupported type {kind!r}")
        if index == 0:
            if kind != "session":
                raise PiEventError("trace must begin with one session event")
            saw_session = True
            continue
        if kind == "session":
            raise PiEventError("duplicate or out-of-order session event")
        if kind == "agent_start":
            if saw_agent_start or saw_user or terminal_status is not None:
                raise PiEventError("duplicate or out-of-order agent_start")
            saw_agent_start = True
            continue
        if not saw_agent_start:
            raise PiEventError("trace event appeared before agent_start")
        if kind == "agent_settled":
            if saw_agent_settled or not saw_agent_end or terminal_status is None:
                raise PiEventError("agent_settled has incomplete ordering")
            saw_agent_settled = True
            continue
        if saw_agent_settled:
            raise PiEventError("events appeared after agent_settled")
        if kind == "message_end":
            message = event.get("message")
            if not isinstance(message, Mapping):
                raise PiEventError("message_end requires a message object")
            role = message.get("role")
            if role == "user":
                if saw_user or pending_action is not None or saw_final or terminal_status is not None:
                    raise PiEventError("user message ordering is incomplete or duplicated")
                emitted = _text_parts(message.get("content"), where="user")
                if emitted != user_prompt:
                    raise PiEventError("Pi user message does not equal supplied user prompt")
                saw_user = True
            elif role == "assistant":
                if not saw_user or pending_action is not None or saw_final or terminal_status is not None:
                    raise PiEventError("assistant message ordering is incomplete or duplicated")
                raw_content = message.get("content")
                if message.get("stopReason", event.get("stopReason")) == "error" and raw_content == []:
                    terminal_status = "error"
                    terminal_error = message.get("errorMessage", message.get("error", event.get("error")))
                    continue
                text, call_id = _assistant_message(message)
                if call_id is None:
                    saw_final = True
                    terminal_status = "success"
                    messages.append({"role": "assistant", "text": text, "loss": 0})
                else:
                    call = next(item for item in message["content"] if item.get("type") == "toolCall")
                    arguments = dict(call["arguments"])
                    messages.append({"role": "assistant", "text": text, "loss": 0, "tool_call_id": call_id})
                    pending_action = (call_id, call["name"], arguments, text)
            elif role == "toolResult":
                if pending_action is not None:
                    raise PiEventError("tool result arrived before execution events completed")
                call_id, text, is_error = _tool_result_message(message)
                if call_id not in ended or call_id in tool_result_ids:
                    raise PiEventError("tool result has missing or duplicate execution linkage")
                end = ended[call_id]
                if end["text"] != text or end["is_error"] != is_error:
                    raise PiEventError("tool result disagrees with tool_execution_end")
                messages.append({"role": "tool", "text": text, "loss": 0, "tool_call_id": call_id})
                tool_result_ids.add(call_id)
            else:
                raise PiEventError("message_end role must be assistant or toolResult")
        elif kind == "tool_execution_start":
            if pending_action is None or saw_final:
                raise PiEventError("tool execution start is not preceded by one assistant toolCall")
            call_id = _call_id(event.get("toolCallId"))
            if call_id != pending_action[0] or call_id in seen_ids:
                raise PiEventError("tool execution start has malformed or duplicate linkage")
            if event.get("toolName") != pending_action[1] or event.get("args") != pending_action[2]:
                raise PiEventError("tool execution start does not match assistant toolCall")
            seen_ids.add(call_id)
            started[call_id] = {"name": event["toolName"], "args": dict(event["args"])}
        elif kind == "tool_execution_end":
            if pending_action is None:
                raise PiEventError("tool execution end has no pending action")
            call_id = _call_id(event.get("toolCallId"))
            if call_id != pending_action[0] or call_id not in started or call_id in ended:
                raise PiEventError("tool execution end has malformed or duplicate linkage")
            if "toolName" in event and event["toolName"] != pending_action[1]:
                raise PiEventError("tool execution end toolName disagrees with assistant toolCall")
            is_error = event.get("isError")
            if not isinstance(is_error, bool):
                raise PiEventError("tool execution end requires boolean isError")
            if "result" not in event:
                raise PiEventError("tool execution end requires result")
            raw_result = event["result"]
            text = _result_text(raw_result, is_error=is_error)
            ended[call_id] = {"text": text, "is_error": is_error}
            actions.append(PiAction(call_id, pending_action[1], pending_action[2], pending_action[3], text, raw_result, is_error))
            pending_action = None
        elif kind == "message_start" or kind == "message_update":
            # These events are framing/streaming evidence.  Their complete
            # message_end remains the sole authority for canonical messages.
            continue
        elif kind == "agent_end":
            if saw_agent_end:
                raise PiEventError("duplicate agent_end")
            if terminal_status is None:
                raise PiEventError("agent_end arrived before terminal assistant outcome")
            saw_agent_end = True
            if pending_action is not None or len(ended) != len(started):
                raise PiEventError("agent_end arrived with incomplete tool execution")
        elif kind in {"turn_start", "turn_end"}:
            continue

    if pending_action is not None:
        raise PiEventError("trace ended with incomplete action/result ordering")
    if not saw_session or not saw_agent_start or not saw_user:
        raise PiEventError("trace lacks required session, agent_start, or user message")
    if terminal_status is None:
        raise PiEventError("trace has no terminal assistant outcome")
    if not saw_agent_end:
        raise PiEventError("trace has no agent_end")
    if not saw_agent_settled:
        raise PiEventError("trace has no terminal agent_settled")
    tool_messages = [m for m in messages if m["role"] == "tool"]
    if len(tool_messages) != len(actions) or len(tool_result_ids) != len(actions):
        raise PiEventError("every execution must have one canonical toolResult")
    return PiTranscript(
        messages=messages, actions=actions, event_count=len(rows),
        terminal_status=terminal_status, terminal_error=terminal_error,
    )


def apply_no_progress(transcript: PiTranscript, state_receipts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Join explicit post-action receipts and record detector decisions."""
    if len(state_receipts) != len(transcript.actions):
        raise PiEventError("state receipt count does not match tool actions")
    detector = NoProgressDetector()
    decisions: list[dict[str, Any]] = []
    for index, (action, raw) in enumerate(zip(transcript.actions, state_receipts)):
        if not isinstance(raw, Mapping) or set(raw) != {
            "sequence", "tool_call_id", "tool_name", "arguments", "observation_sha256",
            "workspace_state", "source_ledger", "acquired_observations",
        }:
            raise PiEventError(f"state receipt {index} fields are not exact")
        if raw["sequence"] != index or raw["tool_call_id"] != action.tool_call_id:
            raise PiEventError(f"state receipt {index} sequence or tool linkage mismatch")
        if raw["tool_name"] != action.tool_name or raw["arguments"] != action.arguments:
            raise PiEventError(f"state receipt {index} action mismatch")
        if not isinstance(raw["observation_sha256"], str) or not _DIGEST.fullmatch(raw["observation_sha256"]):
            raise PiEventError(f"state receipt {index} observation digest is invalid")
        if raw["observation_sha256"] != sha256_text(action.observation_text):
            raise PiEventError(f"state receipt {index} observation digest mismatch")
        decision = detector.observe(ActionProgressReceipt(
            action.tool_name, action.arguments, raw["workspace_state"],
            raw["source_ledger"], raw["acquired_observations"],
        ))
        decisions.append({
            "sequence": index,
            "tool_call_id": action.tool_call_id,
            "kind": decision.kind,
            "action_fingerprint": decision.action_fingerprint,
            "progress_fingerprint": decision.progress_fingerprint,
            **({"observation": decision.observation} if decision.observation else {}),
        })
    return decisions


def validate_collection_inputs(
    *, task_bundle: Mapping[str, Any], source_registry: Mapping[str, Any],
    task_bundle_sha256: str, source_registry_sha256: str,
    actual_task_bundle_sha256: str, actual_source_registry_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate both file pins before any event or receipt is interpreted."""
    _digest(task_bundle_sha256, "task bundle SHA-256")
    _digest(source_registry_sha256, "source registry SHA-256")
    if task_bundle_sha256 != actual_task_bundle_sha256:
        raise PiEventError("task bundle SHA-256 mismatch")
    if source_registry_sha256 != actual_source_registry_sha256:
        raise PiEventError("source registry SHA-256 mismatch")
    try:
        registry = validate_source_registry(source_registry)
        bundle = validate_task_bundle(task_bundle, registry=registry)
    except ValueError as exc:
        raise PiEventError(f"task/source validation failed: {exc}") from exc
    return bundle, registry


def receipt_fingerprint(receipt: Mapping[str, Any]) -> str:
    return sha256_text(f"{PHASE_C_RECEIPT_SCHEMA}\0{canonical_json(dict(receipt))}")


__all__ = [
    "PHASE_C_RECEIPT_SCHEMA", "PiEventError", "PiAction", "PiTranscript",
    "parse_pi_events", "apply_no_progress", "validate_collection_inputs",
    "receipt_fingerprint",
]
