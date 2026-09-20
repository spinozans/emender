#!/usr/bin/env python3
"""E97 public intermingled chat+tool-call codec spike (STAGED; never trains).

Renders OpenAI-style public function-calling conversations into the Pi-native
five-line Analysis/Commentary/Action/Arguments frame protocol
(scripts/e97_pi_native_codec.py) and measures the chat->tool->chat seam.

Rendering policy (docs/validation/e97-public-intermingled-tool-talk-dataset-survey-v1.md,
'Rendering plan sketch'):
  - tool manifest -> one preamble Protocol registry line (never per turn);
  - assistant tool_calls[] -> one five-line frame per call:
    Action: <function name> / Arguments: <verbatim JSON when the source
    serializes strict JSON (never re-serialised); canonical compact() form only
    when the source stores structured arguments (Python literals / dicts);
  - assistant text in the same turn as, or immediately preceding, calls ->
    Commentary of the first call frame; source reasoning (smoltalk2 think
    blocks, reasoning_content, Tool-Reasoning reasoning blocks) -> Analysis
    per the think/no-think codec policy, never concatenated into answers;
  - plain-text assistant answers (no calls) -> Action: finish /
    Arguments: {"message": <answer>} (finish.message is the public channel);
    SFT transcripts may continue with further User sections after a finish
    frame -- supervision is per frame, unlike the runtime episode lifecycle;
  - role:tool results -> ToolResult sections (causal context, never targets);
  - parallel calls -> serialized as consecutive frame/ToolResult pairs (the
    codec admits exactly one call per turn).

Subcommands: download, render, overlap, verify-licenses. Each render produces
an evidence-dir-style authority per source under --output. STAGED: nothing
here enters a prep without operator sign-off.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import shutil
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import HfApi, hf_hub_download

from scripts.e97_open_swe_native_codec import compact, strict_json
from scripts.e97_pi_native_codec import FRAME, PROFILE, PSEUDO_SPECS, parse_turn

SEED_PREFIX = "e97-public-tooltalk-codec-spike-v1"
SPIKE_SCHEMA = "emender-e97-public-tooltalk-codec-spike-v1"
TOKENIZER = "p50k_base"
CONTEXT_WINDOW_TOKENS = 65536

SOURCES = {
    "smoltalk2-smolagents": {
        "repo": "HuggingFaceTB/smoltalk2",
        "revision": "fc6cc2103c066455aade5d7fbb346039ae36ca5e",
        "kind": "smoltalk2",
        "files": ["SFT/smolagents_toolcalling_traces_think-00000-of-00001.parquet"],
        "claimed_rows": 9079,
        "license": "Apache-2.0 (new smoltalk2 subsets)",
    },
    "smoltalk2-xlam": {
        "repo": "HuggingFaceTB/smoltalk2",
        "revision": "fc6cc2103c066455aade5d7fbb346039ae36ca5e",
        "kind": "smoltalk2",
        "files": ["SFT/xlam_traces_no_think-00000-of-00001.parquet"],
        "claimed_rows": 59962,
        "license": "upstream Salesforce xLAM CC-BY-4.0 (via smoltalk2)",
    },
    "smoltalk2-hermes": {
        "repo": "HuggingFaceTB/smoltalk2",
        "revision": "fc6cc2103c066455aade5d7fbb346039ae36ca5e",
        "kind": "smoltalk2",
        "files": ["SFT/hermes_function_calling_v1_no_think-00000-of-00001.parquet"],
        "claimed_rows": 8961,
        "license": "Apache-2.0 (upstream NousResearch/hermes-function-calling-v1)",
    },
    "toucan-sft": {
        "repo": "Agent-Ark/Toucan-1.5M",
        "revision": "0df3cf37f2abefb380370cfb02eabea2a35ae782",
        "kind": "toucan",
        "files": ["SFT/train-00000-of-00003.parquet", "SFT/train-00001-of-00003.parquet",
                  "SFT/train-00002-of-00003.parquet"],
        "claimed_rows": 119287,
        "license": "Apache-2.0",
    },
    "tool-reasoning-31k": {
        "repo": "DomofonResearch/Tool-Reasoning-31K",
        "revision": "a586dac8eee6f376c4d50a56f3407ca0019aed50",
        "kind": "toolreason",
        "files": ["data/train-00000.parquet"],
        "claimed_rows": 30764,
        "license": "Apache-2.0",
        "when2call_source_value": "Nvidia-When2Call",
    },
    "nexus-stage1": {
        "repo": "NexusProjectsAI/Nexus-Agents-ToolCalling",
        "revision": "4fdb25354d8ecb334c978827b876acc5a333ca74",
        "kind": "nexus",
        "files": ["stage1/train.jsonl"],
        "claimed_rows": 60185,
        "license": "Apache-2.0",
    },
    "nemotron-v1-interactive": {
        "repo": "nvidia/Nemotron-Agentic-v1",
        "revision": "650d590978ca35c8f1ecea2faf136e5fac421b62",
        "kind": "nemotron",
        # Only the 448 MB interactive_agent split is fetched (for exact counts
        # and uniform sampling); the 5.3 GB tool_calling split is not fetched.
        "files": ["data/interactive_agent.jsonl"],
        "claimed_rows": None,
        "license": "CC-BY-4.0",
    },
}

SAMPLE_SIZES = {"nemotron-v1-interactive": 50}
DEFAULT_SAMPLE = 200

PROTECTED_PANELS = (
    ("/mnt/nvme1n1/erikg/sft/pi-core-eval-v3-blind-family-heldout/manifest.json",
     "/mnt/nvme1n1/erikg/sft/pi-core-eval-v3-blind-family-heldout/records.jsonl"),
    ("/mnt/nvme1n1/erikg/sft/pi-core-eval-v4-post-broad-heldout/manifest.json",
     "/mnt/nvme1n1/erikg/sft/pi-core-eval-v4-post-broad-heldout/records.jsonl"),
    ("/mnt/nvme1n1/erikg/evals/e97-real-repo-holdout-v1/authority/manifest.json",
     "/mnt/nvme1n1/erikg/evals/e97-real-repo-holdout-v1/authority/tasks.jsonl"),
)

_LT = chr(60)
_THINK_OPEN = _LT + "think" + chr(62)
_THINK_CLOSE = _LT + "/think" + chr(62)
_CALL_OPEN = _LT + "tool_call" + chr(62)
_CALL_CLOSE = _LT + "/tool_call" + chr(62)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


# --------------------------------------------------------------------------
# Lenient JSON / Python-literal value scanner with span capture.
# --------------------------------------------------------------------------

def _scan_ws(text: str, i: int) -> int:
    while i < len(text) and text[i] in " \t\r\n":
        i += 1
    return i


def _scan_string(text: str, i: int) -> tuple[str, int]:
    quote = text[i]
    i += 1
    out = []
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            esc = text[i + 1]
            out.append({"n": "\n", "t": "\t", "r": "\r", "0": "\0"}.get(esc, esc))
            i += 2
            continue
        if ch == quote:
            return "".join(out), i + 1
        out.append(ch)
        i += 1
    raise ValueError("unterminated string")


def _scan_value(text: str, i: int) -> tuple[object, int]:
    """Parse one JSON/Python-literal value; objects come back plain."""
    i = _scan_ws(text, i)
    ch = text[i] if i < len(text) else ""
    if ch in "\"'":
        return _scan_string(text, i)
    if ch == "{":
        i = _scan_ws(text, i + 1)
        obj = {}
        if i < len(text) and text[i] == "}":
            return obj, i + 1
        while True:
            key, i = _scan_string(text, i)
            i = _scan_ws(text, i)
            if text[i] != ":":
                raise ValueError("expected colon")
            value, i = _scan_value(text, _scan_ws(text, i + 1))
            obj[key] = value
            i = _scan_ws(text, i)
            if text[i] == ",":
                i = _scan_ws(text, i + 1)
                continue
            if text[i] == "}":
                return obj, i + 1
            raise ValueError("bad object")
    if ch == "[":
        i = _scan_ws(text, i + 1)
        arr = []
        if i < len(text) and text[i] == "]":
            return arr, i + 1
        while True:
            value, i = _scan_value(text, i)
            arr.append(value)
            i = _scan_ws(text, i)
            if text[i] == ",":
                i = _scan_ws(text, i + 1)
                continue
            if text[i] == "]":
                return arr, i + 1
            raise ValueError("bad array")
    m = re.match(r"-?\d+(\.\d+)?([eE][+-]?\d+)?", text[i:])
    if m:
        raw = m.group(0)
        return (float(raw) if any(c in raw for c in ".eE") else int(raw)), i + len(raw)
    for literal, value in (("true", True), ("false", False), ("null", None),
                           ("True", True), ("False", False), ("None", None)):
        if text.startswith(literal, i):
            return value, i + len(literal)
    raise ValueError(f"cannot parse value at offset {i}")


def _scan_object_spans(text: str, i: int) -> tuple[dict, dict, int]:
    """Parse an object while recording each top-level value's source span."""
    i = _scan_ws(text, i + 1)
    obj, spans = {}, {}
    if i < len(text) and text[i] == "}":
        return obj, spans, i + 1
    while True:
        key, i = _scan_string(text, i)
        i = _scan_ws(text, i)
        if text[i] != ":":
            raise ValueError("expected colon")
        start = _scan_ws(text, i + 1)
        value, end = _scan_value(text, start)
        obj[key] = value
        spans[key] = (start, end)
        i = _scan_ws(text, end)
        if text[i] == ",":
            i = _scan_ws(text, i + 1)
            continue
        if text[i] == "}":
            return obj, spans, i + 1
        raise ValueError("bad object")


def _strict_object(raw: str):
    """Return the parsed dict iff raw is strict JSON denoting an object."""
    try:
        value = strict_json(raw)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _call_result(name, value, raw=None) -> dict:
    """Normalize one parsed call to {name, arguments, arguments_raw, verbatim}.

    arguments_raw is the source's own strict-JSON-object serialization when
    available (verbatim Arguments line); otherwise None and the frame falls
    back to the canonical compact(arguments) form.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("call without a name")
    if isinstance(value, str):
        parsed = _strict_object(value)
        if parsed is not None:
            return {"name": name, "arguments": parsed, "arguments_raw": value, "verbatim": True}
        return {"name": name, "arguments": value, "arguments_raw": None, "verbatim": False}
    if raw is not None:
        parsed = _strict_object(raw)
        if parsed is not None:
            return {"name": name, "arguments": parsed, "arguments_raw": raw, "verbatim": True}
    if isinstance(value, dict):
        return {"name": name, "arguments": value, "arguments_raw": None, "verbatim": False}
    raise ValueError("call arguments must be an object or a JSON object string")


def parse_call_block(block: str) -> dict:
    """Parse one smoltalk2 tool_call block (JSON or Python literal)."""
    stripped = block.strip()
    name, arguments, raw = None, None, None
    try:
        obj = strict_json(stripped)
        if not isinstance(obj, dict):
            raise ValueError("call block is not an object")
        name, arguments = obj.get("name"), obj.get("arguments")
        raw = arguments_span(stripped)
    except Exception:
        scanned, spans, _end = _scan_object_spans(stripped, 0)
        extra = set(scanned) - {"name", "arguments"}
        if extra:
            raise ValueError(f"unexpected call block keys {sorted(extra)}")
        name = scanned.get("name")
        arguments = scanned.get("arguments")
        raw = None
        if "arguments" in spans:
            raw = stripped[spans["arguments"][0]:spans["arguments"][1]]
    if not isinstance(name, str) or not name:
        raise ValueError("call block without a name")
    return _call_result(name, arguments, raw)


def arguments_span(text: str):
    """Verbatim strict-JSON span of the 'arguments' value in a JSON call text."""
    m = re.search(r"(?<![A-Za-z0-9_])arguments['\"]?\s*:\s*", text)
    if not m:
        return None
    try:
        start = _scan_ws(text, m.end())
        _value, end = _scan_value(text, start)
    except Exception:
        return None
    raw = text[start:end]
    return raw if _strict_object(raw) is not None else None


# --------------------------------------------------------------------------
# Tool-spec normalization -> Pi registry shape.
# --------------------------------------------------------------------------

def pi_tool(spec) -> dict:
    if not isinstance(spec, dict):
        raise ValueError("tool spec must be an object")
    fn = spec.get("function") if spec.get("type") == "function" and isinstance(spec.get("function"), dict) else spec
    name = fn.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("tool spec without a name")
    params = fn.get("parameters")
    if not isinstance(params, dict):
        params = {"type": "object", "properties": {}}
    return {"name": name, "label": name, "description": fn.get("description") or "",
            "parameters": params}


def smoltalk2_tools(kwargs) -> list:
    """Parse smoltalk2 chat_template_kwargs tool registries.

    Observed shapes: newline-separated strict-JSON specs (smolagents), a
    '<tools>'-wrapped JSON list (xlam), a '<tools>'-wrapped Python-literal
    list (hermes).
    """
    specs = []
    for column in ("python_tools", "xml_tools"):
        for chunk in (kwargs.get(column) or []):
            if not isinstance(chunk, str):
                continue
            body = chunk.strip()
            open_marker, close_marker = chr(60) + "tools>", chr(60) + "/tools>"
            if open_marker in body:
                # Some rows embed the registry in a prompt whose prose mentions the
                # markers; the registry itself is the LAST marker-delimited span.
                start = body.rfind(open_marker)
                end = body.rfind(close_marker)
                body = body[start + len(open_marker): end if end > start else len(body)].strip()
            items = None
            for parser in (strict_json, ast.literal_eval):
                try:
                    value = parser(body)
                except Exception:
                    continue
                if isinstance(value, list):
                    items = value
                    break
            if items is None:
                items = []
                for line in body.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    for parser in (strict_json, ast.literal_eval):
                        try:
                            items.append(parser(line))
                            break
                        except Exception:
                            continue
                    else:
                        raise ValueError("unparsable tool spec line")
            specs.extend(pi_tool(item) for item in items)
    return specs


def plain_text(content):
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [block.get("text") or "" for block in content
                 if isinstance(block, dict) and block.get("type") == "text"]
        return "".join(parts) if parts else None
    raise ValueError("unsupported content shape")


def _assistant_turn(reasoning=None, text=None, calls=None, final_text=None):
    return {"role": "assistant", "reasoning": reasoning, "text": text,
            "calls": calls or [], "final_text": final_text}


# --------------------------------------------------------------------------
# Per-source loaders.  Normalized turn shapes:
#   {'role':'system'|'user','content':str}
#   {'role':'assistant','reasoning','text','calls':[{'id','name','arguments',
#    'arguments_raw'}],'final_text'}
#   {'role':'tool','tool_call_id','tool_name','content'}
# --------------------------------------------------------------------------

def load_smoltalk2(raw_root: Path, spec: dict):
    for row_index, row in _parquet_rows(raw_root, spec["files"]):
        try:
            tools = smoltalk2_tools(row["chat_template_kwargs"] or {})
        except Exception as exc:
            yield row_index, None, f"tool-parse {type(exc).__name__}"
            continue
        turns, failed = [], None
        for message in row["messages"]:
            role, content = message["role"], message["content"]
            if role in ("system", "user"):
                turns.append({"role": role, "content": content})
            elif role == "assistant":
                reasoning, rest = None, content
                if _THINK_OPEN in content:
                    a = content.index(_THINK_OPEN) + len(_THINK_OPEN)
                    b = content.index(_THINK_CLOSE, a)
                    reasoning = content[a:b].strip()
                    rest = content[b + len(_THINK_CLOSE):]
                calls, head, tail = [], rest, ""
                if _CALL_OPEN in rest:
                    head = rest[:rest.index(_CALL_OPEN)]
                    cursor = rest
                    while _CALL_OPEN in cursor:
                        a = cursor.index(_CALL_OPEN) + len(_CALL_OPEN)
                        b = cursor.index(_CALL_CLOSE, a)
                        try:
                            parsed = parse_call_block(cursor[a:b])
                        except Exception as exc:
                            failed = f"call-block {type(exc).__name__}"
                            break
                        calls.append({"id": None, "name": parsed["name"],
                                     "arguments": parsed["arguments"],
                                     "arguments_raw": parsed["arguments_raw"]})
                        cursor = cursor[b + len(_CALL_CLOSE):]
                    tail = cursor
                if failed:
                    break
                text = (head + tail).strip() or None if calls else (rest.strip() or None)
                turns.append(_assistant_turn(reasoning, text, calls))
            elif role == "tool":
                turns.append({"role": "tool", "tool_call_id": None,
                              "tool_name": None, "content": content})
            else:
                failed = f"role {role}"
                break
        yield row_index, (None if failed else
                         {"id": None, "tools": tools, "turns": turns, "flags": {}}), failed


def load_toucan(raw_root: Path, spec: dict):
    for row_index, row in _parquet_rows(raw_root, spec["files"]):
        try:
            tools = [pi_tool(s) for s in strict_json(row["tools"])]
        except Exception as exc:
            yield row_index, None, f"tool-parse {type(exc).__name__}"
            continue
        turns, pending_text, call_group, failed = [], None, [], None

        def flush():
            nonlocal pending_text, call_group
            if call_group:
                turns.append(_assistant_turn(None, pending_text, call_group))
            elif pending_text:
                turns.append(_assistant_turn(None, pending_text))
            pending_text, call_group = None, []

        for message in strict_json(row["messages"]):
            role = message["role"]
            if role == "user":
                flush()
                turns.append({"role": "user", "content": message["content"]})
            elif role == "assistant":
                flush()
                pending_text = message["content"]
            elif role == "tool_call":
                try:
                    try:
                        call = ast.literal_eval(message["content"])
                    except Exception:
                        call = strict_json(message["content"])
                    if not isinstance(call, dict):
                        raise ValueError("call is not an object")
                except Exception as exc:
                    failed = f"tool_call-parse {type(exc).__name__}"
                    break
                arguments = call.get("arguments")
                arguments_raw = None
                if isinstance(arguments, str):
                    arguments_raw = arguments
                    arguments = _strict_object(arguments)
                    if arguments is None:
                        failed = "tool_call-arguments"
                        break
                if not isinstance(call.get("name"), str):
                    failed = "tool_call-name"
                    break
                call_group.append({"id": None, "name": call["name"],
                                   "arguments": arguments, "arguments_raw": arguments_raw})
            elif role == "tool_response":
                flush()
                turns.append({"role": "tool", "tool_call_id": None, "tool_name": None,
                              "content": message["content"]})
            else:
                failed = f"role {role}"
                break
        flush()
        yield row_index, (None if failed else
                         {"id": row.get("uuid"), "tools": tools, "turns": turns,
                          "flags": {"subset_name": row.get("subset_name")}}), failed


def load_toolreason(raw_root: Path, spec: dict):
    for row_index, row in _parquet_rows(raw_root, spec["files"]):
        flags = {"decision_type": row["decision_type"],
                 "has_tool_call": bool(row["has_tool_call"]),
                 "num_tool_calls": int(row["num_tool_calls"]),
                 "upstream_source": row["source"],
                 "when2call_excluded": row["source"] == spec["when2call_source_value"]}
        turns, tools, failed = [], [], None
        try:
            messages = strict_json(row["messages"])
        except Exception as exc:
            yield row_index, None, f"messages-parse {type(exc).__name__}"
            continue
        for message in messages:
            role = message["role"]
            content = message.get("content")
            if role == "available_tools":
                try:
                    tools = [pi_tool(s) for s in strict_json(plain_text(content) or "[]")]
                except Exception as exc:
                    failed = f"available_tools-parse {type(exc).__name__}"
                    break
            elif role in ("system", "user"):
                turns.append({"role": role, "content": plain_text(content) or ""})
            elif role == "assistant":
                reasoning_parts, calls, final_text = [], [], None
                if isinstance(content, list):
                    for block in content:
                        kind = block.get("type")
                        if kind == "reasoning":
                            reasoning_parts.append(block.get("text") or "")
                        elif kind == "tool_call":
                            try:
                                parsed = strict_json(block.get("text") or "")
                            except Exception as exc:
                                failed = f"tool_call-parse {type(exc).__name__}"
                                break
                            normalized = _call_result(parsed["name"], parsed.get("arguments"),
                                                      arguments_span(block.get("text") or ""))
                            calls.append({"id": parsed.get("id"), "name": normalized["name"],
                                          "arguments": normalized["arguments"],
                                          "arguments_raw": normalized["arguments_raw"]})
                        elif kind == "final":
                            final_text = block.get("text")
                        else:
                            failed = f"assistant-block {kind}"
                            break
                else:
                    final_text = plain_text(content)
                if failed:
                    break
                reasoning = " ".join(p for p in reasoning_parts if p).strip() or None
                turns.append(_assistant_turn(reasoning, None, calls, final_text))
            elif role == "tool":
                if isinstance(content, list):
                    parts = [b.get("text") or "" for b in content
                             if isinstance(b, dict) and b.get("type") in ("text", "tool_result")]
                    text = "".join(parts) if parts else ""
                else:
                    text = plain_text(content) or ""
                tool_call_id, obs = None, text
                parsed = _strict_object(text)
                if parsed is not None and "tool_call_id" in parsed:
                    tool_call_id = parsed["tool_call_id"]
                    inner = parsed.get("content")
                    obs = compact(inner) if not isinstance(inner, str) else inner
                turns.append({"role": "tool", "tool_call_id": tool_call_id,
                              "tool_name": None, "content": obs})
            else:
                failed = f"role {role}"
                break
        yield row_index, (None if failed else
                         {"id": None, "tools": tools, "turns": turns, "flags": flags}), failed


def _openai_assistant(message, call_error):
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = function.get("arguments")
        arguments_raw = None
        if isinstance(arguments, str):
            arguments_raw = arguments
            arguments = _strict_object(arguments)
            if arguments is None:
                return None, f"tool_calls-arguments"
        calls.append({"id": call.get("id"), "name": function.get("name"),
                      "arguments": arguments, "arguments_raw": arguments_raw})
    return calls, None


def load_nexus(raw_root: Path, spec: dict):
    for row_index, row in _jsonl_rows(raw_root, spec["files"]):
        raw_tools = row.get("tools") or []
        if isinstance(raw_tools, str):
            raw_tools = strict_json(raw_tools)
        try:
            tools = [pi_tool(s) for s in raw_tools]
        except Exception as exc:
            yield row_index, None, f"tool-parse {type(exc).__name__}"
            continue
        turns, failed = [], None
        for message in row["messages"]:
            role = message["role"]
            if role in ("system", "user"):
                turns.append({"role": role, "content": message.get("content") or ""})
            elif role == "assistant":
                calls, failed = _openai_assistant(message, failed)
                if failed:
                    break
                turns.append(_assistant_turn(message.get("reasoning_content"),
                                             message.get("content") or None, calls))
            elif role == "tool":
                turns.append({"role": "tool", "tool_call_id": message.get("tool_call_id"),
                              "tool_name": message.get("name"),
                              "content": message.get("content") or ""})
            else:
                failed = f"role {role}"
                break
        yield row_index, (None if failed else
                         {"id": None, "tools": tools, "turns": turns, "flags": {}}), failed


def load_nemotron(raw_root: Path, spec: dict):
    for row_index, row in _jsonl_rows(raw_root, spec["files"]):
        flags = {"license": row.get("license"), "used_in": row.get("used_in"),
                 "reasoning_mode": row.get("reasoning")}
        try:
            tools = [pi_tool(s) for s in (row.get("tools") or [])]
        except Exception as exc:
            yield row_index, None, f"tool-parse {type(exc).__name__}"
            continue
        turns, failed = [], None
        for message in row["messages"]:
            role = message["role"]
            if role in ("system", "user"):
                turns.append({"role": role, "content": message.get("content") or ""})
            elif role == "assistant":
                calls, failed = _openai_assistant(message, failed)
                if failed:
                    break
                content = (message.get("content") or "").strip() or None
                turns.append(_assistant_turn(message.get("reasoning_content") or None,
                                             content, calls))
            elif role == "tool":
                turns.append({"role": "tool", "tool_call_id": message.get("tool_call_id"),
                              "tool_name": message.get("name"),
                              "content": message.get("content") or ""})
            else:
                failed = f"role {role}"
                break
        yield row_index, (None if failed else
                         {"id": row.get("uuid"), "tools": tools, "turns": turns,
                          "flags": flags}), failed


LOADERS = {"smoltalk2": load_smoltalk2, "toucan": load_toucan, "toolreason": load_toolreason,
           "nexus": load_nexus, "nemotron": load_nemotron}


def _count_rows(raw_root: Path, files: list[str], kind: str) -> int:
    total = 0
    if kind in ("smoltalk2", "toucan", "toolreason"):
        for name in files:
            total += pq.ParquetFile(raw_root / name).metadata.num_rows
    else:
        with open(raw_root / files[0]) as f:
            total = sum(1 for line in f if line.strip())
    return total


def _parquet_rows(raw_root: Path, files: list[str]):
    index = 0
    for name in files:
        table = pq.read_table(raw_root / name)
        columns = {c: table.column(c).to_pylist() for c in table.column_names}
        for offset in range(table.num_rows):
            yield index, {c: values[offset] for c, values in columns.items()}
            index += 1


def _jsonl_rows(raw_root: Path, files: list[str]):
    index = 0
    for name in files:
        with open(raw_root / name) as f:
            for line in f:
                if line.strip():
                    yield index, strict_json(line)
                    index += 1


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

class Renderer:
    def __init__(self, encoding):
        self.encoding = encoding

    def protocol_block(self, tools: list) -> str:
        return "Protocol:\n" + compact({"profile": PROFILE, "instructions": FRAME,
                                        "tools": tools, "pseudo_actions": PSEUDO_SPECS})

    def frame(self, analysis, commentary, action, arguments_line) -> str:
        return "\n".join(("Analysis: " + compact(analysis),
                          "Commentary: " + compact(commentary),
                          "Think: " + compact(None),
                          "Action: " + action,
                          "Arguments: " + arguments_line))

    @staticmethod
    def valid_frame(frame_text: str, action: str, names: set):
        """Return None when valid, else a reason: malformed|undeclared|bad_finish."""
        try:
            parsed = parse_turn(frame_text)
        except Exception:
            return "malformed"
        if not isinstance(parsed["arguments"], dict):
            return "malformed"
        if action == "finish":
            if set(parsed["arguments"]) == {"message"} and isinstance(parsed["arguments"]["message"], str):
                return None
            return "bad_finish"
        if action not in names:
            return "undeclared"
        return None

    def render(self, record: dict) -> dict:
        sections = []  # (kind, text) with kind in {context, assistant, toolresult}
        names = {t["name"] for t in record["tools"]}
        assistant_frames = []
        sections.append(("context", self.protocol_block(record["tools"])))
        m = {"n_calls": 0, "n_results": 0, "orphan_calls": 0, "unmatched_results": 0,
             "frames": 0, "invalid_frames": 0, "undeclared_action_frames": 0,
             "malformed_frames": 0, "bad_finish_frames": 0,
             "call_frames_with_analysis": 0,
             "finish_frames": 0, "seam_finish_after_result": 0, "final_answer_frames": 0,
             "final_answer_after_result": 0, "direct_answer_only": 0,
             "verbatim_argument_frames": 0, "calls_and_final_anomaly": 0,
             "n_tool_specs": len(record["tools"]), "oversized": 0}
        emitted_calls = []      # emitted call entries awaiting a result, in order
        deferred_calls = []     # calls of a parallel group not yet serialized as frames
        all_call_entries = []   # every call entry, for terminal-orphan accounting
        last_call_entry = None
        matched_calls = 0
        previous_kind = "context"

        def emit_call_frame(call, analysis, commentary):
            nonlocal last_call_entry, previous_kind
            arguments_line = call["arguments_raw"]
            if arguments_line is not None and ("\n" in arguments_line
                                                or _strict_object(arguments_line) is None):
                # The five-line frame admits single-line Arguments values only;
                # multi-line verbatim sources are canonicalized instead.
                m["canonicalized_multiline_argument_frames"] = m.get(
                    "canonicalized_multiline_argument_frames", 0) + 1
                arguments_line = None
            if arguments_line is None:
                arguments_line = compact(call["arguments"])
            else:
                m["verbatim_argument_frames"] += 1
            frame_text = self.frame(analysis, commentary, call["name"], arguments_line)
            m["frames"] += 1
            reason = self.valid_frame(frame_text, call["name"], names)
            if reason:
                m["invalid_frames"] += 1
                m["undeclared_action_frames" if reason == "undeclared" else
                  ("bad_finish_frames" if reason == "bad_finish" else "malformed_frames")] += 1
            if analysis:
                m["call_frames_with_analysis"] += 1
            if call["name"] == "final_answer":
                m["final_answer_frames"] += 1
                if previous_kind == "toolresult":
                    m["final_answer_after_result"] = 1
            sections.append(("assistant", "Assistant:\n" + frame_text))
            assistant_frames.append(frame_text)
            m["n_calls"] += 1
            entry = {"id": call["id"], "name": call["name"], "matched": False}
            emitted_calls.append(entry)
            all_call_entries.append(entry)
            last_call_entry = entry
            previous_kind = "assistant"

        for turn in record["turns"]:
            role = turn["role"]
            if role in ("system", "user"):
                sections.append(("context",
                                 role.title() + ":\n" + compact({"role": role, "content": turn["content"]})))
                last_call_entry = None
                previous_kind = "context"
            elif role == "tool":
                m["n_results"] += 1
                rid = turn.get("tool_call_id")
                if rid is not None:
                    pending = next((c for c in emitted_calls if c["id"] == rid), None)
                else:
                    pending = next((c for c in emitted_calls if c["id"] is None), None)
                if pending is None and deferred_calls:
                    # The result belongs to a call whose frame is not yet serialized
                    # (parallel group); emit that frame first, then its result.
                    if rid is not None:
                        candidates = [i for i, c in enumerate(deferred_calls) if c["id"] == rid]
                    else:
                        candidates = [i for i, c in enumerate(deferred_calls) if c["id"] is None]
                    if candidates:
                        emit_call_frame(deferred_calls.pop(candidates[0]), None, None)
                        pending = emitted_calls[-1]
                if pending is None:
                    m["unmatched_results"] += 1
                else:
                    pending["matched"] = True
                    matched_calls += 1
                    emitted_calls.remove(pending)
                sections.append(("toolresult", "ToolResult:\n" + compact({
                    "role": "toolResult",
                    "toolCallId": turn.get("tool_call_id") or "unkeyed",
                    "toolName": turn.get("tool_name") or (pending or {}).get("name") or "tool",
                    "content": [{"type": "text", "text": turn["content"]}],
                    "isError": False})))
                previous_kind = "toolresult"
                if deferred_calls:
                    # Interleave: after each result, serialize the next deferred call
                    # frame so parallel groups render as frame/result pairs.
                    emit_call_frame(deferred_calls.pop(0), None, None)
            elif role == "assistant":
                calls, text, reasoning = turn["calls"], turn.get("text"), turn.get("reasoning")
                if calls and turn.get("final_text"):
                    m["calls_and_final_anomaly"] += 1
                if calls:
                    emit_call_frame(calls[0], reasoning, text if text else None)
                    deferred_calls.extend(calls[1:])
                else:
                    answer = text if text is not None else turn.get("final_text")
                    if not (answer and answer.strip()):
                        continue
                    if previous_kind == "toolresult":
                        m["seam_finish_after_result"] = 1
                    arguments_line = compact({"message": answer})
                    frame_text = self.frame(reasoning, None, "finish", arguments_line)
                    m["frames"] += 1
                    reason = self.valid_frame(frame_text, "finish", names)
                    if reason:
                        m["invalid_frames"] += 1
                        m["bad_finish_frames"] += 1
                    m["finish_frames"] += 1
                    sections.append(("assistant", "Assistant:\n" + frame_text))
                    assistant_frames.append(frame_text)
                    last_call_entry = None
                    previous_kind = "assistant"
        for call in deferred_calls:
            # trailing calls of parallel groups that never received results
            emit_call_frame(call, None, None)
        m["orphan_calls"] = len(emitted_calls)
        m["orphan_terminal_calls"] = (1 if last_call_entry is not None and not last_call_entry["matched"] else 0)
        m["orphan_mid_calls"] = m["orphan_calls"] - m["orphan_terminal_calls"]
        m["direct_answer_only"] = 1 if (m["n_calls"] and m["n_results"] == 0
                                        and all(e["name"] == "final_answer" for e in all_call_entries)) else 0
        transcript = "\n\n".join(text for _kind, text in sections)
        assistant_text = "\n\n".join(assistant_frames)
        m["tokens"] = len(self.encoding.encode_ordinary(transcript))
        m["assistant_tokens"] = len(self.encoding.encode_ordinary(assistant_text))
        m["context_tokens"] = m["tokens"] - m["assistant_tokens"]
        if m["tokens"] > CONTEXT_WINDOW_TOKENS:
            m["oversized"] = 1
        return {"metrics": m, "transcript": transcript, "sections": sections}


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _median(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def render_source(source_key: str, spec: dict, raw_root: Path, output_root: Path,
                  encoding, sample_size: int, license_receipt: dict) -> Path:
    loader = LOADERS[spec["kind"]]
    started = time.time()
    rows_total = _count_rows(raw_root, spec["files"], spec["kind"])
    rng = random.Random(f"{SEED_PREFIX}:{source_key}")
    n = min(sample_size, rows_total)
    chosen = set(rng.sample(range(rows_total), n))
    renderer = Renderer(encoding)
    out_dir = output_root / source_key
    if out_dir.exists():
        shutil.rmtree(out_dir)  # fresh authority per run; no stale spot-checks
    (out_dir / "spot-checks").mkdir(parents=True, exist_ok=True)
    per_record, failures = [], Counter()
    sampled = 0
    for row_index, record, error in loader(raw_root, spec):
        if error is not None:
            failures[error] += 1
        if row_index not in chosen:
            continue
        sampled += 1
        if error is not None or record is None:
            continue
        rendered = renderer.render(record)
        record_id = record.get("id") or f"{source_key}:{row_index:06d}"
        per_record.append({"id": record_id, "source": source_key, "row_index": row_index,
                           "flags": record["flags"], "metrics": rendered["metrics"],
                           "turns": record["turns"],
                           "sections": [{"kind": kind, "text": text}
                                        for kind, text in rendered["sections"]],
                           "transcript": rendered["transcript"]})
    if sampled != n:
        raise SystemExit(f"{source_key}: sampled {sampled} != expected {n}")
    if not per_record:
        raise SystemExit(f"{source_key}: no parsable sampled records (failures: {dict(failures)})")

    eligible = [r for r in per_record if not r["flags"].get("when2call_excluded")]
    excluded = [r for r in per_record if r["flags"].get("when2call_excluded")]

    def aggregate(items):
        count = len(items)
        if count == 0:
            return {"records": 0}
        metrics = [r["metrics"] for r in items]
        frames = sum(m["frames"] for m in metrics)
        calls = sum(m["n_calls"] for m in metrics)
        return {
            "records": count,
            "seam_ratio": sum(m["seam_finish_after_result"] for m in metrics) / count,
            "records_with_calls": sum(1 for m in metrics if m["n_calls"]) / count,
            "call_result_alternation_valid_ratio": sum(
                1 for m in metrics if m["orphan_calls"] == 0 and m["unmatched_results"] == 0) / count,
            "alternation_valid_no_terminal_orphan_ratio": sum(
                1 for m in metrics if m["orphan_mid_calls"] == 0 and m["unmatched_results"] == 0) / count,
            "final_answer_after_result_ratio": sum(m.get("final_answer_after_result", 0) for m in metrics) / count,
            "direct_answer_only_ratio": sum(m.get("direct_answer_only", 0) for m in metrics) / count,
            "records_with_orphan_calls": sum(1 for m in metrics if m["orphan_calls"]) / count,
            "reasoning_before_call_record_ratio": sum(
                1 for m in metrics if m["call_frames_with_analysis"]) / max(1, sum(1 for m in metrics if m["n_calls"])),
            "call_frames_with_analysis_ratio": (
                sum(m["call_frames_with_analysis"] for m in metrics) / calls) if calls else 0.0,
            "relevance_no_tool_ratio": sum(
                1 for m in metrics if m["n_tool_specs"] and m["n_calls"] == 0 and m["finish_frames"]) / count,
            "finish_frames_per_record": sum(m["finish_frames"] for m in metrics) / count,
            "final_answer_frames_per_record": sum(m["final_answer_frames"] for m in metrics) / count,
            "invalid_frame_ratio": (sum(m["invalid_frames"] for m in metrics) / frames) if frames else 0.0,
            "undeclared_action_frame_ratio": (sum(m.get("undeclared_action_frames", 0) for m in metrics) / frames) if frames else 0.0,
            "malformed_frame_ratio": (sum(m.get("malformed_frames", 0) for m in metrics) / frames) if frames else 0.0,
            "verbatim_argument_frame_ratio": (
                sum(m["verbatim_argument_frames"] for m in metrics) / calls) if calls else 0.0,
            "canonicalized_multiline_argument_frame_ratio": (
                sum(m.get("canonicalized_multiline_argument_frames", 0) for m in metrics) / calls) if calls else 0.0,
            "calls_and_final_anomaly_records": sum(m["calls_and_final_anomaly"] for m in metrics),
            "oversized_record_ratio": sum(m["oversized"] for m in metrics) / count,
            "mean_tokens": _mean([m["tokens"] for m in metrics]),
            "median_tokens": _median([m["tokens"] for m in metrics]),
            "mean_assistant_tokens": _mean([m["assistant_tokens"] for m in metrics]),
            "median_assistant_tokens": _median([m["assistant_tokens"] for m in metrics]),
            "mean_calls_per_record": calls / count,
        }

    eligible_aggregate = aggregate(eligible)
    when2call_rows = 0
    if spec["kind"] == "toolreason":
        when2call_rows = sum(1 for _row_index, row in _parquet_rows(raw_root, spec["files"])
                             if row["source"] == spec["when2call_source_value"])
    eligible_rows = rows_total - when2call_rows - sum(failures.values())
    mean_assistant = eligible_aggregate.get("mean_assistant_tokens", 0.0)
    mean_tokens = eligible_aggregate.get("mean_tokens", 0.0)
    manifest = {
        "schema": SPIKE_SCHEMA, "source": source_key, "repo": spec["repo"],
        "revision": spec["revision"], "kind": spec["kind"],
        "files": [{"name": name, "sha256": sha256_file(raw_root / name),
                   "bytes": (raw_root / name).stat().st_size} for name in spec["files"]],
        "claimed_rows": spec["claimed_rows"], "rows_total": rows_total,
        "when2call_rows_tagged_excluded": when2call_rows,
        "parse_failure_rows": sum(failures.values()),
        "eligible_rows": eligible_rows,
        "sample": {"requested": sample_size, "rendered": len(per_record),
                    "seed": f"{SEED_PREFIX}:{source_key}",
                    "parse_failures_full_pass": {k: v for k, v in failures.items()}},
        "license_claim": spec["license"],
        "license_verified": license_receipt.get("sources", {}).get(source_key, {}),
        "tokenizer": TOKENIZER, "context_window_tokens": CONTEXT_WINDOW_TOKENS,
        "training_eligible": False, "packing_authorized": False,
        "optimizer_updates_authorized": 0,
        "policy": ("rendered per the survey's rendering plan sketch; plain-text answers are "
                   "finish frames; pre-call text is Commentary of the first call frame; "
                   "parallel calls serialize as frame/ToolResult pairs; observations are "
                   "context-only"),
        "cohort_estimates_full_intake": {
            "assistant_target_tokens": int(round(eligible_rows * mean_assistant)),
            "total_tokens": int(round(eligible_rows * mean_tokens)),
            "records": eligible_rows},
        "rendered_at_unix": time.time(), "elapsed_s": round(time.time() - started, 1),
        "checker_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json(out_dir / "manifest.json", manifest)
    with open(out_dir / "records.jsonl", "w") as f:
        for payload in per_record:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
    write_json(out_dir / "metrics.json", {
        "eligible": eligible_aggregate,
        "all_sampled": aggregate(per_record),
        "when2call_excluded_sample": aggregate(excluded) if excluded else {"records": 0},
        "per_record": [{"id": r["id"], "row_index": r["row_index"], "flags": r["flags"],
                        "metrics": {k: v for k, v in r["metrics"].items()}} for r in per_record],
    })
    # spot-check transcripts: 5 spread deterministically across the sample
    spots = sorted({0, len(per_record) // 4, len(per_record) // 2,
                    3 * len(per_record) // 4, len(per_record) - 1})
    for position in spots:
        payload = per_record[position]
        (out_dir / "spot-checks" / f"{payload['id'].replace(':', '_')}.txt").write_text(
            payload["transcript"] + "\n")
    print(f"RENDER {source_key}: rows={rows_total} sampled={n} rendered={len(per_record)} "
          f"seam={eligible_aggregate['seam_ratio']:.3f} "
          f"valid={eligible_aggregate['call_result_alternation_valid_ratio']:.3f} "
          f"assistant_tok_mean={mean_assistant:.0f} parse_failures={sum(failures.values())}", flush=True)
    return out_dir


# --------------------------------------------------------------------------
# download / licenses / overlap
# --------------------------------------------------------------------------

def cmd_download(args):
    raw_root = args.raw_root
    raw_root.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": SPIKE_SCHEMA + ":download", "sources": []}
    for source_key, spec in SOURCES.items():
        for name in spec["files"]:
            target = raw_root / name
            if not target.exists():
                hf_hub_download(repo_id=spec["repo"], revision=spec["revision"],
                                filename=name, repo_type="dataset",
                                local_dir=raw_root)
            receipt["sources"].append({"source": source_key, "repo": spec["repo"],
                                        "revision": spec["revision"], "file": name,
                                        "bytes": target.stat().st_size,
                                        "sha256": sha256_file(target)})
    write_json(args.output_root / "download-receipt.json", receipt)
    print("DOWNLOAD complete:", len(receipt["sources"]), "files")


def cmd_verify_licenses(args):
    api = HfApi()
    receipt = {"schema": SPIKE_SCHEMA + ":licenses", "fetched_unix": time.time(),
               "sources": {}}
    readme_cache = Path("/tmp/e97-tooltalk-readmes")
    readme_cache.mkdir(parents=True, exist_ok=True)
    for source_key, spec in SOURCES.items():
        info = api.dataset_info(spec["repo"], revision=spec["revision"])
        tags = [t.removeprefix("license:") for t in (info.tags or []) if t.startswith("license:")]
        card_license = info.cardData.get("license") if info.cardData else None
        readme = readme_cache / f"{source_key}.md"
        if not readme.exists():
            hf_hub_download(repo_id=spec["repo"], revision=spec["revision"],
                            filename="README.md", repo_type="dataset",
                            local_dir=readme_cache,
                            local_dir_use_symlinks=False)
            downloaded = readme_cache / "README.md"
            if downloaded.exists() and not readme.exists():
                downloaded.replace(readme)
        readme_lines = []
        if readme.exists():
            for line in readme.read_text(errors="replace").splitlines():
                if re.search(r"licen[cs]e", line, re.I) or re.match(r"#.*licen[cs]e", line, re.I):
                    readme_lines.append(line.strip()[:400])
        claim = receipt["sources"][source_key] = {
            "repo": spec["repo"], "resolved_revision": info.sha,
            "revision_matches_pin": info.sha == spec["revision"], "gated": bool(info.gated),
            "api_license_tags": tags, "card_license_field": card_license,
            "readme_license_lines": readme_lines[:12],
            "survey_license_claim": spec["license"],
            "permissive": (bool(tags) and all(t in ("apache-2.0", "cc-by-4.0", "mit") for t in tags))
                          or (isinstance(card_license, str)
                              and card_license in ("apache-2.0", "cc-by-4.0", "mit"))
                          or (isinstance(card_license, list)
                              and bool(card_license)
                              and all(t in ("apache-2.0", "cc-by-4.0", "mit")
                                      for t in card_license)),
        }
        print(source_key, "tags=", tags, "card=", claim["card_license_field"],
              "gated", claim["gated"], "permissive", claim["permissive"])
    write_json(args.output_root / "license-receipt.json", receipt)
    print("LICENSE receipt written")


_NONFINITE = re.compile(r"\b(?:Infinity|-Infinity|NaN)\b")


def _sanitize_observation(text: str) -> str:
    """Make an observation safe for the protected-overlap scalar extractor.
    Some Toucan MCP observations embed non-JSON-compliant float literals
    (bare Infinity/NaN, or exponents like 1e999 that parse to inf); the
    extractor's json round-trip rejects them. Only rewrites when a non-finite
    value is actually present, so exact scalar text is otherwise preserved."""
    import math
    def _nonfinite(o):
        if isinstance(o, float):
            return not math.isfinite(o)
        if isinstance(o, dict):
            return any(_nonfinite(v) for v in o.values())
        if isinstance(o, list):
            return any(_nonfinite(v) for v in o)
        return False
    def _fix(o):
        if isinstance(o, float) and not math.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: _fix(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_fix(v) for v in o]
        return o
    try:
        parsed = json.loads(text)
    except Exception:
        return _NONFINITE.sub("null", text)
    if not isinstance(parsed, (dict, list)):
        return text
    if _nonfinite(parsed):
        return json.dumps(_fix(parsed))
    return text


def _overlap_records(payloads):
    from ndm.e97_protected_overlap import extract_exact_scalars
    out = []
    for payload in payloads:
        observations = [_sanitize_observation(t["content"])[:65536]
                        for t in payload.get("turns", []) if t.get("role") == "tool"]
        observations = [obs for obs in observations if obs]
        first_user = next((t["content"] for t in payload.get("turns", [])
                           if t.get("role") == "user"), "")
        out.append({"task_id": payload["id"], "family_id": payload["source"],
                    "repository": "public/" + SOURCES[payload["source"]]["repo"],
                    "prompt_template": first_user[:65536],
                    "fixture_files": [{"path": f"observation-{i}", "content": obs}
                                      for i, obs in enumerate(observations)],
                    "exact_scalars": sorted(extract_exact_scalars(observations))})
    return out


def cmd_overlap(args):
    from ndm.e97_protected_overlap import (_domains, load_protected_panel,
                                           normalize_content)
    protected, panel_ids = [], []
    for manifest_path, records_path in PROTECTED_PANELS:
        manifest_sha, records_sha, items = load_protected_panel(Path(manifest_path), Path(records_path))
        protected.extend(items)
        panel_ids.append((manifest_sha, records_sha))
    protected_domains = _domains(protected)
    receipt = {"schema": SPIKE_SCHEMA + ":overlap", "status": "pass",
                "policy": ("exact-and-significant-entity-v1; structural family/path-template "
                           "reuse and trivial sub-8-byte numeric scalars are reported but "
                           "are not entity collisions"),
                "protected_panels": [list(pair) for pair in panel_ids],
                "sources": {}}
    for source_dir in sorted(p for p in args.output_root.iterdir() if p.is_dir()):
        records_path = source_dir / "records.jsonl"
        if not records_path.exists():
            continue
        payloads = [json.loads(line) for line in records_path.read_text().splitlines()]
        candidate = _domains(_overlap_records(payloads))
        counts = {field: len(candidate[field] & protected_domains[field])
                  for field in sorted(protected_domains)}
        significant_scalars = ({x for x in candidate["exact_scalars"] if len(x.encode()) >= 8}
                               & {x for x in protected_domains["exact_scalars"] if len(x.encode()) >= 8})
        significant_content = ({normalize_content(x) for x in candidate["fixture_contents_full"]
                                if len(x.encode()) >= 16}
                               & {normalize_content(x) for x in protected_domains["fixture_contents_full"]
                                  if len(x.encode()) >= 16})
        trivial = sorted(x for x in candidate["exact_scalars"] & protected_domains["exact_scalars"]
                         if len(x.encode()) < 8)
        blocking = {k: v for k, v in counts.items() if k != "exact_scalars" and v}
        blocking["significant_exact_scalars"] = len(significant_scalars)
        blocking["significant_normalized_contents"] = len(significant_content)
        entry = {"records": len(payloads), "entity_collision_counts": counts,
                 "significant_exact_scalars": len(significant_scalars),
                 "significant_normalized_contents": len(significant_content),
                 "trivial_sub8byte_scalars": trivial,
                 "status": "pass" if not any(blocking.values()) else "fail"}
        receipt["sources"][source_dir.name] = entry
        if entry["status"] != "pass":
            receipt["status"] = "fail"
        print("OVERLAP", source_dir.name, entry["status"],
              {k: v for k, v in counts.items() if v},
              "sig_scalars", len(significant_scalars),
              "sig_content", len(significant_content))
    write_json(args.output_root / "overlap-audit.json", receipt)
    if receipt["status"] != "pass":
        raise SystemExit("protected overlap failed")
    print("OVERLAP_AUDIT_PASS")


def cmd_render(args):
    encoding = tiktoken.get_encoding(TOKENIZER)
    license_path = args.output_root / "license-receipt.json"
    license_receipt = json.loads(license_path.read_text()) if license_path.exists() else {}
    args.output_root.mkdir(parents=True, exist_ok=True)
    sources = args.sources or list(SOURCES)
    unknown = [s for s in sources if s not in SOURCES]
    if unknown:
        raise SystemExit(f"unknown sources: {unknown}")
    for source_key in sources:
        # an explicit --sample overrides every per-source default (the spike's
        # small Nemotron default stays the fallback when --sample is absent)
        size = args.sample if args.sample is not None else SAMPLE_SIZES.get(source_key, DEFAULT_SAMPLE)
        render_source(source_key, SOURCES[source_key], args.raw_root, args.output_root,
                      encoding, size, license_receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = lambda p: (p.add_argument("--raw-root", type=Path, required=True),
                        p.add_argument("--output-root", type=Path, required=True))
    p_download = sub.add_parser("download"); common(p_download)
    p_licenses = sub.add_parser("verify-licenses")
    p_licenses.add_argument("--output-root", type=Path, required=True)
    p_render = sub.add_parser("render"); common(p_render)
    p_render.add_argument("--sample", type=int, default=None,
                          help="explicit per-source sample size; overrides the spike's per-source defaults when given")
    p_render.add_argument("--sources", nargs="*", default=None)
    p_overlap = sub.add_parser("overlap")
    p_overlap.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    {"download": cmd_download, "verify-licenses": lambda a: cmd_verify_licenses(a),
     "render": cmd_render, "overlap": cmd_overlap}[args.command](args)


if __name__ == "__main__":
    main()
