"""Deterministic read-observe acquisition controller for an E97 OpenAI service.

The controller, rather than a Pi extension hook, owns the serial turn loop.  It
keeps each executor result in the audit receipt and separately chooses the
bounded observation placed in the next OpenAI request.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import multiprocessing
import os
import stat
from pathlib import Path, PurePosixPath
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ndm.e97_agent_protocol import RS, AgentProtocolError, parse_agent_turn, serialize_pi_messages, validate_generated_tool
from ndm.e97_onpolicy_records import (
    ActionProgressReceipt,
    NoProgressDetector,
    action_fingerprint,
    canonical_json,
    progress_fingerprint,
    sha256_json,
    sha256_text,
)


ACQUISITION_CONTROLLER_SCHEMA = "emender-e97-dedicated-acquisition-controller-v1"
SERVICE_ATTESTATION_SCHEMA = "emender-e97-agent-service-attestation-v1"
_DIGEST_LENGTH = 64


def _require_digest(value: Any, name: str) -> str:
    if (not isinstance(value, str) or len(value) != _DIGEST_LENGTH
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def validate_service_attestation(value: Any) -> dict[str, Any]:
    """Normalize the startup-only identity returned by an external service."""
    required = {
        "schema", "checkpoint_path", "checkpoint_sha256", "args_json_sha256", "config_sha256",
        "weight_mode", "tokenizer", "model_id", "server_build_sha256", "controller_build_sha256",
        "device", "dtype", "use_triton", "ingest_mode", "runtime_image_path", "runtime_image_sha256", "tool_schema_sha256",
        "system_prompt_override_sha256", "runtime_identity_schema", "max_output_tokens", "max_sessions",
        "python_implementation", "python_version", "torch_version", "cuda_runtime", "cuda_available",
        "platform", "machine",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("service attestation fields are invalid")
    if value["schema"] != SERVICE_ATTESTATION_SCHEMA:
        raise ValueError("unsupported service attestation schema")
    normalized = dict(value)
    for key in ("checkpoint_sha256", "args_json_sha256", "config_sha256", "server_build_sha256",
                "system_prompt_override_sha256", "controller_build_sha256", "runtime_image_sha256",
                "tool_schema_sha256"):
        _require_digest(normalized[key], f"attestation {key}")
    if not isinstance(normalized["checkpoint_path"], str) or not normalized["checkpoint_path"]:
        raise ValueError("attestation checkpoint_path is invalid")
    if not isinstance(normalized["runtime_image_path"], str) or not normalized["runtime_image_path"]:
        raise ValueError("attestation runtime_image_path is invalid")
    if not isinstance(normalized["model_id"], str) or not normalized["model_id"]:
        raise ValueError("attestation model_id is invalid")
    if not isinstance(normalized["device"], str) or not isinstance(normalized["dtype"], str):
        raise ValueError("attestation device/dtype is invalid")
    if not isinstance(normalized["use_triton"], bool) or normalized["ingest_mode"] not in {"tokenwise", "segment"}:
        raise ValueError("attestation decode runtime is invalid")
    if normalized["weight_mode"] not in {"saved", "train"}:
        raise ValueError("attestation weight_mode is invalid")
    if not isinstance(normalized["tokenizer"], str) or not normalized["tokenizer"]:
        raise ValueError("attestation tokenizer is invalid")
    if normalized["runtime_identity_schema"] != "emender-e97-runtime-identity-v1":
        raise ValueError("attestation runtime identity schema is invalid")
    if (isinstance(normalized["max_output_tokens"], bool) or not isinstance(normalized["max_output_tokens"], int)
            or isinstance(normalized["max_sessions"], bool) or not isinstance(normalized["max_sessions"], int)
            or normalized["max_output_tokens"] <= 0 or normalized["max_sessions"] <= 0):
        raise ValueError("attestation service limits are invalid")
    for key in ("python_implementation", "python_version", "torch_version", "cuda_runtime", "platform", "machine"):
        if not isinstance(normalized[key], str):
            raise ValueError(f"attestation {key} is invalid")
    if not isinstance(normalized["cuda_available"], bool):
        raise ValueError("attestation cuda availability is invalid")
    return json.loads(canonical_json(normalized))


READ_OBSERVE_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List a bounded workspace directory without following symlink directories.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path", "depth", "limit"],
                "properties": {
                    "path": {"type": "string"},
                    "depth": {"type": "integer", "minimum": 0, "maximum": 16},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a bounded, line-numbered UTF-8 text slice from the workspace.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path", "offset", "limit"],
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 1},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 2000},
                },
            },
        },
    },
)


class CompletionClient(Protocol):
    """A replaceable deadline-aware OpenAI-compatible completion boundary."""

    def complete(
        self, request: Mapping[str, Any], *, deadline: float,
    ) -> Mapping[str, Any]: ...


class OpenAICompletionClient:
    """Small standard-library client for one pinned OpenAI Chat Completions URL.

    It deliberately exposes no tool execution or retry policy; those remain in
    the controller.  Tests can supply a ``CompletionClient`` fake instead.
    """

    def __init__(
        self, completion_url: str, *, api_key: str | None = None,
        max_response_bytes: int = 1 << 20,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(completion_url, str) or not completion_url.startswith(("http://", "https://")):
            raise ValueError("completion_url must be an http(s) URL")
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        self.completion_url = completion_url
        self.api_key = api_key
        self.max_response_bytes = max_response_bytes
        self.clock = clock

    def complete(self, request: Mapping[str, Any], *, deadline: float) -> Mapping[str, Any]:
        timeout_seconds = deadline - self.clock()
        if timeout_seconds <= 0:
            raise AgentProtocolError("completion deadline expired")
        body = canonical_json(dict(request)).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            with urlopen(Request(self.completion_url, body, headers, method="POST"), timeout=timeout_seconds) as response:
                length = response.headers.get("Content-Length")
                if length is not None and int(length) > self.max_response_bytes:
                    raise AgentProtocolError("OpenAI completion response exceeds byte limit")
                chunks: list[bytes] = []
                remaining = self.max_response_bytes
                while True:
                    if self.clock() >= deadline:
                        raise AgentProtocolError("OpenAI completion deadline expired")
                    chunk = response.read(min(64 << 10, remaining + 1))
                    if not chunk:
                        break
                    if len(chunk) > remaining:
                        raise AgentProtocolError("OpenAI completion response exceeds byte limit")
                    chunks.append(chunk)
                    remaining -= len(chunk)
                payload = json.loads(b"".join(chunks))
        except AgentProtocolError:
            raise
        except (HTTPError, OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise AgentProtocolError("OpenAI completion request failed") from exc
        if not isinstance(payload, Mapping):
            raise AgentProtocolError("OpenAI completion response must be an object")
        return payload


@dataclass(frozen=True)
class ToolExecution:
    """Authentic executor output and optional post-access source ledger entries."""

    raw_observation: Any
    effective_observation: str
    is_error: bool = False
    source_ledger: Mapping[str, Any] = field(default_factory=dict)


class ToolExecutor(Protocol):
    """Typed serial tool boundary required by :class:`AcquisitionController`."""

    def execute(self, tool_name: str, arguments: Mapping[str, Any]) -> ToolExecution: ...
    def workspace_state(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ActionReceipt:
    sequence: int
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    arguments_json: str
    raw_observation: Any
    raw_observation_sha256: str
    effective_observation: str
    observation_sha256: str
    is_error: bool
    workspace_state: Mapping[str, Any]
    source_ledger: Mapping[str, Any]
    acquired_observations: tuple[Mapping[str, Any], ...]
    action_fingerprint: str
    progress_fingerprint: str
    completion_tokens: int
    decision: str


@dataclass(frozen=True)
class TerminalReceipt:
    status: str
    turns: int
    elapsed_seconds: float
    messages: tuple[dict[str, Any], ...]
    actions: tuple[ActionReceipt, ...]
    metadata: Mapping[str, Any]
    error: str | None = None


class WorkspaceToolExecutor:
    """Contained, deterministic implementations of the initial read-observe tools."""

    def __init__(
        self,
        root: Path | str,
        *,
        max_output_bytes: int = 16_384,
        max_list_depth: int = 16,
        max_list_entries: int = 1_000,
        max_read_lines: int = 2_000,
        max_read_bytes: int = 1 << 20,
        max_read_line_bytes: int | None = None,
        max_tree_entries: int = 10_000,
        max_tree_bytes: int = 16 << 20,
    ) -> None:
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("workspace root must be a directory")
        if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
            raise RuntimeError("descriptor-safe workspace tools require Linux openat flags")
        self._root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        for value, name in ((max_list_entries, "max_list_entries"), (max_read_lines, "max_read_lines"),
                            (max_read_bytes, "max_read_bytes"), (max_tree_entries, "max_tree_entries"),
                            (max_tree_bytes, "max_tree_bytes")):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if max_output_bytes < 128:
            raise ValueError("max_output_bytes must leave room for a compact error")
        if max_list_depth < 0:
            raise ValueError("max_list_depth must be non-negative")
        self.max_output_bytes = max_output_bytes
        self.max_list_depth = max_list_depth
        self.max_list_entries = max_list_entries
        self.max_read_lines = max_read_lines
        self.max_read_bytes = max_read_bytes
        self.max_read_line_bytes = max_read_line_bytes or max_output_bytes
        if self.max_read_line_bytes <= 0:
            raise ValueError("max_read_line_bytes must be positive")
        self.max_tree_entries = max_tree_entries
        self.max_tree_bytes = max_tree_bytes

    def close(self) -> None:
        if getattr(self, "_root_fd", -1) >= 0:
            os.close(self._root_fd)
            self._root_fd = -1

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except OSError:
            pass

    def _relative_path(self, value: Any) -> tuple[tuple[str, ...], str]:
        if (not isinstance(value, str) or not value or "\x00" in value
                or len(value.encode("utf-8")) > 4096):
            raise ValueError("path must be a non-empty bounded relative string")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("path escapes workspace")
        if not path.parts and path.as_posix() != ".":
            raise ValueError("path escapes workspace")
        return tuple(path.parts), path.as_posix()

    def _open_dir(self, parts: Sequence[str]) -> int:
        fd = os.dup(self._root_fd)
        try:
            for part in parts:
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
                os.close(fd)
                fd = next_fd
            return fd
        except BaseException:
            os.close(fd)
            raise

    def _open_file(self, parts: Sequence[str]) -> int:
        if not parts:
            raise ValueError("path must name a file")
        parent = self._open_dir(parts[:-1])
        try:
            return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
        finally:
            os.close(parent)

    def _bounded_text(self, value: Mapping[str, Any]) -> str:
        """Return compact canonical JSON, dropping tail entries before overflow."""
        rendered = canonical_json(value)
        if len(rendered.encode("utf-8")) <= self.max_output_bytes:
            return rendered
        mutable = dict(value)
        entries = mutable.get("entries", mutable.get("lines"))
        if isinstance(entries, list):
            key = "entries" if "entries" in mutable else "lines"
            bounded_entries = list(entries)
            while bounded_entries:
                bounded_entries.pop()
                mutable[key] = bounded_entries
                mutable["truncated"] = True
                rendered = canonical_json(mutable)
                if len(rendered.encode("utf-8")) <= self.max_output_bytes:
                    return rendered
        return canonical_json({"ok": False, "error": {"code": "output_limit", "message": "tool output exceeded limit"}})

    def _error(self, tool: str, code: str, message: str) -> ToolExecution:
        raw = {"ok": False, "tool": tool, "error": {"code": code, "message": message}}
        return ToolExecution(raw, self._bounded_text(raw), True)

    @staticmethod
    def _integer(arguments: Mapping[str, Any], name: str, *, minimum: int, maximum: int) -> int:
        value = arguments.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f"{name} is outside its bounded range")
        return value

    def execute(self, tool_name: str, arguments: Mapping[str, Any]) -> ToolExecution:
        if not isinstance(arguments, Mapping):
            return self._error(str(tool_name), "invalid_arguments", "arguments must be an object")
        try:
            if tool_name == "list_files":
                return self._list_files(arguments)
            if tool_name == "read":
                return self._read(arguments)
            return self._error(str(tool_name), "unknown_tool", "tool is not registered")
        except ValueError as exc:
            return self._error(str(tool_name), "invalid_arguments", str(exc))
        except (OSError, UnicodeError) as exc:
            return self._error(str(tool_name), "io_error", str(exc))

    def _list_files(self, arguments: Mapping[str, Any]) -> ToolExecution:
        if set(arguments) != {"path", "depth", "limit"}:
            raise ValueError("list_files requires exactly path, depth, and limit")
        parts, display_path = self._relative_path(arguments["path"])
        depth = self._integer(arguments, "depth", minimum=0, maximum=self.max_list_depth)
        limit = self._integer(arguments, "limit", minimum=1, maximum=self.max_list_entries)
        entries: list[dict[str, str]] = []
        pending: list[tuple[int, str, int]] = [(self._open_dir(parts), display_path, 0)]
        truncated = False
        try:
            while pending:
                current, relative, current_depth = pending.pop(0)
                try:
                    for name in sorted(os.listdir(current)):
                        child_relative = f"{relative}/{name}" if relative != "." else name
                        info = os.stat(name, dir_fd=current, follow_symlinks=False)
                        if stat.S_ISLNK(info.st_mode):
                            kind = "symlink"
                        elif stat.S_ISDIR(info.st_mode):
                            kind = "directory"
                        else:
                            kind = "file"
                        if len(entries) >= limit:
                            truncated = True
                            break
                        entries.append({"path": child_relative, "type": kind})
                        if kind == "directory" and current_depth < depth:
                            pending.append((os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=current), child_relative, current_depth + 1))
                finally:
                    os.close(current)
                if truncated:
                    break
        finally:
            for current, _, _ in pending:
                os.close(current)
        raw = {"ok": True, "tool": "list_files", "path": display_path, "entries": entries, "truncated": truncated}
        ledger = {display_path: sha256_json(entries)}
        return ToolExecution(raw, self._bounded_text(raw), False, ledger)

    def _read(self, arguments: Mapping[str, Any]) -> ToolExecution:
        if set(arguments) != {"path", "offset", "limit"}:
            raise ValueError("read requires exactly path, offset, and limit")
        parts, display_path = self._relative_path(arguments["path"])
        offset = self._integer(arguments, "offset", minimum=1, maximum=2**31 - 1)
        limit = self._integer(arguments, "limit", minimum=1, maximum=self.max_read_lines)
        fd = self._open_file(parts)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("path is not a regular file")
            if info.st_size > self.max_read_bytes:
                return self._error("read", "input_limit", "file exceeds readable input limit")
            digest = hashlib.sha256()
            rendered_lines: list[dict[str, Any]] = []
            line_number = 1
            truncated = False
            pending = b""
            bytes_seen = 0

            def consume(encoded: bytes) -> None:
                nonlocal line_number, truncated
                if len(encoded) > self.max_read_line_bytes:
                    raise ValueError("line exceeds readable input limit")
                text = encoded.decode("utf-8")
                if text.endswith("\r"):
                    text = text[:-1]
                if offset <= line_number < offset + limit:
                    rendered_lines.append({"line": line_number, "text": text})
                elif line_number >= offset + limit:
                    truncated = True
                line_number += 1

            while chunk := os.read(fd, 64 << 10):
                bytes_seen += len(chunk)
                if bytes_seen > self.max_read_bytes:
                    return self._error("read", "input_limit", "file exceeds readable input limit")
                digest.update(chunk)
                pending += chunk
                split = pending.split(b"\n")
                pending = split.pop()
                for encoded in split:
                    consume(encoded)
                if len(pending) > self.max_read_line_bytes:
                    raise ValueError("line exceeds readable input limit")
            if pending:
                consume(pending)
        finally:
            os.close(fd)
        raw = {"ok": True, "tool": "read", "path": display_path, "offset": offset,
               "lines": rendered_lines, "truncated": truncated}
        return ToolExecution(raw, self._bounded_text(raw), False, {display_path: digest.hexdigest()})

    def workspace_state(self) -> Mapping[str, Any]:
        """Descriptor-safe recursive tree hash; every file is opened once by fd."""
        rows: list[dict[str, Any]] = []
        bytes_seen = 0

        def visit(directory: int, prefix: str) -> None:
            nonlocal bytes_seen
            for name in sorted(os.listdir(directory)):
                relative = f"{prefix}/{name}" if prefix else name
                info = os.stat(name, dir_fd=directory, follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    rows.append({"path": relative, "type": "symlink", "target": os.readlink(name, dir_fd=directory)})
                elif stat.S_ISDIR(info.st_mode):
                    rows.append({"path": relative, "type": "directory"})
                    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
                    try:
                        visit(child, relative)
                    finally:
                        os.close(child)
                elif stat.S_ISREG(info.st_mode):
                    child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
                    try:
                        opened = os.fstat(child)
                        if not stat.S_ISREG(opened.st_mode):
                            raise ValueError("workspace file changed type")
                        digest = hashlib.sha256()
                        while chunk := os.read(child, 64 << 10):
                            bytes_seen += len(chunk)
                            if bytes_seen > self.max_tree_bytes:
                                raise ValueError("workspace tree exceeds byte limit")
                            digest.update(chunk)
                        rows.append({"path": relative, "type": "file", "sha256": digest.hexdigest()})
                    finally:
                        os.close(child)
                else:
                    rows.append({"path": relative, "type": "other", "mode": info.st_mode})
                if len(rows) > self.max_tree_entries:
                    raise ValueError("workspace tree exceeds entry limit")

        root = os.dup(self._root_fd)
        try:
            visit(root, "")
        finally:
            os.close(root)
        return {"tree_sha256": sha256_json(rows), "entries": len(rows)}


_MAX_WORKER_RESULT_BYTES = 1 << 20
_MAX_WORKER_ERROR_BYTES = 4096


def _bounded_json(value: Any, limit: int = _MAX_WORKER_RESULT_BYTES) -> bytes:
    pieces: list[str] = []
    size = 0
    for piece in json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).iterencode(value):
        encoded = piece.encode("utf-8")
        size += len(encoded)
        if size > limit:
            raise ValueError("IPC payload exceeds byte limit")
        pieces.append(piece)
    return "".join(pieces).encode("utf-8")


def _worker_error(exc: BaseException) -> bytes:
    text = f"{type(exc).__name__}: {exc}".encode("utf-8")[:_MAX_WORKER_ERROR_BYTES].decode("utf-8", "ignore")
    return _bounded_json({"ok": False, "error": text}, _MAX_WORKER_ERROR_BYTES + 64)


def _worker_loop(kind: str, target: Any, connection: Any) -> None:
    try:
        while True:
            command = json.loads(connection.recv_bytes(_MAX_WORKER_RESULT_BYTES))
            if command == {"op": "close"}:
                return
            try:
                if kind == "completion":
                    result = target.complete(command["request"], deadline=float(command["deadline"]))
                else:
                    execution = target.execute(command["tool_name"], command["arguments"])
                    if not isinstance(execution, ToolExecution):
                        raise TypeError("tool executor must return ToolExecution")
                    result = {"execution": {"raw_observation": execution.raw_observation, "effective_observation": execution.effective_observation, "is_error": execution.is_error, "source_ledger": dict(execution.source_ledger)}, "workspace_state": target.workspace_state()}
                connection.send_bytes(_bounded_json({"ok": True, "value": result}))
            except BaseException as exc:
                connection.send_bytes(_worker_error(exc))
    finally:
        connection.close()


class _ForkWorker:
    def __init__(self, kind: str, target: Any) -> None:
        try:
            context = multiprocessing.get_context("fork")
        except ValueError as exc:  # pragma: no cover
            raise RuntimeError("controller requires Linux fork worker isolation") from exc
        parent, child = context.Pipe(duplex=True)
        self.connection = parent
        self.process = context.Process(target=_worker_loop, args=(kind, target, child))
        self.process.start()
        child.close()

    def call(self, command: Mapping[str, Any], *, deadline: float, clock: Callable[[], float]) -> Any:
        request = _bounded_json(dict(command))
        self.connection.send_bytes(request)
        remaining = deadline - clock()
        if remaining <= 0 or not self.connection.poll(remaining):
            raise TimeoutError("controller deadline expired")
        payload = self.connection.recv_bytes(_MAX_WORKER_RESULT_BYTES + 1)
        if len(payload) > _MAX_WORKER_RESULT_BYTES:
            raise RuntimeError("worker result exceeds byte limit")
        result = json.loads(payload)
        if result.get("ok") is not True:
            raise RuntimeError(str(result.get("error", "worker failure")))
        return result["value"]

    def close(self) -> None:
        try:
            if self.process.is_alive():
                try:
                    self.connection.send_bytes(b'{"op":"close"}')
                    # A controller deadline has already expired; prefer prompt
                    # process-tree reaping over a second long graceful wait.
                    self.process.join(timeout=0.02)
                except (BrokenPipeError, OSError):
                    pass
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=0.1)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
        finally:
            self.connection.close()
            self.process.close()


class AcquisitionController:
    """Own a bounded, serial E97 turn loop with deterministic cycle receipts."""

    def __init__(
        self,
        completion_client: CompletionClient,
        tool_executor: ToolExecutor,
        *,
        checkpoint_sha256: str,
        expected_service_attestation: Mapping[str, Any],
        controller_build_sha256: str,
        model_id: str = "e97-dense-agent",
        max_turns: int = 12,
        max_seconds: float = 300,
        max_completion_tokens: int = 512,
        max_observation_bytes: int = 16_384,
        sealed_limits: Mapping[str, int] | None = None,
        tools: Sequence[Mapping[str, Any]] = READ_OBSERVE_TOOLS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.checkpoint_sha256 = _require_digest(checkpoint_sha256, "checkpoint_sha256")
        self.controller_build_sha256 = _require_digest(controller_build_sha256, "controller_build_sha256")
        self.expected_service_attestation = validate_service_attestation(expected_service_attestation)
        if self.expected_service_attestation["checkpoint_sha256"] != self.checkpoint_sha256:
            raise ValueError("service attestation checkpoint does not match controller checkpoint")
        if self.expected_service_attestation["controller_build_sha256"] != self.controller_build_sha256:
            raise ValueError("service attestation controller build does not match controller build")
        if self.expected_service_attestation["system_prompt_override_sha256"] != sha256_text(""):
            raise ValueError("external service must not override the controller system prompt")
        if max_turns <= 0 or max_seconds <= 0 or max_completion_tokens <= 0 or max_observation_bytes <= 0:
            raise ValueError("turn, time, completion-token, and observation limits must be positive")
        self.completion_client = completion_client
        self.tool_executor = tool_executor
        self.model_id = model_id
        if self.expected_service_attestation["model_id"] != self.model_id:
            raise ValueError("service attestation model does not match controller model")
        self.max_turns = max_turns
        self.max_seconds = max_seconds
        self.max_completion_tokens = max_completion_tokens
        self.max_observation_bytes = max_observation_bytes
        if sealed_limits is not None:
            expected_limits = {"turns", "seconds", "completion_tokens", "output_bytes", "disk_bytes", "processes"}
            if (not isinstance(sealed_limits, Mapping) or set(sealed_limits) != expected_limits
                    or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0
                           for item in sealed_limits.values())):
                raise ValueError("sealed task limits are invalid")
            if (sealed_limits["turns"] != max_turns or sealed_limits["seconds"] != int(max_seconds)
                    or sealed_limits["completion_tokens"] != max_completion_tokens
                    or sealed_limits["output_bytes"] != max_observation_bytes):
                raise ValueError("controller limits do not exactly match sealed task limits")
            self.sealed_limits: dict[str, int] | None = dict(sealed_limits)
        else:
            self.sealed_limits = None
        self.tools = tuple(dict(tool) for tool in tools)
        if self.expected_service_attestation["tool_schema_sha256"] != sha256_json(self.tools):
            raise ValueError("service attestation tool schema does not match controller tools")
        self.clock = clock
        self._completion_worker: _ForkWorker | None = None
        self._tool_worker: _ForkWorker | None = None
        self._completion_usage: list[dict[str, int]] = []
        self._consumed = False

    @classmethod
    def for_task_bundle(
        cls,
        completion_client: CompletionClient,
        tool_executor: ToolExecutor,
        *,
        bundle: Mapping[str, Any],
        checkpoint_sha256: str,
        expected_service_attestation: Mapping[str, Any],
        controller_build_sha256: str,
        model_id: str = "e97-dense-agent",
        clock: Callable[[], float] = time.monotonic,
    ) -> "AcquisitionController":
        """Construct a controller whose executable limits come only from a bundle."""

        limits = bundle.get("limits") if isinstance(bundle, Mapping) else None
        if not isinstance(limits, Mapping):
            raise ValueError("sealed task bundle limits are required")
        return cls(
            completion_client,
            tool_executor,
            checkpoint_sha256=checkpoint_sha256,
            expected_service_attestation=expected_service_attestation,
            controller_build_sha256=controller_build_sha256,
            model_id=model_id,
            max_turns=limits.get("turns", 0),
            max_seconds=limits.get("seconds", 0),
            max_completion_tokens=limits.get("completion_tokens", 0),
            max_observation_bytes=limits.get("output_bytes", 0),
            sealed_limits=limits,
            clock=clock,
        )

    def _metadata(
        self,
        system_prompt: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        turns: int,
        actions: int,
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        metadata = {
            "schema": ACQUISITION_CONTROLLER_SCHEMA,
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_id": self.model_id,
            "system_prompt_sha256": sha256_text(system_prompt),
            "tool_schema_sha256": sha256_json(self.tools),
            "controller_build_sha256": self.controller_build_sha256,
            "service_attestation": self.expected_service_attestation,
            "serialized_messages_sha256": sha256_text(
                serialize_pi_messages(messages, append_assistant_header=False)),
            "configured_limits": dict(self.sealed_limits) if self.sealed_limits is not None else {
                "turns": self.max_turns,
                "seconds": int(self.max_seconds),
                "completion_tokens": self.max_completion_tokens,
                "output_bytes": self.max_observation_bytes,
            },
            "turn_count": turns,
            "action_count": actions,
            "elapsed_seconds": elapsed_seconds,
            "completion_usage": [dict(item) for item in self._completion_usage],
            "completion_tokens": sum(item["completion_tokens"] for item in self._completion_usage),
        }
        return metadata

    def _terminal(self, status: str, started: float, messages: list[dict[str, Any]], actions: list[ActionReceipt], system_prompt: str, error: str | None = None) -> TerminalReceipt:
        for worker in (self._completion_worker, self._tool_worker):
            if worker is not None:
                worker.close()
        self._completion_worker = self._tool_worker = None
        turns = len([message for message in messages if message["role"] == "assistant"])
        elapsed = self.clock() - started
        return TerminalReceipt(
            status,
            turns,
            elapsed,
            tuple(messages),
            tuple(actions),
            self._metadata(system_prompt, messages, turns=turns, actions=len(actions), elapsed_seconds=elapsed),
            error,
        )

    def _bounded_observation(self, text: str) -> str:
        if not isinstance(text, str):
            return canonical_json({"ok": False, "error": {"code": "invalid_observation", "message": "executor returned non-text observation"}})
        encoded = text.encode("utf-8")
        if len(encoded) <= self.max_observation_bytes:
            return text
        # Do not slice arbitrary JSON/text and thereby fabricate a partial result.
        return canonical_json({"ok": False, "error": {"code": "output_limit", "message": "tool output exceeded limit"}})

    def _completion_tokens(self, response: Mapping[str, Any]) -> int:
        """Require the service's bounded token accounting for every accepted turn."""

        usage = response.get("usage")
        tokens = usage.get("completion_tokens") if isinstance(usage, Mapping) else None
        if isinstance(tokens, bool) or not isinstance(tokens, int) or not 1 <= tokens <= self.max_completion_tokens:
            raise AgentProtocolError("completion response lacks bounded usage.completion_tokens")
        return tokens

    def _validate_assistant_body_limit(self, message: Mapping[str, Any]) -> None:
        """Apply a deterministic UTF-8 ceiling when tokenizer details are unavailable."""

        try:
            body = serialize_pi_messages([message], append_assistant_header=False).removeprefix("Assistant:\n")
        except AgentProtocolError as exc:
            raise AgentProtocolError("completion assistant serialization is invalid") from exc
        if len(body.encode("utf-8")) > self.max_completion_tokens * 8:
            raise AgentProtocolError("completion assistant body exceeds deterministic token byte ceiling")

    @staticmethod
    def _assistant(response: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
        try:
            choice = response["choices"][0]
            raw_message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AgentProtocolError("completion lacks one OpenAI assistant message") from exc
        if not isinstance(raw_message, Mapping) or raw_message.get("role") != "assistant":
            raise AgentProtocolError("completion message must be an assistant message")
        if raw_message.get("tool_calls"):
            if raw_message.get("content") is not None:
                raise AgentProtocolError("completion tool call must have null content")
            tool_calls = raw_message["tool_calls"]
            if not isinstance(tool_calls, list) or not tool_calls:
                raise AgentProtocolError("completion tool_calls must be a non-empty list")
            for call in tool_calls:
                function = call.get("function") if isinstance(call, Mapping) else None
                if not isinstance(function, Mapping) or not isinstance(function.get("arguments"), str):
                    raise AgentProtocolError("completion function.arguments must be a string")
            message = {"role": "assistant", "content": None, "tool_calls": list(tool_calls)}
        else:
            content = raw_message.get("content")
            if not isinstance(content, str):
                raise AgentProtocolError("completion final must have text content")
            if RS in content:
                raise AgentProtocolError("completion final contains a record-separator suffix")
            message = {"role": "assistant", "content": content}
        # serialize_pi_messages is the exact native serialization authority.
        native = serialize_pi_messages([{"role": "user", "content": "validation"}, message], append_assistant_header=False)
        turn = parse_agent_turn(native.split("Assistant:\n", 1)[1])
        return message, turn

    def run(self, *, system_prompt: str, user_prompt: str) -> TerminalReceipt:
        if self._consumed:
            raise RuntimeError("AcquisitionController is one-shot")
        self._consumed = True
        self._completion_usage = []
        if not isinstance(system_prompt, str) or not system_prompt or not isinstance(user_prompt, str) or not user_prompt:
            raise ValueError("non-empty system and user prompts are required")
        started = self.clock()
        deadline = started + self.max_seconds
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        actions: list[ActionReceipt] = []
        detector = NoProgressDetector()
        source_ledger: dict[str, Any] = {}
        acquired: dict[str, Mapping[str, Any]] = {}
        requests = 0
        while True:
            if self.clock() >= deadline:
                return self._terminal("time_limit", started, messages, actions, system_prompt)
            if requests >= self.max_turns:
                return self._terminal("turn_limit", started, messages, actions, system_prompt)
            request = {
                "model": self.model_id, "messages": [dict(message) for message in messages],
                "tools": list(self.tools), "temperature": 0,
                "max_completion_tokens": self.max_completion_tokens,
            }
            try:
                if self._completion_worker is None:
                    self._completion_worker = _ForkWorker("completion", self.completion_client)
                response = self._completion_worker.call({"request": request, "deadline": deadline}, deadline=deadline, clock=self.clock)
                if response.get("model") != self.model_id:
                    raise AgentProtocolError("completion model does not match controller model")
                expected_request_identity = {
                    "messages_sha256": sha256_text(canonical_json(request["messages"])),
                    "system_prompt_sha256": sha256_text(system_prompt),
                    "tool_schema_sha256": sha256_json(request["tools"]),
                    "request_sha256": sha256_json(request),
                }
                if response.get("emender_request_identity") != expected_request_identity:
                    raise AgentProtocolError("completion request identity mismatch")
                if validate_service_attestation(response.get("emender_service_attestation")) != self.expected_service_attestation:
                    raise AgentProtocolError("service attestation mismatch")
                completion_tokens = self._completion_tokens(response)
                assistant, turn = self._assistant(response)
                self._validate_assistant_body_limit(assistant)
                validate_generated_tool(turn, list(self.tools))
            except TimeoutError:
                return self._terminal("time_limit", started, messages, actions, system_prompt)
            except Exception as exc:
                return self._terminal("completion_error", started, messages, actions, system_prompt, str(exc))
            self._completion_usage.append({"sequence": len(self._completion_usage), "completion_tokens": completion_tokens})
            requests += 1
            if self.clock() >= deadline:
                return self._terminal("time_limit", started, messages, actions, system_prompt)
            messages.append(assistant)
            if turn.kind == "final":
                return self._terminal("success", started, messages, actions, system_prompt)
            if turn.tool_name is None or turn.arguments is None:
                return self._terminal("protocol_error", started, messages, actions, system_prompt, "tool turn is incomplete")
            call = assistant["tool_calls"][0]
            call_id = call.get("id") if isinstance(call, Mapping) else None
            if not isinstance(call_id, str) or not call_id:
                return self._terminal("protocol_error", started, messages, actions, system_prompt, "tool call id is required")
            # A completion can consume the final available moment after it was
            # parsed; never begin a model-selected action beyond the deadline.
            if self.clock() >= deadline:
                return self._terminal("time_limit", started, messages, actions, system_prompt)
            try:
                if self._tool_worker is None:
                    self._tool_worker = _ForkWorker("tool", self.tool_executor)
                tool_result = self._tool_worker.call({"tool_name": turn.tool_name, "arguments": turn.arguments}, deadline=deadline, clock=self.clock)
                raw_execution = tool_result["execution"]
                execution = ToolExecution(raw_execution["raw_observation"], raw_execution["effective_observation"], raw_execution["is_error"], raw_execution["source_ledger"])
                workspace_state = tool_result["workspace_state"]
            except TimeoutError:
                return self._terminal("time_limit", started, messages, actions, system_prompt)
            except Exception as exc:
                return self._terminal("tool_controller_error", started, messages, actions, system_prompt, str(exc))
            if self.clock() >= deadline:
                return self._terminal("time_limit", started, messages, actions, system_prompt)
            try:
                raw_digest = sha256_json(execution.raw_observation)
                authentic_effective = self._bounded_observation(execution.effective_observation)
                source_ledger.update(dict(execution.source_ledger))
                observation_key = sha256_json({"tool": turn.tool_name, "raw_observation": execution.raw_observation})
                acquired.setdefault(observation_key, {"tool": turn.tool_name, "raw_observation_sha256": raw_digest})
                acquired_observations = tuple(acquired[key] for key in sorted(acquired))
                receipt_state = ActionProgressReceipt(turn.tool_name, turn.arguments, workspace_state, source_ledger, acquired_observations)
                decision = detector.observe(receipt_state)
                effective = decision.observation if decision.observation is not None else authentic_effective
                receipt = ActionReceipt(
                    len(actions), call_id, turn.tool_name, dict(turn.arguments), turn.arguments_json or "", execution.raw_observation, raw_digest,
                    effective, sha256_text(effective), execution.is_error, dict(workspace_state), dict(source_ledger),
                    acquired_observations, action_fingerprint(turn.tool_name, turn.arguments),
                    progress_fingerprint(workspace_state=workspace_state, source_ledger=source_ledger, acquired_observations=acquired_observations), completion_tokens, decision.kind,
                )
            except Exception as exc:
                return self._terminal("tool_controller_error", started, messages, actions, system_prompt, str(exc))
            actions.append(receipt)
            if decision.kind == "terminate":
                return self._terminal("no_progress_terminated", started, messages, actions, system_prompt)
            messages.append({"role": "tool", "tool_call_id": call_id, "content": effective})
