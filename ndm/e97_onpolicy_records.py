"""Canonical records and no-progress detection for E97 on-policy correction.

Model and teacher outputs are untrusted.  This module only describes immutable
execution receipts and validates already replayed, deterministically accepted
correction records.  Consumed V3/V4 evaluation identities are inadmissible.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from ndm.e97_agent_protocol import AgentProtocolError, parse_agent_turn


RECOVERY_RECORD_SCHEMA = "emender-e97-student-state-correction-v1"
TASK_IDENTITY_SCHEMA = "emender-e97-onpolicy-task-identity-v1"
NO_PROGRESS_SCHEMA = "emender-e97-no-progress-v1"
CONSUMED_V3_MANIFEST_SHA256 = (
    "ef481c637fde5916b8b0fe1f80cc2b4f0a6b88262088cbb33aefe0fed6bd6d09"
)
CONSUMED_V4_MANIFEST_SHA256 = (
    "8d0d5d350a39c9f5c007d98e4a0010f4a06f39e5a7fb89ecd9ccc4e37e66b8d8"
)
CONSUMED_PANEL_MANIFEST_SHA256S = (
    CONSUMED_V3_MANIFEST_SHA256,
    CONSUMED_V4_MANIFEST_SHA256,
)
CONSUMED_ID_PREFIXES = ("pi-eval-v3-", "pi-eval-v4-")
CANONICAL_NO_PROGRESS_OBSERVATION = (
    "no_progress: this identical action produced no new workspace, source, or "
    "observation state; choose a different action or finish with a grounded blocker."
)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def canonical_json(value: Any) -> str:
    """Return deterministic UTF-8 JSON; reject NaN and non-JSON objects."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def _require_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_name(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError(f"{name} is invalid")
    return value


def canonical_action(tool_name: str, arguments: Mapping[str, Any]) -> str:
    """Canonicalize a single bounded tool action for comparison and storage."""

    _require_name(tool_name, "tool_name")
    if not isinstance(arguments, Mapping):
        raise ValueError("tool arguments must be an object")
    arguments_json = canonical_json(dict(arguments))
    # Reuse the live protocol parser as the final syntax authority.
    turn = parse_agent_turn(f"Action: {tool_name}\nArguments: {arguments_json}")
    if turn.kind != "tool_call":  # pragma: no cover - parser contract
        raise AssertionError("canonical action did not parse as a tool call")
    return turn.raw_text


def action_fingerprint(tool_name: str, arguments: Mapping[str, Any]) -> str:
    """Hash the tool name and normalized JSON arguments."""

    action = canonical_action(tool_name, arguments)
    return sha256_text(f"{NO_PROGRESS_SCHEMA}\0action\0{action}")


def _canonical_observation_state(value: Any) -> Any:
    """Treat acquired-observation sequences as a deduplicated information set."""

    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        encoded = sorted({canonical_json(item) for item in value})
        return [json.loads(item) for item in encoded]
    raise ValueError("acquired_observations must be an object or sequence")


def task_identity(
    *,
    namespace: str,
    family_id: str,
    generator_source_digest: str,
    fixture_tree_digest: str,
    intent_digest: str,
) -> str:
    """Bind a task identity to its immutable generator, fixture, and intent."""

    _require_name(namespace, "task namespace")
    _require_name(family_id, "task family_id")
    values = {
        "schema": TASK_IDENTITY_SCHEMA,
        "namespace": namespace,
        "family_id": family_id,
        "generator_source_digest": _require_digest(
            generator_source_digest, "task generator_source_digest"),
        "fixture_tree_digest": _require_digest(
            fixture_tree_digest, "task fixture_tree_digest"),
        "intent_digest": _require_digest(intent_digest, "task intent_digest"),
    }
    return sha256_json(values)


def progress_fingerprint(
    *,
    workspace_state: Any,
    source_ledger: Any,
    acquired_observations: Any,
) -> str:
    """Hash the canonical task-visible world and acquired information state."""

    state = {
        "schema": NO_PROGRESS_SCHEMA,
        "workspace_state": workspace_state,
        "source_ledger": source_ledger,
        "acquired_observations": _canonical_observation_state(acquired_observations),
    }
    return sha256_json(state)


@dataclass(frozen=True)
class ActionProgressReceipt:
    """Post-execution state used to decide whether an action made progress."""

    tool_name: str
    arguments: Mapping[str, Any]
    workspace_state: Any
    source_ledger: Any
    acquired_observations: Any

    @property
    def action_fingerprint(self) -> str:
        return action_fingerprint(self.tool_name, self.arguments)

    @property
    def progress_fingerprint(self) -> str:
        return progress_fingerprint(
            workspace_state=self.workspace_state,
            source_ledger=self.source_ledger,
            acquired_observations=self.acquired_observations,
        )


@dataclass(frozen=True)
class NoProgressDecision:
    kind: str
    action_fingerprint: str
    progress_fingerprint: str
    observation: str | None = None


class NoProgressDetector:
    """Inject one recovery observation, then fail closed on another stale repeat.

    Receipts describe state *after* tool execution.  Consequently a duplicate
    read result does not expand the deduplicated observation set, while an edit,
    changed source ledger, or genuinely new observation changes the progress
    fingerprint and permits the same action again.
    """

    def __init__(self) -> None:
        self._seen: set[tuple[str, str]] = set()
        self._recovery_injected: set[tuple[str, str]] = set()

    def observe(self, receipt: ActionProgressReceipt) -> NoProgressDecision:
        action = receipt.action_fingerprint
        progress = receipt.progress_fingerprint
        pair = (action, progress)
        if pair in self._seen:
            if pair in self._recovery_injected:
                return NoProgressDecision("terminate", action, progress)
            self._recovery_injected.add(pair)
            return NoProgressDecision(
                "inject_no_progress", action, progress,
                CANONICAL_NO_PROGRESS_OBSERVATION,
            )
        self._seen.add(pair)
        return NoProgressDecision("continue", action, progress)


def _require_fields(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(f"{name} fields mismatch: {observed}")
    return value


def _validate_messages(
    messages: Any,
    *,
    divergence_index: int,
    target_start_index: int,
    observation_digest: str,
    system_prompt_sha256: str,
    max_messages: int,
    max_message_bytes: int,
) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not 5 <= len(messages) <= max_messages:
        raise ValueError("messages length is invalid")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(messages):
        message = _require_fields(raw, {"role", "text", "loss"}, f"message {index}")
        role, text, loss = message["role"], message["text"], message["loss"]
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"message {index} role is invalid")
        if not isinstance(text, str) or len(text.encode("utf-8")) > max_message_bytes:
            raise ValueError(f"message {index} text is invalid or oversized")
        if isinstance(loss, bool) or loss not in {0, 1}:
            raise ValueError(f"message {index} loss must be zero or one")
        if role != "assistant" and loss:
            raise ValueError("only assistant messages may carry loss")
        normalized.append({"role": role, "text": text, "loss": int(loss)})
    if [normalized[0]["role"], normalized[1]["role"]] != ["system", "user"]:
        raise ValueError("records must begin with system then user")
    if sha256_text(normalized[0]["text"]) != system_prompt_sha256:
        raise ValueError("system prompt digest mismatch")
    if normalized[-1]["role"] != "assistant" or not normalized[-1]["text"].startswith("Final:"):
        raise ValueError("records require one terminal Final assistant turn")
    if normalized[-1]["loss"] != 1:
        raise ValueError("terminal Final must be a target")
    if not 2 <= divergence_index < len(normalized) - 2:
        raise ValueError("first divergence message index is invalid")
    if not divergence_index < target_start_index < len(normalized):
        raise ValueError("target start must follow the first divergence")
    if normalized[divergence_index]["role"] != "assistant" or normalized[divergence_index]["loss"]:
        raise ValueError("first divergent student action must be zero-loss assistant context")
    if normalized[divergence_index + 1]["role"] != "tool":
        raise ValueError("first divergent action must retain its authentic tool observation")
    if sha256_text(normalized[divergence_index + 1]["text"]) != observation_digest:
        raise ValueError("first divergence observation digest mismatch")
    if normalized[target_start_index]["role"] != "assistant":
        raise ValueError("target suffix must begin on an assistant message")
    for index, message in enumerate(normalized):
        if message["role"] == "assistant":
            expected_loss = int(index >= target_start_index)
            if message["loss"] != expected_loss:
                raise ValueError("assistant targets must be one contiguous corrective suffix")
    # After the initial user turn, assistant/tool messages must alternate until Final.
    expected = "assistant"
    for index, message in enumerate(normalized[2:], start=2):
        if message["role"] != expected:
            raise ValueError(f"message {index} violates assistant/tool alternation")
        expected = "tool" if expected == "assistant" else "assistant"
    return normalized


def validate_recovery_record(
    value: Any,
    *,
    forbidden_source_digests: Sequence[str] = CONSUMED_PANEL_MANIFEST_SHA256S,
    forbidden_id_prefixes: Sequence[str] = CONSUMED_ID_PREFIXES,
    max_messages: int = 32,
    max_message_bytes: int = 64 << 10,
) -> dict[str, Any]:
    """Validate and normalize one deterministically accepted correction record."""

    record = _require_fields(value, {
        "schema", "split", "task", "student", "teacher", "runtime",
        "first_divergence", "messages", "validator_receipt", "source_provenance",
    }, "record")
    if record["schema"] != RECOVERY_RECORD_SCHEMA:
        raise ValueError("unsupported recovery record schema")
    if record["split"] not in {"train", "validation"}:
        raise ValueError("record split must be train or validation")

    task = _require_fields(record["task"], {
        "namespace", "family_id", "identity", "generator_source_digest",
        "fixture_tree_digest", "intent_digest",
    }, "task")
    namespace = _require_name(task["namespace"], "task namespace")
    family = _require_name(task["family_id"], "task family_id")
    expected_namespace = "e97-train-" if record["split"] == "train" else "e97-dev-"
    if not namespace.startswith(expected_namespace):
        raise ValueError("task namespace does not match record split")
    if any(namespace.startswith(prefix) or family.startswith(prefix) for prefix in forbidden_id_prefixes):
        raise ValueError("consumed evaluation identity is inadmissible")
    for key in ("identity", "generator_source_digest", "fixture_tree_digest", "intent_digest"):
        _require_digest(task[key], f"task {key}")
    derived_task_identity = task_identity(
        namespace=namespace,
        family_id=family,
        generator_source_digest=task["generator_source_digest"],
        fixture_tree_digest=task["fixture_tree_digest"],
        intent_digest=task["intent_digest"],
    )
    if task["identity"] != derived_task_identity:
        raise ValueError("task identity is not bound to its immutable inputs")

    student = _require_fields(record["student"], {
        "rollout_identity", "checkpoint_sha256", "decode",
    }, "student")
    _require_digest(student["rollout_identity"], "student rollout_identity")
    _require_digest(student["checkpoint_sha256"], "student checkpoint_sha256")
    if not isinstance(student["decode"], Mapping) or not student["decode"]:
        raise ValueError("student decode identity must be a non-empty object")
    canonical_json(student["decode"])

    teacher = _require_fields(record["teacher"], {"tier", "model_revision"}, "teacher")
    if teacher["tier"] not in {"luna", "terra", "sol", "human"}:
        raise ValueError("teacher tier is invalid")
    if not isinstance(teacher["model_revision"], str) or not teacher["model_revision"]:
        raise ValueError("teacher model revision is required")

    runtime = _require_fields(record["runtime"], {
        "schema_digest", "controller_digest", "system_prompt_sha256", "tool_schema_digest",
    }, "runtime")
    for key, item in runtime.items():
        _require_digest(item, f"runtime {key}")

    divergence = _require_fields(record["first_divergence"], {
        "message_index", "target_start_message_index", "class",
        "student_action_canonical", "pre_state_digest", "observation_digest",
    }, "first_divergence")
    if isinstance(divergence["message_index"], bool) or not isinstance(divergence["message_index"], int):
        raise ValueError("first divergence message_index must be an integer")
    if (isinstance(divergence["target_start_message_index"], bool)
            or not isinstance(divergence["target_start_message_index"], int)):
        raise ValueError("target start message index must be an integer")
    _require_name(divergence["class"], "first divergence class")
    _require_digest(divergence["pre_state_digest"], "first divergence pre_state_digest")
    observation_digest = _require_digest(
        divergence["observation_digest"], "first divergence observation_digest")

    messages = _validate_messages(
        record["messages"],
        divergence_index=divergence["message_index"],
        target_start_index=divergence["target_start_message_index"],
        observation_digest=observation_digest,
        system_prompt_sha256=runtime["system_prompt_sha256"],
        max_messages=max_messages,
        max_message_bytes=max_message_bytes,
    )
    try:
        parsed = parse_agent_turn(messages[divergence["message_index"]]["text"])
    except AgentProtocolError as exc:
        raise ValueError("first divergent action is not canonical protocol") from exc
    if parsed.kind != "tool_call" or parsed.tool_name is None or parsed.arguments is None:
        raise ValueError("first divergence must be a tool call")
    expected_action = canonical_action(parsed.tool_name, parsed.arguments)
    if divergence["student_action_canonical"] != expected_action:
        raise ValueError("stored divergent action is not canonical or does not match messages")

    receipt = _require_fields(record["validator_receipt"], {
        "validator_digest", "input_digest", "postcondition_digest",
        "action_graph_digest", "passed", "cycle_free",
    }, "validator_receipt")
    for key in ("validator_digest", "input_digest", "postcondition_digest", "action_graph_digest"):
        _require_digest(receipt[key], f"validator receipt {key}")
    if receipt["passed"] is not True or receipt["cycle_free"] is not True:
        raise ValueError("only passed, cycle-free validator receipts are trainable")

    provenance = _require_fields(record["source_provenance"], {
        "source_digests", "forbidden_panel_digests_checked",
    }, "source_provenance")
    if not isinstance(provenance["source_digests"], list):
        raise ValueError("source provenance digests must be a list")
    sources = [_require_digest(item, "source provenance digest")
               for item in provenance["source_digests"]]
    if len(sources) != len(set(sources)):
        raise ValueError("source provenance digests must be unique")
    forbidden = {_require_digest(item, "forbidden source digest")
                 for item in forbidden_source_digests}
    if forbidden.intersection(sources):
        raise ValueError("consumed evaluation source is inadmissible")
    checked = provenance["forbidden_panel_digests_checked"]
    if not isinstance(checked, list) or not forbidden.issubset(checked):
        raise ValueError("record lacks the required forbidden-panel check receipt")
    for item in checked:
        _require_digest(item, "checked forbidden panel digest")

    # Round-trip through canonical JSON to detach callers' mutable mappings.
    return json.loads(canonical_json(dict(record)))


def recovery_record_fingerprint(record: Mapping[str, Any]) -> str:
    """Return the immutable identity of one validated correction record."""

    normalized = validate_recovery_record(record)
    return sha256_text(f"{RECOVERY_RECORD_SCHEMA}\0{canonical_json(normalized)}")
