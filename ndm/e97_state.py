"""Portable, integrity-checked recurrent state artifacts for E97.

The compact format intentionally omits transcript token IDs. Restored states
therefore continue through explicit token deltas and never replay the prefix.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import secrets
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .e97 import E97RecurrentCache, LoadedE97Checkpoint


MAGIC = b"EMENDER_E97STATE_V1\n"
SCHEMA = "emender-e97-portable-state-v1"
RUNTIME_SCHEMA = "emender-e97-tokenwise-runtime-v1"
_HEADER_LENGTHS = struct.Struct(">QQ")
_TRAILER_BYTES = 32


def _sha256_text(value: str, field: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64:
        raise ValueError(f"{field} must be a 64-character SHA-256 digest")
    try:
        bytes.fromhex(normalized)
    except ValueError as exc:
        raise ValueError(f"{field} must be hexadecimal") from exc
    return normalized


def _encode_tree(value: Any, tensors: dict[str, torch.Tensor]) -> Mapping[str, Any]:
    if torch.is_tensor(value):
        key = f"tensor-{len(tensors):06d}"
        tensor = value.detach().to(device="cpu").contiguous()
        tensors[key] = tensor
        return {
            "kind": "tensor",
            "key": key,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
    if value is None:
        return {"kind": "none"}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_encode_tree(item, tensors) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_encode_tree(item, tensors) for item in value]}
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("portable E97 state mappings require string keys")
        return {
            "kind": "dict",
            "items": [[key, _encode_tree(value[key], tensors)] for key in sorted(value)],
        }
    raise TypeError(f"unsupported recurrent-state value: {type(value).__name__}")


def _decode_tree(node: Mapping[str, Any], tensors: Mapping[str, torch.Tensor]) -> Any:
    kind = node.get("kind")
    if kind == "none":
        return None
    if kind == "tensor":
        key = node.get("key")
        if not isinstance(key, str) or key not in tensors:
            raise ValueError("portable state references a missing tensor")
        tensor = tensors[key]
        if str(tensor.dtype) != node.get("dtype") or list(tensor.shape) != node.get("shape"):
            raise ValueError(f"portable state tensor metadata mismatch for {key}")
        return tensor
    if kind in {"list", "tuple"}:
        items = node.get("items")
        if not isinstance(items, list):
            raise ValueError("portable state sequence metadata is invalid")
        decoded = [_decode_tree(item, tensors) for item in items]
        return tuple(decoded) if kind == "tuple" else decoded
    if kind == "dict":
        items = node.get("items")
        if not isinstance(items, list):
            raise ValueError("portable state mapping metadata is invalid")
        result = {}
        for item in items:
            if not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str):
                raise ValueError("portable state mapping entry is invalid")
            result[item[0]] = _decode_tree(item[1], tensors)
        return result
    raise ValueError(f"unsupported portable state tree node: {kind!r}")


def _clone_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _clone_tree(item) for key, item in value.items()}
    if value is None:
        return None
    raise TypeError(f"unsupported recurrent-state value: {type(value).__name__}")


def clone_e97_cache(cache: E97RecurrentCache) -> E97RecurrentCache:
    """Clone tensor storage so branches cannot mutate one another."""

    return E97RecurrentCache(
        token_ids=tuple(cache.token_ids),
        hidden=_clone_tree(cache.hidden),
        next_logits=cache.next_logits.detach().clone(),
        checkpoint=cache.checkpoint,
        token_count=cache.total_token_count,
        token_lineage_sha256=cache.token_lineage_sha256,
    )


def _aesgcm():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:  # pragma: no cover - depends on deployment extras
        raise RuntimeError("encrypted E97 state requires the cryptography package") from exc
    return AESGCM


def _validate_key(encryption_key: bytes | None) -> bytes | None:
    if encryption_key is None:
        return None
    if not isinstance(encryption_key, bytes) or len(encryption_key) != 32:
        raise ValueError("encryption_key must contain exactly 32 bytes")
    return encryption_key


@dataclass(frozen=True)
class RestoredE97State:
    cache: E97RecurrentCache
    metadata: Mapping[str, Any]
    artifact_sha256: str
    artifact_bytes: int


def save_e97_state(
    path: str | Path,
    cache: E97RecurrentCache,
    *,
    model_checkpoint_sha256: str,
    tokenizer: str,
    system_prompt_sha256: str,
    runtime_schema: str = RUNTIME_SCHEMA,
    encryption_key: bytes | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Atomically save a compact `.e97state` artifact.

    Transcript tokens are excluded. ``token_count`` and the append-only lineage
    digest bind the state to its exact ingestion history without making artifact
    size grow with the transcript.
    """

    destination = Path(path)
    if destination.suffix != ".e97state":
        raise ValueError("portable state path must end in .e97state")
    checkpoint_sha = _sha256_text(model_checkpoint_sha256, "model_checkpoint_sha256")
    prompt_sha = _sha256_text(system_prompt_sha256, "system_prompt_sha256")
    if not tokenizer:
        raise ValueError("tokenizer identity cannot be empty")
    if not runtime_schema:
        raise ValueError("runtime schema cannot be empty")
    key = _validate_key(encryption_key)

    tensors: dict[str, torch.Tensor] = {}
    hidden_tree = _encode_tree(cache.hidden, tensors)
    logits_tree = _encode_tree(cache.next_logits, tensors)
    tensor_buffer = io.BytesIO()
    torch.save(tensors, tensor_buffer)
    payload = tensor_buffer.getvalue()

    nonce = secrets.token_bytes(12) if key is not None else None
    header = {
        "schema": SCHEMA,
        "runtime_schema": runtime_schema,
        "model_checkpoint_sha256": checkpoint_sha,
        "tokenizer": tokenizer,
        "system_prompt_sha256": prompt_sha,
        "token_count": cache.total_token_count,
        "token_lineage_sha256": cache.token_lineage_sha256,
        "transcript_tokens_included": False,
        "state_bytes": cache.state_bytes,
        "hidden_tree": hidden_tree,
        "next_logits_tree": logits_tree,
        "encryption": (
            {"algorithm": "AES-256-GCM", "nonce_base64": base64.b64encode(nonce).decode("ascii")}
            if nonce is not None else {"algorithm": "none"}
        ),
        "metadata": dict(metadata or {}),
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if key is not None:
        payload = _aesgcm()(key).encrypt(nonce, payload, header_bytes)
    prefix = MAGIC + _HEADER_LENGTHS.pack(len(header_bytes), len(payload)) + header_bytes + payload
    internal_digest = hashlib.sha256(prefix).digest()
    artifact = prefix + internal_digest

    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(artifact)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)

    return {
        "path": str(destination),
        "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
        "artifact_bytes": len(artifact),
        "state_bytes": cache.state_bytes,
        "token_count": cache.total_token_count,
        "encrypted": key is not None,
    }


def load_e97_state(
    path: str | Path,
    *,
    loaded: LoadedE97Checkpoint,
    model_checkpoint_sha256: str,
    tokenizer: str,
    system_prompt_sha256: str,
    runtime_schema: str = RUNTIME_SCHEMA,
    encryption_key: bytes | None = None,
    map_location: str | torch.device | None = None,
) -> RestoredE97State:
    """Verify identity and integrity, then restore state for ``loaded``."""

    artifact = Path(path).read_bytes()
    minimum = len(MAGIC) + _HEADER_LENGTHS.size + _TRAILER_BYTES
    if len(artifact) < minimum or not artifact.startswith(MAGIC):
        raise ValueError("not an Emender E97 portable state artifact")
    prefix, trailer = artifact[:-_TRAILER_BYTES], artifact[-_TRAILER_BYTES:]
    if hashlib.sha256(prefix).digest() != trailer:
        raise ValueError("portable E97 state checksum mismatch")

    offset = len(MAGIC)
    header_length, payload_length = _HEADER_LENGTHS.unpack_from(artifact, offset)
    offset += _HEADER_LENGTHS.size
    expected_length = offset + header_length + payload_length + _TRAILER_BYTES
    if expected_length != len(artifact):
        raise ValueError("portable E97 state length metadata mismatch")
    header_bytes = artifact[offset:offset + header_length]
    offset += header_length
    payload = artifact[offset:offset + payload_length]
    header = json.loads(header_bytes)
    if not isinstance(header, dict) or header.get("schema") != SCHEMA:
        raise ValueError("unsupported portable E97 state schema")

    expected = {
        "model_checkpoint_sha256": _sha256_text(model_checkpoint_sha256, "model_checkpoint_sha256"),
        "tokenizer": tokenizer,
        "system_prompt_sha256": _sha256_text(system_prompt_sha256, "system_prompt_sha256"),
        "runtime_schema": runtime_schema,
    }
    for field, value in expected.items():
        if header.get(field) != value:
            raise ValueError(f"portable E97 state {field} mismatch")
    if loaded.tokenizer_name != tokenizer:
        raise ValueError("loaded checkpoint tokenizer does not match portable state tokenizer")

    encryption = header.get("encryption")
    if not isinstance(encryption, dict):
        raise ValueError("portable E97 state encryption metadata is invalid")
    algorithm = encryption.get("algorithm")
    key = _validate_key(encryption_key)
    if algorithm == "AES-256-GCM":
        if key is None:
            raise ValueError("portable E97 state is encrypted; encryption_key is required")
        try:
            nonce = base64.b64decode(encryption["nonce_base64"], validate=True)
            payload = _aesgcm()(key).decrypt(nonce, payload, header_bytes)
        except Exception as exc:
            raise ValueError("portable E97 state decryption failed") from exc
    elif algorithm == "none":
        if key is not None:
            raise ValueError("encryption_key supplied for an unencrypted portable state")
    else:
        raise ValueError("unsupported portable E97 state encryption algorithm")

    tensors = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(tensors, dict) or not all(
        isinstance(name, str) and torch.is_tensor(tensor)
        for name, tensor in tensors.items()
    ):
        raise ValueError("portable E97 tensor payload is invalid")
    hidden = _decode_tree(header["hidden_tree"], tensors)
    next_logits = _decode_tree(header["next_logits_tree"], tensors)
    if not torch.is_tensor(next_logits):
        raise ValueError("portable E97 next logits must be a tensor")
    target = next(loaded.model.parameters()).device if map_location is None else torch.device(map_location)

    def move(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.to(device=target)
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, Mapping):
            return {name: move(item) for name, item in value.items()}
        return value

    cache = E97RecurrentCache(
        token_ids=(),
        hidden=move(hidden),
        next_logits=move(next_logits),
        checkpoint=str(loaded.checkpoint_path),
        token_count=int(header["token_count"]),
        token_lineage_sha256=header.get("token_lineage_sha256"),
    )
    if cache.state_bytes != int(header["state_bytes"]):
        raise ValueError("portable E97 state byte accounting mismatch")
    return RestoredE97State(
        cache=cache,
        metadata=header,
        artifact_sha256=hashlib.sha256(artifact).hexdigest(),
        artifact_bytes=len(artifact),
    )


__all__ = [
    "MAGIC",
    "RUNTIME_SCHEMA",
    "SCHEMA",
    "RestoredE97State",
    "clone_e97_cache",
    "load_e97_state",
    "save_e97_state",
]
