import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ndm.e97 import E97RecurrentCache, e97_cache_suffix
from ndm.e97_state import clone_e97_cache, load_e97_state, save_e97_state


MODEL_SHA = "1" * 64
PROMPT_SHA = hashlib.sha256(b"system prompt").hexdigest()


def loaded(tmp_path: Path):
    return SimpleNamespace(
        checkpoint_path=tmp_path / "checkpoint.pt",
        tokenizer_name="gpt2",
        model=torch.nn.Linear(2, 2, bias=False),
    )


def cache():
    return E97RecurrentCache(
        token_ids=(11, 22, 33),
        hidden=[torch.arange(6, dtype=torch.float32).reshape(2, 3), (None, torch.ones(4))],
        next_logits=torch.tensor([0.25, -0.5], dtype=torch.float32),
        checkpoint="/source/checkpoint.pt",
        token_count=3,
        token_lineage_sha256="2" * 64,
    )


def test_portable_state_roundtrip_is_compact_bound_and_integrity_checked(tmp_path):
    destination = tmp_path / "session.e97state"
    receipt = save_e97_state(
        destination,
        cache(),
        model_checkpoint_sha256=MODEL_SHA,
        tokenizer="gpt2",
        system_prompt_sha256=PROMPT_SHA,
        metadata={"source_ledger_sha256": "3" * 64},
    )
    restored = load_e97_state(
        destination,
        loaded=loaded(tmp_path),
        model_checkpoint_sha256=MODEL_SHA,
        tokenizer="gpt2",
        system_prompt_sha256=PROMPT_SHA,
    )

    assert receipt["artifact_sha256"] == restored.artifact_sha256
    assert restored.cache.token_ids == ()
    assert restored.cache.total_token_count == 3
    assert restored.cache.token_lineage_sha256 == "2" * 64
    assert restored.cache.checkpoint == str(tmp_path / "checkpoint.pt")
    assert e97_cache_suffix(restored.cache, (11, 22, 33, 44)) is None
    assert torch.equal(restored.cache.hidden[0], cache().hidden[0])
    assert torch.equal(restored.cache.hidden[1][1], cache().hidden[1][1])
    assert torch.equal(restored.cache.next_logits, cache().next_logits)
    assert restored.metadata["transcript_tokens_included"] is False
    assert restored.metadata["metadata"]["source_ledger_sha256"] == "3" * 64
    assert destination.stat().st_mode & 0o777 == 0o600

    damaged = bytearray(destination.read_bytes())
    damaged[-40] ^= 1
    destination.write_bytes(damaged)
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_e97_state(
            destination,
            loaded=loaded(tmp_path),
            model_checkpoint_sha256=MODEL_SHA,
            tokenizer="gpt2",
            system_prompt_sha256=PROMPT_SHA,
        )


def test_portable_state_fails_closed_on_identity_mismatch(tmp_path):
    destination = tmp_path / "session.e97state"
    save_e97_state(
        destination,
        cache(),
        model_checkpoint_sha256=MODEL_SHA,
        tokenizer="gpt2",
        system_prompt_sha256=PROMPT_SHA,
    )
    with pytest.raises(ValueError, match="model_checkpoint_sha256 mismatch"):
        load_e97_state(
            destination,
            loaded=loaded(tmp_path),
            model_checkpoint_sha256="4" * 64,
            tokenizer="gpt2",
            system_prompt_sha256=PROMPT_SHA,
        )
    with pytest.raises(ValueError, match="system_prompt_sha256 mismatch"):
        load_e97_state(
            destination,
            loaded=loaded(tmp_path),
            model_checkpoint_sha256=MODEL_SHA,
            tokenizer="gpt2",
            system_prompt_sha256="5" * 64,
        )


def test_portable_state_optional_encryption(tmp_path):
    pytest.importorskip("cryptography")
    destination = tmp_path / "private.e97state"
    key = bytes(range(32))
    save_e97_state(
        destination,
        cache(),
        model_checkpoint_sha256=MODEL_SHA,
        tokenizer="gpt2",
        system_prompt_sha256=PROMPT_SHA,
        encryption_key=key,
    )
    artifact_bytes = destination.read_bytes()
    assert b'"algorithm":"AES-256-GCM"' in artifact_bytes
    with pytest.raises(ValueError, match="encryption_key is required"):
        load_e97_state(
            destination,
            loaded=loaded(tmp_path),
            model_checkpoint_sha256=MODEL_SHA,
            tokenizer="gpt2",
            system_prompt_sha256=PROMPT_SHA,
        )
    with pytest.raises(ValueError, match="decryption failed"):
        load_e97_state(
            destination,
            loaded=loaded(tmp_path),
            model_checkpoint_sha256=MODEL_SHA,
            tokenizer="gpt2",
            system_prompt_sha256=PROMPT_SHA,
            encryption_key=b"x" * 32,
        )
    restored = load_e97_state(
        destination,
        loaded=loaded(tmp_path),
        model_checkpoint_sha256=MODEL_SHA,
        tokenizer="gpt2",
        system_prompt_sha256=PROMPT_SHA,
        encryption_key=key,
    )
    assert torch.equal(restored.cache.next_logits, cache().next_logits)


def test_clone_e97_cache_has_independent_tensor_storage():
    original = cache()
    branch = clone_e97_cache(original)
    branch.hidden[0].add_(100)
    branch.next_logits.zero_()
    assert not torch.equal(branch.hidden[0], original.hidden[0])
    assert not torch.equal(branch.next_logits, original.next_logits)
    assert branch.total_token_count == original.total_token_count
    assert branch.token_lineage_sha256 == original.token_lineage_sha256
