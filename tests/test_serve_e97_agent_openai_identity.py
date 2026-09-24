import errno
import hashlib
import os
from pathlib import Path
import sys
import types

import pytest

from scripts import serve_e97_agent_openai as serving


def _read_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    while True:
        chunk = os.read(descriptor, 65536)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def test_private_checkpoint_snapshot_survives_directory_entry_swap(tmp_path, monkeypatch):
    original_parent = tmp_path / "checkpoint-dir"; original_parent.mkdir()
    checkpoint = original_parent / "checkpoint.pt"
    original = b"tiny-original-checkpoint"
    checkpoint.write_bytes(original)
    saved_parent = tmp_path / "saved-checkpoint-dir"

    def swap_then_report_unsupported(_destination, _request, _source):
        original_parent.rename(saved_parent)
        original_parent.mkdir()
        (original_parent / "checkpoint.pt").write_bytes(b"substituted-directory-entry")
        raise OSError(errno.EOPNOTSUPP, "reflink unsupported")

    monkeypatch.setattr(serving.fcntl, "ioctl", swap_then_report_unsupported)
    temporary, descriptor, digest = serving._private_checkpoint_snapshot(
        checkpoint, copy_fallback_maximum=1024,
    )
    try:
        assert digest == hashlib.sha256(original).hexdigest()
        assert _read_descriptor(descriptor) == original
        assert checkpoint.read_bytes() == b"substituted-directory-entry"
    finally:
        os.close(descriptor)
        temporary.cleanup()


def test_private_checkpoint_snapshot_hash_and_load_bytes_ignore_same_inode_rewrite(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    original = b"tiny-original-checkpoint"
    checkpoint.write_bytes(original)
    monkeypatch.setattr(
        serving.fcntl, "ioctl",
        lambda *_args: (_ for _ in ()).throw(OSError(errno.EOPNOTSUPP, "unsupported")),
    )
    temporary, descriptor, digest = serving._private_checkpoint_snapshot(
        checkpoint, copy_fallback_maximum=1024,
    )
    try:
        # This rewrites the same inode after startup hashing.  The unlinked
        # service-owned descriptor remains the sole load authority.
        checkpoint.write_bytes(b"same-inode-rewrite-after-attestation")
        assert digest == hashlib.sha256(original).hexdigest()
        assert _read_descriptor(descriptor) == original
    finally:
        os.close(descriptor)
        temporary.cleanup()


def test_checkpoint_snapshot_fails_closed_without_reflink_or_explicit_tiny_fallback(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"; checkpoint.write_bytes(b"tiny")
    monkeypatch.setattr(
        serving.fcntl, "ioctl",
        lambda *_args: (_ for _ in ()).throw(OSError(errno.EOPNOTSUPP, "unsupported")),
    )
    with pytest.raises(ValueError, match="fallback is disabled"):
        serving._private_checkpoint_snapshot(checkpoint)


def test_args_and_runtime_identity_snapshots_retain_pre_substitution_bytes(tmp_path, monkeypatch):
    args = tmp_path / "args.json"; runtime = tmp_path / "runtime.sqsh"
    args.write_bytes(b'{"level":"E97"}'); runtime.write_bytes(b"runtime-original")
    original_reader = serving.read_regular_file_no_follow

    def substitute_after_snapshot(path, *, maximum):
        payload = original_reader(path, maximum=maximum)
        Path(path).write_bytes(b"substituted-after-snapshot")
        return payload

    monkeypatch.setattr(serving, "read_regular_file_no_follow", substitute_after_snapshot)
    assert serving._snapshot_small_identity(args, name="args") == b'{"level":"E97"}'
    assert serving._snapshot_small_identity(runtime, name="runtime") == b"runtime-original"


def test_runtime_build_identity_uses_loaded_code_objects_without_source_reopen(monkeypatch):
    import ndm.e97_agent_server as server_module

    def fail_read(*_args, **_kwargs):
        raise AssertionError("runtime code identity must not reread source paths")

    monkeypatch.setattr(Path, "read_bytes", fail_read)
    digest = serving._loaded_module_code_sha256(server_module)
    assert len(digest) == 64


def test_runtime_build_identity_includes_explicit_serving_launcher_root(monkeypatch):
    import ndm.e97_agent_server as server_module

    original = serving._loaded_module_code_sha256(serving, server_module)

    def substituted_snapshot(*_args, **_kwargs):
        raise AssertionError("substituted launcher behavior")

    monkeypatch.setattr(serving, "_private_checkpoint_snapshot", substituted_snapshot)
    assert serving._loaded_module_code_sha256(serving, server_module) != original


def test_loaded_code_digest_recurses_into_nested_properties_and_wrapped_functions():
    module = types.ModuleType("ndm.identity_digest_fixture")
    exec(
        "class Probe:\n"
        "    MODE = 'original-mode'\n"
        "    @property\n"
        "    def behavior(self):\n"
        "        return 'original'\n"
        "def inner_one():\n"
        "    return 'one'\n"
        "def inner_two():\n"
        "    return 'two'\n"
        "def wrapped():\n"
        "    return wrapped.__wrapped__()\n"
        "wrapped.__wrapped__ = inner_one\n"
        "def make_closure(value):\n"
        "    def closure():\n"
        "        return value\n"
        "    return closure\n"
        "NESTED = {'behavior': (Probe, {'wrapper': wrapped, 'closure': make_closure('original')})}\n",
        module.__dict__,
    )
    sys.modules[module.__name__] = module
    try:
        original = serving._loaded_module_code_sha256(module)
        module.Probe.MODE = "changed-mode"
        assert serving._loaded_module_code_sha256(module) != original

        class_attribute_changed = serving._loaded_module_code_sha256(module)

        def changed_property(self):
            return "changed"

        changed_property.__module__ = module.__name__
        module.Probe.behavior = property(changed_property)
        assert serving._loaded_module_code_sha256(module) != class_attribute_changed

        property_changed = serving._loaded_module_code_sha256(module)
        module.wrapped.__wrapped__ = module.inner_two
        assert serving._loaded_module_code_sha256(module) != property_changed

        wrapper_changed = serving._loaded_module_code_sha256(module)
        module.NESTED["behavior"][1]["closure"].__closure__[0].cell_contents = "changed"
        assert serving._loaded_module_code_sha256(module) != wrapper_changed
    finally:
        sys.modules.pop(module.__name__, None)


def test_main_passes_exact_unlinked_checkpoint_stream_through_runtime_identity_path(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"; checkpoint.write_bytes(b"checkpoint-original")
    args_json = tmp_path / "args.json"; args_json.write_text("{}")
    runtime = tmp_path / "runtime.sqsh"; runtime.write_bytes(b"runtime-original")
    runtime_digest = hashlib.sha256(b"runtime-original").hexdigest()
    captured: dict[str, object] = {}
    real_hash = serving._sha256_and_rewind

    monkeypatch.setattr(
        serving.fcntl, "ioctl",
        lambda *_args: (_ for _ in ()).throw(OSError(errno.EOPNOTSUPP, "unsupported")),
    )

    def hash_runtime_descriptor_then_substitute(descriptor):
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) == (runtime.stat().st_dev, runtime.stat().st_ino):
            result = real_hash(descriptor)
            runtime.write_bytes(b"runtime-same-inode-substitution")
            captured["runtime_hashed_by_fd"] = True
            return result
        return real_hash(descriptor)

    def fake_load(stream, **kwargs):
        captured["checkpoint_stream_bytes"] = _read_descriptor(stream.fileno())
        captured["checkpoint_stream_nlink"] = os.fstat(stream.fileno()).st_nlink
        captured["checkpoint_path"] = kwargs["checkpoint_path"]
        # Rewrite the original source inode, then replace its directory entry.
        # Neither operation may affect the exact private stream passed here.
        checkpoint.write_bytes(b"checkpoint-same-inode-substitution")
        checkpoint.rename(tmp_path / "saved-checkpoint.pt")
        checkpoint.write_bytes(b"checkpoint-path-substitution")
        return types.SimpleNamespace(checkpoint_path=checkpoint, config={"tokenizer": "unit"})

    class FakeEngine:
        def __init__(self, loaded, **kwargs):
            self.loaded = loaded
            self.kwargs = kwargs

    class FakeService:
        def __init__(self, engine, **kwargs):
            captured["engine"] = engine
            captured["service_kwargs"] = kwargs

    monkeypatch.setattr(serving, "_sha256_and_rewind", hash_runtime_descriptor_then_substitute)
    monkeypatch.setattr(serving, "load_e97_checkpoint", fake_load)
    monkeypatch.setattr(serving, "TorchE97AgentEngine", FakeEngine)
    monkeypatch.setattr(serving, "AgentCompletionService", FakeService)
    monkeypatch.setattr(serving, "run_openai_server", lambda service, **kwargs: captured.update(server=(service, kwargs)))
    monkeypatch.setattr(sys, "argv", [
        "serve_e97_agent_openai.py", "--checkpoint", str(checkpoint),
        "--checkpoint-sha256", hashlib.sha256(b"checkpoint-original").hexdigest(),
        "--args-json", str(args_json), "--runtime-image", str(runtime), "--runtime-image-sha256", runtime_digest,
        "--checkpoint-copy-fallback-bytes", "1024", "--device", "cpu",
    ])
    serving.main()

    assert captured["runtime_hashed_by_fd"] is True
    assert captured["checkpoint_stream_bytes"] == b"checkpoint-original"
    assert captured["checkpoint_stream_nlink"] == 0
    assert captured["checkpoint_path"] == checkpoint
    assert checkpoint.read_bytes() == b"checkpoint-path-substitution"
    assert runtime.read_bytes() == b"runtime-same-inode-substitution"
    checkpoint_digest = hashlib.sha256(b"checkpoint-original").hexdigest()
    assert captured["engine"].kwargs["checkpoint_sha256"] == checkpoint_digest
    assert captured["service_kwargs"]["checkpoint_sha256"] == checkpoint_digest
    assert captured["service_kwargs"]["runtime_image_sha256"] == runtime_digest


def test_main_rejects_checkpoint_pin_mismatch_before_loader_or_service(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"; checkpoint.write_bytes(b"checkpoint-original")
    args_json = tmp_path / "args.json"; args_json.write_text("{}")
    monkeypatch.setattr(
        serving.fcntl, "ioctl",
        lambda *_args: (_ for _ in ()).throw(OSError(errno.EOPNOTSUPP, "unsupported")),
    )
    monkeypatch.setattr(serving, "_archived_controller_source_sha256", lambda: "a" * 64)
    monkeypatch.setattr(serving, "_loaded_module_code_sha256", lambda *_args: "b" * 64)
    monkeypatch.setattr(serving, "load_e97_checkpoint", lambda *_args, **_kwargs: pytest.fail("loader must not run"))
    monkeypatch.setattr(sys, "argv", [
        "serve_e97_agent_openai.py", "--checkpoint", str(checkpoint),
        "--checkpoint-sha256", "0" * 64, "--args-json", str(args_json),
        "--checkpoint-copy-fallback-bytes", "1024", "--device", "cpu",
    ])
    with pytest.raises(SystemExit) as result:
        serving.main()
    assert result.value.code == 2


def test_actual_checkpoint_loader_accepts_the_descriptor_stream_without_reopening_path(tmp_path, monkeypatch):
    """Exercise the real descriptor branch without constructing a model-scale checkpoint."""

    from ndm import e97

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"tiny descriptor fixture")
    observed = {}

    def fake_torch_load(stream, **kwargs):
        observed["stream"] = stream
        observed["kwargs"] = kwargs
        return {}

    monkeypatch.setattr(e97.torch, "load", fake_torch_load)
    with checkpoint.open("rb") as stream:
        with pytest.raises(ValueError, match="not a train.py checkpoint"):
            e97.load_e97_checkpoint(
                stream, checkpoint_path=checkpoint, device="cpu", mmap=True)
        assert observed["stream"] is stream
        assert observed["kwargs"]["mmap"] is False
