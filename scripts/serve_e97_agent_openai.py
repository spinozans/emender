#!/usr/bin/env python3
"""Serve dense recurrent E97 through bounded OpenAI Chat Completions."""
from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import marshal
import os
from pathlib import Path
import stat
import secrets
import sys
import tempfile
import types
from dataclasses import dataclass

import torch

from ndm.e97_atomic import open_directory_no_follow, read_regular_file_no_follow
from ndm.e97_first_party_source_archive import (
    verified_loaded_module_closure_sha256,
    verified_source_member_payload,
)

from ndm.e97 import load_e97_checkpoint
from ndm.e97_acquisition_controller import READ_OBSERVE_TOOLS
from ndm.e97_agent_protocol import DENSE_AGENT_CLI_DIRECT_SYSTEM, DENSE_AGENT_CLI_SYSTEM, DENSE_AGENT_V1_SYSTEM, DENSE_AGENT_V2_SYSTEM, E97_PI_AGENT_ANALYSIS_SYSTEM_V1, E97_PI_AGENT_SYSTEM_V2, E97_PI_CORE_SYSTEM
from ndm.e97_agent_server import (
    AgentCompletionService,
    TorchE97AgentEngine,
    run_openai_server,
)


def _open_regular_no_follow(path: Path) -> int:
    """Open one regular source artifact through a pinned parent descriptor."""

    parent_fd = open_directory_no_follow(path.parent)
    try:
        descriptor = os.open(
            path.name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent_fd,
        )
    finally:
        os.close(parent_fd)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("identity artifact is not a regular file")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _sha256_and_rewind(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1 << 20)
        if not chunk:
            break
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _snapshot_small_identity(path: Path, *, name: str, maximum: int = 16 << 20) -> bytes:
    try:
        return read_regular_file_no_follow(path, maximum=maximum)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"{name} cannot be snapshotted safely") from exc


def _archived_controller_source_sha256() -> str:
    """Bind loaded controller code to retained source without reopening checkout files."""

    root = Path(__file__).resolve().parents[1]
    manifest = _snapshot_small_identity(
        root / "configs/pi/e97-firstparty-generator-manifest-v1.json", name="controller source manifest")
    archive = _snapshot_small_identity(
        root / "configs/pi/e97-firstparty-source-v1.tar", name="controller source archive", maximum=256 << 20)
    # Verify that this is the expected archived source member before comparing
    # the actual already-imported callable closure with compiled retained code.
    verified_source_member_payload(manifest, archive, "ndm/e97_acquisition_controller.py")
    return verified_loaded_module_closure_sha256(
        manifest, archive, root_module="ndm.e97_acquisition_controller")


def _loaded_module_code_sha256(module: types.ModuleType, *additional_modules: types.ModuleType) -> str:
    """Hash loaded launcher roots and bounded reachable first-party behavior.

    Startup attestation is established by this launcher as well as ``ndm``
    code.  Explicit roots may therefore be outside ``ndm`` (the serving
    launcher), while transitive discovery remains bounded to ``ndm`` and those
    roots.  Function defaults, keyword defaults, wrappers, and closure cells
    are behavior, not annotations, and are part of the identity.
    """

    roots = (module, *additional_modules)
    if any(not isinstance(item, types.ModuleType) for item in roots):
        raise ValueError("runtime identity roots must be modules")
    explicit_root_names = {item.__name__ for item in roots}
    pending = list(roots)
    visited_modules: set[str] = set()
    visited_values: set[int] = set()
    entries: list[tuple[str, bytes]] = []

    def permitted_name(name: object) -> bool:
        return isinstance(name, str) and (name.startswith("ndm.") or name in explicit_root_names)

    def value_identity(value: object, active: set[int] | None = None) -> bytes:
        """Return a deterministic runtime value identity without memory addresses."""

        active = set() if active is None else active
        digest = hashlib.sha256()

        def add(tag: str, payload: bytes) -> None:
            encoded = tag.encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "big")); digest.update(encoded)
            digest.update(len(payload).to_bytes(8, "big")); digest.update(payload)

        def visit_value(item: object) -> None:
            if item is None:
                add("none", b"")
            elif isinstance(item, bool):
                add("bool", b"1" if item else b"0")
            elif isinstance(item, int):
                add("int", str(item).encode("ascii"))
            elif isinstance(item, float):
                if item != item or item in (float("inf"), float("-inf")):
                    raise ValueError("runtime code identity contains a non-finite value")
                add("float", item.hex().encode("ascii"))
            elif isinstance(item, str):
                add("str", item.encode("utf-8"))
            elif isinstance(item, bytes):
                add("bytes", item)
            elif isinstance(item, Path):
                add("path", str(item).encode("utf-8"))
            elif isinstance(item, types.CodeType):
                add("code", marshal.dumps(item))
            elif isinstance(item, (types.BuiltinFunctionType, types.BuiltinMethodType)):
                add("builtin", f"{item.__module__}.{item.__qualname__}".encode("utf-8"))
            elif isinstance(item, types.FunctionType):
                identifier = id(item)
                if identifier in active:
                    add("function-cycle", f"{item.__module__}.{item.__qualname__}".encode("utf-8"))
                    return
                active.add(identifier)
                try:
                    add("function", function_identity(item, active))
                finally:
                    active.remove(identifier)
            elif isinstance(item, types.MethodType):
                add("method", function_identity(item.__func__, active))
            elif isinstance(item, type):
                add("type", f"{item.__module__}.{item.__qualname__}".encode("utf-8"))
            elif isinstance(item, (tuple, list)):
                identifier = id(item)
                if identifier in active:
                    add(type(item).__name__ + "-cycle", type(item).__qualname__.encode("utf-8"))
                    return
                active.add(identifier)
                try:
                    encoded = bytearray()
                    for child in item:
                        child_identity = value_identity(child, active)
                        encoded.extend(len(child_identity).to_bytes(8, "big")); encoded.extend(child_identity)
                    add(type(item).__name__, bytes(encoded))
                finally:
                    active.remove(identifier)
            elif isinstance(item, (set, frozenset)):
                identifier = id(item)
                if identifier in active:
                    add(type(item).__name__ + "-cycle", type(item).__qualname__.encode("utf-8"))
                    return
                active.add(identifier)
                try:
                    values = sorted(value_identity(child, active) for child in item)
                    add(type(item).__name__, b"".join(values))
                finally:
                    active.remove(identifier)
            elif isinstance(item, dict):
                if not all(isinstance(key, str) for key in item):
                    raise ValueError("runtime code identity mapping key is invalid")
                identifier = id(item)
                if identifier in active:
                    add("dict-cycle", b"builtins.dict")
                    return
                active.add(identifier)
                try:
                    encoded = bytearray()
                    for key in sorted(item):
                        key_bytes, child_identity = key.encode("utf-8"), value_identity(item[key], active)
                        encoded.extend(len(key_bytes).to_bytes(4, "big")); encoded.extend(key_bytes)
                        encoded.extend(child_identity)
                    add("dict", bytes(encoded))
                finally:
                    active.remove(identifier)
            else:
                module_name = getattr(item, "__module__", None)
                qualname = getattr(item, "__qualname__", None)
                if isinstance(module_name, str) and isinstance(qualname, str):
                    add("external-symbol", f"{module_name}.{qualname}".encode("utf-8"))
                    return
                item_type = type(item)
                type_name = f"{item_type.__module__}.{item_type.__qualname__}"
                state = getattr(item, "__dict__", None)
                if isinstance(state, dict):
                    add("external-instance", type_name.encode("utf-8") + value_identity(state))
                else:
                    # Extension/typing sentinels have no portable value state;
                    # retain their stable type rather than a repr containing an
                    # address.  First-party closures/defaults above are all
                    # recursively value-bound.
                    add("external-instance", type_name.encode("utf-8"))

        visit_value(value)
        return digest.digest()

    def function_identity(function: types.FunctionType, active: set[int]) -> bytes:
        digest = hashlib.sha256(b"emender-e97-loaded-function-semantics-v2\0")
        digest.update(marshal.dumps(function.__code__))
        for index, value in enumerate(function.__defaults__ or ()):
            digest.update(b"default\0" + str(index).encode("ascii") + b"\0" + value_identity(value, active))
        for name, value in sorted((function.__kwdefaults__ or {}).items()):
            digest.update(b"kwdefault\0" + name.encode("utf-8") + b"\0" + value_identity(value, active))
        for index, cell in enumerate(function.__closure__ or ()):
            try:
                value = cell.cell_contents
            except ValueError:
                digest.update(b"closure-empty\0" + str(index).encode("ascii"))
            else:
                digest.update(b"closure\0" + str(index).encode("ascii") + b"\0" + value_identity(value, active))
        return digest.digest()

    def enqueue_module(value: object) -> None:
        if isinstance(value, types.ModuleType) and permitted_name(value.__name__) and value.__name__ not in visited_modules:
            pending.append(value)
            return
        module_name = getattr(value, "__module__", None)
        candidate = sys.modules.get(module_name) if permitted_name(module_name) else None
        if isinstance(candidate, types.ModuleType) and candidate.__name__ not in visited_modules:
            pending.append(candidate)

    def record_value(value: object, label: str) -> None:
        entries.append((label, value_identity(value)))

    def visit(value: object, label: str) -> None:
        if isinstance(value, types.ModuleType):
            enqueue_module(value)
            return
        if isinstance(value, (tuple, list, set, frozenset)):
            identity = id(value)
            if identity in visited_values:
                return
            visited_values.add(identity)
            record_value(value, label)
            for index, child in enumerate(value):
                visit(child, f"{label}[{index}]")
            return
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise ValueError("runtime code identity mapping key is invalid")
            identity = id(value)
            if identity in visited_values:
                return
            visited_values.add(identity)
            record_value(value, label)
            for name, child in sorted(value.items()):
                visit(child, f"{label}[{name}]")
            return
        if not isinstance(value, (types.FunctionType, types.MethodType, types.CodeType, type,
                                  property, staticmethod, classmethod)) and not hasattr(value, "__wrapped__"):
            return
        identity = id(value)
        if identity in visited_values:
            return
        visited_values.add(identity)
        if isinstance(value, property):
            visit(value.fget, label + ".fget"); visit(value.fset, label + ".fset"); visit(value.fdel, label + ".fdel")
            return
        if isinstance(value, (staticmethod, classmethod)):
            visit(value.__func__, label + ".__func__")
            return
        if isinstance(value, types.MethodType):
            visit(value.__func__, label + ".__func__")
            return
        if isinstance(value, types.CodeType):
            record_value(value, label)
            return
        if isinstance(value, types.FunctionType):
            if not permitted_name(value.__module__):
                return
            canonical = f"{value.__module__}.{value.__qualname__}"
            record_value(value, canonical)
            enqueue_module(value)
            wrapped = getattr(value, "__wrapped__", None)
            if wrapped is not None:
                record_value(wrapped, canonical + ".__wrapped__"); visit(wrapped, canonical + ".__wrapped__")
            for index, cell in enumerate(value.__closure__ or ()):
                try:
                    captured = cell.cell_contents
                except ValueError:
                    continue
                record_value(captured, f"{canonical}.closure[{index}]"); visit(captured, f"{canonical}.closure[{index}]")
            for index, default in enumerate(value.__defaults__ or ()):
                record_value(default, f"{canonical}.default[{index}]"); visit(default, f"{canonical}.default[{index}]")
            for name, default in sorted((value.__kwdefaults__ or {}).items()):
                record_value(default, f"{canonical}.kwdefault[{name}]"); visit(default, f"{canonical}.kwdefault[{name}]")
            for name in value.__code__.co_names:
                if name not in value.__globals__:
                    continue
                global_value = value.__globals__[name]
                global_module = getattr(global_value, "__module__", "")
                if isinstance(global_value, types.ModuleType) and permitted_name(global_value.__name__):
                    visit(global_value, f"{canonical}.global[{name}]")
                elif permitted_name(global_module):
                    visit(global_value, f"{canonical}.global[{name}]")
                elif isinstance(global_value, (str, bytes, int, float, bool, type(None), tuple, list, dict, set, frozenset, Path)):
                    record_value(global_value, f"{canonical}.global[{name}]")
            return
        if isinstance(value, type):
            if not permitted_name(value.__module__):
                return
            canonical = f"{value.__module__}.{value.__qualname__}"
            enqueue_module(value)
            wrapped = getattr(value, "__wrapped__", None)
            if wrapped is not None:
                record_value(wrapped, canonical + ".__wrapped__"); visit(wrapped, canonical + ".__wrapped__")
            for base in value.__bases__:
                if permitted_name(getattr(base, "__module__", "")):
                    visit(base, canonical + ".base")
            for name, child in sorted(value.__dict__.items()):
                child_label = f"{canonical}.{name}"
                # Class data attributes can change property/method behavior
                # without appearing in a method's co_names (for example,
                # ``self.MODE``).  Bind supported immutable values as well as
                # recursively visiting their nested callable/container forms.
                if isinstance(child, (str, bytes, int, float, bool, type(None),
                                      tuple, list, dict, set, frozenset, Path)):
                    record_value(child, child_label)
                visit(child, child_label)
            return
        wrapped = getattr(value, "__wrapped__", None)
        if wrapped is not None:
            record_value(wrapped, label + ".__wrapped__"); visit(wrapped, label + ".__wrapped__")

    while pending:
        current = pending.pop()
        if current.__name__ in visited_modules:
            continue
        visited_modules.add(current.__name__)
        for name, value in sorted(current.__dict__.items()):
            visit(value, f"{current.__name__}.{name}")
    digest = hashlib.sha256(b"emender-e97-runtime-code-object-package-v3\0")
    for name, payload in sorted(entries):
        digest.update(name.encode("utf-8") + b"\0" + payload)
    return digest.hexdigest()


_FICLONE = 0x40049409
# A regular production checkpoint must use a COW reflink.  The copy path is
# deliberately limited to tiny local/CPU fixtures, never a model-scale escape.
_MAX_CHECKPOINT_COPY_FALLBACK_BYTES = 64 << 20


def _copy_descriptor(source_fd: int, destination_fd: int, *, maximum: int) -> None:
    """Copy one pinned regular source into a bounded service-owned file."""

    source_size = os.fstat(source_fd).st_size
    if source_size > maximum:
        raise ValueError("checkpoint copy fallback exceeds its configured bound")
    os.lseek(source_fd, 0, os.SEEK_SET)
    remaining = source_size
    while remaining:
        chunk = os.read(source_fd, min(1 << 20, remaining))
        if not chunk:
            raise ValueError("checkpoint changed while creating private snapshot")
        view = memoryview(chunk)
        while view:
            written = os.write(destination_fd, view)
            if written <= 0:  # pragma: no cover - regular-file write contract
                raise OSError("checkpoint private snapshot write failed")
            view = view[written:]
        remaining -= len(chunk)
    if os.read(source_fd, 1):
        raise ValueError("checkpoint changed while creating private snapshot")


@dataclass
class _PrivateCheckpointSnapshot:
    """Descriptor-relative ownership of a private snapshot directory."""

    parent_fd: int
    private_fd: int
    name: str

    def cleanup(self) -> None:
        if self.private_fd >= 0:
            try:
                os.close(self.private_fd)
            except OSError:
                pass
            self.private_fd = -1
        if self.parent_fd >= 0:
            try:
                os.rmdir(self.name, dir_fd=self.parent_fd)
                os.fsync(self.parent_fd)
            except OSError:
                pass
            try:
                os.close(self.parent_fd)
            except OSError:
                pass
            self.parent_fd = -1


def _private_checkpoint_snapshot(
    checkpoint: Path, *, copy_fallback_maximum: int = 0,
) -> tuple[_PrivateCheckpointSnapshot, int, str]:
    """Return an unlinked private checkpoint descriptor and its exact digest.

    A same-filesystem FICLONE COW snapshot is required by default.  A bounded
    byte-copy fallback is deliberately opt-in for tiny CPU/test artifacts; it
    never hashes or loads the independently writable source inode.
    """

    if (isinstance(copy_fallback_maximum, bool) or not isinstance(copy_fallback_maximum, int)
            or not 0 <= copy_fallback_maximum <= _MAX_CHECKPOINT_COPY_FALLBACK_BYTES):
        raise ValueError("checkpoint copy fallback bound is invalid")
    parent_fd = open_directory_no_follow(checkpoint.parent)
    source_fd: int | None = None
    private: _PrivateCheckpointSnapshot | None = None
    destination_fd: int | None = None
    try:
        source_fd = os.open(
            checkpoint.name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent_fd,
        )
        if not stat.S_ISREG(os.fstat(source_fd).st_mode):
            raise ValueError("identity artifact is not a regular file")
        for _ in range(32):
            name = f".e97-private-checkpoint-{secrets.token_hex(16)}"
            try:
                os.mkdir(name, mode=0o700, dir_fd=parent_fd)
            except FileExistsError:
                continue
            private_fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                                 dir_fd=parent_fd)
            private = _PrivateCheckpointSnapshot(parent_fd, private_fd, name)
            parent_fd = -1  # ownership transferred to ``private``
            break
        else:  # pragma: no cover - cryptographic collision is not practical
            raise RuntimeError("could not reserve private checkpoint directory")
        destination_fd = os.open(
            "checkpoint.pt", os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600, dir_fd=private.private_fd)
        try:
            fcntl.ioctl(destination_fd, _FICLONE, source_fd)
        except OSError as exc:
            if copy_fallback_maximum <= 0:
                raise ValueError("checkpoint reflink is unavailable and copy fallback is disabled") from exc
            if exc.errno not in {errno.EOPNOTSUPP, errno.ENOTTY, errno.EINVAL, errno.EXDEV, errno.ENOSYS}:
                raise ValueError("checkpoint reflink failed") from exc
            _copy_descriptor(source_fd, destination_fd, maximum=copy_fallback_maximum)
        os.fchmod(destination_fd, 0o400)
        os.fsync(destination_fd)
        checkpoint_sha256 = _sha256_and_rewind(destination_fd)
        # The service owns the open descriptor; remove its basename through
        # the pinned private directory before loading it.
        os.unlink("checkpoint.pt", dir_fd=private.private_fd)
        os.close(source_fd)
        source_fd = None
        return private, destination_fd, checkpoint_sha256
    except BaseException:
        if destination_fd is not None:
            os.close(destination_fd)
        if private is not None:
            try:
                os.unlink("checkpoint.pt", dir_fd=private.private_fd)
            except OSError:
                pass
            private.cleanup()
        raise
    finally:
        if source_fd is not None:
            os.close(source_fd)
        if parent_fd >= 0:
            os.close(parent_fd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True,
                        help="required lowercase SHA-256 pin for the exact private checkpoint snapshot")
    parser.add_argument("--args-json", type=Path, required=True)
    parser.add_argument("--model-id", default="e97-dense-agent")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8797)
    parser.add_argument("--api-key", default=os.environ.get("EMENDER_AGENT_API_KEY"))
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--max-sessions", type=int, default=8)
    parser.add_argument("--max-body-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--ingest-mode", choices=("tokenwise", "segment"), default="tokenwise")
    parser.add_argument(
        "--weight-mode",
        choices=("saved", "train"),
        default="saved",
        help="serve saved Schedule-Free x weights or reconstruct train/y weights",
    )
    system_group = parser.add_mutually_exclusive_group()
    system_group.add_argument("--v1-canonical-system", action="store_true")
    system_group.add_argument("--v2-canonical-system", action="store_true")
    system_group.add_argument("--cli-canonical-system", action="store_true")
    system_group.add_argument("--cli-direct-canonical-system", action="store_true")
    system_group.add_argument("--pi-core-canonical-system", action="store_true")
    system_group.add_argument("--pi-agent-v2-canonical-system", action="store_true")
    system_group.add_argument("--pi-agent-analysis-v1-canonical-system", action="store_true")
    parser.add_argument(
        "--private-analysis-protocol", action="store_true",
        help="require canonical Analysis JSON plus Action/Final and round-trip reasoning_content",
    )
    parser.add_argument("--trace-generated-errors", action="store_true")
    parser.add_argument(
        "--external-controller",
        action="store_true",
        help="delegate repeat-cycle policy to the trusted dedicated acquisition controller",
    )
    parser.add_argument("--runtime-image", type=Path, help="required immutable sandbox/runtime image artifact for external-controller mode")
    parser.add_argument("--runtime-image-sha256", help="expected SHA-256 for --runtime-image")
    parser.add_argument(
        "--checkpoint-copy-fallback-bytes", type=int, default=0,
        help="opt-in bounded private-copy fallback when same-filesystem FICLONE is unavailable (CPU/test only)",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    system_prompt_override = (
        DENSE_AGENT_V1_SYSTEM if args.v1_canonical_system else
        DENSE_AGENT_V2_SYSTEM if args.v2_canonical_system else
        DENSE_AGENT_CLI_DIRECT_SYSTEM if args.cli_direct_canonical_system else
        DENSE_AGENT_CLI_SYSTEM if args.cli_canonical_system else
        E97_PI_CORE_SYSTEM if args.pi_core_canonical_system else
        E97_PI_AGENT_SYSTEM_V2 if args.pi_agent_v2_canonical_system else
        E97_PI_AGENT_ANALYSIS_SYSTEM_V1 if args.pi_agent_analysis_v1_canonical_system else None
    )
    if args.pi_agent_analysis_v1_canonical_system:
        args.private_analysis_protocol = True
    if args.external_controller and system_prompt_override is not None:
        parser.error("--external-controller cannot be combined with a server-side canonical system override")
    if (not isinstance(args.checkpoint_sha256, str) or len(args.checkpoint_sha256) != 64
            or any(character not in "0123456789abcdef" for character in args.checkpoint_sha256)):
        parser.error("--checkpoint-sha256 must be a lowercase SHA-256 digest")
    if args.external_controller and (args.runtime_image is None
                                    or not isinstance(args.runtime_image_sha256, str) or len(args.runtime_image_sha256) != 64):
        parser.error("--external-controller requires --runtime-image and --runtime-image-sha256")
    if args.device == "cuda":
        torch.cuda.set_device(0)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    dtype_name = "bfloat16" if args.device == "cuda" else "float32"
    try:
        args_payload = _snapshot_small_identity(args.args_json, name="args-json")
        runtime_descriptor = _open_regular_no_follow(args.runtime_image or args.checkpoint)
        try:
            runtime_image_sha256 = _sha256_and_rewind(runtime_descriptor)
        finally:
            os.close(runtime_descriptor)
        if args.runtime_image_sha256 and runtime_image_sha256 != args.runtime_image_sha256:
            parser.error("--runtime-image SHA-256 does not match --runtime-image-sha256")
        # These digests bind the code objects already imported and executed by
        # this process.  They deliberately do not claim to hash later mutable
        # checkout source paths.
        import ndm.e97_acquisition_controller as controller_module
        import ndm.e97_agent_server as server_module
        # The controller field is the archived task-bundle source identity;
        # the server field is the loaded package code-object closure that
        # actually executes this process (and includes that controller module).
        controller_build_sha256 = _archived_controller_source_sha256()
        server_build_sha256 = _loaded_module_code_sha256(
            sys.modules[__name__], server_module, controller_module)
        checkpoint_temporary, checkpoint_descriptor, checkpoint_sha256 = _private_checkpoint_snapshot(
            args.checkpoint, copy_fallback_maximum=args.checkpoint_copy_fallback_bytes)
        if checkpoint_sha256 != args.checkpoint_sha256:
            os.close(checkpoint_descriptor)
            checkpoint_temporary.cleanup()
            raise ValueError("--checkpoint-sha256 does not match the private checkpoint snapshot")
    except ValueError as exc:
        parser.error(str(exc))
    try:
        # ``load_e97_checkpoint`` consumes the same unlinked service-owned
        # snapshot that was hashed above.  The private args copy exists only
        # for its duration and comes from one retained no-follow snapshot.
        with tempfile.TemporaryDirectory(prefix="e97-serve-identity-") as temporary:
            private_args = Path(temporary) / "args.json"
            private_args.write_bytes(args_payload)
            with os.fdopen(checkpoint_descriptor, "rb", closefd=False) as checkpoint_stream:
                loaded = load_e97_checkpoint(
                    checkpoint_stream,
                    checkpoint_path=args.checkpoint,
                    args_json=private_args,
                    device=args.device,
                    dtype=dtype,
                    weight_mode=args.weight_mode,
                    use_triton=args.device == "cuda",
                    mmap=False,
                )
    finally:
        os.close(checkpoint_descriptor)
        checkpoint_temporary.cleanup()
    engine = TorchE97AgentEngine(loaded, ingest_mode=args.ingest_mode, weight_mode=args.weight_mode,
                                 device=args.device, dtype=dtype_name, use_triton=args.device == "cuda",
                                 checkpoint_sha256=checkpoint_sha256,
                                 private_analysis=args.private_analysis_protocol)
    # These values are snapshots consumed at startup, not pathnames the service
    # is permitted to reopen while constructing an attestation.
    engine.args_json_sha256 = hashlib.sha256(args_payload).hexdigest()
    engine.runtime_image_sha256 = runtime_image_sha256
    engine.controller_build_sha256 = controller_build_sha256
    engine.server_build_sha256 = server_build_sha256
    service = AgentCompletionService(
        engine,
        model_id=args.model_id,
        max_output_tokens=args.max_output_tokens,
        max_sessions=args.max_sessions,
        trace_generated_errors=args.trace_generated_errors,
        system_prompt_override=system_prompt_override,
        require_tool_call=args.v2_canonical_system or args.cli_canonical_system or args.cli_direct_canonical_system,
        external_controller=args.external_controller,
        checkpoint_sha256=checkpoint_sha256, checkpoint_path=str(args.checkpoint),
        args_json_path=str(args.args_json), args_json_sha256=hashlib.sha256(args_payload).hexdigest(), config_sha256=hashlib.sha256(json.dumps(loaded.config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest(),
        weight_mode=args.weight_mode, tokenizer=str(loaded.config["tokenizer"]),
        device=args.device, dtype=dtype_name,
        use_triton=args.device == "cuda", ingest_mode=args.ingest_mode,
        runtime_image_path=str(args.runtime_image or args.checkpoint), runtime_image_sha256=runtime_image_sha256,
        tool_schema_sha256=hashlib.sha256(json.dumps(READ_OBSERVE_TOOLS, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest(),
        controller_build_sha256=controller_build_sha256,
        server_build_sha256=server_build_sha256,
        private_analysis=args.private_analysis_protocol,
    )
    print(
        f"serving model={args.model_id} checkpoint={loaded.checkpoint_path} "
        f"address={args.host}:{args.port} max_sessions={args.max_sessions} "
        f"ingest_mode={args.ingest_mode} external_controller={args.external_controller} "
        f"private_analysis={args.private_analysis_protocol}",
        flush=True,
    )
    run_openai_server(
        service,
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        max_body_bytes=args.max_body_bytes,
    )


if __name__ == "__main__":
    main()
