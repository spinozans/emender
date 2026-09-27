"""Deterministic, checkout-verifiable source archives for first-party generators."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import io
import json
import marshal
import os
from pathlib import Path
import re
import sys
import tarfile
import types
from typing import Any, Mapping

from ndm.e97_atomic import publish_bytes_atomically, read_regular_file_no_follow

_SOURCE_MANIFEST_SCHEMA = "emender-e97-firstparty-generator-manifest-v3"
_MAX_MANIFEST_BYTES = 8 << 20
_MAX_COMPONENT_BYTES = 64 << 20
_MAX_ARCHIVE_BYTES = 256 << 20

# This is deliberately a source-closure allowlist, not a manifest assertion.
# It covers only first-party generation, protected-overlap checking, replay, and
# validation plus their local dependencies.  Correction packing, leases, and
# artifact-store code are separate authorities and must not silently enter this
# generator archive.
EXPECTED_GENERATOR_COMPONENT_PATHS = frozenset({
    "configs/pi/e97-firstparty-authorization-license-v1.json",
    "configs/pi/e97-firstparty-cpu-environment-v1.json",
    "configs/pi/e97-firstparty-overlap-firewall-audit-v1.json",
    "ndm/e97_acquisition_controller.py",
    "ndm/e97_agent_protocol.py",
    "ndm/e97_atomic.py",
    "ndm/e97_first_party_read_observe.py",
    "ndm/e97_first_party_source_archive.py",
    "ndm/e97_onpolicy_records.py",
    "ndm/e97_protected_overlap.py",
    "ndm/e97_task_lake.py",
    "scripts/build_e97_first_party_source_archive.py",
    "scripts/check_e97_protected_overlap.py",
    "scripts/e97_first_party_validator.py",
    "scripts/validate_e97_first_party_task.py",
})


@dataclass(frozen=True)
class _CheckoutSnapshot:
    """One retained manifest and complete component-byte authority snapshot."""

    manifest_payload: bytes
    components: tuple[tuple[dict[str, str], bytes], ...]

    @property
    def manifest_sha256(self) -> str:
        return sha256_bytes(self.manifest_payload)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _snapshot_file(path: Path, *, name: str, maximum: int) -> bytes:
    """Read one bounded authority file once, without following any component."""

    try:
        return read_regular_file_no_follow(path, maximum=maximum)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"{name} is unreadable") from exc


def sha256_path(path: Path) -> str:
    """Hash one bounded no-follow source authority file."""

    return sha256_bytes(_snapshot_file(path, name="source file", maximum=_MAX_COMPONENT_BYTES))


def _manifest_components_payload(payload: bytes) -> list[dict[str, str]]:
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("source component manifest is unreadable") from exc
    if not isinstance(manifest, Mapping) or manifest.get("schema") != _SOURCE_MANIFEST_SCHEMA:
        raise ValueError("source component manifest schema is invalid")
    components = manifest.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("source component manifest must contain components")
    normalized: list[dict[str, str]] = []
    names: set[str] = set()
    for raw in components:
        if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256"}:
            raise ValueError("source component is invalid")
        name, digest = raw["path"], raw["sha256"]
        path = Path(name) if isinstance(name, str) else None
        if (not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or path is None or path.is_absolute() or ".." in path.parts
                or name in names or name != path.as_posix()):
            raise ValueError("source component path or SHA-256 is invalid")
        names.add(name)
        normalized.append({"path": name, "sha256": digest})
    if normalized != sorted(normalized, key=lambda component: component["path"]):
        raise ValueError("source component manifest must be path-sorted")
    if {component["path"] for component in normalized} != EXPECTED_GENERATOR_COMPONENT_PATHS:
        raise ValueError("source component manifest does not match the closed generator authority")
    return normalized


def _manifest_components(manifest_path: Path) -> list[dict[str, str]]:
    """Read and validate one retained component manifest snapshot."""

    return _manifest_components_payload(
        _snapshot_file(manifest_path, name="source component manifest", maximum=_MAX_MANIFEST_BYTES))


def _snapshot_checkout_components(manifest_path: Path, *, checkout_root: Path) -> _CheckoutSnapshot:
    """Retain manifest and every closed checkout component before any comparison.

    Each file is opened once through descriptor-relative no-follow traversal.
    Later archive construction and verification consume these exact retained
    bytes, so a pathname substitution cannot split recorded hashes from tar
    members.
    """

    manifest_payload = _snapshot_file(
        manifest_path, name="source component manifest", maximum=_MAX_MANIFEST_BYTES)
    components = _manifest_components_payload(manifest_payload)
    retained: list[tuple[dict[str, str], bytes]] = []
    for component in components:
        payload = _snapshot_file(
            checkout_root / component["path"],
            name=f"source component {component['path']}",
            maximum=_MAX_COMPONENT_BYTES,
        )
        if sha256_bytes(payload) != component["sha256"]:
            raise ValueError(f"source component bytes changed: {component['path']}")
        retained.append((component, payload))
    return _CheckoutSnapshot(manifest_payload, tuple(retained))


def verify_checkout_components(manifest_path: Path, *, checkout_root: Path) -> list[dict[str, str]]:
    """Verify the closed component list against one retained checkout snapshot."""

    return [dict(component) for component, _payload in _snapshot_checkout_components(
        manifest_path, checkout_root=checkout_root).components]


def _canonical_ustar_bytes(components: list[dict[str, str]], payloads: Mapping[str, bytes]) -> bytes:
    """Build the one permitted byte encoding for the closed source archive."""

    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for component in components:
            name = component["path"]
            payload = payloads[name]
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.type = tarfile.REGTYPE
            archive.addfile(info, io.BytesIO(payload))
    return stream.getvalue()


def _verify_archive_payload(
    manifest_payload: bytes,
    archive_payload: bytes,
    *,
    checkout_snapshot: _CheckoutSnapshot | None = None,
) -> str:
    """Verify retained archive bytes against the exact canonical USTAR closure."""

    components = _manifest_components_payload(manifest_payload)
    expected = {component["path"]: component["sha256"] for component in components}
    checkout_payloads = (
        {component["path"]: payload for component, payload in checkout_snapshot.components}
        if checkout_snapshot is not None else None
    )
    observed_payloads: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_payload), mode="r:") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if names != [component["path"] for component in components]:
                raise ValueError("source archive members are not exactly manifest components")
            for member in members:
                if (member.type != tarfile.REGTYPE or member.mode != 0o644 or member.uid != 0 or member.gid != 0
                        or member.uname or member.gname or member.mtime != 0):
                    raise ValueError("source archive member metadata is not normalized")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("source archive regular member is unreadable")
                payload = stream.read()
                if sha256_bytes(payload) != expected[member.name]:
                    raise ValueError("source archive member digest mismatch")
                if checkout_payloads is not None and payload != checkout_payloads[member.name]:
                    raise ValueError("source archive member differs from checkout")
                observed_payloads[member.name] = payload
    except tarfile.TarError as exc:
        raise ValueError("source archive is invalid") from exc
    canonical_payload = _canonical_ustar_bytes(components, checkout_payloads or observed_payloads)
    if archive_payload != canonical_payload:
        raise ValueError("source archive bytes are not the canonical USTAR encoding")
    return sha256_bytes(archive_payload)


def build_source_archive(manifest_path: Path, output: Path, *, checkout_root: Path) -> str:
    """Build, byte-verify, then atomically publish one retained canonical USTAR."""

    snapshot = _snapshot_checkout_components(manifest_path, checkout_root=checkout_root)
    components = [component for component, _payload in snapshot.components]
    payloads = {component["path"]: payload for component, payload in snapshot.components}
    archive_payload = _canonical_ustar_bytes(components, payloads)
    digest = _verify_archive_payload(
        snapshot.manifest_payload, archive_payload, checkout_snapshot=snapshot)
    publish_bytes_atomically(output, archive_payload)
    return digest


def verify_archive_members(manifest_path: Path, archive_path: Path) -> str:
    """Verify one manifest and archive snapshot without reopening either pathname."""

    manifest_payload = _snapshot_file(
        manifest_path, name="source component manifest", maximum=_MAX_MANIFEST_BYTES)
    archive_payload = _snapshot_file(archive_path, name="source archive", maximum=_MAX_ARCHIVE_BYTES)
    return _verify_archive_payload(manifest_payload, archive_payload)


def verified_source_archive_members(
    manifest_payload: bytes, archive_payload: bytes,
) -> dict[str, bytes]:
    """Return every verified canonical archive member from retained authority."""

    components = _manifest_components_payload(manifest_payload)
    _verify_archive_payload(manifest_payload, archive_payload)
    payloads: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_payload), mode="r:") as archive:
            for component in components:
                member = archive.getmember(component["path"])
                if not member.isreg() or member.issym() or member.islnk():
                    raise ValueError("source archive member is not a regular file")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("source archive member is unreadable")
                payload = stream.read()
                if sha256_bytes(payload) != component["sha256"]:
                    raise ValueError("source archive member digest mismatch")
                payloads[component["path"]] = payload
    except (KeyError, tarfile.TarError) as exc:
        raise ValueError("source archive is invalid") from exc
    return payloads


def _semantic_code_digest(code: types.CodeType) -> bytes:
    """Canonicalize executable code while deliberately excluding source paths."""

    digest = hashlib.sha256()

    def add(tag: bytes, value: bytes) -> None:
        digest.update(tag)
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)

    def add_constant(value: object) -> None:
        if isinstance(value, types.CodeType):
            add(b"code", _semantic_code_digest(value))
        elif isinstance(value, tuple):
            add(b"tuple", len(value).to_bytes(8, "big"))
            for item in value:
                add_constant(item)
        elif isinstance(value, frozenset):
            encoded = sorted(marshal.dumps(item) for item in value)
            add(b"frozenset", len(encoded).to_bytes(8, "big"))
            for item in encoded:
                add(b"item", item)
        else:
            # CPython compiler constants are primitives accepted by marshal;
            # rejecting an unexpected value is safer than silently omitting it.
            try:
                add(b"constant", marshal.dumps(value))
            except (TypeError, ValueError) as exc:  # pragma: no cover - compiler contract
                raise ValueError("unsupported code constant in semantic digest") from exc

    add(b"code", code.co_code)
    add(b"exception", code.co_exceptiontable)
    for name in ("co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_nlocals", "co_stacksize", "co_flags"):
        add(name.encode("ascii"), str(getattr(code, name)).encode("ascii"))
    for name, values in (("names", code.co_names), ("varnames", code.co_varnames),
                         ("freevars", code.co_freevars), ("cellvars", code.co_cellvars)):
        add(name.encode("ascii"), "\0".join(values).encode("utf-8"))
    for value in code.co_consts:
        add_constant(value)
    return digest.digest()


def _source_direct_callable_names(source: bytes, *, path: str) -> list[str]:
    """Return every explicitly source-defined module/class callable name."""

    try:
        tree = ast.parse(source.decode("utf-8"), filename=f"<verified-archive:{path}>", mode="exec")
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ValueError("verified source archive Python member is not parsable") from exc
    names: set[str] = set()

    def visit(statements: list[ast.stmt], prefix: str = "") -> None:
        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(prefix + statement.name)
            elif isinstance(statement, ast.ClassDef):
                visit(statement.body, prefix + statement.name + ".")

    visit(tree.body)
    return sorted(names)


def _source_class_names(source: bytes, *, path: str) -> list[str]:
    """Return every source-defined class so generated descriptors are bound too."""

    try:
        tree = ast.parse(source.decode("utf-8"), filename=f"<verified-archive:{path}>", mode="exec")
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ValueError("verified source archive Python member is not parsable") from exc
    names: set[str] = set()

    def visit(statements: list[ast.stmt], prefix: str = "") -> None:
        for statement in statements:
            if isinstance(statement, ast.ClassDef):
                name = prefix + statement.name
                names.add(name)
                visit(statement.body, name + ".")

    visit(tree.body)
    return sorted(names)


def _logical_module_name(module_name: object, module_aliases: Mapping[str, str]) -> str:
    if not isinstance(module_name, str):
        return "<unknown-module>"
    return module_aliases.get(module_name, module_name)


def _semantic_value_digest(
    value: object,
    *,
    active: set[int] | None = None,
    module_aliases: Mapping[str, str] | None = None,
    first_party_modules: frozenset[str] = frozenset(),
) -> bytes:
    """Digest runtime values, recursively binding reachable first-party behavior.

    The archived module graph is executed under isolated module names.  The
    alias map makes its symbolic identities equal to their already-loaded
    counterparts while the recursive cases bind class descriptors, imported
    aliases, defaults, closures, and immutable containers.  A cycle records a
    stable symbolic edge rather than a process-local object address.
    """

    active = set() if active is None else active
    module_aliases = {} if module_aliases is None else module_aliases
    digest = hashlib.sha256()

    def add(tag: str, payload: bytes) -> None:
        encoded = tag.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big")); digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big")); digest.update(payload)

    def is_first_party(item: object) -> bool:
        return _logical_module_name(getattr(item, "__module__", None), module_aliases) in first_party_modules

    def cycle_label(item: object) -> bytes:
        module_name = _logical_module_name(getattr(item, "__module__", None), module_aliases)
        qualname = getattr(item, "__qualname__", type(item).__qualname__)
        return f"{module_name}.{qualname}".encode("utf-8")

    def nested(item: object) -> bytes:
        return _semantic_value_digest(
            item, active=active, module_aliases=module_aliases,
            first_party_modules=first_party_modules)

    def visit_sequence(tag: str, values: object) -> None:
        identifier = id(values)
        if identifier in active:
            add(f"{tag}-cycle", cycle_label(values))
            return
        active.add(identifier)
        try:
            encoded = bytearray()
            for member in values:  # type: ignore[union-attr]
                value_digest = nested(member)
                encoded.extend(len(value_digest).to_bytes(8, "big")); encoded.extend(value_digest)
            add(tag, bytes(encoded))
        finally:
            active.remove(identifier)

    def visit_type(item: type) -> None:
        module_name = _logical_module_name(item.__module__, module_aliases)
        if module_name not in first_party_modules:
            add("external-type", f"{module_name}.{item.__qualname__}".encode("utf-8"))
            return
        identifier = id(item)
        if identifier in active:
            add("type-cycle", cycle_label(item))
            return
        active.add(identifier)
        try:
            encoded = bytearray(f"{module_name}.{item.__qualname__}".encode("utf-8"))
            for name, member in sorted(item.__dict__.items()):
                # These interpreter/dataclass reflection fields are not
                # executable descriptors.  Generated __init__/repr/eq and all
                # other callable descriptors remain included below.
                if name in {"__module__", "__doc__", "__dict__", "__weakref__",
                            "__dataclass_fields__", "__dataclass_params__", "__annotations__"}:
                    continue
                member_digest = nested(member)
                name_bytes = name.encode("utf-8")
                encoded.extend(len(name_bytes).to_bytes(4, "big")); encoded.extend(name_bytes)
                encoded.extend(member_digest)
            add("first-party-type", bytes(encoded))
        finally:
            active.remove(identifier)

    def visit_module(item: types.ModuleType) -> None:
        module_name = _logical_module_name(item.__name__, module_aliases)
        if module_name not in first_party_modules:
            add("external-module", module_name.encode("utf-8"))
            return
        identifier = id(item)
        if identifier in active:
            add("module-cycle", module_name.encode("utf-8"))
            return
        active.add(identifier)
        try:
            encoded = bytearray(module_name.encode("utf-8"))
            for name, member in sorted(item.__dict__.items()):
                if name in {"__builtins__", "__cached__", "__file__", "__loader__", "__name__",
                            "__package__", "__path__", "__spec__"}:
                    continue
                member_digest = nested(member)
                name_bytes = name.encode("utf-8")
                encoded.extend(len(name_bytes).to_bytes(4, "big")); encoded.extend(name_bytes)
                encoded.extend(member_digest)
            add("first-party-module", bytes(encoded))
        finally:
            active.remove(identifier)

    def visit(item: object) -> None:
        if item is None:
            add("none", b"")
        elif isinstance(item, bool):
            add("bool", b"1" if item else b"0")
        elif isinstance(item, int):
            add("int", str(item).encode("ascii"))
        elif isinstance(item, float):
            if item != item or item in (float("inf"), float("-inf")):
                raise ValueError("loaded callable semantic value is non-finite")
            add("float", item.hex().encode("ascii"))
        elif isinstance(item, str):
            add("str", item.encode("utf-8"))
        elif isinstance(item, bytes):
            add("bytes", item)
        elif isinstance(item, Path):
            add("path", str(item).encode("utf-8"))
        elif isinstance(item, types.CodeType):
            add("code", _semantic_code_digest(item))
        elif isinstance(item, types.ModuleType):
            visit_module(item)
        elif isinstance(item, (types.BuiltinFunctionType, types.BuiltinMethodType)):
            add("builtin", f"{item.__module__}.{item.__qualname__}".encode("utf-8"))
        elif isinstance(item, types.FunctionType):
            if not is_first_party(item):
                add("external-function", cycle_label(item))
                return
            identifier = id(item)
            if identifier in active:
                add("function-cycle", cycle_label(item))
                return
            active.add(identifier)
            try:
                add("first-party-function", _semantic_function_digest(
                    item, active=active, module_aliases=module_aliases,
                    first_party_modules=first_party_modules))
            finally:
                active.remove(identifier)
        elif isinstance(item, types.MethodType):
            visit(item.__func__)
        elif isinstance(item, type):
            visit_type(item)
        elif isinstance(item, property):
            visit_sequence("property", (item.fget, item.fset, item.fdel))
        elif isinstance(item, (staticmethod, classmethod)):
            visit(item.__func__)
        elif isinstance(item, tuple):
            visit_sequence("tuple", item)
        elif isinstance(item, list):
            visit_sequence("list", item)
        elif isinstance(item, (set, frozenset)):
            identifier = id(item)
            if identifier in active:
                add(f"{type(item).__name__}-cycle", cycle_label(item))
                return
            active.add(identifier)
            try:
                add(type(item).__name__, b"".join(sorted(nested(member) for member in item)))
            finally:
                active.remove(identifier)
        elif isinstance(item, re.Pattern):
            # Compiled regular expressions are immutable executable globals;
            # their pattern and flags determine every match decision.
            add("regex", item.pattern.encode("utf-8") + b"\0" + str(item.flags).encode("ascii"))
        elif isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise ValueError("loaded callable semantic mapping key is invalid")
            identifier = id(item)
            if identifier in active:
                add("dict-cycle", cycle_label(item))
                return
            active.add(identifier)
            try:
                encoded = bytearray()
                for key in sorted(item):
                    key_bytes, value_digest = key.encode("utf-8"), nested(item[key])
                    encoded.extend(len(key_bytes).to_bytes(4, "big")); encoded.extend(key_bytes)
                    encoded.extend(value_digest)
                add("dict", bytes(encoded))
            finally:
                active.remove(identifier)
        else:
            # Imported sentinels do not carry first-party behavior.  Bind their
            # stable symbolic identity rather than repr(), which can leak an
            # address.  Objects with a wrapper remain behavior-bearing.
            wrapped = getattr(item, "__wrapped__", None)
            if wrapped is not None:
                visit_sequence("wrapped", (type(item), wrapped))
                return
            module, qualname = getattr(item, "__module__", None), getattr(item, "__qualname__", None)
            if isinstance(module, str) and isinstance(qualname, str):
                add("external-symbol", f"{module}.{qualname}".encode("utf-8"))
            else:
                add("external-instance", f"{type(item).__module__}.{type(item).__qualname__}".encode("utf-8"))

    visit(value)
    return digest.digest()


def _semantic_function_digest(
    function: types.FunctionType,
    *,
    active: set[int] | None = None,
    module_aliases: Mapping[str, str] | None = None,
    first_party_modules: frozenset[str] = frozenset(),
) -> bytes:
    """Digest code, defaults, closures, and every runtime-resolved global."""

    active = set() if active is None else active
    module_aliases = {} if module_aliases is None else module_aliases
    digest = hashlib.sha256()
    digest.update(b"emender-e97-function-semantics-v3\0")
    digest.update(_semantic_code_digest(function.__code__))

    def value_digest(value: object) -> bytes:
        return _semantic_value_digest(
            value, active=active, module_aliases=module_aliases,
            first_party_modules=first_party_modules)

    for index, value in enumerate(function.__defaults__ or ()):
        digest.update(b"default\0" + str(index).encode("ascii") + b"\0" + value_digest(value))
    for name, value in sorted((function.__kwdefaults__ or {}).items()):
        digest.update(b"kwdefault\0" + name.encode("utf-8") + b"\0" + value_digest(value))
    for index, cell in enumerate(function.__closure__ or ()):
        try:
            value = cell.cell_contents
        except ValueError:
            digest.update(b"closure-empty\0" + str(index).encode("ascii"))
        else:
            digest.update(b"closure\0" + str(index).encode("ascii") + b"\0" + value_digest(value))
    # ``co_names`` is the interpreter's exact set of names a code object can
    # resolve through globals.  Bind only names actually present in globals;
    # builtins remain Python/runtime identity, not a mutable module alias.
    for name in sorted(set(function.__code__.co_names) & set(function.__globals__)):
        value = function.__globals__[name]
        # Isolated archived modules necessarily have a temporary __name__, but
        # code reading it observes the logical source module identity.  Bind
        # that logical value, not the materialization-private alias.
        if name == "__name__" and isinstance(value, str):
            value = module_aliases.get(value, value)
        digest.update(b"global\0" + name.encode("utf-8") + b"\0" + value_digest(value))
    return digest.digest()


def _callable_by_qualname(module: types.ModuleType, qualname: str) -> types.FunctionType:
    """Resolve one source-defined function/property/class descriptor exactly."""

    current: object = module
    for part in qualname.split("."):
        if not isinstance(current, (types.ModuleType, type)):
            raise ValueError("verified source archive callable closure is incomplete")
        namespace = current.__dict__
        if part not in namespace:
            raise ValueError("verified source archive callable closure is incomplete")
        current = namespace[part]
    if isinstance(current, property):
        current = current.fget
    elif isinstance(current, (staticmethod, classmethod)):
        current = current.__func__
    if not isinstance(current, types.FunctionType):
        raise ValueError("verified source archive callable closure is incomplete")
    return current


def _class_by_qualname(module: types.ModuleType, qualname: str) -> type:
    """Resolve one source-defined class, including its generated descriptors."""

    current: object = module
    for part in qualname.split("."):
        if not isinstance(current, (types.ModuleType, type)) or part not in current.__dict__:
            raise ValueError("verified source archive class closure is incomplete")
        current = current.__dict__[part]
    if not isinstance(current, type):
        raise ValueError("verified source archive class closure is incomplete")
    return current


def _archive_module_graph(
    source_modules: Mapping[str, tuple[str, bytes]], closure: set[str],
) -> tuple[dict[str, types.ModuleType], dict[str, str]]:
    """Materialize one isolated archived graph with its own first-party imports.

    Archive modules are temporarily installed under their real import names
    only while their retained source is executed.  Their function globals then
    retain imported archived aliases after the process module table is restored.
    This prevents an expected-side function from accidentally resolving a
    mutable checkout module during its semantic comparison.
    """

    ordered: list[str] = []
    visited: set[str] = set()

    def visit(module_name: str) -> None:
        if module_name in visited:
            return
        visited.add(module_name)
        _path, source = source_modules[module_name]
        for dependency in sorted(_archive_imports(
                source, module_name=module_name, available_modules=set(source_modules)) & closure):
            visit(dependency)
        ordered.append(module_name)

    for module_name in sorted(closure):
        visit(module_name)
    loaded = {name: sys.modules.get(name) for name in ordered}
    if not all(isinstance(module, types.ModuleType) for module in loaded.values()):
        raise ValueError("required replay controller module is not loaded")

    archived: dict[str, types.ModuleType] = {}
    aliases: dict[str, str] = {name: name for name in ordered}
    restore_modules: list[tuple[str, object]] = []
    restore_attributes: list[tuple[types.ModuleType, str, object]] = []
    missing = object()
    try:
        for module_name in ordered:
            path, source = source_modules[module_name]
            temporary_name = "_emender_verified_archive_" + hashlib.sha256(
                module_name.encode("utf-8") + b"\0" + source).hexdigest()
            module = types.ModuleType(temporary_name)
            module.__file__ = str(getattr(loaded[module_name], "__file__", f"<verified-archive:{path}>"))
            module.__package__ = module_name.rpartition(".")[0]
            aliases[temporary_name] = module_name
            sys.modules[temporary_name] = module
            restore_modules.append((temporary_name, missing))
            restore_modules.append((module_name, sys.modules.get(module_name, missing)))
            sys.modules[module_name] = module
            parent_name, _, child_name = module_name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if not isinstance(parent, types.ModuleType):  # pragma: no cover - ndm is already imported
                raise ValueError("verified source archive package is unavailable")
            prior = getattr(parent, child_name, missing)
            restore_attributes.append((parent, child_name, prior))
            setattr(parent, child_name, module)
            try:
                exec(compile(source, f"<verified-archive:{path}>", "exec", dont_inherit=True), module.__dict__)
            except BaseException as exc:
                raise ValueError("verified source archive Python member cannot materialize callable semantics") from exc
            archived[module_name] = module
    finally:
        for parent, child_name, prior in reversed(restore_attributes):
            if prior is missing:
                delattr(parent, child_name)
            else:
                setattr(parent, child_name, prior)
        for name, prior in reversed(restore_modules):
            if prior is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior  # type: ignore[assignment]
    return archived, aliases


def _archive_direct_function_codes(
    archived: types.ModuleType,
    source: bytes,
    *,
    module_name: str,
    path: str,
    module_aliases: Mapping[str, str],
    first_party_modules: frozenset[str],
) -> dict[str, bytes]:
    """Return retained function semantics with archived imported aliases bound."""

    return {
        f"{module_name}.{qualname}": _semantic_function_digest(
            _callable_by_qualname(archived, qualname), module_aliases=module_aliases,
            first_party_modules=first_party_modules)
        for qualname in _source_direct_callable_names(source, path=path)
    }


def _loaded_direct_function_codes(
    module: types.ModuleType,
    *,
    expected_names: list[str],
    first_party_modules: frozenset[str],
) -> dict[str, bytes]:
    """Read the exact loaded counterpart of each retained source callable."""

    result: dict[str, bytes] = {}
    for qualname in expected_names:
        function = _callable_by_qualname(module, qualname)
        if function.__module__ != module.__name__:
            raise ValueError(
                f"loaded replay controller code does not match verified archived source: {module.__name__}.{qualname}")
        result[f"{module.__name__}.{qualname}"] = _semantic_function_digest(
            function, first_party_modules=first_party_modules)
    return result


def _archive_class_codes(
    archived: types.ModuleType,
    source: bytes,
    *,
    module_name: str,
    path: str,
    module_aliases: Mapping[str, str],
    first_party_modules: frozenset[str],
) -> dict[str, bytes]:
    return {
        f"{module_name}.<class>.{qualname}": _semantic_value_digest(
            _class_by_qualname(archived, qualname), module_aliases=module_aliases,
            first_party_modules=first_party_modules)
        for qualname in _source_class_names(source, path=path)
    }


def _loaded_class_codes(
    module: types.ModuleType,
    *,
    source: bytes,
    path: str,
    first_party_modules: frozenset[str],
) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for qualname in _source_class_names(source, path=path):
        cls = _class_by_qualname(module, qualname)
        if cls.__module__ != module.__name__:
            raise ValueError(
                f"loaded replay controller class does not match verified archived source: {module.__name__}.{qualname}")
        result[f"{module.__name__}.<class>.{qualname}"] = _semantic_value_digest(
            cls, first_party_modules=first_party_modules)
    return result


def _archive_imports(source: bytes, *, module_name: str, available_modules: set[str]) -> set[str]:
    """Resolve archived first-party imports without importing from checkout paths."""

    try:
        tree = ast.parse(source.decode("utf-8"), filename=f"<verified-archive:{module_name}>", mode="exec")
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ValueError("verified source archive Python member is not parsable") from exc
    imports: set[str] = set()
    package = module_name.rpartition(".")[0]
    for statement in ast.walk(tree):
        if isinstance(statement, ast.Import):
            imports.update(alias.name for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom):
            if statement.level:
                base = package.split(".")
                if statement.level > len(base) + 1:
                    continue
                parent = ".".join(base[:len(base) - statement.level + 1])
                target = ".".join(part for part in (parent, statement.module or "") if part)
            else:
                target = statement.module or ""
            if target:
                imports.add(target)
    # ``from package import member`` may name either a module or a symbol.  A
    # source closure includes only exact archived module names.
    return {name for name in imports if name in available_modules}


def verified_loaded_module_closure_sha256(
    manifest_payload: bytes, archive_payload: bytes, *, root_module: str,
) -> str:
    """Bind loaded replay behavior to an isolated graph compiled from archive bytes.

    In addition to every explicitly declared callable, this comparison binds
    each source-defined class's complete descriptor set (including dataclass
    generated methods) and every global named by executable code.  Imported
    first-party aliases are resolved only in the isolated archived graph on the
    expected side; no checkout pathname is read after process import.
    """

    members = verified_source_archive_members(manifest_payload, archive_payload)
    source_modules = {
        path.removesuffix(".py").replace("/", "."): (path, payload)
        for path, payload in members.items()
        if path.startswith("ndm/") and path.endswith(".py")
    }
    if root_module not in source_modules:
        raise ValueError("verified source archive lacks requested loaded-code root")
    closure: set[str] = set()
    pending = [root_module]
    while pending:
        module_name = pending.pop()
        if module_name in closure:
            continue
        try:
            _path, source = source_modules[module_name]
        except KeyError as exc:  # pragma: no cover - guarded at ingress
            raise ValueError("verified source archive closure is incomplete") from exc
        closure.add(module_name)
        pending.extend(sorted(_archive_imports(
            source, module_name=module_name, available_modules=set(source_modules)) - closure,
                              reverse=True))
    first_party_modules = frozenset(closure)
    archived_modules, module_aliases = _archive_module_graph(source_modules, closure)
    expected: dict[str, bytes] = {}
    actual: dict[str, bytes] = {}
    for module_name in sorted(closure):
        path, source = source_modules[module_name]
        module = sys.modules.get(module_name)
        if not isinstance(module, types.ModuleType):
            raise ValueError("required replay controller module is not loaded")
        names = _source_direct_callable_names(source, path=path)
        expected.update(_archive_direct_function_codes(
            archived_modules[module_name], source, module_name=module_name, path=path,
            module_aliases=module_aliases, first_party_modules=first_party_modules))
        actual.update(_loaded_direct_function_codes(
            module, expected_names=names, first_party_modules=first_party_modules))
        expected.update(_archive_class_codes(
            archived_modules[module_name], source, module_name=module_name, path=path,
            module_aliases=module_aliases, first_party_modules=first_party_modules))
        actual.update(_loaded_class_codes(
            module, source=source, path=path, first_party_modules=first_party_modules))
    if set(actual) != set(expected):
        raise ValueError("loaded replay controller code does not match verified archived source")
    for name, expected_digest in expected.items():
        if actual[name] != expected_digest:
            raise ValueError("loaded replay controller code does not match verified archived source")
    digest = hashlib.sha256(b"emender-e97-verified-loaded-controller-closure-v2\0")
    for name in sorted(expected):
        digest.update(name.encode("utf-8") + b"\0" + expected[name])
    return digest.hexdigest()


def verified_source_member_payload(
    manifest_payload: bytes, archive_payload: bytes, member_path: str,
) -> tuple[bytes, str]:
    """Return one member from already-retained, canonical archive authority.

    Callers that already own source snapshots use this instead of extracting to
    a pathname and reopening it.  Canonical verification happens before the
    member is returned, so the bytes and digest are necessarily from the same
    source-archive snapshot as every other component.
    """

    components = {component["path"]: component["sha256"]
                  for component in _manifest_components_payload(manifest_payload)}
    _verify_archive_payload(manifest_payload, archive_payload)
    expected = components.get(member_path)
    if expected is None:
        raise ValueError("requested source member is not in the closed manifest")
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_payload), mode="r:") as archive:
            member = archive.getmember(member_path)
            if not member.isreg() or member.issym() or member.islnk():
                raise ValueError("requested source member is not a regular archive member")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("requested source member is unreadable")
            payload = stream.read()
    except (KeyError, tarfile.TarError) as exc:
        raise ValueError("requested source member is absent from archive") from exc
    if sha256_bytes(payload) != expected:
        raise ValueError("requested source member digest does not match manifest")
    return payload, expected


def extract_verified_source_member(
    manifest_path: Path, archive_path: Path, member_path: str, destination: Path,
) -> str:
    """Extract one manifest-bound member from retained archive bytes to a fresh path."""

    manifest_payload = _snapshot_file(
        manifest_path, name="source component manifest", maximum=_MAX_MANIFEST_BYTES)
    archive_payload = _snapshot_file(archive_path, name="source archive", maximum=_MAX_ARCHIVE_BYTES)
    payload, expected = verified_source_member_payload(
        manifest_payload, archive_payload, member_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError as exc:
        raise ValueError("requested source extraction destination already exists") from exc
    with os.fdopen(fd, "wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    return expected


def source_archive_payload_verification(
    manifest_path: Path, archive_path: Path, *, checkout_root: Path,
) -> tuple[bytes, bytes, str, str]:
    """Retain and verify every source authority, returning the consumed bytes."""

    snapshot = _snapshot_checkout_components(manifest_path, checkout_root=checkout_root)
    archive_payload = _snapshot_file(archive_path, name="source archive", maximum=_MAX_ARCHIVE_BYTES)
    archive_sha256 = _verify_archive_payload(
        snapshot.manifest_payload, archive_payload, checkout_snapshot=snapshot)
    return snapshot.manifest_payload, archive_payload, snapshot.manifest_sha256, archive_sha256


def source_archive_verification(
    manifest_path: Path, archive_path: Path, *, checkout_root: Path,
) -> tuple[str, str]:
    """Verify source archive closure and return manifest/archive digests from snapshots."""

    _manifest_payload, _archive_payload, manifest_sha256, archive_sha256 = source_archive_payload_verification(
        manifest_path, archive_path, checkout_root=checkout_root)
    return manifest_sha256, archive_sha256


def verify_source_archive(manifest_path: Path, archive_path: Path, *, checkout_root: Path) -> str:
    """Verify source archive closure from single snapshots of every input authority."""

    _manifest_sha256, archive_sha256 = source_archive_verification(
        manifest_path, archive_path, checkout_root=checkout_root)
    return archive_sha256
