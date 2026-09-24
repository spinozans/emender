import hashlib
import io
import json
import os
from pathlib import Path
import tarfile

import pytest

import ndm.e97_first_party_source_archive as source_archive
from ndm.e97_first_party_read_observe import safe_extract_fixture_archive
from ndm.e97_onpolicy_records import sha256_json
from scripts import validate_e97_first_party_task as replay_cli


def _write_manifest(path: Path, component: Path, payload: bytes) -> None:
    path.write_text(json.dumps({
        "schema": "emender-e97-firstparty-generator-manifest-v3",
        "components": [{"path": component.as_posix(), "sha256": hashlib.sha256(payload).hexdigest()}],
    }, separators=(",", ":")))


def test_source_archive_build_uses_one_retained_component_snapshot(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    component = Path("generator.py")
    original, replacement = b"first source authority", b"substituted source authority"
    (checkout / component).write_bytes(original)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, component, original)
    output = tmp_path / "source.tar"
    monkeypatch.setattr(source_archive, "EXPECTED_GENERATOR_COMPONENT_PATHS", frozenset({component.as_posix()}))
    actual_reader = source_archive.read_regular_file_no_follow

    def replace_after_component_snapshot(path, *, maximum):
        payload = actual_reader(path, maximum=maximum)
        if Path(path) == checkout / component:
            (checkout / component).write_bytes(replacement)
        return payload

    monkeypatch.setattr(source_archive, "read_regular_file_no_follow", replace_after_component_snapshot)
    digest = source_archive.build_source_archive(manifest, output, checkout_root=checkout)
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    with tarfile.open(output, "r:") as archive:
        assert archive.extractfile(component.as_posix()).read() == original
    assert (checkout / component).read_bytes() == replacement


@pytest.mark.parametrize("kind", ("symlink", "fifo"))
def test_source_archive_component_snapshot_rejects_special_or_linked_inputs(tmp_path, monkeypatch, kind):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    component = Path("generator.py")
    expected = b"sealed component"
    target = checkout / component
    if kind == "symlink":
        outside = tmp_path / "outside.py"
        outside.write_bytes(expected)
        target.symlink_to(outside)
    else:
        os.mkfifo(target)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, component, expected)
    monkeypatch.setattr(source_archive, "EXPECTED_GENERATOR_COMPONENT_PATHS", frozenset({component.as_posix()}))
    with pytest.raises(ValueError, match="source component generator.py is unreadable"):
        source_archive.build_source_archive(manifest, tmp_path / "source.tar", checkout_root=checkout)


@pytest.mark.parametrize("kind", ("symlink", "fifo"))
def test_source_archive_verification_rejects_linked_or_fifo_archive(tmp_path, monkeypatch, kind):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    component = Path("generator.py")
    payload = b"sealed component"
    (checkout / component).write_bytes(payload)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, component, payload)
    archive = tmp_path / "source.tar"
    monkeypatch.setattr(source_archive, "EXPECTED_GENERATOR_COMPONENT_PATHS", frozenset({component.as_posix()}))
    source_archive.build_source_archive(manifest, archive, checkout_root=checkout)
    if kind == "symlink":
        outside = tmp_path / "outside.tar"
        outside.write_bytes(archive.read_bytes())
        archive.unlink()
        archive.symlink_to(outside)
    else:
        archive.unlink()
        os.mkfifo(archive)
    with pytest.raises(ValueError, match="source archive is unreadable"):
        source_archive.verify_source_archive(manifest, archive, checkout_root=checkout)


def test_source_archive_requires_exact_canonical_ustar_bytes_and_rebuilds_identically(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"; checkout.mkdir()
    component = Path("generator.py"); payload = b"canonical source"
    (checkout / component).write_bytes(payload)
    manifest = tmp_path / "manifest.json"; _write_manifest(manifest, component, payload)
    first, second = tmp_path / "one.tar", tmp_path / "two.tar"
    monkeypatch.setattr(source_archive, "EXPECTED_GENERATOR_COMPONENT_PATHS", frozenset({component.as_posix()}))
    source_archive.build_source_archive(manifest, first, checkout_root=checkout)
    source_archive.build_source_archive(manifest, second, checkout_root=checkout)
    assert first.read_bytes() == second.read_bytes()
    first.write_bytes(first.read_bytes() + b"trailing bytes")
    with pytest.raises(ValueError, match="canonical USTAR"):
        source_archive.verify_source_archive(manifest, first, checkout_root=checkout)


@pytest.mark.parametrize("archive_format", (tarfile.GNU_FORMAT, tarfile.PAX_FORMAT))
def test_source_archive_rejects_valid_gnu_and_pax_encodings(tmp_path, monkeypatch, archive_format):
    checkout = tmp_path / "checkout"; checkout.mkdir()
    component = Path("generator.py"); payload = b"canonical source"
    (checkout / component).write_bytes(payload)
    manifest = tmp_path / "manifest.json"; _write_manifest(manifest, component, payload)
    archive = tmp_path / "source.tar"
    monkeypatch.setattr(source_archive, "EXPECTED_GENERATOR_COMPONENT_PATHS", frozenset({component.as_posix()}))
    with tarfile.open(archive, "w", format=archive_format) as stream:
        info = tarfile.TarInfo(component.as_posix())
        info.size = len(payload); info.mode = 0o644; info.uid = info.gid = 0
        info.uname = info.gname = ""; info.mtime = 0
        if archive_format == tarfile.PAX_FORMAT:
            info.pax_headers = {"comment": "force-a-valid-pax-extension"}
        stream.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="canonical USTAR"):
        source_archive.verify_source_archive(manifest, archive, checkout_root=checkout)


def test_source_archive_rejects_valid_noncanonical_header_padding(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"; checkout.mkdir()
    component = Path("generator.py"); payload = b"canonical source"
    (checkout / component).write_bytes(payload)
    manifest = tmp_path / "manifest.json"; _write_manifest(manifest, component, payload)
    archive = tmp_path / "source.tar"
    monkeypatch.setattr(source_archive, "EXPECTED_GENERATOR_COMPONENT_PATHS", frozenset({component.as_posix()}))
    source_archive.build_source_archive(manifest, archive, checkout_root=checkout)
    mutated = bytearray(archive.read_bytes())
    # ``devmajor`` is ignored for a regular member by tarfile, but changing it
    # and recomputing the header checksum yields a valid noncanonical archive.
    mutated[329:337] = b"0000001\0"
    mutated[148:156] = b" " * 8
    checksum = sum(mutated[:512])
    mutated[148:156] = f"{checksum:06o}\0 ".encode("ascii")
    archive.write_bytes(mutated)
    with tarfile.open(archive, "r:") as stream:
        assert stream.getmembers()[0].name == component.as_posix()
    with pytest.raises(ValueError, match="canonical USTAR"):
        source_archive.verify_source_archive(manifest, archive, checkout_root=checkout)


def test_source_archive_build_publishes_the_retained_buffer_without_output_reread(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"; checkout.mkdir()
    component = Path("generator.py"); payload = b"sealed"
    (checkout / component).write_bytes(payload)
    manifest = tmp_path / "manifest.json"; _write_manifest(manifest, component, payload)
    output = tmp_path / "source.tar"
    monkeypatch.setattr(source_archive, "EXPECTED_GENERATOR_COMPONENT_PATHS", frozenset({component.as_posix()}))
    observed = []
    real_publish = source_archive.publish_bytes_atomically

    def replace_output(path, retained):
        output.write_bytes(b"attacker output")
        observed.append(retained)
        real_publish(path, retained)

    monkeypatch.setattr(source_archive, "publish_bytes_atomically", replace_output)
    source_archive.build_source_archive(manifest, output, checkout_root=checkout)
    assert output.read_bytes() == observed[0]


def test_safe_fixture_extraction_consumes_retained_archive_bytes(tmp_path):
    content = b"token=sealed\n"
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        member = tarfile.TarInfo("fixture/token.txt")
        member.size = len(content)
        member.mode = 0o644
        archive.addfile(member, io.BytesIO(content))
    retained = archive_bytes.getvalue()
    tree_digest = sha256_json([
        {"path": "fixture", "type": "directory"},
        {"path": "fixture/token.txt", "type": "file", "sha256": hashlib.sha256(content).hexdigest()},
    ])
    destination = tmp_path / "fixture"
    destination.mkdir()
    assert safe_extract_fixture_archive(
        retained, destination,
        expected_sha256=hashlib.sha256(retained).hexdigest(),
        expected_tree_digest=tree_digest, disk_limit=1024,
    ) == len(content)
    assert (destination / "fixture" / "token.txt").read_bytes() == content


def test_fixture_tree_digest_preserves_directory_identity_observed_by_list_files(tmp_path):
    from ndm.e97_acquisition_controller import WorkspaceToolExecutor
    from ndm.e97_first_party_read_observe import _archive, _tree_digest

    source = tmp_path / "source"; (source / "empty").mkdir(parents=True)
    (source / "nested").mkdir(); (source / "nested" / "token.txt").write_text("token=sealed\n")
    declared = _tree_digest(source)
    archive = tmp_path / "fixture.tar"; _archive(source, archive)
    destination = tmp_path / "destination"; destination.mkdir()
    safe_extract_fixture_archive(
        archive.read_bytes(), destination, expected_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        expected_tree_digest=declared, disk_limit=1024,
    )
    assert _tree_digest(destination) == declared
    executor = WorkspaceToolExecutor(destination, max_output_bytes=1024, max_tree_bytes=1024)
    try:
        observed = executor.execute("list_files", {"path": ".", "depth": 2, "limit": 16})
    finally:
        executor.close()
    assert {entry["path"] for entry in observed.raw_observation["entries"]} == {
        "empty", "nested", "nested/token.txt",
    }


def test_replay_cli_snapshot_reader_retains_substitution_and_rejects_links_and_fifos(tmp_path, monkeypatch):
    authority = tmp_path / "bundle.json"
    original, replacement = b'{"sealed":true}', b'{"substituted":true}'
    authority.write_bytes(original)
    actual_reader = replay_cli.read_regular_file_no_follow

    def replace_after_snapshot(path, *, maximum):
        payload = actual_reader(path, maximum=maximum)
        if Path(path) == authority:
            authority.write_bytes(replacement)
        return payload

    monkeypatch.setattr(replay_cli, "read_regular_file_no_follow", replace_after_snapshot)
    assert replay_cli._snapshot_input(authority, name="bundle", maximum=1024) == original
    assert authority.read_bytes() == replacement

    linked = tmp_path / "linked.json"
    linked.symlink_to(authority)
    with pytest.raises(ValueError, match="cannot be snapshotted safely"):
        replay_cli._snapshot_input(linked, name="bundle", maximum=1024)
    fifo = tmp_path / "archive.tar"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="cannot be snapshotted safely"):
        replay_cli._snapshot_input(fifo, name="fixture archive", maximum=1024)
