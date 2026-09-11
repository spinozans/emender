"""Owned, bounded native tool container; host-side paused filesystem snapshots."""
import hashlib
import io
import json
import os
from pathlib import Path
import selectors
import subprocess
import tarfile
import time
import uuid
from scripts.run_e97_openhands_native_qualification import (
    BOOTSTRAP, DOCKER, container_args, inspect_container, validate_container,
)


def archive_file(data):
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        entries = archive.getmembers()
        if len(entries) != 1 or not entries[0].isfile() or entries[0].size > 65536:
            raise ValueError('nonregular or oversized outcome')
        return archive.extractfile(entries[0]).read().decode('utf-8')


def bounded_output(command, limit=131072, timeout=15):
    """Bound Docker archive bytes before buffering; directory/symlink outputs fail closed."""
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    data = bytearray()
    sel = selectors.DefaultSelector(); sel.register(p.stdout, selectors.EVENT_READ)
    end = time.monotonic() + timeout
    try:
        while True:
            if not sel.select(max(0, end - time.monotonic())):
                raise TimeoutError('archive deadline')
            chunk = os.read(p.stdout.fileno(), 65536)
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > limit:
                raise ValueError('archive byte bound')
        code = p.wait(timeout=max(.1, end-time.monotonic()))
        if code:
            return None
        return bytes(data)
    finally:
        sel.close()
        if p.poll() is None:
            p.kill(); p.wait(timeout=5)
        p.stdout.close()


class NativeSandbox:
    def __init__(self, panel, evidence):
        self.panel = panel; self.evidence = Path(evidence)
        self.evidence.mkdir(mode=0o700)
        self.identity = None; self.proc = None; self.sel = None; self.log = None
        self.nonce = uuid.uuid4().hex; self.buffer = b''

    def __enter__(self):
        try:
            payload = io.BytesIO()
            with tarfile.open(fileobj=payload, mode='w:') as tar:
                for name, digest in self.panel['bundle_files'].items():
                    data = Path(name).read_bytes()
                    if hashlib.sha256(data).hexdigest() != digest:
                        raise ValueError('bundle source identity')
                    info = tarfile.TarInfo(name); info.size = len(data); info.mode = 0o400
                    tar.addfile(info, io.BytesIO(data))
            data = payload.getvalue()
            command = container_args(self.panel['image_id'], self.nonce, self.panel['bundle_files'], len(data))
            if command[-3] != BOOTSTRAP:
                raise ValueError('bootstrap anchor')
            command[-3] = BOOTSTRAP.replace('qualify_e97_openhands_native_execution.py', 'e97_native_execution_rpc.py')
            self.identity = subprocess.check_output(command, timeout=30).decode().strip()
            spec = inspect_container(self.identity)
            validate_container(spec, self.panel['image_id'], self.nonce)
            (self.evidence/'container-before.json').write_text(json.dumps(spec, indent=2))
            self.log = (self.evidence/'backend-private.log').open('wb')
            self.proc = subprocess.Popen(DOCKER+['start', '--attach', '--interactive', self.identity],
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log)
            self.sel = selectors.DefaultSelector(); self.sel.register(self.proc.stdout, selectors.EVENT_READ)
            self.proc.stdin.write(data); self.proc.stdin.flush()
            ready = self.receive(120)
            wanted = {t['function']['name']: t for t in self.panel['tools']}
            actual = {t['function']['name']: t for t in ready.get('tools', [])}
            if not ready.get('ready') or actual != wanted or not all(ready['sandbox'].values()):
                raise ValueError('backend startup/schema isolation')
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def receive(self, seconds):
        deadline = time.monotonic() + seconds
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError('native RPC deadline')
            while b'\n' in self.buffer:
                line, self.buffer = self.buffer.split(b'\n', 1)
                if line.startswith(b'E97_RPC '):
                    return json.loads(line[8:])
            if not self.sel.select(max(0, deadline-time.monotonic())):
                raise TimeoutError('native RPC deadline')
            chunk = os.read(self.proc.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError('native RPC terminated')
            self.log.write(chunk); self.log.flush()
            self.buffer += chunk
            if len(self.buffer) > 16*1024*1024:
                raise ValueError('native RPC byte bound')

    def request(self, op, **kwargs):
        identity = uuid.uuid4().hex
        self.proc.stdin.write((json.dumps(dict(op=op, id=identity, **kwargs))+'\n').encode())
        self.proc.stdin.flush()
        reply = self.receive(45)
        if reply.get('id') != identity:
            raise ValueError('RPC response identity')
        return reply

    def snapshot(self, names):
        # Freeze all generated processes before a daemon-side, non-executing read.
        subprocess.run(DOCKER+['pause', self.identity], check=True, timeout=10, stdout=subprocess.DEVNULL)
        spec = inspect_container(self.identity)
        if not spec['State']['Paused']:
            raise ValueError('snapshot requires paused container')
        result = {}
        for name in names:
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('snapshot path')
            try:
                data = bounded_output(DOCKER+['cp', self.identity+':/testbed/'+name, '-'])
                result[name] = archive_file(data) if data is not None else None
            except (ValueError, UnicodeError, tarfile.TarError):
                result[name] = None
        return result

    def __exit__(self, *args):
        try:
            if self.identity:
                spec = inspect_container(self.identity)
                if spec['Config']['Labels'].get('emender.native-qualification') != self.nonce:
                    raise ValueError('refuse unowned cleanup')
                (self.evidence/'container-terminal.json').write_text(json.dumps(spec, indent=2))
                if spec['State']['Paused']:
                    subprocess.run(DOCKER+['unpause', self.identity], check=True, timeout=10, stdout=subprocess.DEVNULL)
                if spec['State']['Running']:
                    subprocess.run(DOCKER+['stop', '--time', '1', self.identity], check=True, timeout=10, stdout=subprocess.DEVNULL)
                subprocess.run(DOCKER+['rm', self.identity], check=True, timeout=10, stdout=subprocess.DEVNULL)
                (self.evidence/'cleanup.json').write_text(json.dumps(dict(container_id=self.identity, removed=True)))
        finally:
            if self.proc:
                if self.proc.poll() is None:
                    self.proc.terminate(); self.proc.wait(timeout=10)
                self.proc.stdin.close(); self.proc.stdout.close()
            if self.sel:
                self.sel.close()
            if self.log:
                self.log.close()
