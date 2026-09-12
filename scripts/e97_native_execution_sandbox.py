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


def bounded_output(command, limit=131072, timeout=15, stderr=subprocess.DEVNULL):
    """Bound subprocess bytes before buffering and kill on deadline/overflow."""
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr)
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

    def resume_for_continuation(self):
        spec = inspect_container(self.identity)
        if (spec['Config']['Labels'].get('emender.native-qualification') != self.nonce
                or not spec['State']['Paused'] or not spec['State']['Running']):
            raise ValueError('continuation requires owned paused running container')
        subprocess.run(DOCKER+['unpause', self.identity], check=True, timeout=10, stdout=subprocess.DEVNULL)

    def snapshot(self, names, *, label=None):
        evidence = self.evidence
        if label is not None:
            if label != 'teacher':
                raise ValueError('unsupported snapshot label')
            evidence = self.evidence/label
            evidence.mkdir(mode=0o700)
        # Docker archive API on this host cannot see the agent's tmpfs mounts.
        # Freeze all agent processes, then use a separate, immutable-code reader
        # in the agent PID namespace (not its cgroup or Python environment).
        subprocess.run(DOCKER+['pause', self.identity], check=True, timeout=10, stdout=subprocess.DEVNULL)
        spec = inspect_container(self.identity)
        if not spec['State']['Paused']:
            raise ValueError('snapshot requires paused container')
        reader = Path('scripts/e97_native_snapshot_reader.py').read_bytes()
        if hashlib.sha256(reader).hexdigest() != self.panel['snapshot_reader_sha256']:
            raise ValueError('snapshot reader identity')
        nonce = uuid.uuid4().hex; helper = None
        command = container_args(self.panel['image_id'], nonce, {}, 1)
        image_index = command.index(self.panel['image_id'])
        command = command[:image_index] + ['--pid', 'container:'+self.identity, self.panel['image_id'],
                   'python', '-I', '-S', '-c', reader.decode(), json.dumps(names)]
        try:
            helper = subprocess.check_output(command, timeout=30).decode().strip()
            before = inspect_container(helper)
            validate_container(before, self.panel['image_id'], nonce)
            if before['HostConfig']['PidMode'] != 'container:'+self.identity:
                raise ValueError('snapshot PID namespace')
            (evidence/'reader-before.json').write_text(json.dumps(before, indent=2))
            with (evidence/'reader.stderr').open('wb') as stderr:
                data = bounded_output(DOCKER+['start', '--attach', helper], limit=4*1024*1024,
                                      timeout=30, stderr=stderr)
            if data is None:
                raise RuntimeError('trusted snapshot reader failed; see reader.stderr')
            (evidence/'reader.stdout').write_bytes(data)
            state = inspect_container(helper)['State']
            if state['Running'] or state['ExitCode'] or state['OOMKilled']:
                raise RuntimeError('trusted reader did not complete cleanly')
            result = json.loads(data)
            if set(result) != set(names) or any(v is not None and not isinstance(v, str) for v in result.values()):
                raise ValueError('snapshot coverage/type')
            if not inspect_container(self.identity)['State']['Paused']:
                raise ValueError('agent resumed during snapshot')
            return result
        finally:
            if helper:
                terminal = inspect_container(helper)
                if terminal['Config']['Labels'].get('emender.native-qualification') != nonce:
                    raise ValueError('refuse unowned reader cleanup')
                (evidence/'reader-terminal.json').write_text(json.dumps(terminal, indent=2))
                subprocess.run(DOCKER+['rm', '--force', helper], check=True, timeout=10, stdout=subprocess.DEVNULL)
                (evidence/'reader-cleanup.json').write_text(json.dumps(dict(container_id=helper, removed=True)))

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
