"""W1: an open live prefix and an immutable, in-memory m20 fixture. No answer cache."""
from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from contextlib import AbstractContextManager
from functools import lru_cache
from pathlib import Path

EXE = Path.home() / '.cache/xsa-b2-target/release/xsa'
EXE_SHA = 'fbc436aac93190368bbed637a5f69cca0fa5848fc81a311e60c4ebe1b4d3bf50'
DEPTH = 16


@lru_cache(maxsize=1)
def check_binary():
    if hashlib.sha256(EXE.read_bytes()).hexdigest() != EXE_SHA:
        raise ValueError(f'wrong native session executable: {EXE}')


class _Session(AbstractContextManager):
    def __init__(self):
        check_binary()
        self.proc = subprocess.Popen([EXE, 'session', str(DEPTH)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE)
        self.visible = 0

    def close(self):
        if not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
            finally:
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill(); self.proc.wait()
                finally:
                    self.proc.stdout.close()

    def __exit__(self, *exc):
        self.close()

    def _rpc(self, line):
        self.proc.stdin.write(line.encode('ascii') + b'\n'); self.proc.stdin.flush()
        data = self.proc.stdout.readline(32769)
        if not data.endswith(b'\n') or len(data) > 32768:
            raise ValueError('missing or oversized native reply')
        return json.loads(data)

    def _append(self, data):
        if not isinstance(data, bytes) or not data:
            raise ValueError('append requires nonempty bytes')
        if self._rpc('append ' + data.hex()) != {'visible': self.visible + len(data)}:
            raise ValueError('incorrect native visibility')
        self.visible += len(data)

    def branch(self, key: bytes) -> tuple[int, ...]:
        if not isinstance(key, bytes) or not 0 < len(key) <= DEPTH:
            raise ValueError('outside native branch depth')
        r = self._rpc('branch ' + key.hex())
        if r.keys() != {'counts', 'end'} or not isinstance(r['counts'], list):
            raise ValueError('invalid branch reply')
        h = (*r['counts'], r['end'])
        if (len(h) != 257 or any(type(v) is not int or not 0 <= v < 2**24 for v in h)
                or sum(h) > self.visible):
            raise ValueError('invalid or out-of-W1 FP32-range histogram')
        return h


class Live(_Session):
    def __init__(self, prefix=b''):
        super().__init__()
        self.session_id = uuid.uuid4().hex
        self.tail = b''
        try:
            if prefix:
                self.append(prefix)
        except BaseException:
            self.close(); raise

    def append(self, data):
        self._append(data)
        self.tail = (self.tail + data)[-8:]


class Archive(_Session):
    """Trusted local whole-bank fixture. No public append/end/seal after construction."""
    name = 'fixture'

    def __init__(self, documents, *, allowed=True, excluded_docs=()):
        if not allowed or excluded_docs:
            raise PermissionError('W1 supports only an admitted whole fixture')
        docs = tuple(documents)
        if not docs or any(not isinstance(d, bytes) or not d for d in docs):
            raise ValueError('expected nonempty fixture documents')
        self.version = hashlib.sha256(b''.join(len(d).to_bytes(8, 'big') + d for d in docs)).hexdigest()
        super().__init__()
        try:
            for doc in docs:
                self._append(doc)
                if self._rpc('end') != {'ended': True}:
                    raise ValueError('unconfirmed document end')
            if self._rpc('seal') != {'sealed': self.visible}:
                raise ValueError('unsealed archive')
        except BaseException:
            self.close(); raise
